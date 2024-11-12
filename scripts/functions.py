import numpy as np

def xop2_2d(g, f, a0, a):
    """
    Performs the following operation on a C-grid:
    g[i-igo, j] = a0 * g[i-igo, j] + a[0] * f[i, j] + a[1] * f[i-1, j]
    for i = 2, ..., M+igo and j = 1, ..., N
    
    Parameters:
        g (numpy.ndarray): Output array, modified in place.
        f (numpy.ndarray): Input array.
        a0 (float): Coefficient for the g array.
        a (list or numpy.ndarray): Coefficients a1 and a2.
        
    Note:
        For an x-derivative set a[0] = 1/dx and a[1] = -1/dx.
        For an x-average set a[0] = 1/2 and a[1] = 1/2.
        Set a0 = 0 to assign result to array g.
        Set a0 = 1 to add result to array g.
    """
    
    # Determine the shape of g and f
    Mg = g.shape
    Mf = f.shape

    # Determine offset of array g (igo)
    if Mg[0] > Mf[0]:
        igo = 0  # Operation from p to u-points, set igo=0
    else:
        igo = 1  # Operation from u to p-points, set igo=1

    # Check that the second dimension of f and g match
    if Mg[1] != Mf[1]:
        raise ValueError("Error: 2nd dimension of arrays do not conform")

    # Number of x-cells
    M = min(Mg[0], Mf[0])

    # Perform the operation
    g[1-igo:M, :] = (a0 * g[1-igo:M, :] +
                     a[0] * f[1:M+igo, :] +
                     a[1] * f[:M+igo-1, :])

    return g


def yop2_2d(g, f, a0, a):
    """
    Performs the following operation:
    g[i, j-jgo] = a0 * g[i, j-jgo] + a[0] * f[i, j] + a[1] * f[i, j-1]
    for i = 1, ..., M and j = 2, ..., N+jgo

    Parameters:
        g (numpy.ndarray): Output array, modified in place.
        f (numpy.ndarray): Input array.
        a0 (float): Coefficient for the g array.
        a (list or numpy.ndarray): Coefficients a1 and a2.

    Note:
        For a y-derivative set a[0] = 1/dy and a[1] = -1/dy.
        For a y-average set a[0] = 1/2 and a[1] = 1/2.
        Set a0 = 0 to assign result to array g.
        Set a0 = 1 to add result to array g.
    """

    # Determine the shape of g and f
    Mg = g.shape
    Mf = f.shape

    # Determine offset of array g (jgo)
    if Mg[1] > Mf[1]:
        jgo = 0  # Operation from p to v-points, set jgo=0
    else:
        jgo = 1  # Operation from v to p-points, set jgo=1

    # Check that the first dimension of f and g match
    if Mg[0] != Mf[0]:
        raise ValueError("Error: 1st dimension of arrays do not conform")

    # Number of y-cells
    N = min(Mg[1], Mf[1])

    # Perform the operation
    g[:, 1-jgo:N] = (a0 * g[:, 1-jgo:N] +
                     a[0] * f[:, 1:N+jgo] +
                     a[1] * f[:, :N+jgo-1])

    return g

def swerhs(u, v, p, depth, fcoriolis, gravity, dxs):
    """
    Returns the tendencies of the non-linear Shallow Water Equations (SWE).
    
    Parameters:
        u (numpy.ndarray): Zonal velocity component.
        v (numpy.ndarray): Meridional velocity component.
        p (numpy.ndarray): Pressure field.
        depth (numpy.ndarray): Depth field.
        fcoriolis (numpy.ndarray): Coriolis parameter.
        gravity (float): Gravitational constant.
        dxs (list): Grid spacings [dx, dy].
    
    Returns:
        tuple: (ru, rv, rp) - tendencies for u, v, and p.
    """
    am = np.array([0.5, 0.5])     # averaging coefficients
    ad = np.array([1.0, -1.0])    # gradient coefficients
    alin = 0                      # set to 0/1 for linear/non-linear equations

    Ms = p.shape

    h = depth + alin * p

    hbx = np.zeros_like(u)
    hbx = xop2_2d(hbx, h, 0.0, am)
    hbx[0, :] = h[0, :]
    hbx[Ms[0], :] = h[Ms[0]-1, :]
    U = u * hbx

    hby = np.zeros_like(v)
    hby = yop2_2d(hby, h, 0.0, am)
    hby[:, 0] = h[:, 0]
    hby[:, Ms[1]] = h[:, Ms[1]-1]
    V = v * hby

    hbxy = np.zeros_like(fcoriolis)
    hbxy = yop2_2d(hbxy, hbx, 0.0, am)
    hbxy[:, 0] = hbx[:, 0]
    hbxy[:, Ms[1]] = hbx[:, Ms[1]-1]

    # Total Head
    th = -gravity * p
    ru = u**2
    th = xop2_2d(th, ru, 1.0, -0.5 * alin * am)
    rv = v**2
    th = yop2_2d(th, rv, 1.0, -0.5 * alin * am)

    # Potential Vorticity
    q = fcoriolis
    q = xop2_2d(q, v, 1.0, alin * ad / dxs[0])
    q = yop2_2d(q, u, 1.0, -alin * ad / dxs[1])
    q = q / hbxy

    # Mass-Flux Divergence and Tendencies
    rp = np.zeros_like(p)
    rp = xop2_2d(rp, U, 0.0, -ad / dxs[0])
    rp = yop2_2d(rp, V, 1.0, -ad / dxs[1])

    ru = xop2_2d(ru, th, 0.0, ad / dxs[0])
    rv = yop2_2d(rv, th, 0.0, ad / dxs[1])

    # Add Rotational Terms
    hbxy = xop2_2d(hbxy, V, 0.0, am)
    hbxy[0, :] = 0.0
    hbxy[Ms[0], :] = 0.0
    hbxy = hbxy * q
    ru = yop2_2d(ru, hbxy, 1.0, am)

    hbxy = yop2_2d(hbxy, U, 0.0, am)
    hbxy[:, 0] = 0.0
    hbxy[:, Ms[1]] = 0.0
    hbxy = hbxy * q
    rv = xop2_2d(rv, hbxy, 1.0, -am)

    # Apply No-Flow Boundary Conditions
    ru[0, :] = 0.0
    ru[Ms[0], :] = 0.0
    rv[:, 0] = 0.0
    rv[:, Ms[1]] = 0.0

    return ru, rv, rp


def rk3step(u, v, p, depth, fcoriolis, gravity, dxs, dt):
    """
    Time steps the variables (u, v, p) through a single Runge-Kutta step of size dt.
    
    Parameters:
        u (numpy.ndarray): Zonal velocity component at the present time level.
        v (numpy.ndarray): Meridional velocity component at the present time level.
        p (numpy.ndarray): Pressure field at the present time level.
        depth (numpy.ndarray): Depth field.
        fcoriolis (numpy.ndarray): Coriolis parameter.
        gravity (float): Gravitational constant.
        dxs (list): Array holding the grid sizes [dx, dy].
        dt (float): Time step.
    
    Returns:
        tuple: Updated (u, v, p) after one time step.
    """
    
    # First stage
    ru, rv, rp = swerhs(u, v, p, depth, fcoriolis, gravity, dxs)
    ut = u + dt * ru
    vt = v + dt * rv
    pt = p + dt * rp

    # Second stage
    ru, rv, rp = swerhs(ut, vt, pt, depth, fcoriolis, gravity, dxs)
    ut = 0.75 * u + 0.25 * (ut + dt * ru)
    vt = 0.75 * v + 0.25 * (vt + dt * rv)
    pt = 0.75 * p + 0.25 * (pt + dt * rp)

    # Third stage
    ru, rv, rp = swerhs(ut, vt, pt, depth, fcoriolis, gravity, dxs)
    a1 = 1.0 / 3.0
    a2 = 1.0 - a1
    u = a1 * u + a2 * (ut + dt * ru)
    v = a1 * v + a2 * (vt + dt * rv)
    p = a1 * p + a2 * (pt + dt * rp)

    return u, v, p

def initcond(xe, ye, timein=None):
    """
    Initializes the velocity and pressure fields.
    
    Parameters:
        xe (numpy.ndarray): x-coordinates at cell edges.
        ye (numpy.ndarray): y-coordinates at cell edges.
        timein (float, optional): Time parameter for initial conditions.
        
    Returns:
        tuple: Arrays for the initial velocity components (u, v) and pressure (p).
    """
    time = timein if timein is not None else 0

    nx1 = len(xe)
    ny1 = len(ye)
    nx = nx1 - 1
    ny = ny1 - 1

    # Calculate cell centers in x and y directions
    xc = 0.5 * (xe[:nx] + xe[1:nx+1])
    yc = 0.5 * (ye[:ny] + ye[1:ny+1])

    # Physical constants and parameters
    f = 0.0                   # Central Coriolis parameter (1/s)
    gravity = 10.0            # Reduced gravity (m/s^2)
    amp = 0.10
    depth = 10.0

    a = xe[-1] - xe[0]
    b = ye[-1] - ye[0]

    m = 2.0
    n = 1.0

    c = np.sqrt(gravity * depth)
    kx = m * np.pi / a
    ky = n * np.pi / b
    freq = c * np.sqrt(kx**2 + ky**2)
    Uamp = amp * gravity * kx / freq
    Vamp = amp * gravity * ky / freq

    # Create meshgrids for positions
    yp, xp = np.meshgrid(yc, xc)
    yu, xu = np.meshgrid(yc, xe)
    yv, xv = np.meshgrid(ye, xc)

    # Vortex initialization
    ct = np.cos(freq * time)
    st = np.sin(freq * time)

    # Initialize velocity and pressure fields
    u = (Uamp * st) * np.sin(kx * xu) * np.cos(ky * yu)
    v = (Vamp * st) * np.cos(kx * xv) * np.sin(ky * yv)
    p = (amp * ct) * np.cos(kx * xp) * np.cos(ky * yp)

    # Apply boundary conditions for u and v
    u[0, :] = 0.0
    u[-1, :] = 0.0
    v[:, 0] = 0.0
    v[:, -1] = 0.0

    return u, v, p

import matplotlib.pyplot as plt

def sweplot(u, v, p, xp, yp, xe, ye, scalef, plevs):
    """
    Plots the pressure field as filled contours and overlays velocity vectors.
    
    Parameters:
        u (numpy.ndarray): x-component of the velocity field.
        v (numpy.ndarray): y-component of the velocity field.
        p (numpy.ndarray): pressure field.
        xp (numpy.ndarray): x-coordinates of pressure grid.
        yp (numpy.ndarray): y-coordinates of pressure grid.
        xe (numpy.ndarray): x-coordinates of velocity grid in the x direction.
        ye (numpy.ndarray): y-coordinates of velocity grid in the y direction.
        scalef (list): Scaling factors for the plot.
        plevs (list): Pressure levels for contour plotting.
    """
    Ms = p.shape
    nx, ny = 4, 4  # Interval for plotting velocity vectors

    # Plot pressure field as filled contours
    plt.contourf(xp * scalef[3], yp * scalef[3], scalef[2] * p.T, levels=plevs)
    plt.colorbar(label='Pressure')

    # Interpolate velocity fields to cell centers
    up = 0.5 * (u[:Ms[0], :Ms[1]] + u[1:Ms[0]+1, :Ms[1]])
    vp = 0.5 * (v[:Ms[0], :Ms[1]] + v[:Ms[0], 1:Ms[1]+1])

    # Plot velocity vectors using quiver
    plt.quiver(
        xp[::nx] * scalef[3], yp[::ny] * scalef[3],
        up[::nx, ::ny].T, vp[::nx, ::ny].T,
        scale=scalef[0], color='k'
    )

    # Set plot limits
    lim1=scalef[3]*xe[0]
    lim2=scalef[3]*xe[Ms[0]]
    lim3=scalef[3]*ye[0]
    lim4=scalef[3]*ye[Ms[1]]
    
    plt.axis([lim1,lim2,lim3,lim4])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Shallow Water Equations - Pressure and Velocity Field')
    plt.show()

    # """ 
    # MATLAB to Python code conversion assistance was provided by 
    # OpenAI’s ChatGPT, used as a tool to facilitate code translation. 
    # The translated code was adapted to study numerical properties 
    # of a two-vortex problem, with modifications as needed to focus 
    # on error analysis and numerical assumptions.
    # """