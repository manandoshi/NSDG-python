import numpy as np
import interpolation
import lobatto


def massMatrix(nodes, weights, exact=False):
    if exact:
        n = weights.size
        integration_nodes, integration_weights = lobatto.compute_nodes(n + 1)
        mat = interpolation.compute_matrix(nodes, integration_nodes)
        return mat.T.dot(mat * np.reshape(integration_weights, [-1, 1]))
    else:
        return np.diag(weights)


def massMatrix_2D(nodes_x, nodes_y, weights_x, weights_y, exact=False, exact_x=False, exact_y=False):
    nx = weights_x.size
    ny = weights_y.size
    mx = massMatrix(nodes_x, weights_x, exact or exact_x)
    my = massMatrix(nodes_y, weights_y, exact or exact_y)
    mx = np.concatenate([mx for i in xrange(ny)], 0)
    mx = np.concatenate([mx for i in xrange(ny)], 1)
    my = np.repeat(my, nx, 0)
    my = np.repeat(my, nx, 1)
    return mx * my


def derMatrix(nodes, weights):
    n = nodes.size
    mat = np.zeros([n, n])
    for i, node in enumerate(nodes):
        mat[i, :] = interpolation.dlagrangian(i, nodes) * weights[i]
    return mat


def derMatrix_2D(nodes_x, nodes_y, weights_x, weights_y, exact=True):
    nx = weights_x.size
    ny = weights_y.size

    dx = derMatrix(nodes_x, weights_x)
    mx = massMatrix(nodes_x, weights_x, exact)

    dy = derMatrix(nodes_y, weights_y)
    my = massMatrix(nodes_y, weights_y, exact)

    dx = np.concatenate([dx for i in xrange(ny)], 0)
    dx = np.concatenate([dx for i in xrange(ny)], 1)

    mx = np.concatenate([mx for i in xrange(ny)], 0)
    mx = np.concatenate([mx for i in xrange(ny)], 1)

    dy = np.repeat(dy, nx, 0)
    dy = np.repeat(dy, nx, 1)

    my = np.repeat(my, nx, 0)
    my = np.repeat(my, nx, 1)

    der_x = dx * my
    der_y = mx * dy

    return der_x.T, der_y.T


def fluxMatrix(nodes_x, nodes_y, weights_x, weights_y, exact=True):
    return massMatrix(nodes_y, weights_y, exact), massMatrix(nodes_x, weights_x, exact)


def lapInv(nx,ny,mx,my,M,Dx,Dy,Fx,Fy,M_inv):
    global_M_inv = np.zeros([nx*mx*ny*my,nx*mx*ny*my])
    global_Dx    = np.zeros([nx*mx*ny*my,nx*mx*ny*my])
    global_Dy    = np.zeros([nx*mx*ny*my,nx*mx*ny*my])
    global_Fx    = np.zeros([nx*mx*ny*my,nx*mx*ny*my])
    global_Fy    = np.zeros([nx*mx*ny*my,nx*mx*ny*my])

    a = np.zeros_like(Dx)
    a[::my,::mx] = Fx.copy()
    a[my-1::my,mx-1::mx] = -1*Fx.copy()
    Fx_full = a

    a = np.zeros_like(Dy)
    a[:my,:mx] = Fy.copy()
    a[-my:,-mx:] = -1*Fy.copy()

    s.Fy_full = a

    for ix in xrange(nx):
        for iy in xrange(ny):
            for row_dest in xrange(n):
                for row_source in xrange(n):
                    temp_dest   = mx*nx*my*iy + mx*nx*row_dest   + mx*ix
                    temp_source = mx*nx*my*iy + mx*nx*row_source + mx*ix
                    
                    l_source = n*row_source
                    l_dest   = n*row_dest
                    
                    global_M_inv[temp_dest:temp_dest+n,temp_source:temp_source+n] = M_inv[l_dest:l_dest+n,l_source:l_source+n]
                    
                    global_Dx[temp_dest:temp_dest+n,temp_source:temp_source+n]  = Dx[l_dest:l_dest+n,l_source:l_source+n]
                    global_Dy[temp_dest:temp_dest+n,temp_source:temp_source+n]  = Dy[l_dest:l_dest+n,l_source:l_source+n]
                                                                               
                    global_Fx[temp_dest:temp_dest+n,temp_source:temp_source+n]  = Fx_full[l_dest:l_dest+n,l_source:l_source+n]
                    global_Fy[temp_dest:temp_dest+n,temp_source:temp_source+n]  = Fy_full[l_dest:l_dest+n,l_source:l_source+n]

    global_bx = np.eye(nx*mx*ny*my)
    global_by = np.eye(nx*mx*ny*my)

    for ix in xrange(nx-1):
        for iy in xrange(ny):
            for row in xrange(n):
                index_west = mx*nx*my*iy+mx*nx*row+mx*ix
                index_east = index_west + mx -1
                global_bx[index_east,index_east] = 0.5
                global_bx[index_east+1,index_east] = 0.5
                global_bx[index_east,index_east+1] = 0.5
                global_bx[index_east+1,index_east+1] = 0.5

    for ix in xrange(nx):
        for iy in xrange(ny-1):
            for col in xrange(n):
                index_south = mx*nx*my*iy+mx*ix+col
                index_north = index_south + mx*nx*(n-1)
                global_by[index_north,index_north] = 0.5
                global_by[index_north+n*nx,index_north] = 0.5
                global_by[index_north,index_north+n*nx] = 0.5
                global_by[index_north+n*nx,index_north+n*nx] = 0.5
                
    global_Fx = global_Fx.dot(global_bx)
    global_Fy = global_Fy.dot(global_by)
    derx = -1*global_M_inv.dot(global_Fx + global_Dx)
    dery = -1*global_M_inv.dot(global_Fy + global_Dy)
    der2x = derx.dot(derx)
    der2y = dery.dot(dery)
    lap = der2x + der2y

    return lap
