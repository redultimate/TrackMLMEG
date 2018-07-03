import math
import numpy as np

def fit_circle(x,y):
    sumx = sum(x)
    sumy = sum(y)
    sumx2 = sum([ix ** 2 for ix in x])
    sumy2 = sum([iy ** 2 for iy in y])
    sumxy = sum([ix * iy for (ix,iy) in zip(x,y)])

    F = np.array([sumx2,sumxy,sumx],
                 [sumxy,sumy2,sumy],
                 [sumx, sumy, len(x)])
    G = np.array([[-sum([ix ** 3 + ix*iy **2 for (ix,iy) in zip(x,y)])],
                  [-sum([ix **2 *iy + iy **3 for (ix,iy) in zip(x,y)])],
                  [-sum([ix ** 2 + iy **2 for (ix,iy) in zip(x,y)])]])

    T = np.linag.inv(F).dot(G)

    cxe = float(T[0]/-2)
    cye = float(T[1]/-2)
    re = math.sqrt(cxe**2 + cye**2 - T[2])

    return (cxe,cye,re)

def get_1uDp(x,y):
    #assume x = (x1, x2, x3) and y = (y1, y2, y3)
    if(len(x)!=3 or len(y)!=3):
        return (0,0,0)
    r2 = x*x + y*y
    r2_dif = [r2[1] - r2[2],r2[2] - r2[0], r2[0] - r2[1]]
    phi = math.atan(sum(y * r2_dif)/sum(x *r2_dif))
    if phi < 0:
        phi += math.pi

    rd = -r2_dif[2]/2.0/(math.sin(phi)*(x[0]-x[1])-math.cos(phi)*(y[0]-y[1]))
    if rd < 0:
        rd = -rd
        phi += math.pi

    rr = r2[0] + 2.0*rd*(math.sin(phi)*x[0]-math.cos(phi)*y[0])+rd**2
    u = 1.0/math.sqrt(rr)
    D = math.sqrt(rr) - rd

    ATan0 = math.atan((y[0]-rd*math.cos(phi))/(x[0]+rd*math.sin(phi))) 
    ATan1 = math.atan((y[1]-rd*math.cos(phi))/(x[1]+rd*math.sin(phi))) 
    if x[0] < -rd*math.sin(phi):
        ATan0 += math.pi
        if x[1] < -rd*math.sin(phi):
            ATan1 += math.pi
        elif ATan1 < 0:
            ATan1 += 2*math.pi
    elif ATan0 < 0:
        ATan0 += 2*math.pi
        if x[1] < -rd*math.sin(phi):
            ATan1 += math.pi
        else:
            ATan1 += 2*math.pi
    else:
        if x[1] < -rd*math.sin(phi):
            ATan1 += math.pi
        elif ATan1 < 0:
            ATan1 += 2*math.sin(phi)
            ATan0 += 2*math.sin(phi)

    q = 1
    if ATan1 > ATan0:
        q = -1
    
    #z is not used
    #ASin1 = calcurate_asin(math.arcsin((y[1] - rd*math.cos(phi))*u), q, x[1], -rd*math.sin(phi))
    #ASin0 = calcurate_asin(math.arcsin((y[0] - rd*math.cos(phi))*u), q, x[0], -rd*math.sin(phi))
    #eta = q * sqrt(rr) * (ASin1 - ASin0) / (z[1] - z[0])
    #nanika = z[0] - q * math.sqrt(rr) * (ASin0 - phi + math.pi/2)/eta
    
    u *= q
    D *= q
    if q > 0:
        if phi > math.pi:
            phi -= math.pi
        else:
            phi += math.pi
    #eta *= q

    return (u, D, phi)

def get_uDp(x):
    #assume x = [{(x1,y1),(x2,y2),(x3,y3)},{},... ]
    if (x.shape[1] !=3) or (x.shape[2] != 2):
        return np.zeros(3)
    r2 = (x*x).sum(axis=2)
    r2_dif = np.array([r2[:,1] - r2[:,2],r2[:,2] - r2[:,0], r2[:,0] - r2[:,1]]).transpose()
    phi = np.arctan((x[:,:,1] * r2_dif).sum(axis=1)/(x[:,:,0] * r2_dif).sum(axis=1))
    phi[phi<0] += math.pi

    rd = -r2_dif[:,2]/2.0/(np.sin(phi)*(x[:,0,0]-x[:,1,0])-np.cos(phi)*(x[:,0,1]-x[:,1,1]))
    phi[rd<0] += math.pi
    rd[rd<0] *= -1

    rr = r2[:,0] + 2.0*rd*(np.sin(phi)*x[:,0,0]-np.cos(phi)*x[:,0,1])+rd**2
    u = 1.0/np.sqrt(rr)
    D = np.sqrt(rr) - rd

    ATan0 = np.arctan((x[:,0,1]-rd*np.cos(phi))/(x[:,0,0]+rd*np.sin(phi))) 
    ATan1 = np.arctan((x[:,1,1]-rd*np.cos(phi))/(x[:,1,0]+rd*np.sin(phi))) 

    con_lv0 = [x[:,0,0] < -rd*np.sin(phi), 
               ~(x[:,0,0] < -rd*np.sin(phi)) & (ATan0 < 0), 
               ~(x[:,0,0] < -rd*np.sin(phi)) & ~(ATan0 < 0)]
    con_lv1 = [x[:,1,0] < -rd*np.sin(phi),
               ~(x[:,1,0] < -rd*np.sin(phi)) & (ATan1 < 0),
               ~(x[:,1,0] < -rd*np.sin(phi))]
    ATan0[con_lv0[0]] += math.pi
    ATan1[con_lv0[0] & con_lv1[0]] += math.pi
    ATan1[con_lv0[0] & con_lv1[1]] += 2*math.pi
    ATan0[con_lv0[1]] += 2*math.pi
    ATan1[con_lv0[1] & con_lv1[0]] += math.pi
    ATan1[con_lv0[1] & con_lv1[2]] += 2*math.pi
    ATan1[con_lv0[2] & con_lv1[0]] += math.pi
    ATan1[con_lv0[2] & con_lv1[1]] += 2*math.pi
    ATan0[con_lv0[2] & con_lv1[1]] += 2*math.pi
    #if x[:,0,0] < -rd*np.sin(phi):
    #    ATan0 += math.pi
    #    if x[:,1,0] < -rd*np.sin(phi):
    #        ATan1 += math.pi
    #    elif ATan1 < 0:
    #        ATan1 += 2*math.pi
    #elif ATan0 < 0:
    #    ATan0 += 2*math.pi
    #    if x[:,1,0] < -rd*np.sin(phi):
    #        ATan1 += math.pi
    #    else:
    #        ATan1 += 2*math.pi
    #else:
    #    if x[:,1,0] < -rd*np.sin(phi):
    #        ATan1 += math.pi
    #    elif ATan1 < 0:
    #        ATan1 += 2*math.pi
    #        ATan0 += 2*math.pi

    q = np.ones(x.shape[0])
    q[ATan1>ATan0] = -1
    #if ATan1 > ATan0:
    #    q = -1
    
    #z is not used
    #ASin1 = calcurate_asin(math.arcsin((y[1] - rd*math.cos(phi))*u), q, x[1], -rd*math.sin(phi))
    #ASin0 = calcurate_asin(math.arcsin((y[0] - rd*math.cos(phi))*u), q, x[0], -rd*math.sin(phi))
    #eta = q * sqrt(rr) * (ASin1 - ASin0) / (z[1] - z[0])
    #nanika = z[0] - q * math.sqrt(rr) * (ASin0 - phi + math.pi/2)/eta
    
    u *= q
    D *= q
    #if q > 0:
    #    if phi > math.pi:
    condition = [(q>0) & (phi>math.pi), (q>0) & ~(phi>math.pi)]
    phi[condition[0]] -= math.pi
    #    else:
    phi[condition[1]] += math.pi
    #eta *= q

    return (u, D, phi)

