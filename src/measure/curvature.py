import numpy as np

def calc_local_curvature(curve, stride=1) -> np.ndarray:
    """Calc the curvature at every point of the curve"""
    dx = np.gradient(curve[::stride, 0])
    dy = np.gradient(curve[::stride, 1])
    d2x = np.gradient(dx)
    d2y = np.gradient(dy)
    nominator = dx * d2y - dy * d2x
    denominator = dx ** 2 + dy ** 2 + 1e-11
    curvature = nominator / denominator ** 1.5
    return curvature

def calc_local_curvature_old(contour,stride=1) -> np.ndarray:
    """Calc the curvature at every point of the contour"""
    local_curvatures=[]
    assert stride<len(contour),"stride must be shorther than length of contour"

    for i in range(len(contour)):
        if i-stride<0 or i+stride>=len(contour):
            local_curvatures.append(0)
            continue
        before=i-stride+len(contour) if i-stride<0 else i-stride
        after=i+stride-len(contour) if i+stride>=len(contour) else i+stride

        f1x,f1y=(contour[after]-contour[before])/stride
        f2x,f2y=(contour[after]-2*contour[i]+contour[before])/stride**2
        denominator=(f1x**2+f1y**2)**3+1e-11

        curvature_at_i=np.sqrt(4*(f2y*f1x-f2x*f1y)**2/denominator) if denominator > 1e-12 else -1

        local_curvatures.append(curvature_at_i)

    return np.array(local_curvatures)