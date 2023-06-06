import numpy as np

def calc_curvature(contour,stride=1):
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
    return local_curvatures
