from numpy import ndarray, array

class LeastSqr:
    """
    Single varaible least square method

    Assume fitting equation:

        y = b0 + b1 * x

    So sum of square of residual

        Sr = SUM(yi - b0 - b1 * x1)2
    
    To fit curve we minimise residual sum by differenciating
    `Sr` w.r.t b0 and b1, So we get

    sum_yi   = n * b0      + b1 * sum_xi
    sum_xiyi = b0 * sum_xi + b1 * sum_xi2

    On solving these two equation we get coeff of b0 and b1
    """

    x, y, b0, b1 = [], [], None, None

    def __init__(self, x: ndarray, y: ndarray) -> None:
        self.x = array(x)
        self.y = array(y)

    @property
    def coff_(self):
        return [self.b0, self.b1]

    def test(self, x):
        assert self.b0 is not None and self.b1 is not None, (
            "Please train first"
        )
        return self.b0 + self.b1 * x

    def train(self):
        x, y = self.x, self.y
        n = min(x.size, y.size)
        _x = x.sum()
        _y = y.sum()
        _x2 = (x * x).sum()
        _xy = (x * y).sum()
        self.b0 = (_x2*_y - _x*_xy) / (n*_x2 - _x*_x)
        self.b1 = (n*_xy - _x*_y) / (n*_x2 - _x*_x)
