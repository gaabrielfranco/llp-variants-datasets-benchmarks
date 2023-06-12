from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector
utils = importr('utils')
utils.chooseCRANmirror(ind=1)

package_names = ("doMC", "Matrix", "data.table", "FNN")
utils.install_packages(StrVector(package_names))
