#
# This function determines if the isnan function is available on this
# platform.
#
AC_DEFUN([AC_FUNC_ISNAN],[
AC_SINGLE_CXX_CHECK([HAVE_ISNAN_IN_MATH_H],    [ac_cv_func_isnan_in_math_h],   
                    [isnan], [<math.h>],
                    [#include <math.h>
                     int foo(float f) {return isnan(f);}])
AC_SINGLE_CXX_CHECK([HAVE_ISNAN_IN_CMATH],     [ac_cv_func_isnan_in_cmath],    
                    [isnan], [<cmath>],
                    [#include <cmath>
                     int foo(float f) {return isnan(f);}])
AC_SINGLE_CXX_CHECK([HAVE_STD_ISNAN_IN_CMATH], [ac_cv_func_std_isnan_in_cmath],
                    [std::isnan], [<cmath>],
                    [#include <cmath>
                     using std::isnan; int foo(float f) {return isnan(f);}])
])
