#
# This function determins if the the isinf function isavailable on this
# platform.
#
AC_DEFUN([AC_FUNC_ISINF],[
AC_SINGLE_CXX_CHECK([HAVE_ISINF_IN_MATH_H],    [ac_cv_func_isinf_in_math_h],   
                    [isinf], [<math.h>],
                    [#include <math.h>
                     int foo(float f) {return isinf(f);}])
AC_SINGLE_CXX_CHECK([HAVE_ISINF_IN_CMATH],     [ac_cv_func_isinf_in_cmath],    
                    [isinf], [<cmath>],
                    [#include <cmath>
                     int foo(float f) {return isinf(f);}])
AC_SINGLE_CXX_CHECK([HAVE_STD_ISINF_IN_CMATH], [ac_cv_func_std_isinf_in_cmath],
                    [std::isinf], [<cmath>],
                    [#include <cmath>
                     using std::isinf; int foo(float f) {return isinf(f);}])
AC_SINGLE_CXX_CHECK([HAVE_FINITE_IN_IEEEFP_H], [ac_cv_func_finite_in_ieeefp_h],
                    [finite], [<ieeefp.h>],
                    [#include <ieeefp.h>
                     int foo(float f) {return finite(f);}])
])


