dnl AC_SINGLE_CXX_CHECK(DEFINEVAR, CACHEVAR, FUNCTION, HEADER, PROGRAM)
dnl                     $1,        $2,       $3,       $4,     $5
dnl 
AC_DEFUN([AC_SINGLE_CXX_CHECK],
[AC_CACHE_CHECK([for $3 in $4], [$2],
 [AC_LANG_PUSH(C++)
  AC_COMPILE_IFELSE(AC_LANG_SOURCE([$5]),[$2=yes],[$2=no])
 AC_LANG_POP(C++)])
 if test "$$2" = "yes"
 then
   AC_DEFINE($1, 1, [Define to 1 if your compiler defines $3 in the $4
                     header file.])
 fi])

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

