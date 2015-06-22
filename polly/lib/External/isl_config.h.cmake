
/* most gcc compilers know a function __attribute__((__warn_unused_result__))
   */
#define GCC_WARN_UNUSED_RESULT @GCC_WARN_UNUSED_RESULT@

/* Define to 1 if you have the declaration of `ffs', and to 0 if you don't. */
#define HAVE_DECL_FFS @HAVE_DECL_FFS@

/* Define to 1 if you have the declaration of `__builtin_ffs', and to 0 if you
   don't. */
#define HAVE_DECL___BUILTIN_FFS @HAVE_DECL___BUILTIN_FFS@

/* define if your compiler has __attribute__ */
#cmakedefine HAVE___ATTRIBUTE__ /**/

/* use gmp to implement isl_int */
#cmakedefine USE_GMP_FOR_MP

/* use imath to implement isl_int */
#cmakedefine USE_IMATH_FOR_MP

/* Use small integer optimization */
#cmakedefine USE_SMALL_INT_OPT

#include <isl_config_post.h>
