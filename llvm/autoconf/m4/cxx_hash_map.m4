# Check for hash_map extension.  This is from
# http://www.gnu.org/software/ac-archive/htmldoc/ac_cxx_have_ext_hash_map.html
AC_DEFUN([AC_CXX_HAVE_STD_EXT_HASH_MAP],
[AC_CACHE_CHECK([whether the compiler has <ext/hash_map> defining template class std::hash_map],
 ac_cv_cxx_have_std_ext_hash_map,
 [AC_REQUIRE([AC_CXX_NAMESPACES])
  AC_LANG_SAVE
  AC_LANG_CPLUSPLUS
  AC_TRY_COMPILE([#include <ext/hash_map>
#ifdef HAVE_NAMESPACES
using namespace std;
#endif],[hash_map<int, int> t;],
  [ac_cv_cxx_have_std_ext_hash_map=yes], [ac_cv_cxx_have_std_ext_hash_map=no])
  AC_LANG_RESTORE])
 HAVE_STD_EXT_HASH_MAP=0
 if test "$ac_cv_cxx_have_std_ext_hash_map" = yes
 then
   HAVE_STD_EXT_HASH_MAP=1
 fi
 AC_SUBST(HAVE_STD_EXT_HASH_MAP)])

AC_DEFUN([AC_CXX_HAVE_GNU_EXT_HASH_MAP],
[AC_CACHE_CHECK([whether the compiler has <ext/hash_map> defining template class __gnu_cxx::hash_map],
 ac_cv_cxx_have_gnu_ext_hash_map,
 [AC_REQUIRE([AC_CXX_NAMESPACES])
  AC_LANG_SAVE
  AC_LANG_CPLUSPLUS
  AC_TRY_COMPILE([#include <ext/hash_map>
#ifdef HAVE_NAMESPACES
using namespace __gnu_cxx;
#endif],[hash_map<int,int> t; ],
  [ac_cv_cxx_have_gnu_ext_hash_map=yes],[ac_cv_cxx_have_gnu_ext_hash_map=no])
  AC_LANG_RESTORE])
 HAVE_GNU_EXT_HASH_MAP=0
 if test "$ac_cv_cxx_have_gnu_ext_hash_map" = yes
 then
   HAVE_GNU_EXT_HASH_MAP=1
 fi
 AC_SUBST(HAVE_GNU_EXT_HASH_MAP)])

AC_DEFUN([AC_CXX_HAVE_GLOBAL_HASH_MAP],
[AC_CACHE_CHECK([whether the compiler has <hash_map> defining template class ::hash_map],
 ac_cv_cxx_have_global_hash_map,
 [AC_REQUIRE([AC_CXX_NAMESPACES])
  AC_LANG_SAVE
  AC_LANG_CPLUSPLUS
  AC_TRY_COMPILE([#include <hash_map>],[hash_map<int,int> t; ],
  [ac_cv_cxx_have_global_hash_map=yes], [ac_cv_cxx_have_global_hash_map=no])
  AC_LANG_RESTORE])
 HAVE_GLOBAL_HASH_MAP=0
 if test "$ac_cv_cxx_have_global_hash_map" = yes
 then
   HAVE_GLOBAL_HASH_MAP=1
 fi
 AC_SUBST(HAVE_GLOBAL_HASH_MAP)])

AC_DEFUN([AC_CXX_HAVE_HASH_MAP],
[AC_CXX_HAVE_STD_EXT_HASH_MAP
 AC_CXX_HAVE_GNU_EXT_HASH_MAP
 AC_CXX_HAVE_GLOBAL_HASH_MAP])


