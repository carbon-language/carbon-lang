dnl AC_SINGLE_CXX_CHECK(DEFINEVAR, CACHEVAR, FUNCTION, HEADER, PROGRAM)
dnl                     $1,        $2,       $3,       $4,     $5
dnl 
AC_DEFUN([AC_SINGLE_CXX_CHECK],
[AC_CACHE_CHECK([for $3 in $4], [$2],
 [AC_LANG_PUSH([C++])
  AC_COMPILE_IFELSE(AC_LANG_SOURCE([$5]),[$2=yes],[$2=no])
 AC_LANG_POP([C++])])
 if test "$$2" = "yes"
 then
   AC_DEFINE($1, 1, [Define to 1 if your compiler defines $3 in the $4
                     header file.])
 fi])

