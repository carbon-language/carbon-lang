#
# Determine if the system can handle the -R option being passed to the linker.
#
# This macro is specific to LLVM.
#
AC_DEFUN([AC_LINK_USE_R],
[AC_CACHE_CHECK([for compiler -Wl,-R<path> option],[llvm_cv_link_use_r],
[ AC_LANG_PUSH([C])
  oldcflags="$CFLAGS"
  CFLAGS="$CFLAGS -Wl,-R."
  AC_LINK_IFELSE([AC_LANG_PROGRAM([[]],[[]])],
    [llvm_cv_link_use_r=yes],[llvm_cv_link_use_r=no])
  CFLAGS="$oldcflags"
  AC_LANG_POP([C])
])
if test "$llvm_cv_link_use_r" = yes ; then
  AC_DEFINE([HAVE_LINK_R],[1],[Define if you can use -Wl,-R. to pass -R. to the linker, in order to add the current directory to the dynamic linker search path.])
  fi
])

#
# Determine if the system can handle the -R option being passed to the linker.
#
# This macro is specific to LLVM.
#
AC_DEFUN([AC_LINK_EXPORT_DYNAMIC],
[AC_CACHE_CHECK([for compiler -Wl,-export-dynamic option],
                [llvm_cv_link_use_export_dynamic],
[ AC_LANG_PUSH([C])
  oldcflags="$CFLAGS"
  CFLAGS="$CFLAGS -Wl,-export-dynamic"
  AC_LINK_IFELSE([AC_LANG_PROGRAM([[]],[[]])],
    [llvm_cv_link_use_export_dynamic=yes],[llvm_cv_link_use_export_dynamic=no])
  CFLAGS="$oldcflags"
  AC_LANG_POP([C])
])
if test "$llvm_cv_link_use_export_dynamic" = yes ; then
  AC_DEFINE([HAVE_LINK_EXPORT_DYNAMIC],[1],[Define if you can use -Wl,-export-dynamic.])
  fi
])

#
# Determine if the system can handle the -retain-symbols-file option being
# passed to the linker.
#
# This macro is specific to LLVM.
#
AC_DEFUN([AC_LINK_RETAIN_SYMBOLS_FILE],
[AC_CACHE_CHECK([for compiler -Wl,-retain-symbols-file option],
                [llvm_cv_link_use_retain_symbols_file],
[ AC_LANG_PUSH([C])
  oldcflags="$CFLAGS"

  # The following code is from the autoconf manual,
  # "11.13: Limitations of Usual Tools".
  # Create a temporary directory $tmp in $TMPDIR (default /tmp).
  # Use mktemp if possible; otherwise fall back on mkdir,
  # with $RANDOM to make collisions less likely.
  : ${TMPDIR=/tmp}
  {
    tmp=`
      (umask 077 && mktemp -d "$TMPDIR/fooXXXXXX") 2>/dev/null
    ` &&
    test -n "$tmp" && test -d "$tmp"
  } || {
    tmp=$TMPDIR/foo$$-$RANDOM
    (umask 077 && mkdir "$tmp")
  } || exit $?

  echo "main" > "$tmp/exports"

  CFLAGS="$CFLAGS -Wl,-retain-symbols-file=$tmp/exports"
  AC_LINK_IFELSE([AC_LANG_PROGRAM([[]],[[]])],
    [llvm_cv_link_use_retain_symbols_file=yes],[llvm_cv_link_use_retain_symbols_file=no])
  rm "$tmp/exports"
  rmdir "$tmp"
  CFLAGS="$oldcflags"
  AC_LANG_POP([C])
])
if test "$llvm_cv_link_use_retain_symbols_file" = yes ; then
  AC_SUBST(HAVE_LINK_RETAIN_SYMBOLS_FILE,1)
  fi
])

