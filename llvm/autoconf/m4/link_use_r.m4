#
# Determine if the system can handle the -R option being passed to the linker.
#
# This macro is specific to LLVM.
#
AC_DEFUN([AC_LINK_USE_R],
[
  AC_MSG_CHECKING([for compiler -Wl,-R<path> option])
  AC_LANG_SAVE
  AC_LANG_C
  oldcflags="$CFLAGS"
  CFLAGS="$CFLAGS -Wl,-R."
  AC_LINK_IFELSE([int main() { return 0; }],[ac_cv_link_use_r=yes],[ac_cv_link_use_r=no])
  CFLAGS="$oldcflags"
  AC_LANG_RESTORE
  AC_MSG_RESULT($ac_cv_link_use_r)
  if test "$ac_cv_link_use_r" = yes
  then
    AC_DEFINE([HAVE_LINK_R],[1],[Define if you can use -Wl,-R. to pass -R. to the linker, in order to add the current directory to the dynamic linker search path.])
  fi
])


