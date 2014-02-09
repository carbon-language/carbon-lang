# Combine AC_DEFINE and AC_SUBST
AC_DEFUN([LLVM_DEFINE_SUBST], [
AC_DEFINE([$1], [$2], [$3])
AC_SUBST([$1], ['$2'])
])
