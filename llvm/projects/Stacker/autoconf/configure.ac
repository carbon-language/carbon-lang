dnl **************************************************************************
dnl * Initialize
dnl **************************************************************************
AC_INIT([[[Stacker]]],[[[1.0]]],[rspencer@x10sys.com])

dnl Place all of the extra autoconf files into the config subdirectory
AC_CONFIG_AUX_DIR([autoconf])

dnl Verify that the source directory is valid
AC_CONFIG_SRCDIR([lib/compiler/StackerParser.y])

AC_CONFIG_FILES([Makefile.common])

dnl Configure Makefiles
dnl List every Makefile that exists within your source tree

AC_CONFIG_MAKEFILE(Makefile)
AC_CONFIG_MAKEFILE(lib/Makefile)
AC_CONFIG_MAKEFILE(lib/compiler/Makefile)
AC_CONFIG_MAKEFILE(lib/runtime/Makefile)
AC_CONFIG_MAKEFILE(test/Makefile)
AC_CONFIG_MAKEFILE(tools/Makefile)
AC_CONFIG_MAKEFILE(tools/stkrc/Makefile)

dnl **************************************************************************
dnl * Determine which system we are building on
dnl **************************************************************************

dnl **************************************************************************
dnl * Check for programs.
dnl **************************************************************************
AC_PROG_LIBTOOL
dnl Get libtool's idea of what the shared library suffix is.
dnl (This is a hack; it relies on undocumented behavior.)
AC_MSG_CHECKING([for shared library suffix])
eval "SHLIBEXT=$shrext"
AC_MSG_RESULT($SHLIBEXT)
dnl Propagate it to the Makefiles and config.h (for gccld & bugpoint).
AC_SUBST(SHLIBEXT,$SHLIBEXT)
AC_DEFINE_UNQUOTED(SHLIBEXT,"$SHLIBEXT",
                   [Extension that shared libraries have,
                   e.g., ".so".])
                                                                                                                                            


dnl **************************************************************************
dnl * Check for libraries.
dnl **************************************************************************

dnl **************************************************************************
dnl * Checks for header files.
dnl **************************************************************************

dnl **************************************************************************
dnl * Checks for typedefs, structures, and compiler characteristics.
dnl **************************************************************************

dnl **************************************************************************
dnl * Checks for library functions.
dnl **************************************************************************

dnl **************************************************************************
dnl * Enable various compile-time options
dnl **************************************************************************

dnl **************************************************************************
dnl * Set the location of various third-party software packages
dnl **************************************************************************

dnl Location of LLVM source code
AC_ARG_WITH(llvmsrc,AC_HELP_STRING([--with-llvmsrc],[Location of LLVM Source Code]),AC_SUBST(LLVM_SRC,[$withval]),AC_SUBST(LLVM_SRC,[`cd ${srcdir}/../..; pwd`]))

dnl Location of LLVM object code
AC_ARG_WITH(llvmobj,AC_HELP_STRING([--with-llvmobj],[Location of LLVM Object Code]),AC_SUBST(LLVM_OBJ,[$withval]),AC_SUBST(LLVM_OBJ,[`cd ../..; pwd`]))

dnl **************************************************************************
dnl * Create the output files
dnl **************************************************************************
AC_OUTPUT
