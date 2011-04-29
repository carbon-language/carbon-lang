dnl **************************************************************************
dnl * Initialize
dnl **************************************************************************
AC_INIT([Polly],[0.01],[polly-dev@googlegroups.com])

dnl Identify where LLVM source tree is
LLVM_SRC_ROOT="`(cd $srcdir/../..; pwd)`"
LLVM_OBJ_ROOT="`(cd ../..; pwd)`"

dnl Tell autoconf that this is an LLVM project being configured
dnl This provides the --with-llvmsrc and --with-llvmobj options
LLVM_CONFIG_PROJECT($LLVM_SRC_ROOT,$LLVM_OBJ_ROOT)

dnl Tell autoconf that the auxilliary files are actually located in
dnl the LLVM autoconf directory, not here.
AC_CONFIG_AUX_DIR($LLVM_SRC/autoconf)

dnl Indicate that we require autoconf 2.59 or later. Ths is needed because we
dnl use some autoconf macros only available in 2.59.
AC_PREREQ(2.59)

dnl Verify that the source directory is valid
AC_CONFIG_SRCDIR(["lib/Analysis/ScopInfo.cpp"])

dnl Configure a common Makefile
AC_CONFIG_FILES(Makefile.config)
AC_CONFIG_FILES(Makefile.common)

dnl Configure project makefiles
dnl List every Makefile that exists within your source tree

dnl Quit if the source directory has already been configured.
dnl NOTE: This relies upon undocumented autoconf behavior.
if test ${srcdir} != "." ; then
  if test -f ${srcdir}/include/polly/Config/config.h ; then
    AC_MSG_ERROR([Already configured in ${srcdir}])
  fi
fi

AC_DEFINE([CLOOG_INT_GMP], [1], [Use gmp for isl])

dnl **************************************************************************
dnl * Determine which system we are building on
dnl **************************************************************************

dnl **************************************************************************
dnl * Check for programs.
dnl **************************************************************************

dnl AC_PROG_CPP
dnl AC_PROG_CC(gcc)
dnl AC_PROG_CXX(g++)

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
dnl Find Gmp
find_lib_and_headers([gmp], [gmp.h], [gmp], [required])

dnl Find Isl
find_lib_and_headers([isl], [isl/config.h], [isl], [required])

dnl Find cloog
saved_CXXFLAGS=$CXXFLAGS
CXXFLAGS="$CXXFLAGS $isl_inc"
find_lib_and_headers([cloog], [cloog/isl/cloog.h], [cloog-isl], [required])
CXXFLAGS=$saved_CXXFLAGS


dnl Check if Scoplib there
find_lib_and_headers([openscop], [openscop/scop.h], [openscop])
AS_IF([test "x$openscop_found" = "xyes"],
  [AC_DEFINE([OPENSCOP_FOUND],[1],[Define if openscop found])])

dnl Check if Scoplib there
find_lib_and_headers([scoplib], [scoplib/scop.h], [scoplib])
AS_IF([test "x$scoplib_found" = "xyes"],
  [AC_DEFINE([SCOPLIB_FOUND],[1],[Define if scoplib found])])

if test "x$scoplib_found" = "xyes"; then :
  scoplib_rpath="-Wl,-rpath=$given_lib_path"
else
  scoplib_rpath=""
fi

AC_SUBST(scoplib_rpath)

dnl **************************************************************************
dnl * Create the output files
dnl **************************************************************************

dnl This must be last
AC_CONFIG_HEADERS(include/polly/Config/config.h)
AC_OUTPUT
