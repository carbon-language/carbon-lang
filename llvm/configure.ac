dnl Autoconf requirements
dnl AC_INIT(package, version, bug-report-address)
dnl information on the package
dnl checks for programs
dnl checks for libraries
dnl checks for header files
dnl checks for types
dnl checks for structures
dnl checks for compiler characteristics
dnl checks for library functions
dnl checks for system services
dnl AC_CONFIG_FILES([file...])
dnl AC_OUTPUT

dnl **************************************************************************
dnl * Initialize
dnl **************************************************************************
AC_INIT([[[LLVM]]],[[[1.0]]],[llvmbugs@cs.uiuc.edu])

dnl * Configure a header file
AC_CONFIG_HEADERS(include/Config/config.h)

dnl **************************************************************************
dnl * Determine which system we are building on
dnl **************************************************************************

dnl Check the install program (needs to be done before canonical stuff)
AC_PROG_INSTALL

dnl Check which host for which we're compiling.  This will tell us which LLVM
dnl compiler will be used for compiling SSA into object code.
AC_CANONICAL_TARGET

dnl
dnl Now, for some of our own magic:
dnl We will use the build machine information to set some variables.
dnl
case $build in
	*i*86*)  AC_SUBST(OS,[Linux])
	         AC_SUBST(DISABLE_LLC_DIFFS,[[DISABLE_LLC_DIFFS:=1]])
	         AC_SUBST(LLVMGCCDIR,[/home/vadve/lattner/local/x86/llvm-gcc/])
	         ;;

	*sparc*) AC_SUBST(OS,[SunOS])
	         AC_SUBST(LLVMGCCDIR,[/home/vadve/lattner/local/sparc/llvm-gcc/])
	         ;;

	*)       AC_SUBST(OS,[Unknown])
	         ;;
esac

dnl
dnl If we are on a Solaris machine, pretend that it is V9, since that is all
dnl that we support at the moment, and autoconf will only tell us we're a
dnl sparc.
dnl
case $target in
	*sparc*solaris*)  AC_SUBST(target,[[sparcv9-sun-solaris2.8]])
	                  ;;
esac

dnl **************************************************************************
dnl * Check for programs.
dnl **************************************************************************

dnl Check for compilation tools
AC_PROG_CXX
AC_PROG_CC(gcc)
AC_PROG_CPP

dnl Ensure that compilation tools are GCC; we use GCC specific extensions
if test "$GCC" != "yes"
then
	AC_MSG_ERROR([gcc required but not found])
fi

if test "$GXX" != "yes"
then
	AC_MSG_ERROR([g++ required but not found])
fi

dnl Check for GNU Make.  We use its extensions to, so don't build without it
CHECK_GNU_MAKE
if test -z "$_cv_gnu_make_command"
then
	AC_MSG_ERROR([GNU Make required but not found])
fi

dnl Check for compiler-compiler tools (reminds me of Little Caesar's Pizza)
AC_PROG_FLEX
AC_PROG_BISON

dnl Check for libtool
AC_PROG_LIBTOOL

dnl Check for our special programs
AC_PATH_PROG(AR,[ar])
AC_PATH_PROG(SED,[sed])
AC_PATH_PROG(RM,[rm])
AC_PATH_PROG(ECHO,[echo])
AC_PATH_PROG(MKDIR,[mkdir])
AC_PATH_PROG(DATE,[date])
AC_PATH_PROG(MV,[mv])
AC_PATH_PROG(DOT,[dot])
AC_PATH_PROG(ETAGS,[etags])
AC_PATH_PROG(PURIFY,[purify])

dnl Verify that the source directory is valid
AC_CONFIG_SRCDIR(["Makefile.config.in"])

dnl **************************************************************************
dnl * Check for libraries.
dnl **************************************************************************

dnl libelf is for sparc only; we can ignore it if we don't have it
AC_CHECK_LIB(elf, elf_begin)

dnl dlopen() is required.  If we don't find it, quit.
AC_SEARCH_LIBS(dlopen,dl,,AC_MSG_ERROR([dlopen() required but not found]))

dnl mallinfo is optional; the code can compile (minus features) without it
AC_SEARCH_LIBS(mallinfo,malloc,AC_DEFINE([HAVE_MALLINFO],[1]))

dnl
dnl The math libraries are used by the test code, but not by the actual LLVM
dnl code.
dnl
dnl AC_CHECK_LIB(m, cos)

dnl **************************************************************************
dnl * Checks for header files.
dnl *	Chances are, if the standard C or POSIX type header files are missing,
dnl *	then LLVM just isn't going to compile.  However, it is possible that
dnl *	the necessary functions/macros will be included from other
dnl *	(non-standard and non-obvious) header files.
dnl *
dnl *	So, we'll be gracious, give it a chance, and try to go on without
dnl *	them.
dnl **************************************************************************
AC_HEADER_STDC
AC_HEADER_SYS_WAIT

dnl Check for ANSI C/POSIX header files
AC_CHECK_HEADERS(assert.h fcntl.h limits.h sys/time.h unistd.h errno.h signal.h math.h)

dnl Check for system specific header files
AC_CHECK_HEADERS(malloc.h strings.h sys/mman.h sys/resource.h)

dnl Check for header files associated with dlopen and friends
AC_CHECK_HEADERS(dlfcn.h link.h)

dnl **************************************************************************
dnl * Checks for typedefs, structures, and compiler characteristics.
dnl **************************************************************************

dnl Check for const and inline keywords
AC_C_CONST
AC_C_INLINE

dnl Check for machine endian-ness
AC_C_BIGENDIAN(AC_DEFINE([ENDIAN_BIG]),AC_DEFINE(ENDIAN_LITTLE))

dnl Check for types
AC_TYPE_PID_T
AC_TYPE_SIZE_T
AC_CHECK_TYPES([int64_t],,AC_MSG_ERROR([Type int64_t required but not found]))
AC_CHECK_TYPES([uint64_t],,AC_MSG_ERROR([Type uint64_t required but not found]))
AC_HEADER_TIME
AC_STRUCT_TM

dnl Check for C++ extensions
AC_CXX_HAVE_EXT_HASH_MAP
AC_CXX_HAVE_EXT_HASH_SET
AC_CXX_HAVE_EXT_SLIST
AC_CXX_HAVE_STD_ITERATOR
AC_CXX_HAVE_BI_ITERATOR
AC_CXX_HAVE_FWD_ITERATOR

dnl **************************************************************************
dnl * Checks for library functions.
dnl **************************************************************************
AC_FUNC_ALLOCA
AC_PROG_GCC_TRADITIONAL
AC_FUNC_MEMCMP
AC_FUNC_MMAP
AC_FUNC_MMAP_FILE
if test ${ac_cv_func_mmap_file} = "no"
then
	AC_MSG_ERROR([mmap() of files required but not found])
fi
AC_HEADER_MMAP_ANONYMOUS
AC_TYPE_SIGNAL
AC_CHECK_FUNCS(getcwd gettimeofday strcspn strdup strerror strspn strstr strtod strtol)

dnl
dnl Need to check mmap for MAP_PRIVATE, MAP_ANONYMOUS, MAP_ANON, MAP_FIXED
dnl MAP_FIXED is only needed for Sparc
dnl MAP_ANON is used for Sparc and BSD
dnl Everyone should have MAP_PRIVATE
dnl

dnl Check for certain functions (even if we've already found them) so that we
dnl can quit with an error if they are unavailable.
dnl
dnl As the code is made more portable (i.e. less reliant on these functions,
dnl these checks should go away.
AC_CHECK_FUNC(mmap,,AC_MSG_ERROR([Function mmap() required but not found]))
AC_CHECK_FUNC(mprotect,,AC_MSG_ERROR([Function mprotect() required but not found]))

dnl **************************************************************************
dnl * Enable various compile-time options
dnl **************************************************************************
AC_ARG_ENABLE(purify,AC_HELP_STRING([--enable-purify],[Compile with purify (default is NO)]), AC_SUBST(ENABLE_PURIFY,[[ENABLE_PURIFY=1]]), AC_SUBST(ENABLE_PURIFY,[[]]))
AC_ARG_ENABLE(verbose,AC_HELP_STRING([--enable-verbose],[Enable verbose compilation messages (default is NO)]), AC_SUBST(ENABLE_VERBOSE,[[VERBOSE=1]]), AC_SUBST(ENABLE_VERBOSE,[[]]))
AC_ARG_ENABLE(profiling,AC_HELP_STRING([--enable-profiling],[Compile with profiling information (default is NO)]), AC_SUBST(ENABLE_PROFILING,[[ENABLE_PROFILING=1]]), AC_SUBST(ENABLE_PROFILING,[[]]))
AC_ARG_ENABLE(optimized,AC_HELP_STRING([--enable-optimized],[Compile with optimizations enabled (default is NO)]), AC_SUBST(ENABLE_OPTIMIZED,[[ENABLE_OPTIMIZED=1]]), AC_SUBST(ENABLE_OPTIMIZED,[[]]))
AC_ARG_ENABLE(spec,AC_HELP_STRING([--enable-spec],[Compile SPEC benchmarks (default is NO)]), AC_SUBST(USE_SPEC,[[USE_SPEC=1]]), AC_SUBST(USE_SPEC,[[]]))
AC_ARG_ENABLE(precompiled_bytecode,AC_HELP_STRING([--enable-precompiled_bytecode],[Use pre-compiled bytecode (default is NO)]), AC_SUBST(UPB,[[USE_PRECOMPILED_BYTECODE=1]]), AC_SUBST(UPB,[[]]))
case $build in
	*i*86*)  AC_ARG_ENABLE(jit,AC_HELP_STRING([--enable-jit],[Enable Just In Time Compiling (default is NO)]), AC_SUBST(JIT,[[TARGET_HAS_JIT=1]]), AC_SUBST(JIT,[[]]))
	         ;;
	*)
	         ;;
esac

dnl **************************************************************************
dnl * Set the location of various third-party software packages
dnl **************************************************************************
AC_ARG_WITH(spec,AC_HELP_STRING([--with-spec],[Location of SPEC benchmarks]),AC_SUBST(SPEC_ROOT,[$withval]),AC_SUBST(SPEC_ROOT,[/home/vadve/shared/benchmarks/speccpu2000/benchspec]))
AC_ARG_WITH(llvmgccdir,AC_HELP_STRING([--with-llvmgccdir],[Location of LLVM GCC front-end]),AC_SUBST(LLVMGCCDIR,[$withval]))
AC_ARG_WITH(bcrepos,AC_HELP_STRING([--with-bcrepos],[Location of Bytecode Repository]),AC_SUBST(BCR,[$withval]),AC_SUBST(BCR,[/home/vadve/lattner/LLVMPrograms]))
AC_ARG_WITH(papi,AC_HELP_STRING([--with-papi],[Location of PAPI]),AC_SUBST(PAPIDIR,[$withval]),AC_SUBST(PAPIDIR,[/home/vadve/shared/papi-2.3.4.1]))
AC_ARG_WITH(objroot,AC_HELP_STRING([--with-objroot],[Location where object files should be placed (default is .)]),AC_SUBST(OBJROOT,[$withval]),AC_SUBST(OBJROOT,[.]))
AC_ARG_WITH(purify,AC_HELP_STRING([--with-purify],[Location of purify program]),AC_SUBST(PURIFY,[$withval]))

AC_OUTPUT(Makefile.config)
