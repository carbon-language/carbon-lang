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

dnl Place all of the extra autoconf files into the config subdirectory
AC_CONFIG_AUX_DIR([autoconf])

dnl Configure a header file
AC_CONFIG_HEADERS(include/Config/config.h)
AC_CONFIG_MAKEFILE(Makefile)
AC_CONFIG_MAKEFILE(Makefile.common)
AC_CONFIG_MAKEFILE(Makefile.rules)
AC_CONFIG_MAKEFILE(lib/Makefile)
AC_CONFIG_MAKEFILE(lib/Analysis/Makefile)
AC_CONFIG_MAKEFILE(lib/Analysis/DataStructure/Makefile)
AC_CONFIG_MAKEFILE(lib/Analysis/IPA/Makefile)
AC_CONFIG_MAKEFILE(lib/Analysis/LiveVar/Makefile)
AC_CONFIG_MAKEFILE(lib/AsmParser/Makefile)
AC_CONFIG_MAKEFILE(lib/Bytecode/Makefile)
AC_CONFIG_MAKEFILE(lib/Bytecode/Reader/Makefile)
AC_CONFIG_MAKEFILE(lib/Bytecode/Writer/Makefile)
AC_CONFIG_MAKEFILE(lib/CWriter/Makefile)
AC_CONFIG_MAKEFILE(lib/CodeGen/Makefile)
AC_CONFIG_MAKEFILE(lib/CodeGen/InstrSched/Makefile)
AC_CONFIG_MAKEFILE(lib/CodeGen/InstrSelection/Makefile)
AC_CONFIG_MAKEFILE(lib/CodeGen/ModuloScheduling/Makefile)
AC_CONFIG_MAKEFILE(lib/CodeGen/RegAlloc/Makefile)
AC_CONFIG_MAKEFILE(lib/CodeGen/SelectionDAG/Makefile)
AC_CONFIG_MAKEFILE(lib/ExecutionEngine/Makefile)
AC_CONFIG_MAKEFILE(lib/ExecutionEngine/Interpreter/Makefile)
AC_CONFIG_MAKEFILE(lib/ExecutionEngine/JIT/Makefile)
AC_CONFIG_MAKEFILE(lib/Support/Makefile)
AC_CONFIG_MAKEFILE(lib/Target/Makefile)
AC_CONFIG_MAKEFILE(lib/Target/Sparc/Makefile)
AC_CONFIG_MAKEFILE(lib/Target/X86/Makefile)
AC_CONFIG_MAKEFILE(lib/Transforms/Makefile)
AC_CONFIG_MAKEFILE(lib/Transforms/Hello/Makefile)
AC_CONFIG_MAKEFILE(lib/Transforms/IPO/Makefile)
AC_CONFIG_MAKEFILE(lib/Transforms/Instrumentation/Makefile)
AC_CONFIG_MAKEFILE(lib/Transforms/Instrumentation/ProfilePaths/Makefile)
AC_CONFIG_MAKEFILE(lib/Transforms/Scalar/Makefile)
AC_CONFIG_MAKEFILE(lib/Transforms/Utils/Makefile)
AC_CONFIG_MAKEFILE(lib/VMCore/Makefile)
AC_CONFIG_MAKEFILE(runtime/Makefile)
AC_CONFIG_MAKEFILE(runtime/GCCLibraries/Makefile)
AC_CONFIG_MAKEFILE(runtime/GCCLibraries/crtend/Makefile)
AC_CONFIG_MAKEFILE(runtime/GCCLibraries/libc/Makefile)
AC_CONFIG_MAKEFILE(runtime/GCCLibraries/libcurses/Makefile)
AC_CONFIG_MAKEFILE(runtime/GCCLibraries/libexception/Makefile)
AC_CONFIG_MAKEFILE(runtime/GCCLibraries/libg/Makefile)
AC_CONFIG_MAKEFILE(runtime/GCCLibraries/libgcc/Makefile)
AC_CONFIG_MAKEFILE(runtime/GCCLibraries/libm/Makefile)
AC_CONFIG_MAKEFILE(runtime/GCCLibraries/libmalloc/Makefile)
AC_CONFIG_MAKEFILE(runtime/GCCLibraries/libtermcap/Makefile)
AC_CONFIG_MAKEFILE(runtime/GCCLibraries/libucb/Makefile)
AC_CONFIG_MAKEFILE(runtime/GCCLibraries/libutempter/Makefile)
AC_CONFIG_MAKEFILE(runtime/GCCLibraries/libutil/Makefile)
AC_CONFIG_MAKEFILE(runtime/libdummy/Makefile)
AC_CONFIG_MAKEFILE(runtime/libtrace/Makefile)
AC_CONFIG_MAKEFILE(test/Makefile)
AC_CONFIG_MAKEFILE(test/Makefile.tests)
AC_CONFIG_MAKEFILE(test/Programs/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/Makefile.programs)
AC_CONFIG_MAKEFILE(test/Programs/TEST.aa.Makefile)
AC_CONFIG_MAKEFILE(test/Programs/TEST.dsgraph.report)
AC_CONFIG_MAKEFILE(test/Programs/TEST.micro.report)
AC_CONFIG_MAKEFILE(test/Programs/TEST.aa.report)
AC_CONFIG_MAKEFILE(test/Programs/TEST.example.Makefile)
AC_CONFIG_MAKEFILE(test/Programs/TEST.nightly.Makefile)
AC_CONFIG_MAKEFILE(test/Programs/TEST.buildrepo.Makefile)
AC_CONFIG_MAKEFILE(test/Programs/TEST.jit.Makefile)
AC_CONFIG_MAKEFILE(test/Programs/TEST.nightly.report)
AC_CONFIG_MAKEFILE(test/Programs/TEST.dsgraph.Makefile)
AC_CONFIG_MAKEFILE(test/Programs/TEST.jit.report)
AC_CONFIG_MAKEFILE(test/Programs/TEST.typesafe.Makefile)
AC_CONFIG_MAKEFILE(test/Programs/TEST.dsgraph.gnuplot)
AC_CONFIG_MAKEFILE(test/Programs/TEST.micro.Makefile)
AC_CONFIG_MAKEFILE(test/Programs/External/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/External/SPEC/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/External/SPEC/Makefile.spec)
AC_CONFIG_MAKEFILE(test/Programs/External/SPEC/CFP2000/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/External/SPEC/CFP2000/179.art/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/External/SPEC/CFP2000/183.equake/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/External/SPEC/CFP2000/188.ammp/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/External/SPEC/CINT2000/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/External/SPEC/CINT2000/164.gzip/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/External/SPEC/CINT2000/175.vpr/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/External/SPEC/CINT2000/176.gcc/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/External/SPEC/CINT2000/181.mcf/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/External/SPEC/CINT2000/186.crafty/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/External/SPEC/CINT2000/197.parser/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/External/SPEC/CINT2000/254.gap/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/External/SPEC/CINT2000/255.vortex/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/External/SPEC/CINT2000/256.bzip2/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/External/SPEC/CINT2000/300.twolf/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/LLVMSource/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/MultiSource/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/MultiSource/Makefile.multisrc)
AC_CONFIG_MAKEFILE(test/Programs/MultiSource/Applications/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/MultiSource/Applications/Burg/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/MultiSource/Applications/aha/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/MultiSource/Applications/sgefa/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/MultiSource/Benchmarks/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/MultiSource/Benchmarks/Fhourstones/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/MultiSource/Benchmarks/McCat/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/MultiSource/Benchmarks/McCat/01-qbsort/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/MultiSource/Benchmarks/McCat/03-testtrie/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/MultiSource/Benchmarks/McCat/04-bisect/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/MultiSource/Benchmarks/McCat/05-eks/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/MultiSource/Benchmarks/McCat/08-main/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/MultiSource/Benchmarks/McCat/09-vor/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/MultiSource/Benchmarks/McCat/12-IOtest/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/MultiSource/Benchmarks/McCat/15-trie/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/MultiSource/Benchmarks/McCat/17-bintr/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/MultiSource/Benchmarks/McCat/18-imp/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/MultiSource/Benchmarks/Olden/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/MultiSource/Benchmarks/Olden/bh/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/MultiSource/Benchmarks/Olden/bisort/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/MultiSource/Benchmarks/Olden/em3d/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/MultiSource/Benchmarks/Olden/health/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/MultiSource/Benchmarks/Olden/mst/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/MultiSource/Benchmarks/Olden/perimeter/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/MultiSource/Benchmarks/Olden/power/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/MultiSource/Benchmarks/Olden/treeadd/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/MultiSource/Benchmarks/Olden/tsp/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/MultiSource/Benchmarks/Olden/voronoi/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/MultiSource/Benchmarks/OptimizerEval/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/MultiSource/Benchmarks/Ptrdist/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/MultiSource/Benchmarks/Ptrdist/anagram/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/MultiSource/Benchmarks/Ptrdist/bc/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/MultiSource/Benchmarks/Ptrdist/ft/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/MultiSource/Benchmarks/Ptrdist/ks/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/MultiSource/Benchmarks/Ptrdist/yacr2/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/MultiSource/Benchmarks/llubenchmark/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/MultiSource/Benchmarks/sim/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/NoSource/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/NoSource/Flex/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/NoSource/Larn/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/NoSource/Moria-5.5.2/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/NoSource/Povray31/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/NoSource/m4/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/SingleSource/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/SingleSource/Makefile.singlesrc)
AC_CONFIG_MAKEFILE(test/Programs/SingleSource/Gizmos/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/SingleSource/Benchmarks/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/SingleSource/Benchmarks/Dhrystone/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/SingleSource/Benchmarks/Shootout/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/SingleSource/Benchmarks/Stanford/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/SingleSource/Benchmarks/Misc/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/SingleSource/CustomChecked/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/SingleSource/UnitTests/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/SingleSource/UnitTests/SetjmpLongjmp/Makefile)
AC_CONFIG_MAKEFILE(tools/Makefile)
AC_CONFIG_MAKEFILE(tools/analyze/Makefile)
AC_CONFIG_MAKEFILE(tools/as/Makefile)
AC_CONFIG_MAKEFILE(tools/bugpoint/Makefile)
AC_CONFIG_MAKEFILE(tools/dis/Makefile)
AC_CONFIG_MAKEFILE(tools/extract/Makefile)
AC_CONFIG_MAKEFILE(tools/gccas/Makefile)
AC_CONFIG_MAKEFILE(tools/gccld/Makefile)
AC_CONFIG_MAKEFILE(tools/link/Makefile)
AC_CONFIG_MAKEFILE(tools/llc/Makefile)
AC_CONFIG_MAKEFILE(tools/lli/Makefile)
AC_CONFIG_MAKEFILE(tools/opt/Makefile)
AC_CONFIG_MAKEFILE(tools/llee/Makefile)
AC_CONFIG_MAKEFILE(tools/llvm-as/Makefile)
AC_CONFIG_MAKEFILE(tools/llvm-dis/Makefile)
AC_CONFIG_MAKEFILE(utils/Makefile)
AC_CONFIG_MAKEFILE(utils/Burg/Makefile)
AC_CONFIG_MAKEFILE(utils/Burg/Doc/Makefile)
AC_CONFIG_MAKEFILE(utils/TableGen/Makefile)
AC_CONFIG_MAKEFILE(www/docs/Makefile)
AC_CONFIG_MAKEFILE(projects/Makefile)
AC_CONFIG_MAKEFILE(projects/sample/Makefile)
AC_CONFIG_MAKEFILE(projects/sample/Makefile.common)
AC_CONFIG_MAKEFILE(projects/sample/Makefile.config)
AC_CONFIG_MAKEFILE(projects/sample/lib/Makefile)
AC_CONFIG_MAKEFILE(projects/sample/lib/sample/Makefile)
AC_CONFIG_MAKEFILE(projects/sample/tools/Makefile)
AC_CONFIG_MAKEFILE(projects/sample/tools/sample/Makefile)
AC_CONFIG_MAKEFILE(projects/ModuleMaker/Makefile)
AC_CONFIG_MAKEFILE(projects/ModuleMaker/Makefile.common)
AC_CONFIG_MAKEFILE(projects/ModuleMaker/tools/Makefile)
AC_CONFIG_MAKEFILE(projects/ModuleMaker/tools/ModuleMaker/Makefile)

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
	         AC_SUBST(LLVMGCCDIR,[/home/vadve/lattner/local/x86/llvm-gcc/])
	         ;;

	*sparc*) AC_SUBST(OS,[SunOS])
	         AC_SUBST(LLVMGCCDIR,[/home/vadve/lattner/local/sparc/llvm-gcc/])
	         ;;

	*)       AC_SUBST(OS,[Unknown])
	         ;;
esac

dnl
dnl If we are targetting a Sparc machine running Solaris, pretend that it is
dnl V9, since that is all that we support at the moment, and autoconf will only
dnl tell us we're a sparc.
dnl
case $target in
	*sparc*solaris*)  AC_SUBST(target,[[sparcv9-sun-solaris2.8]])
	                  ;;
esac

dnl
dnl Determine what our target architecture is and configure accordingly.
dnl This will allow Makefiles to make a distinction between the hardware and
dnl the OS.
dnl
case $target in
	*i*86*)           AC_SUBST(ARCH,[x86])
	                  ;;
	*sparc*solaris*)  AC_SUBST(ARCH,[Sparc])
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

dnl Verify that GCC is version 3.0 or higher
gccmajor=`$CC --version | head -n 1 | awk '{print $NF;}' | cut -d. -f1`
if test "$gccmajor" -lt "3"
then
	AC_MSG_ERROR([gcc 3.x required])
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
AC_PATH_PROG(RPWD,[pwd])
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
AC_PATH_PROG(PYTHON,[python])
AC_PATH_PROG(QMTEST,[qmtest])

dnl Verify that the version of python available is high enough for qmtest
pyversion=`$PYTHON -V 2>&1 | cut -d\  -f2`
pymajor=`echo $pyversion | cut -d. -f1`
pyminor=`echo $pyversion | cut -d. -f2`

if test "$pymajor" -ge "2"
then
	if test "$pymajor" -eq "2"
	then
		if test "$pyminor" -lt "2"
		then
			AC_MSG_ERROR([Python 2.2 or greater required])
		fi
	fi
else
	AC_MSG_ERROR([Python 2.2 or greater required])
fi

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

dnl Purify Option
AC_ARG_ENABLE(purify,AC_HELP_STRING([--enable-purify],[Compile with purify (default is NO)]),,enableval="no")
if test ${enableval} = "no"
then
	AC_SUBST(ENABLE_PURIFY,[[]])
else
	AC_SUBST(ENABLE_PURIFY,[[ENABLE_PURIFY=1]])
fi

dnl Optimized Option
AC_ARG_ENABLE(optimized,AC_HELP_STRING([--enable-optimized],[Compile with optimizations enabled (default is NO)]),,enableval=no)
if test ${enableval} = "no"
then
	AC_SUBST(ENABLE_OPTIMIZED,[[]])
else
	AC_SUBST(ENABLE_OPTIMIZED,[[ENABLE_OPTIMIZED=1]])
fi

dnl Spec Benchmarks
AC_ARG_ENABLE(spec,AC_HELP_STRING([--enable-spec],[Compile SPEC benchmarks (default is NO)]),,enableval=no)
if test ${enableval} = "no"
then
	AC_SUBST(USE_SPEC,[[]])
else
	AC_SUBST(USE_SPEC,[[USE_SPEC=1]])
fi

dnl Precompiled Bytecode Option
AC_ARG_ENABLE(precompiled_bytecode,AC_HELP_STRING([--enable-precompiled_bytecode],[Use pre-compiled bytecode (default is NO)]),,enableval=no)
if test ${enableval} = "no"
then
	AC_SUBST(UPB,[[]])
else
	AC_SUBST(UPB,[[USE_PRECOMPILED_BYTECODE=1]])
fi


dnl LLC Diff Option
AC_ARG_ENABLE(llc_diffs,AC_HELP_STRING([--enable-llc_diffs],[Enable LLC Diffs when testing (default is YES)]),,enableval=yes)
if test ${enableval} = "no"
then
	AC_SUBST(DISABLE_LLC_DIFFS,[DISABLE_LLC_DIFFS:=1])
else
	AC_SUBST(DISABLE_LLC_DIFFS,[[]])
fi

dnl JIT Option
AC_ARG_ENABLE(jit,AC_HELP_STRING([--enable-jit],[Enable Just In Time Compiling (default is YES)]),,enableval=default)

if test ${enableval} = "no"
then
	AC_SUBST(JIT,[[]])
else
	case $target in
		*i*86*)
			AC_SUBST(JIT,[[TARGET_HAS_JIT=1]])
			;;
		*sparc*)
			AC_SUBST(JIT,[[TARGET_HAS_JIT=1]])
			;;
		*)
			AC_SUBST(JIT,[[]])
			;;
	esac
fi

dnl **************************************************************************
dnl * Set the location of various third-party software packages
dnl **************************************************************************

dnl Location of SPEC benchmarks
AC_ARG_WITH(spec,AC_HELP_STRING([--with-spec],[Location of SPEC benchmarks]),AC_SUBST(SPEC_ROOT,[$withval]),AC_SUBST(SPEC_ROOT,[/home/vadve/shared/benchmarks/speccpu2000/benchspec]))

dnl Location of the LLVM C front end
AC_ARG_WITH(llvmgccdir,AC_HELP_STRING([--with-llvmgccdir],[Location of LLVM GCC front-end]),AC_SUBST(LLVMGCCDIR,[$withval]))

dnl Location of the bytecode repository
AC_ARG_WITH(bcrepos,AC_HELP_STRING([--with-bcrepos],[Location of Bytecode Repository]),AC_SUBST(BCR,[$withval]),AC_SUBST(BCR,[/home/vadve/lattner/LLVMPrograms]))

dnl Location of PAPI
AC_ARG_WITH(papi,AC_HELP_STRING([--with-papi],[Location of PAPI]),AC_SUBST(PAPIDIR,[$withval]),AC_SUBST(PAPIDIR,[/home/vadve/shared/Sparc/papi-2.3.4.1]))

dnl Location of the purify program
AC_ARG_WITH(purify,AC_HELP_STRING([--with-purify],[Location of purify program]),AC_SUBST(PURIFY,[$withval]))

dnl **************************************************************************
dnl * Configure other software packages (via AC_CONFIG_SUBDIRS)
dnl **************************************************************************

dnl **************************************************************************
dnl * Create the output files
dnl **************************************************************************
AC_OUTPUT(Makefile.config)
