dnl Initialize autoconf
AC_INIT([[llvm]],[[1.4]],[llvmbugs@cs.uiuc.edu])

dnl Place all of the extra autoconf files into the config subdirectory
dnl Tell various tools where the m4 autoconf macros are
dnl Have configure verify that the source directory is valid.
AC_CONFIG_AUX_DIR([autoconf])
dnl AC_CONFIG_MACRO_DIR(autoconf/m4)
dnl Verify that the source directory is valid
AC_CONFIG_SRCDIR(["Makefile.config.in"])

dnl Quit if the source directory has already been configured.
dnl NOTE: This relies upon undocumented autoconf behavior.
if test ${srcdir} != "." ; then
	if test -f ${srcdir}/include/llvm/Config/config.h ; then
		AC_MSG_ERROR([Already configured in ${srcdir}])
	fi
fi

dnl Configure all of the projects present in our source tree.
for i in `ls ${srcdir}/projects`
do
  if test -d ${srcdir}/projects/${i} ; then
    case ${i} in
      "CVS") ;;
      "sample")       AC_CONFIG_SUBDIRS([projects/sample])    ;;
      "Stacker")      AC_CONFIG_SUBDIRS([projects/Stacker])   ;;
      "llvm-test")    AC_CONFIG_SUBDIRS([projects/llvm-test]) ;;
      "llvm-reopt")   AC_CONFIG_SUBDIRS([projects/llvm-reopt]);;
      "llvm-gcc")     AC_CONFIG_SUBDIRS([projects/llvm-gcc])  ;;
      "Java")         AC_CONFIG_SUBDIRS([projects/Java])      ;;
      "llvm-tv")      AC_CONFIG_SUBDIRS([projects/llvm-tv])   ;;
      "llvm-fefw")    AC_CONFIG_SUBDIRS([projects/llvm-fefw]) ;;
      "poolalloc")    AC_CONFIG_SUBDIRS([projects/poolalloc]) ;;
      *)              
        AC_MSG_WARN([Unknown project (${i}) won't be configured automatically])
        ;;
    esac
  fi
done

dnl Configure header files
AC_CONFIG_HEADERS(include/llvm/Config/config.h)

dnl Configure other output file
AC_CONFIG_FILES(Makefile.config)
AC_CONFIG_HEADERS([include/llvm/Support/DataTypes.h])
AC_CONFIG_HEADERS([include/llvm/ADT/hash_map])
AC_CONFIG_HEADERS([include/llvm/ADT/hash_set])
AC_CONFIG_HEADERS([include/llvm/Support/ThreadSupport.h])
AC_CONFIG_HEADERS([include/llvm/ADT/iterator])

dnl Do special configuration of Makefiles
AC_CONFIG_MAKEFILE(Makefile)
AC_CONFIG_MAKEFILE(Makefile.common)
AC_CONFIG_MAKEFILE(examples/Makefile)
AC_CONFIG_MAKEFILE(lib/Makefile)
AC_CONFIG_MAKEFILE(runtime/Makefile)
AC_CONFIG_MAKEFILE(test/Makefile)
AC_CONFIG_MAKEFILE(test/Makefile.tests)
AC_CONFIG_MAKEFILE(test/QMTest/llvm.py)
AC_CONFIG_MAKEFILE(test/QMTest/llvmdb.py)
AC_CONFIG_MAKEFILE(tools/Makefile)
AC_CONFIG_MAKEFILE(tools/Makefile.JIT)
AC_CONFIG_MAKEFILE(utils/Makefile)
AC_CONFIG_MAKEFILE(projects/Makefile)

dnl Find the install program (needs to be done before canonical stuff)
AC_PROG_INSTALL

dnl Check which host for which we're compiling.  This will tell us which LLVM
dnl compiler will be used for compiling SSA into object code.
AC_CANONICAL_TARGET

dnl Set the "OS" Makefile variable based on the system we are building on.
dnl We will use the build machine information to set some variables.

AC_MSG_CHECKING([support for generic build operating system])
case $build in
	*-*-aix*)
             AC_SUBST(OS,[AIX])
             platform_type="AIX"
             ;;
	*-*-cygwin*)
             AC_SUBST(OS,[Cygwin])
             platform_type="Cygwin"
             ;;
	*-*-darwin*)
             AC_SUBST(OS,[Darwin])
             platform_type="Darwin"
             ;;
        *-*-freebsd*)
             AC_SUBST(OS,[Linux])
             platform_type="FreeBSD"
             ;;
        *-*-interix*)
             AC_SUBST(OS,[SunOS])
             platform_type="Interix"
             ;;
	*-*-linux*)
             AC_SUBST(OS,[Linux])
             platform_type="Linux"
             if test -d /home/vadve/lattner/local/x86/llvm-gcc
	     then
	       AC_SUBST(LLVMGCCDIR,[/home/vadve/lattner/local/x86/llvm-gcc/])
             fi
             ;;
	*-*-solaris*)
             AC_SUBST(OS,[SunOS])
             platform_type="SunOS"
             if test -d /home/vadve/lattner/local/sparc/llvm-gcc
             then
               AC_SUBST(LLVMGCCDIR,[/home/vadve/lattner/local/sparc/llvm-gcc/])
             fi
             ;;
        *-*-win32*)
             AC_SUBST(OS,[Win32])
             platform_type="Win32"
	     ;;
        *-*-mingw*)
             AC_SUBST(OS,[Win32])
             platform_type="Win32"
	     ;;
	*)   
             AC_SUBST(OS,[Unknown])
             platform_type="Unknown"
	     ;;
esac

dnl Make sure we aren't attempting to configure for an unknown system
if test "$platform_type" = "Unknown" ; then
  AC_MSG_ERROR([Platform is unknown, configure can't continue])
fi

dnl Make a link from lib/System/platform to lib/System/$platform_type
dnl This helps the #inclusion of the system specific include files
dnl for the operating system abstraction library
AC_CONFIG_LINKS(lib/System/platform:lib/System/$platform_type)


AC_MSG_CHECKING(target architecture)
dnl If we are targetting a Sparc machine running Solaris, pretend that it is
dnl V9, since that is all that we support at the moment, and autoconf will only
dnl tell us we're a sparc.
case $target in
	sparc*-*-solaris*)  AC_SUBST(target,[[sparcv9-sun-solaris2.8]])
	                    ;;
esac

dnl Determine what our target architecture is and configure accordingly.
dnl This will allow Makefiles to make a distinction between the hardware and
dnl the OS.
case $target in
	i*86-*)       
          ARCH="x86"
          AC_SUBST(ARCH,[x86])
	  ;;
	sparc*-*)         
          ARCH="Sparc"
          AC_SUBST(ARCH,[Sparc])
	  ;;
	powerpc*-*)       
          ARCH="PowerPC"
          AC_SUBST(ARCH,[PowerPC])
	  ;;
	*)                
          ARCH="Unknown"
          AC_SUBST(ARCH,[Unknown])
	  ;;
esac

AC_MSG_RESULT($ARCH)

dnl Check for compilation tools
AC_PROG_CXX
AC_PROG_CC(gcc)
dnl Ensure that compilation tools are GCC; we use GCC specific extensions
if test "$GCC" != "yes"
then
	AC_MSG_ERROR([gcc required but not found])
fi
AC_PROG_CPP
dnl Ensure that compilation tools are GCC; we use GCC specific extensions
if test "$GXX" != "yes"
then
	AC_MSG_ERROR([g++ required but not found])
fi

dnl Verify that GCC is version 3.0 or higher
gccmajor=`$CC --version | head -n 1 | awk '{print $NF;}' | cut -d. -f1`
if test "$gccmajor" -lt "3"
then
	AC_MSG_ERROR([gcc 3.x required, but you have a lower version])
fi

dnl Check for GNU Make.  We use its extensions too, so don't build without it
AC_CHECK_GNU_MAKE
if test -z "$_cv_gnu_make_command"
then
	AC_MSG_ERROR([GNU Make required but not found])
fi

dnl Checks for other tools
AC_PROG_FLEX
AC_PROG_BISON
AC_PROG_LIBTOOL
AC_PATH_PROG(TAR,[tar],[gtar])

dnl Checks for tools we can get away with not having:
AC_PATH_PROG(DOT,[dot],[true dot])
AC_PATH_PROG(ETAGS,[etags],[true etags])
dnl Check if we know how to tell etags we are using C++:
etags_version=`$ETAGS --version 2>&1`
case "$etags_version" in
	*[Ee]xuberant*) ETAGSFLAGS="--language-force=c++" ;;
	*GNU\ Emacs*) ETAGSFLAGS="-l c++" ;;
	*) ETAGSFLAGS="" ;;
esac
AC_SUBST(ETAGSFLAGS,$ETAGSFLAGS)
AC_PATH_PROG(PYTHON,[python],[true python])
if test "$PYTHON" = "false"
then
	AC_MSG_WARN([Python is required for the test suite, but it was not found])
fi
AC_PATH_PROG(QMTEST,[qmtest],[true qmtest])
if test "$QMTEST" = "false"
then
	AC_MSG_WARN([QMTest is required for the test suite, but it was not found])
fi

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
			AC_MSG_WARN([QMTest requires Python 2.2 or later])
		fi
	fi
else
	AC_MSG_WARN([QMTest requires Python 2.2 or later])
fi

dnl Checks for libraries:
dnl libelf is for sparc only; we can ignore it if we don't have it
AC_CHECK_LIB(elf, elf_begin)

dnl Check for bzip2 and zlib compression libraries needed for archive reading/writing
AC_CHECK_LIB(bz2,BZ2_bzCompressInit,[bzip2_found=1],[bzip2_found=0])
AC_CHECK_HEADERS([bzlib.h],[bzlib_h_found=1],[bzlib_h_found=0],[])
AC_CHECK_LIB(z,gzopen,[zlib_found=1],[zlib_found=0])
AC_CHECK_HEADERS([zlib.h],[zlib_h_found=1],[zlib_h_found=0],[])
if test $zlib_found -eq 1 -a $zlib_h_found -eq 1; then
  AC_DEFINE([HAVE_ZLIB],[1],[Define if zlib library is available on this platform.])
  AC_SUBST([HAVE_ZLIB],[1])
else
  AC_SUBST([HAVE_ZLIB],[0])
fi
if test $bzip2_found -eq 1 -a $bzlib_h_found -eq 1 ; then
  AC_DEFINE([HAVE_BZIP2],[1],[Define if bzip2 library is available on this platform.])
  AC_SUBST([HAVE_BZIP2],[1])
else
  AC_SUBST([HAVE_BZIP2],[0])
fi
dnl dlopen() is required for plugin support.
AC_SEARCH_LIBS(dlopen,dl,AC_DEFINE([HAVE_DLOPEN],[1],[Define if dlopen() is available on this platform.]),AC_MSG_WARN([dlopen() not found - disabling plugin support]))

dnl mallinfo is optional; the code can compile (minus features) without it
AC_SEARCH_LIBS(mallinfo,malloc,AC_DEFINE([HAVE_MALLINFO],[1],[Define if mallinfo() is available on this platform.]))

dnl pthread locking functions are optional - but llvm will not be thread-safe
dnl without locks.
AC_SEARCH_LIBS(pthread_mutex_lock,pthread,AC_DEFINE([HAVE_PTHREAD_MUTEX_LOCK],[1],[Have pthread_mutex_lock]))
dnl AC_SUBST(HAVE_PTHREAD_MUTEX_LOCK)

dnl Checks for header files.
dnl We don't check for ancient stuff or things that are guaranteed to be there
dnl by the C++ standard. We always use the <cfoo> versions of <foo.h> C headers.
AC_HEADER_STDC
AC_HEADER_SYS_WAIT

dnl Checks for POSIX and other various system-specific header files
AC_CHECK_HEADERS(fcntl.h limits.h sys/time.h unistd.h malloc.h sys/mman.h sys/resource.h dlfcn.h link.h execinfo.h windows.h)

dnl Check for things that need to be included in public headers, and so
dnl for which we may not have access to a HAVE_* preprocessor #define.
dnl (primarily used in DataTypes.h)
AC_CHECK_HEADER([sys/types.h])
AC_CHECK_HEADER([inttypes.h])
AC_CHECK_HEADER([stdint.h])

dnl Check for types
AC_TYPE_PID_T
AC_TYPE_SIZE_T
AC_CHECK_TYPES([int64_t],,AC_MSG_ERROR([Type int64_t required but not found]))
AC_CHECK_TYPES([uint64_t],,
               AC_CHECK_TYPES([u_int64_t],,
                              AC_MSG_ERROR([Type uint64_t or u_int64_t required but not found])))
AC_HEADER_TIME
AC_STRUCT_TM

dnl Check for various C features
AC_C_PRINTF_A

dnl Check for the endianness of the target
AC_C_BIGENDIAN(AC_SUBST([ENDIAN],[big]),AC_SUBST([ENDIAN],[little]))

dnl Check for C++ extensions
AC_CXX_HAVE_HASH_MAP
AC_CXX_HAVE_HASH_SET
AC_CXX_HAVE_STD_ITERATOR
AC_CXX_HAVE_BI_ITERATOR
AC_CXX_HAVE_FWD_ITERATOR

AC_FUNC_ISNAN
AC_FUNC_ISINF
AC_FUNC_RAND48

dnl Checks for library functions.
AC_FUNC_ALLOCA
AC_FUNC_MMAP
if test "$ac_cv_func_mmap_fixed_mapped" = "no"
then
	AC_MSG_WARN([mmap() required but not found])
fi
AC_FUNC_MMAP_FILE
if test "$ac_cv_func_mmap_file" = "no"
then
	AC_MSG_WARN([mmap() of files required but not found])
fi
AC_HEADER_MMAP_ANONYMOUS
AC_TYPE_SIGNAL
AC_CHECK_FUNCS(getcwd gettimeofday strdup strtoq strtoll backtrace isatty mkstemp getrusage)
AC_CHECK_FUNC(mprotect,,AC_MSG_ERROR([Function mprotect() required but not found]))

dnl Determine if the linker supports the -R option.
AC_LINK_USE_R

dnl --enable/--with command-line options:
dnl Check whether they want to do an optimized build:
AC_ARG_ENABLE(optimized,AS_HELP_STRING(--enable-optimized,Compile with optimizations enabled (default is NO)),,enableval=no)
if test ${enableval} = "no"
then
	AC_SUBST(ENABLE_OPTIMIZED,[[]])
else
	AC_SUBST(ENABLE_OPTIMIZED,[[ENABLE_OPTIMIZED=1]])
fi

dnl JIT Option
AC_ARG_ENABLE(jit,AS_HELP_STRING(--enable-jit,Enable Just In Time Compiling (default is YES)),,enableval=default)
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

dnl Find the LLVM GCC-based C/C++ front end
AC_ARG_WITH(llvmgccdir,AS_HELP_STRING(--with-llvmgccdir,Location of LLVM GCC front-end),AC_SUBST(LLVMGCCDIR,[$withval]))
AC_MSG_CHECKING([for llvm-gcc])
LLVM_GCC_CHECK=no
if test -d "$LLVMGCCDIR"
then
	if test -x "$LLVMGCCDIR/bin/gcc"
	then
		LLVM_GCC_CHECK="$LLVMGCCDIR/bin/gcc"
	fi
fi
llvmgccwarn=no
AC_MSG_RESULT($LLVM_GCC_CHECK)
if test "$LLVM_GCC_CHECK" = "no"
then
    llvmgccwarn=yes
fi
AC_MSG_CHECKING([whether llvm-gcc is sane])
LLVM_GCC_SANE=no
if test -x "$LLVM_GCC_CHECK"
then
	cp /dev/null conftest.c
    "$LLVM_GCC_CHECK" -S -o - conftest.c | grep implementation > /dev/null 2>&1
	if test $? -eq 0
	then
		LLVM_GCC_SANE=yes
	fi
	rm conftest.c
	llvmcc1path=`"$LLVM_GCC_CHECK" --print-prog-name=cc1`
	AC_SUBST(LLVMCC1,$llvmcc1path)
	llvmcc1pluspath=`"$LLVM_GCC_CHECK" --print-prog-name=cc1plus`
	AC_SUBST(LLVMCC1PLUS,$llvmcc1pluspath)
fi
AC_MSG_RESULT($LLVM_GCC_SANE)
if test "$LLVM_GCC_SANE" = "no"
then
	llvmgccwarn=yes
fi

dnl Get libtool's idea of what the shared library suffix is.
dnl (This is a hack; it relies on undocumented behavior.)
AC_MSG_CHECKING([for shared library suffix])
eval "SHLIBEXT=$shrext_cmds"
AC_MSG_RESULT($SHLIBEXT)
dnl Propagate it to the Makefiles and config.h (for gccld & bugpoint).
AC_SUBST(SHLIBEXT,$SHLIBEXT)
AC_DEFINE_UNQUOTED(SHLIBEXT,"$SHLIBEXT",
                   [Extension that shared libraries have, e.g., ".so".])

# Translate the various configuration directories and other basic
# information into substitutions that will end up in config.h.in so
# that these configured values can be hard-wired into a program.
eval LLVM_PREFIX="${prefix}";
eval LLVM_BINDIR="${prefix}/bin";
eval LLVM_LIBDIR="${prefix}/lib";
eval LLVM_DATADIR="${prefix}/data";
eval LLVM_DOCSDIR="${prefix}/docs";
eval LLVM_ETCDIR="${prefix}/etc";
eval LLVM_INCLUDEDIR="${prefix}/include";
eval LLVM_INFODIR="${prefix}/info";
eval LLVM_MANDIR="${prefix}/man";
LLVM_CONFIGTIME=`date`
AC_SUBST(LLVM_PREFIX)
AC_SUBST(LLVM_BINDIR)
AC_SUBST(LLVM_LIBDIR)
AC_SUBST(LLVM_DATADIR)
AC_SUBST(LLVM_DOCSDIR)
AC_SUBST(LLVM_ETCDIR)
AC_SUBST(LLVM_INCLUDEDIR)
AC_SUBST(LLVM_INFODIR)
AC_SUBST(LLVM_MANDIR)
AC_SUBST(LLVM_CONFIGTIME)
AC_DEFINE_UNQUOTED(LLVM_PREFIX,"$LLVM_PREFIX", [Installation prefix directory])
AC_DEFINE_UNQUOTED(LLVM_BINDIR, "$LLVM_BINDIR", [Installation directory for binary executables])
AC_DEFINE_UNQUOTED(LLVM_LIBDIR, "$LLVM_LIBDIR", [Installation directory for libraries])
AC_DEFINE_UNQUOTED(LLVM_DATADIR, "$LLVM_DATADIR", [Installation directory for data files])
AC_DEFINE_UNQUOTED(LLVM_DATADIR, "$LLVM_DOCSDIR", [Installation directory for documentation])
AC_DEFINE_UNQUOTED(LLVM_ETCDIR, "$LLVM_ETCDIR", [Installation directory for config files])
AC_DEFINE_UNQUOTED(LLVM_INCLUDEDIR, "$LLVM_INCLUDEDIR", [Installation directory for include files])
AC_DEFINE_UNQUOTED(LLVM_INFODIR, "$LLVM_INFODIR", [Installation directory for .info files])
AC_DEFINE_UNQUOTED(LLVM_MANDIR, "$LLVM_MANDIR", [Installation directory for man pages])
AC_DEFINE_UNQUOTED(LLVM_CONFIGTIME, "$LLVM_CONFIGTIME", [Time at which LLVM was configured])

dnl Create the output files
AC_OUTPUT

dnl Warn if we don't have a compression library
if test $bzip2_found -ne 1 ; then
  if test $zlib_found -ne 1 ; then
    AC_MSG_WARN([*** Neither zlib nor bzip2 compression libraries were found.])
    AC_MSG_WARN([*** Bytecode archives will not support compression!])
    AC_MSG_WARN([*** To correct, install the libraries and and re-run configure.])
  fi
fi

dnl Warn loudly if llvm-gcc was not obviously working
if test $llvmgccwarn = yes
then
	AC_MSG_WARN([***** llvm C/C++ front end was not found, or does not])
	AC_MSG_WARN([***** appear to be working.])
	AC_MSG_WARN([***** ])
	AC_MSG_WARN([***** Please check configure's --with-llvmgccdir option.])
	AC_MSG_WARN([***** Runtime libraries (in llvm/runtime) will not be built,])
	AC_MSG_WARN([***** but you should be able to build the llvm tools.])
fi

