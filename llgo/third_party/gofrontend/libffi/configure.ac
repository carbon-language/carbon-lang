dnl Process this with autoconf to create configure

AC_PREREQ(2.64)

AC_INIT([libffi], [3.0.11], [http://gcc.gnu.org/bugs.html])
AC_CONFIG_HEADERS([fficonfig.h])

AM_ENABLE_MULTILIB(, ..)

AC_CANONICAL_SYSTEM
target_alias=${target_alias-$host_alias}

. ${srcdir}/configure.host

AM_INIT_AUTOMAKE([no-dist])

# See if makeinfo has been installed and is modern enough
# that we can use it.
ACX_CHECK_PROG_VER([MAKEINFO], [makeinfo], [--version],
                   [GNU texinfo.* \([0-9][0-9.]*\)],
                   [4.[4-9]*|4.[1-9][0-9]*|[5-9]*|[1-9][0-9]*])
AM_CONDITIONAL(BUILD_INFO, test $gcc_cv_prog_makeinfo_modern = "yes")

# We would like our source tree to be readonly. However when releases or
# pre-releases are generated, the flex/bison generated files as well as the
# various formats of manuals need to be included along with the rest of the
# sources.  Therefore we have --enable-generated-files-in-srcdir to do
# just that.
AC_MSG_CHECKING(generated-files-in-srcdir)
AC_ARG_ENABLE(generated-files-in-srcdir,
AS_HELP_STRING([--enable-generated-files-in-srcdir],
 [put copies of generated files in source dir intended for creating source tarballs for users without texinfo bison or flex]),
[case "$enableval" in
 yes) enable_generated_files_in_srcdir=yes ;;
 no)  enable_generated_files_in_srcdir=no ;;
 *)   AC_MSG_ERROR([Unknown argument to enable/disable version-specific libs]);;
 esac],
[enable_generated_files_in_srcdir=no])
AC_MSG_RESULT($enable_generated_files_in_srcdir)
AM_CONDITIONAL(GENINSRC, test "$enable_generated_files_in_srcdir" = yes)

# The same as in boehm-gc and libstdc++. Have to borrow it from there.
# We must force CC to /not/ be precious variables; otherwise
# the wrong, non-multilib-adjusted value will be used in multilibs.
# As a side effect, we have to subst CFLAGS ourselves.

m4_rename([_AC_ARG_VAR_PRECIOUS],[real_PRECIOUS])
m4_define([_AC_ARG_VAR_PRECIOUS],[])
AC_PROG_CC
m4_rename_force([real_PRECIOUS],[_AC_ARG_VAR_PRECIOUS])

AC_SUBST(CFLAGS)

AM_PROG_AS
AM_PROG_CC_C_O
AC_PROG_LIBTOOL

AM_MAINTAINER_MODE

AC_CHECK_HEADERS(sys/mman.h)
AC_CHECK_FUNCS(mmap)
AC_FUNC_MMAP_BLACKLIST

dnl The -no-testsuite modules omit the test subdir.
AM_CONDITIONAL(TESTSUBDIR, test -d $srcdir/testsuite)

TARGETDIR="unknown"
HAVE_LONG_DOUBLE_VARIANT=0
case "$host" in
  aarch64*-*-*)
	TARGET=AARCH64; TARGETDIR=aarch64
	;;

  alpha*-*-*)
	TARGET=ALPHA; TARGETDIR=alpha;
	# Support 128-bit long double, changeable via command-line switch.
	HAVE_LONG_DOUBLE='defined(__LONG_DOUBLE_128__)'
	;;

  arm*-*-*)
	TARGET=ARM; TARGETDIR=arm
	;;

  amd64-*-freebsd* | amd64-*-openbsd*)
	TARGET=X86_64; TARGETDIR=x86
  	;;

  amd64-*-freebsd*)
	TARGET=X86_64; TARGETDIR=x86
	;;

  avr32*-*-*)
	TARGET=AVR32; TARGETDIR=avr32
	;;

  bfin*)
  	TARGET=BFIN; TARGETDIR=bfin
  	;;

  cris-*-*)
	TARGET=LIBFFI_CRIS; TARGETDIR=cris
	;;

  frv-*-*)
	TARGET=FRV; TARGETDIR=frv
	;;

  hppa*-*-linux* | parisc*-*-linux* | hppa*-*-openbsd*)
	TARGET=PA_LINUX; TARGETDIR=pa
	;;
  hppa*64-*-hpux*)
	TARGET=PA64_HPUX; TARGETDIR=pa
	;;
  hppa*-*-hpux*)
	TARGET=PA_HPUX; TARGETDIR=pa
	;;

  i?86-*-freebsd* | i?86-*-openbsd*)
	TARGET=X86_FREEBSD; TARGETDIR=x86
	;;
  i?86-win32* | i?86-*-cygwin* | i?86-*-mingw* | i?86-*-os2* | i?86-*-interix*)
	TARGET=X86_WIN32; TARGETDIR=x86
	# All mingw/cygwin/win32 builds require -no-undefined for sharedlib.
	# We must also check with_cross_host to decide if this is a native
	# or cross-build and select where to install dlls appropriately.
	if test -n "$with_cross_host" &&
	   test x"$with_cross_host" != x"no"; then
	  AM_LTLDFLAGS='-no-undefined -bindir "$(toolexeclibdir)"';
	else
	  AM_LTLDFLAGS='-no-undefined -bindir "$(bindir)"';
	fi
	;;
  i?86-*-darwin*)
	TARGET=X86_DARWIN; TARGETDIR=x86
	;;
  i?86-*-solaris2.1[[0-9]]*)
	TARGET=X86_64; TARGETDIR=x86
	;;
  i?86-*-*)
	TARGET=X86_64; TARGETDIR=x86
	;;

  ia64*-*-*)
	TARGET=IA64; TARGETDIR=ia64
	;;

  m32r*-*-*)
	TARGET=M32R; TARGETDIR=m32r
	;;

  m68k-*-*)
	TARGET=M68K; TARGETDIR=m68k
	;;

  mips-sgi-irix5.* | mips-sgi-irix6.* | mips*-*-rtems*)
	TARGET=MIPS; TARGETDIR=mips
	;;
  mips*-*-linux* | mips*-*-openbsd*)
	# Support 128-bit long double for NewABI.
	HAVE_LONG_DOUBLE='defined(__mips64)'
	TARGET=MIPS; TARGETDIR=mips
	;;

  powerpc*-*-linux* | powerpc-*-sysv*)
	TARGET=POWERPC; TARGETDIR=powerpc
	HAVE_LONG_DOUBLE_VARIANT=1
	;;
  powerpc-*-amigaos*)
	TARGET=POWERPC; TARGETDIR=powerpc
	;;
  powerpc-*-beos*)
	TARGET=POWERPC; TARGETDIR=powerpc
	;;
  powerpc-*-darwin* | powerpc64-*-darwin*)
	TARGET=POWERPC_DARWIN; TARGETDIR=powerpc
	;;
  powerpc-*-aix* | rs6000-*-aix*)
	TARGET=POWERPC_AIX; TARGETDIR=powerpc
	;;
  powerpc-*-freebsd* | powerpc-*-openbsd*)
	TARGET=POWERPC_FREEBSD; TARGETDIR=powerpc
	HAVE_LONG_DOUBLE_VARIANT=1
	;;
  powerpc64-*-freebsd*)
	TARGET=POWERPC; TARGETDIR=powerpc
	;;
  powerpc*-*-rtems*)
	TARGET=POWERPC; TARGETDIR=powerpc
	;;

  s390-*-* | s390x-*-*)
	TARGET=S390; TARGETDIR=s390
	;;

  sh-*-* | sh[[34]]*-*-*)
	TARGET=SH; TARGETDIR=sh
	;;
  sh64-*-* | sh5*-*-*)
	TARGET=SH64; TARGETDIR=sh64
	;;

  sparc*-*-*)
	TARGET=SPARC; TARGETDIR=sparc
	;;

  tile*-*)
        TARGET=TILE; TARGETDIR=tile
        ;;

  x86_64-*-darwin*)
	TARGET=X86_DARWIN; TARGETDIR=x86
	;;

  x86_64-*-cygwin* | x86_64-*-mingw*)
	TARGET=X86_WIN64; TARGETDIR=x86
	# All mingw/cygwin/win32 builds require -no-undefined for sharedlib.
	# We must also check with_cross_host to decide if this is a native
	# or cross-build and select where to install dlls appropriately.
	if test -n "$with_cross_host" &&
	   test x"$with_cross_host" != x"no"; then
	  AM_LTLDFLAGS='-no-undefined -bindir "$(toolexeclibdir)"';
	else
	  AM_LTLDFLAGS='-no-undefined -bindir "$(bindir)"';
	fi
	;;

  x86_64-*-*)
	TARGET=X86_64; TARGETDIR=x86
	;;
esac

AC_SUBST(AM_RUNTESTFLAGS)
AC_SUBST(AM_LTLDFLAGS)

if test $TARGETDIR = unknown; then
  AC_MSG_ERROR(["libffi has not been ported to $host."])
fi

AM_CONDITIONAL(MIPS, test x$TARGET = xMIPS)
AM_CONDITIONAL(BFIN, test x$TARGET = xBFIN)
AM_CONDITIONAL(SPARC, test x$TARGET = xSPARC)
AM_CONDITIONAL(X86, test x$TARGET = xX86)
AM_CONDITIONAL(X86_FREEBSD, test x$TARGET = xX86_FREEBSD)
AM_CONDITIONAL(X86_WIN32, test x$TARGET = xX86_WIN32)
AM_CONDITIONAL(X86_WIN64, test x$TARGET = xX86_WIN64)
AM_CONDITIONAL(X86_DARWIN, test x$TARGET = xX86_DARWIN)
AM_CONDITIONAL(ALPHA, test x$TARGET = xALPHA)
AM_CONDITIONAL(IA64, test x$TARGET = xIA64)
AM_CONDITIONAL(M32R, test x$TARGET = xM32R)
AM_CONDITIONAL(M68K, test x$TARGET = xM68K)
AM_CONDITIONAL(POWERPC, test x$TARGET = xPOWERPC)
AM_CONDITIONAL(POWERPC_AIX, test x$TARGET = xPOWERPC_AIX)
AM_CONDITIONAL(POWERPC_DARWIN, test x$TARGET = xPOWERPC_DARWIN)
AM_CONDITIONAL(POWERPC_FREEBSD, test x$TARGET = xPOWERPC_FREEBSD)
AM_CONDITIONAL(AARCH64, test x$TARGET = xAARCH64)
AM_CONDITIONAL(ARM, test x$TARGET = xARM)
AM_CONDITIONAL(AVR32, test x$TARGET = xAVR32)
AM_CONDITIONAL(LIBFFI_CRIS, test x$TARGET = xLIBFFI_CRIS)
AM_CONDITIONAL(FRV, test x$TARGET = xFRV)
AM_CONDITIONAL(S390, test x$TARGET = xS390)
AM_CONDITIONAL(X86_64, test x$TARGET = xX86_64)
AM_CONDITIONAL(SH, test x$TARGET = xSH)
AM_CONDITIONAL(SH64, test x$TARGET = xSH64)
AM_CONDITIONAL(PA_LINUX, test x$TARGET = xPA_LINUX)
AM_CONDITIONAL(PA_HPUX, test x$TARGET = xPA_HPUX)
AM_CONDITIONAL(PA64_HPUX, test x$TARGET = xPA64_HPUX)
AM_CONDITIONAL(TILE, test x$TARGET = xTILE)

AC_HEADER_STDC
AC_CHECK_FUNCS(memcpy)
AC_FUNC_ALLOCA

AC_CHECK_SIZEOF(double)
AC_CHECK_SIZEOF(long double)

# Also AC_SUBST this variable for ffi.h.
if test -z "$HAVE_LONG_DOUBLE"; then
  HAVE_LONG_DOUBLE=0
  if test $ac_cv_sizeof_long_double != 0; then
    if test $HAVE_LONG_DOUBLE_VARIANT != 0; then
      AC_DEFINE(HAVE_LONG_DOUBLE_VARIANT, 1, [Define if you support more than one size of the long double type])
      HAVE_LONG_DOUBLE=1
    else
      if test $ac_cv_sizeof_double != $ac_cv_sizeof_long_double; then
        HAVE_LONG_DOUBLE=1
        AC_DEFINE(HAVE_LONG_DOUBLE, 1, [Define if you have the long double type and it is bigger than a double])
      fi
    fi
  fi
fi
AC_SUBST(HAVE_LONG_DOUBLE)
AC_SUBST(HAVE_LONG_DOUBLE_VARIANT)

AC_C_BIGENDIAN

GCC_AS_CFI_PSEUDO_OP

AC_CACHE_CHECK([if compiler supports -Qunused-arguments],
[libffi_cv_c_unused_arguments],
[CFLAGS_hold=$CFLAGS
CFLAGS="$CFLAGS -Qunused-arguments"
AC_COMPILE_IFELSE([[int i;]],
[libffi_cv_c_unused_arguments=yes],
[libffi_cv_c_unused_arguments=no])
CFLAGS=$CFLAGS_hold])

if test x$TARGET = xSPARC; then
    AC_CACHE_CHECK([assembler and linker support unaligned pc related relocs],
	libffi_cv_as_sparc_ua_pcrel, [
	save_CFLAGS="$CFLAGS"
	save_LDFLAGS="$LDFLAGS"
	CFLAGS="$CFLAGS -fpic"
	LDFLAGS="$LDFLAGS -shared"
	AC_TRY_LINK([asm (".text; foo: nop; .data; .align 4; .byte 0; .uaword %r_disp32(foo); .text");],,
		    [libffi_cv_as_sparc_ua_pcrel=yes],
		    [libffi_cv_as_sparc_ua_pcrel=no])
	CFLAGS="$save_CFLAGS"
	LDFLAGS="$save_LDFLAGS"])
    if test "x$libffi_cv_as_sparc_ua_pcrel" = xyes; then
	AC_DEFINE(HAVE_AS_SPARC_UA_PCREL, 1,
		  [Define if your assembler and linker support unaligned PC relative relocs.])
    fi

    AC_CACHE_CHECK([assembler .register pseudo-op support],
       libffi_cv_as_register_pseudo_op, [
       libffi_cv_as_register_pseudo_op=unknown
       # Check if we have .register
       AC_TRY_COMPILE([asm (".register %g2, #scratch");],,
		       [libffi_cv_as_register_pseudo_op=yes],
		       [libffi_cv_as_register_pseudo_op=no])
    ])
    if test "x$libffi_cv_as_register_pseudo_op" = xyes; then
       AC_DEFINE(HAVE_AS_REGISTER_PSEUDO_OP, 1,
	       [Define if your assembler supports .register.])
    fi
fi

if test x$TARGET = xX86 || test x$TARGET = xX86_WIN32 || test x$TARGET = xX86_64; then
    AC_CACHE_CHECK([assembler supports pc related relocs],
	libffi_cv_as_x86_pcrel, [
	libffi_cv_as_x86_pcrel=yes
	echo '.text; foo: nop; .data; .long foo-.; .text' > conftest.s
	CFLAGS_hold=$CFLAGS
	if test "$libffi_cv_c_unused_arguments" = yes; then
	    CFLAGS="$CFLAGS -Qunused-arguments"
	fi
	if $CC $CFLAGS -c conftest.s 2>&1 | $EGREP -i 'illegal|warning' > /dev/null; then
	    libffi_cv_as_x86_pcrel=no
	fi
	CFLAGS=$CFLAGS_hold
	])
    if test "x$libffi_cv_as_x86_pcrel" = xyes; then
	AC_DEFINE(HAVE_AS_X86_PCREL, 1,
		  [Define if your assembler supports PC relative relocs.])
    fi

    AC_CACHE_CHECK([assembler .ascii pseudo-op support],
       libffi_cv_as_ascii_pseudo_op, [
       libffi_cv_as_ascii_pseudo_op=unknown
       # Check if we have .ascii
       AC_TRY_COMPILE([asm (".ascii \\"string\\"");],,
		       [libffi_cv_as_ascii_pseudo_op=yes],
		       [libffi_cv_as_ascii_pseudo_op=no])
    ])
    if test "x$libffi_cv_as_ascii_pseudo_op" = xyes; then
       AC_DEFINE(HAVE_AS_ASCII_PSEUDO_OP, 1,
	       [Define if your assembler supports .ascii.])
    fi

    AC_CACHE_CHECK([assembler .string pseudo-op support],
       libffi_cv_as_string_pseudo_op, [
       libffi_cv_as_string_pseudo_op=unknown
       # Check if we have .string
       AC_TRY_COMPILE([asm (".string \\"string\\"");],,
		       [libffi_cv_as_string_pseudo_op=yes],
		       [libffi_cv_as_string_pseudo_op=no])
    ])
    if test "x$libffi_cv_as_string_pseudo_op" = xyes; then
       AC_DEFINE(HAVE_AS_STRING_PSEUDO_OP, 1,
	       [Define if your assembler supports .string.])
    fi
fi

# On PaX enable kernels that have MPROTECT enable we can't use PROT_EXEC.
AC_ARG_ENABLE(pax_emutramp,
  [  --enable-pax_emutramp       enable pax emulated trampolines, for we can't use PROT_EXEC],
  if test "$enable_pax_emutramp" = "yes"; then
    AC_DEFINE(FFI_MMAP_EXEC_EMUTRAMP_PAX, 1,
      [Define this if you want to enable pax emulated trampolines])
  fi)

FFI_EXEC_TRAMPOLINE_TABLE=0
case "$target" in
     *arm*-apple-darwin*)
       FFI_EXEC_TRAMPOLINE_TABLE=1
       AC_DEFINE(FFI_EXEC_TRAMPOLINE_TABLE, 1,
                 [Cannot use PROT_EXEC on this target, so, we revert to
                   alternative means])
     ;;
     *-apple-darwin1* | *-*-freebsd* | *-*-kfreebsd* | *-*-openbsd* | *-pc-solaris*)
       AC_DEFINE(FFI_MMAP_EXEC_WRIT, 1,
                 [Cannot use malloc on this target, so, we revert to
                   alternative means])
     ;;
esac
AM_CONDITIONAL(FFI_EXEC_TRAMPOLINE_TABLE, test x$FFI_EXEC_TRAMPOLINE_TABLE = x1)
AC_SUBST(FFI_EXEC_TRAMPOLINE_TABLE)

if test x$TARGET = xX86_64; then
    AC_CACHE_CHECK([assembler supports unwind section type],
	libffi_cv_as_x86_64_unwind_section_type, [
	libffi_cv_as_x86_64_unwind_section_type=yes
	echo '.section .eh_frame,"a",@unwind' > conftest.s
	CFLAGS_hold=$CFLAGS
	if test "$libffi_cv_c_unused_arguments" = yes; then
	    CFLAGS="$CFLAGS -Qunused-arguments"
	fi
	if $CC $CFLAGS -c conftest.s 2>&1 | grep -i warning > /dev/null; then
	    libffi_cv_as_x86_64_unwind_section_type=no
	fi
	CFLAGS=$CFLAGS_hold
	])
    if test "x$libffi_cv_as_x86_64_unwind_section_type" = xyes; then
	AC_DEFINE(HAVE_AS_X86_64_UNWIND_SECTION_TYPE, 1,
		  [Define if your assembler supports unwind section type.])
    fi
fi

if test "x$GCC" = "xyes"; then
  AC_CACHE_CHECK([whether .eh_frame section should be read-only],
      libffi_cv_ro_eh_frame, [
  	libffi_cv_ro_eh_frame=no
  	echo 'extern void foo (void); void bar (void) { foo (); foo (); }' > conftest.c
  	if $CC $CFLAGS -S -fpic -fexceptions -o conftest.s conftest.c > /dev/null 2>&1; then
  	    if grep '.section.*eh_frame.*"a"' conftest.s > /dev/null; then
  		libffi_cv_ro_eh_frame=yes
  	    elif grep '.section.*eh_frame.*#alloc' conftest.c \
  		 | grep -v '#write' > /dev/null; then
  		libffi_cv_ro_eh_frame=yes
  	    fi
  	fi
  	rm -f conftest.*
      ])
  if test "x$libffi_cv_ro_eh_frame" = xyes; then
      AC_DEFINE(HAVE_RO_EH_FRAME, 1,
  	      [Define if .eh_frame sections should be read-only.])
      AC_DEFINE(EH_FRAME_FLAGS, "a",
  	      [Define to the flags needed for the .section .eh_frame directive.  ])
  else
      AC_DEFINE(EH_FRAME_FLAGS, "aw",
  	      [Define to the flags needed for the .section .eh_frame directive.  ])
  fi

  AC_CACHE_CHECK([for __attribute__((visibility("hidden")))],
      libffi_cv_hidden_visibility_attribute, [
  	echo 'int __attribute__ ((visibility ("hidden"))) foo (void) { return 1  ; }' > conftest.c
  	libffi_cv_hidden_visibility_attribute=no
  	if AC_TRY_COMMAND(${CC-cc} -Werror -S conftest.c -o conftest.s 1>&AS_MESSAGE_LOG_FD); then
  	    if grep '\.hidden.*foo' conftest.s >/dev/null; then
  		libffi_cv_hidden_visibility_attribute=yes
  	    fi
  	fi
  	rm -f conftest.*
      ])
  if test $libffi_cv_hidden_visibility_attribute = yes; then
      AC_DEFINE(HAVE_HIDDEN_VISIBILITY_ATTRIBUTE, 1,
  	      [Define if __attribute__((visibility("hidden"))) is supported.])
  fi
fi

AH_BOTTOM([
#ifdef HAVE_HIDDEN_VISIBILITY_ATTRIBUTE
#ifdef LIBFFI_ASM
#define FFI_HIDDEN(name) .hidden name
#else
#define FFI_HIDDEN __attribute__ ((visibility ("hidden")))
#endif
#else
#ifdef LIBFFI_ASM
#define FFI_HIDDEN(name)
#else
#define FFI_HIDDEN
#endif
#endif
])

AC_SUBST(TARGET)
AC_SUBST(TARGETDIR)

AC_SUBST(SHELL)

AC_ARG_ENABLE(debug,
[  --enable-debug          debugging mode],
  if test "$enable_debug" = "yes"; then
    AC_DEFINE(FFI_DEBUG, 1, [Define this if you want extra debugging.])
  fi)
AM_CONDITIONAL(FFI_DEBUG, test "$enable_debug" = "yes")

AC_ARG_ENABLE(structs,
[  --disable-structs       omit code for struct support],
  if test "$enable_structs" = "no"; then
    AC_DEFINE(FFI_NO_STRUCTS, 1, [Define this is you do not want support for aggregate types.])
  fi)

AC_ARG_ENABLE(raw-api,
[  --disable-raw-api       make the raw api unavailable],
  if test "$enable_raw_api" = "no"; then
    AC_DEFINE(FFI_NO_RAW_API, 1, [Define this is you do not want support for the raw API.])
  fi)

AC_ARG_ENABLE(purify-safety,
[  --enable-purify-safety  purify-safe mode],
  if test "$enable_purify_safety" = "yes"; then
    AC_DEFINE(USING_PURIFY, 1, [Define this if you are using Purify and want to suppress spurious messages.])
  fi)

if test -n "$with_cross_host" &&
   test x"$with_cross_host" != x"no"; then
  toolexecdir='$(exec_prefix)/$(target_alias)'
  toolexeclibdir='$(toolexecdir)/lib'
else
  toolexecdir='$(libdir)/gcc-lib/$(target_alias)'
  toolexeclibdir='$(libdir)'
fi
multi_os_directory=`$CC -print-multi-os-directory`
case $multi_os_directory in
  .) ;; # Avoid trailing /.
  *) toolexeclibdir=$toolexeclibdir/$multi_os_directory ;;
esac
AC_SUBST(toolexecdir)
AC_SUBST(toolexeclibdir)

if test "${multilib}" = "yes"; then
  multilib_arg="--enable-multilib"
else
  multilib_arg=
fi

AC_CONFIG_COMMANDS(include, [test -d include || mkdir include])
AC_CONFIG_COMMANDS(src, [
test -d src || mkdir src
test -d src/$TARGETDIR || mkdir src/$TARGETDIR
], [TARGETDIR="$TARGETDIR"])

AC_CONFIG_LINKS(include/ffitarget.h:src/$TARGETDIR/ffitarget.h)

AC_CONFIG_FILES(include/Makefile include/ffi.h Makefile testsuite/Makefile man/Makefile)

AC_OUTPUT
