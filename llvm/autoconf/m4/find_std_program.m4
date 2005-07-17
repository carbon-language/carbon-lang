dnl Check for a standard program that has a bin, include and lib directory
dnl 
dnl Parameters:
dnl   $1 - prefix directory to check
dnl   $2 - program name to check
dnl   $3 - header file to check 
dnl   $4 - library file to check 
AC_DEFUN([CHECK_STD_PROGRAM],
[m4_define([allcapsname],translit($2,a-z,A-Z))
if test -n "$1" -a -d "$1" -a -n "$2" -a -d "$1/bin" -a -x "$1/bin/$2" ; then
  AC_SUBST([USE_]allcapsname(),["USE_]allcapsname()[ = 1"])
  AC_SUBST(allcapsname(),[$1/bin/$2])
  AC_SUBST(allcapsname()[_BIN],[$1/bin])
  AC_SUBST(allcapsname()[_DIR],[$1])
  if test -n "$3" -a -d "$1/include" -a -f "$1/include/$3" ; then 
    AC_SUBST(allcapsname()[_INC],[$1/include])
  fi
  if test -n "$4" -a -d "$1/lib" -a -f "$1/lib/$4" ; then
    AC_SUBST(allcapsname()[_LIB],[$1/lib])
  fi
fi
])

dnl Find a program via --with options, in the path, or well known places
dnl
dnl Parameters:
dnl   $1 - program name
dnl   $2 - header file name to check (optional)
dnl   $3 - library file name to check (optional)
AC_DEFUN([FIND_STD_PROGRAM],
[m4_define([allcapsname],translit($1,a-z,A-Z))
AC_MSG_CHECKING([for ]$1[ bin/lib/include locations])
AC_ARG_WITH($1,
  AS_HELP_STRING([--with-]$1[=DIR],[Specify that ]$1['s install prefix is DIR]),
    $1[pfxdir=$withval],$1[pfxdir=nada])
AC_ARG_WITH($1[-bin],
  AS_HELP_STRING([--with-]$1[-bin=DIR],[Specify that ]$1[ binary are in DIR]),
    $1[bindir=$withval],$1[bindir=nada])
AC_ARG_WITH($1[-lib],
  AS_HELP_STRING([--with-]$1[-lib=DIR],[Specify that ]$1[ libs are in DIR]),
  $1[libdir=$withval],$1[libdir=nada])
AC_ARG_WITH($1[-inc],
  AS_HELP_STRING([--with-]$1[-inc=DIR],[Specify that ]$1[ includes are in DIR]),
  $1[incdir=$withval],$1[incdir=nada])
pfxvar=$1pfxdir
binvar=$1bindir
incvar=$1incdir
libvar=$1libdir
if test "${!pfxvar}" != "nada" ; then
  CHECK_STD_PROGRAM(${!pfxvar},$1,$2,$3)
elif test "${!binvar}" != "nada" ; then
  if test "${!libvar}" != "nada" ; then
    if test "${!incvar}" != "nada" ; then
      if test -d "${!binvar}" ; then
        if test -d "${!incvar}" ; then
          if test -d "${!libvar}" ; then
            AC_SUBST(allcapsname(),${!binvar}/$1)
            AC_SUBST(allcapsname()[_BIN],${!binvar})
            AC_SUBST(allcapsname()[_INC],${!incvar})
            AC_SUBST(allcapsname()[_LIB],${!libvar})
            AC_SUBST([USE_]allcapsname(),[1])
            AC_MSG_RESULT([found via --with options])
          else
            AC_MSG_RESULT([failed])
            AC_MSG_ERROR([The --with-]$1[-libdir value must be a directory])
          fi
        else
          AC_MSG_RESULT([failed])
          AC_MSG_ERROR([The --with-]$1[-incdir value must be a directory])
        fi
      else
        AC_MSG_RESULT([failed])
        AC_MSG_ERROR([The --with-]$1[-bindir value must be a directory])
      fi
    else
      AC_MSG_RESULT([failed])
      AC_MSG_ERROR([The --with-]$1[-incdir option must be specified])
    fi
  else
    AC_MSG_RESULT([failed])
    AC_MSG_ERROR([The --with-]$1[-libdir option must be specified])
  fi
else
  tmppfxdir=`which $1 2>&1`
  if test -n "$tmppfxdir" -a -d "${tmppfxdir%*$1}" -a \
          -d "${tmppfxdir%*$1}/.." ; then
    tmppfxdir=`cd "${tmppfxdir%*$1}/.." ; pwd`
    CHECK_STD_PROGRAM($tmppfxdir,$1,$2,$3)
    AC_MSG_RESULT([found in PATH at ]$tmppfxdir)
  else
    checkresult="yes"
    checkvar="USE_"allcapsname()
    CHECK_STD_PROGRAM([/usr],$1,$2,$3)
    if test -z "${!checkvar}" ; then
      CHECK_STD_PROGRAM([/usr/local],$1,$2,$3)
      if test -z "${!checkvar}" ; then
        CHECK_STD_PROGRAM([/sw],$1,$2,$3)
        if test -z "${!checkvar}" ; then
          CHECK_STD_PROGRAM([/opt],$1,$2,$3)
          if test -z "${!checkvar}" ; then
            CHECK_STD_PROGRAM([/],$1,$2,$3)
            if test -z "${!checkvar}" ; then
              checkresult="no"
            fi
          fi
        fi
      fi
    fi
    AC_MSG_RESULT($checkresult)
  fi
fi
])
