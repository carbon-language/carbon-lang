dnl This macro checks for tclsh which is required to run dejagnu. On some 
dnl platforms (notably FreeBSD), tclsh is named tclshX.Y - this handles
dnl that for us so we can get the latest installed tclsh version.
dnl 
AC_DEFUN([DJ_AC_PATH_TCLSH], [
dirlist=".. ../../ ../../../ ../../../../ ../../../../../ ../../../../../../ ../
../../../../../.. ../../../../../../../.. ../../../../../../../../.. ../../../..
/../../../../../.."
no_itcl=true
AC_MSG_CHECKING(for the tclsh program)
AC_ARG_WITH(tclinclude, [  --with-tclinclude       directory where tcl headers are], with_tclinclude=${withval})
AC_CACHE_VAL(ac_cv_path_tclsh,[
dnl first check to see if --with-itclinclude was specified
if test x"${with_tclinclude}" != x ; then
  if test -f ${with_tclinclude}/tclsh ; then
    ac_cv_path_tclsh=`(cd ${with_tclinclude}; pwd)`
  elif test -f ${with_tclinclude}/src/tclsh ; then
    ac_cv_path_tclsh=`(cd ${with_tclinclude}/src; pwd)`
  else
    AC_MSG_ERROR([${with_tclinclude} directory doesn't contain tclsh])
  fi
fi
])

dnl next check in private source directory
dnl since ls returns lowest version numbers first, reverse its output
if test x"${ac_cv_path_tclsh}" = x ; then
    dnl find the top level Itcl source directory
    for i in $dirlist; do
        if test -n "`ls -dr $srcdir/$i/tcl* 2>/dev/null`" ; then
            tclpath=$srcdir/$i
            break
        fi
    done

    dnl find the exact Itcl source dir. We do it this way, cause there
    dnl might be multiple version of Itcl, and we want the most recent one.
    for i in `ls -dr $tclpath/tcl* 2>/dev/null ` ; do
        if test -f $i/src/tclsh ; then
          ac_cv_path_tclsh=`(cd $i/src; pwd)`/tclsh
          break
        fi
    done
fi

dnl see if one is installed
if test x"${ac_cv_path_tclsh}" = x ; then
   AC_MSG_RESULT(none)
   AC_PATH_PROG(tclsh, tclsh)
else
   AC_MSG_RESULT(${ac_cv_path_tclsh})
fi
TCLSH="${ac_cv_path_tclsh}"
AC_SUBST(TCLSH)
])

