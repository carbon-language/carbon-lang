#
# Check for GNU Make.  This is from
# http://www.gnu.org/software/ac-archive/htmldoc/check_gnu_make.html
#
AC_DEFUN([AC_CHECK_GNU_MAKE],
[ AC_CACHE_CHECK( for GNU make,_cv_gnu_make_command,
                _cv_gnu_make_command='' ;
dnl Search all the common names for GNU make
                for a in "$MAKE" make gmake gnumake ; do
                        if test -z "$a" ; then continue ; fi ;
                        if  ( sh -c "$a --version" 2> /dev/null | grep GNU  2>&1 > /dev/null ) ;  then
                                _cv_gnu_make_command=$a ;
                                break;
                        fi
                done ;
        ) ;
dnl If there was a GNU version, then set @ifGNUmake@ to the empty string, '#' otherwise
        if test  "x$_cv_gnu_make_command" != "x"  ; then
                ifGNUmake='' ;
        else
                ifGNUmake='#' ;
                AC_MSG_RESULT("Not found");
        fi
        AC_SUBST(ifGNUmake)
])
