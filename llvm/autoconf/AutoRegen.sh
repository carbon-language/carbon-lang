#!/bin/sh
die () {
	echo "$@" 1>&2
	exit 1
}
test -d autoconf && test -f autoconf/configure.ac && cd autoconf
test -f configure.ac || die "Can't find 'autoconf' dir; please cd into it first"
autoconf --version | egrep '2\.59' > /dev/null
if test $? -ne 0 ; then
  die "Your autoconf was not detected as being 2.59"
fi
aclocal --version | egrep '1\.9\.1' > /dev/null
if test $? -ne 0 ; then
  die "Your aclocal was not detected as being 1.9.1"
fi
autoheader --version | egrep '2\.59' > /dev/null
if test $? -ne 0 ; then
  die "Your autoheader was not detected as being 2.59"
fi
libtool --version | grep '1.5.10' > /dev/null
if test $? -ne 0 ; then
  die "Your libtool was not detected as being 1.5.10"
fi
echo ""
echo "### NOTE: ############################################################"
echo "### If you get *any* warnings from autoconf below you MUST fix the"
echo "### scripts in the m4 directory because there are future forward"
echo "### compatibility or platform support issues at risk. Please do NOT"
echo "### commit any configure.ac or configure script that was generated "
echo "### with warnings present. You should get just three 'Regenerating..'"
echo "### lines."
echo "######################################################################"
echo ""
echo "Regenerating aclocal.m4 with aclocal"
cwd=`pwd`
aclocal --force -I $cwd/m4 || die "aclocal failed"
echo "Regenerating configure with autoconf 2.5x"
autoconf --force --warnings=all -o ../configure configure.ac || die "autoconf failed"
cd ..
echo "Regenerating config.h.in with autoheader 2.5x"
autoheader -I autoconf -I autoconf/m4 autoconf/configure.ac || die "autoheader failed"
exit 0
