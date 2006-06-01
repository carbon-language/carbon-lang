#!/bin/sh
die () {
	echo "$@" 1>&2
	exit 1
}
outfile=configure
configfile=configure.ac
test -d autoconf && test -f autoconf/$configfile && cd autoconf
test -f $configfile || die "Can't find 'autoconf' dir; please cd into it first"
autoconf --version | grep '2\.59' > /dev/null
test $? -eq 0 || die "Your autoconf was not detected as being 2.59"
aclocal --version | grep '^aclocal.*1\.9\.2' > /dev/null
test $? -eq 0 || die "Your aclocal was not detected as being 1.9.2"
autoheader --version | grep '^autoheader.*2\.59' > /dev/null
test $? -eq 0 || die "Your autoheader was not detected as being 2.59"
libtool --version | grep '1\.5\.10' > /dev/null
test $? -eq 0 || die "Your libtool was not detected as being 1.5.10"
echo ""
echo "### NOTE: ############################################################"
echo "### If you get *any* warnings from autoconf below you MUST fix the"
echo "### scripts in the m4 directory because there are future forward"
echo "### compatibility or platform support issues at risk. Please do NOT"
echo "### commit any configure script that was generated with warnings"
echo "### present. You should get just three 'Regenerating..' lines."
echo "######################################################################"
echo ""
echo "Regenerating aclocal.m4 with aclocal 1.9.2"
cwd=`pwd`
aclocal --force -I $cwd/m4 || die "aclocal failed"
echo "Regenerating configure with autoconf 2.59"
autoconf --force --warnings=all -o ../$outfile $configfile || die "autoconf failed"
cd ..
echo "Regenerating config.h.in with autoheader 2.59"
autoheader --warnings=all -I autoconf -I autoconf/m4 autoconf/$configfile || die "autoheader failed"
exit 0
