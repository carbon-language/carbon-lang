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
echo "Note: Warnings about 'AC_CONFIG_SUBDIRS: you should use literals' are ok"
echo "Regenerating aclocal.m4 with aclocal"
cwd=`pwd`
aclocal --force -I $cwd/m4 || die "aclocal failed"
echo "Regenerating configure with autoconf 2.5x"
autoconf --force --warnings=all -o ../configure configure.ac || die "autoconf failed"
cd ..
echo "Regenerating config.h.in with autoheader 2.5x"
autoheader -I autoconf -I autoconf/m4 autoconf/configure.ac || die "autoheader failed"
exit 0
