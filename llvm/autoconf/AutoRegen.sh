#!/bin/sh
die () {
	echo "$@" 1>&2
	exit 1
}
test -d autoconf && test -f autoconf/configure.ac && cd autoconf
[ -f configure.ac ] || die "Can't find 'autoconf' dir; please cd into it first"
echo "Regenerating aclocal.m4 with aclocal"
aclocal || die "aclocal failed"
if ! autoconf --version | egrep '2\.5[0-9]' > /dev/null
then
	die "Your autoconf was not detected as being 2.5x"
fi
echo "Note: Warnings about 'AC_CONFIG_SUBDIRS: you should use literals' are ok"
echo "Regenerating configure with autoconf 2.5x"
autoconf -o ../configure configure.ac || die "autoconf failed"
cd ..
echo "Regenerating config.h.in with autoheader 2.5x"
autoheader -I autoconf autoconf/configure.ac || die "autoheader failed"
exit 0
