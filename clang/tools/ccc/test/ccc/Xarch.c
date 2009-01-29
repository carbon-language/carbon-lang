// RUN: xcc -ccc-no-clang -### -fsyntax-only -Xarch_i386 -Wall -Xarch_ppc -Wunused -arch i386 -arch ppc %s &> %t &&
// RUN: grep '"-Xarch"' %t | count 0 &&
// RUN: grep '"-Wall"' %t | count 1 &&
// RUN: grep 'i686-apple' %t | grep -v '"-m64"' | count 1 &&
// RUN: grep '"-Wall"' %t | grep 'i686-apple' | grep -v '"-m64"' | count 1 &&
// RUN: grep '"-Wunused"' %t | count 1 &&
// RUN: grep '"-arch" "ppc"' %t | count 1 &&
// RUN: grep '"-Wunused"' %t | grep '"-arch" "ppc"' | count 1
