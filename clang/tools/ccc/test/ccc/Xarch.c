// RUN: xcc -### -fsyntax-only -Xarch_i386 -Wall -Xarch_ppc -Wunused -arch i386 -arch ppc %s &> %t &&
// RUN: grep '"-Xarch"' %t | count 0 &&
// RUN: grep '"-Wall"' %t | count 1 &&
// RUN: grep '"-arch" "i386"' %t | count 1 &&
// RUN: grep '"-Wall"' %t | grep '"-arch" "i386"' | count 1 &&
// RUN: grep '"-Wunused"' %t | count 1 &&
// RUN: grep '"-arch" "ppc"' %t | count 1 &&
// RUN: grep '"-Wunused"' %t | grep '"-arch" "ppc"' | count 1
