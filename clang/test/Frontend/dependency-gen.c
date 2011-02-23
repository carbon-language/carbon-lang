// rdar://6533411
// RUN: %clang -MD -MF %t.d -S -x c -o %t.o %s
// RUN: grep '.*dependency-gen.*:' %t.d
// RUN: grep 'dependency-gen.c' %t.d

// RUN: %clang -S -M -x c %s -o %t.d
// RUN: grep '.*dependency-gen.*:' %t.d
// RUN: grep 'dependency-gen.c' %t.d

// PR8974
// XFAIL: win32
// RUN: rm -rf %t.dir
// RUN: mkdir %t.dir
// RUN: echo > %t.dir/x.h
// RUN: %clang -include %t.dir/x.h -MD -MF %t.d -S -x c -o %t.o %s
// RUN: grep ' %t.dir/x.h' %t.d

