// rdar://6533411
// RUN: %clang -MD -MF %t.d -S -x c -o %t.o %s
// RUN: grep '.*dependency-gen.*:' %t.d
// RUN: grep 'dependency-gen.c' %t.d

// RUN: %clang -S -M -x c %s -o %t.d
// RUN: grep '.*dependency-gen.*:' %t.d
// RUN: grep 'dependency-gen.c' %t.d

// PR8974
// REQUIRES: shell
// "cd %t.dir" requires shell.
// RUN: rm -rf %t.dir
// RUN: mkdir -p %t.dir/a/b
// RUN: echo > %t.dir/a/b/x.h
// RUN: cd %t.dir
// RUN: %clang -include a/b/x.h -MD -MF %t.d -S -x c -o %t.o %s
// RUN: grep ' a/b/x\.h' %t.d

