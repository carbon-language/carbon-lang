// RUN: echo "#include <stdio.h>" > %t.h
// RUN: %clang -S -save-temps -g -include %t.h %s -emit-llvm -o %t.ll
// RUN: grep "i32 5" %t.ll
// RUN: rm -f lineno-dbginfo.i
// outer is at line number 5.
int outer = 42;
