// RUN: %clang_cc1 -O1 -emit-llvm %s -o - | FileCheck %s

// Check that the following construct, which is similar to one which occurs
// in Firefox, is not misfolded (folding it correctly would be a bonus, but
// that doesn't work at the moment, hence the -O1 in the runline).
struct A { char x; };
struct B { char y; };
struct C : A,B {};
unsigned char x = ((char*)(B*)(C*)0x1000) - (char*)0x1000;

// CHECK: @x = global i8 1
