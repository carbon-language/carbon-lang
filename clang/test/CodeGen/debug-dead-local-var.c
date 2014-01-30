// FIXME: Check IR rather than asm, then triple is not needed.
// RUN: %clang_cc1 -mllvm -asm-verbose -triple %itanium_abi_triple -S -O2 -g %s -o - | FileCheck %s
// Radar 8122864

// Code is not generated for function foo, but preserve type information of
// local variable xyz.
static void foo() {
// CHECK: DW_TAG_structure_type 
  struct X { int a; int b; } xyz;
}

int bar() {
  foo();
  return 1;
}
