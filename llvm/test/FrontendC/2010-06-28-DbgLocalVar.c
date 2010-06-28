// RUN: %llvmgcc -S -O2 -g %s -o - | llc -O2 -o %t.s 
// RUN: grep DW_TAG_structure_type %t.s | count 2
// Radar 8122864

// Code is not generated for function foo, but preserve type information of
// local variable xyz.
static foo() {
  struct X { int a; int b; } xyz;
}

int bar() {
  foo();
  return 1;
}
