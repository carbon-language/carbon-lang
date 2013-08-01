// RUN: %clang -flimit-debug-info -emit-llvm -g -S %s -o - | FileCheck %s

// Ensure we emit the full definition of 'foo' even though only its declaration
// is needed, since C has no ODR to ensure that the definition will be the same
// in whatever TU actually uses/requires the definition of 'foo'.
// CHECK: ; [ DW_TAG_structure_type ] [foo] {{.*}} [def]

struct foo {
};

struct foo *f;
