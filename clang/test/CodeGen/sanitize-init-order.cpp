// RUN: %clang_cc1 -fsanitize=address,init-order -emit-llvm -o - %s | FileCheck %s

struct PODStruct {
  int x;
};
PODStruct s1;

struct PODWithDtor {
  ~PODWithDtor() { }
  int x;
};
PODWithDtor s2;

struct PODWithCtorAndDtor {
  PODWithCtorAndDtor() { }
  ~PODWithCtorAndDtor() { }
  int x;
};
PODWithCtorAndDtor s3;

// Check that ASan init-order checking ignores structs with trivial default
// constructor.
// CHECK: !llvm.asan.dynamically_initialized_globals = !{[[GLOB:![0-9]+]]}
// CHECK: [[GLOB]] = metadata !{%struct.PODWithCtorAndDtor
