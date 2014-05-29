// RUN: %clang_cc1 -fsanitize=address,init-order -emit-llvm -o - %s | FileCheck %s

// Test blacklist functionality.
// RUN: echo "global-init-src:%s" > %t-file.blacklist
// RUN: echo "global-init-type:struct.PODWithCtorAndDtor" > %t-type.blacklist
// RUN: %clang_cc1 -fsanitize=address,init-order -fsanitize-blacklist=%t-file.blacklist -emit-llvm -o - %s | FileCheck %s --check-prefix=BLACKLIST
// RUN: %clang_cc1 -fsanitize=address,init-order -fsanitize-blacklist=%t-type.blacklist -emit-llvm -o - %s | FileCheck %s --check-prefix=BLACKLIST
// REQUIRES: shell

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

// BLACKLIST-NOT: llvm.asan.dynamically_initialized_globals
