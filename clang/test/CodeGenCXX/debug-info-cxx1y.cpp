// RUN: %clang_cc1 -emit-llvm-only -std=c++1y -g %s 2>&1 | FileCheck %s

struct foo {
  auto func(); // CHECK: error: debug information for auto is not yet supported
};

foo f;
