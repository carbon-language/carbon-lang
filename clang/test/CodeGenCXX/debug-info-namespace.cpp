// RUN: %clang  -g -S %s -o - | FileCheck %s

// CHECK: TAG_namespace
namespace A {
  enum numbers {
    ZERO,
    ONE
  };
}

using namespace A;
numbers n;
