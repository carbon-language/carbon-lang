// RUN: %clang_cc1 -emit-llvm -std=c++11 -g %s -o -| FileCheck %s

void foo() {
  decltype(nullptr) t = 0;
}

// CHECK: !MDBasicType(tag: DW_TAG_unspecified_type, name: "decltype(nullptr)")
