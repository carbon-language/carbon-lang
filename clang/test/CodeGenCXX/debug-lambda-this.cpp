// RUN: %clang_cc1 -triple x86_64-apple-darwin10.0.0 -emit-llvm -o - %s -fexceptions -std=c++11 -g | FileCheck %s

struct D {
  D();
  D(const D&);
  int x;
  int d(int x);
};
int D::d(int x) {
  [=] {
    return this->x;
  }();
}

// CHECK: {{.*}} [ DW_TAG_member ] [this] [line 11, size 64, align 64, offset 0] [from ]
