// RUN: %clang_cc1 -Wno-microsoft -fms-extensions -fno-rtti -std=c++11 -emit-llvm %s -o - -triple=i386-pc-win32 | FileCheck %s

#pragma pointers_to_members(full_generality, virtual_inheritance)

struct S {
  int a, b;
  void f();
  virtual void g();
};

struct GeneralBase {
  virtual void h();
};
struct MostGeneral : S, virtual GeneralBase {
  virtual void h();
};
template <void (MostGeneral::*MP)()>
struct ClassTemplate {
  ClassTemplate() {}
};

template struct ClassTemplate<&MostGeneral::h>;

// Test that we mangle in the vbptr offset, which is 12 here.
//
// CHECK: define weak_odr x86_thiscallcc %struct.ClassTemplate* @"\01??0?$ClassTemplate@$J??_9MostGeneral@@$BA@AEA@M@3@@QAE@XZ"
