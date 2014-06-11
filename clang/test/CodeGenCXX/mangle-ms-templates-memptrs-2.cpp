// RUN: %clang_cc1 -Wno-microsoft -fms-extensions -fno-rtti -std=c++11 -emit-llvm %s -o - -triple=i386-pc-win32 | FileCheck %s

template <typename T, int (T::*)() = nullptr>
struct J {};

struct __single_inheritance M;
J<M> m;
// CHECK-DAG: @"\01?m@@3U?$J@UM@@$0A@@@A"

struct __multiple_inheritance N;
J<N> n;
// CHECK-DAG: @"\01?n@@3U?$J@UN@@$HA@@@A"

struct __virtual_inheritance O;
J<O> o;
// CHECK-DAG: @"\01?o@@3U?$J@UO@@$IA@A@@@A"

struct P;
J<P> p;
// CHECK-DAG: @"\01?p@@3U?$J@UP@@$JA@A@?0@@A"

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
