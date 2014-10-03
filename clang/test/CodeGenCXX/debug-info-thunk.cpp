// RUN: %clang_cc1 %s -triple %itanium_abi_triple -g -S -emit-llvm -o - | FileCheck %s

struct A {
  virtual void f();
};

struct B {
  virtual void f();
};

struct C : A, B {
  virtual void f();
};

void C::f() { }

// CHECK: metadata !{metadata !"0x2e\00\00\00_ZThn{{[48]}}_N1C1fEv\0015\00{{.*}}", {{.*}} ; [ DW_TAG_subprogram ] [line 15] [def]{{$}}
