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

// CHECK: metadata !"_ZThn{{4|8}}_N1C1fEv", i32 15, {{.*}} ; [ DW_TAG_subprogram ] [line 15] [def]{{$}}
