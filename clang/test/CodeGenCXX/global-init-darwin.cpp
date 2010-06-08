// RUN: %clang_cc1 -triple=x86_64-apple-darwin10 -emit-llvm %s -o - |FileCheck %s

struct A {
  A();
  ~A();
};

A a;
A as[2];

struct B {
  B();
  ~B();
  int f();
};

int i = B().f();

// CHECK: define internal void @__cxx_global_var_init() section "__TEXT,__StaticInit,regular,pure_instructions" {
// CHECK: define internal void @__cxx_global_var_init1() section "__TEXT,__StaticInit,regular,pure_instructions" {
// CHECK: define internal void @__cxx_global_array_dtor(i8*) section "__TEXT,__StaticInit,regular,pure_instructions" {
// CHECK: define internal void @__cxx_global_var_init2() section "__TEXT,__StaticInit,regular,pure_instructions" {
// CHECK: define internal void @_GLOBAL__I_a() section "__TEXT,__StaticInit,regular,pure_instructions" {
