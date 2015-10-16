// RUN: %clang_cc1 -emit-llvm %s -o - -triple=x86_64-pc-linux-gnu | FileCheck %s

// Verifying that we do, in fact, select the correct function in the following
// cases.

void foo(int m) __attribute__((overloadable, enable_if(m > 0, "")));
void foo(int m) __attribute__((overloadable));

// CHECK-LABEL: define void @test1
void test1() {
  // CHECK: store void (i32)* @_Z3fooi
  void (*p)(int) = foo;
  // CHECK: store void (i32)* @_Z3fooi
  void (*p2)(int) = &foo;
  // CHECK: store void (i32)* @_Z3fooi
  p = foo;
  // CHECK: store void (i32)* @_Z3fooi
  p = &foo;

  // CHECK: store i8* bitcast (void (i32)* @_Z3fooi to i8*)
  void *vp1 = (void*)&foo;
  // CHECK: store i8* bitcast (void (i32)* @_Z3fooi to i8*)
  void *vp2 = (void*)foo;
  // CHECK: store i8* bitcast (void (i32)* @_Z3fooi to i8*)
  vp1 = (void*)&foo;
  // CHECK: store i8* bitcast (void (i32)* @_Z3fooi to i8*)
  vp1 = (void*)foo;
}

void bar(int m) __attribute__((overloadable, enable_if(m > 0, "")));
void bar(int m) __attribute__((overloadable, enable_if(1, "")));
// CHECK-LABEL: define void @test2
void test2() {
  // CHECK: store void (i32)* @_Z3barUa9enable_ifIXLi1EEEi
  void (*p)(int) = bar;
  // CHECK: store void (i32)* @_Z3barUa9enable_ifIXLi1EEEi
  void (*p2)(int) = &bar;
  // CHECK: store void (i32)* @_Z3barUa9enable_ifIXLi1EEEi
  p = bar;
  // CHECK: store void (i32)* @_Z3barUa9enable_ifIXLi1EEEi
  p = &bar;

  // CHECK: store i8* bitcast (void (i32)* @_Z3barUa9enable_ifIXLi1EEEi to i8*)
  void *vp1 = (void*)&bar;
  // CHECK: store i8* bitcast (void (i32)* @_Z3barUa9enable_ifIXLi1EEEi to i8*)
  void *vp2 = (void*)bar;
  // CHECK: store i8* bitcast (void (i32)* @_Z3barUa9enable_ifIXLi1EEEi to i8*)
  vp1 = (void*)&bar;
  // CHECK: store i8* bitcast (void (i32)* @_Z3barUa9enable_ifIXLi1EEEi to i8*)
  vp1 = (void*)bar;
}

void baz(int m) __attribute__((overloadable, enable_if(1, "")));
void baz(int m) __attribute__((overloadable));
// CHECK-LABEL: define void @test3
void test3() {
  // CHECK: store void (i32)* @_Z3bazUa9enable_ifIXLi1EEEi
  void (*p)(int) = baz;
  // CHECK: store void (i32)* @_Z3bazUa9enable_ifIXLi1EEEi
  void (*p2)(int) = &baz;
  // CHECK: store void (i32)* @_Z3bazUa9enable_ifIXLi1EEEi
  p = baz;
  // CHECK: store void (i32)* @_Z3bazUa9enable_ifIXLi1EEEi
  p = &baz;
}


const int TRUEFACTS = 1;
void qux(int m) __attribute__((overloadable, enable_if(1, ""),
                               enable_if(TRUEFACTS, "")));
void qux(int m) __attribute__((overloadable, enable_if(1, "")));
// CHECK-LABEL: define void @test4
void test4() {
  // CHECK: store void (i32)* @_Z3quxUa9enable_ifIXLi1EEXL_Z9TRUEFACTSEEEi
  void (*p)(int) = qux;
  // CHECK: store void (i32)* @_Z3quxUa9enable_ifIXLi1EEXL_Z9TRUEFACTSEEEi
  void (*p2)(int) = &qux;
  // CHECK: store void (i32)* @_Z3quxUa9enable_ifIXLi1EEXL_Z9TRUEFACTSEEEi
  p = qux;
  // CHECK: store void (i32)* @_Z3quxUa9enable_ifIXLi1EEXL_Z9TRUEFACTSEEEi
  p = &qux;
}
