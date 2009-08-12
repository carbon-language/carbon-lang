// RUN: clang-cc -triple=x86_64-apple-darwin9 -emit-llvm %s -o %t &&
// RUN: grep "call void @_ZN1AC1Ev" %t | count 1 &&
// RUN: grep "call i32 @__cxa_atexit(void (i8\*)\* bitcast (void (%.truct.A\*)\* @_ZN1AD1Ev to void (i8\*)\*), i8\* getelementptr inbounds (%.truct.A\* @a, i32 0, i32 0), i8\* bitcast (i8\*\* @__dso_handle to i8\*))" %t | count 1 

struct A {
  A();
  ~A();
};

A a;
