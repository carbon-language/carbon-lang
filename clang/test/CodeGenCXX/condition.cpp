// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -o - %s | FileCheck %s
void *f();

template <typename T> T* g() {
 if (T* t = f())
   return t;

 return 0;
}

void h() {
 void *a = g<void>();
}

struct X {
  X();
  ~X();
  operator bool();
};

struct Y {
  Y();
  ~Y();
};

void if_destruct(int z) {
  // Verify that the condition variable is destroyed at the end of the
  // "if" statement.
  // CHECK: call void @_ZN1XC1Ev
  // CHECK: call zeroext i1 @_ZN1XcvbEv
  if (X x = X()) {
    // CHECK: store i32 18
    z = 18;
  }
  // CHECK: call void @_ZN1XD1Ev
  // CHECK: store i32 17
  z = 17;

  // CHECK: call void @_ZN1XC1Ev
  if (X x = X())
    Y y;
  // CHECK: if.then
  // CHECK: call  void @_ZN1YC1Ev
  // CHECK: call  void @_ZN1YD1Ev
  // CHECK: if.end
  // CHECK: call  void @_ZN1XD1Ev
}

struct ConvertibleToInt {
  ConvertibleToInt();
  ~ConvertibleToInt();
  operator int();
};

void switch_destruct(int z) {
  // CHECK: call void @_ZN16ConvertibleToIntC1Ev
  switch (ConvertibleToInt conv = ConvertibleToInt()) {
  case 0:
    break;

  default:
    // CHECK: sw.default:
    // CHECK: store i32 19
    z = 19;
    break;
  }
  // CHECK: sw.epilog:
  // CHECK: call void @_ZN16ConvertibleToIntD1Ev
  // CHECK: store i32 20
  z = 20;
}

int foo();

void while_destruct(int z) {
  // CHECK: define void @_Z14while_destructi
  // CHECK: while.cond: 
  while (X x = X()) {
    // CHECK: call void @_ZN1XC1Ev

    // CHECK: while.body:
    // CHECK: store i32 21
    z = 21;

    // CHECK: while.cleanup:
    // CHECK: call void @_ZN1XD1Ev
  }
  // CHECK: while.end
  // CHECK: store i32 22
  z = 22;
}

void for_destruct(int z) {
  // CHECK: define void @_Z12for_destruct
  // CHECK: call void @_ZN1YC1Ev
  for(Y y = Y(); X x = X(); ++z)
    // CHECK: for.cond:
    // CHECK: call void @_ZN1XC1Ev
    // CHECK: for.body:
    // CHECK: store i32 23
    z = 23;
    // CHECK: for.inc:
    // CHECK: br label %for.cond.cleanup
    // CHECK: for.cond.cleanup:
    // CHECK: call void @_ZN1XD1Ev
  // CHECK: for.end:
  // CHECK: call void @_ZN1YD1Ev
  // CHECK: store i32 24
  z = 24;
}
