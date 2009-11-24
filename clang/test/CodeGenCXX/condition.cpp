// RUN: clang-cc -triple x86_64-apple-darwin10 -emit-llvm -o - %s | FileCheck %s
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
