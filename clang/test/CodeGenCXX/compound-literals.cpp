// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s

struct X {
  X();
  X(const X&);
  X(const char*);
  ~X();
};

struct Y { 
  int i;
  X x;
};

// CHECK: define i32 @_Z1fv()
int f() {
  // CHECK: [[LVALUE:%[a-z0-9.]+]] = alloca
  // CHECK-NEXT: [[I:%[a-z0-9]+]] = getelementptr inbounds {{.*}}* [[LVALUE]], i32 0, i32 0
  // CHECK-NEXT: store i32 17, i32* [[I]]
  // CHECK-NEXT: [[X:%[a-z0-9]+]] = getelementptr inbounds {{.*}} [[LVALUE]], i32 0, i32 1
  // CHECK-NEXT: call void @_ZN1XC1EPKc({{.*}}[[X]]
  // CHECK-NEXT: [[I:%[a-z0-9]+]] = getelementptr inbounds {{.*}} [[LVALUE]], i32 0, i32 0
  // CHECK-NEXT: [[RESULT:%[a-z0-9]+]] = load i32*
  // CHECK-NEXT: call void @_ZN1YD1Ev
  // CHECK-NEXT: ret i32 [[RESULT]]
  return ((Y){17, "seventeen"}).i;
}
