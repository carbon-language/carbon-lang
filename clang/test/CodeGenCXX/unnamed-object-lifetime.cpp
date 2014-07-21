// RUN: %clang_cc1 -triple i386-unknown-unknown -emit-llvm -O1 -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple i386-unknown-unknown -emit-llvm -O1 -fcxx-exceptions -fexceptions -o - %s | FileCheck --check-prefix=CHECK-EH %s

// Test lifetime marker generation for unnamed temporary objects.

struct X {
  X();
  ~X();
  char t[33]; // make the class big enough so that lifetime markers get inserted
};

extern void useX(const X &);

// CHECK-LABEL: define void @_Z6simplev
// CHECK-EH-LABEL: define void @_Z6simplev
void simple() {
  // CHECK: [[ALLOCA:%.*]] = alloca %struct.X
  // CHECK: [[PTR:%.*]] = getelementptr inbounds %struct.X* [[ALLOCA]], i32 0, i32 0, i32 0
  // CHECK: call void @llvm.lifetime.start(i64 33, i8* [[PTR]])
  // CHECK-NEXT: call void @_ZN1XC1Ev
  // CHECK-NEXT: call void @_Z4useXRK1X
  // CHECK-NEXT: call void @_ZN1XD1Ev
  // CHECK-NEXT: call void @llvm.lifetime.end(i64 33, i8* [[PTR]])
  //
  // CHECK-EH: [[ALLOCA:%.*]] = alloca %struct.X
  // CHECK-EH: [[PTR:%.*]] = getelementptr inbounds %struct.X* [[ALLOCA]], i32 0, i32 0, i32 0
  // CHECK-EH: call void @llvm.lifetime.start(i64 33, i8* [[PTR]])
  // CHECK-EH-NEXT: call void @_ZN1XC1Ev
  // CHECK-EH: invoke void @_Z4useXRK1X
  // CHECK-EH: invoke void @_ZN1XD1Ev
  // CHECK-EH: call void @llvm.lifetime.end(i64 33, i8* [[PTR]])
  // CHECK-EH: call void @llvm.lifetime.end(i64 33, i8* [[PTR]])
  useX(X());
}

struct Y {
  Y(){}
  ~Y(){}
  char t[34]; // make the class big enough so that lifetime markers get inserted
};

extern void useY(const Y &);

// Check lifetime markers are inserted, despite Y's trivial constructor & destructor
// CHECK-LABEL: define void @_Z7trivialv
// CHECK-EH-LABEL: define void @_Z7trivialv
void trivial() {
  // CHECK: [[ALLOCA:%.*]] = alloca %struct.Y
  // CHECK: [[PTR:%.*]] = getelementptr inbounds %struct.Y* [[ALLOCA]], i32 0, i32 0, i32 0
  // CHECK: call void @llvm.lifetime.start(i64 34, i8* [[PTR]])
  // CHECK-NEXT: call void @_Z4useYRK1Y
  // CHECK-NEXT: call void @llvm.lifetime.end(i64 34, i8* [[PTR]])
  //
  // CHECK-EH: [[ALLOCA:%.*]] = alloca %struct.Y
  // CHECK-EH: [[PTR:%.*]] = getelementptr inbounds %struct.Y* [[ALLOCA]], i32 0, i32 0, i32 0
  // CHECK-EH: call void @llvm.lifetime.start(i64 34, i8* [[PTR]])
  // CHECK-EH-NEXT: invoke void @_Z4useYRK1Y
  // CHECK-EH: call void @llvm.lifetime.end(i64 34, i8* [[PTR]])
  // CHECK-EH: call void @llvm.lifetime.end(i64 34, i8* [[PTR]])
  useY(Y());
}

struct Z {
  Z();
  ~Z();
  char t;
};

extern void useZ(const Z &);

// Check lifetime markers are not inserted if the unnamed object is too small
// CHECK-LABEL: define void @_Z8tooSmallv
// CHECK-EH-LABEL: define void @_Z8tooSmallv
void tooSmall() {
  // CHECK-NOT: call void @llvm.lifetime.start
  // CHECK: call void @_Z4useZRK1Z
  // CHECK-NOT: call void @llvm.lifetime.end
  // CHECK: ret
  //
  // CHECK-EH-NOT: call void @llvm.lifetime.start
  // CHECK-EH: invoke void @_Z4useZRK1Z
  // CHECK-EH-NOT: call void @llvm.lifetime.end
  // CHECK-EH: ret
  useZ(Z());
}

// Check the lifetime are inserted at the right place in their respective scope
// CHECK-LABEL: define void @_Z6scopesv
void scopes() {
  // CHECK: alloca %struct
  // CHECK: alloca %struct
  // CHECK: call void @llvm.lifetime.start(i64 33, i8* [[X:%.*]])
  // CHECK: call void @llvm.lifetime.end(i64 33, i8* [[X]])
  // CHECK: call void @llvm.lifetime.start(i64 34, i8* [[Y:%.*]])
  // CHECK: call void @llvm.lifetime.end(i64 34, i8* [[Y]])
  useX(X());
  useY(Y());
}

struct L {
  L(int);
  ~L();
  char t[33];
};

// Check the lifetime-extended case
// CHECK-LABEL: define void @_Z16extendedLifetimev
void extendedLifetime() {
  extern void useL(const L&);

  // CHECK: [[A:%.*]] = alloca %struct.L
  // CHECK: [[P:%.*]] = getelementptr inbounds %struct.L* [[A]], i32 0, i32 0, i32 0
  // CHECK: call void @llvm.lifetime.start(i64 33, i8* [[P]])
  // CHECK: call void @_ZN1LC1Ei(%struct.L* [[A]], i32 2)
  // CHECK-NOT: call void @llvm.lifetime.end(i64 33, i8* [[P]])
  // CHECK: call void @_Z4useLRK1L(%struct.L* dereferenceable(33) [[A]])
  // CHECK: call void @_ZN1LD1Ev(%struct.L* [[A]])
  // CHECK-NEXT: call void @llvm.lifetime.end(i64 33, i8* [[P]])
  //
  // CHECK-EH: [[A:%.*]] = alloca %struct.L
  // CHECK-EH: [[P:%.*]] = getelementptr inbounds %struct.L* [[A]], i32 0, i32 0, i32 0
  // CHECK-EH: call void @llvm.lifetime.start(i64 33, i8* [[P]])
  // CHECK-EH: call void @_ZN1LC1Ei(%struct.L* [[A]], i32 2)
  // CHECK-EH-NOT: call void @llvm.lifetime.end(i64 33, i8* [[P]])
  // CHECK-EH: invoke void @_Z4useLRK1L(%struct.L* dereferenceable(33) [[A]])
  // CHECK-EH: invoke void @_ZN1LD1Ev(%struct.L* [[A]])
  // CHECK-EH: call void @llvm.lifetime.end(i64 33, i8* [[P]])
  // CHECK-EH: invoke void @_ZN1LD1Ev(%struct.L* [[A]])
  // CHECK-EH: call void @llvm.lifetime.end(i64 33, i8* [[P]])
  const L &l = 2;
  useL(l);
}
