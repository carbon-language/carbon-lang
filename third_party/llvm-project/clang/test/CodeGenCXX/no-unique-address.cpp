// RUN: %clang_cc1 -std=c++2a %s -emit-llvm -o - -triple x86_64-linux-gnu | FileCheck %s
// RUN: %clang_cc1 -std=c++2a %s -emit-llvm -o - -triple x86_64-linux-gnu -O2 -disable-llvm-passes | FileCheck %s --check-prefix=CHECK-OPT

struct A { ~A(); int n; char c[3]; };
struct B { [[no_unique_address]] A a; char k; };
// CHECK-DAG: @b ={{.*}} global { i32, [3 x i8], i8 } { i32 1, [3 x i8] c"\02\03\04", i8 5 }
B b = {1, 2, 3, 4, 5};

struct C : A {};
struct D : C {};
struct E { int e; [[no_unique_address]] D d; char k; };
// CHECK-DAG: @e ={{.*}} global { i32, i32, [3 x i8], i8 } { i32 1, i32 2, [3 x i8] c"\03\04\05", i8 6 }
E e = {1, 2, 3, 4, 5, 6};

struct Empty1 {};
struct Empty2 {};
struct Empty3 {};
struct HasEmpty {
  [[no_unique_address]] Empty1 e1;
  int a;
  [[no_unique_address]] Empty2 e2;
  int b;
  [[no_unique_address]] Empty3 e3;
};
// CHECK-DAG: @he ={{.*}} global %{{[^ ]*}} { i32 1, i32 2 }
HasEmpty he = {{}, 1, {}, 2, {}};

struct HasEmptyDuplicates {
  [[no_unique_address]] Empty1 e1; // +0
  int a;
  [[no_unique_address]] Empty1 e2; // +4
  int b;
  [[no_unique_address]] Empty1 e3; // +8
};
// CHECK-DAG: @off1 ={{.*}} global i64 0
Empty1 HasEmptyDuplicates::*off1 = &HasEmptyDuplicates::e1;
// CHECK-DAG: @off2 ={{.*}} global i64 4
Empty1 HasEmptyDuplicates::*off2 = &HasEmptyDuplicates::e2;
// CHECK-DAG: @off3 ={{.*}} global i64 8
Empty1 HasEmptyDuplicates::*off3 = &HasEmptyDuplicates::e3;

// CHECK-DAG: @hed ={{.*}} global %{{[^ ]*}} { i32 1, i32 2, [4 x i8] undef }
HasEmptyDuplicates hed = {{}, 1, {}, 2, {}};

struct __attribute__((packed, aligned(2))) PackedAndPadded {
  ~PackedAndPadded();
  char c;
  int n;
};
struct WithPackedAndPadded {
  [[no_unique_address]] PackedAndPadded pap;
  char d;
};
// CHECK-DAG: @wpap ={{.*}} global <{ i8, i32, i8 }> <{ i8 1, i32 2, i8 3 }>
WithPackedAndPadded wpap = {1, 2, 3};

struct FieldOverlap {
  [[no_unique_address]] Empty1 e1, e2, e3, e4;
  int n;
};
static_assert(sizeof(FieldOverlap) == 4);
// CHECK-DAG: @fo ={{.*}} global %{{[^ ]*}} { i32 1234 }
FieldOverlap fo = {{}, {}, {}, {}, 1234};

// CHECK-DAG: @e1 ={{.*}} constant %[[E1:[^ ]*]]* bitcast (%[[FO:[^ ]*]]* @fo to %[[E1]]*)
Empty1 &e1 = fo.e1;
// CHECK-DAG: @e2 ={{.*}} constant %[[E1]]* bitcast (i8* getelementptr (i8, i8* bitcast (%[[FO]]* @fo to i8*), i64 1) to %[[E1]]*)
Empty1 &e2 = fo.e2;

// CHECK-LABEL: accessE1
// CHECK: %[[RET:.*]] = bitcast %[[FO]]* %{{.*}} to %[[E1]]*
// CHECK: ret %[[E1]]* %[[RET]]
Empty1 &accessE1(FieldOverlap &fo) { return fo.e1; }

// CHECK-LABEL: accessE2
// CHECK: %[[AS_I8:.*]] = bitcast %[[FO]]* %{{.*}} to i8*
// CHECK: %[[ADJUSTED:.*]] = getelementptr inbounds i8, i8* %[[AS_I8]], i64 1
// CHECK: %[[RET:.*]] = bitcast i8* %[[ADJUSTED]] to %[[E1]]*
// CHECK: ret %[[E1]]* %[[RET]]
Empty1 &accessE2(FieldOverlap &fo) { return fo.e2; }

struct LaterDeclaredFieldHasLowerOffset {
  int a;
  int b;
  [[no_unique_address]] Empty1 e;
};
// CHECK-OPT-LABEL: @_Z41loadWhereLaterDeclaredFieldHasLowerOffset
int loadWhereLaterDeclaredFieldHasLowerOffset(LaterDeclaredFieldHasLowerOffset &a) {
  // CHECK-OPT: getelementptr
  // CHECK-OPT: load {{.*}}, !tbaa ![[TBAA_AB:[0-9]*]]
  return a.b;
}
// Note, never emit TBAA for zero-size fields.
// CHECK-OPT: ![[TBAA_AB]] = !{![[TBAA_A:[0-9]*]], ![[TBAA_INT:[0-9]*]], i64 4}
// CHECK-OPT: ![[TBAA_A]] = !{!"_ZTS32LaterDeclaredFieldHasLowerOffset", ![[TBAA_INT]], i64 0, ![[TBAA_INT]], i64 4}
