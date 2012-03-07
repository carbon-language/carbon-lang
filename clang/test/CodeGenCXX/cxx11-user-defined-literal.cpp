// RUN: %clang_cc1 -std=c++11 -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s

struct S { S(); ~S(); S(const S &); void operator()(int); };
using size_t = decltype(sizeof(int));
S operator"" _x(const char *, size_t);

void f() {
  // CHECK: call void @_Zli2_xPKcm({{.*}}, i8* getelementptr inbounds ([4 x i8]* @{{.*}}, i32 0, i32 0), i64 3)
  // CHECK: call void @_Zli2_xPKcm({{.*}}, i8* getelementptr inbounds ([4 x i8]* @{{.*}}, i32 0, i32 0), i64 3)
  // CHECK: call void @_ZN1SD1Ev({{.*}}) nounwind
  // CHECK: call void @_ZN1SD1Ev({{.*}}) nounwind
  "foo"_x, "bar"_x;
}

template<typename T> auto g(T t) -> decltype("foo"_x(t)) { return "foo"_x(t); }
template<typename T> auto i(T t) -> decltype(operator"" _x("foo", 3)(t)) { return operator"" _x("foo", 3)(t); }

void h() {
  g(42);
  i(42);
}

// CHECK: define {{.*}} @_Z1hv()
// CHECK:   call void @_Z1gIiEDTclclL_Zli2_xPKcmELA4_S0_ELm3EEfp_EET_(i32 42)
// CHECK:   call void @_Z1iIiEDTclclL_Zli2_xPKcmELA4_S0_ELi3EEfp_EET_(i32 42)

// CHECK: define {{.*}} @_Z1gIiEDTclclL_Zli2_xPKcmELA4_S0_ELm3EEfp_EET_(i32
// CHECK:   call void @_Zli2_xPKcm({{.*}}, i8* getelementptr inbounds ([4 x i8]* @{{.*}}, i32 0, i32 0), i64 3)
// CHECK:   call void @_ZN1SclEi
// CHECK:   call void @_ZN1SD1Ev

// CHECK: define {{.*}} @_Z1iIiEDTclclL_Zli2_xPKcmELA4_S0_ELi3EEfp_EET_(i32
// CHECK:   call void @_Zli2_xPKcm({{.*}}, i8* getelementptr inbounds ([4 x i8]* @{{.*}}, i32 0, i32 0), i64 3)
// CHECK:   call void @_ZN1SclEi
// CHECK:   call void @_ZN1SD1Ev
