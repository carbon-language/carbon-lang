// RUN: %clang_cc1 -no-opaque-pointers -triple i686-pc-win32 -fms-extensions -emit-llvm -O1 -disable-llvm-passes %s -o - | FileCheck %s

class __declspec(dllimport) A {
  virtual void m_fn1();
};
template <typename>
class B : virtual A {};

extern template class __declspec(dllimport) B<int>;
class __declspec(dllexport) C : B<int> {};

// CHECK-DAG: @[[VTABLE_C:.*]] = private unnamed_addr constant { [2 x i8*] } { [2 x i8*] [i8* bitcast (%rtti.CompleteObjectLocator* @"??_R4C@@6B@" to i8*), i8* bitcast (void (%class.A*)* @"?m_fn1@A@@EAEXXZ" to i8*)] }
// CHECK-DAG: @[[VTABLE_B:.*]] = private unnamed_addr constant { [2 x i8*] } { [2 x i8*] [i8* bitcast (%rtti.CompleteObjectLocator* @"??_R4?$B@H@@6B@" to i8*), i8* bitcast (void (%class.A*)* @"?m_fn1@A@@EAEXXZ" to i8*)] }, comdat($"??_S?$B@H@@6B@")
// CHECK-DAG: @[[VTABLE_A:.*]] = private unnamed_addr constant { [2 x i8*] } { [2 x i8*] [i8* bitcast (%rtti.CompleteObjectLocator* @"??_R4A@@6B@" to i8*), i8* bitcast (void (%class.A*)* @"?m_fn1@A@@EAEXXZ" to i8*)] }, comdat($"??_SA@@6B@")

// CHECK-DAG: @"??_7C@@6B@" = dllexport unnamed_addr alias i8*, getelementptr inbounds ({ [2 x i8*] }, { [2 x i8*] }* @[[VTABLE_C]], i32 0, i32 0, i32 1)
// CHECK-DAG: @"??_S?$B@H@@6B@" = unnamed_addr alias i8*, getelementptr inbounds ({ [2 x i8*] }, { [2 x i8*] }* @[[VTABLE_B]], i32 0, i32 0, i32 1)
// CHECK-DAG: @"??_SA@@6B@" = unnamed_addr alias i8*, getelementptr inbounds ({ [2 x i8*] }, { [2 x i8*] }* @[[VTABLE_A]], i32 0, i32 0, i32 1)

// CHECK-DAG: @"??_8?$B@H@@7B@" = available_externally dllimport unnamed_addr constant [2 x i32] [i32 0, i32 4]
