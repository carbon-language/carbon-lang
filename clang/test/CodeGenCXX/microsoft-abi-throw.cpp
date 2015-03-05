// RUN: %clang_cc1 -emit-llvm -o - -triple=i386-pc-win32 %s -fcxx-exceptions | FileCheck %s

// CHECK-DAG: @"\01??_R0?AUY@@@8" = linkonce_odr global %rtti.TypeDescriptor7 { i8** @"\01??_7type_info@@6B@", i8* null, [8 x i8] c".?AUY@@\00" }, comdat
// CHECK-DAG: @"_CT??_R0?AUY@@@88" = linkonce_odr unnamed_addr constant %eh.CatchableType { i32 4, i8* bitcast (%rtti.TypeDescriptor7* @"\01??_R0?AUY@@@8" to i8*), i32 0, i32 -1, i32 0, i32 8, i8* null }, comdat
// CHECK-DAG: @"\01??_R0?AUZ@@@8" = linkonce_odr global %rtti.TypeDescriptor7 { i8** @"\01??_7type_info@@6B@", i8* null, [8 x i8] c".?AUZ@@\00" }, comdat
// CHECK-DAG: @"_CT??_R0?AUZ@@@81" = linkonce_odr unnamed_addr constant %eh.CatchableType { i32 0, i8* bitcast (%rtti.TypeDescriptor7* @"\01??_R0?AUZ@@@8" to i8*), i32 0, i32 -1, i32 0, i32 1, i8* null }, comdat
// CHECK-DAG: @"\01??_R0?AUW@@@8" = linkonce_odr global %rtti.TypeDescriptor7 { i8** @"\01??_7type_info@@6B@", i8* null, [8 x i8] c".?AUW@@\00" }, comdat
// CHECK-DAG: @"_CT??_R0?AUW@@@84" = linkonce_odr unnamed_addr constant %eh.CatchableType { i32 4, i8* bitcast (%rtti.TypeDescriptor7* @"\01??_R0?AUW@@@8" to i8*), i32 4, i32 -1, i32 0, i32 4, i8* null }, comdat
// CHECK-DAG: @"\01??_R0?AUM@@@8" = linkonce_odr global %rtti.TypeDescriptor7 { i8** @"\01??_7type_info@@6B@", i8* null, [8 x i8] c".?AUM@@\00" }, comdat
// CHECK-DAG: @"_CT??_R0?AUM@@@81" = linkonce_odr unnamed_addr constant %eh.CatchableType { i32 0, i8* bitcast (%rtti.TypeDescriptor7* @"\01??_R0?AUM@@@8" to i8*), i32 8, i32 -1, i32 0, i32 1, i8* null }, comdat
// CHECK-DAG: @"\01??_R0?AUV@@@8" = linkonce_odr global %rtti.TypeDescriptor7 { i8** @"\01??_7type_info@@6B@", i8* null, [8 x i8] c".?AUV@@\00" }, comdat
// CHECK-DAG: @"_CT??_R0?AUV@@@81" = linkonce_odr unnamed_addr constant %eh.CatchableType { i32 0, i8* bitcast (%rtti.TypeDescriptor7* @"\01??_R0?AUV@@@8" to i8*), i32 0, i32 4, i32 4, i32 1, i8* null }, comdat
// CHECK-DAG: @"_CTA5?AUY@@" = linkonce_odr unnamed_addr constant %eh.CatchableTypeArray.5 { i32 5, [5 x %eh.CatchableType*] [%eh.CatchableType* @"_CT??_R0?AUY@@@88", %eh.CatchableType* @"_CT??_R0?AUZ@@@81", %eh.CatchableType* @"_CT??_R0?AUW@@@84", %eh.CatchableType* @"_CT??_R0?AUM@@@81", %eh.CatchableType* @"_CT??_R0?AUV@@@81"] }, comdat
// CHECK-DAG: @"_TI5?AUY@@" = linkonce_odr unnamed_addr constant %eh.ThrowInfo { i32 0, i8* bitcast (void (%struct.Y*)* @"\01??_DY@@QAE@XZ" to i8*), i8* null, i8* bitcast (%eh.CatchableTypeArray.5* @"_CTA5?AUY@@" to i8*) }, comdat


struct N { ~N(); };
struct M : private N {};
struct X {};
struct Z {};
struct V : private X {};
struct W : M, virtual V {};
struct Y : Z, W, virtual V {};

void f(const Y &y) {
  // CHECK-LABEL: @"\01?f@@YAXABUY@@@Z"
  // CHECK: call x86_thiscallcc %struct.Y* @"\01??0Y@@QAE@ABU0@@Z"(%struct.Y* %[[mem:.*]], %struct.Y*
  // CHECK: %[[cast:.*]] = bitcast %struct.Y* %[[mem]] to i8*
  // CHECK: call void @_CxxThrowException(i8* %[[cast]], %eh.ThrowInfo* @"_TI5?AUY@@") #4
  throw y;
}
