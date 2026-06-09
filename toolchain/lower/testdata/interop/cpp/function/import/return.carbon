// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
// EXTRA-ARGS: --target=x86_64-linux-gnu
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/function/import/return.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/function/import/return.carbon

// --- ints.h

unsigned char ReturnU8();
unsigned short ReturnU16();
unsigned int ReturnU32();
unsigned long ReturnU64();

signed char ReturnI8();
signed short ReturnI16();
signed int ReturnI32();
signed long ReturnI64();

// --- import_ints.carbon

library "[[@TEST_NAME]]";

import Cpp library "ints.h";

fn GetU8() -> u8 { return Cpp.ReturnU8(); }
fn GetU16() -> u16 { return Cpp.ReturnU16(); }
fn GetU32() -> u32 { return Cpp.ReturnU32(); }
fn GetU64() -> u64 { return Cpp.ReturnU64(); }

fn GetI8() -> i8 { return Cpp.ReturnI8(); }
fn GetI16() -> i16 { return Cpp.ReturnI16(); }
fn GetI32() -> i32 { return Cpp.ReturnI32(); }
fn GetI64() -> i64 { return Cpp.ReturnI64(); }

fn Use(a: u8, b: u16, c: u32, d: u64, e: i8, f: i16, g: i32, h: i64);

fn Lets() {
  let my_u8: u8 = Cpp.ReturnU8();
  let my_u16: u16 = Cpp.ReturnU16();
  let my_u32: u32 = Cpp.ReturnU32();
  let my_u64: u64 = Cpp.ReturnU64();

  let my_i8: i8 = Cpp.ReturnI8();
  let my_i16: i16 = Cpp.ReturnI16();
  let my_i32: i32 = Cpp.ReturnI32();
  let my_i64: i64 = Cpp.ReturnI64();

  Use(my_u8, my_u16, my_u32, my_u64, my_i8, my_i16, my_i32, my_i64);
}

fn Vars() {
  var my_u8: u8 = Cpp.ReturnU8();
  var my_u16: u16 = Cpp.ReturnU16();
  var my_u32: u32 = Cpp.ReturnU32();
  var my_u64: u64 = Cpp.ReturnU64();

  var my_i8: i8 = Cpp.ReturnI8();
  var my_i16: i16 = Cpp.ReturnI16();
  var my_i32: i32 = Cpp.ReturnI32();
  var my_i64: i64 = Cpp.ReturnI64();

  Use(my_u8, my_u16, my_u32, my_u64, my_i8, my_i16, my_i32, my_i64);
}

// --- class.h

struct X {
  X();
  X(const X&);
  ~X();

  void* _Nonnull p;
  void* _Nonnull q;
};

X Make();

// --- import_class.carbon

library "[[@TEST_NAME]]";

import Cpp library "class.h";

fn GetX() -> Cpp.X { return Cpp.Make(); }

fn Let() {
  let _: Cpp.X = Cpp.Make();
}

fn Var() {
  var _: Cpp.X = Cpp.Make();
}

// --- indirect_return_with_args.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
class C {};
class D {};
C f(D);
''';

fn Call(x: Cpp.D) -> Cpp.C {
  return Cpp.f(x);
}

// CHECK:STDOUT: ; ModuleID = 'import_ints.carbon'
// CHECK:STDOUT: source_filename = "import_ints.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i8 @_CGetU8.Main() #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc6_40.1.temp = alloca i8, align 1, !dbg !15
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc6_40.1.temp), !dbg !15
// CHECK:STDOUT:   call void @_Z8ReturnU8v.carbon_thunk.(ptr %.loc6_40.1.temp), !dbg !15
// CHECK:STDOUT:   %0 = load i8, ptr %.loc6_40.1.temp, align 1, !dbg !16
// CHECK:STDOUT:   ret i8 %0, !dbg !16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z8ReturnU8v.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !17
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !17
// CHECK:STDOUT:   %call = call noundef zeroext i8 @_Z8ReturnU8v()
// CHECK:STDOUT:   store i8 %call, ptr %0, align 1, !tbaa !20
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i16 @_CGetU16.Main() #0 !dbg !21 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_43.1.temp = alloca i16, align 2, !dbg !25
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_43.1.temp), !dbg !25
// CHECK:STDOUT:   call void @_Z9ReturnU16v.carbon_thunk.(ptr %.loc7_43.1.temp), !dbg !25
// CHECK:STDOUT:   %0 = load i16, ptr %.loc7_43.1.temp, align 2, !dbg !26
// CHECK:STDOUT:   ret i16 %0, !dbg !26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z9ReturnU16v.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !27
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !27
// CHECK:STDOUT:   %call = call noundef zeroext i16 @_Z9ReturnU16v()
// CHECK:STDOUT:   store i16 %call, ptr %0, align 2, !tbaa !29
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CGetU32.Main() #0 !dbg !31 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ReturnU32.call = call i32 @_Z9ReturnU32v(), !dbg !35
// CHECK:STDOUT:   ret i32 %ReturnU32.call, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef i32 @_Z9ReturnU32v() #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CGetU64.Main() #0 !dbg !37 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ReturnU64.call = call i64 @_Z9ReturnU64v(), !dbg !41
// CHECK:STDOUT:   ret i64 %ReturnU64.call, !dbg !42
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef i64 @_Z9ReturnU64v() #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i8 @_CGetI8.Main() #0 !dbg !43 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc11_40.1.temp = alloca i8, align 1, !dbg !47
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc11_40.1.temp), !dbg !47
// CHECK:STDOUT:   call void @_Z8ReturnI8v.carbon_thunk.(ptr %.loc11_40.1.temp), !dbg !47
// CHECK:STDOUT:   %0 = load i8, ptr %.loc11_40.1.temp, align 1, !dbg !48
// CHECK:STDOUT:   ret i8 %0, !dbg !48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z8ReturnI8v.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !17
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !17
// CHECK:STDOUT:   %call = call noundef signext i8 @_Z8ReturnI8v()
// CHECK:STDOUT:   store i8 %call, ptr %0, align 1, !tbaa !20
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i16 @_CGetI16.Main() #0 !dbg !49 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc12_43.1.temp = alloca i16, align 2, !dbg !53
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc12_43.1.temp), !dbg !53
// CHECK:STDOUT:   call void @_Z9ReturnI16v.carbon_thunk.(ptr %.loc12_43.1.temp), !dbg !53
// CHECK:STDOUT:   %0 = load i16, ptr %.loc12_43.1.temp, align 2, !dbg !54
// CHECK:STDOUT:   ret i16 %0, !dbg !54
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z9ReturnI16v.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !27
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !27
// CHECK:STDOUT:   %call = call noundef signext i16 @_Z9ReturnI16v()
// CHECK:STDOUT:   store i16 %call, ptr %0, align 2, !tbaa !29
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CGetI32.Main() #0 !dbg !55 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ReturnI32.call = call i32 @_Z9ReturnI32v(), !dbg !59
// CHECK:STDOUT:   ret i32 %ReturnI32.call, !dbg !60
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef i32 @_Z9ReturnI32v() #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CGetI64.Main() #0 !dbg !61 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ReturnI64.call = call i64 @_Z9ReturnI64v(), !dbg !65
// CHECK:STDOUT:   ret i64 %ReturnI64.call, !dbg !66
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef i64 @_Z9ReturnI64v() #2
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CUse.Main(i8, i16, i32, i64, i8, i16, i32, i64)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CLets.Main() #0 !dbg !67 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc19_32.1.temp = alloca i8, align 1, !dbg !70
// CHECK:STDOUT:   %.loc20_35.1.temp = alloca i16, align 2, !dbg !71
// CHECK:STDOUT:   %.loc24_32.1.temp = alloca i8, align 1, !dbg !72
// CHECK:STDOUT:   %.loc25_35.1.temp = alloca i16, align 2, !dbg !73
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc19_32.1.temp), !dbg !70
// CHECK:STDOUT:   call void @_Z8ReturnU8v.carbon_thunk.(ptr %.loc19_32.1.temp), !dbg !70
// CHECK:STDOUT:   %.loc19_32.4 = load i8, ptr %.loc19_32.1.temp, align 1, !dbg !70
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc20_35.1.temp), !dbg !71
// CHECK:STDOUT:   call void @_Z9ReturnU16v.carbon_thunk.(ptr %.loc20_35.1.temp), !dbg !71
// CHECK:STDOUT:   %.loc20_35.4 = load i16, ptr %.loc20_35.1.temp, align 2, !dbg !71
// CHECK:STDOUT:   %ReturnU32.call = call i32 @_Z9ReturnU32v(), !dbg !74
// CHECK:STDOUT:   %ReturnU64.call = call i64 @_Z9ReturnU64v(), !dbg !75
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc24_32.1.temp), !dbg !72
// CHECK:STDOUT:   call void @_Z8ReturnI8v.carbon_thunk.(ptr %.loc24_32.1.temp), !dbg !72
// CHECK:STDOUT:   %.loc24_32.4 = load i8, ptr %.loc24_32.1.temp, align 1, !dbg !72
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc25_35.1.temp), !dbg !73
// CHECK:STDOUT:   call void @_Z9ReturnI16v.carbon_thunk.(ptr %.loc25_35.1.temp), !dbg !73
// CHECK:STDOUT:   %.loc25_35.4 = load i16, ptr %.loc25_35.1.temp, align 2, !dbg !73
// CHECK:STDOUT:   %ReturnI32.call = call i32 @_Z9ReturnI32v(), !dbg !76
// CHECK:STDOUT:   %ReturnI64.call = call i64 @_Z9ReturnI64v(), !dbg !77
// CHECK:STDOUT:   call void @_CUse.Main(i8 %.loc19_32.4, i16 %.loc20_35.4, i32 %ReturnU32.call, i64 %ReturnU64.call, i8 %.loc24_32.4, i16 %.loc25_35.4, i32 %ReturnI32.call, i64 %ReturnI64.call), !dbg !78
// CHECK:STDOUT:   call void @"_COp.eb6702152d47e49f:core.Destroy.Core"(ptr %.loc25_35.1.temp), !dbg !73
// CHECK:STDOUT:   call void @"_COp.578b3934f85dbca6:core.Destroy.Core"(ptr %.loc24_32.1.temp), !dbg !72
// CHECK:STDOUT:   call void @"_COp.6fd0ed7da7a22bb4:core.Destroy.Core"(ptr %.loc20_35.1.temp), !dbg !71
// CHECK:STDOUT:   call void @"_COp.c3b9dc9c7e0d5d3d:core.Destroy.Core"(ptr %.loc19_32.1.temp), !dbg !70
// CHECK:STDOUT:   ret void, !dbg !79
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.eb6702152d47e49f:core.Destroy.Core"(ptr %self) #0 !dbg !80 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !85
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.578b3934f85dbca6:core.Destroy.Core"(ptr %self) #0 !dbg !86 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !91
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.6fd0ed7da7a22bb4:core.Destroy.Core"(ptr %self) #0 !dbg !92 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !97
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.c3b9dc9c7e0d5d3d:core.Destroy.Core"(ptr %self) #0 !dbg !98 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !103
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CVars.Main() #0 !dbg !104 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %my_u8.var = alloca i8, align 1, !dbg !105
// CHECK:STDOUT:   %my_u16.var = alloca i16, align 2, !dbg !106
// CHECK:STDOUT:   %my_u32.var = alloca i32, align 4, !dbg !107
// CHECK:STDOUT:   %my_u64.var = alloca i64, align 8, !dbg !108
// CHECK:STDOUT:   %my_i8.var = alloca i8, align 1, !dbg !109
// CHECK:STDOUT:   %my_i16.var = alloca i16, align 2, !dbg !110
// CHECK:STDOUT:   %my_i32.var = alloca i32, align 4, !dbg !111
// CHECK:STDOUT:   %my_i64.var = alloca i64, align 8, !dbg !112
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %my_u8.var), !dbg !105
// CHECK:STDOUT:   call void @_Z8ReturnU8v.carbon_thunk.(ptr %my_u8.var), !dbg !113
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %my_u16.var), !dbg !106
// CHECK:STDOUT:   call void @_Z9ReturnU16v.carbon_thunk.(ptr %my_u16.var), !dbg !114
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %my_u32.var), !dbg !107
// CHECK:STDOUT:   %ReturnU32.call = call i32 @_Z9ReturnU32v(), !dbg !115
// CHECK:STDOUT:   store i32 %ReturnU32.call, ptr %my_u32.var, align 4, !dbg !107
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %my_u64.var), !dbg !108
// CHECK:STDOUT:   %ReturnU64.call = call i64 @_Z9ReturnU64v(), !dbg !116
// CHECK:STDOUT:   store i64 %ReturnU64.call, ptr %my_u64.var, align 8, !dbg !108
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %my_i8.var), !dbg !109
// CHECK:STDOUT:   call void @_Z8ReturnI8v.carbon_thunk.(ptr %my_i8.var), !dbg !117
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %my_i16.var), !dbg !110
// CHECK:STDOUT:   call void @_Z9ReturnI16v.carbon_thunk.(ptr %my_i16.var), !dbg !118
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %my_i32.var), !dbg !111
// CHECK:STDOUT:   %ReturnI32.call = call i32 @_Z9ReturnI32v(), !dbg !119
// CHECK:STDOUT:   store i32 %ReturnI32.call, ptr %my_i32.var, align 4, !dbg !111
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %my_i64.var), !dbg !112
// CHECK:STDOUT:   %ReturnI64.call = call i64 @_Z9ReturnI64v(), !dbg !120
// CHECK:STDOUT:   store i64 %ReturnI64.call, ptr %my_i64.var, align 8, !dbg !112
// CHECK:STDOUT:   %.loc43_7 = load i8, ptr %my_u8.var, align 1, !dbg !121
// CHECK:STDOUT:   %.loc43_14 = load i16, ptr %my_u16.var, align 2, !dbg !122
// CHECK:STDOUT:   %.loc43_22 = load i32, ptr %my_u32.var, align 4, !dbg !123
// CHECK:STDOUT:   %.loc43_30 = load i64, ptr %my_u64.var, align 8, !dbg !124
// CHECK:STDOUT:   %.loc43_38 = load i8, ptr %my_i8.var, align 1, !dbg !125
// CHECK:STDOUT:   %.loc43_45 = load i16, ptr %my_i16.var, align 2, !dbg !126
// CHECK:STDOUT:   %.loc43_53 = load i32, ptr %my_i32.var, align 4, !dbg !127
// CHECK:STDOUT:   %.loc43_61 = load i64, ptr %my_i64.var, align 8, !dbg !128
// CHECK:STDOUT:   call void @_CUse.Main(i8 %.loc43_7, i16 %.loc43_14, i32 %.loc43_22, i64 %.loc43_30, i8 %.loc43_38, i16 %.loc43_45, i32 %.loc43_53, i64 %.loc43_61), !dbg !129
// CHECK:STDOUT:   call void @"_COp.fbec46972c1998a0:core.Destroy.Core"(ptr %my_i64.var), !dbg !112
// CHECK:STDOUT:   call void @"_COp.5e27612b9dd31a14:core.Destroy.Core"(ptr %my_i32.var), !dbg !111
// CHECK:STDOUT:   call void @"_COp.eb6702152d47e49f:core.Destroy.Core"(ptr %my_i16.var), !dbg !110
// CHECK:STDOUT:   call void @"_COp.578b3934f85dbca6:core.Destroy.Core"(ptr %my_i8.var), !dbg !109
// CHECK:STDOUT:   call void @"_COp.df57f49075d1fe6b:core.Destroy.Core"(ptr %my_u64.var), !dbg !108
// CHECK:STDOUT:   call void @"_COp.45ecfb19e8cc5ab3:core.Destroy.Core"(ptr %my_u32.var), !dbg !107
// CHECK:STDOUT:   call void @"_COp.6fd0ed7da7a22bb4:core.Destroy.Core"(ptr %my_u16.var), !dbg !106
// CHECK:STDOUT:   call void @"_COp.c3b9dc9c7e0d5d3d:core.Destroy.Core"(ptr %my_u8.var), !dbg !105
// CHECK:STDOUT:   ret void, !dbg !130
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.fbec46972c1998a0:core.Destroy.Core"(ptr %self) #0 !dbg !131 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !136
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.5e27612b9dd31a14:core.Destroy.Core"(ptr %self) #0 !dbg !137 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !142
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.df57f49075d1fe6b:core.Destroy.Core"(ptr %self) #0 !dbg !143 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !148
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.45ecfb19e8cc5ab3:core.Destroy.Core"(ptr %self) #0 !dbg !149 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !154
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef zeroext i8 @_Z8ReturnU8v() #2
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef zeroext i16 @_Z9ReturnU16v() #2
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef signext i8 @_Z8ReturnI8v() #2
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef signext i16 @_Z9ReturnI16v() #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1, !2, !3, !4}
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!7}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !3 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !4 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !6, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIFile(filename: "import_ints.carbon", directory: "")
// CHECK:STDOUT: !7 = !{!8, !8, i64 0}
// CHECK:STDOUT: !8 = !{!"int", !9, i64 0}
// CHECK:STDOUT: !9 = !{!"omnipotent char", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "GetU8", linkageName: "_CGetU8.Main", scope: null, file: !6, line: 6, type: !12, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !12 = !DISubroutineType(types: !13)
// CHECK:STDOUT: !13 = !{!14}
// CHECK:STDOUT: !14 = !DIBasicType(name: "int", size: 8, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !15 = !DILocation(line: 6, column: 27, scope: !11)
// CHECK:STDOUT: !16 = !DILocation(line: 6, column: 20, scope: !11)
// CHECK:STDOUT: !17 = !{!18, !18, i64 0}
// CHECK:STDOUT: !18 = !{!"p1 omnipotent char", !19, i64 0}
// CHECK:STDOUT: !19 = !{!"any pointer", !9, i64 0}
// CHECK:STDOUT: !20 = !{!9, !9, i64 0}
// CHECK:STDOUT: !21 = distinct !DISubprogram(name: "GetU16", linkageName: "_CGetU16.Main", scope: null, file: !6, line: 7, type: !22, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !22 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !23 = !{!24}
// CHECK:STDOUT: !24 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !25 = !DILocation(line: 7, column: 29, scope: !21)
// CHECK:STDOUT: !26 = !DILocation(line: 7, column: 22, scope: !21)
// CHECK:STDOUT: !27 = !{!28, !28, i64 0}
// CHECK:STDOUT: !28 = !{!"p1 short", !19, i64 0}
// CHECK:STDOUT: !29 = !{!30, !30, i64 0}
// CHECK:STDOUT: !30 = !{!"short", !9, i64 0}
// CHECK:STDOUT: !31 = distinct !DISubprogram(name: "GetU32", linkageName: "_CGetU32.Main", scope: null, file: !6, line: 8, type: !32, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !32 = !DISubroutineType(types: !33)
// CHECK:STDOUT: !33 = !{!34}
// CHECK:STDOUT: !34 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !35 = !DILocation(line: 8, column: 29, scope: !31)
// CHECK:STDOUT: !36 = !DILocation(line: 8, column: 22, scope: !31)
// CHECK:STDOUT: !37 = distinct !DISubprogram(name: "GetU64", linkageName: "_CGetU64.Main", scope: null, file: !6, line: 9, type: !38, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !38 = !DISubroutineType(types: !39)
// CHECK:STDOUT: !39 = !{!40}
// CHECK:STDOUT: !40 = !DIBasicType(name: "int", size: 64, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !41 = !DILocation(line: 9, column: 29, scope: !37)
// CHECK:STDOUT: !42 = !DILocation(line: 9, column: 22, scope: !37)
// CHECK:STDOUT: !43 = distinct !DISubprogram(name: "GetI8", linkageName: "_CGetI8.Main", scope: null, file: !6, line: 11, type: !44, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !44 = !DISubroutineType(types: !45)
// CHECK:STDOUT: !45 = !{!46}
// CHECK:STDOUT: !46 = !DIBasicType(name: "int", size: 8, encoding: DW_ATE_signed)
// CHECK:STDOUT: !47 = !DILocation(line: 11, column: 27, scope: !43)
// CHECK:STDOUT: !48 = !DILocation(line: 11, column: 20, scope: !43)
// CHECK:STDOUT: !49 = distinct !DISubprogram(name: "GetI16", linkageName: "_CGetI16.Main", scope: null, file: !6, line: 12, type: !50, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !50 = !DISubroutineType(types: !51)
// CHECK:STDOUT: !51 = !{!52}
// CHECK:STDOUT: !52 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_signed)
// CHECK:STDOUT: !53 = !DILocation(line: 12, column: 29, scope: !49)
// CHECK:STDOUT: !54 = !DILocation(line: 12, column: 22, scope: !49)
// CHECK:STDOUT: !55 = distinct !DISubprogram(name: "GetI32", linkageName: "_CGetI32.Main", scope: null, file: !6, line: 13, type: !56, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !56 = !DISubroutineType(types: !57)
// CHECK:STDOUT: !57 = !{!58}
// CHECK:STDOUT: !58 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !59 = !DILocation(line: 13, column: 29, scope: !55)
// CHECK:STDOUT: !60 = !DILocation(line: 13, column: 22, scope: !55)
// CHECK:STDOUT: !61 = distinct !DISubprogram(name: "GetI64", linkageName: "_CGetI64.Main", scope: null, file: !6, line: 14, type: !62, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !62 = !DISubroutineType(types: !63)
// CHECK:STDOUT: !63 = !{!64}
// CHECK:STDOUT: !64 = !DIBasicType(name: "int", size: 64, encoding: DW_ATE_signed)
// CHECK:STDOUT: !65 = !DILocation(line: 14, column: 29, scope: !61)
// CHECK:STDOUT: !66 = !DILocation(line: 14, column: 22, scope: !61)
// CHECK:STDOUT: !67 = distinct !DISubprogram(name: "Lets", linkageName: "_CLets.Main", scope: null, file: !6, line: 18, type: !68, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !68 = !DISubroutineType(types: !69)
// CHECK:STDOUT: !69 = !{null}
// CHECK:STDOUT: !70 = !DILocation(line: 19, column: 19, scope: !67)
// CHECK:STDOUT: !71 = !DILocation(line: 20, column: 21, scope: !67)
// CHECK:STDOUT: !72 = !DILocation(line: 24, column: 19, scope: !67)
// CHECK:STDOUT: !73 = !DILocation(line: 25, column: 21, scope: !67)
// CHECK:STDOUT: !74 = !DILocation(line: 21, column: 21, scope: !67)
// CHECK:STDOUT: !75 = !DILocation(line: 22, column: 21, scope: !67)
// CHECK:STDOUT: !76 = !DILocation(line: 26, column: 21, scope: !67)
// CHECK:STDOUT: !77 = !DILocation(line: 27, column: 21, scope: !67)
// CHECK:STDOUT: !78 = !DILocation(line: 29, column: 3, scope: !67)
// CHECK:STDOUT: !79 = !DILocation(line: 18, column: 1, scope: !67)
// CHECK:STDOUT: !80 = distinct !DISubprogram(name: "Op", linkageName: "_COp.eb6702152d47e49f:core.Destroy.Core", scope: null, file: !6, line: 25, type: !81, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !83)
// CHECK:STDOUT: !81 = !DISubroutineType(types: !82)
// CHECK:STDOUT: !82 = !{null, !52}
// CHECK:STDOUT: !83 = !{!84}
// CHECK:STDOUT: !84 = !DILocalVariable(arg: 1, scope: !80, type: !52)
// CHECK:STDOUT: !85 = !DILocation(line: 25, column: 21, scope: !80)
// CHECK:STDOUT: !86 = distinct !DISubprogram(name: "Op", linkageName: "_COp.578b3934f85dbca6:core.Destroy.Core", scope: null, file: !6, line: 24, type: !87, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !89)
// CHECK:STDOUT: !87 = !DISubroutineType(types: !88)
// CHECK:STDOUT: !88 = !{null, !46}
// CHECK:STDOUT: !89 = !{!90}
// CHECK:STDOUT: !90 = !DILocalVariable(arg: 1, scope: !86, type: !46)
// CHECK:STDOUT: !91 = !DILocation(line: 24, column: 19, scope: !86)
// CHECK:STDOUT: !92 = distinct !DISubprogram(name: "Op", linkageName: "_COp.6fd0ed7da7a22bb4:core.Destroy.Core", scope: null, file: !6, line: 20, type: !93, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !95)
// CHECK:STDOUT: !93 = !DISubroutineType(types: !94)
// CHECK:STDOUT: !94 = !{null, !24}
// CHECK:STDOUT: !95 = !{!96}
// CHECK:STDOUT: !96 = !DILocalVariable(arg: 1, scope: !92, type: !24)
// CHECK:STDOUT: !97 = !DILocation(line: 20, column: 21, scope: !92)
// CHECK:STDOUT: !98 = distinct !DISubprogram(name: "Op", linkageName: "_COp.c3b9dc9c7e0d5d3d:core.Destroy.Core", scope: null, file: !6, line: 19, type: !99, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !101)
// CHECK:STDOUT: !99 = !DISubroutineType(types: !100)
// CHECK:STDOUT: !100 = !{null, !14}
// CHECK:STDOUT: !101 = !{!102}
// CHECK:STDOUT: !102 = !DILocalVariable(arg: 1, scope: !98, type: !14)
// CHECK:STDOUT: !103 = !DILocation(line: 19, column: 19, scope: !98)
// CHECK:STDOUT: !104 = distinct !DISubprogram(name: "Vars", linkageName: "_CVars.Main", scope: null, file: !6, line: 32, type: !68, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !105 = !DILocation(line: 33, column: 3, scope: !104)
// CHECK:STDOUT: !106 = !DILocation(line: 34, column: 3, scope: !104)
// CHECK:STDOUT: !107 = !DILocation(line: 35, column: 3, scope: !104)
// CHECK:STDOUT: !108 = !DILocation(line: 36, column: 3, scope: !104)
// CHECK:STDOUT: !109 = !DILocation(line: 38, column: 3, scope: !104)
// CHECK:STDOUT: !110 = !DILocation(line: 39, column: 3, scope: !104)
// CHECK:STDOUT: !111 = !DILocation(line: 40, column: 3, scope: !104)
// CHECK:STDOUT: !112 = !DILocation(line: 41, column: 3, scope: !104)
// CHECK:STDOUT: !113 = !DILocation(line: 33, column: 19, scope: !104)
// CHECK:STDOUT: !114 = !DILocation(line: 34, column: 21, scope: !104)
// CHECK:STDOUT: !115 = !DILocation(line: 35, column: 21, scope: !104)
// CHECK:STDOUT: !116 = !DILocation(line: 36, column: 21, scope: !104)
// CHECK:STDOUT: !117 = !DILocation(line: 38, column: 19, scope: !104)
// CHECK:STDOUT: !118 = !DILocation(line: 39, column: 21, scope: !104)
// CHECK:STDOUT: !119 = !DILocation(line: 40, column: 21, scope: !104)
// CHECK:STDOUT: !120 = !DILocation(line: 41, column: 21, scope: !104)
// CHECK:STDOUT: !121 = !DILocation(line: 43, column: 7, scope: !104)
// CHECK:STDOUT: !122 = !DILocation(line: 43, column: 14, scope: !104)
// CHECK:STDOUT: !123 = !DILocation(line: 43, column: 22, scope: !104)
// CHECK:STDOUT: !124 = !DILocation(line: 43, column: 30, scope: !104)
// CHECK:STDOUT: !125 = !DILocation(line: 43, column: 38, scope: !104)
// CHECK:STDOUT: !126 = !DILocation(line: 43, column: 45, scope: !104)
// CHECK:STDOUT: !127 = !DILocation(line: 43, column: 53, scope: !104)
// CHECK:STDOUT: !128 = !DILocation(line: 43, column: 61, scope: !104)
// CHECK:STDOUT: !129 = !DILocation(line: 43, column: 3, scope: !104)
// CHECK:STDOUT: !130 = !DILocation(line: 32, column: 1, scope: !104)
// CHECK:STDOUT: !131 = distinct !DISubprogram(name: "Op", linkageName: "_COp.fbec46972c1998a0:core.Destroy.Core", scope: null, file: !6, line: 41, type: !132, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !134)
// CHECK:STDOUT: !132 = !DISubroutineType(types: !133)
// CHECK:STDOUT: !133 = !{null, !64}
// CHECK:STDOUT: !134 = !{!135}
// CHECK:STDOUT: !135 = !DILocalVariable(arg: 1, scope: !131, type: !64)
// CHECK:STDOUT: !136 = !DILocation(line: 41, column: 3, scope: !131)
// CHECK:STDOUT: !137 = distinct !DISubprogram(name: "Op", linkageName: "_COp.5e27612b9dd31a14:core.Destroy.Core", scope: null, file: !6, line: 40, type: !138, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !140)
// CHECK:STDOUT: !138 = !DISubroutineType(types: !139)
// CHECK:STDOUT: !139 = !{null, !58}
// CHECK:STDOUT: !140 = !{!141}
// CHECK:STDOUT: !141 = !DILocalVariable(arg: 1, scope: !137, type: !58)
// CHECK:STDOUT: !142 = !DILocation(line: 40, column: 3, scope: !137)
// CHECK:STDOUT: !143 = distinct !DISubprogram(name: "Op", linkageName: "_COp.df57f49075d1fe6b:core.Destroy.Core", scope: null, file: !6, line: 36, type: !144, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !146)
// CHECK:STDOUT: !144 = !DISubroutineType(types: !145)
// CHECK:STDOUT: !145 = !{null, !40}
// CHECK:STDOUT: !146 = !{!147}
// CHECK:STDOUT: !147 = !DILocalVariable(arg: 1, scope: !143, type: !40)
// CHECK:STDOUT: !148 = !DILocation(line: 36, column: 3, scope: !143)
// CHECK:STDOUT: !149 = distinct !DISubprogram(name: "Op", linkageName: "_COp.45ecfb19e8cc5ab3:core.Destroy.Core", scope: null, file: !6, line: 35, type: !150, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !152)
// CHECK:STDOUT: !150 = !DISubroutineType(types: !151)
// CHECK:STDOUT: !151 = !{null, !34}
// CHECK:STDOUT: !152 = !{!153}
// CHECK:STDOUT: !153 = !DILocalVariable(arg: 1, scope: !149, type: !34)
// CHECK:STDOUT: !154 = !DILocation(line: 35, column: 3, scope: !149)
// CHECK:STDOUT: ; ModuleID = 'import_class.carbon'
// CHECK:STDOUT: source_filename = "import_class.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %struct.X = type { ptr, ptr }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CGetX.Main(ptr sret([16 x i8]) %return) #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z4Makev.carbon_thunk.(ptr %return), !dbg !15
// CHECK:STDOUT:   ret void, !dbg !16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z4Makev.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !17
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !17
// CHECK:STDOUT:   call void @_Z4Makev(ptr dead_on_unwind writable sret(%struct.X) align 8 %0)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CLet.Main() #0 !dbg !20 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_27.1.temp = alloca [16 x i8], align 8, !dbg !23
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc9_27.1.temp), !dbg !23
// CHECK:STDOUT:   call void @_Z4Makev.carbon_thunk.(ptr %.loc9_27.1.temp), !dbg !23
// CHECK:STDOUT:   call void @_ZN1XD1Ev(ptr %.loc9_27.1.temp), !dbg !23
// CHECK:STDOUT:   ret void, !dbg !24
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: declare void @_ZN1XD1Ev(ptr noundef nonnull align 8 dead_on_return(16) dereferenceable(16)) unnamed_addr #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CVar.Main() #0 !dbg !25 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca [16 x i8], align 8, !dbg !26
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !26
// CHECK:STDOUT:   call void @_Z4Makev.carbon_thunk.(ptr %_.var), !dbg !27
// CHECK:STDOUT:   call void @_ZN1XD1Ev(ptr %_.var), !dbg !26
// CHECK:STDOUT:   ret void, !dbg !28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z4Makev(ptr dead_on_unwind writable sret(%struct.X) align 8) #4
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #4 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1, !2, !3, !4}
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!7}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !3 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !4 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !6, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIFile(filename: "import_class.carbon", directory: "")
// CHECK:STDOUT: !7 = !{!8, !8, i64 0}
// CHECK:STDOUT: !8 = !{!"int", !9, i64 0}
// CHECK:STDOUT: !9 = !{!"omnipotent char", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "GetX", linkageName: "_CGetX.Main", scope: null, file: !6, line: 6, type: !12, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !12 = !DISubroutineType(types: !13)
// CHECK:STDOUT: !13 = !{!14}
// CHECK:STDOUT: !14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !15 = !DILocation(line: 6, column: 29, scope: !11)
// CHECK:STDOUT: !16 = !DILocation(line: 6, column: 22, scope: !11)
// CHECK:STDOUT: !17 = !{!18, !18, i64 0}
// CHECK:STDOUT: !18 = !{!"p1 _ZTS1X", !19, i64 0}
// CHECK:STDOUT: !19 = !{!"any pointer", !9, i64 0}
// CHECK:STDOUT: !20 = distinct !DISubprogram(name: "Let", linkageName: "_CLet.Main", scope: null, file: !6, line: 8, type: !21, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !21 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !22 = !{null}
// CHECK:STDOUT: !23 = !DILocation(line: 9, column: 18, scope: !20)
// CHECK:STDOUT: !24 = !DILocation(line: 8, column: 1, scope: !20)
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "Var", linkageName: "_CVar.Main", scope: null, file: !6, line: 12, type: !21, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !26 = !DILocation(line: 13, column: 3, scope: !25)
// CHECK:STDOUT: !27 = !DILocation(line: 13, column: 18, scope: !25)
// CHECK:STDOUT: !28 = !DILocation(line: 12, column: 1, scope: !25)
// CHECK:STDOUT: ; ModuleID = 'indirect_return_with_args.carbon'
// CHECK:STDOUT: source_filename = "indirect_return_with_args.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %class.D = type { i8 }
// CHECK:STDOUT: %class.C = type { i8 }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCall.Main(ptr sret({}) %return, ptr %x) #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z1f1D.carbon_thunk._(ptr %x, ptr %return), !dbg !17
// CHECK:STDOUT:   ret void, !dbg !18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z1f1D.carbon_thunk._(ptr noundef %0, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %class.D, align 1
// CHECK:STDOUT:   %undef.agg.tmp = alloca %class.C, align 1
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !19
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !22
// CHECK:STDOUT:   %1 = load ptr, ptr %return.addr, align 8, !tbaa !22
// CHECK:STDOUT:   %2 = load ptr, ptr %.addr, align 8, !tbaa !19
// CHECK:STDOUT:   call void @_Z1f1D()
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z1f1D() #2
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1, !2, !3, !4}
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!7}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !3 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !4 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !6, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIFile(filename: "indirect_return_with_args.carbon", directory: "")
// CHECK:STDOUT: !7 = !{!8, !8, i64 0}
// CHECK:STDOUT: !8 = !{!"int", !9, i64 0}
// CHECK:STDOUT: !9 = !{!"omnipotent char", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "Call", linkageName: "_CCall.Main", scope: null, file: !6, line: 10, type: !12, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !15)
// CHECK:STDOUT: !12 = !DISubroutineType(types: !13)
// CHECK:STDOUT: !13 = !{!14, !14}
// CHECK:STDOUT: !14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !15 = !{!16}
// CHECK:STDOUT: !16 = !DILocalVariable(arg: 1, scope: !11, type: !14)
// CHECK:STDOUT: !17 = !DILocation(line: 11, column: 10, scope: !11)
// CHECK:STDOUT: !18 = !DILocation(line: 11, column: 3, scope: !11)
// CHECK:STDOUT: !19 = !{!20, !20, i64 0}
// CHECK:STDOUT: !20 = !{!"p1 _ZTS1D", !21, i64 0}
// CHECK:STDOUT: !21 = !{!"any pointer", !9, i64 0}
// CHECK:STDOUT: !22 = !{!23, !23, i64 0}
// CHECK:STDOUT: !23 = !{!"p1 _ZTS1C", !21, i64 0}
