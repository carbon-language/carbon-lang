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

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'import_ints.carbon'
// CHECK:STDOUT: source_filename = "import_ints.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i8 @_CGetU8.Main() #0 !dbg !53 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc6_40.1.temp = alloca i8, align 1, !dbg !56
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc6_40.1.temp), !dbg !56
// CHECK:STDOUT:   call void @_Z8ReturnU8v.carbon_thunk.(ptr %.loc6_40.1.temp), !dbg !56
// CHECK:STDOUT:   %0 = load i8, ptr %.loc6_40.1.temp, align 1, !dbg !57
// CHECK:STDOUT:   ret i8 %0, !dbg !57
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z8ReturnU8v.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !263
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !263
// CHECK:STDOUT:   %call = call noundef zeroext i8 @_Z8ReturnU8v()
// CHECK:STDOUT:   store i8 %call, ptr %0, align 1, !tbaa !264
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i16 @_CGetU16.Main() #0 !dbg !58 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_43.1.temp = alloca i16, align 2, !dbg !61
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_43.1.temp), !dbg !61
// CHECK:STDOUT:   call void @_Z9ReturnU16v.carbon_thunk.(ptr %.loc7_43.1.temp), !dbg !61
// CHECK:STDOUT:   %0 = load i16, ptr %.loc7_43.1.temp, align 2, !dbg !62
// CHECK:STDOUT:   ret i16 %0, !dbg !62
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z9ReturnU16v.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !267
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !267
// CHECK:STDOUT:   %call = call noundef zeroext i16 @_Z9ReturnU16v()
// CHECK:STDOUT:   store i16 %call, ptr %0, align 2, !tbaa !268
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CGetU32.Main() #0 !dbg !63 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ReturnU32.call = call i32 @_Z9ReturnU32v(), !dbg !66
// CHECK:STDOUT:   ret i32 %ReturnU32.call, !dbg !67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef i32 @_Z9ReturnU32v() #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CGetU64.Main() #0 !dbg !68 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ReturnU64.call = call i64 @_Z9ReturnU64v(), !dbg !71
// CHECK:STDOUT:   ret i64 %ReturnU64.call, !dbg !72
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef i64 @_Z9ReturnU64v() #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i8 @_CGetI8.Main() #0 !dbg !73 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc11_40.1.temp = alloca i8, align 1, !dbg !76
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc11_40.1.temp), !dbg !76
// CHECK:STDOUT:   call void @_Z8ReturnI8v.carbon_thunk.(ptr %.loc11_40.1.temp), !dbg !76
// CHECK:STDOUT:   %0 = load i8, ptr %.loc11_40.1.temp, align 1, !dbg !77
// CHECK:STDOUT:   ret i8 %0, !dbg !77
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z8ReturnI8v.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !263
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !263
// CHECK:STDOUT:   %call = call noundef signext i8 @_Z8ReturnI8v()
// CHECK:STDOUT:   store i8 %call, ptr %0, align 1, !tbaa !264
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i16 @_CGetI16.Main() #0 !dbg !78 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc12_43.1.temp = alloca i16, align 2, !dbg !81
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc12_43.1.temp), !dbg !81
// CHECK:STDOUT:   call void @_Z9ReturnI16v.carbon_thunk.(ptr %.loc12_43.1.temp), !dbg !81
// CHECK:STDOUT:   %0 = load i16, ptr %.loc12_43.1.temp, align 2, !dbg !82
// CHECK:STDOUT:   ret i16 %0, !dbg !82
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z9ReturnI16v.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !267
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !267
// CHECK:STDOUT:   %call = call noundef signext i16 @_Z9ReturnI16v()
// CHECK:STDOUT:   store i16 %call, ptr %0, align 2, !tbaa !268
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CGetI32.Main() #0 !dbg !83 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ReturnI32.call = call i32 @_Z9ReturnI32v(), !dbg !86
// CHECK:STDOUT:   ret i32 %ReturnI32.call, !dbg !87
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef i32 @_Z9ReturnI32v() #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CGetI64.Main() #0 !dbg !88 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ReturnI64.call = call i64 @_Z9ReturnI64v(), !dbg !91
// CHECK:STDOUT:   ret i64 %ReturnI64.call, !dbg !92
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef i64 @_Z9ReturnI64v() #2
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CUse.Main(i8, i16, i32, i64, i8, i16, i32, i64)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CLets.Main() #0 !dbg !93 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc19_32.1.temp = alloca i8, align 1, !dbg !94
// CHECK:STDOUT:   %.loc20_35.1.temp = alloca i16, align 2, !dbg !97
// CHECK:STDOUT:   %.loc24_32.1.temp = alloca i8, align 1, !dbg !106
// CHECK:STDOUT:   %.loc25_35.1.temp = alloca i16, align 2, !dbg !109
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc19_32.1.temp), !dbg !94
// CHECK:STDOUT:   call void @_Z8ReturnU8v.carbon_thunk.(ptr %.loc19_32.1.temp), !dbg !94
// CHECK:STDOUT:   %.loc19_32.4 = load i8, ptr %.loc19_32.1.temp, align 1, !dbg !94
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc20_35.1.temp), !dbg !97
// CHECK:STDOUT:   call void @_Z9ReturnU16v.carbon_thunk.(ptr %.loc20_35.1.temp), !dbg !97
// CHECK:STDOUT:   %.loc20_35.4 = load i16, ptr %.loc20_35.1.temp, align 2, !dbg !97
// CHECK:STDOUT:   %ReturnU32.call = call i32 @_Z9ReturnU32v(), !dbg !100
// CHECK:STDOUT:   %ReturnU64.call = call i64 @_Z9ReturnU64v(), !dbg !103
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc24_32.1.temp), !dbg !106
// CHECK:STDOUT:   call void @_Z8ReturnI8v.carbon_thunk.(ptr %.loc24_32.1.temp), !dbg !106
// CHECK:STDOUT:   %.loc24_32.4 = load i8, ptr %.loc24_32.1.temp, align 1, !dbg !106
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc25_35.1.temp), !dbg !109
// CHECK:STDOUT:   call void @_Z9ReturnI16v.carbon_thunk.(ptr %.loc25_35.1.temp), !dbg !109
// CHECK:STDOUT:   %.loc25_35.4 = load i16, ptr %.loc25_35.1.temp, align 2, !dbg !109
// CHECK:STDOUT:   %ReturnI32.call = call i32 @_Z9ReturnI32v(), !dbg !112
// CHECK:STDOUT:   %ReturnI64.call = call i64 @_Z9ReturnI64v(), !dbg !115
// CHECK:STDOUT:   call void @_CUse.Main(i8 %.loc19_32.4, i16 %.loc20_35.4, i32 %ReturnU32.call, i64 %ReturnU64.call, i8 %.loc24_32.4, i16 %.loc25_35.4, i32 %ReturnI32.call, i64 %ReturnI64.call), !dbg !126
// CHECK:STDOUT:   call void @"_CSelfDestruct.d57cb32fcf12ec07:core.Destroy.Core"(ptr %.loc25_35.1.temp), !dbg !109
// CHECK:STDOUT:   call void @"_CSelfDestruct.4954785a8b134801:core.Destroy.Core"(ptr %.loc24_32.1.temp), !dbg !106
// CHECK:STDOUT:   call void @"_CSelfDestruct.3bc9e5dacf328f74:core.Destroy.Core"(ptr %.loc20_35.1.temp), !dbg !97
// CHECK:STDOUT:   call void @"_CSelfDestruct.be60742f00557eec:core.Destroy.Core"(ptr %.loc19_32.1.temp), !dbg !94
// CHECK:STDOUT:   ret void, !dbg !127
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.ea6dff92ea7b44ef:core.Destroy.Core"(ptr %self) #0 !dbg !128 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !130
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.d57cb32fcf12ec07:core.Destroy.Core"(ptr %self) #0 !dbg !132 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !134
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.d57cb32fcf12ec07:core.Destroy.Core"(ptr %self) #0 !dbg !136 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.d57cb32fcf12ec07:core.Destroy.Core"(ptr %self), !dbg !138
// CHECK:STDOUT:   ret void, !dbg !138
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.c4a4fcc00062c7f0:core.Destroy.Core"(ptr %self) #0 !dbg !140 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !142
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.4954785a8b134801:core.Destroy.Core"(ptr %self) #0 !dbg !144 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !146
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.4954785a8b134801:core.Destroy.Core"(ptr %self) #0 !dbg !148 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.4954785a8b134801:core.Destroy.Core"(ptr %self), !dbg !150
// CHECK:STDOUT:   ret void, !dbg !150
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.99892fadd247b492:core.Destroy.Core"(ptr %self) #0 !dbg !152 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !154
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.3bc9e5dacf328f74:core.Destroy.Core"(ptr %self) #0 !dbg !156 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !158
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.3bc9e5dacf328f74:core.Destroy.Core"(ptr %self) #0 !dbg !160 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.3bc9e5dacf328f74:core.Destroy.Core"(ptr %self), !dbg !162
// CHECK:STDOUT:   ret void, !dbg !162
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.04514f1c5a56b0cb:core.Destroy.Core"(ptr %self) #0 !dbg !164 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !166
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.be60742f00557eec:core.Destroy.Core"(ptr %self) #0 !dbg !168 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !170
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.be60742f00557eec:core.Destroy.Core"(ptr %self) #0 !dbg !172 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.be60742f00557eec:core.Destroy.Core"(ptr %self), !dbg !174
// CHECK:STDOUT:   ret void, !dbg !174
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CVars.Main() #0 !dbg !176 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %my_u8.var = alloca i8, align 1, !dbg !177
// CHECK:STDOUT:   %my_u16.var = alloca i16, align 2, !dbg !180
// CHECK:STDOUT:   %my_u32.var = alloca i32, align 4, !dbg !183
// CHECK:STDOUT:   %my_u64.var = alloca i64, align 8, !dbg !186
// CHECK:STDOUT:   %my_i8.var = alloca i8, align 1, !dbg !189
// CHECK:STDOUT:   %my_i16.var = alloca i16, align 2, !dbg !192
// CHECK:STDOUT:   %my_i32.var = alloca i32, align 4, !dbg !195
// CHECK:STDOUT:   %my_i64.var = alloca i64, align 8, !dbg !198
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %my_u8.var), !dbg !177
// CHECK:STDOUT:   call void @_Z8ReturnU8v.carbon_thunk.(ptr %my_u8.var), !dbg !178
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %my_u16.var), !dbg !180
// CHECK:STDOUT:   call void @_Z9ReturnU16v.carbon_thunk.(ptr %my_u16.var), !dbg !181
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %my_u32.var), !dbg !183
// CHECK:STDOUT:   %ReturnU32.call = call i32 @_Z9ReturnU32v(), !dbg !184
// CHECK:STDOUT:   store i32 %ReturnU32.call, ptr %my_u32.var, align 4, !dbg !183
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %my_u64.var), !dbg !186
// CHECK:STDOUT:   %ReturnU64.call = call i64 @_Z9ReturnU64v(), !dbg !187
// CHECK:STDOUT:   store i64 %ReturnU64.call, ptr %my_u64.var, align 8, !dbg !186
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %my_i8.var), !dbg !189
// CHECK:STDOUT:   call void @_Z8ReturnI8v.carbon_thunk.(ptr %my_i8.var), !dbg !190
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %my_i16.var), !dbg !192
// CHECK:STDOUT:   call void @_Z9ReturnI16v.carbon_thunk.(ptr %my_i16.var), !dbg !193
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %my_i32.var), !dbg !195
// CHECK:STDOUT:   %ReturnI32.call = call i32 @_Z9ReturnI32v(), !dbg !196
// CHECK:STDOUT:   store i32 %ReturnI32.call, ptr %my_i32.var, align 4, !dbg !195
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %my_i64.var), !dbg !198
// CHECK:STDOUT:   %ReturnI64.call = call i64 @_Z9ReturnI64v(), !dbg !199
// CHECK:STDOUT:   store i64 %ReturnI64.call, ptr %my_i64.var, align 8, !dbg !198
// CHECK:STDOUT:   %.loc43_7 = load i8, ptr %my_u8.var, align 1, !dbg !201
// CHECK:STDOUT:   %.loc43_14 = load i16, ptr %my_u16.var, align 2, !dbg !202
// CHECK:STDOUT:   %.loc43_22 = load i32, ptr %my_u32.var, align 4, !dbg !203
// CHECK:STDOUT:   %.loc43_30 = load i64, ptr %my_u64.var, align 8, !dbg !204
// CHECK:STDOUT:   %.loc43_38 = load i8, ptr %my_i8.var, align 1, !dbg !205
// CHECK:STDOUT:   %.loc43_45 = load i16, ptr %my_i16.var, align 2, !dbg !206
// CHECK:STDOUT:   %.loc43_53 = load i32, ptr %my_i32.var, align 4, !dbg !207
// CHECK:STDOUT:   %.loc43_61 = load i64, ptr %my_i64.var, align 8, !dbg !208
// CHECK:STDOUT:   call void @_CUse.Main(i8 %.loc43_7, i16 %.loc43_14, i32 %.loc43_22, i64 %.loc43_30, i8 %.loc43_38, i16 %.loc43_45, i32 %.loc43_53, i64 %.loc43_61), !dbg !209
// CHECK:STDOUT:   call void @"_CSelfDestruct.5aacfd7dcc330b17:core.Destroy.Core"(ptr %my_i64.var), !dbg !198
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %my_i32.var), !dbg !195
// CHECK:STDOUT:   call void @"_CSelfDestruct.d57cb32fcf12ec07:core.Destroy.Core"(ptr %my_i16.var), !dbg !192
// CHECK:STDOUT:   call void @"_CSelfDestruct.4954785a8b134801:core.Destroy.Core"(ptr %my_i8.var), !dbg !189
// CHECK:STDOUT:   call void @"_CSelfDestruct.18a2a1ac0d16cf80:core.Destroy.Core"(ptr %my_u64.var), !dbg !186
// CHECK:STDOUT:   call void @"_CSelfDestruct.ea3ccedbd6a1c75d:core.Destroy.Core"(ptr %my_u32.var), !dbg !183
// CHECK:STDOUT:   call void @"_CSelfDestruct.3bc9e5dacf328f74:core.Destroy.Core"(ptr %my_u16.var), !dbg !180
// CHECK:STDOUT:   call void @"_CSelfDestruct.be60742f00557eec:core.Destroy.Core"(ptr %my_u8.var), !dbg !177
// CHECK:STDOUT:   ret void, !dbg !210
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.6e5a4ef51c9e705a:core.Destroy.Core"(ptr %self) #0 !dbg !211 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !213
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.5aacfd7dcc330b17:core.Destroy.Core"(ptr %self) #0 !dbg !215 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !217
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.5aacfd7dcc330b17:core.Destroy.Core"(ptr %self) #0 !dbg !219 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.5aacfd7dcc330b17:core.Destroy.Core"(ptr %self), !dbg !221
// CHECK:STDOUT:   ret void, !dbg !221
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.6dae29ece9a708c4:core.Destroy.Core"(ptr %self) #0 !dbg !223 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !225
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !227 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !229
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !231 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self), !dbg !233
// CHECK:STDOUT:   ret void, !dbg !233
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.98b8c1354df43b20:core.Destroy.Core"(ptr %self) #0 !dbg !235 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !237
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.18a2a1ac0d16cf80:core.Destroy.Core"(ptr %self) #0 !dbg !239 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !241
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.18a2a1ac0d16cf80:core.Destroy.Core"(ptr %self) #0 !dbg !243 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.18a2a1ac0d16cf80:core.Destroy.Core"(ptr %self), !dbg !245
// CHECK:STDOUT:   ret void, !dbg !245
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.b18ca5df050aa3af:core.Destroy.Core"(ptr %self) #0 !dbg !247 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !249
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.ea3ccedbd6a1c75d:core.Destroy.Core"(ptr %self) #0 !dbg !251 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !253
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.ea3ccedbd6a1c75d:core.Destroy.Core"(ptr %self) #0 !dbg !255 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.ea3ccedbd6a1c75d:core.Destroy.Core"(ptr %self), !dbg !257
// CHECK:STDOUT:   ret void, !dbg !257
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.module.flags = !{!269, !270, !271, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!274}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "import_ints.carbon", directory: "")
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 8, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !7 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !9 = !DIBasicType(name: "int", size: 64, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !10 = !DIBasicType(name: "int", size: 8, encoding: DW_ATE_signed)
// CHECK:STDOUT: !11 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_signed)
// CHECK:STDOUT: !12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !13 = !DIBasicType(name: "int", size: 64, encoding: DW_ATE_signed)
// CHECK:STDOUT: !14 = !{!6}
// CHECK:STDOUT: !15 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !19 = !{!7}
// CHECK:STDOUT: !20 = !DISubroutineType(types: !19)
// CHECK:STDOUT: !21 = !{!8}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = !{!9}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !25 = !{!10}
// CHECK:STDOUT: !26 = !DISubroutineType(types: !25)
// CHECK:STDOUT: !27 = !{!11}
// CHECK:STDOUT: !28 = !DISubroutineType(types: !27)
// CHECK:STDOUT: !29 = !{!12}
// CHECK:STDOUT: !30 = !DISubroutineType(types: !29)
// CHECK:STDOUT: !31 = !{!13}
// CHECK:STDOUT: !32 = !DISubroutineType(types: !31)
// CHECK:STDOUT: !35 = !{null}
// CHECK:STDOUT: !36 = !DISubroutineType(types: !35)
// CHECK:STDOUT: !37 = !{null, !11}
// CHECK:STDOUT: !38 = !DISubroutineType(types: !37)
// CHECK:STDOUT: !39 = !{null, !10}
// CHECK:STDOUT: !40 = !DISubroutineType(types: !39)
// CHECK:STDOUT: !41 = !{null, !7}
// CHECK:STDOUT: !42 = !DISubroutineType(types: !41)
// CHECK:STDOUT: !43 = !{null, !6}
// CHECK:STDOUT: !44 = !DISubroutineType(types: !43)
// CHECK:STDOUT: !45 = !{null, !13}
// CHECK:STDOUT: !46 = !DISubroutineType(types: !45)
// CHECK:STDOUT: !47 = !{null, !12}
// CHECK:STDOUT: !48 = !DISubroutineType(types: !47)
// CHECK:STDOUT: !49 = !{null, !9}
// CHECK:STDOUT: !50 = !DISubroutineType(types: !49)
// CHECK:STDOUT: !51 = !{null, !8}
// CHECK:STDOUT: !52 = !DISubroutineType(types: !51)
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "GetU8", linkageName: "_CGetU8.Main", scope: null, file: !4, line: 6, type: !15, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !56 = !DILocation(line: 6, column: 27, scope: !53)
// CHECK:STDOUT: !57 = !DILocation(line: 6, column: 20, scope: !53)
// CHECK:STDOUT: !58 = distinct !DISubprogram(name: "GetU16", linkageName: "_CGetU16.Main", scope: null, file: !4, line: 7, type: !20, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !61 = !DILocation(line: 7, column: 29, scope: !58)
// CHECK:STDOUT: !62 = !DILocation(line: 7, column: 22, scope: !58)
// CHECK:STDOUT: !63 = distinct !DISubprogram(name: "GetU32", linkageName: "_CGetU32.Main", scope: null, file: !4, line: 8, type: !22, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !66 = !DILocation(line: 8, column: 29, scope: !63)
// CHECK:STDOUT: !67 = !DILocation(line: 8, column: 22, scope: !63)
// CHECK:STDOUT: !68 = distinct !DISubprogram(name: "GetU64", linkageName: "_CGetU64.Main", scope: null, file: !4, line: 9, type: !24, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !71 = !DILocation(line: 9, column: 29, scope: !68)
// CHECK:STDOUT: !72 = !DILocation(line: 9, column: 22, scope: !68)
// CHECK:STDOUT: !73 = distinct !DISubprogram(name: "GetI8", linkageName: "_CGetI8.Main", scope: null, file: !4, line: 11, type: !26, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !76 = !DILocation(line: 11, column: 27, scope: !73)
// CHECK:STDOUT: !77 = !DILocation(line: 11, column: 20, scope: !73)
// CHECK:STDOUT: !78 = distinct !DISubprogram(name: "GetI16", linkageName: "_CGetI16.Main", scope: null, file: !4, line: 12, type: !28, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !81 = !DILocation(line: 12, column: 29, scope: !78)
// CHECK:STDOUT: !82 = !DILocation(line: 12, column: 22, scope: !78)
// CHECK:STDOUT: !83 = distinct !DISubprogram(name: "GetI32", linkageName: "_CGetI32.Main", scope: null, file: !4, line: 13, type: !30, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !86 = !DILocation(line: 13, column: 29, scope: !83)
// CHECK:STDOUT: !87 = !DILocation(line: 13, column: 22, scope: !83)
// CHECK:STDOUT: !88 = distinct !DISubprogram(name: "GetI64", linkageName: "_CGetI64.Main", scope: null, file: !4, line: 14, type: !32, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !91 = !DILocation(line: 14, column: 29, scope: !88)
// CHECK:STDOUT: !92 = !DILocation(line: 14, column: 22, scope: !88)
// CHECK:STDOUT: !93 = distinct !DISubprogram(name: "Lets", linkageName: "_CLets.Main", scope: null, file: !4, line: 18, type: !36, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !94 = !DILocation(line: 19, column: 19, scope: !93)
// CHECK:STDOUT: !97 = !DILocation(line: 20, column: 21, scope: !93)
// CHECK:STDOUT: !100 = !DILocation(line: 21, column: 21, scope: !93)
// CHECK:STDOUT: !103 = !DILocation(line: 22, column: 21, scope: !93)
// CHECK:STDOUT: !106 = !DILocation(line: 24, column: 19, scope: !93)
// CHECK:STDOUT: !109 = !DILocation(line: 25, column: 21, scope: !93)
// CHECK:STDOUT: !112 = !DILocation(line: 26, column: 21, scope: !93)
// CHECK:STDOUT: !115 = !DILocation(line: 27, column: 21, scope: !93)
// CHECK:STDOUT: !126 = !DILocation(line: 29, column: 3, scope: !93)
// CHECK:STDOUT: !127 = !DILocation(line: 18, column: 1, scope: !93)
// CHECK:STDOUT: !128 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.ea6dff92ea7b44ef:core.Destroy.Core", scope: null, file: !4, line: 25, type: !38, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !131)
// CHECK:STDOUT: !129 = !DILocalVariable(arg: 1, scope: !128, type: !11)
// CHECK:STDOUT: !130 = !DILocation(line: 25, column: 21, scope: !128)
// CHECK:STDOUT: !131 = !{!129}
// CHECK:STDOUT: !132 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.d57cb32fcf12ec07:core.Destroy.Core", scope: null, file: !4, line: 25, type: !38, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !135)
// CHECK:STDOUT: !133 = !DILocalVariable(arg: 1, scope: !132, type: !11)
// CHECK:STDOUT: !134 = !DILocation(line: 25, column: 21, scope: !132)
// CHECK:STDOUT: !135 = !{!133}
// CHECK:STDOUT: !136 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.d57cb32fcf12ec07:core.Destroy.Core", scope: null, file: !4, line: 25, type: !38, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !139)
// CHECK:STDOUT: !137 = !DILocalVariable(arg: 1, scope: !136, type: !11)
// CHECK:STDOUT: !138 = !DILocation(line: 25, column: 21, scope: !136)
// CHECK:STDOUT: !139 = !{!137}
// CHECK:STDOUT: !140 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.c4a4fcc00062c7f0:core.Destroy.Core", scope: null, file: !4, line: 24, type: !40, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !143)
// CHECK:STDOUT: !141 = !DILocalVariable(arg: 1, scope: !140, type: !10)
// CHECK:STDOUT: !142 = !DILocation(line: 24, column: 19, scope: !140)
// CHECK:STDOUT: !143 = !{!141}
// CHECK:STDOUT: !144 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.4954785a8b134801:core.Destroy.Core", scope: null, file: !4, line: 24, type: !40, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !147)
// CHECK:STDOUT: !145 = !DILocalVariable(arg: 1, scope: !144, type: !10)
// CHECK:STDOUT: !146 = !DILocation(line: 24, column: 19, scope: !144)
// CHECK:STDOUT: !147 = !{!145}
// CHECK:STDOUT: !148 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.4954785a8b134801:core.Destroy.Core", scope: null, file: !4, line: 24, type: !40, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !151)
// CHECK:STDOUT: !149 = !DILocalVariable(arg: 1, scope: !148, type: !10)
// CHECK:STDOUT: !150 = !DILocation(line: 24, column: 19, scope: !148)
// CHECK:STDOUT: !151 = !{!149}
// CHECK:STDOUT: !152 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.99892fadd247b492:core.Destroy.Core", scope: null, file: !4, line: 20, type: !42, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !155)
// CHECK:STDOUT: !153 = !DILocalVariable(arg: 1, scope: !152, type: !7)
// CHECK:STDOUT: !154 = !DILocation(line: 20, column: 21, scope: !152)
// CHECK:STDOUT: !155 = !{!153}
// CHECK:STDOUT: !156 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.3bc9e5dacf328f74:core.Destroy.Core", scope: null, file: !4, line: 20, type: !42, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !159)
// CHECK:STDOUT: !157 = !DILocalVariable(arg: 1, scope: !156, type: !7)
// CHECK:STDOUT: !158 = !DILocation(line: 20, column: 21, scope: !156)
// CHECK:STDOUT: !159 = !{!157}
// CHECK:STDOUT: !160 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.3bc9e5dacf328f74:core.Destroy.Core", scope: null, file: !4, line: 20, type: !42, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !163)
// CHECK:STDOUT: !161 = !DILocalVariable(arg: 1, scope: !160, type: !7)
// CHECK:STDOUT: !162 = !DILocation(line: 20, column: 21, scope: !160)
// CHECK:STDOUT: !163 = !{!161}
// CHECK:STDOUT: !164 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.04514f1c5a56b0cb:core.Destroy.Core", scope: null, file: !4, line: 19, type: !44, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !167)
// CHECK:STDOUT: !165 = !DILocalVariable(arg: 1, scope: !164, type: !6)
// CHECK:STDOUT: !166 = !DILocation(line: 19, column: 19, scope: !164)
// CHECK:STDOUT: !167 = !{!165}
// CHECK:STDOUT: !168 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.be60742f00557eec:core.Destroy.Core", scope: null, file: !4, line: 19, type: !44, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !171)
// CHECK:STDOUT: !169 = !DILocalVariable(arg: 1, scope: !168, type: !6)
// CHECK:STDOUT: !170 = !DILocation(line: 19, column: 19, scope: !168)
// CHECK:STDOUT: !171 = !{!169}
// CHECK:STDOUT: !172 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.be60742f00557eec:core.Destroy.Core", scope: null, file: !4, line: 19, type: !44, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !175)
// CHECK:STDOUT: !173 = !DILocalVariable(arg: 1, scope: !172, type: !6)
// CHECK:STDOUT: !174 = !DILocation(line: 19, column: 19, scope: !172)
// CHECK:STDOUT: !175 = !{!173}
// CHECK:STDOUT: !176 = distinct !DISubprogram(name: "Vars", linkageName: "_CVars.Main", scope: null, file: !4, line: 32, type: !36, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !177 = !DILocation(line: 33, column: 3, scope: !176)
// CHECK:STDOUT: !178 = !DILocation(line: 33, column: 19, scope: !176)
// CHECK:STDOUT: !180 = !DILocation(line: 34, column: 3, scope: !176)
// CHECK:STDOUT: !181 = !DILocation(line: 34, column: 21, scope: !176)
// CHECK:STDOUT: !183 = !DILocation(line: 35, column: 3, scope: !176)
// CHECK:STDOUT: !184 = !DILocation(line: 35, column: 21, scope: !176)
// CHECK:STDOUT: !186 = !DILocation(line: 36, column: 3, scope: !176)
// CHECK:STDOUT: !187 = !DILocation(line: 36, column: 21, scope: !176)
// CHECK:STDOUT: !189 = !DILocation(line: 38, column: 3, scope: !176)
// CHECK:STDOUT: !190 = !DILocation(line: 38, column: 19, scope: !176)
// CHECK:STDOUT: !192 = !DILocation(line: 39, column: 3, scope: !176)
// CHECK:STDOUT: !193 = !DILocation(line: 39, column: 21, scope: !176)
// CHECK:STDOUT: !195 = !DILocation(line: 40, column: 3, scope: !176)
// CHECK:STDOUT: !196 = !DILocation(line: 40, column: 21, scope: !176)
// CHECK:STDOUT: !198 = !DILocation(line: 41, column: 3, scope: !176)
// CHECK:STDOUT: !199 = !DILocation(line: 41, column: 21, scope: !176)
// CHECK:STDOUT: !201 = !DILocation(line: 43, column: 7, scope: !176)
// CHECK:STDOUT: !202 = !DILocation(line: 43, column: 14, scope: !176)
// CHECK:STDOUT: !203 = !DILocation(line: 43, column: 22, scope: !176)
// CHECK:STDOUT: !204 = !DILocation(line: 43, column: 30, scope: !176)
// CHECK:STDOUT: !205 = !DILocation(line: 43, column: 38, scope: !176)
// CHECK:STDOUT: !206 = !DILocation(line: 43, column: 45, scope: !176)
// CHECK:STDOUT: !207 = !DILocation(line: 43, column: 53, scope: !176)
// CHECK:STDOUT: !208 = !DILocation(line: 43, column: 61, scope: !176)
// CHECK:STDOUT: !209 = !DILocation(line: 43, column: 3, scope: !176)
// CHECK:STDOUT: !210 = !DILocation(line: 32, column: 1, scope: !176)
// CHECK:STDOUT: !211 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.6e5a4ef51c9e705a:core.Destroy.Core", scope: null, file: !4, line: 41, type: !46, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !214)
// CHECK:STDOUT: !212 = !DILocalVariable(arg: 1, scope: !211, type: !13)
// CHECK:STDOUT: !213 = !DILocation(line: 41, column: 3, scope: !211)
// CHECK:STDOUT: !214 = !{!212}
// CHECK:STDOUT: !215 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.5aacfd7dcc330b17:core.Destroy.Core", scope: null, file: !4, line: 41, type: !46, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !218)
// CHECK:STDOUT: !216 = !DILocalVariable(arg: 1, scope: !215, type: !13)
// CHECK:STDOUT: !217 = !DILocation(line: 41, column: 3, scope: !215)
// CHECK:STDOUT: !218 = !{!216}
// CHECK:STDOUT: !219 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.5aacfd7dcc330b17:core.Destroy.Core", scope: null, file: !4, line: 41, type: !46, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !222)
// CHECK:STDOUT: !220 = !DILocalVariable(arg: 1, scope: !219, type: !13)
// CHECK:STDOUT: !221 = !DILocation(line: 41, column: 3, scope: !219)
// CHECK:STDOUT: !222 = !{!220}
// CHECK:STDOUT: !223 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.6dae29ece9a708c4:core.Destroy.Core", scope: null, file: !4, line: 40, type: !48, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !226)
// CHECK:STDOUT: !224 = !DILocalVariable(arg: 1, scope: !223, type: !12)
// CHECK:STDOUT: !225 = !DILocation(line: 40, column: 3, scope: !223)
// CHECK:STDOUT: !226 = !{!224}
// CHECK:STDOUT: !227 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core", scope: null, file: !4, line: 40, type: !48, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !230)
// CHECK:STDOUT: !228 = !DILocalVariable(arg: 1, scope: !227, type: !12)
// CHECK:STDOUT: !229 = !DILocation(line: 40, column: 3, scope: !227)
// CHECK:STDOUT: !230 = !{!228}
// CHECK:STDOUT: !231 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f58524831c37807e:core.Destroy.Core", scope: null, file: !4, line: 40, type: !48, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !234)
// CHECK:STDOUT: !232 = !DILocalVariable(arg: 1, scope: !231, type: !12)
// CHECK:STDOUT: !233 = !DILocation(line: 40, column: 3, scope: !231)
// CHECK:STDOUT: !234 = !{!232}
// CHECK:STDOUT: !235 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.98b8c1354df43b20:core.Destroy.Core", scope: null, file: !4, line: 36, type: !50, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !238)
// CHECK:STDOUT: !236 = !DILocalVariable(arg: 1, scope: !235, type: !9)
// CHECK:STDOUT: !237 = !DILocation(line: 36, column: 3, scope: !235)
// CHECK:STDOUT: !238 = !{!236}
// CHECK:STDOUT: !239 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.18a2a1ac0d16cf80:core.Destroy.Core", scope: null, file: !4, line: 36, type: !50, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !242)
// CHECK:STDOUT: !240 = !DILocalVariable(arg: 1, scope: !239, type: !9)
// CHECK:STDOUT: !241 = !DILocation(line: 36, column: 3, scope: !239)
// CHECK:STDOUT: !242 = !{!240}
// CHECK:STDOUT: !243 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.18a2a1ac0d16cf80:core.Destroy.Core", scope: null, file: !4, line: 36, type: !50, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !246)
// CHECK:STDOUT: !244 = !DILocalVariable(arg: 1, scope: !243, type: !9)
// CHECK:STDOUT: !245 = !DILocation(line: 36, column: 3, scope: !243)
// CHECK:STDOUT: !246 = !{!244}
// CHECK:STDOUT: !247 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.b18ca5df050aa3af:core.Destroy.Core", scope: null, file: !4, line: 35, type: !52, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !250)
// CHECK:STDOUT: !248 = !DILocalVariable(arg: 1, scope: !247, type: !8)
// CHECK:STDOUT: !249 = !DILocation(line: 35, column: 3, scope: !247)
// CHECK:STDOUT: !250 = !{!248}
// CHECK:STDOUT: !251 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.ea3ccedbd6a1c75d:core.Destroy.Core", scope: null, file: !4, line: 35, type: !52, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !254)
// CHECK:STDOUT: !252 = !DILocalVariable(arg: 1, scope: !251, type: !8)
// CHECK:STDOUT: !253 = !DILocation(line: 35, column: 3, scope: !251)
// CHECK:STDOUT: !254 = !{!252}
// CHECK:STDOUT: !255 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.ea3ccedbd6a1c75d:core.Destroy.Core", scope: null, file: !4, line: 35, type: !52, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !258)
// CHECK:STDOUT: !256 = !DILocalVariable(arg: 1, scope: !255, type: !8)
// CHECK:STDOUT: !257 = !DILocation(line: 35, column: 3, scope: !255)
// CHECK:STDOUT: !258 = !{!256}
// CHECK:STDOUT: !259 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !260 = !{!"omnipotent char", !259, i64 0}
// CHECK:STDOUT: !261 = !{!"any pointer", !260, i64 0}
// CHECK:STDOUT: !262 = !{!"p1 omnipotent char", !261, i64 0}
// CHECK:STDOUT: !263 = !{!262, !262, i64 0}
// CHECK:STDOUT: !264 = !{!260, !260, i64 0}
// CHECK:STDOUT: !265 = !{!"short", !260, i64 0}
// CHECK:STDOUT: !266 = !{!"p1 short", !261, i64 0}
// CHECK:STDOUT: !267 = !{!266, !266, i64 0}
// CHECK:STDOUT: !268 = !{!265, !265, i64 0}
// CHECK:STDOUT: !269 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !270 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !271 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !272 = !{!"int", !260, i64 0}
// CHECK:STDOUT: !273 = !{!"__libc_errno", !272, i64 0}
// CHECK:STDOUT: !274 = !{!273, !272, i64 0}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'import_class.carbon'
// CHECK:STDOUT: source_filename = "import_class.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %struct.X = type { ptr, ptr }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CGetX.Main(ptr sret([16 x i8]) %return) #0 !dbg !279 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z4Makev.carbon_thunk.(ptr %return), !dbg !282
// CHECK:STDOUT:   ret void, !dbg !283
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z4Makev.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !299
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !299
// CHECK:STDOUT:   call void @_Z4Makev(ptr dead_on_unwind writable sret(%struct.X) align 8 %0)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CLet.Main() #0 !dbg !284 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_27.1.temp = alloca [16 x i8], align 8, !dbg !285
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc9_27.1.temp), !dbg !285
// CHECK:STDOUT:   call void @_Z4Makev.carbon_thunk.(ptr %.loc9_27.1.temp), !dbg !285
// CHECK:STDOUT:   call void @"_CSelfDestruct.3d7b8a41b4c67e71:core.Destroy.Core"(ptr %.loc9_27.1.temp), !dbg !285
// CHECK:STDOUT:   ret void, !dbg !288
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: declare void @_ZN1XD1Ev(ptr noundef nonnull align 8 dead_on_return(16) dereferenceable(16)) unnamed_addr #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.3d7b8a41b4c67e71:core.Destroy.Core"(ptr %self) #0 !dbg !289 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN1XD1Ev(ptr %self), !dbg !291
// CHECK:STDOUT:   ret void, !dbg !291
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CVar.Main() #0 !dbg !293 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca [16 x i8], align 8, !dbg !294
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !294
// CHECK:STDOUT:   call void @_Z4Makev.carbon_thunk.(ptr %_.var), !dbg !295
// CHECK:STDOUT:   call void @"_CSelfDestruct.3d7b8a41b4c67e71:core.Destroy.Core"(ptr %_.var), !dbg !294
// CHECK:STDOUT:   ret void, !dbg !297
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!276}
// CHECK:STDOUT: !llvm.module.flags = !{!269, !270, !271, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!274}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !16 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !17 = !{null, !16}
// CHECK:STDOUT: !18 = !DISubroutineType(types: !17)
// CHECK:STDOUT: !35 = !{null}
// CHECK:STDOUT: !36 = !DISubroutineType(types: !35)
// CHECK:STDOUT: !259 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !260 = !{!"omnipotent char", !259, i64 0}
// CHECK:STDOUT: !261 = !{!"any pointer", !260, i64 0}
// CHECK:STDOUT: !269 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !270 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !271 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !272 = !{!"int", !260, i64 0}
// CHECK:STDOUT: !273 = !{!"__libc_errno", !272, i64 0}
// CHECK:STDOUT: !274 = !{!273, !272, i64 0}
// CHECK:STDOUT: !275 = !DIFile(filename: "import_class.carbon", directory: "")
// CHECK:STDOUT: !276 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !275, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !277 = !{!16}
// CHECK:STDOUT: !278 = !DISubroutineType(types: !277)
// CHECK:STDOUT: !279 = distinct !DISubprogram(name: "GetX", linkageName: "_CGetX.Main", scope: null, file: !275, line: 6, type: !278, spFlags: DISPFlagDefinition, unit: !276)
// CHECK:STDOUT: !282 = !DILocation(line: 6, column: 29, scope: !279)
// CHECK:STDOUT: !283 = !DILocation(line: 6, column: 22, scope: !279)
// CHECK:STDOUT: !284 = distinct !DISubprogram(name: "Let", linkageName: "_CLet.Main", scope: null, file: !275, line: 8, type: !36, spFlags: DISPFlagDefinition, unit: !276)
// CHECK:STDOUT: !285 = !DILocation(line: 9, column: 18, scope: !284)
// CHECK:STDOUT: !288 = !DILocation(line: 8, column: 1, scope: !284)
// CHECK:STDOUT: !289 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.3d7b8a41b4c67e71:core.Destroy.Core", scope: null, file: !275, line: 9, type: !18, spFlags: DISPFlagDefinition, unit: !276, retainedNodes: !292)
// CHECK:STDOUT: !290 = !DILocalVariable(arg: 1, scope: !289, type: !16)
// CHECK:STDOUT: !291 = !DILocation(line: 9, column: 18, scope: !289)
// CHECK:STDOUT: !292 = !{!290}
// CHECK:STDOUT: !293 = distinct !DISubprogram(name: "Var", linkageName: "_CVar.Main", scope: null, file: !275, line: 12, type: !36, spFlags: DISPFlagDefinition, unit: !276)
// CHECK:STDOUT: !294 = !DILocation(line: 13, column: 3, scope: !293)
// CHECK:STDOUT: !295 = !DILocation(line: 13, column: 18, scope: !293)
// CHECK:STDOUT: !297 = !DILocation(line: 12, column: 1, scope: !293)
// CHECK:STDOUT: !298 = !{!"p1 _ZTS1X", !261, i64 0}
// CHECK:STDOUT: !299 = !{!298, !298, i64 0}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'indirect_return_with_args.carbon'
// CHECK:STDOUT: source_filename = "indirect_return_with_args.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %class.D = type { i8 }
// CHECK:STDOUT: %class.C = type { i8 }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCall.Main(ptr sret({}) %return, ptr %x) #0 !dbg !307 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z1f1D.carbon_thunk._(ptr %x, ptr %return), !dbg !313
// CHECK:STDOUT:   ret void, !dbg !314
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z1f1D.carbon_thunk._(ptr noundef %0, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %class.D, align 1
// CHECK:STDOUT:   %undef.agg.tmp = alloca %class.C, align 1
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !317
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !319
// CHECK:STDOUT:   %1 = load ptr, ptr %return.addr, align 8, !tbaa !319
// CHECK:STDOUT:   %2 = load ptr, ptr %.addr, align 8, !tbaa !317
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!302}
// CHECK:STDOUT: !llvm.module.flags = !{!269, !270, !271, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!274}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !16 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !259 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !260 = !{!"omnipotent char", !259, i64 0}
// CHECK:STDOUT: !261 = !{!"any pointer", !260, i64 0}
// CHECK:STDOUT: !269 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !270 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !271 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !272 = !{!"int", !260, i64 0}
// CHECK:STDOUT: !273 = !{!"__libc_errno", !272, i64 0}
// CHECK:STDOUT: !274 = !{!273, !272, i64 0}
// CHECK:STDOUT: !301 = !DIFile(filename: "indirect_return_with_args.carbon", directory: "")
// CHECK:STDOUT: !302 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !301, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !303 = !{!16, !16}
// CHECK:STDOUT: !304 = !DISubroutineType(types: !303)
// CHECK:STDOUT: !307 = distinct !DISubprogram(name: "Call", linkageName: "_CCall.Main", scope: null, file: !301, line: 10, type: !304, spFlags: DISPFlagDefinition, unit: !302, retainedNodes: !315)
// CHECK:STDOUT: !308 = !DILocalVariable(arg: 1, scope: !307, type: !16)
// CHECK:STDOUT: !313 = !DILocation(line: 11, column: 10, scope: !307)
// CHECK:STDOUT: !314 = !DILocation(line: 11, column: 3, scope: !307)
// CHECK:STDOUT: !315 = !{!308}
// CHECK:STDOUT: !316 = !{!"p1 _ZTS1D", !261, i64 0}
// CHECK:STDOUT: !317 = !{!316, !316, i64 0}
// CHECK:STDOUT: !318 = !{!"p1 _ZTS1C", !261, i64 0}
// CHECK:STDOUT: !319 = !{!318, !318, i64 0}
// CHECK:STDOUT:
