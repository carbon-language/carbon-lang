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
// CHECK:STDOUT: define i8 @_CGetU8.Main() #0 !dbg !37 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc6_40.1.temp = alloca i8, align 1, !dbg !40
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc6_40.1.temp), !dbg !40
// CHECK:STDOUT:   call void @_Z8ReturnU8v.carbon_thunk.(ptr %.loc6_40.1.temp), !dbg !40
// CHECK:STDOUT:   %0 = load i8, ptr %.loc6_40.1.temp, align 1, !dbg !41
// CHECK:STDOUT:   ret i8 %0, !dbg !41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z8ReturnU8v.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !224
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !224
// CHECK:STDOUT:   %call = call noundef zeroext i8 @_Z8ReturnU8v()
// CHECK:STDOUT:   store i8 %call, ptr %0, align 1, !tbaa !225
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i16 @_CGetU16.Main() #0 !dbg !42 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_43.1.temp = alloca i16, align 2, !dbg !45
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_43.1.temp), !dbg !45
// CHECK:STDOUT:   call void @_Z9ReturnU16v.carbon_thunk.(ptr %.loc7_43.1.temp), !dbg !45
// CHECK:STDOUT:   %0 = load i16, ptr %.loc7_43.1.temp, align 2, !dbg !46
// CHECK:STDOUT:   ret i16 %0, !dbg !46
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z9ReturnU16v.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !228
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !228
// CHECK:STDOUT:   %call = call noundef zeroext i16 @_Z9ReturnU16v()
// CHECK:STDOUT:   store i16 %call, ptr %0, align 2, !tbaa !229
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CGetU32.Main() #0 !dbg !47 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ReturnU32.call = call i32 @_Z9ReturnU32v(), !dbg !50
// CHECK:STDOUT:   ret i32 %ReturnU32.call, !dbg !51
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef i32 @_Z9ReturnU32v() #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CGetU64.Main() #0 !dbg !52 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ReturnU64.call = call i64 @_Z9ReturnU64v(), !dbg !55
// CHECK:STDOUT:   ret i64 %ReturnU64.call, !dbg !56
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef i64 @_Z9ReturnU64v() #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i8 @_CGetI8.Main() #0 !dbg !57 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc11_40.1.temp = alloca i8, align 1, !dbg !60
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc11_40.1.temp), !dbg !60
// CHECK:STDOUT:   call void @_Z8ReturnI8v.carbon_thunk.(ptr %.loc11_40.1.temp), !dbg !60
// CHECK:STDOUT:   %0 = load i8, ptr %.loc11_40.1.temp, align 1, !dbg !61
// CHECK:STDOUT:   ret i8 %0, !dbg !61
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z8ReturnI8v.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !224
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !224
// CHECK:STDOUT:   %call = call noundef signext i8 @_Z8ReturnI8v()
// CHECK:STDOUT:   store i8 %call, ptr %0, align 1, !tbaa !225
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i16 @_CGetI16.Main() #0 !dbg !62 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc12_43.1.temp = alloca i16, align 2, !dbg !65
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc12_43.1.temp), !dbg !65
// CHECK:STDOUT:   call void @_Z9ReturnI16v.carbon_thunk.(ptr %.loc12_43.1.temp), !dbg !65
// CHECK:STDOUT:   %0 = load i16, ptr %.loc12_43.1.temp, align 2, !dbg !66
// CHECK:STDOUT:   ret i16 %0, !dbg !66
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z9ReturnI16v.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !228
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !228
// CHECK:STDOUT:   %call = call noundef signext i16 @_Z9ReturnI16v()
// CHECK:STDOUT:   store i16 %call, ptr %0, align 2, !tbaa !229
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CGetI32.Main() #0 !dbg !67 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ReturnI32.call = call i32 @_Z9ReturnI32v(), !dbg !70
// CHECK:STDOUT:   ret i32 %ReturnI32.call, !dbg !71
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef i32 @_Z9ReturnI32v() #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CGetI64.Main() #0 !dbg !72 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ReturnI64.call = call i64 @_Z9ReturnI64v(), !dbg !75
// CHECK:STDOUT:   ret i64 %ReturnI64.call, !dbg !76
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef i64 @_Z9ReturnI64v() #2
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CUse.Main(i8, i16, i32, i64, i8, i16, i32, i64)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CLets.Main() #0 !dbg !77 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc19_32.1.temp = alloca i8, align 1, !dbg !78
// CHECK:STDOUT:   %.loc20_35.1.temp = alloca i16, align 2, !dbg !81
// CHECK:STDOUT:   %.loc24_32.1.temp = alloca i8, align 1, !dbg !90
// CHECK:STDOUT:   %.loc25_35.1.temp = alloca i16, align 2, !dbg !93
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc19_32.1.temp), !dbg !78
// CHECK:STDOUT:   call void @_Z8ReturnU8v.carbon_thunk.(ptr %.loc19_32.1.temp), !dbg !78
// CHECK:STDOUT:   %.loc19_32.4 = load i8, ptr %.loc19_32.1.temp, align 1, !dbg !78
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc20_35.1.temp), !dbg !81
// CHECK:STDOUT:   call void @_Z9ReturnU16v.carbon_thunk.(ptr %.loc20_35.1.temp), !dbg !81
// CHECK:STDOUT:   %.loc20_35.4 = load i16, ptr %.loc20_35.1.temp, align 2, !dbg !81
// CHECK:STDOUT:   %ReturnU32.call = call i32 @_Z9ReturnU32v(), !dbg !84
// CHECK:STDOUT:   %ReturnU64.call = call i64 @_Z9ReturnU64v(), !dbg !87
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc24_32.1.temp), !dbg !90
// CHECK:STDOUT:   call void @_Z8ReturnI8v.carbon_thunk.(ptr %.loc24_32.1.temp), !dbg !90
// CHECK:STDOUT:   %.loc24_32.4 = load i8, ptr %.loc24_32.1.temp, align 1, !dbg !90
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc25_35.1.temp), !dbg !93
// CHECK:STDOUT:   call void @_Z9ReturnI16v.carbon_thunk.(ptr %.loc25_35.1.temp), !dbg !93
// CHECK:STDOUT:   %.loc25_35.4 = load i16, ptr %.loc25_35.1.temp, align 2, !dbg !93
// CHECK:STDOUT:   %ReturnI32.call = call i32 @_Z9ReturnI32v(), !dbg !96
// CHECK:STDOUT:   %ReturnI64.call = call i64 @_Z9ReturnI64v(), !dbg !99
// CHECK:STDOUT:   call void @_CUse.Main(i8 %.loc19_32.4, i16 %.loc20_35.4, i32 %ReturnU32.call, i64 %ReturnU64.call, i8 %.loc24_32.4, i16 %.loc25_35.4, i32 %ReturnI32.call, i64 %ReturnI64.call), !dbg !110
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.e43967308dd20a9f(ptr %.loc25_35.1.temp), !dbg !93
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.6fb12317fdf5c9b3(ptr %.loc24_32.1.temp), !dbg !90
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.44e7310a0b394feb(ptr %.loc20_35.1.temp), !dbg !81
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.73549b49ca05bf4c(ptr %.loc19_32.1.temp), !dbg !78
// CHECK:STDOUT:   ret void, !dbg !119
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CVars.Main() #0 !dbg !120 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %my_u8.var = alloca i8, align 1, !dbg !121
// CHECK:STDOUT:   %my_u16.var = alloca i16, align 2, !dbg !124
// CHECK:STDOUT:   %my_u32.var = alloca i32, align 4, !dbg !127
// CHECK:STDOUT:   %my_u64.var = alloca i64, align 8, !dbg !130
// CHECK:STDOUT:   %my_i8.var = alloca i8, align 1, !dbg !133
// CHECK:STDOUT:   %my_i16.var = alloca i16, align 2, !dbg !136
// CHECK:STDOUT:   %my_i32.var = alloca i32, align 4, !dbg !139
// CHECK:STDOUT:   %my_i64.var = alloca i64, align 8, !dbg !142
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %my_u8.var), !dbg !121
// CHECK:STDOUT:   call void @_Z8ReturnU8v.carbon_thunk.(ptr %my_u8.var), !dbg !122
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %my_u16.var), !dbg !124
// CHECK:STDOUT:   call void @_Z9ReturnU16v.carbon_thunk.(ptr %my_u16.var), !dbg !125
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %my_u32.var), !dbg !127
// CHECK:STDOUT:   %ReturnU32.call = call i32 @_Z9ReturnU32v(), !dbg !128
// CHECK:STDOUT:   store i32 %ReturnU32.call, ptr %my_u32.var, align 4, !dbg !127
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %my_u64.var), !dbg !130
// CHECK:STDOUT:   %ReturnU64.call = call i64 @_Z9ReturnU64v(), !dbg !131
// CHECK:STDOUT:   store i64 %ReturnU64.call, ptr %my_u64.var, align 8, !dbg !130
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %my_i8.var), !dbg !133
// CHECK:STDOUT:   call void @_Z8ReturnI8v.carbon_thunk.(ptr %my_i8.var), !dbg !134
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %my_i16.var), !dbg !136
// CHECK:STDOUT:   call void @_Z9ReturnI16v.carbon_thunk.(ptr %my_i16.var), !dbg !137
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %my_i32.var), !dbg !139
// CHECK:STDOUT:   %ReturnI32.call = call i32 @_Z9ReturnI32v(), !dbg !140
// CHECK:STDOUT:   store i32 %ReturnI32.call, ptr %my_i32.var, align 4, !dbg !139
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %my_i64.var), !dbg !142
// CHECK:STDOUT:   %ReturnI64.call = call i64 @_Z9ReturnI64v(), !dbg !143
// CHECK:STDOUT:   store i64 %ReturnI64.call, ptr %my_i64.var, align 8, !dbg !142
// CHECK:STDOUT:   %.loc43_7 = load i8, ptr %my_u8.var, align 1, !dbg !145
// CHECK:STDOUT:   %.loc43_14 = load i16, ptr %my_u16.var, align 2, !dbg !146
// CHECK:STDOUT:   %.loc43_22 = load i32, ptr %my_u32.var, align 4, !dbg !147
// CHECK:STDOUT:   %.loc43_30 = load i64, ptr %my_u64.var, align 8, !dbg !148
// CHECK:STDOUT:   %.loc43_38 = load i8, ptr %my_i8.var, align 1, !dbg !149
// CHECK:STDOUT:   %.loc43_45 = load i16, ptr %my_i16.var, align 2, !dbg !150
// CHECK:STDOUT:   %.loc43_53 = load i32, ptr %my_i32.var, align 4, !dbg !151
// CHECK:STDOUT:   %.loc43_61 = load i64, ptr %my_i64.var, align 8, !dbg !152
// CHECK:STDOUT:   call void @_CUse.Main(i8 %.loc43_7, i16 %.loc43_14, i32 %.loc43_22, i64 %.loc43_30, i8 %.loc43_38, i16 %.loc43_45, i32 %.loc43_53, i64 %.loc43_61), !dbg !153
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.7e9df21fc3951ad1(ptr %my_i64.var), !dbg !142
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %my_i32.var), !dbg !139
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.e43967308dd20a9f(ptr %my_i16.var), !dbg !136
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.6fb12317fdf5c9b3(ptr %my_i8.var), !dbg !133
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.3266e5c7a8b68ec1(ptr %my_u64.var), !dbg !130
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.ff339b9821763c59(ptr %my_u32.var), !dbg !127
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.44e7310a0b394feb(ptr %my_u16.var), !dbg !124
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.73549b49ca05bf4c(ptr %my_u8.var), !dbg !121
// CHECK:STDOUT:   ret void, !dbg !162
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.e43967308dd20a9f(ptr %t) #0 !dbg !164 {
// CHECK:STDOUT:   ret void, !dbg !169
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.6fb12317fdf5c9b3(ptr %t) #0 !dbg !171 {
// CHECK:STDOUT:   ret void, !dbg !176
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.44e7310a0b394feb(ptr %t) #0 !dbg !178 {
// CHECK:STDOUT:   ret void, !dbg !183
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.73549b49ca05bf4c(ptr %t) #0 !dbg !185 {
// CHECK:STDOUT:   ret void, !dbg !190
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.7e9df21fc3951ad1(ptr %t) #0 !dbg !192 {
// CHECK:STDOUT:   ret void, !dbg !197
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.f127f30143696d51(ptr %t) #0 !dbg !199 {
// CHECK:STDOUT:   ret void, !dbg !204
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.3266e5c7a8b68ec1(ptr %t) #0 !dbg !206 {
// CHECK:STDOUT:   ret void, !dbg !211
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.ff339b9821763c59(ptr %t) #0 !dbg !213 {
// CHECK:STDOUT:   ret void, !dbg !218
// CHECK:STDOUT: }
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
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.e43967308dd20a9f, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.6fb12317fdf5c9b3, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.44e7310a0b394feb, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.73549b49ca05bf4c, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.module.flags = !{!230, !231, !232, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!235}
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
// CHECK:STDOUT: !37 = distinct !DISubprogram(name: "GetU8", linkageName: "_CGetU8.Main", scope: null, file: !4, line: 6, type: !15, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !40 = !DILocation(line: 6, column: 27, scope: !37)
// CHECK:STDOUT: !41 = !DILocation(line: 6, column: 20, scope: !37)
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "GetU16", linkageName: "_CGetU16.Main", scope: null, file: !4, line: 7, type: !20, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !45 = !DILocation(line: 7, column: 29, scope: !42)
// CHECK:STDOUT: !46 = !DILocation(line: 7, column: 22, scope: !42)
// CHECK:STDOUT: !47 = distinct !DISubprogram(name: "GetU32", linkageName: "_CGetU32.Main", scope: null, file: !4, line: 8, type: !22, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !50 = !DILocation(line: 8, column: 29, scope: !47)
// CHECK:STDOUT: !51 = !DILocation(line: 8, column: 22, scope: !47)
// CHECK:STDOUT: !52 = distinct !DISubprogram(name: "GetU64", linkageName: "_CGetU64.Main", scope: null, file: !4, line: 9, type: !24, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !55 = !DILocation(line: 9, column: 29, scope: !52)
// CHECK:STDOUT: !56 = !DILocation(line: 9, column: 22, scope: !52)
// CHECK:STDOUT: !57 = distinct !DISubprogram(name: "GetI8", linkageName: "_CGetI8.Main", scope: null, file: !4, line: 11, type: !26, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !60 = !DILocation(line: 11, column: 27, scope: !57)
// CHECK:STDOUT: !61 = !DILocation(line: 11, column: 20, scope: !57)
// CHECK:STDOUT: !62 = distinct !DISubprogram(name: "GetI16", linkageName: "_CGetI16.Main", scope: null, file: !4, line: 12, type: !28, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !65 = !DILocation(line: 12, column: 29, scope: !62)
// CHECK:STDOUT: !66 = !DILocation(line: 12, column: 22, scope: !62)
// CHECK:STDOUT: !67 = distinct !DISubprogram(name: "GetI32", linkageName: "_CGetI32.Main", scope: null, file: !4, line: 13, type: !30, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !70 = !DILocation(line: 13, column: 29, scope: !67)
// CHECK:STDOUT: !71 = !DILocation(line: 13, column: 22, scope: !67)
// CHECK:STDOUT: !72 = distinct !DISubprogram(name: "GetI64", linkageName: "_CGetI64.Main", scope: null, file: !4, line: 14, type: !32, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !75 = !DILocation(line: 14, column: 29, scope: !72)
// CHECK:STDOUT: !76 = !DILocation(line: 14, column: 22, scope: !72)
// CHECK:STDOUT: !77 = distinct !DISubprogram(name: "Lets", linkageName: "_CLets.Main", scope: null, file: !4, line: 18, type: !36, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !78 = !DILocation(line: 19, column: 19, scope: !77)
// CHECK:STDOUT: !81 = !DILocation(line: 20, column: 21, scope: !77)
// CHECK:STDOUT: !84 = !DILocation(line: 21, column: 21, scope: !77)
// CHECK:STDOUT: !87 = !DILocation(line: 22, column: 21, scope: !77)
// CHECK:STDOUT: !90 = !DILocation(line: 24, column: 19, scope: !77)
// CHECK:STDOUT: !93 = !DILocation(line: 25, column: 21, scope: !77)
// CHECK:STDOUT: !96 = !DILocation(line: 26, column: 21, scope: !77)
// CHECK:STDOUT: !99 = !DILocation(line: 27, column: 21, scope: !77)
// CHECK:STDOUT: !110 = !DILocation(line: 29, column: 3, scope: !77)
// CHECK:STDOUT: !111 = !{null, !11}
// CHECK:STDOUT: !112 = !DISubroutineType(types: !111)
// CHECK:STDOUT: !113 = !{null, !10}
// CHECK:STDOUT: !114 = !DISubroutineType(types: !113)
// CHECK:STDOUT: !115 = !{null, !7}
// CHECK:STDOUT: !116 = !DISubroutineType(types: !115)
// CHECK:STDOUT: !117 = !{null, !6}
// CHECK:STDOUT: !118 = !DISubroutineType(types: !117)
// CHECK:STDOUT: !119 = !DILocation(line: 18, column: 1, scope: !77)
// CHECK:STDOUT: !120 = distinct !DISubprogram(name: "Vars", linkageName: "_CVars.Main", scope: null, file: !4, line: 32, type: !36, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !121 = !DILocation(line: 33, column: 3, scope: !120)
// CHECK:STDOUT: !122 = !DILocation(line: 33, column: 19, scope: !120)
// CHECK:STDOUT: !124 = !DILocation(line: 34, column: 3, scope: !120)
// CHECK:STDOUT: !125 = !DILocation(line: 34, column: 21, scope: !120)
// CHECK:STDOUT: !127 = !DILocation(line: 35, column: 3, scope: !120)
// CHECK:STDOUT: !128 = !DILocation(line: 35, column: 21, scope: !120)
// CHECK:STDOUT: !130 = !DILocation(line: 36, column: 3, scope: !120)
// CHECK:STDOUT: !131 = !DILocation(line: 36, column: 21, scope: !120)
// CHECK:STDOUT: !133 = !DILocation(line: 38, column: 3, scope: !120)
// CHECK:STDOUT: !134 = !DILocation(line: 38, column: 19, scope: !120)
// CHECK:STDOUT: !136 = !DILocation(line: 39, column: 3, scope: !120)
// CHECK:STDOUT: !137 = !DILocation(line: 39, column: 21, scope: !120)
// CHECK:STDOUT: !139 = !DILocation(line: 40, column: 3, scope: !120)
// CHECK:STDOUT: !140 = !DILocation(line: 40, column: 21, scope: !120)
// CHECK:STDOUT: !142 = !DILocation(line: 41, column: 3, scope: !120)
// CHECK:STDOUT: !143 = !DILocation(line: 41, column: 21, scope: !120)
// CHECK:STDOUT: !145 = !DILocation(line: 43, column: 7, scope: !120)
// CHECK:STDOUT: !146 = !DILocation(line: 43, column: 14, scope: !120)
// CHECK:STDOUT: !147 = !DILocation(line: 43, column: 22, scope: !120)
// CHECK:STDOUT: !148 = !DILocation(line: 43, column: 30, scope: !120)
// CHECK:STDOUT: !149 = !DILocation(line: 43, column: 38, scope: !120)
// CHECK:STDOUT: !150 = !DILocation(line: 43, column: 45, scope: !120)
// CHECK:STDOUT: !151 = !DILocation(line: 43, column: 53, scope: !120)
// CHECK:STDOUT: !152 = !DILocation(line: 43, column: 61, scope: !120)
// CHECK:STDOUT: !153 = !DILocation(line: 43, column: 3, scope: !120)
// CHECK:STDOUT: !154 = !{null, !13}
// CHECK:STDOUT: !155 = !DISubroutineType(types: !154)
// CHECK:STDOUT: !156 = !{null, !12}
// CHECK:STDOUT: !157 = !DISubroutineType(types: !156)
// CHECK:STDOUT: !158 = !{null, !9}
// CHECK:STDOUT: !159 = !DISubroutineType(types: !158)
// CHECK:STDOUT: !160 = !{null, !8}
// CHECK:STDOUT: !161 = !DISubroutineType(types: !160)
// CHECK:STDOUT: !162 = !DILocation(line: 32, column: 1, scope: !120)
// CHECK:STDOUT: !163 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !164 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.e43967308dd20a9f", scope: null, file: !163, line: 35, type: !112, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !170)
// CHECK:STDOUT: !165 = !DILocalVariable(arg: 1, scope: !164, type: !11)
// CHECK:STDOUT: !169 = !DILocation(line: 35, column: 1, scope: !164)
// CHECK:STDOUT: !170 = !{!165}
// CHECK:STDOUT: !171 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.6fb12317fdf5c9b3", scope: null, file: !163, line: 35, type: !114, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !177)
// CHECK:STDOUT: !172 = !DILocalVariable(arg: 1, scope: !171, type: !10)
// CHECK:STDOUT: !176 = !DILocation(line: 35, column: 1, scope: !171)
// CHECK:STDOUT: !177 = !{!172}
// CHECK:STDOUT: !178 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.44e7310a0b394feb", scope: null, file: !163, line: 35, type: !116, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !184)
// CHECK:STDOUT: !179 = !DILocalVariable(arg: 1, scope: !178, type: !7)
// CHECK:STDOUT: !183 = !DILocation(line: 35, column: 1, scope: !178)
// CHECK:STDOUT: !184 = !{!179}
// CHECK:STDOUT: !185 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.73549b49ca05bf4c", scope: null, file: !163, line: 35, type: !118, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !191)
// CHECK:STDOUT: !186 = !DILocalVariable(arg: 1, scope: !185, type: !6)
// CHECK:STDOUT: !190 = !DILocation(line: 35, column: 1, scope: !185)
// CHECK:STDOUT: !191 = !{!186}
// CHECK:STDOUT: !192 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.7e9df21fc3951ad1", scope: null, file: !163, line: 35, type: !155, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !198)
// CHECK:STDOUT: !193 = !DILocalVariable(arg: 1, scope: !192, type: !13)
// CHECK:STDOUT: !197 = !DILocation(line: 35, column: 1, scope: !192)
// CHECK:STDOUT: !198 = !{!193}
// CHECK:STDOUT: !199 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.f127f30143696d51", scope: null, file: !163, line: 35, type: !157, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !205)
// CHECK:STDOUT: !200 = !DILocalVariable(arg: 1, scope: !199, type: !12)
// CHECK:STDOUT: !204 = !DILocation(line: 35, column: 1, scope: !199)
// CHECK:STDOUT: !205 = !{!200}
// CHECK:STDOUT: !206 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.3266e5c7a8b68ec1", scope: null, file: !163, line: 35, type: !159, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !212)
// CHECK:STDOUT: !207 = !DILocalVariable(arg: 1, scope: !206, type: !9)
// CHECK:STDOUT: !211 = !DILocation(line: 35, column: 1, scope: !206)
// CHECK:STDOUT: !212 = !{!207}
// CHECK:STDOUT: !213 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.ff339b9821763c59", scope: null, file: !163, line: 35, type: !161, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !219)
// CHECK:STDOUT: !214 = !DILocalVariable(arg: 1, scope: !213, type: !8)
// CHECK:STDOUT: !218 = !DILocation(line: 35, column: 1, scope: !213)
// CHECK:STDOUT: !219 = !{!214}
// CHECK:STDOUT: !220 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !221 = !{!"omnipotent char", !220, i64 0}
// CHECK:STDOUT: !222 = !{!"any pointer", !221, i64 0}
// CHECK:STDOUT: !223 = !{!"p1 omnipotent char", !222, i64 0}
// CHECK:STDOUT: !224 = !{!223, !223, i64 0}
// CHECK:STDOUT: !225 = !{!221, !221, i64 0}
// CHECK:STDOUT: !226 = !{!"short", !221, i64 0}
// CHECK:STDOUT: !227 = !{!"p1 short", !222, i64 0}
// CHECK:STDOUT: !228 = !{!227, !227, i64 0}
// CHECK:STDOUT: !229 = !{!226, !226, i64 0}
// CHECK:STDOUT: !230 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !231 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !232 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !233 = !{!"int", !221, i64 0}
// CHECK:STDOUT: !234 = !{!"__libc_errno", !233, i64 0}
// CHECK:STDOUT: !235 = !{!234, !233, i64 0}
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
// CHECK:STDOUT: define void @_CGetX.Main(ptr sret([16 x i8]) %return) #0 !dbg !240 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z4Makev.carbon_thunk.(ptr %return), !dbg !243
// CHECK:STDOUT:   ret void, !dbg !244
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z4Makev.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !263
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !263
// CHECK:STDOUT:   call void @_Z4Makev(ptr dead_on_unwind writable sret(%struct.X) align 8 %0)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CLet.Main() #0 !dbg !245 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_27.1.temp = alloca [16 x i8], align 8, !dbg !246
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc9_27.1.temp), !dbg !246
// CHECK:STDOUT:   call void @_Z4Makev.carbon_thunk.(ptr %.loc9_27.1.temp), !dbg !246
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.2d682b12c0c20e08(ptr %.loc9_27.1.temp), !dbg !246
// CHECK:STDOUT:   ret void, !dbg !249
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: declare void @_ZN1XD1Ev(ptr noundef nonnull align 8 dead_on_return(16) dereferenceable(16)) unnamed_addr #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CVar.Main() #0 !dbg !250 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca [16 x i8], align 8, !dbg !251
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !251
// CHECK:STDOUT:   call void @_Z4Makev.carbon_thunk.(ptr %_.var), !dbg !252
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.2d682b12c0c20e08(ptr %_.var), !dbg !251
// CHECK:STDOUT:   ret void, !dbg !254
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.2d682b12c0c20e08(ptr %t) #0 !dbg !255 {
// CHECK:STDOUT:   call void @_ZN1XD1Ev(ptr %t), !dbg !259
// CHECK:STDOUT:   ret void, !dbg !260
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z4Makev(ptr dead_on_unwind writable sret(%struct.X) align 8) #4
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.2d682b12c0c20e08, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #4 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!237}
// CHECK:STDOUT: !llvm.module.flags = !{!230, !231, !232, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!235}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !16 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !17 = !{null, !16}
// CHECK:STDOUT: !18 = !DISubroutineType(types: !17)
// CHECK:STDOUT: !35 = !{null}
// CHECK:STDOUT: !36 = !DISubroutineType(types: !35)
// CHECK:STDOUT: !163 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !220 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !221 = !{!"omnipotent char", !220, i64 0}
// CHECK:STDOUT: !222 = !{!"any pointer", !221, i64 0}
// CHECK:STDOUT: !230 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !231 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !232 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !233 = !{!"int", !221, i64 0}
// CHECK:STDOUT: !234 = !{!"__libc_errno", !233, i64 0}
// CHECK:STDOUT: !235 = !{!234, !233, i64 0}
// CHECK:STDOUT: !236 = !DIFile(filename: "import_class.carbon", directory: "")
// CHECK:STDOUT: !237 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !236, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !238 = !{!16}
// CHECK:STDOUT: !239 = !DISubroutineType(types: !238)
// CHECK:STDOUT: !240 = distinct !DISubprogram(name: "GetX", linkageName: "_CGetX.Main", scope: null, file: !236, line: 6, type: !239, spFlags: DISPFlagDefinition, unit: !237)
// CHECK:STDOUT: !243 = !DILocation(line: 6, column: 29, scope: !240)
// CHECK:STDOUT: !244 = !DILocation(line: 6, column: 22, scope: !240)
// CHECK:STDOUT: !245 = distinct !DISubprogram(name: "Let", linkageName: "_CLet.Main", scope: null, file: !236, line: 8, type: !36, spFlags: DISPFlagDefinition, unit: !237)
// CHECK:STDOUT: !246 = !DILocation(line: 9, column: 18, scope: !245)
// CHECK:STDOUT: !249 = !DILocation(line: 8, column: 1, scope: !245)
// CHECK:STDOUT: !250 = distinct !DISubprogram(name: "Var", linkageName: "_CVar.Main", scope: null, file: !236, line: 12, type: !36, spFlags: DISPFlagDefinition, unit: !237)
// CHECK:STDOUT: !251 = !DILocation(line: 13, column: 3, scope: !250)
// CHECK:STDOUT: !252 = !DILocation(line: 13, column: 18, scope: !250)
// CHECK:STDOUT: !254 = !DILocation(line: 12, column: 1, scope: !250)
// CHECK:STDOUT: !255 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.2d682b12c0c20e08", scope: null, file: !163, line: 35, type: !18, spFlags: DISPFlagDefinition, unit: !237, retainedNodes: !261)
// CHECK:STDOUT: !256 = !DILocalVariable(arg: 1, scope: !255, type: !16)
// CHECK:STDOUT: !259 = !DILocation(line: 37, column: 5, scope: !255)
// CHECK:STDOUT: !260 = !DILocation(line: 35, column: 1, scope: !255)
// CHECK:STDOUT: !261 = !{!256}
// CHECK:STDOUT: !262 = !{!"p1 _ZTS1X", !222, i64 0}
// CHECK:STDOUT: !263 = !{!262, !262, i64 0}
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
// CHECK:STDOUT: define void @_CCall.Main(ptr sret({}) %return, ptr %x) #0 !dbg !271 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z1f1D.carbon_thunk._(ptr %x, ptr %return), !dbg !277
// CHECK:STDOUT:   ret void, !dbg !278
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z1f1D.carbon_thunk._(ptr noundef %0, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %class.D, align 1
// CHECK:STDOUT:   %undef.agg.tmp = alloca %class.C, align 1
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !281
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !283
// CHECK:STDOUT:   %1 = load ptr, ptr %return.addr, align 8, !tbaa !283
// CHECK:STDOUT:   %2 = load ptr, ptr %.addr, align 8, !tbaa !281
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!266}
// CHECK:STDOUT: !llvm.module.flags = !{!230, !231, !232, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!235}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !16 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !220 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !221 = !{!"omnipotent char", !220, i64 0}
// CHECK:STDOUT: !222 = !{!"any pointer", !221, i64 0}
// CHECK:STDOUT: !230 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !231 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !232 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !233 = !{!"int", !221, i64 0}
// CHECK:STDOUT: !234 = !{!"__libc_errno", !233, i64 0}
// CHECK:STDOUT: !235 = !{!234, !233, i64 0}
// CHECK:STDOUT: !265 = !DIFile(filename: "indirect_return_with_args.carbon", directory: "")
// CHECK:STDOUT: !266 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !265, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !267 = !{!16, !16}
// CHECK:STDOUT: !268 = !DISubroutineType(types: !267)
// CHECK:STDOUT: !271 = distinct !DISubprogram(name: "Call", linkageName: "_CCall.Main", scope: null, file: !265, line: 10, type: !268, spFlags: DISPFlagDefinition, unit: !266, retainedNodes: !279)
// CHECK:STDOUT: !272 = !DILocalVariable(arg: 1, scope: !271, type: !16)
// CHECK:STDOUT: !277 = !DILocation(line: 11, column: 10, scope: !271)
// CHECK:STDOUT: !278 = !DILocation(line: 11, column: 3, scope: !271)
// CHECK:STDOUT: !279 = !{!272}
// CHECK:STDOUT: !280 = !{!"p1 _ZTS1D", !222, i64 0}
// CHECK:STDOUT: !281 = !{!280, !280, i64 0}
// CHECK:STDOUT: !282 = !{!"p1 _ZTS1C", !222, i64 0}
// CHECK:STDOUT: !283 = !{!282, !282, i64 0}
// CHECK:STDOUT:
