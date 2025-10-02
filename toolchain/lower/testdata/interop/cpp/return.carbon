// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
// EXTRA-ARGS: --target=x86_64-linux-gnu
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/return.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/return.carbon

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

  void *p, *q;
};

X Make();

// --- import_class.carbon

library "[[@TEST_NAME]]";

import Cpp library "class.h";

fn GetX() -> Cpp.X { return Cpp.Make(); }

fn Let() {
  let x: Cpp.X = Cpp.Make();
}

fn Var() {
  var x: Cpp.X = Cpp.Make();
}

// CHECK:STDOUT: ; ModuleID = 'import_ints.carbon'
// CHECK:STDOUT: source_filename = "import_ints.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: define i8 @_CGetU8.Main() !dbg !7 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc6_40.1.temp = alloca i8, align 1, !dbg !10
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc6_40.1.temp), !dbg !10
// CHECK:STDOUT:   call void @_Z8ReturnU8v.carbon_thunk(ptr %.loc6_40.1.temp), !dbg !10
// CHECK:STDOUT:   %.loc6_40.2 = load i8, ptr %.loc6_40.1.temp, align 1, !dbg !10
// CHECK:STDOUT:   ret i8 %.loc6_40.2, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i16 @_CGetU16.Main() !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_43.1.temp = alloca i16, align 2, !dbg !13
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_43.1.temp), !dbg !13
// CHECK:STDOUT:   call void @_Z9ReturnU16v.carbon_thunk(ptr %.loc7_43.1.temp), !dbg !13
// CHECK:STDOUT:   %.loc7_43.2 = load i16, ptr %.loc7_43.1.temp, align 2, !dbg !13
// CHECK:STDOUT:   ret i16 %.loc7_43.2, !dbg !14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_CGetU32.Main() !dbg !15 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ReturnU32.call = call i32 @_Z9ReturnU32v(), !dbg !16
// CHECK:STDOUT:   ret i32 %ReturnU32.call, !dbg !17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @_Z9ReturnU32v()
// CHECK:STDOUT:
// CHECK:STDOUT: define i64 @_CGetU64.Main() !dbg !18 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ReturnU64.call = call i64 @_Z9ReturnU64v(), !dbg !19
// CHECK:STDOUT:   ret i64 %ReturnU64.call, !dbg !20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i64 @_Z9ReturnU64v()
// CHECK:STDOUT:
// CHECK:STDOUT: define i8 @_CGetI8.Main() !dbg !21 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc11_40.1.temp = alloca i8, align 1, !dbg !22
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc11_40.1.temp), !dbg !22
// CHECK:STDOUT:   call void @_Z8ReturnI8v.carbon_thunk(ptr %.loc11_40.1.temp), !dbg !22
// CHECK:STDOUT:   %.loc11_40.2 = load i8, ptr %.loc11_40.1.temp, align 1, !dbg !22
// CHECK:STDOUT:   ret i8 %.loc11_40.2, !dbg !23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i16 @_CGetI16.Main() !dbg !24 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc12_43.1.temp = alloca i16, align 2, !dbg !25
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc12_43.1.temp), !dbg !25
// CHECK:STDOUT:   call void @_Z9ReturnI16v.carbon_thunk(ptr %.loc12_43.1.temp), !dbg !25
// CHECK:STDOUT:   %.loc12_43.2 = load i16, ptr %.loc12_43.1.temp, align 2, !dbg !25
// CHECK:STDOUT:   ret i16 %.loc12_43.2, !dbg !26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_CGetI32.Main() !dbg !27 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ReturnI32.call = call i32 @_Z9ReturnI32v(), !dbg !28
// CHECK:STDOUT:   ret i32 %ReturnI32.call, !dbg !29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @_Z9ReturnI32v()
// CHECK:STDOUT:
// CHECK:STDOUT: define i64 @_CGetI64.Main() !dbg !30 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ReturnI64.call = call i64 @_Z9ReturnI64v(), !dbg !31
// CHECK:STDOUT:   ret i64 %ReturnI64.call, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i64 @_Z9ReturnI64v()
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CUse.Main(i8, i16, i32, i64, i8, i16, i32, i64)
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CLets.Main() !dbg !33 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc19_32.1.temp = alloca i8, align 1, !dbg !34
// CHECK:STDOUT:   %.loc20_35.1.temp = alloca i16, align 2, !dbg !35
// CHECK:STDOUT:   %.loc24_32.1.temp = alloca i8, align 1, !dbg !36
// CHECK:STDOUT:   %.loc25_35.1.temp = alloca i16, align 2, !dbg !37
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc19_32.1.temp), !dbg !34
// CHECK:STDOUT:   call void @_Z8ReturnU8v.carbon_thunk(ptr %.loc19_32.1.temp), !dbg !34
// CHECK:STDOUT:   %.loc19_32.2 = load i8, ptr %.loc19_32.1.temp, align 1, !dbg !34
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc20_35.1.temp), !dbg !35
// CHECK:STDOUT:   call void @_Z9ReturnU16v.carbon_thunk(ptr %.loc20_35.1.temp), !dbg !35
// CHECK:STDOUT:   %.loc20_35.2 = load i16, ptr %.loc20_35.1.temp, align 2, !dbg !35
// CHECK:STDOUT:   %ReturnU32.call = call i32 @_Z9ReturnU32v(), !dbg !38
// CHECK:STDOUT:   %ReturnU64.call = call i64 @_Z9ReturnU64v(), !dbg !39
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc24_32.1.temp), !dbg !36
// CHECK:STDOUT:   call void @_Z8ReturnI8v.carbon_thunk(ptr %.loc24_32.1.temp), !dbg !36
// CHECK:STDOUT:   %.loc24_32.2 = load i8, ptr %.loc24_32.1.temp, align 1, !dbg !36
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc25_35.1.temp), !dbg !37
// CHECK:STDOUT:   call void @_Z9ReturnI16v.carbon_thunk(ptr %.loc25_35.1.temp), !dbg !37
// CHECK:STDOUT:   %.loc25_35.2 = load i16, ptr %.loc25_35.1.temp, align 2, !dbg !37
// CHECK:STDOUT:   %ReturnI32.call = call i32 @_Z9ReturnI32v(), !dbg !40
// CHECK:STDOUT:   %ReturnI64.call = call i64 @_Z9ReturnI64v(), !dbg !41
// CHECK:STDOUT:   call void @_CUse.Main(i8 %.loc19_32.2, i16 %.loc20_35.2, i32 %ReturnU32.call, i64 %ReturnU64.call, i8 %.loc24_32.2, i16 %.loc25_35.2, i32 %ReturnI32.call, i64 %ReturnI64.call), !dbg !42
// CHECK:STDOUT:   ret void, !dbg !43
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CVars.Main() !dbg !44 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %my_u8.var = alloca i8, align 1, !dbg !45
// CHECK:STDOUT:   %.loc33_32.1.temp = alloca i8, align 1, !dbg !46
// CHECK:STDOUT:   %my_u16.var = alloca i16, align 2, !dbg !47
// CHECK:STDOUT:   %.loc34_35.1.temp = alloca i16, align 2, !dbg !48
// CHECK:STDOUT:   %my_u32.var = alloca i32, align 4, !dbg !49
// CHECK:STDOUT:   %my_u64.var = alloca i64, align 8, !dbg !50
// CHECK:STDOUT:   %my_i8.var = alloca i8, align 1, !dbg !51
// CHECK:STDOUT:   %.loc38_32.1.temp = alloca i8, align 1, !dbg !52
// CHECK:STDOUT:   %my_i16.var = alloca i16, align 2, !dbg !53
// CHECK:STDOUT:   %.loc39_35.1.temp = alloca i16, align 2, !dbg !54
// CHECK:STDOUT:   %my_i32.var = alloca i32, align 4, !dbg !55
// CHECK:STDOUT:   %my_i64.var = alloca i64, align 8, !dbg !56
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %my_u8.var), !dbg !45
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc33_32.1.temp), !dbg !46
// CHECK:STDOUT:   call void @_Z8ReturnU8v.carbon_thunk(ptr %.loc33_32.1.temp), !dbg !46
// CHECK:STDOUT:   %.loc33_32.2 = load i8, ptr %.loc33_32.1.temp, align 1, !dbg !46
// CHECK:STDOUT:   store i8 %.loc33_32.2, ptr %my_u8.var, align 1, !dbg !45
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %my_u16.var), !dbg !47
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc34_35.1.temp), !dbg !48
// CHECK:STDOUT:   call void @_Z9ReturnU16v.carbon_thunk(ptr %.loc34_35.1.temp), !dbg !48
// CHECK:STDOUT:   %.loc34_35.2 = load i16, ptr %.loc34_35.1.temp, align 2, !dbg !48
// CHECK:STDOUT:   store i16 %.loc34_35.2, ptr %my_u16.var, align 2, !dbg !47
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %my_u32.var), !dbg !49
// CHECK:STDOUT:   %ReturnU32.call = call i32 @_Z9ReturnU32v(), !dbg !57
// CHECK:STDOUT:   store i32 %ReturnU32.call, ptr %my_u32.var, align 4, !dbg !49
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %my_u64.var), !dbg !50
// CHECK:STDOUT:   %ReturnU64.call = call i64 @_Z9ReturnU64v(), !dbg !58
// CHECK:STDOUT:   store i64 %ReturnU64.call, ptr %my_u64.var, align 4, !dbg !50
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %my_i8.var), !dbg !51
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc38_32.1.temp), !dbg !52
// CHECK:STDOUT:   call void @_Z8ReturnI8v.carbon_thunk(ptr %.loc38_32.1.temp), !dbg !52
// CHECK:STDOUT:   %.loc38_32.2 = load i8, ptr %.loc38_32.1.temp, align 1, !dbg !52
// CHECK:STDOUT:   store i8 %.loc38_32.2, ptr %my_i8.var, align 1, !dbg !51
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %my_i16.var), !dbg !53
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc39_35.1.temp), !dbg !54
// CHECK:STDOUT:   call void @_Z9ReturnI16v.carbon_thunk(ptr %.loc39_35.1.temp), !dbg !54
// CHECK:STDOUT:   %.loc39_35.2 = load i16, ptr %.loc39_35.1.temp, align 2, !dbg !54
// CHECK:STDOUT:   store i16 %.loc39_35.2, ptr %my_i16.var, align 2, !dbg !53
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %my_i32.var), !dbg !55
// CHECK:STDOUT:   %ReturnI32.call = call i32 @_Z9ReturnI32v(), !dbg !59
// CHECK:STDOUT:   store i32 %ReturnI32.call, ptr %my_i32.var, align 4, !dbg !55
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %my_i64.var), !dbg !56
// CHECK:STDOUT:   %ReturnI64.call = call i64 @_Z9ReturnI64v(), !dbg !60
// CHECK:STDOUT:   store i64 %ReturnI64.call, ptr %my_i64.var, align 4, !dbg !56
// CHECK:STDOUT:   %.loc43_7 = load i8, ptr %my_u8.var, align 1, !dbg !61
// CHECK:STDOUT:   %.loc43_14 = load i16, ptr %my_u16.var, align 2, !dbg !62
// CHECK:STDOUT:   %.loc43_22 = load i32, ptr %my_u32.var, align 4, !dbg !63
// CHECK:STDOUT:   %.loc43_30 = load i64, ptr %my_u64.var, align 4, !dbg !64
// CHECK:STDOUT:   %.loc43_38 = load i8, ptr %my_i8.var, align 1, !dbg !65
// CHECK:STDOUT:   %.loc43_45 = load i16, ptr %my_i16.var, align 2, !dbg !66
// CHECK:STDOUT:   %.loc43_53 = load i32, ptr %my_i32.var, align 4, !dbg !67
// CHECK:STDOUT:   %.loc43_61 = load i64, ptr %my_i64.var, align 4, !dbg !68
// CHECK:STDOUT:   call void @_CUse.Main(i8 %.loc43_7, i16 %.loc43_14, i32 %.loc43_22, i64 %.loc43_30, i8 %.loc43_38, i16 %.loc43_45, i32 %.loc43_53, i64 %.loc43_61), !dbg !69
// CHECK:STDOUT:   ret void, !dbg !70
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #0
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress
// CHECK:STDOUT: define dso_local void @_Z8ReturnU8v.carbon_thunk(ptr %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8
// CHECK:STDOUT:   %call = call zeroext i8 @_Z8ReturnU8v()
// CHECK:STDOUT:   store i8 %call, ptr %0, align 1
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare zeroext i8 @_Z8ReturnU8v() #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress
// CHECK:STDOUT: define dso_local void @_Z9ReturnU16v.carbon_thunk(ptr %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8
// CHECK:STDOUT:   %call = call zeroext i16 @_Z9ReturnU16v()
// CHECK:STDOUT:   store i16 %call, ptr %0, align 2
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare zeroext i16 @_Z9ReturnU16v() #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress
// CHECK:STDOUT: define dso_local void @_Z8ReturnI8v.carbon_thunk(ptr %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8
// CHECK:STDOUT:   %call = call signext i8 @_Z8ReturnI8v()
// CHECK:STDOUT:   store i8 %call, ptr %0, align 1
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare signext i8 @_Z8ReturnI8v() #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress
// CHECK:STDOUT: define dso_local void @_Z9ReturnI16v.carbon_thunk(ptr %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8
// CHECK:STDOUT:   %call = call signext i16 @_Z9ReturnI16v()
// CHECK:STDOUT:   store i16 %call, ptr %0, align 2
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare signext i16 @_Z9ReturnI16v() #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="0" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { "no-trapping-math"="true" "stack-protector-buffer-size"="0" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1, !2, !3, !4}
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = !{i32 1, !"wchar_size", i32 4}
// CHECK:STDOUT: !3 = !{i32 8, !"PIC Level", i32 0}
// CHECK:STDOUT: !4 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C, file: !6, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIFile(filename: "import_ints.carbon", directory: "")
// CHECK:STDOUT: !7 = distinct !DISubprogram(name: "GetU8", linkageName: "_CGetU8.Main", scope: null, file: !6, line: 6, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !8 = !DISubroutineType(types: !9)
// CHECK:STDOUT: !9 = !{}
// CHECK:STDOUT: !10 = !DILocation(line: 6, column: 27, scope: !7)
// CHECK:STDOUT: !11 = !DILocation(line: 6, column: 20, scope: !7)
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "GetU16", linkageName: "_CGetU16.Main", scope: null, file: !6, line: 7, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !13 = !DILocation(line: 7, column: 29, scope: !12)
// CHECK:STDOUT: !14 = !DILocation(line: 7, column: 22, scope: !12)
// CHECK:STDOUT: !15 = distinct !DISubprogram(name: "GetU32", linkageName: "_CGetU32.Main", scope: null, file: !6, line: 8, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !16 = !DILocation(line: 8, column: 29, scope: !15)
// CHECK:STDOUT: !17 = !DILocation(line: 8, column: 22, scope: !15)
// CHECK:STDOUT: !18 = distinct !DISubprogram(name: "GetU64", linkageName: "_CGetU64.Main", scope: null, file: !6, line: 9, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !19 = !DILocation(line: 9, column: 29, scope: !18)
// CHECK:STDOUT: !20 = !DILocation(line: 9, column: 22, scope: !18)
// CHECK:STDOUT: !21 = distinct !DISubprogram(name: "GetI8", linkageName: "_CGetI8.Main", scope: null, file: !6, line: 11, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !22 = !DILocation(line: 11, column: 27, scope: !21)
// CHECK:STDOUT: !23 = !DILocation(line: 11, column: 20, scope: !21)
// CHECK:STDOUT: !24 = distinct !DISubprogram(name: "GetI16", linkageName: "_CGetI16.Main", scope: null, file: !6, line: 12, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !25 = !DILocation(line: 12, column: 29, scope: !24)
// CHECK:STDOUT: !26 = !DILocation(line: 12, column: 22, scope: !24)
// CHECK:STDOUT: !27 = distinct !DISubprogram(name: "GetI32", linkageName: "_CGetI32.Main", scope: null, file: !6, line: 13, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !28 = !DILocation(line: 13, column: 29, scope: !27)
// CHECK:STDOUT: !29 = !DILocation(line: 13, column: 22, scope: !27)
// CHECK:STDOUT: !30 = distinct !DISubprogram(name: "GetI64", linkageName: "_CGetI64.Main", scope: null, file: !6, line: 14, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !31 = !DILocation(line: 14, column: 29, scope: !30)
// CHECK:STDOUT: !32 = !DILocation(line: 14, column: 22, scope: !30)
// CHECK:STDOUT: !33 = distinct !DISubprogram(name: "Lets", linkageName: "_CLets.Main", scope: null, file: !6, line: 18, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !34 = !DILocation(line: 19, column: 19, scope: !33)
// CHECK:STDOUT: !35 = !DILocation(line: 20, column: 21, scope: !33)
// CHECK:STDOUT: !36 = !DILocation(line: 24, column: 19, scope: !33)
// CHECK:STDOUT: !37 = !DILocation(line: 25, column: 21, scope: !33)
// CHECK:STDOUT: !38 = !DILocation(line: 21, column: 21, scope: !33)
// CHECK:STDOUT: !39 = !DILocation(line: 22, column: 21, scope: !33)
// CHECK:STDOUT: !40 = !DILocation(line: 26, column: 21, scope: !33)
// CHECK:STDOUT: !41 = !DILocation(line: 27, column: 21, scope: !33)
// CHECK:STDOUT: !42 = !DILocation(line: 29, column: 3, scope: !33)
// CHECK:STDOUT: !43 = !DILocation(line: 18, column: 1, scope: !33)
// CHECK:STDOUT: !44 = distinct !DISubprogram(name: "Vars", linkageName: "_CVars.Main", scope: null, file: !6, line: 32, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !45 = !DILocation(line: 33, column: 3, scope: !44)
// CHECK:STDOUT: !46 = !DILocation(line: 33, column: 19, scope: !44)
// CHECK:STDOUT: !47 = !DILocation(line: 34, column: 3, scope: !44)
// CHECK:STDOUT: !48 = !DILocation(line: 34, column: 21, scope: !44)
// CHECK:STDOUT: !49 = !DILocation(line: 35, column: 3, scope: !44)
// CHECK:STDOUT: !50 = !DILocation(line: 36, column: 3, scope: !44)
// CHECK:STDOUT: !51 = !DILocation(line: 38, column: 3, scope: !44)
// CHECK:STDOUT: !52 = !DILocation(line: 38, column: 19, scope: !44)
// CHECK:STDOUT: !53 = !DILocation(line: 39, column: 3, scope: !44)
// CHECK:STDOUT: !54 = !DILocation(line: 39, column: 21, scope: !44)
// CHECK:STDOUT: !55 = !DILocation(line: 40, column: 3, scope: !44)
// CHECK:STDOUT: !56 = !DILocation(line: 41, column: 3, scope: !44)
// CHECK:STDOUT: !57 = !DILocation(line: 35, column: 21, scope: !44)
// CHECK:STDOUT: !58 = !DILocation(line: 36, column: 21, scope: !44)
// CHECK:STDOUT: !59 = !DILocation(line: 40, column: 21, scope: !44)
// CHECK:STDOUT: !60 = !DILocation(line: 41, column: 21, scope: !44)
// CHECK:STDOUT: !61 = !DILocation(line: 43, column: 7, scope: !44)
// CHECK:STDOUT: !62 = !DILocation(line: 43, column: 14, scope: !44)
// CHECK:STDOUT: !63 = !DILocation(line: 43, column: 22, scope: !44)
// CHECK:STDOUT: !64 = !DILocation(line: 43, column: 30, scope: !44)
// CHECK:STDOUT: !65 = !DILocation(line: 43, column: 38, scope: !44)
// CHECK:STDOUT: !66 = !DILocation(line: 43, column: 45, scope: !44)
// CHECK:STDOUT: !67 = !DILocation(line: 43, column: 53, scope: !44)
// CHECK:STDOUT: !68 = !DILocation(line: 43, column: 61, scope: !44)
// CHECK:STDOUT: !69 = !DILocation(line: 43, column: 3, scope: !44)
// CHECK:STDOUT: !70 = !DILocation(line: 32, column: 1, scope: !44)
// CHECK:STDOUT: ; ModuleID = 'import_class.carbon'
// CHECK:STDOUT: source_filename = "import_class.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %struct.X = type { ptr, ptr }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CGetX.Main(ptr sret([16 x i8]) %return) !dbg !7 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z4Makev.carbon_thunk(ptr %return), !dbg !10
// CHECK:STDOUT:   ret void, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CLet.Main() !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_27.1.temp = alloca [16 x i8], align 1, !dbg !13
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc9_27.1.temp), !dbg !13
// CHECK:STDOUT:   call void @_Z4Makev.carbon_thunk(ptr %.loc9_27.1.temp), !dbg !13
// CHECK:STDOUT:   ret void, !dbg !14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CVar.Main() !dbg !15 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x.var = alloca [16 x i8], align 1, !dbg !16
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %x.var), !dbg !16
// CHECK:STDOUT:   call void @_Z4Makev.carbon_thunk(ptr %x.var), !dbg !17
// CHECK:STDOUT:   ret void, !dbg !18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #0
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress
// CHECK:STDOUT: define dso_local void @_Z4Makev.carbon_thunk(ptr %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8
// CHECK:STDOUT:   call void @_Z4Makev(ptr dead_on_unwind writable sret(%struct.X) align 8 %0)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z4Makev(ptr dead_on_unwind writable sret(%struct.X) align 8) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="0" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { "no-trapping-math"="true" "stack-protector-buffer-size"="0" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1, !2, !3, !4}
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = !{i32 1, !"wchar_size", i32 4}
// CHECK:STDOUT: !3 = !{i32 8, !"PIC Level", i32 0}
// CHECK:STDOUT: !4 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C, file: !6, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIFile(filename: "import_class.carbon", directory: "")
// CHECK:STDOUT: !7 = distinct !DISubprogram(name: "GetX", linkageName: "_CGetX.Main", scope: null, file: !6, line: 6, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !8 = !DISubroutineType(types: !9)
// CHECK:STDOUT: !9 = !{}
// CHECK:STDOUT: !10 = !DILocation(line: 6, column: 29, scope: !7)
// CHECK:STDOUT: !11 = !DILocation(line: 6, column: 22, scope: !7)
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "Let", linkageName: "_CLet.Main", scope: null, file: !6, line: 8, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !13 = !DILocation(line: 9, column: 18, scope: !12)
// CHECK:STDOUT: !14 = !DILocation(line: 8, column: 1, scope: !12)
// CHECK:STDOUT: !15 = distinct !DISubprogram(name: "Var", linkageName: "_CVar.Main", scope: null, file: !6, line: 12, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !16 = !DILocation(line: 13, column: 3, scope: !15)
// CHECK:STDOUT: !17 = !DILocation(line: 13, column: 18, scope: !15)
// CHECK:STDOUT: !18 = !DILocation(line: 12, column: 1, scope: !15)
