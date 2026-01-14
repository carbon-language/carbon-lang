// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
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
// CHECK:STDOUT: define i8 @_CGetU8.Main() #0 !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc6_40.1.temp = alloca i8, align 1, !dbg !16
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc6_40.1.temp), !dbg !16
// CHECK:STDOUT:   call void @_Z8ReturnU8v.carbon_thunk(ptr %.loc6_40.1.temp), !dbg !16
// CHECK:STDOUT:   %.loc6_40.2 = load i8, ptr %.loc6_40.1.temp, align 1, !dbg !16
// CHECK:STDOUT:   ret i8 %.loc6_40.2, !dbg !17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i16 @_CGetU16.Main() #0 !dbg !18 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_43.1.temp = alloca i16, align 2, !dbg !22
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_43.1.temp), !dbg !22
// CHECK:STDOUT:   call void @_Z9ReturnU16v.carbon_thunk(ptr %.loc7_43.1.temp), !dbg !22
// CHECK:STDOUT:   %.loc7_43.2 = load i16, ptr %.loc7_43.1.temp, align 2, !dbg !22
// CHECK:STDOUT:   ret i16 %.loc7_43.2, !dbg !23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CGetU32.Main() #0 !dbg !24 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ReturnU32.call = call i32 @_Z9ReturnU32v(), !dbg !28
// CHECK:STDOUT:   ret i32 %ReturnU32.call, !dbg !29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @_Z9ReturnU32v()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CGetU64.Main() #0 !dbg !30 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ReturnU64.call = call i64 @_Z9ReturnU64v(), !dbg !34
// CHECK:STDOUT:   ret i64 %ReturnU64.call, !dbg !35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i64 @_Z9ReturnU64v()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i8 @_CGetI8.Main() #0 !dbg !36 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc11_40.1.temp = alloca i8, align 1, !dbg !40
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc11_40.1.temp), !dbg !40
// CHECK:STDOUT:   call void @_Z8ReturnI8v.carbon_thunk(ptr %.loc11_40.1.temp), !dbg !40
// CHECK:STDOUT:   %.loc11_40.2 = load i8, ptr %.loc11_40.1.temp, align 1, !dbg !40
// CHECK:STDOUT:   ret i8 %.loc11_40.2, !dbg !41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i16 @_CGetI16.Main() #0 !dbg !42 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc12_43.1.temp = alloca i16, align 2, !dbg !46
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc12_43.1.temp), !dbg !46
// CHECK:STDOUT:   call void @_Z9ReturnI16v.carbon_thunk(ptr %.loc12_43.1.temp), !dbg !46
// CHECK:STDOUT:   %.loc12_43.2 = load i16, ptr %.loc12_43.1.temp, align 2, !dbg !46
// CHECK:STDOUT:   ret i16 %.loc12_43.2, !dbg !47
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CGetI32.Main() #0 !dbg !48 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ReturnI32.call = call i32 @_Z9ReturnI32v(), !dbg !52
// CHECK:STDOUT:   ret i32 %ReturnI32.call, !dbg !53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @_Z9ReturnI32v()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CGetI64.Main() #0 !dbg !54 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ReturnI64.call = call i64 @_Z9ReturnI64v(), !dbg !58
// CHECK:STDOUT:   ret i64 %ReturnI64.call, !dbg !59
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i64 @_Z9ReturnI64v()
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CUse.Main(i8, i16, i32, i64, i8, i16, i32, i64)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CLets.Main() #0 !dbg !60 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc19_32.1.temp = alloca i8, align 1, !dbg !63
// CHECK:STDOUT:   %.loc20_35.1.temp = alloca i16, align 2, !dbg !64
// CHECK:STDOUT:   %.loc24_32.1.temp = alloca i8, align 1, !dbg !65
// CHECK:STDOUT:   %.loc25_35.1.temp = alloca i16, align 2, !dbg !66
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc19_32.1.temp), !dbg !63
// CHECK:STDOUT:   call void @_Z8ReturnU8v.carbon_thunk(ptr %.loc19_32.1.temp), !dbg !63
// CHECK:STDOUT:   %.loc19_32.2 = load i8, ptr %.loc19_32.1.temp, align 1, !dbg !63
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc20_35.1.temp), !dbg !64
// CHECK:STDOUT:   call void @_Z9ReturnU16v.carbon_thunk(ptr %.loc20_35.1.temp), !dbg !64
// CHECK:STDOUT:   %.loc20_35.2 = load i16, ptr %.loc20_35.1.temp, align 2, !dbg !64
// CHECK:STDOUT:   %ReturnU32.call = call i32 @_Z9ReturnU32v(), !dbg !67
// CHECK:STDOUT:   %ReturnU64.call = call i64 @_Z9ReturnU64v(), !dbg !68
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc24_32.1.temp), !dbg !65
// CHECK:STDOUT:   call void @_Z8ReturnI8v.carbon_thunk(ptr %.loc24_32.1.temp), !dbg !65
// CHECK:STDOUT:   %.loc24_32.2 = load i8, ptr %.loc24_32.1.temp, align 1, !dbg !65
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc25_35.1.temp), !dbg !66
// CHECK:STDOUT:   call void @_Z9ReturnI16v.carbon_thunk(ptr %.loc25_35.1.temp), !dbg !66
// CHECK:STDOUT:   %.loc25_35.2 = load i16, ptr %.loc25_35.1.temp, align 2, !dbg !66
// CHECK:STDOUT:   %ReturnI32.call = call i32 @_Z9ReturnI32v(), !dbg !69
// CHECK:STDOUT:   %ReturnI64.call = call i64 @_Z9ReturnI64v(), !dbg !70
// CHECK:STDOUT:   call void @_CUse.Main(i8 %.loc19_32.2, i16 %.loc20_35.2, i32 %ReturnU32.call, i64 %ReturnU64.call, i8 %.loc24_32.2, i16 %.loc25_35.2, i32 %ReturnI32.call, i64 %ReturnI64.call), !dbg !71
// CHECK:STDOUT:   ret void, !dbg !72
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CVars.Main() #0 !dbg !73 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %my_u8.var = alloca i8, align 1, !dbg !74
// CHECK:STDOUT:   %.loc33_32.1.temp = alloca i8, align 1, !dbg !75
// CHECK:STDOUT:   %my_u16.var = alloca i16, align 2, !dbg !76
// CHECK:STDOUT:   %.loc34_35.1.temp = alloca i16, align 2, !dbg !77
// CHECK:STDOUT:   %my_u32.var = alloca i32, align 4, !dbg !78
// CHECK:STDOUT:   %my_u64.var = alloca i64, align 8, !dbg !79
// CHECK:STDOUT:   %my_i8.var = alloca i8, align 1, !dbg !80
// CHECK:STDOUT:   %.loc38_32.1.temp = alloca i8, align 1, !dbg !81
// CHECK:STDOUT:   %my_i16.var = alloca i16, align 2, !dbg !82
// CHECK:STDOUT:   %.loc39_35.1.temp = alloca i16, align 2, !dbg !83
// CHECK:STDOUT:   %my_i32.var = alloca i32, align 4, !dbg !84
// CHECK:STDOUT:   %my_i64.var = alloca i64, align 8, !dbg !85
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %my_u8.var), !dbg !74
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc33_32.1.temp), !dbg !75
// CHECK:STDOUT:   call void @_Z8ReturnU8v.carbon_thunk(ptr %.loc33_32.1.temp), !dbg !75
// CHECK:STDOUT:   %.loc33_32.2 = load i8, ptr %.loc33_32.1.temp, align 1, !dbg !75
// CHECK:STDOUT:   store i8 %.loc33_32.2, ptr %my_u8.var, align 1, !dbg !74
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %my_u16.var), !dbg !76
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc34_35.1.temp), !dbg !77
// CHECK:STDOUT:   call void @_Z9ReturnU16v.carbon_thunk(ptr %.loc34_35.1.temp), !dbg !77
// CHECK:STDOUT:   %.loc34_35.2 = load i16, ptr %.loc34_35.1.temp, align 2, !dbg !77
// CHECK:STDOUT:   store i16 %.loc34_35.2, ptr %my_u16.var, align 2, !dbg !76
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %my_u32.var), !dbg !78
// CHECK:STDOUT:   %ReturnU32.call = call i32 @_Z9ReturnU32v(), !dbg !86
// CHECK:STDOUT:   store i32 %ReturnU32.call, ptr %my_u32.var, align 4, !dbg !78
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %my_u64.var), !dbg !79
// CHECK:STDOUT:   %ReturnU64.call = call i64 @_Z9ReturnU64v(), !dbg !87
// CHECK:STDOUT:   store i64 %ReturnU64.call, ptr %my_u64.var, align 4, !dbg !79
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %my_i8.var), !dbg !80
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc38_32.1.temp), !dbg !81
// CHECK:STDOUT:   call void @_Z8ReturnI8v.carbon_thunk(ptr %.loc38_32.1.temp), !dbg !81
// CHECK:STDOUT:   %.loc38_32.2 = load i8, ptr %.loc38_32.1.temp, align 1, !dbg !81
// CHECK:STDOUT:   store i8 %.loc38_32.2, ptr %my_i8.var, align 1, !dbg !80
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %my_i16.var), !dbg !82
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc39_35.1.temp), !dbg !83
// CHECK:STDOUT:   call void @_Z9ReturnI16v.carbon_thunk(ptr %.loc39_35.1.temp), !dbg !83
// CHECK:STDOUT:   %.loc39_35.2 = load i16, ptr %.loc39_35.1.temp, align 2, !dbg !83
// CHECK:STDOUT:   store i16 %.loc39_35.2, ptr %my_i16.var, align 2, !dbg !82
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %my_i32.var), !dbg !84
// CHECK:STDOUT:   %ReturnI32.call = call i32 @_Z9ReturnI32v(), !dbg !88
// CHECK:STDOUT:   store i32 %ReturnI32.call, ptr %my_i32.var, align 4, !dbg !84
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %my_i64.var), !dbg !85
// CHECK:STDOUT:   %ReturnI64.call = call i64 @_Z9ReturnI64v(), !dbg !89
// CHECK:STDOUT:   store i64 %ReturnI64.call, ptr %my_i64.var, align 4, !dbg !85
// CHECK:STDOUT:   %.loc43_7 = load i8, ptr %my_u8.var, align 1, !dbg !90
// CHECK:STDOUT:   %.loc43_14 = load i16, ptr %my_u16.var, align 2, !dbg !91
// CHECK:STDOUT:   %.loc43_22 = load i32, ptr %my_u32.var, align 4, !dbg !92
// CHECK:STDOUT:   %.loc43_30 = load i64, ptr %my_u64.var, align 4, !dbg !93
// CHECK:STDOUT:   %.loc43_38 = load i8, ptr %my_i8.var, align 1, !dbg !94
// CHECK:STDOUT:   %.loc43_45 = load i16, ptr %my_i16.var, align 2, !dbg !95
// CHECK:STDOUT:   %.loc43_53 = load i32, ptr %my_i32.var, align 4, !dbg !96
// CHECK:STDOUT:   %.loc43_61 = load i64, ptr %my_i64.var, align 4, !dbg !97
// CHECK:STDOUT:   call void @_CUse.Main(i8 %.loc43_7, i16 %.loc43_14, i32 %.loc43_22, i64 %.loc43_30, i8 %.loc43_38, i16 %.loc43_45, i32 %.loc43_53, i64 %.loc43_61), !dbg !98
// CHECK:STDOUT:   ret void, !dbg !99
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define dso_local void @_Z8ReturnU8v.carbon_thunk(ptr noundef %return) #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !100
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !100
// CHECK:STDOUT:   %call = call noundef zeroext i8 @_Z8ReturnU8v()
// CHECK:STDOUT:   store i8 %call, ptr %0, align 1, !tbaa !103
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef zeroext i8 @_Z8ReturnU8v() #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define dso_local void @_Z9ReturnU16v.carbon_thunk(ptr noundef %return) #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !104
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !104
// CHECK:STDOUT:   %call = call noundef zeroext i16 @_Z9ReturnU16v()
// CHECK:STDOUT:   store i16 %call, ptr %0, align 2, !tbaa !106
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef zeroext i16 @_Z9ReturnU16v() #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define dso_local void @_Z8ReturnI8v.carbon_thunk(ptr noundef %return) #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !100
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !100
// CHECK:STDOUT:   %call = call noundef signext i8 @_Z8ReturnI8v()
// CHECK:STDOUT:   store i8 %call, ptr %0, align 1, !tbaa !103
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef signext i8 @_Z8ReturnI8v() #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define dso_local void @_Z9ReturnI16v.carbon_thunk(ptr noundef %return) #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !104
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !104
// CHECK:STDOUT:   %call = call noundef signext i16 @_Z9ReturnI16v()
// CHECK:STDOUT:   store i16 %call, ptr %0, align 2, !tbaa !106
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef signext i16 @_Z9ReturnI16v() #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #2 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #3 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1, !2, !3, !4, !5}
// CHECK:STDOUT: !llvm.dbg.cu = !{!6}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!8}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = !{i32 1, !"wchar_size", i32 4}
// CHECK:STDOUT: !3 = !{i32 8, !"PIC Level", i32 0}
// CHECK:STDOUT: !4 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !5 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !6 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !7, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !7 = !DIFile(filename: "import_ints.carbon", directory: "")
// CHECK:STDOUT: !8 = !{!9, !9, i64 0}
// CHECK:STDOUT: !9 = !{!"int", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"omnipotent char", !11, i64 0}
// CHECK:STDOUT: !11 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "GetU8", linkageName: "_CGetU8.Main", scope: null, file: !7, line: 6, type: !13, spFlags: DISPFlagDefinition, unit: !6)
// CHECK:STDOUT: !13 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !14 = !{!15}
// CHECK:STDOUT: !15 = !DIBasicType(name: "int", size: 8, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !16 = !DILocation(line: 6, column: 27, scope: !12)
// CHECK:STDOUT: !17 = !DILocation(line: 6, column: 20, scope: !12)
// CHECK:STDOUT: !18 = distinct !DISubprogram(name: "GetU16", linkageName: "_CGetU16.Main", scope: null, file: !7, line: 7, type: !19, spFlags: DISPFlagDefinition, unit: !6)
// CHECK:STDOUT: !19 = !DISubroutineType(types: !20)
// CHECK:STDOUT: !20 = !{!21}
// CHECK:STDOUT: !21 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !22 = !DILocation(line: 7, column: 29, scope: !18)
// CHECK:STDOUT: !23 = !DILocation(line: 7, column: 22, scope: !18)
// CHECK:STDOUT: !24 = distinct !DISubprogram(name: "GetU32", linkageName: "_CGetU32.Main", scope: null, file: !7, line: 8, type: !25, spFlags: DISPFlagDefinition, unit: !6)
// CHECK:STDOUT: !25 = !DISubroutineType(types: !26)
// CHECK:STDOUT: !26 = !{!27}
// CHECK:STDOUT: !27 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !28 = !DILocation(line: 8, column: 29, scope: !24)
// CHECK:STDOUT: !29 = !DILocation(line: 8, column: 22, scope: !24)
// CHECK:STDOUT: !30 = distinct !DISubprogram(name: "GetU64", linkageName: "_CGetU64.Main", scope: null, file: !7, line: 9, type: !31, spFlags: DISPFlagDefinition, unit: !6)
// CHECK:STDOUT: !31 = !DISubroutineType(types: !32)
// CHECK:STDOUT: !32 = !{!33}
// CHECK:STDOUT: !33 = !DIBasicType(name: "int", size: 64, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !34 = !DILocation(line: 9, column: 29, scope: !30)
// CHECK:STDOUT: !35 = !DILocation(line: 9, column: 22, scope: !30)
// CHECK:STDOUT: !36 = distinct !DISubprogram(name: "GetI8", linkageName: "_CGetI8.Main", scope: null, file: !7, line: 11, type: !37, spFlags: DISPFlagDefinition, unit: !6)
// CHECK:STDOUT: !37 = !DISubroutineType(types: !38)
// CHECK:STDOUT: !38 = !{!39}
// CHECK:STDOUT: !39 = !DIBasicType(name: "int", size: 8, encoding: DW_ATE_signed)
// CHECK:STDOUT: !40 = !DILocation(line: 11, column: 27, scope: !36)
// CHECK:STDOUT: !41 = !DILocation(line: 11, column: 20, scope: !36)
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "GetI16", linkageName: "_CGetI16.Main", scope: null, file: !7, line: 12, type: !43, spFlags: DISPFlagDefinition, unit: !6)
// CHECK:STDOUT: !43 = !DISubroutineType(types: !44)
// CHECK:STDOUT: !44 = !{!45}
// CHECK:STDOUT: !45 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_signed)
// CHECK:STDOUT: !46 = !DILocation(line: 12, column: 29, scope: !42)
// CHECK:STDOUT: !47 = !DILocation(line: 12, column: 22, scope: !42)
// CHECK:STDOUT: !48 = distinct !DISubprogram(name: "GetI32", linkageName: "_CGetI32.Main", scope: null, file: !7, line: 13, type: !49, spFlags: DISPFlagDefinition, unit: !6)
// CHECK:STDOUT: !49 = !DISubroutineType(types: !50)
// CHECK:STDOUT: !50 = !{!51}
// CHECK:STDOUT: !51 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !52 = !DILocation(line: 13, column: 29, scope: !48)
// CHECK:STDOUT: !53 = !DILocation(line: 13, column: 22, scope: !48)
// CHECK:STDOUT: !54 = distinct !DISubprogram(name: "GetI64", linkageName: "_CGetI64.Main", scope: null, file: !7, line: 14, type: !55, spFlags: DISPFlagDefinition, unit: !6)
// CHECK:STDOUT: !55 = !DISubroutineType(types: !56)
// CHECK:STDOUT: !56 = !{!57}
// CHECK:STDOUT: !57 = !DIBasicType(name: "int", size: 64, encoding: DW_ATE_signed)
// CHECK:STDOUT: !58 = !DILocation(line: 14, column: 29, scope: !54)
// CHECK:STDOUT: !59 = !DILocation(line: 14, column: 22, scope: !54)
// CHECK:STDOUT: !60 = distinct !DISubprogram(name: "Lets", linkageName: "_CLets.Main", scope: null, file: !7, line: 18, type: !61, spFlags: DISPFlagDefinition, unit: !6)
// CHECK:STDOUT: !61 = !DISubroutineType(types: !62)
// CHECK:STDOUT: !62 = !{null}
// CHECK:STDOUT: !63 = !DILocation(line: 19, column: 19, scope: !60)
// CHECK:STDOUT: !64 = !DILocation(line: 20, column: 21, scope: !60)
// CHECK:STDOUT: !65 = !DILocation(line: 24, column: 19, scope: !60)
// CHECK:STDOUT: !66 = !DILocation(line: 25, column: 21, scope: !60)
// CHECK:STDOUT: !67 = !DILocation(line: 21, column: 21, scope: !60)
// CHECK:STDOUT: !68 = !DILocation(line: 22, column: 21, scope: !60)
// CHECK:STDOUT: !69 = !DILocation(line: 26, column: 21, scope: !60)
// CHECK:STDOUT: !70 = !DILocation(line: 27, column: 21, scope: !60)
// CHECK:STDOUT: !71 = !DILocation(line: 29, column: 3, scope: !60)
// CHECK:STDOUT: !72 = !DILocation(line: 18, column: 1, scope: !60)
// CHECK:STDOUT: !73 = distinct !DISubprogram(name: "Vars", linkageName: "_CVars.Main", scope: null, file: !7, line: 32, type: !61, spFlags: DISPFlagDefinition, unit: !6)
// CHECK:STDOUT: !74 = !DILocation(line: 33, column: 3, scope: !73)
// CHECK:STDOUT: !75 = !DILocation(line: 33, column: 19, scope: !73)
// CHECK:STDOUT: !76 = !DILocation(line: 34, column: 3, scope: !73)
// CHECK:STDOUT: !77 = !DILocation(line: 34, column: 21, scope: !73)
// CHECK:STDOUT: !78 = !DILocation(line: 35, column: 3, scope: !73)
// CHECK:STDOUT: !79 = !DILocation(line: 36, column: 3, scope: !73)
// CHECK:STDOUT: !80 = !DILocation(line: 38, column: 3, scope: !73)
// CHECK:STDOUT: !81 = !DILocation(line: 38, column: 19, scope: !73)
// CHECK:STDOUT: !82 = !DILocation(line: 39, column: 3, scope: !73)
// CHECK:STDOUT: !83 = !DILocation(line: 39, column: 21, scope: !73)
// CHECK:STDOUT: !84 = !DILocation(line: 40, column: 3, scope: !73)
// CHECK:STDOUT: !85 = !DILocation(line: 41, column: 3, scope: !73)
// CHECK:STDOUT: !86 = !DILocation(line: 35, column: 21, scope: !73)
// CHECK:STDOUT: !87 = !DILocation(line: 36, column: 21, scope: !73)
// CHECK:STDOUT: !88 = !DILocation(line: 40, column: 21, scope: !73)
// CHECK:STDOUT: !89 = !DILocation(line: 41, column: 21, scope: !73)
// CHECK:STDOUT: !90 = !DILocation(line: 43, column: 7, scope: !73)
// CHECK:STDOUT: !91 = !DILocation(line: 43, column: 14, scope: !73)
// CHECK:STDOUT: !92 = !DILocation(line: 43, column: 22, scope: !73)
// CHECK:STDOUT: !93 = !DILocation(line: 43, column: 30, scope: !73)
// CHECK:STDOUT: !94 = !DILocation(line: 43, column: 38, scope: !73)
// CHECK:STDOUT: !95 = !DILocation(line: 43, column: 45, scope: !73)
// CHECK:STDOUT: !96 = !DILocation(line: 43, column: 53, scope: !73)
// CHECK:STDOUT: !97 = !DILocation(line: 43, column: 61, scope: !73)
// CHECK:STDOUT: !98 = !DILocation(line: 43, column: 3, scope: !73)
// CHECK:STDOUT: !99 = !DILocation(line: 32, column: 1, scope: !73)
// CHECK:STDOUT: !100 = !{!101, !101, i64 0}
// CHECK:STDOUT: !101 = !{!"p1 omnipotent char", !102, i64 0}
// CHECK:STDOUT: !102 = !{!"any pointer", !10, i64 0}
// CHECK:STDOUT: !103 = !{!10, !10, i64 0}
// CHECK:STDOUT: !104 = !{!105, !105, i64 0}
// CHECK:STDOUT: !105 = !{!"p1 short", !102, i64 0}
// CHECK:STDOUT: !106 = !{!107, !107, i64 0}
// CHECK:STDOUT: !107 = !{!"short", !10, i64 0}
// CHECK:STDOUT: ; ModuleID = 'import_class.carbon'
// CHECK:STDOUT: source_filename = "import_class.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %struct.X = type { ptr, ptr }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CGetX.Main(ptr sret([16 x i8]) %return) #0 !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z4Makev.carbon_thunk(ptr %return), !dbg !16
// CHECK:STDOUT:   ret void, !dbg !17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CLet.Main() #0 !dbg !18 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_27.1.temp = alloca [16 x i8], align 1, !dbg !21
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc9_27.1.temp), !dbg !21
// CHECK:STDOUT:   call void @_Z4Makev.carbon_thunk(ptr %.loc9_27.1.temp), !dbg !21
// CHECK:STDOUT:   call void @_ZN1XD1Ev(ptr %.loc9_27.1.temp), !dbg !21
// CHECK:STDOUT:   ret void, !dbg !22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_ZN1XD1Ev(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CVar.Main() #0 !dbg !23 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca [16 x i8], align 1, !dbg !24
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !24
// CHECK:STDOUT:   call void @_Z4Makev.carbon_thunk(ptr %_.var), !dbg !25
// CHECK:STDOUT:   call void @_ZN1XD1Ev(ptr %_.var), !dbg !24
// CHECK:STDOUT:   ret void, !dbg !26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define dso_local void @_Z4Makev.carbon_thunk(ptr noundef %return) #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !27
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !27
// CHECK:STDOUT:   call void @_Z4Makev(ptr dead_on_unwind writable sret(%struct.X) align 8 %0)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z4Makev(ptr dead_on_unwind writable sret(%struct.X) align 8) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #2 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #3 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1, !2, !3, !4, !5}
// CHECK:STDOUT: !llvm.dbg.cu = !{!6}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!8}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = !{i32 1, !"wchar_size", i32 4}
// CHECK:STDOUT: !3 = !{i32 8, !"PIC Level", i32 0}
// CHECK:STDOUT: !4 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !5 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !6 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !7, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !7 = !DIFile(filename: "import_class.carbon", directory: "")
// CHECK:STDOUT: !8 = !{!9, !9, i64 0}
// CHECK:STDOUT: !9 = !{!"int", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"omnipotent char", !11, i64 0}
// CHECK:STDOUT: !11 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "GetX", linkageName: "_CGetX.Main", scope: null, file: !7, line: 6, type: !13, spFlags: DISPFlagDefinition, unit: !6)
// CHECK:STDOUT: !13 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !14 = !{!15}
// CHECK:STDOUT: !15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 8)
// CHECK:STDOUT: !16 = !DILocation(line: 6, column: 29, scope: !12)
// CHECK:STDOUT: !17 = !DILocation(line: 6, column: 22, scope: !12)
// CHECK:STDOUT: !18 = distinct !DISubprogram(name: "Let", linkageName: "_CLet.Main", scope: null, file: !7, line: 8, type: !19, spFlags: DISPFlagDefinition, unit: !6)
// CHECK:STDOUT: !19 = !DISubroutineType(types: !20)
// CHECK:STDOUT: !20 = !{null}
// CHECK:STDOUT: !21 = !DILocation(line: 9, column: 18, scope: !18)
// CHECK:STDOUT: !22 = !DILocation(line: 8, column: 1, scope: !18)
// CHECK:STDOUT: !23 = distinct !DISubprogram(name: "Var", linkageName: "_CVar.Main", scope: null, file: !7, line: 12, type: !19, spFlags: DISPFlagDefinition, unit: !6)
// CHECK:STDOUT: !24 = !DILocation(line: 13, column: 3, scope: !23)
// CHECK:STDOUT: !25 = !DILocation(line: 13, column: 18, scope: !23)
// CHECK:STDOUT: !26 = !DILocation(line: 12, column: 1, scope: !23)
// CHECK:STDOUT: !27 = !{!28, !28, i64 0}
// CHECK:STDOUT: !28 = !{!"p1 _ZTS1X", !29, i64 0}
// CHECK:STDOUT: !29 = !{!"any pointer", !10, i64 0}
// CHECK:STDOUT: ; ModuleID = 'indirect_return_with_args.carbon'
// CHECK:STDOUT: source_filename = "indirect_return_with_args.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %class.D = type { i8 }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCall.Main(ptr sret({}) %return, ptr %x) #0 !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z1f1D.carbon_thunk(ptr %x, ptr %return), !dbg !18
// CHECK:STDOUT:   ret void, !dbg !19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define dso_local void @_Z1f1D.carbon_thunk(ptr noundef %0, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %class.D, align 1
// CHECK:STDOUT:   %undef.agg.tmp = alloca %class.D, align 1
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !20
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !23
// CHECK:STDOUT:   %1 = load ptr, ptr %return.addr, align 8, !tbaa !23
// CHECK:STDOUT:   %2 = load ptr, ptr %.addr, align 8, !tbaa !20
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
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1, !2, !3, !4, !5}
// CHECK:STDOUT: !llvm.dbg.cu = !{!6}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!8}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = !{i32 1, !"wchar_size", i32 4}
// CHECK:STDOUT: !3 = !{i32 8, !"PIC Level", i32 0}
// CHECK:STDOUT: !4 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !5 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !6 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !7, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !7 = !DIFile(filename: "indirect_return_with_args.carbon", directory: "")
// CHECK:STDOUT: !8 = !{!9, !9, i64 0}
// CHECK:STDOUT: !9 = !{!"int", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"omnipotent char", !11, i64 0}
// CHECK:STDOUT: !11 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "Call", linkageName: "_CCall.Main", scope: null, file: !7, line: 10, type: !13, spFlags: DISPFlagDefinition, unit: !6, retainedNodes: !16)
// CHECK:STDOUT: !13 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !14 = !{!15, !15}
// CHECK:STDOUT: !15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 8)
// CHECK:STDOUT: !16 = !{!17}
// CHECK:STDOUT: !17 = !DILocalVariable(arg: 1, scope: !12, type: !15)
// CHECK:STDOUT: !18 = !DILocation(line: 11, column: 10, scope: !12)
// CHECK:STDOUT: !19 = !DILocation(line: 11, column: 3, scope: !12)
// CHECK:STDOUT: !20 = !{!21, !21, i64 0}
// CHECK:STDOUT: !21 = !{!"p1 _ZTS1D", !22, i64 0}
// CHECK:STDOUT: !22 = !{!"any pointer", !10, i64 0}
// CHECK:STDOUT: !23 = !{!24, !24, i64 0}
// CHECK:STDOUT: !24 = !{!"p1 _ZTS1C", !22, i64 0}
