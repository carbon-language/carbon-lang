// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/function/import/parameters.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/function/import/parameters.carbon

// ============================================================================
// Parameter requires a thunk to be generated
// ============================================================================

// --- ints.h

// TODO: Also test `long long` once it's supported.

void pass_signed(signed char, short, int, long);

void pass_unsigned(unsigned char, unsigned short, unsigned int, unsigned long);

void pass_short(short);

// --- import_ints.carbon

library "[[@TEST_NAME]]";

import Cpp library "ints.h";

fn MyF() {
  Cpp.pass_signed(1, 2, 3, 4);

  Cpp.pass_unsigned(1, 2, 3, 4);
}

fn MakeShort() -> i16;

var c: i16;

fn PassShort(a: i16, b: i16*) {
  Cpp.pass_short(a);
  Cpp.pass_short(*b);
  Cpp.pass_short(c);
  Cpp.pass_short(MakeShort());
}

// --- struct.h

struct X {
  int a;
  int b;
  int c;
};

void pass_struct(X);

// --- import_struct.carbon

library "[[@TEST_NAME]]";

import Cpp library "struct.h";

fn Test() {
  var x: Cpp.X;
  x.a = 1;
  x.b = 2;
  x.c = 3;
  Cpp.pass_struct(x);
}

// --- class_with_nontrivial_copy.h

class Y {
 public:
  Y(const Y&);

  int a;
  int b;
  int c;
};

void pass_struct(Y);

// --- import_class_with_nontrivial_copy.carbon

library "[[@TEST_NAME]]";

import Cpp library "class_with_nontrivial_copy.h";

fn PassRefExpr() {
  var y: Cpp.Y;
  y.a = 1;
  y.b = 2;
  y.c = 3;
  Cpp.pass_struct(y);
}

fn Make() -> Cpp.Y;

fn PassInitExpr() {
  Cpp.pass_struct(Make());
}

fn PassValueExpr(y: Cpp.Y) {
  Cpp.pass_struct(y);
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'import_ints.carbon'
// CHECK:STDOUT: source_filename = "import_ints.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: @int_1.f1e = internal constant i8 1
// CHECK:STDOUT: @int_2.5c2 = internal constant i16 2
// CHECK:STDOUT: @int_1.067 = internal constant i8 1
// CHECK:STDOUT: @int_2.ec3 = internal constant i16 2
// CHECK:STDOUT: @_Cc.Main = global i16 0
// CHECK:STDOUT: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 0, ptr @_C__global_init.Main, ptr null }]
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMyF.Main() #0 !dbg !61 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_19.3.temp = alloca i8, align 1, !dbg !62
// CHECK:STDOUT:   %.loc7_22.3.temp = alloca i16, align 2, !dbg !63
// CHECK:STDOUT:   %.loc9_21.3.temp = alloca i8, align 1, !dbg !65
// CHECK:STDOUT:   %.loc9_24.3.temp = alloca i16, align 2, !dbg !66
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_19.3.temp), !dbg !62
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_22.3.temp), !dbg !63
// CHECK:STDOUT:   call void @_Z11pass_signedasil.carbon_thunk.____(ptr @int_1.f1e, ptr @int_2.5c2, i32 3, i64 4), !dbg !64
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.a9999828a34c3ec7(ptr @int_2.5c2), !dbg !63
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.2e4de4ddc495f63e(ptr @int_1.f1e), !dbg !62
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc9_21.3.temp), !dbg !65
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc9_24.3.temp), !dbg !66
// CHECK:STDOUT:   call void @_Z13pass_unsignedhtjm.carbon_thunk.____(ptr @int_1.067, ptr @int_2.ec3, i32 3, i64 4), !dbg !67
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.04aa151d00e71a1b(ptr @int_2.ec3), !dbg !66
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.591d11926ce9716f(ptr @int_1.067), !dbg !65
// CHECK:STDOUT:   ret void, !dbg !68
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z11pass_signedasil.carbon_thunk.____(ptr noundef %0, ptr noundef %1, i32 noundef %2, i64 noundef %3) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %.addr1 = alloca ptr, align 8
// CHECK:STDOUT:   %.addr2 = alloca i32, align 4
// CHECK:STDOUT:   %.addr3 = alloca i64, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !136
// CHECK:STDOUT:   store ptr %1, ptr %.addr1, align 8, !tbaa !139
// CHECK:STDOUT:   store i32 %2, ptr %.addr2, align 4, !tbaa !141
// CHECK:STDOUT:   store i64 %3, ptr %.addr3, align 8, !tbaa !143
// CHECK:STDOUT:   %4 = load ptr, ptr %.addr, align 8, !tbaa !136
// CHECK:STDOUT:   %5 = load i8, ptr %4, align 1, !tbaa !144
// CHECK:STDOUT:   %6 = load ptr, ptr %.addr1, align 8, !tbaa !139
// CHECK:STDOUT:   %7 = load i16, ptr %6, align 2, !tbaa !145
// CHECK:STDOUT:   %8 = load i32, ptr %.addr2, align 4, !tbaa !141
// CHECK:STDOUT:   %9 = load i64, ptr %.addr3, align 8, !tbaa !143
// CHECK:STDOUT:   call void @_Z11pass_signedasil(i8 noundef signext %5, i16 noundef signext %7, i32 noundef %8, i64 noundef %9)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.d57cb32fcf12ec07:core.Destroy.Core"(ptr %self) #0 !dbg !69 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !71
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.4954785a8b134801:core.Destroy.Core"(ptr %self) #0 !dbg !73 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !75
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z13pass_unsignedhtjm.carbon_thunk.____(ptr noundef %0, ptr noundef %1, i32 noundef %2, i64 noundef %3) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %.addr1 = alloca ptr, align 8
// CHECK:STDOUT:   %.addr2 = alloca i32, align 4
// CHECK:STDOUT:   %.addr3 = alloca i64, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !136
// CHECK:STDOUT:   store ptr %1, ptr %.addr1, align 8, !tbaa !139
// CHECK:STDOUT:   store i32 %2, ptr %.addr2, align 4, !tbaa !141
// CHECK:STDOUT:   store i64 %3, ptr %.addr3, align 8, !tbaa !143
// CHECK:STDOUT:   %4 = load ptr, ptr %.addr, align 8, !tbaa !136
// CHECK:STDOUT:   %5 = load i8, ptr %4, align 1, !tbaa !144
// CHECK:STDOUT:   %6 = load ptr, ptr %.addr1, align 8, !tbaa !139
// CHECK:STDOUT:   %7 = load i16, ptr %6, align 2, !tbaa !145
// CHECK:STDOUT:   %8 = load i32, ptr %.addr2, align 4, !tbaa !141
// CHECK:STDOUT:   %9 = load i64, ptr %.addr3, align 8, !tbaa !143
// CHECK:STDOUT:   call void @_Z13pass_unsignedhtjm(i8 noundef zeroext %5, i16 noundef zeroext %7, i32 noundef %8, i64 noundef %9)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.3bc9e5dacf328f74:core.Destroy.Core"(ptr %self) #0 !dbg !77 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !79
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.be60742f00557eec:core.Destroy.Core"(ptr %self) #0 !dbg !81 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !83
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i16 @_CMakeShort.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassShort.Main(i16 %a, ptr %b) #0 !dbg !85 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc17_18.1.temp = alloca i16, align 2, !dbg !90
// CHECK:STDOUT:   %.loc18_18.3.temp = alloca i16, align 2, !dbg !93
// CHECK:STDOUT:   %.loc19_18.2.temp = alloca i16, align 2, !dbg !95
// CHECK:STDOUT:   %.loc20_28.3.temp = alloca i16, align 2, !dbg !97
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc17_18.1.temp), !dbg !90
// CHECK:STDOUT:   store i16 %a, ptr %.loc17_18.1.temp, align 2, !dbg !90
// CHECK:STDOUT:   call void @_Z10pass_shorts.carbon_thunk._(ptr %.loc17_18.1.temp), !dbg !91
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.a9999828a34c3ec7(ptr %.loc17_18.1.temp), !dbg !90
// CHECK:STDOUT:   %.loc18_18.2 = load i16, ptr %b, align 2, !dbg !93
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc18_18.3.temp), !dbg !93
// CHECK:STDOUT:   store i16 %.loc18_18.2, ptr %.loc18_18.3.temp, align 2, !dbg !93
// CHECK:STDOUT:   call void @_Z10pass_shorts.carbon_thunk._(ptr %.loc18_18.3.temp), !dbg !94
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.a9999828a34c3ec7(ptr %.loc18_18.3.temp), !dbg !93
// CHECK:STDOUT:   %.loc19_18.1 = load i16, ptr @_Cc.Main, align 2, !dbg !95
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc19_18.2.temp), !dbg !95
// CHECK:STDOUT:   store i16 %.loc19_18.1, ptr %.loc19_18.2.temp, align 2, !dbg !95
// CHECK:STDOUT:   call void @_Z10pass_shorts.carbon_thunk._(ptr %.loc19_18.2.temp), !dbg !96
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.a9999828a34c3ec7(ptr %.loc19_18.2.temp), !dbg !95
// CHECK:STDOUT:   %MakeShort.call = call i16 @_CMakeShort.Main(), !dbg !97
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc20_28.3.temp), !dbg !97
// CHECK:STDOUT:   store i16 %MakeShort.call, ptr %.loc20_28.3.temp, align 2, !dbg !97
// CHECK:STDOUT:   call void @_Z10pass_shorts.carbon_thunk._(ptr %.loc20_28.3.temp), !dbg !98
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.a9999828a34c3ec7(ptr %.loc20_28.3.temp), !dbg !97
// CHECK:STDOUT:   ret void, !dbg !99
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z10pass_shorts.carbon_thunk._(ptr noundef %0) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !139
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !139
// CHECK:STDOUT:   %2 = load i16, ptr %1, align 2, !tbaa !145
// CHECK:STDOUT:   call void @_Z10pass_shorts(i16 noundef signext %2)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.a9999828a34c3ec7(ptr %self) #0 !dbg !104 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.d57cb32fcf12ec07:core.Destroy.Core"(ptr %self), !dbg !108
// CHECK:STDOUT:   ret void, !dbg !109
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.2e4de4ddc495f63e(ptr %self) #0 !dbg !111 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.4954785a8b134801:core.Destroy.Core"(ptr %self), !dbg !115
// CHECK:STDOUT:   ret void, !dbg !116
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.04aa151d00e71a1b(ptr %self) #0 !dbg !118 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.3bc9e5dacf328f74:core.Destroy.Core"(ptr %self), !dbg !122
// CHECK:STDOUT:   ret void, !dbg !123
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.591d11926ce9716f(ptr %self) #0 !dbg !125 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.be60742f00557eec:core.Destroy.Core"(ptr %self), !dbg !129
// CHECK:STDOUT:   ret void, !dbg !130
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define internal void @_C__global_init.Main() #0 !dbg !101 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   store i16 poison, ptr @_Cc.Main, align 2, !dbg !102
// CHECK:STDOUT:   ret void, !dbg !103
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z11pass_signedasil(i8 noundef signext, i16 noundef signext, i32 noundef, i64 noundef) #3
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z13pass_unsignedhtjm(i8 noundef zeroext, i16 noundef zeroext, i32 noundef, i64 noundef) #3
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z10pass_shorts(i16 noundef signext) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_Z10pass_shorts.carbon_thunk._, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 7, 6, 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Destroy.Core.a9999828a34c3ec7, { 4, 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!33}
// CHECK:STDOUT: !llvm.module.flags = !{!146, !147, !148, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!150}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !8 = !DIBasicType(name: "int", size: 8, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !28 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !32 = !DIFile(filename: "import_ints.carbon", directory: "")
// CHECK:STDOUT: !33 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !32, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !36 = !DIBasicType(name: "int", size: 8, encoding: DW_ATE_signed)
// CHECK:STDOUT: !37 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_signed)
// CHECK:STDOUT: !39 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !40 = !{null}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !42 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !45 = !{null, !37}
// CHECK:STDOUT: !46 = !DISubroutineType(types: !45)
// CHECK:STDOUT: !47 = !{null, !36}
// CHECK:STDOUT: !48 = !DISubroutineType(types: !47)
// CHECK:STDOUT: !51 = !{null, !39}
// CHECK:STDOUT: !52 = !DISubroutineType(types: !51)
// CHECK:STDOUT: !53 = !{null, !8}
// CHECK:STDOUT: !54 = !DISubroutineType(types: !53)
// CHECK:STDOUT: !57 = !{null, !37, !42}
// CHECK:STDOUT: !58 = !DISubroutineType(types: !57)
// CHECK:STDOUT: !61 = distinct !DISubprogram(name: "MyF", linkageName: "_CMyF.Main", scope: null, file: !32, line: 6, type: !41, spFlags: DISPFlagDefinition, unit: !33)
// CHECK:STDOUT: !62 = !DILocation(line: 7, column: 19, scope: !61)
// CHECK:STDOUT: !63 = !DILocation(line: 7, column: 22, scope: !61)
// CHECK:STDOUT: !64 = !DILocation(line: 7, column: 3, scope: !61)
// CHECK:STDOUT: !65 = !DILocation(line: 9, column: 21, scope: !61)
// CHECK:STDOUT: !66 = !DILocation(line: 9, column: 24, scope: !61)
// CHECK:STDOUT: !67 = !DILocation(line: 9, column: 3, scope: !61)
// CHECK:STDOUT: !68 = !DILocation(line: 6, column: 1, scope: !61)
// CHECK:STDOUT: !69 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.d57cb32fcf12ec07:core.Destroy.Core", scope: null, file: !32, line: 7, type: !46, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !72)
// CHECK:STDOUT: !70 = !DILocalVariable(arg: 1, scope: !69, type: !37)
// CHECK:STDOUT: !71 = !DILocation(line: 7, column: 22, scope: !69)
// CHECK:STDOUT: !72 = !{!70}
// CHECK:STDOUT: !73 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.4954785a8b134801:core.Destroy.Core", scope: null, file: !32, line: 7, type: !48, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !76)
// CHECK:STDOUT: !74 = !DILocalVariable(arg: 1, scope: !73, type: !36)
// CHECK:STDOUT: !75 = !DILocation(line: 7, column: 19, scope: !73)
// CHECK:STDOUT: !76 = !{!74}
// CHECK:STDOUT: !77 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.3bc9e5dacf328f74:core.Destroy.Core", scope: null, file: !32, line: 9, type: !52, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !80)
// CHECK:STDOUT: !78 = !DILocalVariable(arg: 1, scope: !77, type: !39)
// CHECK:STDOUT: !79 = !DILocation(line: 9, column: 24, scope: !77)
// CHECK:STDOUT: !80 = !{!78}
// CHECK:STDOUT: !81 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.be60742f00557eec:core.Destroy.Core", scope: null, file: !32, line: 9, type: !54, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !84)
// CHECK:STDOUT: !82 = !DILocalVariable(arg: 1, scope: !81, type: !8)
// CHECK:STDOUT: !83 = !DILocation(line: 9, column: 21, scope: !81)
// CHECK:STDOUT: !84 = !{!82}
// CHECK:STDOUT: !85 = distinct !DISubprogram(name: "PassShort", linkageName: "_CPassShort.Main", scope: null, file: !32, line: 16, type: !58, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !100)
// CHECK:STDOUT: !86 = !DILocalVariable(arg: 1, scope: !85, type: !37)
// CHECK:STDOUT: !87 = !DILocalVariable(arg: 2, scope: !85, type: !42)
// CHECK:STDOUT: !90 = !DILocation(line: 17, column: 18, scope: !85)
// CHECK:STDOUT: !91 = !DILocation(line: 17, column: 3, scope: !85)
// CHECK:STDOUT: !93 = !DILocation(line: 18, column: 18, scope: !85)
// CHECK:STDOUT: !94 = !DILocation(line: 18, column: 3, scope: !85)
// CHECK:STDOUT: !95 = !DILocation(line: 19, column: 18, scope: !85)
// CHECK:STDOUT: !96 = !DILocation(line: 19, column: 3, scope: !85)
// CHECK:STDOUT: !97 = !DILocation(line: 20, column: 18, scope: !85)
// CHECK:STDOUT: !98 = !DILocation(line: 20, column: 3, scope: !85)
// CHECK:STDOUT: !99 = !DILocation(line: 16, column: 1, scope: !85)
// CHECK:STDOUT: !100 = !{!86, !87}
// CHECK:STDOUT: !101 = distinct !DISubprogram(name: "__global_init", linkageName: "_C__global_init.Main", scope: null, file: !32, type: !41, spFlags: DISPFlagDefinition, unit: !33)
// CHECK:STDOUT: !102 = !DILocation(line: 14, column: 1, scope: !101)
// CHECK:STDOUT: !103 = !DILocation(line: 0, scope: !101)
// CHECK:STDOUT: !104 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.a9999828a34c3ec7", scope: null, file: !28, line: 24, type: !46, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !110)
// CHECK:STDOUT: !105 = !DILocalVariable(arg: 1, scope: !104, type: !37)
// CHECK:STDOUT: !108 = !DILocation(line: 26, column: 5, scope: !104)
// CHECK:STDOUT: !109 = !DILocation(line: 24, column: 3, scope: !104)
// CHECK:STDOUT: !110 = !{!105}
// CHECK:STDOUT: !111 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.2e4de4ddc495f63e", scope: null, file: !28, line: 24, type: !48, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !117)
// CHECK:STDOUT: !112 = !DILocalVariable(arg: 1, scope: !111, type: !36)
// CHECK:STDOUT: !115 = !DILocation(line: 26, column: 5, scope: !111)
// CHECK:STDOUT: !116 = !DILocation(line: 24, column: 3, scope: !111)
// CHECK:STDOUT: !117 = !{!112}
// CHECK:STDOUT: !118 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.04aa151d00e71a1b", scope: null, file: !28, line: 24, type: !52, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !124)
// CHECK:STDOUT: !119 = !DILocalVariable(arg: 1, scope: !118, type: !39)
// CHECK:STDOUT: !122 = !DILocation(line: 26, column: 5, scope: !118)
// CHECK:STDOUT: !123 = !DILocation(line: 24, column: 3, scope: !118)
// CHECK:STDOUT: !124 = !{!119}
// CHECK:STDOUT: !125 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.591d11926ce9716f", scope: null, file: !28, line: 24, type: !54, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !131)
// CHECK:STDOUT: !126 = !DILocalVariable(arg: 1, scope: !125, type: !8)
// CHECK:STDOUT: !129 = !DILocation(line: 26, column: 5, scope: !125)
// CHECK:STDOUT: !130 = !DILocation(line: 24, column: 3, scope: !125)
// CHECK:STDOUT: !131 = !{!126}
// CHECK:STDOUT: !132 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !133 = !{!"omnipotent char", !132, i64 0}
// CHECK:STDOUT: !134 = !{!"any pointer", !133, i64 0}
// CHECK:STDOUT: !135 = !{!"p1 omnipotent char", !134, i64 0}
// CHECK:STDOUT: !136 = !{!135, !135, i64 0}
// CHECK:STDOUT: !137 = !{!"short", !133, i64 0}
// CHECK:STDOUT: !138 = !{!"p1 short", !134, i64 0}
// CHECK:STDOUT: !139 = !{!138, !138, i64 0}
// CHECK:STDOUT: !140 = !{!"int", !133, i64 0}
// CHECK:STDOUT: !141 = !{!140, !140, i64 0}
// CHECK:STDOUT: !142 = !{!"long", !133, i64 0}
// CHECK:STDOUT: !143 = !{!142, !142, i64 0}
// CHECK:STDOUT: !144 = !{!133, !133, i64 0}
// CHECK:STDOUT: !145 = !{!137, !137, i64 0}
// CHECK:STDOUT: !146 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !147 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !148 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !149 = !{!"__libc_errno", !140, i64 0}
// CHECK:STDOUT: !150 = !{!149, !140, i64 0}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'import_struct.carbon'
// CHECK:STDOUT: source_filename = "import_struct.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %struct.X = type { i32, i32, i32 }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTest.Main() #0 !dbg !155 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x.var = alloca [12 x i8], align 4, !dbg !156
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %x.var), !dbg !156
// CHECK:STDOUT:   call void @"_COp.51f3fb62cb986374:DefaultOrUnformed.Core.102476b7ae87d0a1"(ptr %x.var), !dbg !156
// CHECK:STDOUT:   %.loc8_4.a = getelementptr inbounds nuw [12 x i8], ptr %x.var, i32 0, i32 0, !dbg !158
// CHECK:STDOUT:   store i32 1, ptr %.loc8_4.a, align 4, !dbg !158
// CHECK:STDOUT:   %.loc9_4.b = getelementptr inbounds nuw [12 x i8], ptr %x.var, i32 0, i32 4, !dbg !159
// CHECK:STDOUT:   store i32 2, ptr %.loc9_4.b, align 4, !dbg !159
// CHECK:STDOUT:   %.loc10_4.c = getelementptr inbounds nuw [12 x i8], ptr %x.var, i32 0, i32 8, !dbg !160
// CHECK:STDOUT:   store i32 3, ptr %.loc10_4.c, align 4, !dbg !160
// CHECK:STDOUT:   call void @_Z11pass_struct1X.carbon_thunk._(ptr %x.var), !dbg !162
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.155369379014d6e1(ptr %x.var), !dbg !156
// CHECK:STDOUT:   ret void, !dbg !163
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN1XC1Ev.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !180
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !180
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:Default.eed9ddf3aaac414a.Core:X.Cpp"(ptr sret([12 x i8]) %return) #2 !dbg !165 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN1XC1Ev.carbon_thunk.(ptr %return), !dbg !166
// CHECK:STDOUT:   ret void, !dbg !166
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z11pass_struct1X.carbon_thunk._(ptr noundef %0) #3 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %struct.X, align 4
// CHECK:STDOUT:   %agg.tmp.coerce = alloca { i64, i32 }, align 4
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !180
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !180
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp, ptr align 4 %1, i64 12, i1 false), !tbaa.struct !182
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp.coerce, ptr align 4 %agg.tmp, i64 12, i1 false)
// CHECK:STDOUT:   %2 = getelementptr inbounds nuw { i64, i32 }, ptr %agg.tmp.coerce, i32 0, i32 0
// CHECK:STDOUT:   %3 = load i64, ptr %2, align 4
// CHECK:STDOUT:   %4 = getelementptr inbounds nuw { i64, i32 }, ptr %agg.tmp.coerce, i32 0, i32 1
// CHECK:STDOUT:   %5 = load i32, ptr %4, align 4
// CHECK:STDOUT:   call void @_Z11pass_struct1X(i64 %3, i32 %5)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #4
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.51f3fb62cb986374:DefaultOrUnformed.Core.102476b7ae87d0a1"(ptr sret([12 x i8]) %return) #0 !dbg !167 {
// CHECK:STDOUT:   call void @"_COp:thunk:Default.eed9ddf3aaac414a.Core:X.Cpp"(ptr %return), !dbg !170
// CHECK:STDOUT:   ret void, !dbg !171
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.155369379014d6e1(ptr %self) #0 !dbg !172 {
// CHECK:STDOUT:   ret void, !dbg !177
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z11pass_struct1X(i64, i32) #5
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #4
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #3 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #4 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #5 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!152}
// CHECK:STDOUT: !llvm.module.flags = !{!146, !147, !148, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!150}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !26 = !DIFile(filename: "min_prelude/parts/default.carbon", directory: "")
// CHECK:STDOUT: !28 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !40 = !{null}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !42 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !59 = !{null, !42}
// CHECK:STDOUT: !60 = !DISubroutineType(types: !59)
// CHECK:STDOUT: !132 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !133 = !{!"omnipotent char", !132, i64 0}
// CHECK:STDOUT: !134 = !{!"any pointer", !133, i64 0}
// CHECK:STDOUT: !140 = !{!"int", !133, i64 0}
// CHECK:STDOUT: !141 = !{!140, !140, i64 0}
// CHECK:STDOUT: !146 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !147 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !148 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !149 = !{!"__libc_errno", !140, i64 0}
// CHECK:STDOUT: !150 = !{!149, !140, i64 0}
// CHECK:STDOUT: !151 = !DIFile(filename: "import_struct.carbon", directory: "")
// CHECK:STDOUT: !152 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !151, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !153 = !{!42}
// CHECK:STDOUT: !154 = !DISubroutineType(types: !153)
// CHECK:STDOUT: !155 = distinct !DISubprogram(name: "Test", linkageName: "_CTest.Main", scope: null, file: !151, line: 6, type: !41, spFlags: DISPFlagDefinition, unit: !152)
// CHECK:STDOUT: !156 = !DILocation(line: 7, column: 3, scope: !155)
// CHECK:STDOUT: !158 = !DILocation(line: 8, column: 3, scope: !155)
// CHECK:STDOUT: !159 = !DILocation(line: 9, column: 3, scope: !155)
// CHECK:STDOUT: !160 = !DILocation(line: 10, column: 3, scope: !155)
// CHECK:STDOUT: !162 = !DILocation(line: 11, column: 3, scope: !155)
// CHECK:STDOUT: !163 = !DILocation(line: 6, column: 1, scope: !155)
// CHECK:STDOUT: !164 = !DIFile(filename: "./struct.h", directory: "")
// CHECK:STDOUT: !165 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:Default.eed9ddf3aaac414a.Core:X.Cpp", scope: null, file: !164, line: 2, type: !154, spFlags: DISPFlagDefinition, unit: !152)
// CHECK:STDOUT: !166 = !DILocation(line: 2, column: 8, scope: !165)
// CHECK:STDOUT: !167 = distinct !DISubprogram(name: "Op", linkageName: "_COp.51f3fb62cb986374:DefaultOrUnformed.Core.102476b7ae87d0a1", scope: null, file: !26, line: 9, type: !154, spFlags: DISPFlagDefinition, unit: !152)
// CHECK:STDOUT: !170 = !DILocation(line: 9, column: 28, scope: !167)
// CHECK:STDOUT: !171 = !DILocation(line: 9, column: 21, scope: !167)
// CHECK:STDOUT: !172 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.155369379014d6e1", scope: null, file: !28, line: 24, type: !60, spFlags: DISPFlagDefinition, unit: !152, retainedNodes: !178)
// CHECK:STDOUT: !173 = !DILocalVariable(arg: 1, scope: !172, type: !42)
// CHECK:STDOUT: !177 = !DILocation(line: 24, column: 3, scope: !172)
// CHECK:STDOUT: !178 = !{!173}
// CHECK:STDOUT: !179 = !{!"p1 _ZTS1X", !134, i64 0}
// CHECK:STDOUT: !180 = !{!179, !179, i64 0}
// CHECK:STDOUT: !182 = !{i64 0, i64 4, !141, i64 4, i64 4, !141, i64 8, i64 4, !141}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'import_class_with_nontrivial_copy.carbon'
// CHECK:STDOUT: source_filename = "import_class_with_nontrivial_copy.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %class.Y = type { i32, i32, i32 }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassRefExpr.Main() #0 !dbg !185 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %y.var = alloca [12 x i8], align 4, !dbg !186
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %y.var), !dbg !186
// CHECK:STDOUT:   %.loc8_4.a = getelementptr inbounds nuw [12 x i8], ptr %y.var, i32 0, i32 0, !dbg !188
// CHECK:STDOUT:   store i32 1, ptr %.loc8_4.a, align 4, !dbg !188
// CHECK:STDOUT:   %.loc9_4.b = getelementptr inbounds nuw [12 x i8], ptr %y.var, i32 0, i32 4, !dbg !189
// CHECK:STDOUT:   store i32 2, ptr %.loc9_4.b, align 4, !dbg !189
// CHECK:STDOUT:   %.loc10_4.c = getelementptr inbounds nuw [12 x i8], ptr %y.var, i32 0, i32 8, !dbg !190
// CHECK:STDOUT:   store i32 3, ptr %.loc10_4.c, align 4, !dbg !190
// CHECK:STDOUT:   call void @_Z11pass_struct1Y.carbon_thunk._(ptr %y.var), !dbg !192
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.452c01d28c253470(ptr %y.var), !dbg !186
// CHECK:STDOUT:   ret void, !dbg !193
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z11pass_struct1Y.carbon_thunk._(ptr noundef %0) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %class.Y, align 4
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !213
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !213
// CHECK:STDOUT:   call void @_ZN1YC1ERKS_(ptr noundef nonnull align 4 dereferenceable(12) %agg.tmp, ptr noundef nonnull align 4 dereferenceable(12) %1)
// CHECK:STDOUT:   call void @_Z11pass_struct1Y(ptr nofreeobj noundef align 4 dead_on_return dereferenceable(12) %agg.tmp)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CMake.Main(ptr sret([12 x i8]))
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassInitExpr.Main() #0 !dbg !194 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc17_24.1.temp = alloca [12 x i8], align 4, !dbg !195
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc17_24.1.temp), !dbg !195
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.loc17_24.1.temp), !dbg !195
// CHECK:STDOUT:   call void @_Z11pass_struct1Y.carbon_thunk._(ptr %.loc17_24.1.temp), !dbg !196
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.452c01d28c253470(ptr %.loc17_24.1.temp), !dbg !195
// CHECK:STDOUT:   ret void, !dbg !197
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassValueExpr.Main(ptr %y) #0 !dbg !198 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z11pass_struct1Y.carbon_thunk._(ptr %y), !dbg !202
// CHECK:STDOUT:   ret void, !dbg !203
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.452c01d28c253470(ptr %self) #0 !dbg !205 {
// CHECK:STDOUT:   ret void, !dbg !210
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z11pass_struct1Y(ptr nofreeobj noundef align 4 dead_on_return dereferenceable(12)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_ZN1YC1ERKS_(ptr noundef nonnull align 4 dereferenceable(12), ptr noundef nonnull align 4 dereferenceable(12)) unnamed_addr #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Destroy.Core.452c01d28c253470, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!184}
// CHECK:STDOUT: !llvm.module.flags = !{!146, !147, !148, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!150}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !28 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !40 = !{null}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !42 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !59 = !{null, !42}
// CHECK:STDOUT: !60 = !DISubroutineType(types: !59)
// CHECK:STDOUT: !132 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !133 = !{!"omnipotent char", !132, i64 0}
// CHECK:STDOUT: !134 = !{!"any pointer", !133, i64 0}
// CHECK:STDOUT: !140 = !{!"int", !133, i64 0}
// CHECK:STDOUT: !146 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !147 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !148 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !149 = !{!"__libc_errno", !140, i64 0}
// CHECK:STDOUT: !150 = !{!149, !140, i64 0}
// CHECK:STDOUT: !183 = !DIFile(filename: "import_class_with_nontrivial_copy.carbon", directory: "")
// CHECK:STDOUT: !184 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !183, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !185 = distinct !DISubprogram(name: "PassRefExpr", linkageName: "_CPassRefExpr.Main", scope: null, file: !183, line: 6, type: !41, spFlags: DISPFlagDefinition, unit: !184)
// CHECK:STDOUT: !186 = !DILocation(line: 7, column: 3, scope: !185)
// CHECK:STDOUT: !188 = !DILocation(line: 8, column: 3, scope: !185)
// CHECK:STDOUT: !189 = !DILocation(line: 9, column: 3, scope: !185)
// CHECK:STDOUT: !190 = !DILocation(line: 10, column: 3, scope: !185)
// CHECK:STDOUT: !192 = !DILocation(line: 11, column: 3, scope: !185)
// CHECK:STDOUT: !193 = !DILocation(line: 6, column: 1, scope: !185)
// CHECK:STDOUT: !194 = distinct !DISubprogram(name: "PassInitExpr", linkageName: "_CPassInitExpr.Main", scope: null, file: !183, line: 16, type: !41, spFlags: DISPFlagDefinition, unit: !184)
// CHECK:STDOUT: !195 = !DILocation(line: 17, column: 19, scope: !194)
// CHECK:STDOUT: !196 = !DILocation(line: 17, column: 3, scope: !194)
// CHECK:STDOUT: !197 = !DILocation(line: 16, column: 1, scope: !194)
// CHECK:STDOUT: !198 = distinct !DISubprogram(name: "PassValueExpr", linkageName: "_CPassValueExpr.Main", scope: null, file: !183, line: 20, type: !60, spFlags: DISPFlagDefinition, unit: !184, retainedNodes: !204)
// CHECK:STDOUT: !199 = !DILocalVariable(arg: 1, scope: !198, type: !42)
// CHECK:STDOUT: !202 = !DILocation(line: 21, column: 3, scope: !198)
// CHECK:STDOUT: !203 = !DILocation(line: 20, column: 1, scope: !198)
// CHECK:STDOUT: !204 = !{!199}
// CHECK:STDOUT: !205 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.452c01d28c253470", scope: null, file: !28, line: 24, type: !60, spFlags: DISPFlagDefinition, unit: !184, retainedNodes: !211)
// CHECK:STDOUT: !206 = !DILocalVariable(arg: 1, scope: !205, type: !42)
// CHECK:STDOUT: !210 = !DILocation(line: 24, column: 3, scope: !205)
// CHECK:STDOUT: !211 = !{!206}
// CHECK:STDOUT: !212 = !{!"p1 _ZTS1Y", !134, i64 0}
// CHECK:STDOUT: !213 = !{!212, !212, i64 0}
// CHECK:STDOUT:
