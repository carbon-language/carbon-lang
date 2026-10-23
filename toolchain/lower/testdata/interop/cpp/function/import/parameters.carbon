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
// CHECK:STDOUT: @int_1.2eb = internal constant i8 1
// CHECK:STDOUT: @int_2.fee = internal constant i16 2
// CHECK:STDOUT: @int_1.5dc = internal constant i8 1
// CHECK:STDOUT: @int_2.2d4 = internal constant i16 2
// CHECK:STDOUT: @_Cc.Main = global i16 0
// CHECK:STDOUT: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 0, ptr @_C__global_init.Main, ptr null }]
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMyF.Main() #0 !dbg !55 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_19.3.temp = alloca i8, align 1, !dbg !56
// CHECK:STDOUT:   %.loc7_22.3.temp = alloca i16, align 2, !dbg !57
// CHECK:STDOUT:   %.loc9_21.3.temp = alloca i8, align 1, !dbg !63
// CHECK:STDOUT:   %.loc9_24.3.temp = alloca i16, align 2, !dbg !64
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_19.3.temp), !dbg !56
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_22.3.temp), !dbg !57
// CHECK:STDOUT:   call void @_Z11pass_signedasil.carbon_thunk.____(ptr @int_1.2eb, ptr @int_2.fee, i32 3, i64 4), !dbg !58
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.e43967308dd20a9f(ptr @int_2.fee), !dbg !57
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.6fb12317fdf5c9b3(ptr @int_1.2eb), !dbg !56
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc9_21.3.temp), !dbg !63
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc9_24.3.temp), !dbg !64
// CHECK:STDOUT:   call void @_Z13pass_unsignedhtjm.carbon_thunk.____(ptr @int_1.5dc, ptr @int_2.2d4, i32 3, i64 4), !dbg !65
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.44e7310a0b394feb(ptr @int_2.2d4), !dbg !64
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.73549b49ca05bf4c(ptr @int_1.5dc), !dbg !63
// CHECK:STDOUT:   ret void, !dbg !70
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z11pass_signedasil.carbon_thunk.____(ptr noundef %0, ptr noundef %1, i32 noundef %2, i64 noundef %3) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %.addr1 = alloca ptr, align 8
// CHECK:STDOUT:   %.addr2 = alloca i32, align 4
// CHECK:STDOUT:   %.addr3 = alloca i64, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !122
// CHECK:STDOUT:   store ptr %1, ptr %.addr1, align 8, !tbaa !125
// CHECK:STDOUT:   store i32 %2, ptr %.addr2, align 4, !tbaa !127
// CHECK:STDOUT:   store i64 %3, ptr %.addr3, align 8, !tbaa !129
// CHECK:STDOUT:   %4 = load ptr, ptr %.addr, align 8, !tbaa !122
// CHECK:STDOUT:   %5 = load i8, ptr %4, align 1, !tbaa !130
// CHECK:STDOUT:   %6 = load ptr, ptr %.addr1, align 8, !tbaa !125
// CHECK:STDOUT:   %7 = load i16, ptr %6, align 2, !tbaa !131
// CHECK:STDOUT:   %8 = load i32, ptr %.addr2, align 4, !tbaa !127
// CHECK:STDOUT:   %9 = load i64, ptr %.addr3, align 8, !tbaa !129
// CHECK:STDOUT:   call void @_Z11pass_signedasil(i8 noundef signext %5, i16 noundef signext %7, i32 noundef %8, i64 noundef %9)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z13pass_unsignedhtjm.carbon_thunk.____(ptr noundef %0, ptr noundef %1, i32 noundef %2, i64 noundef %3) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %.addr1 = alloca ptr, align 8
// CHECK:STDOUT:   %.addr2 = alloca i32, align 4
// CHECK:STDOUT:   %.addr3 = alloca i64, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !122
// CHECK:STDOUT:   store ptr %1, ptr %.addr1, align 8, !tbaa !125
// CHECK:STDOUT:   store i32 %2, ptr %.addr2, align 4, !tbaa !127
// CHECK:STDOUT:   store i64 %3, ptr %.addr3, align 8, !tbaa !129
// CHECK:STDOUT:   %4 = load ptr, ptr %.addr, align 8, !tbaa !122
// CHECK:STDOUT:   %5 = load i8, ptr %4, align 1, !tbaa !130
// CHECK:STDOUT:   %6 = load ptr, ptr %.addr1, align 8, !tbaa !125
// CHECK:STDOUT:   %7 = load i16, ptr %6, align 2, !tbaa !131
// CHECK:STDOUT:   %8 = load i32, ptr %.addr2, align 4, !tbaa !127
// CHECK:STDOUT:   %9 = load i64, ptr %.addr3, align 8, !tbaa !129
// CHECK:STDOUT:   call void @_Z13pass_unsignedhtjm(i8 noundef zeroext %5, i16 noundef zeroext %7, i32 noundef %8, i64 noundef %9)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i16 @_CMakeShort.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassShort.Main(i16 %a, ptr %b) #0 !dbg !71 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc17_18.1.temp = alloca i16, align 2, !dbg !76
// CHECK:STDOUT:   %.loc18_18.3.temp = alloca i16, align 2, !dbg !79
// CHECK:STDOUT:   %.loc19_18.2.temp = alloca i16, align 2, !dbg !81
// CHECK:STDOUT:   %.loc20_28.3.temp = alloca i16, align 2, !dbg !83
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc17_18.1.temp), !dbg !76
// CHECK:STDOUT:   store i16 %a, ptr %.loc17_18.1.temp, align 2, !dbg !76
// CHECK:STDOUT:   call void @_Z10pass_shorts.carbon_thunk._(ptr %.loc17_18.1.temp), !dbg !77
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.e43967308dd20a9f(ptr %.loc17_18.1.temp), !dbg !76
// CHECK:STDOUT:   %.loc18_18.2 = load i16, ptr %b, align 2, !dbg !79
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc18_18.3.temp), !dbg !79
// CHECK:STDOUT:   store i16 %.loc18_18.2, ptr %.loc18_18.3.temp, align 2, !dbg !79
// CHECK:STDOUT:   call void @_Z10pass_shorts.carbon_thunk._(ptr %.loc18_18.3.temp), !dbg !80
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.e43967308dd20a9f(ptr %.loc18_18.3.temp), !dbg !79
// CHECK:STDOUT:   %.loc19_18.1 = load i16, ptr @_Cc.Main, align 2, !dbg !81
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc19_18.2.temp), !dbg !81
// CHECK:STDOUT:   store i16 %.loc19_18.1, ptr %.loc19_18.2.temp, align 2, !dbg !81
// CHECK:STDOUT:   call void @_Z10pass_shorts.carbon_thunk._(ptr %.loc19_18.2.temp), !dbg !82
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.e43967308dd20a9f(ptr %.loc19_18.2.temp), !dbg !81
// CHECK:STDOUT:   %MakeShort.call = call i16 @_CMakeShort.Main(), !dbg !83
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc20_28.3.temp), !dbg !83
// CHECK:STDOUT:   store i16 %MakeShort.call, ptr %.loc20_28.3.temp, align 2, !dbg !83
// CHECK:STDOUT:   call void @_Z10pass_shorts.carbon_thunk._(ptr %.loc20_28.3.temp), !dbg !84
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.e43967308dd20a9f(ptr %.loc20_28.3.temp), !dbg !83
// CHECK:STDOUT:   ret void, !dbg !85
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z10pass_shorts.carbon_thunk._(ptr noundef %0) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !125
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !125
// CHECK:STDOUT:   %2 = load i16, ptr %1, align 2, !tbaa !131
// CHECK:STDOUT:   call void @_Z10pass_shorts(i16 noundef signext %2)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.e43967308dd20a9f(ptr %t) #0 !dbg !90 {
// CHECK:STDOUT:   ret void, !dbg !95
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.6fb12317fdf5c9b3(ptr %t) #0 !dbg !97 {
// CHECK:STDOUT:   ret void, !dbg !102
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.44e7310a0b394feb(ptr %t) #0 !dbg !104 {
// CHECK:STDOUT:   ret void, !dbg !109
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.73549b49ca05bf4c(ptr %t) #0 !dbg !111 {
// CHECK:STDOUT:   ret void, !dbg !116
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define internal void @_C__global_init.Main() #0 !dbg !87 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   store i16 poison, ptr @_Cc.Main, align 2, !dbg !88
// CHECK:STDOUT:   ret void, !dbg !89
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
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.e43967308dd20a9f, { 4, 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!35}
// CHECK:STDOUT: !llvm.module.flags = !{!132, !133, !134, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!136}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !10 = !DIBasicType(name: "int", size: 8, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !26 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !34 = !DIFile(filename: "import_ints.carbon", directory: "")
// CHECK:STDOUT: !35 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !34, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !38 = !DIBasicType(name: "int", size: 8, encoding: DW_ATE_signed)
// CHECK:STDOUT: !39 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_signed)
// CHECK:STDOUT: !41 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !42 = !{null}
// CHECK:STDOUT: !43 = !DISubroutineType(types: !42)
// CHECK:STDOUT: !44 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !51 = !{null, !39, !44}
// CHECK:STDOUT: !52 = !DISubroutineType(types: !51)
// CHECK:STDOUT: !55 = distinct !DISubprogram(name: "MyF", linkageName: "_CMyF.Main", scope: null, file: !34, line: 6, type: !43, spFlags: DISPFlagDefinition, unit: !35)
// CHECK:STDOUT: !56 = !DILocation(line: 7, column: 19, scope: !55)
// CHECK:STDOUT: !57 = !DILocation(line: 7, column: 22, scope: !55)
// CHECK:STDOUT: !58 = !DILocation(line: 7, column: 3, scope: !55)
// CHECK:STDOUT: !59 = !{null, !39}
// CHECK:STDOUT: !60 = !DISubroutineType(types: !59)
// CHECK:STDOUT: !61 = !{null, !38}
// CHECK:STDOUT: !62 = !DISubroutineType(types: !61)
// CHECK:STDOUT: !63 = !DILocation(line: 9, column: 21, scope: !55)
// CHECK:STDOUT: !64 = !DILocation(line: 9, column: 24, scope: !55)
// CHECK:STDOUT: !65 = !DILocation(line: 9, column: 3, scope: !55)
// CHECK:STDOUT: !66 = !{null, !41}
// CHECK:STDOUT: !67 = !DISubroutineType(types: !66)
// CHECK:STDOUT: !68 = !{null, !10}
// CHECK:STDOUT: !69 = !DISubroutineType(types: !68)
// CHECK:STDOUT: !70 = !DILocation(line: 6, column: 1, scope: !55)
// CHECK:STDOUT: !71 = distinct !DISubprogram(name: "PassShort", linkageName: "_CPassShort.Main", scope: null, file: !34, line: 16, type: !52, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !86)
// CHECK:STDOUT: !72 = !DILocalVariable(arg: 1, scope: !71, type: !39)
// CHECK:STDOUT: !73 = !DILocalVariable(arg: 2, scope: !71, type: !44)
// CHECK:STDOUT: !76 = !DILocation(line: 17, column: 18, scope: !71)
// CHECK:STDOUT: !77 = !DILocation(line: 17, column: 3, scope: !71)
// CHECK:STDOUT: !79 = !DILocation(line: 18, column: 18, scope: !71)
// CHECK:STDOUT: !80 = !DILocation(line: 18, column: 3, scope: !71)
// CHECK:STDOUT: !81 = !DILocation(line: 19, column: 18, scope: !71)
// CHECK:STDOUT: !82 = !DILocation(line: 19, column: 3, scope: !71)
// CHECK:STDOUT: !83 = !DILocation(line: 20, column: 18, scope: !71)
// CHECK:STDOUT: !84 = !DILocation(line: 20, column: 3, scope: !71)
// CHECK:STDOUT: !85 = !DILocation(line: 16, column: 1, scope: !71)
// CHECK:STDOUT: !86 = !{!72, !73}
// CHECK:STDOUT: !87 = distinct !DISubprogram(name: "__global_init", linkageName: "_C__global_init.Main", scope: null, file: !34, type: !43, spFlags: DISPFlagDefinition, unit: !35)
// CHECK:STDOUT: !88 = !DILocation(line: 14, column: 1, scope: !87)
// CHECK:STDOUT: !89 = !DILocation(line: 0, scope: !87)
// CHECK:STDOUT: !90 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.e43967308dd20a9f", scope: null, file: !26, line: 32, type: !60, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !96)
// CHECK:STDOUT: !91 = !DILocalVariable(arg: 1, scope: !90, type: !39)
// CHECK:STDOUT: !95 = !DILocation(line: 32, column: 1, scope: !90)
// CHECK:STDOUT: !96 = !{!91}
// CHECK:STDOUT: !97 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.6fb12317fdf5c9b3", scope: null, file: !26, line: 32, type: !62, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !103)
// CHECK:STDOUT: !98 = !DILocalVariable(arg: 1, scope: !97, type: !38)
// CHECK:STDOUT: !102 = !DILocation(line: 32, column: 1, scope: !97)
// CHECK:STDOUT: !103 = !{!98}
// CHECK:STDOUT: !104 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.44e7310a0b394feb", scope: null, file: !26, line: 32, type: !67, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !110)
// CHECK:STDOUT: !105 = !DILocalVariable(arg: 1, scope: !104, type: !41)
// CHECK:STDOUT: !109 = !DILocation(line: 32, column: 1, scope: !104)
// CHECK:STDOUT: !110 = !{!105}
// CHECK:STDOUT: !111 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.73549b49ca05bf4c", scope: null, file: !26, line: 32, type: !69, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !117)
// CHECK:STDOUT: !112 = !DILocalVariable(arg: 1, scope: !111, type: !10)
// CHECK:STDOUT: !116 = !DILocation(line: 32, column: 1, scope: !111)
// CHECK:STDOUT: !117 = !{!112}
// CHECK:STDOUT: !118 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !119 = !{!"omnipotent char", !118, i64 0}
// CHECK:STDOUT: !120 = !{!"any pointer", !119, i64 0}
// CHECK:STDOUT: !121 = !{!"p1 omnipotent char", !120, i64 0}
// CHECK:STDOUT: !122 = !{!121, !121, i64 0}
// CHECK:STDOUT: !123 = !{!"short", !119, i64 0}
// CHECK:STDOUT: !124 = !{!"p1 short", !120, i64 0}
// CHECK:STDOUT: !125 = !{!124, !124, i64 0}
// CHECK:STDOUT: !126 = !{!"int", !119, i64 0}
// CHECK:STDOUT: !127 = !{!126, !126, i64 0}
// CHECK:STDOUT: !128 = !{!"long", !119, i64 0}
// CHECK:STDOUT: !129 = !{!128, !128, i64 0}
// CHECK:STDOUT: !130 = !{!119, !119, i64 0}
// CHECK:STDOUT: !131 = !{!123, !123, i64 0}
// CHECK:STDOUT: !132 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !133 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !134 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !135 = !{!"__libc_errno", !126, i64 0}
// CHECK:STDOUT: !136 = !{!135, !126, i64 0}
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
// CHECK:STDOUT: define void @_CTest.Main() #0 !dbg !141 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x.var = alloca [12 x i8], align 4, !dbg !142
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %x.var), !dbg !142
// CHECK:STDOUT:   call void @"_COp.51f3fb62cb986374:DefaultOrUnformed.Core.102476b7ae87d0a1"(ptr %x.var), !dbg !142
// CHECK:STDOUT:   %.loc8_4.a = getelementptr inbounds nuw [12 x i8], ptr %x.var, i32 0, i32 0, !dbg !144
// CHECK:STDOUT:   store i32 1, ptr %.loc8_4.a, align 4, !dbg !144
// CHECK:STDOUT:   %.loc9_4.b = getelementptr inbounds nuw [12 x i8], ptr %x.var, i32 0, i32 4, !dbg !145
// CHECK:STDOUT:   store i32 2, ptr %.loc9_4.b, align 4, !dbg !145
// CHECK:STDOUT:   %.loc10_4.c = getelementptr inbounds nuw [12 x i8], ptr %x.var, i32 0, i32 8, !dbg !146
// CHECK:STDOUT:   store i32 3, ptr %.loc10_4.c, align 4, !dbg !146
// CHECK:STDOUT:   call void @_Z11pass_struct1X.carbon_thunk._(ptr %x.var), !dbg !148
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.4e766327b9d07a26(ptr %x.var), !dbg !142
// CHECK:STDOUT:   ret void, !dbg !149
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN1XC1Ev.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !166
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !166
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:Default.eed9ddf3aaac414a.Core:X.Cpp"(ptr sret([12 x i8]) %return) #2 !dbg !151 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN1XC1Ev.carbon_thunk.(ptr %return), !dbg !152
// CHECK:STDOUT:   ret void, !dbg !152
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z11pass_struct1X.carbon_thunk._(ptr noundef %0) #3 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %struct.X, align 4
// CHECK:STDOUT:   %agg.tmp.coerce = alloca { i64, i32 }, align 4
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !166
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !166
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp, ptr align 4 %1, i64 12, i1 false), !tbaa.struct !168
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
// CHECK:STDOUT: define linkonce_odr void @"_COp.51f3fb62cb986374:DefaultOrUnformed.Core.102476b7ae87d0a1"(ptr sret([12 x i8]) %return) #0 !dbg !153 {
// CHECK:STDOUT:   call void @"_COp:thunk:Default.eed9ddf3aaac414a.Core:X.Cpp"(ptr %return), !dbg !156
// CHECK:STDOUT:   ret void, !dbg !157
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.4e766327b9d07a26(ptr %t) #0 !dbg !158 {
// CHECK:STDOUT:   ret void, !dbg !163
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!138}
// CHECK:STDOUT: !llvm.module.flags = !{!132, !133, !134, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!136}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !22 = !DIFile(filename: "min_prelude/parts/default.carbon", directory: "")
// CHECK:STDOUT: !26 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !42 = !{null}
// CHECK:STDOUT: !43 = !DISubroutineType(types: !42)
// CHECK:STDOUT: !44 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !53 = !{null, !44}
// CHECK:STDOUT: !54 = !DISubroutineType(types: !53)
// CHECK:STDOUT: !118 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !119 = !{!"omnipotent char", !118, i64 0}
// CHECK:STDOUT: !120 = !{!"any pointer", !119, i64 0}
// CHECK:STDOUT: !126 = !{!"int", !119, i64 0}
// CHECK:STDOUT: !127 = !{!126, !126, i64 0}
// CHECK:STDOUT: !132 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !133 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !134 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !135 = !{!"__libc_errno", !126, i64 0}
// CHECK:STDOUT: !136 = !{!135, !126, i64 0}
// CHECK:STDOUT: !137 = !DIFile(filename: "import_struct.carbon", directory: "")
// CHECK:STDOUT: !138 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !137, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !139 = !{!44}
// CHECK:STDOUT: !140 = !DISubroutineType(types: !139)
// CHECK:STDOUT: !141 = distinct !DISubprogram(name: "Test", linkageName: "_CTest.Main", scope: null, file: !137, line: 6, type: !43, spFlags: DISPFlagDefinition, unit: !138)
// CHECK:STDOUT: !142 = !DILocation(line: 7, column: 3, scope: !141)
// CHECK:STDOUT: !144 = !DILocation(line: 8, column: 3, scope: !141)
// CHECK:STDOUT: !145 = !DILocation(line: 9, column: 3, scope: !141)
// CHECK:STDOUT: !146 = !DILocation(line: 10, column: 3, scope: !141)
// CHECK:STDOUT: !148 = !DILocation(line: 11, column: 3, scope: !141)
// CHECK:STDOUT: !149 = !DILocation(line: 6, column: 1, scope: !141)
// CHECK:STDOUT: !150 = !DIFile(filename: "./struct.h", directory: "")
// CHECK:STDOUT: !151 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:Default.eed9ddf3aaac414a.Core:X.Cpp", scope: null, file: !150, line: 2, type: !140, spFlags: DISPFlagDefinition, unit: !138)
// CHECK:STDOUT: !152 = !DILocation(line: 2, column: 8, scope: !151)
// CHECK:STDOUT: !153 = distinct !DISubprogram(name: "Op", linkageName: "_COp.51f3fb62cb986374:DefaultOrUnformed.Core.102476b7ae87d0a1", scope: null, file: !22, line: 9, type: !140, spFlags: DISPFlagDefinition, unit: !138)
// CHECK:STDOUT: !156 = !DILocation(line: 9, column: 28, scope: !153)
// CHECK:STDOUT: !157 = !DILocation(line: 9, column: 21, scope: !153)
// CHECK:STDOUT: !158 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.4e766327b9d07a26", scope: null, file: !26, line: 32, type: !54, spFlags: DISPFlagDefinition, unit: !138, retainedNodes: !164)
// CHECK:STDOUT: !159 = !DILocalVariable(arg: 1, scope: !158, type: !44)
// CHECK:STDOUT: !163 = !DILocation(line: 32, column: 1, scope: !158)
// CHECK:STDOUT: !164 = !{!159}
// CHECK:STDOUT: !165 = !{!"p1 _ZTS1X", !120, i64 0}
// CHECK:STDOUT: !166 = !{!165, !165, i64 0}
// CHECK:STDOUT: !168 = !{i64 0, i64 4, !127, i64 4, i64 4, !127, i64 8, i64 4, !127}
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
// CHECK:STDOUT: define void @_CPassRefExpr.Main() #0 !dbg !171 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %y.var = alloca [12 x i8], align 4, !dbg !172
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %y.var), !dbg !172
// CHECK:STDOUT:   %.loc8_4.a = getelementptr inbounds nuw [12 x i8], ptr %y.var, i32 0, i32 0, !dbg !174
// CHECK:STDOUT:   store i32 1, ptr %.loc8_4.a, align 4, !dbg !174
// CHECK:STDOUT:   %.loc9_4.b = getelementptr inbounds nuw [12 x i8], ptr %y.var, i32 0, i32 4, !dbg !175
// CHECK:STDOUT:   store i32 2, ptr %.loc9_4.b, align 4, !dbg !175
// CHECK:STDOUT:   %.loc10_4.c = getelementptr inbounds nuw [12 x i8], ptr %y.var, i32 0, i32 8, !dbg !176
// CHECK:STDOUT:   store i32 3, ptr %.loc10_4.c, align 4, !dbg !176
// CHECK:STDOUT:   call void @_Z11pass_struct1Y.carbon_thunk._(ptr %y.var), !dbg !178
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.6b885745e47b1554(ptr %y.var), !dbg !172
// CHECK:STDOUT:   ret void, !dbg !179
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z11pass_struct1Y.carbon_thunk._(ptr noundef %0) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %class.Y, align 4
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !199
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !199
// CHECK:STDOUT:   call void @_ZN1YC1ERKS_(ptr noundef nonnull align 4 dereferenceable(12) %agg.tmp, ptr noundef nonnull align 4 dereferenceable(12) %1)
// CHECK:STDOUT:   call void @_Z11pass_struct1Y(ptr nofreeobj noundef align 4 dead_on_return dereferenceable(12) %agg.tmp)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CMake.Main(ptr sret([12 x i8]))
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassInitExpr.Main() #0 !dbg !180 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc17_24.1.temp = alloca [12 x i8], align 4, !dbg !181
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc17_24.1.temp), !dbg !181
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.loc17_24.1.temp), !dbg !181
// CHECK:STDOUT:   call void @_Z11pass_struct1Y.carbon_thunk._(ptr %.loc17_24.1.temp), !dbg !182
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.6b885745e47b1554(ptr %.loc17_24.1.temp), !dbg !181
// CHECK:STDOUT:   ret void, !dbg !183
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassValueExpr.Main(ptr %y) #0 !dbg !184 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z11pass_struct1Y.carbon_thunk._(ptr %y), !dbg !188
// CHECK:STDOUT:   ret void, !dbg !189
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.6b885745e47b1554(ptr %t) #0 !dbg !191 {
// CHECK:STDOUT:   ret void, !dbg !196
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z11pass_struct1Y(ptr nofreeobj noundef align 4 dead_on_return dereferenceable(12)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_ZN1YC1ERKS_(ptr noundef nonnull align 4 dereferenceable(12), ptr noundef nonnull align 4 dereferenceable(12)) unnamed_addr #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.6b885745e47b1554, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!170}
// CHECK:STDOUT: !llvm.module.flags = !{!132, !133, !134, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!136}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !26 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !42 = !{null}
// CHECK:STDOUT: !43 = !DISubroutineType(types: !42)
// CHECK:STDOUT: !44 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !53 = !{null, !44}
// CHECK:STDOUT: !54 = !DISubroutineType(types: !53)
// CHECK:STDOUT: !118 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !119 = !{!"omnipotent char", !118, i64 0}
// CHECK:STDOUT: !120 = !{!"any pointer", !119, i64 0}
// CHECK:STDOUT: !126 = !{!"int", !119, i64 0}
// CHECK:STDOUT: !132 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !133 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !134 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !135 = !{!"__libc_errno", !126, i64 0}
// CHECK:STDOUT: !136 = !{!135, !126, i64 0}
// CHECK:STDOUT: !169 = !DIFile(filename: "import_class_with_nontrivial_copy.carbon", directory: "")
// CHECK:STDOUT: !170 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !169, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !171 = distinct !DISubprogram(name: "PassRefExpr", linkageName: "_CPassRefExpr.Main", scope: null, file: !169, line: 6, type: !43, spFlags: DISPFlagDefinition, unit: !170)
// CHECK:STDOUT: !172 = !DILocation(line: 7, column: 3, scope: !171)
// CHECK:STDOUT: !174 = !DILocation(line: 8, column: 3, scope: !171)
// CHECK:STDOUT: !175 = !DILocation(line: 9, column: 3, scope: !171)
// CHECK:STDOUT: !176 = !DILocation(line: 10, column: 3, scope: !171)
// CHECK:STDOUT: !178 = !DILocation(line: 11, column: 3, scope: !171)
// CHECK:STDOUT: !179 = !DILocation(line: 6, column: 1, scope: !171)
// CHECK:STDOUT: !180 = distinct !DISubprogram(name: "PassInitExpr", linkageName: "_CPassInitExpr.Main", scope: null, file: !169, line: 16, type: !43, spFlags: DISPFlagDefinition, unit: !170)
// CHECK:STDOUT: !181 = !DILocation(line: 17, column: 19, scope: !180)
// CHECK:STDOUT: !182 = !DILocation(line: 17, column: 3, scope: !180)
// CHECK:STDOUT: !183 = !DILocation(line: 16, column: 1, scope: !180)
// CHECK:STDOUT: !184 = distinct !DISubprogram(name: "PassValueExpr", linkageName: "_CPassValueExpr.Main", scope: null, file: !169, line: 20, type: !54, spFlags: DISPFlagDefinition, unit: !170, retainedNodes: !190)
// CHECK:STDOUT: !185 = !DILocalVariable(arg: 1, scope: !184, type: !44)
// CHECK:STDOUT: !188 = !DILocation(line: 21, column: 3, scope: !184)
// CHECK:STDOUT: !189 = !DILocation(line: 20, column: 1, scope: !184)
// CHECK:STDOUT: !190 = !{!185}
// CHECK:STDOUT: !191 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.6b885745e47b1554", scope: null, file: !26, line: 32, type: !54, spFlags: DISPFlagDefinition, unit: !170, retainedNodes: !197)
// CHECK:STDOUT: !192 = !DILocalVariable(arg: 1, scope: !191, type: !44)
// CHECK:STDOUT: !196 = !DILocation(line: 32, column: 1, scope: !191)
// CHECK:STDOUT: !197 = !{!192}
// CHECK:STDOUT: !198 = !{!"p1 _ZTS1Y", !120, i64 0}
// CHECK:STDOUT: !199 = !{!198, !198, i64 0}
// CHECK:STDOUT:
