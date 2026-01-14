// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/parameters.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/parameters.carbon

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

// CHECK:STDOUT: ; ModuleID = 'import_ints.carbon'
// CHECK:STDOUT: source_filename = "import_ints.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: @_Cc.Main = global i16 0
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMyF.Main() #0 !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_19.3.temp = alloca i8, align 1, !dbg !15
// CHECK:STDOUT:   %.loc7_22.3.temp = alloca i16, align 2, !dbg !16
// CHECK:STDOUT:   %.loc9_21.3.temp = alloca i8, align 1, !dbg !17
// CHECK:STDOUT:   %.loc9_24.3.temp = alloca i16, align 2, !dbg !18
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_19.3.temp), !dbg !15
// CHECK:STDOUT:   store i8 1, ptr %.loc7_19.3.temp, align 1, !dbg !15
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_22.3.temp), !dbg !16
// CHECK:STDOUT:   store i16 2, ptr %.loc7_22.3.temp, align 2, !dbg !16
// CHECK:STDOUT:   call void @_Z11pass_signedasil.carbon_thunk(ptr %.loc7_19.3.temp, ptr %.loc7_22.3.temp, i32 3, i64 4), !dbg !19
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc9_21.3.temp), !dbg !17
// CHECK:STDOUT:   store i8 1, ptr %.loc9_21.3.temp, align 1, !dbg !17
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc9_24.3.temp), !dbg !18
// CHECK:STDOUT:   store i16 2, ptr %.loc9_24.3.temp, align 2, !dbg !18
// CHECK:STDOUT:   call void @_Z13pass_unsignedhtjm.carbon_thunk(ptr %.loc9_21.3.temp, ptr %.loc9_24.3.temp, i32 3, i64 4), !dbg !20
// CHECK:STDOUT:   ret void, !dbg !21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i16 @_CMakeShort.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassShort.Main(i16 %a, ptr %b) #0 !dbg !22 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc17_18.1.temp = alloca i16, align 2, !dbg !30
// CHECK:STDOUT:   %.loc18_18.3.temp = alloca i16, align 2, !dbg !31
// CHECK:STDOUT:   %.loc19_18.2.temp = alloca i16, align 2, !dbg !32
// CHECK:STDOUT:   %.loc20_28.3.temp = alloca i16, align 2, !dbg !33
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc17_18.1.temp), !dbg !30
// CHECK:STDOUT:   store i16 %a, ptr %.loc17_18.1.temp, align 2, !dbg !30
// CHECK:STDOUT:   call void @_Z10pass_shorts.carbon_thunk(ptr %.loc17_18.1.temp), !dbg !34
// CHECK:STDOUT:   %.loc18_18.2 = load i16, ptr %b, align 2, !dbg !31
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc18_18.3.temp), !dbg !31
// CHECK:STDOUT:   store i16 %.loc18_18.2, ptr %.loc18_18.3.temp, align 2, !dbg !31
// CHECK:STDOUT:   call void @_Z10pass_shorts.carbon_thunk(ptr %.loc18_18.3.temp), !dbg !35
// CHECK:STDOUT:   %.loc19_18.1 = load i16, ptr @_Cc.Main, align 2, !dbg !32
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc19_18.2.temp), !dbg !32
// CHECK:STDOUT:   store i16 %.loc19_18.1, ptr %.loc19_18.2.temp, align 2, !dbg !32
// CHECK:STDOUT:   call void @_Z10pass_shorts.carbon_thunk(ptr %.loc19_18.2.temp), !dbg !36
// CHECK:STDOUT:   %MakeShort.call = call i16 @_CMakeShort.Main(), !dbg !33
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc20_28.3.temp), !dbg !33
// CHECK:STDOUT:   store i16 %MakeShort.call, ptr %.loc20_28.3.temp, align 2, !dbg !33
// CHECK:STDOUT:   call void @_Z10pass_shorts.carbon_thunk(ptr %.loc20_28.3.temp), !dbg !37
// CHECK:STDOUT:   ret void, !dbg !38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define dso_local void @_Z11pass_signedasil.carbon_thunk(ptr noundef %0, ptr noundef %1, i32 noundef %2, i64 noundef %3) #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %.addr1 = alloca ptr, align 8
// CHECK:STDOUT:   %.addr2 = alloca i32, align 4
// CHECK:STDOUT:   %.addr3 = alloca i64, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !39
// CHECK:STDOUT:   store ptr %1, ptr %.addr1, align 8, !tbaa !42
// CHECK:STDOUT:   store i32 %2, ptr %.addr2, align 4, !tbaa !8
// CHECK:STDOUT:   store i64 %3, ptr %.addr3, align 8, !tbaa !44
// CHECK:STDOUT:   %4 = load ptr, ptr %.addr, align 8, !tbaa !39
// CHECK:STDOUT:   %5 = load i8, ptr %4, align 1, !tbaa !46
// CHECK:STDOUT:   %6 = load ptr, ptr %.addr1, align 8, !tbaa !42
// CHECK:STDOUT:   %7 = load i16, ptr %6, align 2, !tbaa !47
// CHECK:STDOUT:   %8 = load i32, ptr %.addr2, align 4, !tbaa !8
// CHECK:STDOUT:   %9 = load i64, ptr %.addr3, align 8, !tbaa !44
// CHECK:STDOUT:   call void @_Z11pass_signedasil(i8 noundef signext %5, i16 noundef signext %7, i32 noundef %8, i64 noundef %9)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z11pass_signedasil(i8 noundef signext, i16 noundef signext, i32 noundef, i64 noundef) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define dso_local void @_Z13pass_unsignedhtjm.carbon_thunk(ptr noundef %0, ptr noundef %1, i32 noundef %2, i64 noundef %3) #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %.addr1 = alloca ptr, align 8
// CHECK:STDOUT:   %.addr2 = alloca i32, align 4
// CHECK:STDOUT:   %.addr3 = alloca i64, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !39
// CHECK:STDOUT:   store ptr %1, ptr %.addr1, align 8, !tbaa !42
// CHECK:STDOUT:   store i32 %2, ptr %.addr2, align 4, !tbaa !8
// CHECK:STDOUT:   store i64 %3, ptr %.addr3, align 8, !tbaa !44
// CHECK:STDOUT:   %4 = load ptr, ptr %.addr, align 8, !tbaa !39
// CHECK:STDOUT:   %5 = load i8, ptr %4, align 1, !tbaa !46
// CHECK:STDOUT:   %6 = load ptr, ptr %.addr1, align 8, !tbaa !42
// CHECK:STDOUT:   %7 = load i16, ptr %6, align 2, !tbaa !47
// CHECK:STDOUT:   %8 = load i32, ptr %.addr2, align 4, !tbaa !8
// CHECK:STDOUT:   %9 = load i64, ptr %.addr3, align 8, !tbaa !44
// CHECK:STDOUT:   call void @_Z13pass_unsignedhtjm(i8 noundef zeroext %5, i16 noundef zeroext %7, i32 noundef %8, i64 noundef %9)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z13pass_unsignedhtjm(i8 noundef zeroext, i16 noundef zeroext, i32 noundef, i64 noundef) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define dso_local void @_Z10pass_shorts.carbon_thunk(ptr noundef %0) #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !42
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !42
// CHECK:STDOUT:   %2 = load i16, ptr %1, align 2, !tbaa !47
// CHECK:STDOUT:   call void @_Z10pass_shorts(i16 noundef signext %2)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z10pass_shorts(i16 noundef signext) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 7, 6, 5, 4, 3, 2, 1, 0 }
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
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "MyF", linkageName: "_CMyF.Main", scope: null, file: !7, line: 6, type: !13, spFlags: DISPFlagDefinition, unit: !6)
// CHECK:STDOUT: !13 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !14 = !{null}
// CHECK:STDOUT: !15 = !DILocation(line: 7, column: 19, scope: !12)
// CHECK:STDOUT: !16 = !DILocation(line: 7, column: 22, scope: !12)
// CHECK:STDOUT: !17 = !DILocation(line: 9, column: 21, scope: !12)
// CHECK:STDOUT: !18 = !DILocation(line: 9, column: 24, scope: !12)
// CHECK:STDOUT: !19 = !DILocation(line: 7, column: 3, scope: !12)
// CHECK:STDOUT: !20 = !DILocation(line: 9, column: 3, scope: !12)
// CHECK:STDOUT: !21 = !DILocation(line: 6, column: 1, scope: !12)
// CHECK:STDOUT: !22 = distinct !DISubprogram(name: "PassShort", linkageName: "_CPassShort.Main", scope: null, file: !7, line: 16, type: !23, spFlags: DISPFlagDefinition, unit: !6, retainedNodes: !27)
// CHECK:STDOUT: !23 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !24 = !{null, !25, !26}
// CHECK:STDOUT: !25 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_signed)
// CHECK:STDOUT: !26 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 8)
// CHECK:STDOUT: !27 = !{!28, !29}
// CHECK:STDOUT: !28 = !DILocalVariable(arg: 1, scope: !22, type: !25)
// CHECK:STDOUT: !29 = !DILocalVariable(arg: 2, scope: !22, type: !26)
// CHECK:STDOUT: !30 = !DILocation(line: 17, column: 18, scope: !22)
// CHECK:STDOUT: !31 = !DILocation(line: 18, column: 18, scope: !22)
// CHECK:STDOUT: !32 = !DILocation(line: 19, column: 18, scope: !22)
// CHECK:STDOUT: !33 = !DILocation(line: 20, column: 18, scope: !22)
// CHECK:STDOUT: !34 = !DILocation(line: 17, column: 3, scope: !22)
// CHECK:STDOUT: !35 = !DILocation(line: 18, column: 3, scope: !22)
// CHECK:STDOUT: !36 = !DILocation(line: 19, column: 3, scope: !22)
// CHECK:STDOUT: !37 = !DILocation(line: 20, column: 3, scope: !22)
// CHECK:STDOUT: !38 = !DILocation(line: 16, column: 1, scope: !22)
// CHECK:STDOUT: !39 = !{!40, !40, i64 0}
// CHECK:STDOUT: !40 = !{!"p1 omnipotent char", !41, i64 0}
// CHECK:STDOUT: !41 = !{!"any pointer", !10, i64 0}
// CHECK:STDOUT: !42 = !{!43, !43, i64 0}
// CHECK:STDOUT: !43 = !{!"p1 short", !41, i64 0}
// CHECK:STDOUT: !44 = !{!45, !45, i64 0}
// CHECK:STDOUT: !45 = !{!"long", !10, i64 0}
// CHECK:STDOUT: !46 = !{!10, !10, i64 0}
// CHECK:STDOUT: !47 = !{!48, !48, i64 0}
// CHECK:STDOUT: !48 = !{!"short", !10, i64 0}
// CHECK:STDOUT: ; ModuleID = 'import_struct.carbon'
// CHECK:STDOUT: source_filename = "import_struct.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %struct.X = type { i32, i32, i32 }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTest.Main() #0 !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x.var = alloca [12 x i8], align 1, !dbg !15
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %x.var), !dbg !15
// CHECK:STDOUT:   %.loc8_4.a = getelementptr inbounds nuw [12 x i8], ptr %x.var, i32 0, i32 0, !dbg !16
// CHECK:STDOUT:   store i32 1, ptr %.loc8_4.a, align 4, !dbg !16
// CHECK:STDOUT:   %.loc9_4.b = getelementptr inbounds nuw [12 x i8], ptr %x.var, i32 0, i32 4, !dbg !17
// CHECK:STDOUT:   store i32 2, ptr %.loc9_4.b, align 4, !dbg !17
// CHECK:STDOUT:   %.loc10_4.c = getelementptr inbounds nuw [12 x i8], ptr %x.var, i32 0, i32 8, !dbg !18
// CHECK:STDOUT:   store i32 3, ptr %.loc10_4.c, align 4, !dbg !18
// CHECK:STDOUT:   call void @_Z11pass_struct1X.carbon_thunk(ptr %x.var), !dbg !19
// CHECK:STDOUT:   ret void, !dbg !20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define dso_local void @_Z11pass_struct1X.carbon_thunk(ptr noundef %0) #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %struct.X, align 4
// CHECK:STDOUT:   %agg.tmp.coerce = alloca { i64, i32 }, align 4
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !21
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !21
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp, ptr align 4 %1, i64 12, i1 false), !tbaa.struct !24
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp.coerce, ptr align 4 %agg.tmp, i64 12, i1 false)
// CHECK:STDOUT:   %2 = getelementptr inbounds nuw { i64, i32 }, ptr %agg.tmp.coerce, i32 0, i32 0
// CHECK:STDOUT:   %3 = load i64, ptr %2, align 4
// CHECK:STDOUT:   %4 = getelementptr inbounds nuw { i64, i32 }, ptr %agg.tmp.coerce, i32 0, i32 1
// CHECK:STDOUT:   %5 = load i32, ptr %4, align 4
// CHECK:STDOUT:   call void @_Z11pass_struct1X(i64 %3, i32 %5)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #3
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z11pass_struct1X(i64, i32) #4
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #2 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #4 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
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
// CHECK:STDOUT: !7 = !DIFile(filename: "import_struct.carbon", directory: "")
// CHECK:STDOUT: !8 = !{!9, !9, i64 0}
// CHECK:STDOUT: !9 = !{!"int", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"omnipotent char", !11, i64 0}
// CHECK:STDOUT: !11 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "Test", linkageName: "_CTest.Main", scope: null, file: !7, line: 6, type: !13, spFlags: DISPFlagDefinition, unit: !6)
// CHECK:STDOUT: !13 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !14 = !{null}
// CHECK:STDOUT: !15 = !DILocation(line: 7, column: 3, scope: !12)
// CHECK:STDOUT: !16 = !DILocation(line: 8, column: 3, scope: !12)
// CHECK:STDOUT: !17 = !DILocation(line: 9, column: 3, scope: !12)
// CHECK:STDOUT: !18 = !DILocation(line: 10, column: 3, scope: !12)
// CHECK:STDOUT: !19 = !DILocation(line: 11, column: 3, scope: !12)
// CHECK:STDOUT: !20 = !DILocation(line: 6, column: 1, scope: !12)
// CHECK:STDOUT: !21 = !{!22, !22, i64 0}
// CHECK:STDOUT: !22 = !{!"p1 _ZTS1X", !23, i64 0}
// CHECK:STDOUT: !23 = !{!"any pointer", !10, i64 0}
// CHECK:STDOUT: !24 = !{i64 0, i64 4, !8, i64 4, i64 4, !8, i64 8, i64 4, !8}
// CHECK:STDOUT: ; ModuleID = 'import_class_with_nontrivial_copy.carbon'
// CHECK:STDOUT: source_filename = "import_class_with_nontrivial_copy.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %class.Y = type { i32, i32, i32 }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassRefExpr.Main() #0 !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %y.var = alloca [12 x i8], align 1, !dbg !15
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %y.var), !dbg !15
// CHECK:STDOUT:   %.loc8_4.a = getelementptr inbounds nuw [12 x i8], ptr %y.var, i32 0, i32 0, !dbg !16
// CHECK:STDOUT:   store i32 1, ptr %.loc8_4.a, align 4, !dbg !16
// CHECK:STDOUT:   %.loc9_4.b = getelementptr inbounds nuw [12 x i8], ptr %y.var, i32 0, i32 4, !dbg !17
// CHECK:STDOUT:   store i32 2, ptr %.loc9_4.b, align 4, !dbg !17
// CHECK:STDOUT:   %.loc10_4.c = getelementptr inbounds nuw [12 x i8], ptr %y.var, i32 0, i32 8, !dbg !18
// CHECK:STDOUT:   store i32 3, ptr %.loc10_4.c, align 4, !dbg !18
// CHECK:STDOUT:   call void @_Z11pass_struct1Y.carbon_thunk(ptr %y.var), !dbg !19
// CHECK:STDOUT:   ret void, !dbg !20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CMake.Main(ptr sret([12 x i8]))
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassInitExpr.Main() #0 !dbg !21 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc17_24.1.temp = alloca [12 x i8], align 1, !dbg !22
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc17_24.1.temp), !dbg !22
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.loc17_24.1.temp), !dbg !22
// CHECK:STDOUT:   call void @_Z11pass_struct1Y.carbon_thunk(ptr %.loc17_24.1.temp), !dbg !23
// CHECK:STDOUT:   ret void, !dbg !24
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassValueExpr.Main(ptr %y) #0 !dbg !25 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z11pass_struct1Y.carbon_thunk(ptr %y), !dbg !31
// CHECK:STDOUT:   ret void, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define dso_local void @_Z11pass_struct1Y.carbon_thunk(ptr noundef %0) #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %class.Y, align 4
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !33
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !33
// CHECK:STDOUT:   call void @_ZN1YC1ERKS_(ptr noundef nonnull align 4 dereferenceable(12) %agg.tmp, ptr noundef nonnull align 4 dereferenceable(12) %1)
// CHECK:STDOUT:   call void @_Z11pass_struct1Y(ptr dead_on_return noundef %agg.tmp)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_ZN1YC1ERKS_(ptr noundef nonnull align 4 dereferenceable(12), ptr noundef nonnull align 4 dereferenceable(12)) unnamed_addr #3
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z11pass_struct1Y(ptr dead_on_return noundef) #3
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
// CHECK:STDOUT: !7 = !DIFile(filename: "import_class_with_nontrivial_copy.carbon", directory: "")
// CHECK:STDOUT: !8 = !{!9, !9, i64 0}
// CHECK:STDOUT: !9 = !{!"int", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"omnipotent char", !11, i64 0}
// CHECK:STDOUT: !11 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "PassRefExpr", linkageName: "_CPassRefExpr.Main", scope: null, file: !7, line: 6, type: !13, spFlags: DISPFlagDefinition, unit: !6)
// CHECK:STDOUT: !13 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !14 = !{null}
// CHECK:STDOUT: !15 = !DILocation(line: 7, column: 3, scope: !12)
// CHECK:STDOUT: !16 = !DILocation(line: 8, column: 3, scope: !12)
// CHECK:STDOUT: !17 = !DILocation(line: 9, column: 3, scope: !12)
// CHECK:STDOUT: !18 = !DILocation(line: 10, column: 3, scope: !12)
// CHECK:STDOUT: !19 = !DILocation(line: 11, column: 3, scope: !12)
// CHECK:STDOUT: !20 = !DILocation(line: 6, column: 1, scope: !12)
// CHECK:STDOUT: !21 = distinct !DISubprogram(name: "PassInitExpr", linkageName: "_CPassInitExpr.Main", scope: null, file: !7, line: 16, type: !13, spFlags: DISPFlagDefinition, unit: !6)
// CHECK:STDOUT: !22 = !DILocation(line: 17, column: 19, scope: !21)
// CHECK:STDOUT: !23 = !DILocation(line: 17, column: 3, scope: !21)
// CHECK:STDOUT: !24 = !DILocation(line: 16, column: 1, scope: !21)
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "PassValueExpr", linkageName: "_CPassValueExpr.Main", scope: null, file: !7, line: 20, type: !26, spFlags: DISPFlagDefinition, unit: !6, retainedNodes: !29)
// CHECK:STDOUT: !26 = !DISubroutineType(types: !27)
// CHECK:STDOUT: !27 = !{null, !28}
// CHECK:STDOUT: !28 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 8)
// CHECK:STDOUT: !29 = !{!30}
// CHECK:STDOUT: !30 = !DILocalVariable(arg: 1, scope: !25, type: !28)
// CHECK:STDOUT: !31 = !DILocation(line: 21, column: 3, scope: !25)
// CHECK:STDOUT: !32 = !DILocation(line: 20, column: 1, scope: !25)
// CHECK:STDOUT: !33 = !{!34, !34, i64 0}
// CHECK:STDOUT: !34 = !{!"p1 _ZTS1Y", !35, i64 0}
// CHECK:STDOUT: !35 = !{!"any pointer", !10, i64 0}
