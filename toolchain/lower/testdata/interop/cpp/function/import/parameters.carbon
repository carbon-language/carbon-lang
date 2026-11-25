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
// CHECK:STDOUT:   %.loc9_21.3.temp = alloca i8, align 1, !dbg !59
// CHECK:STDOUT:   %.loc9_24.3.temp = alloca i16, align 2, !dbg !60
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_19.3.temp), !dbg !56
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_22.3.temp), !dbg !57
// CHECK:STDOUT:   call void @_Z11pass_signedasil.carbon_thunk.____(ptr @int_1.2eb, ptr @int_2.fee, i32 3, i64 4), !dbg !58
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc9_21.3.temp), !dbg !59
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc9_24.3.temp), !dbg !60
// CHECK:STDOUT:   call void @_Z13pass_unsignedhtjm.carbon_thunk.____(ptr @int_1.5dc, ptr @int_2.2d4, i32 3, i64 4), !dbg !61
// CHECK:STDOUT:   ret void, !dbg !62
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z11pass_signedasil.carbon_thunk.____(ptr noundef %0, ptr noundef %1, i32 noundef %2, i64 noundef %3) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %.addr1 = alloca ptr, align 8
// CHECK:STDOUT:   %.addr2 = alloca i32, align 4
// CHECK:STDOUT:   %.addr3 = alloca i64, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !86
// CHECK:STDOUT:   store ptr %1, ptr %.addr1, align 8, !tbaa !89
// CHECK:STDOUT:   store i32 %2, ptr %.addr2, align 4, !tbaa !91
// CHECK:STDOUT:   store i64 %3, ptr %.addr3, align 8, !tbaa !93
// CHECK:STDOUT:   %4 = load ptr, ptr %.addr, align 8, !tbaa !86
// CHECK:STDOUT:   %5 = load i8, ptr %4, align 1, !tbaa !94
// CHECK:STDOUT:   %6 = load ptr, ptr %.addr1, align 8, !tbaa !89
// CHECK:STDOUT:   %7 = load i16, ptr %6, align 2, !tbaa !95
// CHECK:STDOUT:   %8 = load i32, ptr %.addr2, align 4, !tbaa !91
// CHECK:STDOUT:   %9 = load i64, ptr %.addr3, align 8, !tbaa !93
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
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !86
// CHECK:STDOUT:   store ptr %1, ptr %.addr1, align 8, !tbaa !89
// CHECK:STDOUT:   store i32 %2, ptr %.addr2, align 4, !tbaa !91
// CHECK:STDOUT:   store i64 %3, ptr %.addr3, align 8, !tbaa !93
// CHECK:STDOUT:   %4 = load ptr, ptr %.addr, align 8, !tbaa !86
// CHECK:STDOUT:   %5 = load i8, ptr %4, align 1, !tbaa !94
// CHECK:STDOUT:   %6 = load ptr, ptr %.addr1, align 8, !tbaa !89
// CHECK:STDOUT:   %7 = load i16, ptr %6, align 2, !tbaa !95
// CHECK:STDOUT:   %8 = load i32, ptr %.addr2, align 4, !tbaa !91
// CHECK:STDOUT:   %9 = load i64, ptr %.addr3, align 8, !tbaa !93
// CHECK:STDOUT:   call void @_Z13pass_unsignedhtjm(i8 noundef zeroext %5, i16 noundef zeroext %7, i32 noundef %8, i64 noundef %9)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i16 @_CMakeShort.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassShort.Main(i16 %a, ptr %b) #0 !dbg !63 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc17_18.1.temp = alloca i16, align 2, !dbg !68
// CHECK:STDOUT:   %.loc18_18.3.temp = alloca i16, align 2, !dbg !71
// CHECK:STDOUT:   %.loc19_18.2.temp = alloca i16, align 2, !dbg !73
// CHECK:STDOUT:   %.loc20_28.3.temp = alloca i16, align 2, !dbg !75
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc17_18.1.temp), !dbg !68
// CHECK:STDOUT:   store i16 %a, ptr %.loc17_18.1.temp, align 2, !dbg !68
// CHECK:STDOUT:   call void @_Z10pass_shorts.carbon_thunk._(ptr %.loc17_18.1.temp), !dbg !69
// CHECK:STDOUT:   %.loc18_18.2 = load i16, ptr %b, align 2, !dbg !71
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc18_18.3.temp), !dbg !71
// CHECK:STDOUT:   store i16 %.loc18_18.2, ptr %.loc18_18.3.temp, align 2, !dbg !71
// CHECK:STDOUT:   call void @_Z10pass_shorts.carbon_thunk._(ptr %.loc18_18.3.temp), !dbg !72
// CHECK:STDOUT:   %.loc19_18.1 = load i16, ptr @_Cc.Main, align 2, !dbg !73
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc19_18.2.temp), !dbg !73
// CHECK:STDOUT:   store i16 %.loc19_18.1, ptr %.loc19_18.2.temp, align 2, !dbg !73
// CHECK:STDOUT:   call void @_Z10pass_shorts.carbon_thunk._(ptr %.loc19_18.2.temp), !dbg !74
// CHECK:STDOUT:   %MakeShort.call = call i16 @_CMakeShort.Main(), !dbg !75
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc20_28.3.temp), !dbg !75
// CHECK:STDOUT:   store i16 %MakeShort.call, ptr %.loc20_28.3.temp, align 2, !dbg !75
// CHECK:STDOUT:   call void @_Z10pass_shorts.carbon_thunk._(ptr %.loc20_28.3.temp), !dbg !76
// CHECK:STDOUT:   ret void, !dbg !77
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z10pass_shorts.carbon_thunk._(ptr noundef %0) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !89
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !89
// CHECK:STDOUT:   %2 = load i16, ptr %1, align 2, !tbaa !95
// CHECK:STDOUT:   call void @_Z10pass_shorts(i16 noundef signext %2)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define internal void @_C__global_init.Main() #0 !dbg !79 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   store i16 poison, ptr @_Cc.Main, align 2, !dbg !80
// CHECK:STDOUT:   ret void, !dbg !81
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z11pass_signedasil(i8 noundef, i16 noundef, i32 noundef, i64 noundef) #3
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z13pass_unsignedhtjm(i8 noundef, i16 noundef, i32 noundef, i64 noundef) #3
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z10pass_shorts(i16 noundef) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_Z10pass_shorts.carbon_thunk._, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 7, 6, 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!35}
// CHECK:STDOUT: !llvm.module.flags = !{!96, !97, !98, !99, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!101}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !34 = !DIFile(filename: "import_ints.carbon", directory: "")
// CHECK:STDOUT: !35 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !34, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !39 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_signed)
// CHECK:STDOUT: !42 = !{null}
// CHECK:STDOUT: !43 = !DISubroutineType(types: !42)
// CHECK:STDOUT: !44 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !51 = !{null, !39, !44}
// CHECK:STDOUT: !52 = !DISubroutineType(types: !51)
// CHECK:STDOUT: !55 = distinct !DISubprogram(name: "MyF", linkageName: "_CMyF.Main", scope: null, file: !34, line: 6, type: !43, spFlags: DISPFlagDefinition, unit: !35)
// CHECK:STDOUT: !56 = !DILocation(line: 7, column: 19, scope: !55)
// CHECK:STDOUT: !57 = !DILocation(line: 7, column: 22, scope: !55)
// CHECK:STDOUT: !58 = !DILocation(line: 7, column: 3, scope: !55)
// CHECK:STDOUT: !59 = !DILocation(line: 9, column: 21, scope: !55)
// CHECK:STDOUT: !60 = !DILocation(line: 9, column: 24, scope: !55)
// CHECK:STDOUT: !61 = !DILocation(line: 9, column: 3, scope: !55)
// CHECK:STDOUT: !62 = !DILocation(line: 6, column: 1, scope: !55)
// CHECK:STDOUT: !63 = distinct !DISubprogram(name: "PassShort", linkageName: "_CPassShort.Main", scope: null, file: !34, line: 16, type: !52, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !78)
// CHECK:STDOUT: !64 = !DILocalVariable(arg: 1, scope: !63, type: !39)
// CHECK:STDOUT: !65 = !DILocalVariable(arg: 2, scope: !63, type: !44)
// CHECK:STDOUT: !68 = !DILocation(line: 17, column: 18, scope: !63)
// CHECK:STDOUT: !69 = !DILocation(line: 17, column: 3, scope: !63)
// CHECK:STDOUT: !71 = !DILocation(line: 18, column: 18, scope: !63)
// CHECK:STDOUT: !72 = !DILocation(line: 18, column: 3, scope: !63)
// CHECK:STDOUT: !73 = !DILocation(line: 19, column: 18, scope: !63)
// CHECK:STDOUT: !74 = !DILocation(line: 19, column: 3, scope: !63)
// CHECK:STDOUT: !75 = !DILocation(line: 20, column: 18, scope: !63)
// CHECK:STDOUT: !76 = !DILocation(line: 20, column: 3, scope: !63)
// CHECK:STDOUT: !77 = !DILocation(line: 16, column: 1, scope: !63)
// CHECK:STDOUT: !78 = !{!64, !65}
// CHECK:STDOUT: !79 = distinct !DISubprogram(name: "__global_init", linkageName: "_C__global_init.Main", scope: null, file: !34, type: !43, spFlags: DISPFlagDefinition, unit: !35)
// CHECK:STDOUT: !80 = !DILocation(line: 14, column: 1, scope: !79)
// CHECK:STDOUT: !81 = !DILocation(line: 0, scope: !79)
// CHECK:STDOUT: !82 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !83 = !{!"omnipotent char", !82, i64 0}
// CHECK:STDOUT: !84 = !{!"any pointer", !83, i64 0}
// CHECK:STDOUT: !85 = !{!"p1 omnipotent char", !84, i64 0}
// CHECK:STDOUT: !86 = !{!85, !85, i64 0}
// CHECK:STDOUT: !87 = !{!"short", !83, i64 0}
// CHECK:STDOUT: !88 = !{!"p1 short", !84, i64 0}
// CHECK:STDOUT: !89 = !{!88, !88, i64 0}
// CHECK:STDOUT: !90 = !{!"int", !83, i64 0}
// CHECK:STDOUT: !91 = !{!90, !90, i64 0}
// CHECK:STDOUT: !92 = !{!"long", !83, i64 0}
// CHECK:STDOUT: !93 = !{!92, !92, i64 0}
// CHECK:STDOUT: !94 = !{!83, !83, i64 0}
// CHECK:STDOUT: !95 = !{!87, !87, i64 0}
// CHECK:STDOUT: !96 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !97 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !98 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !99 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !100 = !{!"__libc_errno", !90, i64 0}
// CHECK:STDOUT: !101 = !{!100, !90, i64 0}
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
// CHECK:STDOUT: define void @_CTest.Main() #0 !dbg !106 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x.var = alloca [12 x i8], align 4, !dbg !143
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %x.var), !dbg !143
// CHECK:STDOUT:   call void @"_COp.51f3fb62cb986374:DefaultOrUnformed.Core.102476b7ae87d0a1"(ptr %x.var), !dbg !143
// CHECK:STDOUT:   %.loc8_4.a = getelementptr inbounds nuw i8, ptr %x.var, i64 0, !dbg !145
// CHECK:STDOUT:   store i32 1, ptr %.loc8_4.a, align 4, !dbg !145
// CHECK:STDOUT:   %.loc9_4.b = getelementptr inbounds nuw i8, ptr %x.var, i64 4, !dbg !146
// CHECK:STDOUT:   store i32 2, ptr %.loc9_4.b, align 4, !dbg !146
// CHECK:STDOUT:   %.loc10_4.c = getelementptr inbounds nuw i8, ptr %x.var, i64 8, !dbg !147
// CHECK:STDOUT:   store i32 3, ptr %.loc10_4.c, align 4, !dbg !147
// CHECK:STDOUT:   call void @_Z11pass_struct1X.carbon_thunk._(ptr %x.var), !dbg !149
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.2d682b12c0c20e08(ptr %x.var), !dbg !143
// CHECK:STDOUT:   ret void, !dbg !150
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN1XC1Ev.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !124
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !124
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:Default.eed9ddf3aaac414a.Core:X.Cpp"(ptr sret([12 x i8]) %return) #2 !dbg !116 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN1XC1Ev.carbon_thunk.(ptr %return), !dbg !117
// CHECK:STDOUT:   ret void, !dbg !117
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z11pass_struct1X.carbon_thunk._(ptr noundef %0) #3 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %struct.X, align 4
// CHECK:STDOUT:   %agg.tmp.coerce = alloca { i64, i32 }, align 4
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !124
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !124
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp, ptr align 4 %1, i64 12, i1 false), !tbaa.struct !126
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
// CHECK:STDOUT: define linkonce_odr void @"_COp.51f3fb62cb986374:DefaultOrUnformed.Core.102476b7ae87d0a1"(ptr sret([12 x i8]) %return) #0 !dbg !118 {
// CHECK:STDOUT:   call void @"_COp:thunk:Default.eed9ddf3aaac414a.Core:X.Cpp"(ptr %return), !dbg !121
// CHECK:STDOUT:   ret void, !dbg !122
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!103}
// CHECK:STDOUT: !llvm.module.flags = !{!96, !97, !98, !99, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!101}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !22 = !DIFile(filename: "min_prelude/parts/default.carbon", directory: "")
// CHECK:STDOUT: !42 = !{null}
// CHECK:STDOUT: !43 = !DISubroutineType(types: !42)
// CHECK:STDOUT: !44 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !82 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !83 = !{!"omnipotent char", !82, i64 0}
// CHECK:STDOUT: !84 = !{!"any pointer", !83, i64 0}
// CHECK:STDOUT: !90 = !{!"int", !83, i64 0}
// CHECK:STDOUT: !91 = !{!90, !90, i64 0}
// CHECK:STDOUT: !96 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !97 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !98 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !99 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !100 = !{!"__libc_errno", !90, i64 0}
// CHECK:STDOUT: !101 = !{!100, !90, i64 0}
// CHECK:STDOUT: !102 = !DIFile(filename: "import_struct.carbon", directory: "")
// CHECK:STDOUT: !103 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !102, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !104 = !{!44}
// CHECK:STDOUT: !105 = !DISubroutineType(types: !104)
// CHECK:STDOUT: !106 = distinct !DISubprogram(name: "Test", linkageName: "_CTest.Main", scope: null, file: !102, line: 6, type: !43, spFlags: DISPFlagDefinition, unit: !103)
// CHECK:STDOUT: !107 = !DILocation(line: 7, column: 3, scope: !106)
// CHECK:STDOUT: !109 = !DILocation(line: 8, column: 3, scope: !106)
// CHECK:STDOUT: !110 = !DILocation(line: 9, column: 3, scope: !106)
// CHECK:STDOUT: !111 = !DILocation(line: 10, column: 3, scope: !106)
// CHECK:STDOUT: !113 = !DILocation(line: 11, column: 3, scope: !106)
// CHECK:STDOUT: !114 = !DILocation(line: 6, column: 1, scope: !106)
// CHECK:STDOUT: !115 = !DIFile(filename: "./struct.h", directory: "")
// CHECK:STDOUT: !116 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:Default.eed9ddf3aaac414a.Core:X.Cpp", scope: null, file: !115, line: 2, type: !105, spFlags: DISPFlagDefinition, unit: !103)
// CHECK:STDOUT: !117 = !DILocation(line: 2, column: 8, scope: !116)
// CHECK:STDOUT: !118 = distinct !DISubprogram(name: "Op", linkageName: "_COp.51f3fb62cb986374:DefaultOrUnformed.Core.102476b7ae87d0a1", scope: null, file: !22, line: 9, type: !105, spFlags: DISPFlagDefinition, unit: !103)
// CHECK:STDOUT: !121 = !DILocation(line: 9, column: 28, scope: !118)
// CHECK:STDOUT: !122 = !DILocation(line: 9, column: 21, scope: !118)
// CHECK:STDOUT: !123 = !{!"p1 _ZTS1X", !84, i64 0}
// CHECK:STDOUT: !124 = !{!123, !123, i64 0}
// CHECK:STDOUT: !126 = !{i64 0, i64 4, !91, i64 4, i64 4, !91, i64 8, i64 4, !91}
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
// CHECK:STDOUT: define void @_CPassRefExpr.Main() #0 !dbg !129 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %y.var = alloca [12 x i8], align 4, !dbg !173
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %y.var), !dbg !173
// CHECK:STDOUT:   %.loc8_4.a = getelementptr inbounds nuw i8, ptr %y.var, i64 0, !dbg !175
// CHECK:STDOUT:   store i32 1, ptr %.loc8_4.a, align 4, !dbg !175
// CHECK:STDOUT:   %.loc9_4.b = getelementptr inbounds nuw i8, ptr %y.var, i64 4, !dbg !176
// CHECK:STDOUT:   store i32 2, ptr %.loc9_4.b, align 4, !dbg !176
// CHECK:STDOUT:   %.loc10_4.c = getelementptr inbounds nuw i8, ptr %y.var, i64 8, !dbg !177
// CHECK:STDOUT:   store i32 3, ptr %.loc10_4.c, align 4, !dbg !177
// CHECK:STDOUT:   call void @_Z11pass_struct1Y.carbon_thunk._(ptr %y.var), !dbg !179
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.5d1f4d5634dc3faa(ptr %y.var), !dbg !173
// CHECK:STDOUT:   ret void, !dbg !180
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z11pass_struct1Y.carbon_thunk._(ptr noundef %0) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %class.Y, align 4
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !150
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !150
// CHECK:STDOUT:   call void @_ZN1YC1ERKS_(ptr noundef nonnull align 4 dereferenceable(12) %agg.tmp, ptr noundef nonnull align 4 dereferenceable(12) %1)
// CHECK:STDOUT:   call void @_Z11pass_struct1Y(ptr nofreeobj noundef align 4 dead_on_return dereferenceable(12) %agg.tmp)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CMake.Main(ptr sret([12 x i8]))
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassInitExpr.Main() #0 !dbg !138 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc17_24.1.temp = alloca [12 x i8], align 4, !dbg !139
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc17_24.1.temp), !dbg !139
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.loc17_24.1.temp), !dbg !139
// CHECK:STDOUT:   call void @_Z11pass_struct1Y.carbon_thunk._(ptr %.loc17_24.1.temp), !dbg !140
// CHECK:STDOUT:   ret void, !dbg !141
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassValueExpr.Main(ptr %y) #0 !dbg !142 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z11pass_struct1Y.carbon_thunk._(ptr %y), !dbg !146
// CHECK:STDOUT:   ret void, !dbg !147
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z11pass_struct1Y(ptr nofreeobj noundef align 4 dead_on_return dereferenceable(12)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_ZN1YC1ERKS_(ptr noundef nonnull align 4 dereferenceable(12), ptr noundef nonnull align 4 dereferenceable(12)) unnamed_addr #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!128}
// CHECK:STDOUT: !llvm.module.flags = !{!96, !97, !98, !99, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!101}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !42 = !{null}
// CHECK:STDOUT: !43 = !DISubroutineType(types: !42)
// CHECK:STDOUT: !44 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !53 = !{null, !44}
// CHECK:STDOUT: !54 = !DISubroutineType(types: !53)
// CHECK:STDOUT: !82 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !83 = !{!"omnipotent char", !82, i64 0}
// CHECK:STDOUT: !84 = !{!"any pointer", !83, i64 0}
// CHECK:STDOUT: !90 = !{!"int", !83, i64 0}
// CHECK:STDOUT: !96 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !97 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !98 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !99 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !100 = !{!"__libc_errno", !90, i64 0}
// CHECK:STDOUT: !101 = !{!100, !90, i64 0}
// CHECK:STDOUT: !127 = !DIFile(filename: "import_class_with_nontrivial_copy.carbon", directory: "")
// CHECK:STDOUT: !128 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !127, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !129 = distinct !DISubprogram(name: "PassRefExpr", linkageName: "_CPassRefExpr.Main", scope: null, file: !127, line: 6, type: !43, spFlags: DISPFlagDefinition, unit: !128)
// CHECK:STDOUT: !130 = !DILocation(line: 7, column: 3, scope: !129)
// CHECK:STDOUT: !132 = !DILocation(line: 8, column: 3, scope: !129)
// CHECK:STDOUT: !133 = !DILocation(line: 9, column: 3, scope: !129)
// CHECK:STDOUT: !134 = !DILocation(line: 10, column: 3, scope: !129)
// CHECK:STDOUT: !136 = !DILocation(line: 11, column: 3, scope: !129)
// CHECK:STDOUT: !137 = !DILocation(line: 6, column: 1, scope: !129)
// CHECK:STDOUT: !138 = distinct !DISubprogram(name: "PassInitExpr", linkageName: "_CPassInitExpr.Main", scope: null, file: !127, line: 16, type: !43, spFlags: DISPFlagDefinition, unit: !128)
// CHECK:STDOUT: !139 = !DILocation(line: 17, column: 19, scope: !138)
// CHECK:STDOUT: !140 = !DILocation(line: 17, column: 3, scope: !138)
// CHECK:STDOUT: !141 = !DILocation(line: 16, column: 1, scope: !138)
// CHECK:STDOUT: !142 = distinct !DISubprogram(name: "PassValueExpr", linkageName: "_CPassValueExpr.Main", scope: null, file: !127, line: 20, type: !54, spFlags: DISPFlagDefinition, unit: !128, retainedNodes: !148)
// CHECK:STDOUT: !143 = !DILocalVariable(arg: 1, scope: !142, type: !44)
// CHECK:STDOUT: !146 = !DILocation(line: 21, column: 3, scope: !142)
// CHECK:STDOUT: !147 = !DILocation(line: 20, column: 1, scope: !142)
// CHECK:STDOUT: !148 = !{!143}
// CHECK:STDOUT: !149 = !{!"p1 _ZTS1Y", !84, i64 0}
// CHECK:STDOUT: !150 = !{!149, !149, i64 0}
// CHECK:STDOUT:
