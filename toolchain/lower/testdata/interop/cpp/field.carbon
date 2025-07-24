// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/field.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/field.carbon

// --- struct.h

struct A {
  int n;
  alignas(16) int m;
};

// --- access_struct.carbon

library "[[@TEST_NAME]]";

import Cpp library "struct.h";

// TODO: We generate 4-byte-aligned loads here, and should generate
// 16-byte-aligned loads as Clang does.
fn AccessN(a: Cpp.A) -> i32 {
  return a.n;
}

fn AccessM(a: Cpp.A) -> i32 {
  return a.m;
}

// --- assign_struct.carbon

library "[[@TEST_NAME]]";

import Cpp library "struct.h";

// TODO: We generate 4-byte-aligned stores here, and should generate
// 16-byte-aligned stores as Clang does.
fn AssignN(p: Cpp.A*) {
  p->n = 1;
}

fn AssignM(p: Cpp.A*) {
  p->m = 1;
}

// --- union.h

union A {
  int n;
  int m;
};

// --- access_union.carbon

library "[[@TEST_NAME]]";

import Cpp library "union.h";

fn AccessN(a: Cpp.A) -> i32 {
  return a.n;
}

fn AccessM(a: Cpp.A) -> i32 {
  return a.m;
}

// --- assign_union.carbon

library "[[@TEST_NAME]]";

import Cpp library "union.h";

fn AssignN(p: Cpp.A*) {
  p->n = 1;
}

fn AssignM(p: Cpp.A*) {
  p->m = 1;
}

// CHECK:STDOUT: ; ModuleID = 'access_struct.carbon'
// CHECK:STDOUT: source_filename = "access_struct.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_CAccessN.Main(ptr %a) !dbg !7 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_11.1.n = getelementptr inbounds nuw [32 x i8], ptr %a, i32 0, i32 0, !dbg !10
// CHECK:STDOUT:   %.loc9_11.2 = load i32, ptr %.loc9_11.1.n, align 4, !dbg !10
// CHECK:STDOUT:   ret i32 %.loc9_11.2, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_CAccessM.Main(ptr %a) !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc13_11.1.m = getelementptr inbounds nuw [32 x i8], ptr %a, i32 0, i32 16, !dbg !13
// CHECK:STDOUT:   %.loc13_11.2 = load i32, ptr %.loc13_11.1.m, align 4, !dbg !13
// CHECK:STDOUT:   ret i32 %.loc13_11.2, !dbg !14
// CHECK:STDOUT: }
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
// CHECK:STDOUT: !6 = !DIFile(filename: "access_struct.carbon", directory: "")
// CHECK:STDOUT: !7 = distinct !DISubprogram(name: "AccessN", linkageName: "_CAccessN.Main", scope: null, file: !6, line: 8, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !8 = !DISubroutineType(types: !9)
// CHECK:STDOUT: !9 = !{}
// CHECK:STDOUT: !10 = !DILocation(line: 9, column: 10, scope: !7)
// CHECK:STDOUT: !11 = !DILocation(line: 9, column: 3, scope: !7)
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "AccessM", linkageName: "_CAccessM.Main", scope: null, file: !6, line: 12, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !13 = !DILocation(line: 13, column: 10, scope: !12)
// CHECK:STDOUT: !14 = !DILocation(line: 13, column: 3, scope: !12)
// CHECK:STDOUT: ; ModuleID = 'assign_struct.carbon'
// CHECK:STDOUT: source_filename = "assign_struct.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CAssignN.Main(ptr %p) !dbg !7 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_4.2.n = getelementptr inbounds nuw [32 x i8], ptr %p, i32 0, i32 0, !dbg !10
// CHECK:STDOUT:   store i32 1, ptr %.loc9_4.2.n, align 4, !dbg !10
// CHECK:STDOUT:   ret void, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CAssignM.Main(ptr %p) !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc13_4.2.m = getelementptr inbounds nuw [32 x i8], ptr %p, i32 0, i32 16, !dbg !13
// CHECK:STDOUT:   store i32 1, ptr %.loc13_4.2.m, align 4, !dbg !13
// CHECK:STDOUT:   ret void, !dbg !14
// CHECK:STDOUT: }
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
// CHECK:STDOUT: !6 = !DIFile(filename: "assign_struct.carbon", directory: "")
// CHECK:STDOUT: !7 = distinct !DISubprogram(name: "AssignN", linkageName: "_CAssignN.Main", scope: null, file: !6, line: 8, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !8 = !DISubroutineType(types: !9)
// CHECK:STDOUT: !9 = !{}
// CHECK:STDOUT: !10 = !DILocation(line: 9, column: 3, scope: !7)
// CHECK:STDOUT: !11 = !DILocation(line: 8, column: 1, scope: !7)
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "AssignM", linkageName: "_CAssignM.Main", scope: null, file: !6, line: 12, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !13 = !DILocation(line: 13, column: 3, scope: !12)
// CHECK:STDOUT: !14 = !DILocation(line: 12, column: 1, scope: !12)
// CHECK:STDOUT: ; ModuleID = 'access_union.carbon'
// CHECK:STDOUT: source_filename = "access_union.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_CAccessN.Main(ptr %a) !dbg !7 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_11.1.n = getelementptr inbounds nuw [4 x i8], ptr %a, i32 0, i32 0, !dbg !10
// CHECK:STDOUT:   %.loc7_11.2 = load i32, ptr %.loc7_11.1.n, align 4, !dbg !10
// CHECK:STDOUT:   ret i32 %.loc7_11.2, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_CAccessM.Main(ptr %a) !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc11_11.1.m = getelementptr inbounds nuw [4 x i8], ptr %a, i32 0, i32 0, !dbg !13
// CHECK:STDOUT:   %.loc11_11.2 = load i32, ptr %.loc11_11.1.m, align 4, !dbg !13
// CHECK:STDOUT:   ret i32 %.loc11_11.2, !dbg !14
// CHECK:STDOUT: }
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
// CHECK:STDOUT: !6 = !DIFile(filename: "access_union.carbon", directory: "")
// CHECK:STDOUT: !7 = distinct !DISubprogram(name: "AccessN", linkageName: "_CAccessN.Main", scope: null, file: !6, line: 6, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !8 = !DISubroutineType(types: !9)
// CHECK:STDOUT: !9 = !{}
// CHECK:STDOUT: !10 = !DILocation(line: 7, column: 10, scope: !7)
// CHECK:STDOUT: !11 = !DILocation(line: 7, column: 3, scope: !7)
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "AccessM", linkageName: "_CAccessM.Main", scope: null, file: !6, line: 10, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !13 = !DILocation(line: 11, column: 10, scope: !12)
// CHECK:STDOUT: !14 = !DILocation(line: 11, column: 3, scope: !12)
// CHECK:STDOUT: ; ModuleID = 'assign_union.carbon'
// CHECK:STDOUT: source_filename = "assign_union.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CAssignN.Main(ptr %p) !dbg !7 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_4.2.n = getelementptr inbounds nuw [4 x i8], ptr %p, i32 0, i32 0, !dbg !10
// CHECK:STDOUT:   store i32 1, ptr %.loc7_4.2.n, align 4, !dbg !10
// CHECK:STDOUT:   ret void, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CAssignM.Main(ptr %p) !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc11_4.2.m = getelementptr inbounds nuw [4 x i8], ptr %p, i32 0, i32 0, !dbg !13
// CHECK:STDOUT:   store i32 1, ptr %.loc11_4.2.m, align 4, !dbg !13
// CHECK:STDOUT:   ret void, !dbg !14
// CHECK:STDOUT: }
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
// CHECK:STDOUT: !6 = !DIFile(filename: "assign_union.carbon", directory: "")
// CHECK:STDOUT: !7 = distinct !DISubprogram(name: "AssignN", linkageName: "_CAssignN.Main", scope: null, file: !6, line: 6, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !8 = !DISubroutineType(types: !9)
// CHECK:STDOUT: !9 = !{}
// CHECK:STDOUT: !10 = !DILocation(line: 7, column: 3, scope: !7)
// CHECK:STDOUT: !11 = !DILocation(line: 6, column: 1, scope: !7)
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "AssignM", linkageName: "_CAssignM.Main", scope: null, file: !6, line: 10, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !13 = !DILocation(line: 11, column: 3, scope: !12)
// CHECK:STDOUT: !14 = !DILocation(line: 10, column: 1, scope: !12)
