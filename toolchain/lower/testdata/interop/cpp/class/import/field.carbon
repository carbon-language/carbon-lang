// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/class/import/field.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/class/import/field.carbon

// --- struct.h

struct StructA {
  int n;
  alignas(16) int m;
};

// --- access_struct.carbon

library "[[@TEST_NAME]]";

import Cpp library "struct.h";

// TODO: We generate 4-byte-aligned loads here, and should generate
// 16-byte-aligned loads as Clang does.
fn AccessN(a: Cpp.StructA) -> i32 {
  return a.n;
}

fn AccessM(a: Cpp.StructA) -> i32 {
  return a.m;
}

// --- assign_struct.carbon

library "[[@TEST_NAME]]";

import Cpp library "struct.h";

// TODO: We generate 4-byte-aligned stores here, and should generate
// 16-byte-aligned stores as Clang does.
fn AssignN(p: Cpp.StructA*) {
  p->n = 1;
}

fn AssignM(p: Cpp.StructA*) {
  p->m = 1;
}

// --- union.h

union UnionA {
  int n;
  int m;
};

// --- access_union.carbon

library "[[@TEST_NAME]]";

import Cpp library "union.h";

fn AccessN(a: Cpp.UnionA) -> i32 {
  return a.n;
}

fn AccessM(a: Cpp.UnionA) -> i32 {
  return a.m;
}

// --- assign_union.carbon

library "[[@TEST_NAME]]";

import Cpp library "union.h";

fn AssignN(p: Cpp.UnionA*) {
  p->n = 1;
}

fn AssignM(p: Cpp.UnionA*) {
  p->m = 1;
}

// --- anon_union.h

struct AnonUnionA {
  int x;
  union {
    int n;
    int *_Nonnull p;
  };
};

// --- access_anon_union.carbon

library "[[@TEST_NAME]]";

import Cpp library "anon_union.h";

fn AccessN(a: Cpp.AnonUnionA) -> i32 {
  return a.n;
}

fn AccessP(a: Cpp.AnonUnionA) -> i32 {
  return *a.p;
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'access_struct.carbon'
// CHECK:STDOUT: source_filename = "access_struct.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CAccessN.Main(ptr %a) #0 !dbg !26 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_11.1.n = getelementptr inbounds nuw [32 x i8], ptr %a, i32 0, i32 0, !dbg !31
// CHECK:STDOUT:   %.loc9_11.2 = load i32, ptr %.loc9_11.1.n, align 4, !dbg !31
// CHECK:STDOUT:   ret i32 %.loc9_11.2, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CAccessM.Main(ptr %a) #0 !dbg !34 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc13_11.1.m = getelementptr inbounds nuw [32 x i8], ptr %a, i32 0, i32 16, !dbg !39
// CHECK:STDOUT:   %.loc13_11.2 = load i32, ptr %.loc13_11.1.m, align 4, !dbg !39
// CHECK:STDOUT:   ret i32 %.loc13_11.2, !dbg !40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!21}
// CHECK:STDOUT: !llvm.module.flags = !{!42, !43, !44, !45, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!50}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIFile(filename: "access_struct.carbon", directory: "")
// CHECK:STDOUT: !21 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !20, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !22 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{!22, !23}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "AccessN", linkageName: "_CAccessN.Main", scope: null, file: !20, line: 8, type: !25, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !33)
// CHECK:STDOUT: !27 = !DILocalVariable(arg: 1, scope: !26, type: !23)
// CHECK:STDOUT: !31 = !DILocation(line: 9, column: 10, scope: !26)
// CHECK:STDOUT: !32 = !DILocation(line: 9, column: 3, scope: !26)
// CHECK:STDOUT: !33 = !{!27}
// CHECK:STDOUT: !34 = distinct !DISubprogram(name: "AccessM", linkageName: "_CAccessM.Main", scope: null, file: !20, line: 12, type: !25, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !41)
// CHECK:STDOUT: !35 = !DILocalVariable(arg: 1, scope: !34, type: !23)
// CHECK:STDOUT: !39 = !DILocation(line: 13, column: 10, scope: !34)
// CHECK:STDOUT: !40 = !DILocation(line: 13, column: 3, scope: !34)
// CHECK:STDOUT: !41 = !{!35}
// CHECK:STDOUT: !42 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !43 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !44 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !45 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !46 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !47 = !{!"omnipotent char", !46, i64 0}
// CHECK:STDOUT: !48 = !{!"int", !47, i64 0}
// CHECK:STDOUT: !49 = !{!"__libc_errno", !48, i64 0}
// CHECK:STDOUT: !50 = !{!49, !48, i64 0}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'assign_struct.carbon'
// CHECK:STDOUT: source_filename = "assign_struct.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CAssignN.Main(ptr %p) #0 !dbg !55 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_4.2.n = getelementptr inbounds nuw [32 x i8], ptr %p, i32 0, i32 0, !dbg !58
// CHECK:STDOUT:   store i32 1, ptr %.loc9_4.2.n, align 4, !dbg !58
// CHECK:STDOUT:   ret void, !dbg !59
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CAssignM.Main(ptr %p) #0 !dbg !61 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc13_4.2.m = getelementptr inbounds nuw [32 x i8], ptr %p, i32 0, i32 16, !dbg !64
// CHECK:STDOUT:   store i32 1, ptr %.loc13_4.2.m, align 4, !dbg !64
// CHECK:STDOUT:   ret void, !dbg !65
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!52}
// CHECK:STDOUT: !llvm.module.flags = !{!42, !43, !44, !45, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!50}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !42 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !43 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !44 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !45 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !46 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !47 = !{!"omnipotent char", !46, i64 0}
// CHECK:STDOUT: !48 = !{!"int", !47, i64 0}
// CHECK:STDOUT: !49 = !{!"__libc_errno", !48, i64 0}
// CHECK:STDOUT: !50 = !{!49, !48, i64 0}
// CHECK:STDOUT: !51 = !DIFile(filename: "assign_struct.carbon", directory: "")
// CHECK:STDOUT: !52 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !51, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !53 = !{null, !23}
// CHECK:STDOUT: !54 = !DISubroutineType(types: !53)
// CHECK:STDOUT: !55 = distinct !DISubprogram(name: "AssignN", linkageName: "_CAssignN.Main", scope: null, file: !51, line: 8, type: !54, spFlags: DISPFlagDefinition, unit: !52, retainedNodes: !60)
// CHECK:STDOUT: !56 = !DILocalVariable(arg: 1, scope: !55, type: !23)
// CHECK:STDOUT: !58 = !DILocation(line: 9, column: 3, scope: !55)
// CHECK:STDOUT: !59 = !DILocation(line: 8, column: 1, scope: !55)
// CHECK:STDOUT: !60 = !{!56}
// CHECK:STDOUT: !61 = distinct !DISubprogram(name: "AssignM", linkageName: "_CAssignM.Main", scope: null, file: !51, line: 12, type: !54, spFlags: DISPFlagDefinition, unit: !52, retainedNodes: !66)
// CHECK:STDOUT: !62 = !DILocalVariable(arg: 1, scope: !61, type: !23)
// CHECK:STDOUT: !64 = !DILocation(line: 13, column: 3, scope: !61)
// CHECK:STDOUT: !65 = !DILocation(line: 12, column: 1, scope: !61)
// CHECK:STDOUT: !66 = !{!62}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'access_union.carbon'
// CHECK:STDOUT: source_filename = "access_union.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CAccessN.Main(ptr %a) #0 !dbg !69 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_11.1.n = getelementptr inbounds nuw [4 x i8], ptr %a, i32 0, i32 0, !dbg !74
// CHECK:STDOUT:   %.loc7_11.2 = load i32, ptr %.loc7_11.1.n, align 4, !dbg !74
// CHECK:STDOUT:   ret i32 %.loc7_11.2, !dbg !75
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CAccessM.Main(ptr %a) #0 !dbg !77 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc11_11.1.m = getelementptr inbounds nuw [4 x i8], ptr %a, i32 0, i32 0, !dbg !82
// CHECK:STDOUT:   %.loc11_11.2 = load i32, ptr %.loc11_11.1.m, align 4, !dbg !82
// CHECK:STDOUT:   ret i32 %.loc11_11.2, !dbg !83
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!68}
// CHECK:STDOUT: !llvm.module.flags = !{!42, !43, !44, !45, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!50}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !22 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{!22, !23}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !42 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !43 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !44 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !45 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !46 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !47 = !{!"omnipotent char", !46, i64 0}
// CHECK:STDOUT: !48 = !{!"int", !47, i64 0}
// CHECK:STDOUT: !49 = !{!"__libc_errno", !48, i64 0}
// CHECK:STDOUT: !50 = !{!49, !48, i64 0}
// CHECK:STDOUT: !67 = !DIFile(filename: "access_union.carbon", directory: "")
// CHECK:STDOUT: !68 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !67, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !69 = distinct !DISubprogram(name: "AccessN", linkageName: "_CAccessN.Main", scope: null, file: !67, line: 6, type: !25, spFlags: DISPFlagDefinition, unit: !68, retainedNodes: !76)
// CHECK:STDOUT: !70 = !DILocalVariable(arg: 1, scope: !69, type: !23)
// CHECK:STDOUT: !74 = !DILocation(line: 7, column: 10, scope: !69)
// CHECK:STDOUT: !75 = !DILocation(line: 7, column: 3, scope: !69)
// CHECK:STDOUT: !76 = !{!70}
// CHECK:STDOUT: !77 = distinct !DISubprogram(name: "AccessM", linkageName: "_CAccessM.Main", scope: null, file: !67, line: 10, type: !25, spFlags: DISPFlagDefinition, unit: !68, retainedNodes: !84)
// CHECK:STDOUT: !78 = !DILocalVariable(arg: 1, scope: !77, type: !23)
// CHECK:STDOUT: !82 = !DILocation(line: 11, column: 10, scope: !77)
// CHECK:STDOUT: !83 = !DILocation(line: 11, column: 3, scope: !77)
// CHECK:STDOUT: !84 = !{!78}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'assign_union.carbon'
// CHECK:STDOUT: source_filename = "assign_union.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CAssignN.Main(ptr %p) #0 !dbg !87 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_4.2.n = getelementptr inbounds nuw [4 x i8], ptr %p, i32 0, i32 0, !dbg !90
// CHECK:STDOUT:   store i32 1, ptr %.loc7_4.2.n, align 4, !dbg !90
// CHECK:STDOUT:   ret void, !dbg !91
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CAssignM.Main(ptr %p) #0 !dbg !93 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc11_4.2.m = getelementptr inbounds nuw [4 x i8], ptr %p, i32 0, i32 0, !dbg !96
// CHECK:STDOUT:   store i32 1, ptr %.loc11_4.2.m, align 4, !dbg !96
// CHECK:STDOUT:   ret void, !dbg !97
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!86}
// CHECK:STDOUT: !llvm.module.flags = !{!42, !43, !44, !45, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!50}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !42 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !43 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !44 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !45 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !46 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !47 = !{!"omnipotent char", !46, i64 0}
// CHECK:STDOUT: !48 = !{!"int", !47, i64 0}
// CHECK:STDOUT: !49 = !{!"__libc_errno", !48, i64 0}
// CHECK:STDOUT: !50 = !{!49, !48, i64 0}
// CHECK:STDOUT: !53 = !{null, !23}
// CHECK:STDOUT: !54 = !DISubroutineType(types: !53)
// CHECK:STDOUT: !85 = !DIFile(filename: "assign_union.carbon", directory: "")
// CHECK:STDOUT: !86 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !85, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !87 = distinct !DISubprogram(name: "AssignN", linkageName: "_CAssignN.Main", scope: null, file: !85, line: 6, type: !54, spFlags: DISPFlagDefinition, unit: !86, retainedNodes: !92)
// CHECK:STDOUT: !88 = !DILocalVariable(arg: 1, scope: !87, type: !23)
// CHECK:STDOUT: !90 = !DILocation(line: 7, column: 3, scope: !87)
// CHECK:STDOUT: !91 = !DILocation(line: 6, column: 1, scope: !87)
// CHECK:STDOUT: !92 = !{!88}
// CHECK:STDOUT: !93 = distinct !DISubprogram(name: "AssignM", linkageName: "_CAssignM.Main", scope: null, file: !85, line: 10, type: !54, spFlags: DISPFlagDefinition, unit: !86, retainedNodes: !98)
// CHECK:STDOUT: !94 = !DILocalVariable(arg: 1, scope: !93, type: !23)
// CHECK:STDOUT: !96 = !DILocation(line: 11, column: 3, scope: !93)
// CHECK:STDOUT: !97 = !DILocation(line: 10, column: 1, scope: !93)
// CHECK:STDOUT: !98 = !{!94}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'access_anon_union.carbon'
// CHECK:STDOUT: source_filename = "access_anon_union.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CAccessN.Main(ptr %a) #0 !dbg !101 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_11.1.n = getelementptr inbounds nuw [16 x i8], ptr %a, i32 0, i32 8, !dbg !106
// CHECK:STDOUT:   %.loc7_11.2 = load i32, ptr %.loc7_11.1.n, align 4, !dbg !106
// CHECK:STDOUT:   ret i32 %.loc7_11.2, !dbg !107
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CAccessP.Main(ptr %a) #0 !dbg !109 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc11_12.1.p = getelementptr inbounds nuw [16 x i8], ptr %a, i32 0, i32 8, !dbg !114
// CHECK:STDOUT:   %.loc11_12.2 = load ptr, ptr %.loc11_12.1.p, align 8, !dbg !114
// CHECK:STDOUT:   %.loc11_10.2 = load i32, ptr %.loc11_12.2, align 4, !dbg !115
// CHECK:STDOUT:   ret i32 %.loc11_10.2, !dbg !116
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!100}
// CHECK:STDOUT: !llvm.module.flags = !{!42, !43, !44, !45, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!50}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !22 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{!22, !23}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !42 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !43 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !44 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !45 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !46 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !47 = !{!"omnipotent char", !46, i64 0}
// CHECK:STDOUT: !48 = !{!"int", !47, i64 0}
// CHECK:STDOUT: !49 = !{!"__libc_errno", !48, i64 0}
// CHECK:STDOUT: !50 = !{!49, !48, i64 0}
// CHECK:STDOUT: !99 = !DIFile(filename: "access_anon_union.carbon", directory: "")
// CHECK:STDOUT: !100 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !99, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !101 = distinct !DISubprogram(name: "AccessN", linkageName: "_CAccessN.Main", scope: null, file: !99, line: 6, type: !25, spFlags: DISPFlagDefinition, unit: !100, retainedNodes: !108)
// CHECK:STDOUT: !102 = !DILocalVariable(arg: 1, scope: !101, type: !23)
// CHECK:STDOUT: !106 = !DILocation(line: 7, column: 10, scope: !101)
// CHECK:STDOUT: !107 = !DILocation(line: 7, column: 3, scope: !101)
// CHECK:STDOUT: !108 = !{!102}
// CHECK:STDOUT: !109 = distinct !DISubprogram(name: "AccessP", linkageName: "_CAccessP.Main", scope: null, file: !99, line: 10, type: !25, spFlags: DISPFlagDefinition, unit: !100, retainedNodes: !117)
// CHECK:STDOUT: !110 = !DILocalVariable(arg: 1, scope: !109, type: !23)
// CHECK:STDOUT: !114 = !DILocation(line: 11, column: 11, scope: !109)
// CHECK:STDOUT: !115 = !DILocation(line: 11, column: 10, scope: !109)
// CHECK:STDOUT: !116 = !DILocation(line: 11, column: 3, scope: !109)
// CHECK:STDOUT: !117 = !{!110}
// CHECK:STDOUT:
