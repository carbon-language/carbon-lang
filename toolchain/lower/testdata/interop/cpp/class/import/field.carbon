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
// CHECK:STDOUT: define i32 @_CAccessN.Main(ptr %a) #0 !dbg !24 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_11.1.n = getelementptr inbounds nuw [32 x i8], ptr %a, i32 0, i32 0, !dbg !29
// CHECK:STDOUT:   %.loc9_11.2 = load i32, ptr %.loc9_11.1.n, align 4, !dbg !29
// CHECK:STDOUT:   ret i32 %.loc9_11.2, !dbg !30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CAccessM.Main(ptr %a) #0 !dbg !32 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc13_11.1.m = getelementptr inbounds nuw [32 x i8], ptr %a, i32 0, i32 16, !dbg !37
// CHECK:STDOUT:   %.loc13_11.2 = load i32, ptr %.loc13_11.1.m, align 4, !dbg !37
// CHECK:STDOUT:   ret i32 %.loc13_11.2, !dbg !38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!40, !41, !42, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!47}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !18 = !DIFile(filename: "access_struct.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{!20, !21}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !24 = distinct !DISubprogram(name: "AccessN", linkageName: "_CAccessN.Main", scope: null, file: !18, line: 8, type: !23, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !31)
// CHECK:STDOUT: !25 = !DILocalVariable(arg: 1, scope: !24, type: !21)
// CHECK:STDOUT: !29 = !DILocation(line: 9, column: 10, scope: !24)
// CHECK:STDOUT: !30 = !DILocation(line: 9, column: 3, scope: !24)
// CHECK:STDOUT: !31 = !{!25}
// CHECK:STDOUT: !32 = distinct !DISubprogram(name: "AccessM", linkageName: "_CAccessM.Main", scope: null, file: !18, line: 12, type: !23, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !39)
// CHECK:STDOUT: !33 = !DILocalVariable(arg: 1, scope: !32, type: !21)
// CHECK:STDOUT: !37 = !DILocation(line: 13, column: 10, scope: !32)
// CHECK:STDOUT: !38 = !DILocation(line: 13, column: 3, scope: !32)
// CHECK:STDOUT: !39 = !{!33}
// CHECK:STDOUT: !40 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !41 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !42 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !43 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !44 = !{!"omnipotent char", !43, i64 0}
// CHECK:STDOUT: !45 = !{!"int", !44, i64 0}
// CHECK:STDOUT: !46 = !{!"__libc_errno", !45, i64 0}
// CHECK:STDOUT: !47 = !{!46, !45, i64 0}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'assign_struct.carbon'
// CHECK:STDOUT: source_filename = "assign_struct.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CAssignN.Main(ptr %p) #0 !dbg !52 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_4.2.n = getelementptr inbounds nuw [32 x i8], ptr %p, i32 0, i32 0, !dbg !55
// CHECK:STDOUT:   store i32 1, ptr %.loc9_4.2.n, align 4, !dbg !55
// CHECK:STDOUT:   ret void, !dbg !56
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CAssignM.Main(ptr %p) #0 !dbg !58 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc13_4.2.m = getelementptr inbounds nuw [32 x i8], ptr %p, i32 0, i32 16, !dbg !61
// CHECK:STDOUT:   store i32 1, ptr %.loc13_4.2.m, align 4, !dbg !61
// CHECK:STDOUT:   ret void, !dbg !62
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!49}
// CHECK:STDOUT: !llvm.module.flags = !{!40, !41, !42, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!47}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !40 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !41 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !42 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !43 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !44 = !{!"omnipotent char", !43, i64 0}
// CHECK:STDOUT: !45 = !{!"int", !44, i64 0}
// CHECK:STDOUT: !46 = !{!"__libc_errno", !45, i64 0}
// CHECK:STDOUT: !47 = !{!46, !45, i64 0}
// CHECK:STDOUT: !48 = !DIFile(filename: "assign_struct.carbon", directory: "")
// CHECK:STDOUT: !49 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !48, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !50 = !{null, !21}
// CHECK:STDOUT: !51 = !DISubroutineType(types: !50)
// CHECK:STDOUT: !52 = distinct !DISubprogram(name: "AssignN", linkageName: "_CAssignN.Main", scope: null, file: !48, line: 8, type: !51, spFlags: DISPFlagDefinition, unit: !49, retainedNodes: !57)
// CHECK:STDOUT: !53 = !DILocalVariable(arg: 1, scope: !52, type: !21)
// CHECK:STDOUT: !55 = !DILocation(line: 9, column: 3, scope: !52)
// CHECK:STDOUT: !56 = !DILocation(line: 8, column: 1, scope: !52)
// CHECK:STDOUT: !57 = !{!53}
// CHECK:STDOUT: !58 = distinct !DISubprogram(name: "AssignM", linkageName: "_CAssignM.Main", scope: null, file: !48, line: 12, type: !51, spFlags: DISPFlagDefinition, unit: !49, retainedNodes: !63)
// CHECK:STDOUT: !59 = !DILocalVariable(arg: 1, scope: !58, type: !21)
// CHECK:STDOUT: !61 = !DILocation(line: 13, column: 3, scope: !58)
// CHECK:STDOUT: !62 = !DILocation(line: 12, column: 1, scope: !58)
// CHECK:STDOUT: !63 = !{!59}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'access_union.carbon'
// CHECK:STDOUT: source_filename = "access_union.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CAccessN.Main(ptr %a) #0 !dbg !66 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_11.1.n = getelementptr inbounds nuw [4 x i8], ptr %a, i32 0, i32 0, !dbg !71
// CHECK:STDOUT:   %.loc7_11.2 = load i32, ptr %.loc7_11.1.n, align 4, !dbg !71
// CHECK:STDOUT:   ret i32 %.loc7_11.2, !dbg !72
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CAccessM.Main(ptr %a) #0 !dbg !74 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc11_11.1.m = getelementptr inbounds nuw [4 x i8], ptr %a, i32 0, i32 0, !dbg !79
// CHECK:STDOUT:   %.loc11_11.2 = load i32, ptr %.loc11_11.1.m, align 4, !dbg !79
// CHECK:STDOUT:   ret i32 %.loc11_11.2, !dbg !80
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!65}
// CHECK:STDOUT: !llvm.module.flags = !{!40, !41, !42, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!47}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{!20, !21}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !40 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !41 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !42 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !43 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !44 = !{!"omnipotent char", !43, i64 0}
// CHECK:STDOUT: !45 = !{!"int", !44, i64 0}
// CHECK:STDOUT: !46 = !{!"__libc_errno", !45, i64 0}
// CHECK:STDOUT: !47 = !{!46, !45, i64 0}
// CHECK:STDOUT: !64 = !DIFile(filename: "access_union.carbon", directory: "")
// CHECK:STDOUT: !65 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !64, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !66 = distinct !DISubprogram(name: "AccessN", linkageName: "_CAccessN.Main", scope: null, file: !64, line: 6, type: !23, spFlags: DISPFlagDefinition, unit: !65, retainedNodes: !73)
// CHECK:STDOUT: !67 = !DILocalVariable(arg: 1, scope: !66, type: !21)
// CHECK:STDOUT: !71 = !DILocation(line: 7, column: 10, scope: !66)
// CHECK:STDOUT: !72 = !DILocation(line: 7, column: 3, scope: !66)
// CHECK:STDOUT: !73 = !{!67}
// CHECK:STDOUT: !74 = distinct !DISubprogram(name: "AccessM", linkageName: "_CAccessM.Main", scope: null, file: !64, line: 10, type: !23, spFlags: DISPFlagDefinition, unit: !65, retainedNodes: !81)
// CHECK:STDOUT: !75 = !DILocalVariable(arg: 1, scope: !74, type: !21)
// CHECK:STDOUT: !79 = !DILocation(line: 11, column: 10, scope: !74)
// CHECK:STDOUT: !80 = !DILocation(line: 11, column: 3, scope: !74)
// CHECK:STDOUT: !81 = !{!75}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'assign_union.carbon'
// CHECK:STDOUT: source_filename = "assign_union.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CAssignN.Main(ptr %p) #0 !dbg !84 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_4.2.n = getelementptr inbounds nuw [4 x i8], ptr %p, i32 0, i32 0, !dbg !87
// CHECK:STDOUT:   store i32 1, ptr %.loc7_4.2.n, align 4, !dbg !87
// CHECK:STDOUT:   ret void, !dbg !88
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CAssignM.Main(ptr %p) #0 !dbg !90 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc11_4.2.m = getelementptr inbounds nuw [4 x i8], ptr %p, i32 0, i32 0, !dbg !93
// CHECK:STDOUT:   store i32 1, ptr %.loc11_4.2.m, align 4, !dbg !93
// CHECK:STDOUT:   ret void, !dbg !94
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!83}
// CHECK:STDOUT: !llvm.module.flags = !{!40, !41, !42, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!47}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !40 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !41 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !42 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !43 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !44 = !{!"omnipotent char", !43, i64 0}
// CHECK:STDOUT: !45 = !{!"int", !44, i64 0}
// CHECK:STDOUT: !46 = !{!"__libc_errno", !45, i64 0}
// CHECK:STDOUT: !47 = !{!46, !45, i64 0}
// CHECK:STDOUT: !50 = !{null, !21}
// CHECK:STDOUT: !51 = !DISubroutineType(types: !50)
// CHECK:STDOUT: !82 = !DIFile(filename: "assign_union.carbon", directory: "")
// CHECK:STDOUT: !83 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !82, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !84 = distinct !DISubprogram(name: "AssignN", linkageName: "_CAssignN.Main", scope: null, file: !82, line: 6, type: !51, spFlags: DISPFlagDefinition, unit: !83, retainedNodes: !89)
// CHECK:STDOUT: !85 = !DILocalVariable(arg: 1, scope: !84, type: !21)
// CHECK:STDOUT: !87 = !DILocation(line: 7, column: 3, scope: !84)
// CHECK:STDOUT: !88 = !DILocation(line: 6, column: 1, scope: !84)
// CHECK:STDOUT: !89 = !{!85}
// CHECK:STDOUT: !90 = distinct !DISubprogram(name: "AssignM", linkageName: "_CAssignM.Main", scope: null, file: !82, line: 10, type: !51, spFlags: DISPFlagDefinition, unit: !83, retainedNodes: !95)
// CHECK:STDOUT: !91 = !DILocalVariable(arg: 1, scope: !90, type: !21)
// CHECK:STDOUT: !93 = !DILocation(line: 11, column: 3, scope: !90)
// CHECK:STDOUT: !94 = !DILocation(line: 10, column: 1, scope: !90)
// CHECK:STDOUT: !95 = !{!91}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'access_anon_union.carbon'
// CHECK:STDOUT: source_filename = "access_anon_union.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CAccessN.Main(ptr %a) #0 !dbg !98 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_11.1.n = getelementptr inbounds nuw [16 x i8], ptr %a, i32 0, i32 8, !dbg !103
// CHECK:STDOUT:   %.loc7_11.2 = load i32, ptr %.loc7_11.1.n, align 4, !dbg !103
// CHECK:STDOUT:   ret i32 %.loc7_11.2, !dbg !104
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CAccessP.Main(ptr %a) #0 !dbg !106 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc11_12.1.p = getelementptr inbounds nuw [16 x i8], ptr %a, i32 0, i32 8, !dbg !111
// CHECK:STDOUT:   %.loc11_12.2 = load ptr, ptr %.loc11_12.1.p, align 8, !dbg !111
// CHECK:STDOUT:   %.loc11_10.2 = load i32, ptr %.loc11_12.2, align 4, !dbg !112
// CHECK:STDOUT:   ret i32 %.loc11_10.2, !dbg !113
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!97}
// CHECK:STDOUT: !llvm.module.flags = !{!40, !41, !42, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!47}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{!20, !21}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !40 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !41 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !42 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !43 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !44 = !{!"omnipotent char", !43, i64 0}
// CHECK:STDOUT: !45 = !{!"int", !44, i64 0}
// CHECK:STDOUT: !46 = !{!"__libc_errno", !45, i64 0}
// CHECK:STDOUT: !47 = !{!46, !45, i64 0}
// CHECK:STDOUT: !96 = !DIFile(filename: "access_anon_union.carbon", directory: "")
// CHECK:STDOUT: !97 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !96, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !98 = distinct !DISubprogram(name: "AccessN", linkageName: "_CAccessN.Main", scope: null, file: !96, line: 6, type: !23, spFlags: DISPFlagDefinition, unit: !97, retainedNodes: !105)
// CHECK:STDOUT: !99 = !DILocalVariable(arg: 1, scope: !98, type: !21)
// CHECK:STDOUT: !103 = !DILocation(line: 7, column: 10, scope: !98)
// CHECK:STDOUT: !104 = !DILocation(line: 7, column: 3, scope: !98)
// CHECK:STDOUT: !105 = !{!99}
// CHECK:STDOUT: !106 = distinct !DISubprogram(name: "AccessP", linkageName: "_CAccessP.Main", scope: null, file: !96, line: 10, type: !23, spFlags: DISPFlagDefinition, unit: !97, retainedNodes: !114)
// CHECK:STDOUT: !107 = !DILocalVariable(arg: 1, scope: !106, type: !21)
// CHECK:STDOUT: !111 = !DILocation(line: 11, column: 11, scope: !106)
// CHECK:STDOUT: !112 = !DILocation(line: 11, column: 10, scope: !106)
// CHECK:STDOUT: !113 = !DILocation(line: 11, column: 3, scope: !106)
// CHECK:STDOUT: !114 = !{!107}
// CHECK:STDOUT:
