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
// CHECK:STDOUT: !llvm.module.flags = !{!40, !41, !42, !43, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!48}
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
// CHECK:STDOUT: !40 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !41 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !42 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !43 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !44 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !45 = !{!"omnipotent char", !44, i64 0}
// CHECK:STDOUT: !46 = !{!"int", !45, i64 0}
// CHECK:STDOUT: !47 = !{!"__libc_errno", !46, i64 0}
// CHECK:STDOUT: !48 = !{!47, !46, i64 0}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'assign_struct.carbon'
// CHECK:STDOUT: source_filename = "assign_struct.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CAssignN.Main(ptr %p) #0 !dbg !53 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_4.2.n = getelementptr inbounds nuw [32 x i8], ptr %p, i32 0, i32 0, !dbg !56
// CHECK:STDOUT:   store i32 1, ptr %.loc9_4.2.n, align 4, !dbg !56
// CHECK:STDOUT:   ret void, !dbg !57
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CAssignM.Main(ptr %p) #0 !dbg !59 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc13_4.2.m = getelementptr inbounds nuw [32 x i8], ptr %p, i32 0, i32 16, !dbg !62
// CHECK:STDOUT:   store i32 1, ptr %.loc13_4.2.m, align 4, !dbg !62
// CHECK:STDOUT:   ret void, !dbg !63
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!50}
// CHECK:STDOUT: !llvm.module.flags = !{!40, !41, !42, !43, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!48}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !40 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !41 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !42 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !43 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !44 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !45 = !{!"omnipotent char", !44, i64 0}
// CHECK:STDOUT: !46 = !{!"int", !45, i64 0}
// CHECK:STDOUT: !47 = !{!"__libc_errno", !46, i64 0}
// CHECK:STDOUT: !48 = !{!47, !46, i64 0}
// CHECK:STDOUT: !49 = !DIFile(filename: "assign_struct.carbon", directory: "")
// CHECK:STDOUT: !50 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !49, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !51 = !{null, !21}
// CHECK:STDOUT: !52 = !DISubroutineType(types: !51)
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "AssignN", linkageName: "_CAssignN.Main", scope: null, file: !49, line: 8, type: !52, spFlags: DISPFlagDefinition, unit: !50, retainedNodes: !58)
// CHECK:STDOUT: !54 = !DILocalVariable(arg: 1, scope: !53, type: !21)
// CHECK:STDOUT: !56 = !DILocation(line: 9, column: 3, scope: !53)
// CHECK:STDOUT: !57 = !DILocation(line: 8, column: 1, scope: !53)
// CHECK:STDOUT: !58 = !{!54}
// CHECK:STDOUT: !59 = distinct !DISubprogram(name: "AssignM", linkageName: "_CAssignM.Main", scope: null, file: !49, line: 12, type: !52, spFlags: DISPFlagDefinition, unit: !50, retainedNodes: !64)
// CHECK:STDOUT: !60 = !DILocalVariable(arg: 1, scope: !59, type: !21)
// CHECK:STDOUT: !62 = !DILocation(line: 13, column: 3, scope: !59)
// CHECK:STDOUT: !63 = !DILocation(line: 12, column: 1, scope: !59)
// CHECK:STDOUT: !64 = !{!60}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'access_union.carbon'
// CHECK:STDOUT: source_filename = "access_union.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CAccessN.Main(ptr %a) #0 !dbg !67 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_11.1.n = getelementptr inbounds nuw [4 x i8], ptr %a, i32 0, i32 0, !dbg !72
// CHECK:STDOUT:   %.loc7_11.2 = load i32, ptr %.loc7_11.1.n, align 4, !dbg !72
// CHECK:STDOUT:   ret i32 %.loc7_11.2, !dbg !73
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CAccessM.Main(ptr %a) #0 !dbg !75 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc11_11.1.m = getelementptr inbounds nuw [4 x i8], ptr %a, i32 0, i32 0, !dbg !80
// CHECK:STDOUT:   %.loc11_11.2 = load i32, ptr %.loc11_11.1.m, align 4, !dbg !80
// CHECK:STDOUT:   ret i32 %.loc11_11.2, !dbg !81
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!66}
// CHECK:STDOUT: !llvm.module.flags = !{!40, !41, !42, !43, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!48}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{!20, !21}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !40 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !41 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !42 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !43 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !44 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !45 = !{!"omnipotent char", !44, i64 0}
// CHECK:STDOUT: !46 = !{!"int", !45, i64 0}
// CHECK:STDOUT: !47 = !{!"__libc_errno", !46, i64 0}
// CHECK:STDOUT: !48 = !{!47, !46, i64 0}
// CHECK:STDOUT: !65 = !DIFile(filename: "access_union.carbon", directory: "")
// CHECK:STDOUT: !66 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !65, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !67 = distinct !DISubprogram(name: "AccessN", linkageName: "_CAccessN.Main", scope: null, file: !65, line: 6, type: !23, spFlags: DISPFlagDefinition, unit: !66, retainedNodes: !74)
// CHECK:STDOUT: !68 = !DILocalVariable(arg: 1, scope: !67, type: !21)
// CHECK:STDOUT: !72 = !DILocation(line: 7, column: 10, scope: !67)
// CHECK:STDOUT: !73 = !DILocation(line: 7, column: 3, scope: !67)
// CHECK:STDOUT: !74 = !{!68}
// CHECK:STDOUT: !75 = distinct !DISubprogram(name: "AccessM", linkageName: "_CAccessM.Main", scope: null, file: !65, line: 10, type: !23, spFlags: DISPFlagDefinition, unit: !66, retainedNodes: !82)
// CHECK:STDOUT: !76 = !DILocalVariable(arg: 1, scope: !75, type: !21)
// CHECK:STDOUT: !80 = !DILocation(line: 11, column: 10, scope: !75)
// CHECK:STDOUT: !81 = !DILocation(line: 11, column: 3, scope: !75)
// CHECK:STDOUT: !82 = !{!76}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'assign_union.carbon'
// CHECK:STDOUT: source_filename = "assign_union.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CAssignN.Main(ptr %p) #0 !dbg !85 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_4.2.n = getelementptr inbounds nuw [4 x i8], ptr %p, i32 0, i32 0, !dbg !88
// CHECK:STDOUT:   store i32 1, ptr %.loc7_4.2.n, align 4, !dbg !88
// CHECK:STDOUT:   ret void, !dbg !89
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CAssignM.Main(ptr %p) #0 !dbg !91 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc11_4.2.m = getelementptr inbounds nuw [4 x i8], ptr %p, i32 0, i32 0, !dbg !94
// CHECK:STDOUT:   store i32 1, ptr %.loc11_4.2.m, align 4, !dbg !94
// CHECK:STDOUT:   ret void, !dbg !95
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!84}
// CHECK:STDOUT: !llvm.module.flags = !{!40, !41, !42, !43, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!48}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !40 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !41 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !42 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !43 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !44 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !45 = !{!"omnipotent char", !44, i64 0}
// CHECK:STDOUT: !46 = !{!"int", !45, i64 0}
// CHECK:STDOUT: !47 = !{!"__libc_errno", !46, i64 0}
// CHECK:STDOUT: !48 = !{!47, !46, i64 0}
// CHECK:STDOUT: !51 = !{null, !21}
// CHECK:STDOUT: !52 = !DISubroutineType(types: !51)
// CHECK:STDOUT: !83 = !DIFile(filename: "assign_union.carbon", directory: "")
// CHECK:STDOUT: !84 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !83, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !85 = distinct !DISubprogram(name: "AssignN", linkageName: "_CAssignN.Main", scope: null, file: !83, line: 6, type: !52, spFlags: DISPFlagDefinition, unit: !84, retainedNodes: !90)
// CHECK:STDOUT: !86 = !DILocalVariable(arg: 1, scope: !85, type: !21)
// CHECK:STDOUT: !88 = !DILocation(line: 7, column: 3, scope: !85)
// CHECK:STDOUT: !89 = !DILocation(line: 6, column: 1, scope: !85)
// CHECK:STDOUT: !90 = !{!86}
// CHECK:STDOUT: !91 = distinct !DISubprogram(name: "AssignM", linkageName: "_CAssignM.Main", scope: null, file: !83, line: 10, type: !52, spFlags: DISPFlagDefinition, unit: !84, retainedNodes: !96)
// CHECK:STDOUT: !92 = !DILocalVariable(arg: 1, scope: !91, type: !21)
// CHECK:STDOUT: !94 = !DILocation(line: 11, column: 3, scope: !91)
// CHECK:STDOUT: !95 = !DILocation(line: 10, column: 1, scope: !91)
// CHECK:STDOUT: !96 = !{!92}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'access_anon_union.carbon'
// CHECK:STDOUT: source_filename = "access_anon_union.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CAccessN.Main(ptr %a) #0 !dbg !99 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_11.1.n = getelementptr inbounds nuw [16 x i8], ptr %a, i32 0, i32 8, !dbg !104
// CHECK:STDOUT:   %.loc7_11.2 = load i32, ptr %.loc7_11.1.n, align 4, !dbg !104
// CHECK:STDOUT:   ret i32 %.loc7_11.2, !dbg !105
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CAccessP.Main(ptr %a) #0 !dbg !107 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc11_12.1.p = getelementptr inbounds nuw [16 x i8], ptr %a, i32 0, i32 8, !dbg !112
// CHECK:STDOUT:   %.loc11_12.2 = load ptr, ptr %.loc11_12.1.p, align 8, !dbg !112
// CHECK:STDOUT:   %.loc11_10.2 = load i32, ptr %.loc11_12.2, align 4, !dbg !113
// CHECK:STDOUT:   ret i32 %.loc11_10.2, !dbg !114
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!98}
// CHECK:STDOUT: !llvm.module.flags = !{!40, !41, !42, !43, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!48}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{!20, !21}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !40 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !41 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !42 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !43 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !44 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !45 = !{!"omnipotent char", !44, i64 0}
// CHECK:STDOUT: !46 = !{!"int", !45, i64 0}
// CHECK:STDOUT: !47 = !{!"__libc_errno", !46, i64 0}
// CHECK:STDOUT: !48 = !{!47, !46, i64 0}
// CHECK:STDOUT: !97 = !DIFile(filename: "access_anon_union.carbon", directory: "")
// CHECK:STDOUT: !98 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !97, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !99 = distinct !DISubprogram(name: "AccessN", linkageName: "_CAccessN.Main", scope: null, file: !97, line: 6, type: !23, spFlags: DISPFlagDefinition, unit: !98, retainedNodes: !106)
// CHECK:STDOUT: !100 = !DILocalVariable(arg: 1, scope: !99, type: !21)
// CHECK:STDOUT: !104 = !DILocation(line: 7, column: 10, scope: !99)
// CHECK:STDOUT: !105 = !DILocation(line: 7, column: 3, scope: !99)
// CHECK:STDOUT: !106 = !{!100}
// CHECK:STDOUT: !107 = distinct !DISubprogram(name: "AccessP", linkageName: "_CAccessP.Main", scope: null, file: !97, line: 10, type: !23, spFlags: DISPFlagDefinition, unit: !98, retainedNodes: !115)
// CHECK:STDOUT: !108 = !DILocalVariable(arg: 1, scope: !107, type: !21)
// CHECK:STDOUT: !112 = !DILocation(line: 11, column: 11, scope: !107)
// CHECK:STDOUT: !113 = !DILocation(line: 11, column: 10, scope: !107)
// CHECK:STDOUT: !114 = !DILocation(line: 11, column: 3, scope: !107)
// CHECK:STDOUT: !115 = !{!108}
// CHECK:STDOUT:
