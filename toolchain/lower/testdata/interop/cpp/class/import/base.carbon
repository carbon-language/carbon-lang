// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/class/import/base.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/class/import/base.carbon

// --- structs.h

struct A {
  int a;

  void f() {}
};

struct B : A {
  int b;
};

// --- convert.carbon

library "[[@TEST_NAME]]";

import Cpp library "structs.h";

fn ConvertPtr(p: Cpp.B*) -> Cpp.A* {
  return p;
}

fn AcceptVal(a: Cpp.A);

fn ConvertVal(b: Cpp.B) {
  AcceptVal(b);
}

// --- access_field.carbon

library "[[@TEST_NAME]]";

import Cpp library "structs.h";

fn AccessVal(b: Cpp.B) -> i32 {
  return b.a;
}

// --- call_method.carbon

library "[[@TEST_NAME]]";

import Cpp library "structs.h";

fn Call(b: Cpp.B*) {
  b->f();
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'convert.carbon'
// CHECK:STDOUT: source_filename = "convert.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CConvertPtr.Main(ptr %p) #0 !dbg !26 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_11.2.base = getelementptr inbounds nuw [8 x i8], ptr %p, i32 0, i32 0, !dbg !32
// CHECK:STDOUT:   ret ptr %.loc7_11.2.base, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CAcceptVal.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvertVal.Main(ptr %b) #0 !dbg !34 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc13_13.2.a = getelementptr inbounds nuw [4 x i8], ptr %b, i32 0, i32 0, !dbg !37
// CHECK:STDOUT:   call void @_CAcceptVal.Main(ptr %.loc13_13.2.a), !dbg !38
// CHECK:STDOUT:   ret void, !dbg !39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!41, !42, !43, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!48}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !18 = !DIFile(filename: "convert.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{!21, !21}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !24 = !{null, !21}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "ConvertPtr", linkageName: "_CConvertPtr.Main", scope: null, file: !18, line: 6, type: !23, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !33)
// CHECK:STDOUT: !27 = !DILocalVariable(arg: 1, scope: !26, type: !21)
// CHECK:STDOUT: !32 = !DILocation(line: 7, column: 3, scope: !26)
// CHECK:STDOUT: !33 = !{!27}
// CHECK:STDOUT: !34 = distinct !DISubprogram(name: "ConvertVal", linkageName: "_CConvertVal.Main", scope: null, file: !18, line: 12, type: !25, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !40)
// CHECK:STDOUT: !35 = !DILocalVariable(arg: 1, scope: !34, type: !21)
// CHECK:STDOUT: !37 = !DILocation(line: 13, column: 13, scope: !34)
// CHECK:STDOUT: !38 = !DILocation(line: 13, column: 3, scope: !34)
// CHECK:STDOUT: !39 = !DILocation(line: 12, column: 1, scope: !34)
// CHECK:STDOUT: !40 = !{!35}
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
// CHECK:STDOUT: ; ModuleID = 'access_field.carbon'
// CHECK:STDOUT: source_filename = "access_field.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CAccessVal.Main(ptr %b) #0 !dbg !53 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_11.2.a = getelementptr inbounds nuw [4 x i8], ptr %b, i32 0, i32 0, !dbg !58
// CHECK:STDOUT:   %.loc7_11.4.a = getelementptr inbounds nuw [4 x i8], ptr %.loc7_11.2.a, i32 0, i32 0, !dbg !58
// CHECK:STDOUT:   %.loc7_11.5 = load i32, ptr %.loc7_11.4.a, align 4, !dbg !58
// CHECK:STDOUT:   ret i32 %.loc7_11.5, !dbg !59
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!50}
// CHECK:STDOUT: !llvm.module.flags = !{!41, !42, !43, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!48}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !41 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !42 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !43 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !44 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !45 = !{!"omnipotent char", !44, i64 0}
// CHECK:STDOUT: !46 = !{!"int", !45, i64 0}
// CHECK:STDOUT: !47 = !{!"__libc_errno", !46, i64 0}
// CHECK:STDOUT: !48 = !{!47, !46, i64 0}
// CHECK:STDOUT: !49 = !DIFile(filename: "access_field.carbon", directory: "")
// CHECK:STDOUT: !50 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !49, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !51 = !{!20, !21}
// CHECK:STDOUT: !52 = !DISubroutineType(types: !51)
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "AccessVal", linkageName: "_CAccessVal.Main", scope: null, file: !49, line: 6, type: !52, spFlags: DISPFlagDefinition, unit: !50, retainedNodes: !60)
// CHECK:STDOUT: !54 = !DILocalVariable(arg: 1, scope: !53, type: !21)
// CHECK:STDOUT: !58 = !DILocation(line: 7, column: 10, scope: !53)
// CHECK:STDOUT: !59 = !DILocation(line: 7, column: 3, scope: !53)
// CHECK:STDOUT: !60 = !{!54}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'call_method.carbon'
// CHECK:STDOUT: source_filename = "call_method.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN1A1fEv = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCall.Main(ptr %b) #0 !dbg !63 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_4.3.a = getelementptr inbounds nuw [4 x i8], ptr %b, i32 0, i32 0, !dbg !66
// CHECK:STDOUT:   call void @_ZN1A1fEv(ptr %.loc7_4.3.a), !dbg !66
// CHECK:STDOUT:   ret void, !dbg !67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN1A1fEv(ptr noundef nonnull align 4 dereferenceable(4) %this) #1 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !71
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!62}
// CHECK:STDOUT: !llvm.module.flags = !{!41, !42, !43, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!48}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{null, !21}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !41 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !42 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !43 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !44 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !45 = !{!"omnipotent char", !44, i64 0}
// CHECK:STDOUT: !46 = !{!"int", !45, i64 0}
// CHECK:STDOUT: !47 = !{!"__libc_errno", !46, i64 0}
// CHECK:STDOUT: !48 = !{!47, !46, i64 0}
// CHECK:STDOUT: !61 = !DIFile(filename: "call_method.carbon", directory: "")
// CHECK:STDOUT: !62 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !61, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !63 = distinct !DISubprogram(name: "Call", linkageName: "_CCall.Main", scope: null, file: !61, line: 6, type: !25, spFlags: DISPFlagDefinition, unit: !62, retainedNodes: !68)
// CHECK:STDOUT: !64 = !DILocalVariable(arg: 1, scope: !63, type: !21)
// CHECK:STDOUT: !66 = !DILocation(line: 7, column: 3, scope: !63)
// CHECK:STDOUT: !67 = !DILocation(line: 6, column: 1, scope: !63)
// CHECK:STDOUT: !68 = !{!64}
// CHECK:STDOUT: !69 = !{!"any pointer", !45, i64 0}
// CHECK:STDOUT: !70 = !{!"p1 _ZTS1A", !69, i64 0}
// CHECK:STDOUT: !71 = !{!70, !70, i64 0}
// CHECK:STDOUT:
