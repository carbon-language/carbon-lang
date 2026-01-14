// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/base.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/base.carbon

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

// CHECK:STDOUT: ; ModuleID = 'convert.carbon'
// CHECK:STDOUT: source_filename = "convert.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CConvertPtr.Main(ptr %p) #0 !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_11.2.base = getelementptr inbounds nuw [8 x i8], ptr %p, i32 0, i32 0, !dbg !18
// CHECK:STDOUT:   ret ptr %.loc7_11.2.base, !dbg !18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CAcceptVal.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvertVal.Main(ptr %b) #0 !dbg !19 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc13_13.1.base = getelementptr inbounds nuw [8 x i8], ptr %b, i32 0, i32 0, !dbg !24
// CHECK:STDOUT:   call void @_CAcceptVal.Main(ptr %.loc13_13.1.base), !dbg !25
// CHECK:STDOUT:   ret void, !dbg !26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
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
// CHECK:STDOUT: !7 = !DIFile(filename: "convert.carbon", directory: "")
// CHECK:STDOUT: !8 = !{!9, !9, i64 0}
// CHECK:STDOUT: !9 = !{!"int", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"omnipotent char", !11, i64 0}
// CHECK:STDOUT: !11 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "ConvertPtr", linkageName: "_CConvertPtr.Main", scope: null, file: !7, line: 6, type: !13, spFlags: DISPFlagDefinition, unit: !6, retainedNodes: !16)
// CHECK:STDOUT: !13 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !14 = !{!15, !15}
// CHECK:STDOUT: !15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 8)
// CHECK:STDOUT: !16 = !{!17}
// CHECK:STDOUT: !17 = !DILocalVariable(arg: 1, scope: !12, type: !15)
// CHECK:STDOUT: !18 = !DILocation(line: 7, column: 3, scope: !12)
// CHECK:STDOUT: !19 = distinct !DISubprogram(name: "ConvertVal", linkageName: "_CConvertVal.Main", scope: null, file: !7, line: 12, type: !20, spFlags: DISPFlagDefinition, unit: !6, retainedNodes: !22)
// CHECK:STDOUT: !20 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !21 = !{null, !15}
// CHECK:STDOUT: !22 = !{!23}
// CHECK:STDOUT: !23 = !DILocalVariable(arg: 1, scope: !19, type: !15)
// CHECK:STDOUT: !24 = !DILocation(line: 13, column: 13, scope: !19)
// CHECK:STDOUT: !25 = !DILocation(line: 13, column: 3, scope: !19)
// CHECK:STDOUT: !26 = !DILocation(line: 12, column: 1, scope: !19)
// CHECK:STDOUT: ; ModuleID = 'access_field.carbon'
// CHECK:STDOUT: source_filename = "access_field.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CAccessVal.Main(ptr %b) #0 !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_11.1.base = getelementptr inbounds nuw [8 x i8], ptr %b, i32 0, i32 0, !dbg !19
// CHECK:STDOUT:   %.loc7_11.3.a = getelementptr inbounds nuw [4 x i8], ptr %.loc7_11.1.base, i32 0, i32 0, !dbg !19
// CHECK:STDOUT:   %.loc7_11.4 = load i32, ptr %.loc7_11.3.a, align 4, !dbg !19
// CHECK:STDOUT:   ret i32 %.loc7_11.4, !dbg !20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
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
// CHECK:STDOUT: !7 = !DIFile(filename: "access_field.carbon", directory: "")
// CHECK:STDOUT: !8 = !{!9, !9, i64 0}
// CHECK:STDOUT: !9 = !{!"int", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"omnipotent char", !11, i64 0}
// CHECK:STDOUT: !11 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "AccessVal", linkageName: "_CAccessVal.Main", scope: null, file: !7, line: 6, type: !13, spFlags: DISPFlagDefinition, unit: !6, retainedNodes: !17)
// CHECK:STDOUT: !13 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !14 = !{!15, !16}
// CHECK:STDOUT: !15 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !16 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 8)
// CHECK:STDOUT: !17 = !{!18}
// CHECK:STDOUT: !18 = !DILocalVariable(arg: 1, scope: !12, type: !16)
// CHECK:STDOUT: !19 = !DILocation(line: 7, column: 10, scope: !12)
// CHECK:STDOUT: !20 = !DILocation(line: 7, column: 3, scope: !12)
// CHECK:STDOUT: ; ModuleID = 'call_method.carbon'
// CHECK:STDOUT: source_filename = "call_method.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN1A1fEv = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCall.Main(ptr %b) #0 !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_4.2.base = getelementptr inbounds nuw [8 x i8], ptr %b, i32 0, i32 0, !dbg !18
// CHECK:STDOUT:   call void @_ZN1A1fEv(ptr %.loc7_4.2.base), !dbg !18
// CHECK:STDOUT:   ret void, !dbg !19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN1A1fEv(ptr noundef nonnull align 4 dereferenceable(4) %this) #1 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !20
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
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
// CHECK:STDOUT: !7 = !DIFile(filename: "call_method.carbon", directory: "")
// CHECK:STDOUT: !8 = !{!9, !9, i64 0}
// CHECK:STDOUT: !9 = !{!"int", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"omnipotent char", !11, i64 0}
// CHECK:STDOUT: !11 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "Call", linkageName: "_CCall.Main", scope: null, file: !7, line: 6, type: !13, spFlags: DISPFlagDefinition, unit: !6, retainedNodes: !16)
// CHECK:STDOUT: !13 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !14 = !{null, !15}
// CHECK:STDOUT: !15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 8)
// CHECK:STDOUT: !16 = !{!17}
// CHECK:STDOUT: !17 = !DILocalVariable(arg: 1, scope: !12, type: !15)
// CHECK:STDOUT: !18 = !DILocation(line: 7, column: 3, scope: !12)
// CHECK:STDOUT: !19 = !DILocation(line: 6, column: 1, scope: !12)
// CHECK:STDOUT: !20 = !{!21, !21, i64 0}
// CHECK:STDOUT: !21 = !{!"p1 _ZTS1A", !22, i64 0}
// CHECK:STDOUT: !22 = !{!"any pointer", !10, i64 0}
