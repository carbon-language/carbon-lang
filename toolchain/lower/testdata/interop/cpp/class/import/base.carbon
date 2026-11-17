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
// CHECK:STDOUT: define ptr @_CConvertPtr.Main(ptr %p) #0 !dbg !28 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_11.2.base = getelementptr inbounds nuw [8 x i8], ptr %p, i32 0, i32 0, !dbg !34
// CHECK:STDOUT:   ret ptr %.loc7_11.2.base, !dbg !34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CAcceptVal.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvertVal.Main(ptr %b) #0 !dbg !36 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc13_13.1.base = getelementptr inbounds nuw [8 x i8], ptr %b, i32 0, i32 0, !dbg !39
// CHECK:STDOUT:   call void @_CAcceptVal.Main(ptr %.loc13_13.1.base), !dbg !40
// CHECK:STDOUT:   ret void, !dbg !41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!21}
// CHECK:STDOUT: !llvm.module.flags = !{!43, !44, !45, !46, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!51}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIFile(filename: "convert.carbon", directory: "")
// CHECK:STDOUT: !21 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !20, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{!23, !23}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = !{null, !23}
// CHECK:STDOUT: !27 = !DISubroutineType(types: !26)
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "ConvertPtr", linkageName: "_CConvertPtr.Main", scope: null, file: !20, line: 6, type: !25, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !35)
// CHECK:STDOUT: !29 = !DILocalVariable(arg: 1, scope: !28, type: !23)
// CHECK:STDOUT: !34 = !DILocation(line: 7, column: 3, scope: !28)
// CHECK:STDOUT: !35 = !{!29}
// CHECK:STDOUT: !36 = distinct !DISubprogram(name: "ConvertVal", linkageName: "_CConvertVal.Main", scope: null, file: !20, line: 12, type: !27, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !42)
// CHECK:STDOUT: !37 = !DILocalVariable(arg: 1, scope: !36, type: !23)
// CHECK:STDOUT: !39 = !DILocation(line: 13, column: 13, scope: !36)
// CHECK:STDOUT: !40 = !DILocation(line: 13, column: 3, scope: !36)
// CHECK:STDOUT: !41 = !DILocation(line: 12, column: 1, scope: !36)
// CHECK:STDOUT: !42 = !{!37}
// CHECK:STDOUT: !43 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !44 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !45 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !46 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !47 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !48 = !{!"omnipotent char", !47, i64 0}
// CHECK:STDOUT: !49 = !{!"int", !48, i64 0}
// CHECK:STDOUT: !50 = !{!"__libc_errno", !49, i64 0}
// CHECK:STDOUT: !51 = !{!50, !49, i64 0}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'access_field.carbon'
// CHECK:STDOUT: source_filename = "access_field.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CAccessVal.Main(ptr %b) #0 !dbg !56 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_11.1.base = getelementptr inbounds nuw [8 x i8], ptr %b, i32 0, i32 0, !dbg !61
// CHECK:STDOUT:   %.loc7_11.3.a = getelementptr inbounds nuw [4 x i8], ptr %.loc7_11.1.base, i32 0, i32 0, !dbg !61
// CHECK:STDOUT:   %.loc7_11.4 = load i32, ptr %.loc7_11.3.a, align 4, !dbg !61
// CHECK:STDOUT:   ret i32 %.loc7_11.4, !dbg !62
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!53}
// CHECK:STDOUT: !llvm.module.flags = !{!43, !44, !45, !46, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!51}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !22 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !43 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !44 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !45 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !46 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !47 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !48 = !{!"omnipotent char", !47, i64 0}
// CHECK:STDOUT: !49 = !{!"int", !48, i64 0}
// CHECK:STDOUT: !50 = !{!"__libc_errno", !49, i64 0}
// CHECK:STDOUT: !51 = !{!50, !49, i64 0}
// CHECK:STDOUT: !52 = !DIFile(filename: "access_field.carbon", directory: "")
// CHECK:STDOUT: !53 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !52, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !54 = !{!22, !23}
// CHECK:STDOUT: !55 = !DISubroutineType(types: !54)
// CHECK:STDOUT: !56 = distinct !DISubprogram(name: "AccessVal", linkageName: "_CAccessVal.Main", scope: null, file: !52, line: 6, type: !55, spFlags: DISPFlagDefinition, unit: !53, retainedNodes: !63)
// CHECK:STDOUT: !57 = !DILocalVariable(arg: 1, scope: !56, type: !23)
// CHECK:STDOUT: !61 = !DILocation(line: 7, column: 10, scope: !56)
// CHECK:STDOUT: !62 = !DILocation(line: 7, column: 3, scope: !56)
// CHECK:STDOUT: !63 = !{!57}
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
// CHECK:STDOUT: define void @_CCall.Main(ptr %b) #0 !dbg !66 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_4.2.base = getelementptr inbounds nuw [8 x i8], ptr %b, i32 0, i32 0, !dbg !69
// CHECK:STDOUT:   call void @_ZN1A1fEv(ptr %.loc7_4.2.base), !dbg !69
// CHECK:STDOUT:   ret void, !dbg !70
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN1A1fEv(ptr noundef nonnull align 4 dereferenceable(4) %this) #1 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !74
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!65}
// CHECK:STDOUT: !llvm.module.flags = !{!43, !44, !45, !46, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!51}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !26 = !{null, !23}
// CHECK:STDOUT: !27 = !DISubroutineType(types: !26)
// CHECK:STDOUT: !43 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !44 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !45 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !46 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !47 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !48 = !{!"omnipotent char", !47, i64 0}
// CHECK:STDOUT: !49 = !{!"int", !48, i64 0}
// CHECK:STDOUT: !50 = !{!"__libc_errno", !49, i64 0}
// CHECK:STDOUT: !51 = !{!50, !49, i64 0}
// CHECK:STDOUT: !64 = !DIFile(filename: "call_method.carbon", directory: "")
// CHECK:STDOUT: !65 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !64, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !66 = distinct !DISubprogram(name: "Call", linkageName: "_CCall.Main", scope: null, file: !64, line: 6, type: !27, spFlags: DISPFlagDefinition, unit: !65, retainedNodes: !71)
// CHECK:STDOUT: !67 = !DILocalVariable(arg: 1, scope: !66, type: !23)
// CHECK:STDOUT: !69 = !DILocation(line: 7, column: 3, scope: !66)
// CHECK:STDOUT: !70 = !DILocation(line: 6, column: 1, scope: !66)
// CHECK:STDOUT: !71 = !{!67}
// CHECK:STDOUT: !72 = !{!"any pointer", !48, i64 0}
// CHECK:STDOUT: !73 = !{!"p1 _ZTS1A", !72, i64 0}
// CHECK:STDOUT: !74 = !{!73, !73, i64 0}
// CHECK:STDOUT:
