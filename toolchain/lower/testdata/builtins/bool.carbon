// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/builtins/bool.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/builtins/bool.carbon

fn Bool() -> type = "bool.make_type";

fn Eq(a: Bool(), b: Bool()) -> Bool() = "bool.eq";
fn TestEq(a: Bool(), b: Bool()) -> Bool() { return Eq(a, b); }

fn Neq(a: Bool(), b: Bool()) -> Bool() = "bool.neq";
fn TestNeq(a: Bool(), b: Bool()) -> Bool() { return Neq(a, b); }

fn IfEq(a: Bool(), b: Bool()) -> Bool() {
  if (Eq(a, b)) { return true; }
  return false;
}

// CHECK:STDOUT: ; ModuleID = 'bool.carbon'
// CHECK:STDOUT: source_filename = "bool.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestEq.Main(i1 %a, i1 %b) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Eq.call = icmp eq i1 %a, %b, !dbg !11
// CHECK:STDOUT:   ret i1 %Eq.call, !dbg !12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestNeq.Main(i1 %a, i1 %b) #0 !dbg !13 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Neq.call = icmp ne i1 %a, %b, !dbg !17
// CHECK:STDOUT:   ret i1 %Neq.call, !dbg !18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CIfEq.Main(i1 %a, i1 %b) #0 !dbg !19 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Eq.call = icmp eq i1 %a, %b, !dbg !23
// CHECK:STDOUT:   br i1 %Eq.call, label %if.then, label %if.else, !dbg !24
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret i1 true, !dbg !25
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   ret i1 false, !dbg !26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "bool.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "TestEq", linkageName: "_CTestEq.Main", scope: null, file: !1, line: 16, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !8)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7, !7, !7}
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{!9, !10}
// CHECK:STDOUT: !9 = !DILocalVariable(arg: 1, scope: !4, type: !7)
// CHECK:STDOUT: !10 = !DILocalVariable(arg: 2, scope: !4, type: !7)
// CHECK:STDOUT: !11 = !DILocation(line: 16, column: 52, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 16, column: 45, scope: !4)
// CHECK:STDOUT: !13 = distinct !DISubprogram(name: "TestNeq", linkageName: "_CTestNeq.Main", scope: null, file: !1, line: 19, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !14)
// CHECK:STDOUT: !14 = !{!15, !16}
// CHECK:STDOUT: !15 = !DILocalVariable(arg: 1, scope: !13, type: !7)
// CHECK:STDOUT: !16 = !DILocalVariable(arg: 2, scope: !13, type: !7)
// CHECK:STDOUT: !17 = !DILocation(line: 19, column: 53, scope: !13)
// CHECK:STDOUT: !18 = !DILocation(line: 19, column: 46, scope: !13)
// CHECK:STDOUT: !19 = distinct !DISubprogram(name: "IfEq", linkageName: "_CIfEq.Main", scope: null, file: !1, line: 21, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !20)
// CHECK:STDOUT: !20 = !{!21, !22}
// CHECK:STDOUT: !21 = !DILocalVariable(arg: 1, scope: !19, type: !7)
// CHECK:STDOUT: !22 = !DILocalVariable(arg: 2, scope: !19, type: !7)
// CHECK:STDOUT: !23 = !DILocation(line: 22, column: 7, scope: !19)
// CHECK:STDOUT: !24 = !DILocation(line: 22, column: 6, scope: !19)
// CHECK:STDOUT: !25 = !DILocation(line: 22, column: 19, scope: !19)
// CHECK:STDOUT: !26 = !DILocation(line: 23, column: 3, scope: !19)
