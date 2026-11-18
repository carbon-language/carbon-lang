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

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'bool.carbon'
// CHECK:STDOUT: source_filename = "bool.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestEq.Main(i1 %a, i1 %b) #0 !dbg !19 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Eq.call = icmp eq i1 %a, %b, !dbg !28
// CHECK:STDOUT:   ret i1 %Eq.call, !dbg !29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestNeq.Main(i1 %a, i1 %b) #0 !dbg !31 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Neq.call = icmp ne i1 %a, %b, !dbg !40
// CHECK:STDOUT:   ret i1 %Neq.call, !dbg !41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CIfEq.Main(i1 %a, i1 %b) #0 !dbg !43 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Eq.call = icmp eq i1 %a, %b, !dbg !52
// CHECK:STDOUT:   br i1 %Eq.call, label %if.then, label %if.else, !dbg !53
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret i1 true, !dbg !54
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   ret i1 false, !dbg !55
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!15}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !14 = !DIFile(filename: "bool.carbon", directory: "")
// CHECK:STDOUT: !15 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !14, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !16 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !17 = !{!16, !16, !16}
// CHECK:STDOUT: !18 = !DISubroutineType(types: !17)
// CHECK:STDOUT: !19 = distinct !DISubprogram(name: "TestEq", linkageName: "_CTestEq.Main", scope: null, file: !14, line: 16, type: !18, spFlags: DISPFlagDefinition, unit: !15, retainedNodes: !30)
// CHECK:STDOUT: !20 = !DILocalVariable(arg: 1, scope: !19, type: !16)
// CHECK:STDOUT: !21 = !DILocalVariable(arg: 2, scope: !19, type: !16)
// CHECK:STDOUT: !28 = !DILocation(line: 16, column: 52, scope: !19)
// CHECK:STDOUT: !29 = !DILocation(line: 16, column: 45, scope: !19)
// CHECK:STDOUT: !30 = !{!20, !21}
// CHECK:STDOUT: !31 = distinct !DISubprogram(name: "TestNeq", linkageName: "_CTestNeq.Main", scope: null, file: !14, line: 19, type: !18, spFlags: DISPFlagDefinition, unit: !15, retainedNodes: !42)
// CHECK:STDOUT: !32 = !DILocalVariable(arg: 1, scope: !31, type: !16)
// CHECK:STDOUT: !33 = !DILocalVariable(arg: 2, scope: !31, type: !16)
// CHECK:STDOUT: !40 = !DILocation(line: 19, column: 53, scope: !31)
// CHECK:STDOUT: !41 = !DILocation(line: 19, column: 46, scope: !31)
// CHECK:STDOUT: !42 = !{!32, !33}
// CHECK:STDOUT: !43 = distinct !DISubprogram(name: "IfEq", linkageName: "_CIfEq.Main", scope: null, file: !14, line: 21, type: !18, spFlags: DISPFlagDefinition, unit: !15, retainedNodes: !56)
// CHECK:STDOUT: !44 = !DILocalVariable(arg: 1, scope: !43, type: !16)
// CHECK:STDOUT: !45 = !DILocalVariable(arg: 2, scope: !43, type: !16)
// CHECK:STDOUT: !52 = !DILocation(line: 22, column: 7, scope: !43)
// CHECK:STDOUT: !53 = !DILocation(line: 22, column: 6, scope: !43)
// CHECK:STDOUT: !54 = !DILocation(line: 22, column: 19, scope: !43)
// CHECK:STDOUT: !55 = !DILocation(line: 23, column: 3, scope: !43)
// CHECK:STDOUT: !56 = !{!44, !45}
// CHECK:STDOUT:
