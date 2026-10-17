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
// CHECK:STDOUT: define i1 @_CTestEq.Main(i1 %a, i1 %b) #0 !dbg !21 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Eq.call = icmp eq i1 %a, %b, !dbg !30
// CHECK:STDOUT:   ret i1 %Eq.call, !dbg !31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestNeq.Main(i1 %a, i1 %b) #0 !dbg !33 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Neq.call = icmp ne i1 %a, %b, !dbg !42
// CHECK:STDOUT:   ret i1 %Neq.call, !dbg !43
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CIfEq.Main(i1 %a, i1 %b) #0 !dbg !45 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Eq.call = icmp eq i1 %a, %b, !dbg !54
// CHECK:STDOUT:   br i1 %Eq.call, label %if.then, label %if.else, !dbg !55
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret i1 true, !dbg !56
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   ret i1 false, !dbg !57
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!17}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !16 = !DIFile(filename: "bool.carbon", directory: "")
// CHECK:STDOUT: !17 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !16, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !18 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !19 = !{!18, !18, !18}
// CHECK:STDOUT: !20 = !DISubroutineType(types: !19)
// CHECK:STDOUT: !21 = distinct !DISubprogram(name: "TestEq", linkageName: "_CTestEq.Main", scope: null, file: !16, line: 16, type: !20, spFlags: DISPFlagDefinition, unit: !17, retainedNodes: !32)
// CHECK:STDOUT: !22 = !DILocalVariable(arg: 1, scope: !21, type: !18)
// CHECK:STDOUT: !23 = !DILocalVariable(arg: 2, scope: !21, type: !18)
// CHECK:STDOUT: !30 = !DILocation(line: 16, column: 52, scope: !21)
// CHECK:STDOUT: !31 = !DILocation(line: 16, column: 45, scope: !21)
// CHECK:STDOUT: !32 = !{!22, !23}
// CHECK:STDOUT: !33 = distinct !DISubprogram(name: "TestNeq", linkageName: "_CTestNeq.Main", scope: null, file: !16, line: 19, type: !20, spFlags: DISPFlagDefinition, unit: !17, retainedNodes: !44)
// CHECK:STDOUT: !34 = !DILocalVariable(arg: 1, scope: !33, type: !18)
// CHECK:STDOUT: !35 = !DILocalVariable(arg: 2, scope: !33, type: !18)
// CHECK:STDOUT: !42 = !DILocation(line: 19, column: 53, scope: !33)
// CHECK:STDOUT: !43 = !DILocation(line: 19, column: 46, scope: !33)
// CHECK:STDOUT: !44 = !{!34, !35}
// CHECK:STDOUT: !45 = distinct !DISubprogram(name: "IfEq", linkageName: "_CIfEq.Main", scope: null, file: !16, line: 21, type: !20, spFlags: DISPFlagDefinition, unit: !17, retainedNodes: !58)
// CHECK:STDOUT: !46 = !DILocalVariable(arg: 1, scope: !45, type: !18)
// CHECK:STDOUT: !47 = !DILocalVariable(arg: 2, scope: !45, type: !18)
// CHECK:STDOUT: !54 = !DILocation(line: 22, column: 7, scope: !45)
// CHECK:STDOUT: !55 = !DILocation(line: 22, column: 6, scope: !45)
// CHECK:STDOUT: !56 = !DILocation(line: 22, column: 19, scope: !45)
// CHECK:STDOUT: !57 = !DILocation(line: 23, column: 3, scope: !45)
// CHECK:STDOUT: !58 = !{!46, !47}
// CHECK:STDOUT:
