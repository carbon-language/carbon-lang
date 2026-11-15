// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/definition/eval_musteval.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/definition/eval_musteval.carbon

// --- eval.carbon

library "[[@TEST_NAME]]";

// Should emit a definition of `Eval`.
eval fn Eval(a: i32) -> i32 {
  return a;
}

fn F() -> i32 {
  // Should not emit a call to `Eval`.
  return Eval(1);
}

fn G(n: i32) -> i32 {
  // Should emit a call to `Eval`.
  return Eval(n);
}

// --- musteval.carbon

library "[[@TEST_NAME]]";

// Should not emit a definition of `MustEval`.
musteval fn MustEval(a: i32) -> i32 {
  return a;
}

// Should not emit a call to `MustEval`.
fn F() -> i32 {
  return MustEval(1);
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'eval.carbon'
// CHECK:STDOUT: source_filename = "eval.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CEval.Main(i32 %a) #0 !dbg !27 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %a, !dbg !33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CF.Main() #0 !dbg !35 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 1, !dbg !38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CG.Main(i32 %n) #0 !dbg !39 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Eval.call = call i32 @_CEval.Main(i32 %n), !dbg !45
// CHECK:STDOUT:   ret i32 %Eval.call, !dbg !46
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!21}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIFile(filename: "eval.carbon", directory: "")
// CHECK:STDOUT: !21 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !20, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !22 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !23 = !{!22, !22}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !25 = !{!22}
// CHECK:STDOUT: !26 = !DISubroutineType(types: !25)
// CHECK:STDOUT: !27 = distinct !DISubprogram(name: "Eval", linkageName: "_CEval.Main", scope: null, file: !20, line: 5, type: !24, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !34)
// CHECK:STDOUT: !28 = !DILocalVariable(arg: 1, scope: !27, type: !22)
// CHECK:STDOUT: !33 = !DILocation(line: 6, column: 3, scope: !27)
// CHECK:STDOUT: !34 = !{!28}
// CHECK:STDOUT: !35 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !20, line: 9, type: !26, spFlags: DISPFlagDefinition, unit: !21)
// CHECK:STDOUT: !38 = !DILocation(line: 11, column: 3, scope: !35)
// CHECK:STDOUT: !39 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !20, line: 14, type: !24, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !47)
// CHECK:STDOUT: !40 = !DILocalVariable(arg: 1, scope: !39, type: !22)
// CHECK:STDOUT: !45 = !DILocation(line: 16, column: 10, scope: !39)
// CHECK:STDOUT: !46 = !DILocation(line: 16, column: 3, scope: !39)
// CHECK:STDOUT: !47 = !{!40}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'musteval.carbon'
// CHECK:STDOUT: source_filename = "musteval.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CF.Main() #0 !dbg !50 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 1, !dbg !53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!49}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !22 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !25 = !{!22}
// CHECK:STDOUT: !26 = !DISubroutineType(types: !25)
// CHECK:STDOUT: !48 = !DIFile(filename: "musteval.carbon", directory: "")
// CHECK:STDOUT: !49 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !48, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !50 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !48, line: 10, type: !26, spFlags: DISPFlagDefinition, unit: !49)
// CHECK:STDOUT: !53 = !DILocation(line: 11, column: 3, scope: !50)
// CHECK:STDOUT:
