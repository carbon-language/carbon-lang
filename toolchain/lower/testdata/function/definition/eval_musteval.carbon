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

// CHECK:STDOUT: ; ModuleID = 'eval.carbon'
// CHECK:STDOUT: source_filename = "eval.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CEval.Main(i32 %a) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %a, !dbg !10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CF.Main() #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 1, !dbg !14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CG.Main(i32 %n) #0 !dbg !15 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Eval.call = call i32 @_CEval.Main(i32 %n), !dbg !18
// CHECK:STDOUT:   ret i32 %Eval.call, !dbg !19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "eval.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Eval", linkageName: "_CEval.Main", scope: null, file: !3, line: 5, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !8)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7, !7}
// CHECK:STDOUT: !7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !8 = !{!9}
// CHECK:STDOUT: !9 = !DILocalVariable(arg: 1, scope: !4, type: !7)
// CHECK:STDOUT: !10 = !DILocation(line: 6, column: 3, scope: !4)
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !3, line: 9, type: !12, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !12 = !DISubroutineType(types: !13)
// CHECK:STDOUT: !13 = !{!7}
// CHECK:STDOUT: !14 = !DILocation(line: 11, column: 3, scope: !11)
// CHECK:STDOUT: !15 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !3, line: 14, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !16)
// CHECK:STDOUT: !16 = !{!17}
// CHECK:STDOUT: !17 = !DILocalVariable(arg: 1, scope: !15, type: !7)
// CHECK:STDOUT: !18 = !DILocation(line: 16, column: 10, scope: !15)
// CHECK:STDOUT: !19 = !DILocation(line: 16, column: 3, scope: !15)
// CHECK:STDOUT: ; ModuleID = 'musteval.carbon'
// CHECK:STDOUT: source_filename = "musteval.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CF.Main() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 1, !dbg !8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "musteval.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !3, line: 10, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7}
// CHECK:STDOUT: !7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !8 = !DILocation(line: 11, column: 3, scope: !4)
