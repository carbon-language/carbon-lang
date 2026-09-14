// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/if_expr/basic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/if_expr/basic.carbon

fn F() -> i32 { return 1; }
fn G() -> i32 { return 2; }

fn Select(b: bool) -> i32 {
  return if b then F() else G();
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'basic.carbon'
// CHECK:STDOUT: source_filename = "basic.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CF.Main() #0 !dbg !40 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 1, !dbg !43
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CG.Main() #0 !dbg !44 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 2, !dbg !47
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CSelect.Main(i1 %b) #0 !dbg !48 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   br i1 %b, label %if.expr.then, label %if.expr.else, !dbg !54
// CHECK:STDOUT:
// CHECK:STDOUT: if.expr.then:                                     ; preds = %entry
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main(), !dbg !55
// CHECK:STDOUT:   br label %if.expr.result, !dbg !54
// CHECK:STDOUT:
// CHECK:STDOUT: if.expr.else:                                     ; preds = %entry
// CHECK:STDOUT:   %G.call = call i32 @_CG.Main(), !dbg !56
// CHECK:STDOUT:   br label %if.expr.result, !dbg !54
// CHECK:STDOUT:
// CHECK:STDOUT: if.expr.result:                                   ; preds = %if.expr.else, %if.expr.then
// CHECK:STDOUT:   %0 = phi i32 [ %F.call, %if.expr.then ], [ %G.call, %if.expr.else ]
// CHECK:STDOUT:   ret i32 %0, !dbg !57
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!33}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !32 = !DIFile(filename: "basic.carbon", directory: "")
// CHECK:STDOUT: !33 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !32, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !34 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !35 = !{!34}
// CHECK:STDOUT: !36 = !DISubroutineType(types: !35)
// CHECK:STDOUT: !37 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !38 = !{!34, !37}
// CHECK:STDOUT: !39 = !DISubroutineType(types: !38)
// CHECK:STDOUT: !40 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !32, line: 13, type: !36, spFlags: DISPFlagDefinition, unit: !33)
// CHECK:STDOUT: !43 = !DILocation(line: 13, column: 17, scope: !40)
// CHECK:STDOUT: !44 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !32, line: 14, type: !36, spFlags: DISPFlagDefinition, unit: !33)
// CHECK:STDOUT: !47 = !DILocation(line: 14, column: 17, scope: !44)
// CHECK:STDOUT: !48 = distinct !DISubprogram(name: "Select", linkageName: "_CSelect.Main", scope: null, file: !32, line: 16, type: !39, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !58)
// CHECK:STDOUT: !49 = !DILocalVariable(arg: 1, scope: !48, type: !37)
// CHECK:STDOUT: !54 = !DILocation(line: 17, column: 10, scope: !48)
// CHECK:STDOUT: !55 = !DILocation(line: 17, column: 20, scope: !48)
// CHECK:STDOUT: !56 = !DILocation(line: 17, column: 29, scope: !48)
// CHECK:STDOUT: !57 = !DILocation(line: 17, column: 3, scope: !48)
// CHECK:STDOUT: !58 = !{!49}
// CHECK:STDOUT:
