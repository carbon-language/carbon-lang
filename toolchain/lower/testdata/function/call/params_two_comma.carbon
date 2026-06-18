// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/call/params_two_comma.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/call/params_two_comma.carbon

fn Foo(_: i32, _: i32,) {}

fn Main() {
  Foo(1, 2);
  Foo(1, 2,);
}

// CHECK:STDOUT: ; ModuleID = 'params_two_comma.carbon'
// CHECK:STDOUT: source_filename = "params_two_comma.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CFoo.Main(i32 %_, i32 %_1) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMain.Main() #0 !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CFoo.Main(i32 1, i32 2), !dbg !15
// CHECK:STDOUT:   call void @_CFoo.Main(i32 1, i32 2), !dbg !16
// CHECK:STDOUT:   ret void, !dbg !17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "params_two_comma.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Foo", linkageName: "_CFoo.Main", scope: null, file: !1, line: 13, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !8)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null, !7, !7}
// CHECK:STDOUT: !7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !8 = !{!9, !10}
// CHECK:STDOUT: !9 = !DILocalVariable(arg: 1, scope: !4, type: !7)
// CHECK:STDOUT: !10 = !DILocalVariable(arg: 2, scope: !4, type: !7)
// CHECK:STDOUT: !11 = !DILocation(line: 13, column: 1, scope: !4)
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "Main", linkageName: "_CMain.Main", scope: null, file: !1, line: 15, type: !13, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !13 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !14 = !{null}
// CHECK:STDOUT: !15 = !DILocation(line: 16, column: 3, scope: !12)
// CHECK:STDOUT: !16 = !DILocation(line: 17, column: 3, scope: !12)
// CHECK:STDOUT: !17 = !DILocation(line: 15, column: 1, scope: !12)
