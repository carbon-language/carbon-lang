// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/generic/local_function.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/generic/local_function.carbon

// Test that calling a local function defined inside a generic function works.
// This previously crashed due to the local function not having a parent name
// scope.

fn F(T:! Core.Copy, y: T) -> T {
  fn G(x: T) -> T { return x; }
  return G(y);
}

fn Run() -> i32 {
  return F(i32, 0);
}

// CHECK:STDOUT: ; ModuleID = 'local_function.carbon'
// CHECK:STDOUT: source_filename = "local_function.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @main() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main.4bee6dcdea6ba756(i32 0), !dbg !8
// CHECK:STDOUT:   ret i32 %F.call, !dbg !9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.4bee6dcdea6ba756(i32 %y) #0 !dbg !10 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call i32 @"_CG:enclosed.4bee6dcdea6ba756"(i32 %y), !dbg !15
// CHECK:STDOUT:   ret i32 %G.call, !dbg !16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CG:enclosed.4bee6dcdea6ba756"(i32 %x) #0 !dbg !17 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %x, !dbg !20
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
// CHECK:STDOUT: !3 = !DIFile(filename: "local_function.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !3, line: 22, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7}
// CHECK:STDOUT: !7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !8 = !DILocation(line: 23, column: 10, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 23, column: 3, scope: !4)
// CHECK:STDOUT: !10 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.4bee6dcdea6ba756", scope: null, file: !3, line: 17, type: !11, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !13)
// CHECK:STDOUT: !11 = !DISubroutineType(types: !12)
// CHECK:STDOUT: !12 = !{!7, !7}
// CHECK:STDOUT: !13 = !{!14}
// CHECK:STDOUT: !14 = !DILocalVariable(arg: 1, scope: !10, type: !7)
// CHECK:STDOUT: !15 = !DILocation(line: 19, column: 10, scope: !10)
// CHECK:STDOUT: !16 = !DILocation(line: 19, column: 3, scope: !10)
// CHECK:STDOUT: !17 = distinct !DISubprogram(name: "G", linkageName: "_CG:enclosed.4bee6dcdea6ba756", scope: null, file: !3, line: 18, type: !11, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !18)
// CHECK:STDOUT: !18 = !{!19}
// CHECK:STDOUT: !19 = !DILocalVariable(arg: 1, scope: !17, type: !7)
// CHECK:STDOUT: !20 = !DILocation(line: 18, column: 21, scope: !17)
