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

fn F(generic T: Core.Copy, y: T) -> T {
  fn G(x: T) -> T { return x; }
  return G(y);
}

fn Run() -> i32 {
  return F(i32, 0);
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'local_function.carbon'
// CHECK:STDOUT: source_filename = "local_function.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @main() #0 !dbg !37 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main.4d6ceefeb7f772f4(i32 0), !dbg !40
// CHECK:STDOUT:   ret i32 %F.call, !dbg !43
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.4d6ceefeb7f772f4(i32 %y) #0 !dbg !44 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call i32 @"_CG:enclosed.4d6ceefeb7f772f4"(i32 %y), !dbg !50
// CHECK:STDOUT:   ret i32 %G.call, !dbg !51
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CG:enclosed.4d6ceefeb7f772f4"(i32 %x) #0 !dbg !53 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %x, !dbg !59
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!33}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !32 = !DIFile(filename: "local_function.carbon", directory: "")
// CHECK:STDOUT: !33 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !32, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !34 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !35 = !{!34}
// CHECK:STDOUT: !36 = !DISubroutineType(types: !35)
// CHECK:STDOUT: !37 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !32, line: 22, type: !36, spFlags: DISPFlagDefinition, unit: !33)
// CHECK:STDOUT: !40 = !DILocation(line: 23, column: 10, scope: !37)
// CHECK:STDOUT: !41 = !{!34, !34}
// CHECK:STDOUT: !42 = !DISubroutineType(types: !41)
// CHECK:STDOUT: !43 = !DILocation(line: 23, column: 3, scope: !37)
// CHECK:STDOUT: !44 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.4d6ceefeb7f772f4", scope: null, file: !32, line: 17, type: !42, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !52)
// CHECK:STDOUT: !45 = !DILocalVariable(arg: 1, scope: !44, type: !34)
// CHECK:STDOUT: !50 = !DILocation(line: 19, column: 10, scope: !44)
// CHECK:STDOUT: !51 = !DILocation(line: 19, column: 3, scope: !44)
// CHECK:STDOUT: !52 = !{!45}
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "G", linkageName: "_CG:enclosed.4d6ceefeb7f772f4", scope: null, file: !32, line: 18, type: !42, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !60)
// CHECK:STDOUT: !54 = !DILocalVariable(arg: 1, scope: !53, type: !34)
// CHECK:STDOUT: !59 = !DILocation(line: 18, column: 21, scope: !53)
// CHECK:STDOUT: !60 = !{!54}
// CHECK:STDOUT:
