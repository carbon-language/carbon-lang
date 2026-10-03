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
// CHECK:STDOUT: define i32 @main() #0 !dbg !39 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main.4d6ceefeb7f772f4(i32 0), !dbg !42
// CHECK:STDOUT:   ret i32 %F.call, !dbg !45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.4d6ceefeb7f772f4(i32 %y) #0 !dbg !46 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call i32 @"_CG:enclosed.4d6ceefeb7f772f4"(i32 %y), !dbg !52
// CHECK:STDOUT:   ret i32 %G.call, !dbg !53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CG:enclosed.4d6ceefeb7f772f4"(i32 %x) #0 !dbg !55 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %x, !dbg !61
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!35}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !34 = !DIFile(filename: "local_function.carbon", directory: "")
// CHECK:STDOUT: !35 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !34, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !36 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !37 = !{!36}
// CHECK:STDOUT: !38 = !DISubroutineType(types: !37)
// CHECK:STDOUT: !39 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !34, line: 22, type: !38, spFlags: DISPFlagDefinition, unit: !35)
// CHECK:STDOUT: !42 = !DILocation(line: 23, column: 10, scope: !39)
// CHECK:STDOUT: !43 = !{!36, !36}
// CHECK:STDOUT: !44 = !DISubroutineType(types: !43)
// CHECK:STDOUT: !45 = !DILocation(line: 23, column: 3, scope: !39)
// CHECK:STDOUT: !46 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.4d6ceefeb7f772f4", scope: null, file: !34, line: 17, type: !44, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !54)
// CHECK:STDOUT: !47 = !DILocalVariable(arg: 1, scope: !46, type: !36)
// CHECK:STDOUT: !52 = !DILocation(line: 19, column: 10, scope: !46)
// CHECK:STDOUT: !53 = !DILocation(line: 19, column: 3, scope: !46)
// CHECK:STDOUT: !54 = !{!47}
// CHECK:STDOUT: !55 = distinct !DISubprogram(name: "G", linkageName: "_CG:enclosed.4d6ceefeb7f772f4", scope: null, file: !34, line: 18, type: !44, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !62)
// CHECK:STDOUT: !56 = !DILocalVariable(arg: 1, scope: !55, type: !36)
// CHECK:STDOUT: !61 = !DILocation(line: 18, column: 21, scope: !55)
// CHECK:STDOUT: !62 = !{!56}
// CHECK:STDOUT:
