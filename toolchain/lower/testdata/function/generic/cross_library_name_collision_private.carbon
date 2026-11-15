// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/generic/cross_library_name_collision_private.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/generic/cross_library_name_collision_private.carbon

// --- lib1.carbon

library "[[@TEST_NAME]]";

private fn F[T: Core.Copy](a: T, _: T) -> T {
  return a;
}

fn Lib1CallF[T: Core.Copy](a: T, b: T) -> T {
  return F(a, b);
}

// --- lib2.carbon

library "[[@TEST_NAME]]";

// Duplicate function name in different files. Shouldn't be a name conflict
// because of `private`. However, we currently use the same mangling for both
// functions.
private fn F[T: Core.Copy](_: T, b: T) -> T {
  return b;
}

fn Lib2CallF[T: Core.Copy](a: T, b: T) -> T {
  return F(a, b);
}

// --- todo_use.carbon

// TODO: This should call the `F` from lib1 and the `F` from lib2. Currently
// it calls the same function in both (hashes match for both instances of
// `call i32 @_CF.Main.<hash>`).

import library "lib1";
import library "lib2";

fn Run() {
  let _: i32 = Lib1CallF(1 as i32, 2 as i32);
  let _: i32 = Lib2CallF(1 as i32, 2 as i32);
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'lib1.carbon'
// CHECK:STDOUT: source_filename = "lib1.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!21}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIFile(filename: "lib1.carbon", directory: "")
// CHECK:STDOUT: !21 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !20, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'lib2.carbon'
// CHECK:STDOUT: source_filename = "lib2.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!23}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !22 = !DIFile(filename: "lib2.carbon", directory: "")
// CHECK:STDOUT: !23 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !22, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'todo_use.carbon'
// CHECK:STDOUT: source_filename = "todo_use.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @main() #0 !dbg !29 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Lib1CallF.call = call i32 @_CLib1CallF.Main.2fad9574e050607e(i32 1, i32 2), !dbg !30
// CHECK:STDOUT:   %Lib2CallF.call = call i32 @_CLib2CallF.Main.2fad9574e050607e(i32 1, i32 2), !dbg !35
// CHECK:STDOUT:   ret i32 0, !dbg !38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CLib1CallF.Main.2fad9574e050607e(i32 %a, i32 %b) #0 !dbg !39 {
// CHECK:STDOUT:   %1 = call i32 @_CF.Main.2fad9574e050607e(i32 %a, i32 %b), !dbg !48
// CHECK:STDOUT:   ret i32 %1, !dbg !49
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CLib2CallF.Main.2fad9574e050607e(i32 %a, i32 %b) #0 !dbg !51 {
// CHECK:STDOUT:   %1 = call i32 @_CF.Main.2fad9574e050607e(i32 %a, i32 %b), !dbg !60
// CHECK:STDOUT:   ret i32 %1, !dbg !61
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.2fad9574e050607e(i32 %a, i32 %_) #0 !dbg !63 {
// CHECK:STDOUT:   ret i32 %a, !dbg !71
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CF.Main.2fad9574e050607e, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!25}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIFile(filename: "lib1.carbon", directory: "")
// CHECK:STDOUT: !22 = !DIFile(filename: "lib2.carbon", directory: "")
// CHECK:STDOUT: !24 = !DIFile(filename: "todo_use.carbon", directory: "")
// CHECK:STDOUT: !25 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !24, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !26 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !27 = !{!26}
// CHECK:STDOUT: !28 = !DISubroutineType(types: !27)
// CHECK:STDOUT: !29 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !24, line: 9, type: !28, spFlags: DISPFlagDefinition, unit: !25)
// CHECK:STDOUT: !30 = !DILocation(line: 10, column: 16, scope: !29)
// CHECK:STDOUT: !31 = !{!26, !26, !26}
// CHECK:STDOUT: !32 = !DISubroutineType(types: !31)
// CHECK:STDOUT: !35 = !DILocation(line: 11, column: 16, scope: !29)
// CHECK:STDOUT: !38 = !DILocation(line: 9, column: 1, scope: !29)
// CHECK:STDOUT: !39 = distinct !DISubprogram(name: "Lib1CallF", linkageName: "_CLib1CallF.Main.2fad9574e050607e", scope: null, file: !20, line: 8, type: !32, spFlags: DISPFlagDefinition, unit: !25, retainedNodes: !50)
// CHECK:STDOUT: !40 = !DILocalVariable(arg: 1, scope: !39, type: !26)
// CHECK:STDOUT: !41 = !DILocalVariable(arg: 2, scope: !39, type: !26)
// CHECK:STDOUT: !48 = !DILocation(line: 9, column: 10, scope: !39)
// CHECK:STDOUT: !49 = !DILocation(line: 9, column: 3, scope: !39)
// CHECK:STDOUT: !50 = !{!40, !41}
// CHECK:STDOUT: !51 = distinct !DISubprogram(name: "Lib2CallF", linkageName: "_CLib2CallF.Main.2fad9574e050607e", scope: null, file: !22, line: 11, type: !32, spFlags: DISPFlagDefinition, unit: !25, retainedNodes: !62)
// CHECK:STDOUT: !52 = !DILocalVariable(arg: 1, scope: !51, type: !26)
// CHECK:STDOUT: !53 = !DILocalVariable(arg: 2, scope: !51, type: !26)
// CHECK:STDOUT: !60 = !DILocation(line: 12, column: 10, scope: !51)
// CHECK:STDOUT: !61 = !DILocation(line: 12, column: 3, scope: !51)
// CHECK:STDOUT: !62 = !{!52, !53}
// CHECK:STDOUT: !63 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.2fad9574e050607e", scope: null, file: !20, line: 4, type: !32, spFlags: DISPFlagDefinition, unit: !25, retainedNodes: !72)
// CHECK:STDOUT: !64 = !DILocalVariable(arg: 1, scope: !63, type: !26)
// CHECK:STDOUT: !65 = !DILocalVariable(arg: 2, scope: !63, type: !26)
// CHECK:STDOUT: !71 = !DILocation(line: 5, column: 3, scope: !63)
// CHECK:STDOUT: !72 = !{!64, !65}
// CHECK:STDOUT:
