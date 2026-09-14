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
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !18 = !DIFile(filename: "lib1.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'lib2.carbon'
// CHECK:STDOUT: source_filename = "lib2.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!21}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIFile(filename: "lib2.carbon", directory: "")
// CHECK:STDOUT: !21 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !20, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'todo_use.carbon'
// CHECK:STDOUT: source_filename = "todo_use.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @main() #0 !dbg !27 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Lib1CallF.call = call i32 @_CLib1CallF.Main.4d6ceefeb7f772f4(i32 1, i32 2), !dbg !28
// CHECK:STDOUT:   %Lib2CallF.call = call i32 @_CLib2CallF.Main.4d6ceefeb7f772f4(i32 1, i32 2), !dbg !33
// CHECK:STDOUT:   ret i32 0, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CLib1CallF.Main.4d6ceefeb7f772f4(i32 %a, i32 %b) #0 !dbg !37 {
// CHECK:STDOUT:   %1 = call i32 @_CF.Main.4d6ceefeb7f772f4(i32 %a, i32 %b), !dbg !46
// CHECK:STDOUT:   ret i32 %1, !dbg !47
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CLib2CallF.Main.4d6ceefeb7f772f4(i32 %a, i32 %b) #0 !dbg !49 {
// CHECK:STDOUT:   %1 = call i32 @_CF.Main.4d6ceefeb7f772f4(i32 %a, i32 %b), !dbg !58
// CHECK:STDOUT:   ret i32 %1, !dbg !59
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.4d6ceefeb7f772f4(i32 %a, i32 %_) #0 !dbg !61 {
// CHECK:STDOUT:   ret i32 %a, !dbg !69
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CF.Main.4d6ceefeb7f772f4, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!23}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !18 = !DIFile(filename: "lib1.carbon", directory: "")
// CHECK:STDOUT: !20 = !DIFile(filename: "lib2.carbon", directory: "")
// CHECK:STDOUT: !22 = !DIFile(filename: "todo_use.carbon", directory: "")
// CHECK:STDOUT: !23 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !22, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !24 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !25 = !{!24}
// CHECK:STDOUT: !26 = !DISubroutineType(types: !25)
// CHECK:STDOUT: !27 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !22, line: 9, type: !26, spFlags: DISPFlagDefinition, unit: !23)
// CHECK:STDOUT: !28 = !DILocation(line: 10, column: 16, scope: !27)
// CHECK:STDOUT: !29 = !{!24, !24, !24}
// CHECK:STDOUT: !30 = !DISubroutineType(types: !29)
// CHECK:STDOUT: !33 = !DILocation(line: 11, column: 16, scope: !27)
// CHECK:STDOUT: !36 = !DILocation(line: 9, column: 1, scope: !27)
// CHECK:STDOUT: !37 = distinct !DISubprogram(name: "Lib1CallF", linkageName: "_CLib1CallF.Main.4d6ceefeb7f772f4", scope: null, file: !18, line: 8, type: !30, spFlags: DISPFlagDefinition, unit: !23, retainedNodes: !48)
// CHECK:STDOUT: !38 = !DILocalVariable(arg: 1, scope: !37, type: !24)
// CHECK:STDOUT: !39 = !DILocalVariable(arg: 2, scope: !37, type: !24)
// CHECK:STDOUT: !46 = !DILocation(line: 9, column: 10, scope: !37)
// CHECK:STDOUT: !47 = !DILocation(line: 9, column: 3, scope: !37)
// CHECK:STDOUT: !48 = !{!38, !39}
// CHECK:STDOUT: !49 = distinct !DISubprogram(name: "Lib2CallF", linkageName: "_CLib2CallF.Main.4d6ceefeb7f772f4", scope: null, file: !20, line: 11, type: !30, spFlags: DISPFlagDefinition, unit: !23, retainedNodes: !60)
// CHECK:STDOUT: !50 = !DILocalVariable(arg: 1, scope: !49, type: !24)
// CHECK:STDOUT: !51 = !DILocalVariable(arg: 2, scope: !49, type: !24)
// CHECK:STDOUT: !58 = !DILocation(line: 12, column: 10, scope: !49)
// CHECK:STDOUT: !59 = !DILocation(line: 12, column: 3, scope: !49)
// CHECK:STDOUT: !60 = !{!50, !51}
// CHECK:STDOUT: !61 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.4d6ceefeb7f772f4", scope: null, file: !18, line: 4, type: !30, spFlags: DISPFlagDefinition, unit: !23, retainedNodes: !70)
// CHECK:STDOUT: !62 = !DILocalVariable(arg: 1, scope: !61, type: !24)
// CHECK:STDOUT: !63 = !DILocalVariable(arg: 2, scope: !61, type: !24)
// CHECK:STDOUT: !69 = !DILocation(line: 5, column: 3, scope: !61)
// CHECK:STDOUT: !70 = !{!62, !63}
// CHECK:STDOUT:
