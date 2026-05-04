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

private fn F[T:! Core.Copy](a: T, _: T) -> T {
  return a;
}

fn Lib1CallF[T:! Core.Copy](a: T, b: T) -> T {
  return F(a, b);
}

// --- lib2.carbon

library "[[@TEST_NAME]]";

// Duplicate function name in different files. Shouldn't be a name conflict
// because of `private`. However, we currently use the same mangling for both
// functions.
private fn F[T:! Core.Copy](_: T, b: T) -> T {
  return b;
}

fn Lib2CallF[T:! Core.Copy](a: T, b: T) -> T {
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

// CHECK:STDOUT: ; ModuleID = 'lib1.carbon'
// CHECK:STDOUT: source_filename = "lib1.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "lib1.carbon", directory: "")
// CHECK:STDOUT: ; ModuleID = 'lib2.carbon'
// CHECK:STDOUT: source_filename = "lib2.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "lib2.carbon", directory: "")
// CHECK:STDOUT: ; ModuleID = 'todo_use.carbon'
// CHECK:STDOUT: source_filename = "todo_use.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @main() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Lib1CallF.call = call i32 @_CLib1CallF.Main.2382f01eaa053e7b(i32 1, i32 2), !dbg !7
// CHECK:STDOUT:   %Lib2CallF.call = call i32 @_CLib2CallF.Main.2382f01eaa053e7b(i32 1, i32 2), !dbg !8
// CHECK:STDOUT:   ret void, !dbg !9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CLib1CallF.Main.2382f01eaa053e7b(i32 %a, i32 %b) #0 !dbg !10 {
// CHECK:STDOUT:   %1 = call i32 @_CF.Main.2382f01eaa053e7b(i32 %a, i32 %b), !dbg !18
// CHECK:STDOUT:   ret i32 %1, !dbg !19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CLib2CallF.Main.2382f01eaa053e7b(i32 %a, i32 %b) #0 !dbg !20 {
// CHECK:STDOUT:   %1 = call i32 @_CF.Main.2382f01eaa053e7b(i32 %a, i32 %b), !dbg !25
// CHECK:STDOUT:   ret i32 %1, !dbg !26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.2382f01eaa053e7b(i32 %a, i32 %_) #0 !dbg !27 {
// CHECK:STDOUT:   ret i32 %a, !dbg !31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CF.Main.2382f01eaa053e7b, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "todo_use.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !3, line: 9, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DILocation(line: 10, column: 16, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 11, column: 16, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 9, column: 1, scope: !4)
// CHECK:STDOUT: !10 = distinct !DISubprogram(name: "Lib1CallF", linkageName: "_CLib1CallF.Main.2382f01eaa053e7b", scope: null, file: !11, line: 8, type: !12, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !15)
// CHECK:STDOUT: !11 = !DIFile(filename: "lib1.carbon", directory: "")
// CHECK:STDOUT: !12 = !DISubroutineType(types: !13)
// CHECK:STDOUT: !13 = !{!14, !14, !14}
// CHECK:STDOUT: !14 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !15 = !{!16, !17}
// CHECK:STDOUT: !16 = !DILocalVariable(arg: 1, scope: !10, type: !14)
// CHECK:STDOUT: !17 = !DILocalVariable(arg: 2, scope: !10, type: !14)
// CHECK:STDOUT: !18 = !DILocation(line: 9, column: 10, scope: !10)
// CHECK:STDOUT: !19 = !DILocation(line: 9, column: 3, scope: !10)
// CHECK:STDOUT: !20 = distinct !DISubprogram(name: "Lib2CallF", linkageName: "_CLib2CallF.Main.2382f01eaa053e7b", scope: null, file: !21, line: 11, type: !12, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !22)
// CHECK:STDOUT: !21 = !DIFile(filename: "lib2.carbon", directory: "")
// CHECK:STDOUT: !22 = !{!23, !24}
// CHECK:STDOUT: !23 = !DILocalVariable(arg: 1, scope: !20, type: !14)
// CHECK:STDOUT: !24 = !DILocalVariable(arg: 2, scope: !20, type: !14)
// CHECK:STDOUT: !25 = !DILocation(line: 12, column: 10, scope: !20)
// CHECK:STDOUT: !26 = !DILocation(line: 12, column: 3, scope: !20)
// CHECK:STDOUT: !27 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.2382f01eaa053e7b", scope: null, file: !11, line: 4, type: !12, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !28)
// CHECK:STDOUT: !28 = !{!29, !30}
// CHECK:STDOUT: !29 = !DILocalVariable(arg: 1, scope: !27, type: !14)
// CHECK:STDOUT: !30 = !DILocalVariable(arg: 2, scope: !27, type: !14)
// CHECK:STDOUT: !31 = !DILocation(line: 5, column: 3, scope: !27)
