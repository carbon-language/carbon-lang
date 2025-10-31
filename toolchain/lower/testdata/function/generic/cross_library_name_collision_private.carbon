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

private fn F[T:! Core.Copy](a: T, b: T) -> T {
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
private fn F[T:! Core.Copy](a: T, b: T) -> T {
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
  let a: i32 = Lib1CallF(1 as i32, 2 as i32);
  let b: i32 = Lib2CallF(1 as i32, 2 as i32);
}

// CHECK:STDOUT: ; ModuleID = 'lib1.carbon'
// CHECK:STDOUT: source_filename = "lib1.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "lib1.carbon", directory: "")
// CHECK:STDOUT: ; ModuleID = 'lib2.carbon'
// CHECK:STDOUT: source_filename = "lib2.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "lib2.carbon", directory: "")
// CHECK:STDOUT: ; ModuleID = 'todo_use.carbon'
// CHECK:STDOUT: source_filename = "todo_use.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @main() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Lib1CallF.call = call i32 @_CLib1CallF.Main.99cfc57c95d4cf0a(i32 1, i32 2), !dbg !7
// CHECK:STDOUT:   %Lib2CallF.call = call i32 @_CLib2CallF.Main.99cfc57c95d4cf0a(i32 1, i32 2), !dbg !8
// CHECK:STDOUT:   ret void, !dbg !9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CLib1CallF.Main.99cfc57c95d4cf0a(i32 %a, i32 %b) #0 !dbg !10 {
// CHECK:STDOUT:   %1 = call i32 @_CF.Main.99cfc57c95d4cf0a(i32 %a, i32 %b), !dbg !12
// CHECK:STDOUT:   ret i32 %1, !dbg !13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CLib2CallF.Main.99cfc57c95d4cf0a(i32 %a, i32 %b) #0 !dbg !14 {
// CHECK:STDOUT:   %1 = call i32 @_CF.Main.99cfc57c95d4cf0a(i32 %a, i32 %b), !dbg !16
// CHECK:STDOUT:   ret i32 %1, !dbg !17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.99cfc57c95d4cf0a(i32 %a, i32 %b) #0 !dbg !18 {
// CHECK:STDOUT:   ret i32 %a, !dbg !19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CF.Main.99cfc57c95d4cf0a, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "todo_use.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !3, line: 9, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 10, column: 16, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 11, column: 16, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 9, column: 1, scope: !4)
// CHECK:STDOUT: !10 = distinct !DISubprogram(name: "Lib1CallF", linkageName: "_CLib1CallF.Main.99cfc57c95d4cf0a", scope: null, file: !11, line: 8, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !11 = !DIFile(filename: "lib1.carbon", directory: "")
// CHECK:STDOUT: !12 = !DILocation(line: 9, column: 10, scope: !10)
// CHECK:STDOUT: !13 = !DILocation(line: 9, column: 3, scope: !10)
// CHECK:STDOUT: !14 = distinct !DISubprogram(name: "Lib2CallF", linkageName: "_CLib2CallF.Main.99cfc57c95d4cf0a", scope: null, file: !15, line: 11, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !15 = !DIFile(filename: "lib2.carbon", directory: "")
// CHECK:STDOUT: !16 = !DILocation(line: 12, column: 10, scope: !14)
// CHECK:STDOUT: !17 = !DILocation(line: 12, column: 3, scope: !14)
// CHECK:STDOUT: !18 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.99cfc57c95d4cf0a", scope: null, file: !11, line: 4, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !19 = !DILocation(line: 5, column: 3, scope: !18)
