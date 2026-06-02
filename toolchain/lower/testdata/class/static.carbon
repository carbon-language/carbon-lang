// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/class/static.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/class/static.carbon

// --- static.carbon
library "[[@TEST_NAME]]";

class C {
  static var static_field: i32;
  static var static_field_with_init: i32 = 123;
}

fn F() {
  C.static_field = 456;
  C.static_field_with_init = 789;
}

// CHECK:STDOUT: ; ModuleID = 'static.carbon'
// CHECK:STDOUT: source_filename = "static.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @_Cstatic_field.C.Main = global i32 0
// CHECK:STDOUT: @_Cstatic_field_with_init.C.Main = global i32 0
// CHECK:STDOUT: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 0, ptr @_C__global_init.Main, ptr null }]
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   store i32 456, ptr @_Cstatic_field.C.Main, align 4, !dbg !7
// CHECK:STDOUT:   store i32 789, ptr @_Cstatic_field_with_init.C.Main, align 4, !dbg !8
// CHECK:STDOUT:   ret void, !dbg !9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define internal void @_C__global_init.Main() #0 !dbg !10 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   store i32 poison, ptr @_Cstatic_field.C.Main, align 4, !dbg !11
// CHECK:STDOUT:   store i32 123, ptr @_Cstatic_field_with_init.C.Main, align 4, !dbg !12
// CHECK:STDOUT:   ret void, !dbg !13
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
// CHECK:STDOUT: !3 = !DIFile(filename: "static.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !3, line: 8, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DILocation(line: 9, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 10, column: 3, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 8, column: 1, scope: !4)
// CHECK:STDOUT: !10 = distinct !DISubprogram(name: "__global_init", linkageName: "_C__global_init.Main", scope: null, file: !3, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !11 = !DILocation(line: 4, column: 10, scope: !10)
// CHECK:STDOUT: !12 = !DILocation(line: 5, column: 10, scope: !10)
// CHECK:STDOUT: !13 = !DILocation(line: 0, scope: !10)
