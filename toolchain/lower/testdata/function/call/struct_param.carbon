// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/call/struct_param.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/call/struct_param.carbon

fn F(a: {}, b: {.a: i32}, c: {.a: i32, .b: i32}) {}

fn Main() {
  F({}, {.a = 1}, {.a = 2, .b = 3});
}

// CHECK:STDOUT: ; ModuleID = 'struct_param.carbon'
// CHECK:STDOUT: source_filename = "struct_param.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @struct.9ae.loc14_34.6 = internal constant { i32, i32 } { i32 2, i32 3 }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CF.Main({ i32 } %b, ptr %c) !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CMain.Main() !dbg !8 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main({ i32 } { i32 1 }, ptr @struct.9ae.loc14_34.6), !dbg !9
// CHECK:STDOUT:   ret void, !dbg !10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "struct_param.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !3, line: 11, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 11, column: 1, scope: !4)
// CHECK:STDOUT: !8 = distinct !DISubprogram(name: "Main", linkageName: "_CMain.Main", scope: null, file: !3, line: 13, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !9 = !DILocation(line: 14, column: 3, scope: !8)
// CHECK:STDOUT: !10 = !DILocation(line: 13, column: 1, scope: !8)
