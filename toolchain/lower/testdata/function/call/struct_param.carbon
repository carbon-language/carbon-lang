// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/call/struct_param.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/call/struct_param.carbon

fn F(_: {}, _: {.a: i32}, _: {.a: i32, .b: i32}) {}

fn Main() {
  F({}, {.a = 1}, {.a = 2, .b = 3});
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'struct_param.carbon'
// CHECK:STDOUT: source_filename = "struct_param.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @struct.c6c.loc16_34.6 = internal constant { i32, i32 } { i32 2, i32 3 }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main({ i32 } %_, ptr %_1) #0 !dbg !28 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMain.Main() #0 !dbg !36 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main({ i32 } { i32 1 }, ptr @struct.c6c.loc16_34.6), !dbg !37
// CHECK:STDOUT:   ret void, !dbg !38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!21}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIFile(filename: "struct_param.carbon", directory: "")
// CHECK:STDOUT: !21 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !20, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{null, !23, !23}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = !{null}
// CHECK:STDOUT: !27 = !DISubroutineType(types: !26)
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !20, line: 13, type: !25, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !35)
// CHECK:STDOUT: !29 = !DILocalVariable(arg: 1, scope: !28, type: !23)
// CHECK:STDOUT: !30 = !DILocalVariable(arg: 2, scope: !28, type: !23)
// CHECK:STDOUT: !34 = !DILocation(line: 13, column: 1, scope: !28)
// CHECK:STDOUT: !35 = !{!29, !30}
// CHECK:STDOUT: !36 = distinct !DISubprogram(name: "Main", linkageName: "_CMain.Main", scope: null, file: !20, line: 15, type: !27, spFlags: DISPFlagDefinition, unit: !21)
// CHECK:STDOUT: !37 = !DILocation(line: 16, column: 3, scope: !36)
// CHECK:STDOUT: !38 = !DILocation(line: 15, column: 1, scope: !36)
// CHECK:STDOUT:
