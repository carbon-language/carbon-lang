// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/struct/partially_const.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/struct/partially_const.carbon

fn Make(n: i32) -> {.a: i32, .b: i32, .c: i32} {
  return {.a = 0, .b = n, .c = 0};
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'partially_const.carbon'
// CHECK:STDOUT: source_filename = "partially_const.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMake.Main(ptr sret({ i32, i32, i32 }) %return, i32 %n) #0 !dbg !26 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc14_33.3.a = getelementptr inbounds nuw { i32, i32, i32 }, ptr %return, i32 0, i32 0, !dbg !32
// CHECK:STDOUT:   %.loc14_33.5.b = getelementptr inbounds nuw { i32, i32, i32 }, ptr %return, i32 0, i32 1, !dbg !32
// CHECK:STDOUT:   store i32 %n, ptr %.loc14_33.5.b, align 4, !dbg !32
// CHECK:STDOUT:   %.loc14_33.8.c = getelementptr inbounds nuw { i32, i32, i32 }, ptr %return, i32 0, i32 2, !dbg !32
// CHECK:STDOUT:   store i32 0, ptr %.loc14_33.3.a, align 4, !dbg !32
// CHECK:STDOUT:   store i32 0, ptr %.loc14_33.8.c, align 4, !dbg !32
// CHECK:STDOUT:   ret void, !dbg !33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!21}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIFile(filename: "partially_const.carbon", directory: "")
// CHECK:STDOUT: !21 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !20, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !22 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{!23, !22}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "Make", linkageName: "_CMake.Main", scope: null, file: !20, line: 13, type: !25, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !34)
// CHECK:STDOUT: !27 = !DILocalVariable(arg: 1, scope: !26, type: !22)
// CHECK:STDOUT: !32 = !DILocation(line: 14, column: 10, scope: !26)
// CHECK:STDOUT: !33 = !DILocation(line: 14, column: 3, scope: !26)
// CHECK:STDOUT: !34 = !{!27}
// CHECK:STDOUT:
