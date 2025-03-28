// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/tuple/value_forwarding.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/tuple/value_forwarding.carbon

fn G(ab: ((i32, i32, i32), (i32, i32, i32)));

fn F(a: (i32, i32, i32), b: (i32, i32, i32)) {
  G((a, b));
}

// CHECK:STDOUT: ; ModuleID = 'value_forwarding.carbon'
// CHECK:STDOUT: source_filename = "value_forwarding.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CG.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CF.Main(ptr %a, ptr %b) !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %tuple = alloca { ptr, ptr }, align 8, !dbg !7
// CHECK:STDOUT:   %tuple1 = getelementptr inbounds nuw { ptr, ptr }, ptr %tuple, i32 0, i32 0, !dbg !7
// CHECK:STDOUT:   store ptr %a, ptr %tuple1, align 8, !dbg !7
// CHECK:STDOUT:   %tuple2 = getelementptr inbounds nuw { ptr, ptr }, ptr %tuple, i32 0, i32 1, !dbg !7
// CHECK:STDOUT:   store ptr %b, ptr %tuple2, align 8, !dbg !7
// CHECK:STDOUT:   call void @_CG.Main(ptr %tuple), !dbg !8
// CHECK:STDOUT:   ret void, !dbg !9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "value_forwarding.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !3, line: 13, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 14, column: 5, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 14, column: 3, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 13, column: 1, scope: !4)
