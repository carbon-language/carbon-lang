// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/tuple/one_entry.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/tuple/one_entry.carbon

fn Run() -> i32 {
  var x: (i32, ) = (1, );
  var y: (i32, ) = x;
  return 0;
}

// CHECK:STDOUT: ; ModuleID = 'one_entry.carbon'
// CHECK:STDOUT: source_filename = "one_entry.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @main() !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x.var = alloca { i32 }, align 8, !dbg !7
// CHECK:STDOUT:   store { i32 } { i32 1 }, ptr %x.var, align 4, !dbg !8
// CHECK:STDOUT:   %y.var = alloca { i32 }, align 8, !dbg !9
// CHECK:STDOUT:   %.loc13_20.1.tuple.elem = getelementptr inbounds nuw { i32 }, ptr %x.var, i32 0, i32 0, !dbg !10
// CHECK:STDOUT:   %.loc13_20.2 = load i32, ptr %.loc13_20.1.tuple.elem, align 4, !dbg !10
// CHECK:STDOUT:   %.loc13_20.3.tuple.init = insertvalue { i32 } poison, i32 %.loc13_20.2, 0, !dbg !10
// CHECK:STDOUT:   store { i32 } %.loc13_20.3.tuple.init, ptr %y.var, align 4, !dbg !11
// CHECK:STDOUT:   ret i32 0, !dbg !12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder i32 1, { 0, 2, 1 }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "one_entry.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !3, line: 11, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 12, column: 7, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 12, column: 3, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 13, column: 7, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 13, column: 20, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 13, column: 3, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 14, column: 3, scope: !4)
