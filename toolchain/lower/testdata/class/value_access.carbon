// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/class/value_access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/class/value_access.carbon

class C {
  var a: (i32, i32, i32);
}

fn F(c: C) -> i32 {
  // TODO: `c.a` is a value expression here, which forces a value binding as
  // part of the member access, creating a tuple value temporary. We could
  // defer performing the value binding to avoid creating this temporary.
  return c.a.1;
}

// CHECK:STDOUT: ; ModuleID = 'value_access.carbon'
// CHECK:STDOUT: source_filename = "value_access.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_CF.Main(ptr %c) !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc19_11.1.a = getelementptr inbounds nuw { { i32, i32, i32 } }, ptr %c, i32 0, i32 0, !dbg !7
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %.loc19_11.1.a, i32 0, i32 0, !dbg !7
// CHECK:STDOUT:   %.loc19_11.2 = load i32, ptr %tuple.elem0.tuple.elem, align 4, !dbg !7
// CHECK:STDOUT:   %tuple.elem1.loc19_11.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %.loc19_11.1.a, i32 0, i32 1, !dbg !7
// CHECK:STDOUT:   %.loc19_11.3 = load i32, ptr %tuple.elem1.loc19_11.tuple.elem, align 4, !dbg !7
// CHECK:STDOUT:   %tuple.elem2.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %.loc19_11.1.a, i32 0, i32 2, !dbg !7
// CHECK:STDOUT:   %.loc19_11.4 = load i32, ptr %tuple.elem2.tuple.elem, align 4, !dbg !7
// CHECK:STDOUT:   %tuple = alloca { i32, i32, i32 }, align 8, !dbg !7
// CHECK:STDOUT:   %tuple1 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %tuple, i32 0, i32 0, !dbg !7
// CHECK:STDOUT:   store i32 %.loc19_11.2, ptr %tuple1, align 4, !dbg !7
// CHECK:STDOUT:   %tuple2 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %tuple, i32 0, i32 1, !dbg !7
// CHECK:STDOUT:   store i32 %.loc19_11.3, ptr %tuple2, align 4, !dbg !7
// CHECK:STDOUT:   %tuple3 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %tuple, i32 0, i32 2, !dbg !7
// CHECK:STDOUT:   store i32 %.loc19_11.4, ptr %tuple3, align 4, !dbg !7
// CHECK:STDOUT:   %tuple.elem1.loc19_13.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %tuple, i32 0, i32 1, !dbg !7
// CHECK:STDOUT:   %tuple.elem1.loc19_13.tuple.elem.load = load i32, ptr %tuple.elem1.loc19_13.tuple.elem, align 4, !dbg !7
// CHECK:STDOUT:   ret i32 %tuple.elem1.loc19_13.tuple.elem.load, !dbg !8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "value_access.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !3, line: 15, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 19, column: 10, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 19, column: 3, scope: !4)
