// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/return/return_var.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/return/return_var.carbon

class C {
  var data: i32;
  var next: C*;
}

fn Make() -> C {
  returned var c: C;
  c = {.data = 42, .next = &c};
  return var;
}

// CHECK:STDOUT: ; ModuleID = 'return_var.carbon'
// CHECK:STDOUT: source_filename = "return_var.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CMake.Main(ptr sret({ i32, ptr }) %return) !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc18_30.3.data = getelementptr inbounds nuw { i32, ptr }, ptr %return, i32 0, i32 0, !dbg !7
// CHECK:STDOUT:   %.loc18_30.5.next = getelementptr inbounds nuw { i32, ptr }, ptr %return, i32 0, i32 1, !dbg !7
// CHECK:STDOUT:   store ptr %return, ptr %.loc18_30.5.next, align 8, !dbg !7
// CHECK:STDOUT:   store i32 42, ptr %.loc18_30.3.data, align 4, !dbg !7
// CHECK:STDOUT:   ret void, !dbg !8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "return_var.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Make", linkageName: "_CMake.Main", scope: null, file: !3, line: 16, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 18, column: 7, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 19, column: 3, scope: !4)
