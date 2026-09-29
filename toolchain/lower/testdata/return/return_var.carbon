// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
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

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'return_var.carbon'
// CHECK:STDOUT: source_filename = "return_var.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMake.Main(ptr sret({ i32, ptr }) %return) #0 !dbg !24 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc20_30.3.data = getelementptr inbounds nuw { i32, ptr }, ptr %return, i32 0, i32 0, !dbg !33
// CHECK:STDOUT:   %.loc20_30.5.next = getelementptr inbounds nuw { i32, ptr }, ptr %return, i32 0, i32 1, !dbg !33
// CHECK:STDOUT:   store ptr %return, ptr %.loc20_30.5.next, align 8, !dbg !33
// CHECK:STDOUT:   store i32 42, ptr %.loc20_30.3.data, align 4, !dbg !33
// CHECK:STDOUT:   ret void, !dbg !34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !18 = !DIFile(filename: "return_var.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{!21}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !24 = distinct !DISubprogram(name: "Make", linkageName: "_CMake.Main", scope: null, file: !18, line: 18, type: !23, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !33 = !DILocation(line: 20, column: 7, scope: !24)
// CHECK:STDOUT: !34 = !DILocation(line: 21, column: 3, scope: !24)
// CHECK:STDOUT:
