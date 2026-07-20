// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
// EXTRA-ARGS: --target=x86_64-linux-gnu
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/builtins/pointer/offset.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/builtins/pointer/offset.carbon

fn PointerOffset[T: type](p: T*, offset: u64) -> T* = "pointer.offset";

fn Test(p: i32*, offset: u64) -> i32* {
  return PointerOffset(p, offset);
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'offset.carbon'
// CHECK:STDOUT: source_filename = "offset.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CTest.Main(ptr %p, i64 %offset) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret ptr poison, !dbg !12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "offset.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Test", linkageName: "_CTest.Main", scope: null, file: !1, line: 16, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !9)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7, !7, !8}
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !DIBasicType(name: "int", size: 64, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !9 = !{!10, !11}
// CHECK:STDOUT: !10 = !DILocalVariable(arg: 1, scope: !4, type: !7)
// CHECK:STDOUT: !11 = !DILocalVariable(arg: 2, scope: !4, type: !8)
// CHECK:STDOUT: !12 = !DILocation(line: 17, column: 3, scope: !4)
// CHECK:STDOUT:
