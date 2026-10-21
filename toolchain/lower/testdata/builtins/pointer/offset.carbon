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
// CHECK:STDOUT: define ptr @_CTest.Main(ptr %p, i64 %offset) #0 !dbg !39 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret ptr poison, !dbg !49
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!35}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !19 = !DIBasicType(name: "int", size: 64, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !34 = !DIFile(filename: "offset.carbon", directory: "")
// CHECK:STDOUT: !35 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !34, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !36 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !37 = !{!36, !36, !19}
// CHECK:STDOUT: !38 = !DISubroutineType(types: !37)
// CHECK:STDOUT: !39 = distinct !DISubprogram(name: "Test", linkageName: "_CTest.Main", scope: null, file: !34, line: 16, type: !38, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !50)
// CHECK:STDOUT: !40 = !DILocalVariable(arg: 1, scope: !39, type: !36)
// CHECK:STDOUT: !41 = !DILocalVariable(arg: 2, scope: !39, type: !19)
// CHECK:STDOUT: !49 = !DILocation(line: 17, column: 3, scope: !39)
// CHECK:STDOUT: !50 = !{!40, !41}
// CHECK:STDOUT:
