// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/primitives/type_values.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/primitives/type_values.carbon

// TODO: It would be nice to treat these functions as returning `void` instead
// of `{}`.

fn I32() -> type {
  return i32;
}

fn I48() -> type {
  return i48;
}

fn F64() -> type {
  return f64;
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'type_values.carbon'
// CHECK:STDOUT: source_filename = "type_values.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: %type.26 = type {}
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define %type.26 @_CI32.Main() #0 !dbg !37 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret %type.26 zeroinitializer, !dbg !40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define %type.26 @_CI48.Main() #0 !dbg !41 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret %type.26 zeroinitializer, !dbg !44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define %type.26 @_CF64.Main() #0 !dbg !45 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret %type.26 zeroinitializer, !dbg !48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!33}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !32 = !DIFile(filename: "type_values.carbon", directory: "")
// CHECK:STDOUT: !33 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !32, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !34 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !35 = !{!34}
// CHECK:STDOUT: !36 = !DISubroutineType(types: !35)
// CHECK:STDOUT: !37 = distinct !DISubprogram(name: "I32", linkageName: "_CI32.Main", scope: null, file: !32, line: 16, type: !36, spFlags: DISPFlagDefinition, unit: !33)
// CHECK:STDOUT: !40 = !DILocation(line: 17, column: 3, scope: !37)
// CHECK:STDOUT: !41 = distinct !DISubprogram(name: "I48", linkageName: "_CI48.Main", scope: null, file: !32, line: 20, type: !36, spFlags: DISPFlagDefinition, unit: !33)
// CHECK:STDOUT: !44 = !DILocation(line: 21, column: 3, scope: !41)
// CHECK:STDOUT: !45 = distinct !DISubprogram(name: "F64", linkageName: "_CF64.Main", scope: null, file: !32, line: 24, type: !36, spFlags: DISPFlagDefinition, unit: !33)
// CHECK:STDOUT: !48 = !DILocation(line: 25, column: 3, scope: !45)
// CHECK:STDOUT:
