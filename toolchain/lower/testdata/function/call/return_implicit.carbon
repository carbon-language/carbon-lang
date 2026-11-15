// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/call/return_implicit.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/call/return_implicit.carbon


fn MakeImplicitEmptyTuple() {
}

fn Main() {
  var _: () = MakeImplicitEmptyTuple();
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'return_implicit.carbon'
// CHECK:STDOUT: source_filename = "return_implicit.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMakeImplicitEmptyTuple.Main() #0 !dbg !18 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMain.Main() #0 !dbg !20 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca {}, align 1, !dbg !21
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !21
// CHECK:STDOUT:   call void @_CMakeImplicitEmptyTuple.Main(), !dbg !22
// CHECK:STDOUT:   ret void, !dbg !24
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!15}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !14 = !DIFile(filename: "return_implicit.carbon", directory: "")
// CHECK:STDOUT: !15 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !14, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !16 = !{null}
// CHECK:STDOUT: !17 = !DISubroutineType(types: !16)
// CHECK:STDOUT: !18 = distinct !DISubprogram(name: "MakeImplicitEmptyTuple", linkageName: "_CMakeImplicitEmptyTuple.Main", scope: null, file: !14, line: 14, type: !17, spFlags: DISPFlagDefinition, unit: !15)
// CHECK:STDOUT: !19 = !DILocation(line: 14, column: 1, scope: !18)
// CHECK:STDOUT: !20 = distinct !DISubprogram(name: "Main", linkageName: "_CMain.Main", scope: null, file: !14, line: 17, type: !17, spFlags: DISPFlagDefinition, unit: !15)
// CHECK:STDOUT: !21 = !DILocation(line: 18, column: 3, scope: !20)
// CHECK:STDOUT: !22 = !DILocation(line: 18, column: 15, scope: !20)
// CHECK:STDOUT: !24 = !DILocation(line: 17, column: 1, scope: !20)
// CHECK:STDOUT:
