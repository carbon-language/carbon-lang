// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
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

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'value_forwarding.carbon'
// CHECK:STDOUT: source_filename = "value_forwarding.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CG.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main(ptr %a, ptr %b) #0 !dbg !26 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %tuple = alloca { ptr, ptr }, align 8, !dbg !33
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %tuple), !dbg !33
// CHECK:STDOUT:   %tuple1 = getelementptr inbounds nuw { ptr, ptr }, ptr %tuple, i32 0, i32 0, !dbg !33
// CHECK:STDOUT:   store ptr %a, ptr %tuple1, align 8, !dbg !33
// CHECK:STDOUT:   %tuple2 = getelementptr inbounds nuw { ptr, ptr }, ptr %tuple, i32 0, i32 1, !dbg !33
// CHECK:STDOUT:   store ptr %b, ptr %tuple2, align 8, !dbg !33
// CHECK:STDOUT:   call void @_CG.Main(ptr %tuple), !dbg !34
// CHECK:STDOUT:   ret void, !dbg !35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !18 = !DIFile(filename: "value_forwarding.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{null, !21, !21}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !18, line: 15, type: !25, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !36)
// CHECK:STDOUT: !27 = !DILocalVariable(arg: 1, scope: !26, type: !21)
// CHECK:STDOUT: !28 = !DILocalVariable(arg: 2, scope: !26, type: !21)
// CHECK:STDOUT: !33 = !DILocation(line: 16, column: 5, scope: !26)
// CHECK:STDOUT: !34 = !DILocation(line: 16, column: 3, scope: !26)
// CHECK:STDOUT: !35 = !DILocation(line: 15, column: 1, scope: !26)
// CHECK:STDOUT: !36 = !{!27, !28}
// CHECK:STDOUT:
