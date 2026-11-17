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
// CHECK:STDOUT: define void @_CF.Main(ptr %a, ptr %b) #0 !dbg !28 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %tuple = alloca { ptr, ptr }, align 8, !dbg !35
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %tuple), !dbg !35
// CHECK:STDOUT:   %tuple1 = getelementptr inbounds nuw { ptr, ptr }, ptr %tuple, i32 0, i32 0, !dbg !35
// CHECK:STDOUT:   store ptr %a, ptr %tuple1, align 8, !dbg !35
// CHECK:STDOUT:   %tuple2 = getelementptr inbounds nuw { ptr, ptr }, ptr %tuple, i32 0, i32 1, !dbg !35
// CHECK:STDOUT:   store ptr %b, ptr %tuple2, align 8, !dbg !35
// CHECK:STDOUT:   call void @_CG.Main(ptr %tuple), !dbg !36
// CHECK:STDOUT:   ret void, !dbg !37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!21}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIFile(filename: "value_forwarding.carbon", directory: "")
// CHECK:STDOUT: !21 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !20, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !26 = !{null, !23, !23}
// CHECK:STDOUT: !27 = !DISubroutineType(types: !26)
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !20, line: 15, type: !27, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !38)
// CHECK:STDOUT: !29 = !DILocalVariable(arg: 1, scope: !28, type: !23)
// CHECK:STDOUT: !30 = !DILocalVariable(arg: 2, scope: !28, type: !23)
// CHECK:STDOUT: !35 = !DILocation(line: 16, column: 5, scope: !28)
// CHECK:STDOUT: !36 = !DILocation(line: 16, column: 3, scope: !28)
// CHECK:STDOUT: !37 = !DILocation(line: 15, column: 1, scope: !28)
// CHECK:STDOUT: !38 = !{!29, !30}
// CHECK:STDOUT:
