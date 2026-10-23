// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/array.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/array/array_in_place.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/array/array_in_place.carbon

fn F() -> (i32, i32, i32);

fn G() {
  var _: array((i32, i32, i32), 2) = (F(), F());
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'array_in_place.carbon'
// CHECK:STDOUT: source_filename = "array_in_place.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CF.Main(ptr sret({ i32, i32, i32 }))
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !28 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca [2 x { i32, i32, i32 }], align 4, !dbg !29
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !29
// CHECK:STDOUT:   %.loc16_47.6.array.index = getelementptr inbounds [2 x { i32, i32, i32 }], ptr %_.var, i32 0, i64 0, !dbg !30
// CHECK:STDOUT:   call void @_CF.Main(ptr %.loc16_47.6.array.index), !dbg !31
// CHECK:STDOUT:   %.loc16_47.5.array.index = getelementptr inbounds [2 x { i32, i32, i32 }], ptr %_.var, i32 0, i64 1, !dbg !30
// CHECK:STDOUT:   call void @_CF.Main(ptr %.loc16_47.5.array.index), !dbg !32
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.cb1358a8dfcca74f(ptr %_.var), !dbg !29
// CHECK:STDOUT:   ret void, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.cb1358a8dfcca74f(ptr %t) #0 !dbg !37 {
// CHECK:STDOUT:   ret void, !dbg !42
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!21}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !14 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !20 = !DIFile(filename: "array_in_place.carbon", directory: "")
// CHECK:STDOUT: !21 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !20, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !26 = !{null}
// CHECK:STDOUT: !27 = !DISubroutineType(types: !26)
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !20, line: 15, type: !27, spFlags: DISPFlagDefinition, unit: !21)
// CHECK:STDOUT: !29 = !DILocation(line: 16, column: 3, scope: !28)
// CHECK:STDOUT: !30 = !DILocation(line: 16, column: 38, scope: !28)
// CHECK:STDOUT: !31 = !DILocation(line: 16, column: 39, scope: !28)
// CHECK:STDOUT: !32 = !DILocation(line: 16, column: 44, scope: !28)
// CHECK:STDOUT: !34 = !{null, !23}
// CHECK:STDOUT: !35 = !DISubroutineType(types: !34)
// CHECK:STDOUT: !36 = !DILocation(line: 15, column: 1, scope: !28)
// CHECK:STDOUT: !37 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.cb1358a8dfcca74f", scope: null, file: !14, line: 32, type: !35, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !43)
// CHECK:STDOUT: !38 = !DILocalVariable(arg: 1, scope: !37, type: !23)
// CHECK:STDOUT: !42 = !DILocation(line: 32, column: 1, scope: !37)
// CHECK:STDOUT: !43 = !{!38}
// CHECK:STDOUT:
