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
// CHECK:STDOUT: define void @_CMakeImplicitEmptyTuple.Main() #0 !dbg !20 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMain.Main() #0 !dbg !22 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca {}, align 1, !dbg !23
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !23
// CHECK:STDOUT:   call void @_CMakeImplicitEmptyTuple.Main(), !dbg !24
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.9045abb8230cbcdf(ptr %_.var), !dbg !23
// CHECK:STDOUT:   ret void, !dbg !29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.9045abb8230cbcdf(ptr %t) #0 !dbg !30 {
// CHECK:STDOUT:   ret void, !dbg !35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!17}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !16 = !DIFile(filename: "return_implicit.carbon", directory: "")
// CHECK:STDOUT: !17 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !16, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !18 = !{null}
// CHECK:STDOUT: !19 = !DISubroutineType(types: !18)
// CHECK:STDOUT: !20 = distinct !DISubprogram(name: "MakeImplicitEmptyTuple", linkageName: "_CMakeImplicitEmptyTuple.Main", scope: null, file: !16, line: 14, type: !19, spFlags: DISPFlagDefinition, unit: !17)
// CHECK:STDOUT: !21 = !DILocation(line: 14, column: 1, scope: !20)
// CHECK:STDOUT: !22 = distinct !DISubprogram(name: "Main", linkageName: "_CMain.Main", scope: null, file: !16, line: 17, type: !19, spFlags: DISPFlagDefinition, unit: !17)
// CHECK:STDOUT: !23 = !DILocation(line: 18, column: 3, scope: !22)
// CHECK:STDOUT: !24 = !DILocation(line: 18, column: 15, scope: !22)
// CHECK:STDOUT: !26 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !27 = !{null, !26}
// CHECK:STDOUT: !28 = !DISubroutineType(types: !27)
// CHECK:STDOUT: !29 = !DILocation(line: 17, column: 1, scope: !22)
// CHECK:STDOUT: !30 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.9045abb8230cbcdf", scope: null, file: !6, line: 32, type: !28, spFlags: DISPFlagDefinition, unit: !17, retainedNodes: !36)
// CHECK:STDOUT: !31 = !DILocalVariable(arg: 1, scope: !30, type: !26)
// CHECK:STDOUT: !35 = !DILocation(line: 32, column: 1, scope: !30)
// CHECK:STDOUT: !36 = !{!31}
// CHECK:STDOUT:
