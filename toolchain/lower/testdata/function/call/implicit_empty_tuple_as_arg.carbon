// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/call/implicit_empty_tuple_as_arg.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/call/implicit_empty_tuple_as_arg.carbon


fn Foo() {}

fn Bar(a: ()) -> () { return a; }

fn Main() {
  var _: () = Bar(Foo());
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'implicit_empty_tuple_as_arg.carbon'
// CHECK:STDOUT: source_filename = "implicit_empty_tuple_as_arg.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CFoo.Main() #0 !dbg !20 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CBar.Main() #0 !dbg !22 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMain.Main() #0 !dbg !28 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca {}, align 1, !dbg !29
// CHECK:STDOUT:   %.loc19_23.1.temp = alloca {}, align 1, !dbg !30
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !29
// CHECK:STDOUT:   call void @_CFoo.Main(), !dbg !30
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc19_23.1.temp), !dbg !30
// CHECK:STDOUT:   call void @_CBar.Main(), !dbg !31
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.6d185494fc1773c8(ptr %.loc19_23.1.temp), !dbg !30
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.6d185494fc1773c8(ptr %_.var), !dbg !29
// CHECK:STDOUT:   ret void, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.6d185494fc1773c8(ptr %t) #0 !dbg !37 {
// CHECK:STDOUT:   ret void, !dbg !42
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.6d185494fc1773c8, { 1, 0 }
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
// CHECK:STDOUT: !16 = !DIFile(filename: "implicit_empty_tuple_as_arg.carbon", directory: "")
// CHECK:STDOUT: !17 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !16, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !18 = !{null}
// CHECK:STDOUT: !19 = !DISubroutineType(types: !18)
// CHECK:STDOUT: !20 = distinct !DISubprogram(name: "Foo", linkageName: "_CFoo.Main", scope: null, file: !16, line: 14, type: !19, spFlags: DISPFlagDefinition, unit: !17)
// CHECK:STDOUT: !21 = !DILocation(line: 14, column: 1, scope: !20)
// CHECK:STDOUT: !22 = distinct !DISubprogram(name: "Bar", linkageName: "_CBar.Main", scope: null, file: !16, line: 16, type: !19, spFlags: DISPFlagDefinition, unit: !17)
// CHECK:STDOUT: !27 = !DILocation(line: 16, column: 23, scope: !22)
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "Main", linkageName: "_CMain.Main", scope: null, file: !16, line: 18, type: !19, spFlags: DISPFlagDefinition, unit: !17)
// CHECK:STDOUT: !29 = !DILocation(line: 19, column: 3, scope: !28)
// CHECK:STDOUT: !30 = !DILocation(line: 19, column: 19, scope: !28)
// CHECK:STDOUT: !31 = !DILocation(line: 19, column: 15, scope: !28)
// CHECK:STDOUT: !33 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !34 = !{null, !33}
// CHECK:STDOUT: !35 = !DISubroutineType(types: !34)
// CHECK:STDOUT: !36 = !DILocation(line: 18, column: 1, scope: !28)
// CHECK:STDOUT: !37 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.6d185494fc1773c8", scope: null, file: !6, line: 30, type: !35, spFlags: DISPFlagDefinition, unit: !17, retainedNodes: !43)
// CHECK:STDOUT: !38 = !DILocalVariable(arg: 1, scope: !37, type: !33)
// CHECK:STDOUT: !42 = !DILocation(line: 30, column: 1, scope: !37)
// CHECK:STDOUT: !43 = !{!38}
// CHECK:STDOUT:
