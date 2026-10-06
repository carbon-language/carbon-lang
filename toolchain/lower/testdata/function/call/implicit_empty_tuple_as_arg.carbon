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
// CHECK:STDOUT: define void @_CFoo.Main() #0 !dbg !21 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CBar.Main() #0 !dbg !23 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMain.Main() #0 !dbg !29 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca {}, align 1, !dbg !30
// CHECK:STDOUT:   %.loc19_23.1.temp = alloca {}, align 1, !dbg !31
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !30
// CHECK:STDOUT:   call void @_CFoo.Main(), !dbg !31
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc19_23.1.temp), !dbg !31
// CHECK:STDOUT:   call void @_CBar.Main(), !dbg !32
// CHECK:STDOUT:   call void @"_CSelfDestruct.61ea2aba74ab3bf1:core.Destroy.Core"(ptr %.loc19_23.1.temp), !dbg !31
// CHECK:STDOUT:   call void @"_CSelfDestruct.61ea2aba74ab3bf1:core.Destroy.Core"(ptr %_.var), !dbg !30
// CHECK:STDOUT:   ret void, !dbg !34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.61ea2aba74ab3bf1:core.Destroy.Core"(ptr %self) #0 !dbg !35 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @"_CSelfDestruct.61ea2aba74ab3bf1:core.Destroy.Core", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!15}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !14 = !DIFile(filename: "implicit_empty_tuple_as_arg.carbon", directory: "")
// CHECK:STDOUT: !15 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !14, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !16 = !{null}
// CHECK:STDOUT: !17 = !DISubroutineType(types: !16)
// CHECK:STDOUT: !18 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !19 = !{null, !18}
// CHECK:STDOUT: !20 = !DISubroutineType(types: !19)
// CHECK:STDOUT: !21 = distinct !DISubprogram(name: "Foo", linkageName: "_CFoo.Main", scope: null, file: !14, line: 14, type: !17, spFlags: DISPFlagDefinition, unit: !15)
// CHECK:STDOUT: !22 = !DILocation(line: 14, column: 1, scope: !21)
// CHECK:STDOUT: !23 = distinct !DISubprogram(name: "Bar", linkageName: "_CBar.Main", scope: null, file: !14, line: 16, type: !17, spFlags: DISPFlagDefinition, unit: !15)
// CHECK:STDOUT: !28 = !DILocation(line: 16, column: 23, scope: !23)
// CHECK:STDOUT: !29 = distinct !DISubprogram(name: "Main", linkageName: "_CMain.Main", scope: null, file: !14, line: 18, type: !17, spFlags: DISPFlagDefinition, unit: !15)
// CHECK:STDOUT: !30 = !DILocation(line: 19, column: 3, scope: !29)
// CHECK:STDOUT: !31 = !DILocation(line: 19, column: 19, scope: !29)
// CHECK:STDOUT: !32 = !DILocation(line: 19, column: 15, scope: !29)
// CHECK:STDOUT: !34 = !DILocation(line: 18, column: 1, scope: !29)
// CHECK:STDOUT: !35 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.61ea2aba74ab3bf1:core.Destroy.Core", scope: null, file: !14, line: 19, type: !20, spFlags: DISPFlagDefinition, unit: !15, retainedNodes: !38)
// CHECK:STDOUT: !36 = !DILocalVariable(arg: 1, scope: !35, type: !18)
// CHECK:STDOUT: !37 = !DILocation(line: 19, column: 19, scope: !35)
// CHECK:STDOUT: !38 = !{!36}
// CHECK:STDOUT:
