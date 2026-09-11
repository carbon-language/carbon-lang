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
// CHECK:STDOUT: define void @_CFoo.Main() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CBar.Main() #0 !dbg !8 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMain.Main() #0 !dbg !10 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca {}, align 1, !dbg !11
// CHECK:STDOUT:   %.loc19_23.1.temp = alloca {}, align 1, !dbg !12
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !11
// CHECK:STDOUT:   call void @_CFoo.Main(), !dbg !12
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc19_23.1.temp), !dbg !12
// CHECK:STDOUT:   call void @_CBar.Main(), !dbg !13
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.c7a99711ca3557ac(ptr %.loc19_23.1.temp), !dbg !12
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.c7a99711ca3557ac(ptr %_.var), !dbg !11
// CHECK:STDOUT:   ret void, !dbg !14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.c7a99711ca3557ac(ptr %self) #0 !dbg !15 {
// CHECK:STDOUT:   ret void, !dbg !22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Destroy.Core.c7a99711ca3557ac, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "implicit_empty_tuple_as_arg.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Foo", linkageName: "_CFoo.Main", scope: null, file: !1, line: 14, type: !5, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DILocation(line: 14, column: 1, scope: !4)
// CHECK:STDOUT: !8 = distinct !DISubprogram(name: "Bar", linkageName: "_CBar.Main", scope: null, file: !1, line: 16, type: !5, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !9 = !DILocation(line: 16, column: 23, scope: !8)
// CHECK:STDOUT: !10 = distinct !DISubprogram(name: "Main", linkageName: "_CMain.Main", scope: null, file: !1, line: 18, type: !5, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !11 = !DILocation(line: 19, column: 3, scope: !10)
// CHECK:STDOUT: !12 = !DILocation(line: 19, column: 19, scope: !10)
// CHECK:STDOUT: !13 = !DILocation(line: 19, column: 15, scope: !10)
// CHECK:STDOUT: !14 = !DILocation(line: 18, column: 1, scope: !10)
// CHECK:STDOUT: !15 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.c7a99711ca3557ac", scope: null, file: !16, line: 24, type: !17, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !20)
// CHECK:STDOUT: !16 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !17 = !DISubroutineType(types: !18)
// CHECK:STDOUT: !18 = !{null, !19}
// CHECK:STDOUT: !19 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !20 = !{!21}
// CHECK:STDOUT: !21 = !DILocalVariable(arg: 1, scope: !15, type: !19)
// CHECK:STDOUT: !22 = !DILocation(line: 24, column: 3, scope: !15)
// CHECK:STDOUT:
