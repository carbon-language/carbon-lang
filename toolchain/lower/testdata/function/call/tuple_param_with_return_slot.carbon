// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/call/tuple_param_with_return_slot.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/call/tuple_param_with_return_slot.carbon

fn F(_: (), b: (i32,), c: (i32, i32)) -> (i32, i32, i32) {
  return (b.0, c.0, c.1);
}

fn Main() {
  F((), (1,), (2, 3));
}

// CHECK:STDOUT: ; ModuleID = 'tuple_param_with_return_slot.carbon'
// CHECK:STDOUT: source_filename = "tuple_param_with_return_slot.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @tuple.7e8.loc18_20.6 = internal constant { i32, i32 } { i32 2, i32 3 }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main(ptr sret({ i32, i32, i32 }) %return, { i32 } %b, ptr %c) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %tuple.elem0.loc14_12.tuple.elem = extractvalue { i32 } %b, 0, !dbg !11
// CHECK:STDOUT:   %tuple.elem0.loc14_17.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %c, i32 0, i32 0, !dbg !12
// CHECK:STDOUT:   %tuple.elem0.loc14_17.tuple.elem.load = load i32, ptr %tuple.elem0.loc14_17.tuple.elem, align 4, !dbg !12
// CHECK:STDOUT:   %tuple.elem1.loc14_22.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %c, i32 0, i32 1, !dbg !13
// CHECK:STDOUT:   %tuple.elem1.loc14_22.tuple.elem.load = load i32, ptr %tuple.elem1.loc14_22.tuple.elem, align 4, !dbg !13
// CHECK:STDOUT:   %tuple.elem0.loc14_24.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %return, i32 0, i32 0, !dbg !14
// CHECK:STDOUT:   store i32 %tuple.elem0.loc14_12.tuple.elem, ptr %tuple.elem0.loc14_24.tuple.elem, align 4, !dbg !14
// CHECK:STDOUT:   %tuple.elem1.loc14_24.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %return, i32 0, i32 1, !dbg !14
// CHECK:STDOUT:   store i32 %tuple.elem0.loc14_17.tuple.elem.load, ptr %tuple.elem1.loc14_24.tuple.elem, align 4, !dbg !14
// CHECK:STDOUT:   %tuple.elem2.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %return, i32 0, i32 2, !dbg !14
// CHECK:STDOUT:   store i32 %tuple.elem1.loc14_22.tuple.elem.load, ptr %tuple.elem2.tuple.elem, align 4, !dbg !14
// CHECK:STDOUT:   ret void, !dbg !15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMain.Main() #0 !dbg !16 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc18_21.1.temp = alloca { i32, i32, i32 }, align 4, !dbg !19
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc18_21.1.temp), !dbg !19
// CHECK:STDOUT:   call void @_CF.Main(ptr %.loc18_21.1.temp, { i32 } { i32 1 }, ptr @tuple.7e8.loc18_20.6), !dbg !19
// CHECK:STDOUT:   call void @"_COp.d52e974a95b4c7a0:core.Destroy.Core"(ptr %.loc18_21.1.temp), !dbg !19
// CHECK:STDOUT:   ret void, !dbg !20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.1fa36cd3e044ac37:core.Destroy.Core"(ptr %self) #0 !dbg !21 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.d52e974a95b4c7a0:core.Destroy.Core"(ptr %self) #0 !dbg !28 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "tuple_param_with_return_slot.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !3, line: 13, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !8)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7, !7, !7}
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{!9, !10}
// CHECK:STDOUT: !9 = !DILocalVariable(arg: 1, scope: !4, type: !7)
// CHECK:STDOUT: !10 = !DILocalVariable(arg: 2, scope: !4, type: !7)
// CHECK:STDOUT: !11 = !DILocation(line: 14, column: 11, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 14, column: 16, scope: !4)
// CHECK:STDOUT: !13 = !DILocation(line: 14, column: 21, scope: !4)
// CHECK:STDOUT: !14 = !DILocation(line: 14, column: 10, scope: !4)
// CHECK:STDOUT: !15 = !DILocation(line: 14, column: 3, scope: !4)
// CHECK:STDOUT: !16 = distinct !DISubprogram(name: "Main", linkageName: "_CMain.Main", scope: null, file: !3, line: 17, type: !17, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !17 = !DISubroutineType(types: !18)
// CHECK:STDOUT: !18 = !{null}
// CHECK:STDOUT: !19 = !DILocation(line: 18, column: 3, scope: !16)
// CHECK:STDOUT: !20 = !DILocation(line: 17, column: 1, scope: !16)
// CHECK:STDOUT: !21 = distinct !DISubprogram(name: "Op", linkageName: "_COp.1fa36cd3e044ac37:core.Destroy.Core", scope: null, file: !3, line: 18, type: !22, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !25)
// CHECK:STDOUT: !22 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !23 = !{null, !24}
// CHECK:STDOUT: !24 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !25 = !{!26}
// CHECK:STDOUT: !26 = !DILocalVariable(arg: 1, scope: !21, type: !24)
// CHECK:STDOUT: !27 = !DILocation(line: 18, column: 3, scope: !21)
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "Op", linkageName: "_COp.d52e974a95b4c7a0:core.Destroy.Core", scope: null, file: !3, line: 18, type: !29, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !31)
// CHECK:STDOUT: !29 = !DISubroutineType(types: !30)
// CHECK:STDOUT: !30 = !{null, !7}
// CHECK:STDOUT: !31 = !{!32}
// CHECK:STDOUT: !32 = !DILocalVariable(arg: 1, scope: !28, type: !7)
// CHECK:STDOUT: !33 = !DILocation(line: 18, column: 3, scope: !28)
