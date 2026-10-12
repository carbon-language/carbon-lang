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

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'tuple_param_with_return_slot.carbon'
// CHECK:STDOUT: source_filename = "tuple_param_with_return_slot.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @tuple.50f.loc18_20.6 = internal constant { i32, i32 } { i32 2, i32 3 }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main(ptr sret({ i32, i32, i32 }) %return, { i32 } %b, ptr %c) #0 !dbg !28 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %tuple.elem0.loc14_12.tuple.elem = extractvalue { i32 } %b, 0, !dbg !36
// CHECK:STDOUT:   %tuple.elem0.loc14_17.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %c, i32 0, i32 0, !dbg !37
// CHECK:STDOUT:   %tuple.elem0.loc14_17.tuple.elem.load = load i32, ptr %tuple.elem0.loc14_17.tuple.elem, align 4, !dbg !37
// CHECK:STDOUT:   %tuple.elem1.loc14_22.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %c, i32 0, i32 1, !dbg !38
// CHECK:STDOUT:   %tuple.elem1.loc14_22.tuple.elem.load = load i32, ptr %tuple.elem1.loc14_22.tuple.elem, align 4, !dbg !38
// CHECK:STDOUT:   %tuple.elem0.loc14_24.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %return, i32 0, i32 0, !dbg !39
// CHECK:STDOUT:   store i32 %tuple.elem0.loc14_12.tuple.elem, ptr %tuple.elem0.loc14_24.tuple.elem, align 4, !dbg !39
// CHECK:STDOUT:   %tuple.elem1.loc14_24.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %return, i32 0, i32 1, !dbg !39
// CHECK:STDOUT:   store i32 %tuple.elem0.loc14_17.tuple.elem.load, ptr %tuple.elem1.loc14_24.tuple.elem, align 4, !dbg !39
// CHECK:STDOUT:   %tuple.elem2.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %return, i32 0, i32 2, !dbg !39
// CHECK:STDOUT:   store i32 %tuple.elem1.loc14_22.tuple.elem.load, ptr %tuple.elem2.tuple.elem, align 4, !dbg !39
// CHECK:STDOUT:   ret void, !dbg !40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMain.Main() #0 !dbg !42 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc18_21.1.temp = alloca { i32, i32, i32 }, align 4, !dbg !43
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc18_21.1.temp), !dbg !43
// CHECK:STDOUT:   call void @_CF.Main(ptr %.loc18_21.1.temp, { i32 } { i32 1 }, ptr @tuple.50f.loc18_20.6), !dbg !43
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.e9f152c5b73f0cbf(ptr %.loc18_21.1.temp), !dbg !43
// CHECK:STDOUT:   ret void, !dbg !44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.9f53a29b4522c35d:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !45 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %tuple.elem2.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %self, i32 0, i32 2, !dbg !47
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.0aefc5da911f5051(ptr %tuple.elem2.tuple.elem), !dbg !47
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %self, i32 0, i32 1, !dbg !47
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.0aefc5da911f5051(ptr %tuple.elem1.tuple.elem), !dbg !47
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %self, i32 0, i32 0, !dbg !47
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.0aefc5da911f5051(ptr %tuple.elem0.tuple.elem), !dbg !47
// CHECK:STDOUT:   ret void, !dbg !47
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.e9f152c5b73f0cbf(ptr %t) #0 !dbg !51 {
// CHECK:STDOUT:   call void @"_COp.9f53a29b4522c35d:core.SubobjectDestroy.Core"(ptr %t), !dbg !55
// CHECK:STDOUT:   ret void, !dbg !56
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.0aefc5da911f5051(ptr %t) #0 !dbg !58 {
// CHECK:STDOUT:   ret void, !dbg !63
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.0aefc5da911f5051, { 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !18 = !DIFile(filename: "tuple_param_with_return_slot.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{!21, !21, !21}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !24 = !{null}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = !{null, !21}
// CHECK:STDOUT: !27 = !DISubroutineType(types: !26)
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !18, line: 13, type: !23, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !41)
// CHECK:STDOUT: !29 = !DILocalVariable(arg: 1, scope: !28, type: !21)
// CHECK:STDOUT: !30 = !DILocalVariable(arg: 2, scope: !28, type: !21)
// CHECK:STDOUT: !36 = !DILocation(line: 14, column: 11, scope: !28)
// CHECK:STDOUT: !37 = !DILocation(line: 14, column: 16, scope: !28)
// CHECK:STDOUT: !38 = !DILocation(line: 14, column: 21, scope: !28)
// CHECK:STDOUT: !39 = !DILocation(line: 14, column: 10, scope: !28)
// CHECK:STDOUT: !40 = !DILocation(line: 14, column: 3, scope: !28)
// CHECK:STDOUT: !41 = !{!29, !30}
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "Main", linkageName: "_CMain.Main", scope: null, file: !18, line: 17, type: !25, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !43 = !DILocation(line: 18, column: 3, scope: !42)
// CHECK:STDOUT: !44 = !DILocation(line: 17, column: 1, scope: !42)
// CHECK:STDOUT: !45 = distinct !DISubprogram(name: "Op", linkageName: "_COp.9f53a29b4522c35d:core.SubobjectDestroy.Core", scope: null, file: !18, line: 18, type: !27, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !50)
// CHECK:STDOUT: !46 = !DILocalVariable(arg: 1, scope: !45, type: !21)
// CHECK:STDOUT: !47 = !DILocation(line: 18, column: 3, scope: !45)
// CHECK:STDOUT: !48 = !{null, !20}
// CHECK:STDOUT: !49 = !DISubroutineType(types: !48)
// CHECK:STDOUT: !50 = !{!46}
// CHECK:STDOUT: !51 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.e9f152c5b73f0cbf", scope: null, file: !10, line: 26, type: !27, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !57)
// CHECK:STDOUT: !52 = !DILocalVariable(arg: 1, scope: !51, type: !21)
// CHECK:STDOUT: !55 = !DILocation(line: 28, column: 5, scope: !51)
// CHECK:STDOUT: !56 = !DILocation(line: 26, column: 1, scope: !51)
// CHECK:STDOUT: !57 = !{!52}
// CHECK:STDOUT: !58 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.0aefc5da911f5051", scope: null, file: !10, line: 26, type: !49, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !64)
// CHECK:STDOUT: !59 = !DILocalVariable(arg: 1, scope: !58, type: !20)
// CHECK:STDOUT: !63 = !DILocation(line: 26, column: 1, scope: !58)
// CHECK:STDOUT: !64 = !{!59}
// CHECK:STDOUT:
