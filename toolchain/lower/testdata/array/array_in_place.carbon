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
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !30 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca [2 x { i32, i32, i32 }], align 4, !dbg !31
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !31
// CHECK:STDOUT:   %.loc16_47.6.array.index = getelementptr inbounds [2 x { i32, i32, i32 }], ptr %_.var, i32 0, i64 0, !dbg !32
// CHECK:STDOUT:   call void @_CF.Main(ptr %.loc16_47.6.array.index), !dbg !33
// CHECK:STDOUT:   %.loc16_47.5.array.index = getelementptr inbounds [2 x { i32, i32, i32 }], ptr %_.var, i32 0, i64 1, !dbg !32
// CHECK:STDOUT:   call void @_CF.Main(ptr %.loc16_47.5.array.index), !dbg !34
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.5defec3482a23e54(ptr %_.var), !dbg !31
// CHECK:STDOUT:   ret void, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.9f53a29b4522c35d:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !37 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %tuple.elem2.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %self, i32 0, i32 2, !dbg !39
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.0aefc5da911f5051(ptr %tuple.elem2.tuple.elem), !dbg !39
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %self, i32 0, i32 1, !dbg !39
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.0aefc5da911f5051(ptr %tuple.elem1.tuple.elem), !dbg !39
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %self, i32 0, i32 0, !dbg !39
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.0aefc5da911f5051(ptr %tuple.elem0.tuple.elem), !dbg !39
// CHECK:STDOUT:   ret void, !dbg !39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.ac2160684fa165b4:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !43 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc16_3.1.array.index = getelementptr inbounds [2 x { i32, i32, i32 }], ptr %self, i32 0, i64 1, !dbg !45
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.e9f152c5b73f0cbf(ptr %.loc16_3.1.array.index), !dbg !45
// CHECK:STDOUT:   %.loc16_3.4.array.index = getelementptr inbounds [2 x { i32, i32, i32 }], ptr %self, i32 0, i64 0, !dbg !45
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.e9f152c5b73f0cbf(ptr %.loc16_3.4.array.index), !dbg !45
// CHECK:STDOUT:   ret void, !dbg !45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.5defec3482a23e54(ptr %t) #0 !dbg !47 {
// CHECK:STDOUT:   call void @"_COp.ac2160684fa165b4:core.SubobjectDestroy.Core"(ptr %t), !dbg !51
// CHECK:STDOUT:   ret void, !dbg !52
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.0aefc5da911f5051(ptr %t) #0 !dbg !54 {
// CHECK:STDOUT:   ret void, !dbg !59
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.e9f152c5b73f0cbf(ptr %t) #0 !dbg !61 {
// CHECK:STDOUT:   call void @"_COp.9f53a29b4522c35d:core.SubobjectDestroy.Core"(ptr %t), !dbg !65
// CHECK:STDOUT:   ret void, !dbg !66
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.0aefc5da911f5051, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.e9f152c5b73f0cbf, { 1, 0 }
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
// CHECK:STDOUT: !22 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !26 = !{null}
// CHECK:STDOUT: !27 = !DISubroutineType(types: !26)
// CHECK:STDOUT: !28 = !{null, !23}
// CHECK:STDOUT: !29 = !DISubroutineType(types: !28)
// CHECK:STDOUT: !30 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !20, line: 15, type: !27, spFlags: DISPFlagDefinition, unit: !21)
// CHECK:STDOUT: !31 = !DILocation(line: 16, column: 3, scope: !30)
// CHECK:STDOUT: !32 = !DILocation(line: 16, column: 38, scope: !30)
// CHECK:STDOUT: !33 = !DILocation(line: 16, column: 39, scope: !30)
// CHECK:STDOUT: !34 = !DILocation(line: 16, column: 44, scope: !30)
// CHECK:STDOUT: !36 = !DILocation(line: 15, column: 1, scope: !30)
// CHECK:STDOUT: !37 = distinct !DISubprogram(name: "Op", linkageName: "_COp.9f53a29b4522c35d:core.SubobjectDestroy.Core", scope: null, file: !20, line: 16, type: !29, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !42)
// CHECK:STDOUT: !38 = !DILocalVariable(arg: 1, scope: !37, type: !23)
// CHECK:STDOUT: !39 = !DILocation(line: 16, column: 3, scope: !37)
// CHECK:STDOUT: !40 = !{null, !22}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !42 = !{!38}
// CHECK:STDOUT: !43 = distinct !DISubprogram(name: "Op", linkageName: "_COp.ac2160684fa165b4:core.SubobjectDestroy.Core", scope: null, file: !20, line: 16, type: !29, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !46)
// CHECK:STDOUT: !44 = !DILocalVariable(arg: 1, scope: !43, type: !23)
// CHECK:STDOUT: !45 = !DILocation(line: 16, column: 3, scope: !43)
// CHECK:STDOUT: !46 = !{!44}
// CHECK:STDOUT: !47 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.5defec3482a23e54", scope: null, file: !14, line: 26, type: !29, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !53)
// CHECK:STDOUT: !48 = !DILocalVariable(arg: 1, scope: !47, type: !23)
// CHECK:STDOUT: !51 = !DILocation(line: 28, column: 5, scope: !47)
// CHECK:STDOUT: !52 = !DILocation(line: 26, column: 1, scope: !47)
// CHECK:STDOUT: !53 = !{!48}
// CHECK:STDOUT: !54 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.0aefc5da911f5051", scope: null, file: !14, line: 26, type: !41, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !60)
// CHECK:STDOUT: !55 = !DILocalVariable(arg: 1, scope: !54, type: !22)
// CHECK:STDOUT: !59 = !DILocation(line: 26, column: 1, scope: !54)
// CHECK:STDOUT: !60 = !{!55}
// CHECK:STDOUT: !61 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.e9f152c5b73f0cbf", scope: null, file: !14, line: 26, type: !29, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !67)
// CHECK:STDOUT: !62 = !DILocalVariable(arg: 1, scope: !61, type: !23)
// CHECK:STDOUT: !65 = !DILocation(line: 28, column: 5, scope: !61)
// CHECK:STDOUT: !66 = !DILocation(line: 26, column: 1, scope: !61)
// CHECK:STDOUT: !67 = !{!62}
// CHECK:STDOUT:
