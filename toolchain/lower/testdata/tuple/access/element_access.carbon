// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/tuple/access/element_access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/tuple/access/element_access.carbon

fn Run() -> i32 {
  var a: (i32, i32, i32) = (0, 1, 2);
  var _: i32 = a.0;
  var _: i32 = a.2;
  return 0;
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'element_access.carbon'
// CHECK:STDOUT: source_filename = "element_access.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @tuple.262.loc14_3.1 = internal constant { i32, i32, i32 } { i32 0, i32 1, i32 2 }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @main() #0 !dbg !26 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca { i32, i32, i32 }, align 4, !dbg !29
// CHECK:STDOUT:   %_.var.loc15 = alloca i32, align 4, !dbg !32
// CHECK:STDOUT:   %_.var.loc16 = alloca i32, align 4, !dbg !35
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !29
// CHECK:STDOUT:   %tuple.elem0.loc14.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %a.var, i32 0, i32 0, !dbg !30
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %a.var, i32 0, i32 1, !dbg !30
// CHECK:STDOUT:   %tuple.elem2.loc14.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %a.var, i32 0, i32 2, !dbg !30
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %a.var, ptr align 4 @tuple.262.loc14_3.1, i64 12, i1 false), !dbg !29
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc15), !dbg !32
// CHECK:STDOUT:   %tuple.elem0.loc15.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %a.var, i32 0, i32 0, !dbg !33
// CHECK:STDOUT:   %.loc15_17 = load i32, ptr %tuple.elem0.loc15.tuple.elem, align 4, !dbg !33
// CHECK:STDOUT:   store i32 %.loc15_17, ptr %_.var.loc15, align 4, !dbg !32
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc16), !dbg !35
// CHECK:STDOUT:   %tuple.elem2.loc16.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %a.var, i32 0, i32 2, !dbg !36
// CHECK:STDOUT:   %.loc16_17 = load i32, ptr %tuple.elem2.loc16.tuple.elem, align 4, !dbg !36
// CHECK:STDOUT:   store i32 %.loc16_17, ptr %_.var.loc16, align 4, !dbg !35
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.0aefc5da911f5051(ptr %_.var.loc16), !dbg !35
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.0aefc5da911f5051(ptr %_.var.loc15), !dbg !32
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.e9f152c5b73f0cbf(ptr %a.var), !dbg !29
// CHECK:STDOUT:   ret i32 0, !dbg !40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.9f53a29b4522c35d:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !41 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %tuple.elem2.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %self, i32 0, i32 2, !dbg !43
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.0aefc5da911f5051(ptr %tuple.elem2.tuple.elem), !dbg !43
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %self, i32 0, i32 1, !dbg !43
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.0aefc5da911f5051(ptr %tuple.elem1.tuple.elem), !dbg !43
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %self, i32 0, i32 0, !dbg !43
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.0aefc5da911f5051(ptr %tuple.elem0.tuple.elem), !dbg !43
// CHECK:STDOUT:   ret void, !dbg !43
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.0aefc5da911f5051(ptr %t) #0 !dbg !45 {
// CHECK:STDOUT:   ret void, !dbg !50
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.e9f152c5b73f0cbf(ptr %t) #0 !dbg !52 {
// CHECK:STDOUT:   call void @"_COp.9f53a29b4522c35d:core.SubobjectDestroy.Core"(ptr %t), !dbg !56
// CHECK:STDOUT:   ret void, !dbg !57
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.0aefc5da911f5051, { 4, 3, 2, 1, 0 }
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
// CHECK:STDOUT: !18 = !DIFile(filename: "element_access.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !{!20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{null, !23}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !18, line: 13, type: !22, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !29 = !DILocation(line: 14, column: 3, scope: !26)
// CHECK:STDOUT: !30 = !DILocation(line: 14, column: 28, scope: !26)
// CHECK:STDOUT: !32 = !DILocation(line: 15, column: 3, scope: !26)
// CHECK:STDOUT: !33 = !DILocation(line: 15, column: 16, scope: !26)
// CHECK:STDOUT: !35 = !DILocation(line: 16, column: 3, scope: !26)
// CHECK:STDOUT: !36 = !DILocation(line: 16, column: 16, scope: !26)
// CHECK:STDOUT: !38 = !{null, !20}
// CHECK:STDOUT: !39 = !DISubroutineType(types: !38)
// CHECK:STDOUT: !40 = !DILocation(line: 17, column: 3, scope: !26)
// CHECK:STDOUT: !41 = distinct !DISubprogram(name: "Op", linkageName: "_COp.9f53a29b4522c35d:core.SubobjectDestroy.Core", scope: null, file: !18, line: 14, type: !25, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !44)
// CHECK:STDOUT: !42 = !DILocalVariable(arg: 1, scope: !41, type: !23)
// CHECK:STDOUT: !43 = !DILocation(line: 14, column: 3, scope: !41)
// CHECK:STDOUT: !44 = !{!42}
// CHECK:STDOUT: !45 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.0aefc5da911f5051", scope: null, file: !10, line: 26, type: !39, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !51)
// CHECK:STDOUT: !46 = !DILocalVariable(arg: 1, scope: !45, type: !20)
// CHECK:STDOUT: !50 = !DILocation(line: 26, column: 1, scope: !45)
// CHECK:STDOUT: !51 = !{!46}
// CHECK:STDOUT: !52 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.e9f152c5b73f0cbf", scope: null, file: !10, line: 26, type: !25, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !58)
// CHECK:STDOUT: !53 = !DILocalVariable(arg: 1, scope: !52, type: !23)
// CHECK:STDOUT: !56 = !DILocation(line: 28, column: 5, scope: !52)
// CHECK:STDOUT: !57 = !DILocation(line: 26, column: 1, scope: !52)
// CHECK:STDOUT: !58 = !{!53}
// CHECK:STDOUT:
