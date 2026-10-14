// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/array.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/index/array_element_access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/index/array_element_access.carbon
fn A() -> (i32, i32) { return (1, 2); }

fn B() -> array(i32, 2) { return (1, 2); }

fn Run() {
  var a: array(i32, 2) = A();
  var b: i32 = 1;
  var _: i32 = a[b];
  var _: i32 = B()[1];
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'array_element_access.carbon'
// CHECK:STDOUT: source_filename = "array_element_access.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @tuple.380.loc12_37 = internal constant { i32, i32 } { i32 1, i32 2 }
// CHECK:STDOUT: @array.loc14_40 = internal constant [2 x i32] [i32 1, i32 2]
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CA.Main(ptr sret({ i32, i32 }) %return) #0 !dbg !30 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 0, !dbg !33
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 1, !dbg !33
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %return, ptr align 4 @tuple.380.loc12_37, i64 8, i1 false), !dbg !34
// CHECK:STDOUT:   ret void, !dbg !34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CB.Main(ptr sret([2 x i32]) %return) #0 !dbg !35 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc14_39.3.array.index = getelementptr inbounds [2 x i32], ptr %return, i32 0, i64 0, !dbg !38
// CHECK:STDOUT:   %.loc14_39.6.array.index = getelementptr inbounds [2 x i32], ptr %return, i32 0, i64 1, !dbg !38
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %return, ptr align 4 @array.loc14_40, i64 8, i1 false), !dbg !39
// CHECK:STDOUT:   ret void, !dbg !39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @main() #0 !dbg !40 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca [2 x i32], align 4, !dbg !41
// CHECK:STDOUT:   %.loc17_28.1.temp = alloca { i32, i32 }, align 4, !dbg !42
// CHECK:STDOUT:   %b.var = alloca i32, align 4, !dbg !44
// CHECK:STDOUT:   %_.var.loc19 = alloca i32, align 4, !dbg !46
// CHECK:STDOUT:   %_.var.loc20 = alloca i32, align 4, !dbg !50
// CHECK:STDOUT:   %.loc20_18.1.temp = alloca [2 x i32], align 4, !dbg !51
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !41
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc17_28.1.temp), !dbg !42
// CHECK:STDOUT:   call void @_CA.Main(ptr %.loc17_28.1.temp), !dbg !42
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %.loc17_28.1.temp, i32 0, i32 0, !dbg !42
// CHECK:STDOUT:   %.loc17_28.3 = load i32, ptr %tuple.elem0.tuple.elem, align 4, !dbg !42
// CHECK:STDOUT:   %.loc17_28.4.array.index = getelementptr inbounds [2 x i32], ptr %a.var, i32 0, i64 0, !dbg !42
// CHECK:STDOUT:   store i32 %.loc17_28.3, ptr %.loc17_28.4.array.index, align 4, !dbg !42
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %.loc17_28.1.temp, i32 0, i32 1, !dbg !42
// CHECK:STDOUT:   %.loc17_28.6 = load i32, ptr %tuple.elem1.tuple.elem, align 4, !dbg !42
// CHECK:STDOUT:   %.loc17_28.7.array.index = getelementptr inbounds [2 x i32], ptr %a.var, i32 0, i64 1, !dbg !42
// CHECK:STDOUT:   store i32 %.loc17_28.6, ptr %.loc17_28.7.array.index, align 4, !dbg !42
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b.var), !dbg !44
// CHECK:STDOUT:   store i32 1, ptr %b.var, align 4, !dbg !44
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc19), !dbg !46
// CHECK:STDOUT:   %.loc19_18 = load i32, ptr %b.var, align 4, !dbg !48
// CHECK:STDOUT:   %.loc19_19.1.array.index = getelementptr inbounds [2 x i32], ptr %a.var, i32 0, i32 %.loc19_18, !dbg !47
// CHECK:STDOUT:   %.loc19_19.2 = load i32, ptr %.loc19_19.1.array.index, align 4, !dbg !47
// CHECK:STDOUT:   store i32 %.loc19_19.2, ptr %_.var.loc19, align 4, !dbg !46
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc20), !dbg !50
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc20_18.1.temp), !dbg !51
// CHECK:STDOUT:   call void @_CB.Main(ptr %.loc20_18.1.temp), !dbg !51
// CHECK:STDOUT:   %.loc20_21.1.array.index = getelementptr inbounds [2 x i32], ptr %.loc20_18.1.temp, i32 0, i32 1, !dbg !51
// CHECK:STDOUT:   %.loc20_21.2 = load i32, ptr %.loc20_21.1.array.index, align 4, !dbg !51
// CHECK:STDOUT:   store i32 %.loc20_21.2, ptr %_.var.loc20, align 4, !dbg !50
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.a9c4ae3b4c910d5b(ptr %.loc20_18.1.temp), !dbg !51
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.dc5658e0ac74b2aa(ptr %_.var.loc20), !dbg !50
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.dc5658e0ac74b2aa(ptr %_.var.loc19), !dbg !46
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.dc5658e0ac74b2aa(ptr %b.var), !dbg !44
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.eda669d623214d41(ptr %.loc17_28.1.temp), !dbg !42
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.a9c4ae3b4c910d5b(ptr %a.var), !dbg !41
// CHECK:STDOUT:   ret i32 0, !dbg !55
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.641010f66e2ea46f:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !56 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc20_18.1.array.index = getelementptr inbounds [2 x i32], ptr %self, i32 0, i64 1, !dbg !58
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.dc5658e0ac74b2aa(ptr %.loc20_18.1.array.index), !dbg !58
// CHECK:STDOUT:   %.loc20_18.4.array.index = getelementptr inbounds [2 x i32], ptr %self, i32 0, i64 0, !dbg !58
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.dc5658e0ac74b2aa(ptr %.loc20_18.4.array.index), !dbg !58
// CHECK:STDOUT:   ret void, !dbg !58
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.35f6012581c5aabb:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !60 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 1, !dbg !62
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.dc5658e0ac74b2aa(ptr %tuple.elem1.tuple.elem), !dbg !62
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 0, !dbg !62
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.dc5658e0ac74b2aa(ptr %tuple.elem0.tuple.elem), !dbg !62
// CHECK:STDOUT:   ret void, !dbg !62
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.a9c4ae3b4c910d5b(ptr %t) #0 !dbg !64 {
// CHECK:STDOUT:   call void @"_COp.641010f66e2ea46f:core.SubobjectDestroy.Core"(ptr %t), !dbg !68
// CHECK:STDOUT:   ret void, !dbg !69
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.dc5658e0ac74b2aa(ptr %t) #0 !dbg !71 {
// CHECK:STDOUT:   ret void, !dbg !76
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.eda669d623214d41(ptr %t) #0 !dbg !78 {
// CHECK:STDOUT:   call void @"_COp.35f6012581c5aabb:core.SubobjectDestroy.Core"(ptr %t), !dbg !82
// CHECK:STDOUT:   ret void, !dbg !83
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.a9c4ae3b4c910d5b, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.dc5658e0ac74b2aa, { 6, 5, 4, 3, 2, 1, 0 }
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
// CHECK:STDOUT: !20 = !DIFile(filename: "array_element_access.carbon", directory: "")
// CHECK:STDOUT: !21 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !20, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !22 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{!23}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = !{!22}
// CHECK:STDOUT: !27 = !DISubroutineType(types: !26)
// CHECK:STDOUT: !28 = !{null, !23}
// CHECK:STDOUT: !29 = !DISubroutineType(types: !28)
// CHECK:STDOUT: !30 = distinct !DISubprogram(name: "A", linkageName: "_CA.Main", scope: null, file: !20, line: 12, type: !25, spFlags: DISPFlagDefinition, unit: !21)
// CHECK:STDOUT: !33 = !DILocation(line: 12, column: 31, scope: !30)
// CHECK:STDOUT: !34 = !DILocation(line: 12, column: 24, scope: !30)
// CHECK:STDOUT: !35 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main", scope: null, file: !20, line: 14, type: !25, spFlags: DISPFlagDefinition, unit: !21)
// CHECK:STDOUT: !38 = !DILocation(line: 14, column: 34, scope: !35)
// CHECK:STDOUT: !39 = !DILocation(line: 14, column: 27, scope: !35)
// CHECK:STDOUT: !40 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !20, line: 16, type: !27, spFlags: DISPFlagDefinition, unit: !21)
// CHECK:STDOUT: !41 = !DILocation(line: 17, column: 3, scope: !40)
// CHECK:STDOUT: !42 = !DILocation(line: 17, column: 26, scope: !40)
// CHECK:STDOUT: !44 = !DILocation(line: 18, column: 3, scope: !40)
// CHECK:STDOUT: !46 = !DILocation(line: 19, column: 3, scope: !40)
// CHECK:STDOUT: !47 = !DILocation(line: 19, column: 16, scope: !40)
// CHECK:STDOUT: !48 = !DILocation(line: 19, column: 18, scope: !40)
// CHECK:STDOUT: !50 = !DILocation(line: 20, column: 3, scope: !40)
// CHECK:STDOUT: !51 = !DILocation(line: 20, column: 16, scope: !40)
// CHECK:STDOUT: !53 = !{null, !22}
// CHECK:STDOUT: !54 = !DISubroutineType(types: !53)
// CHECK:STDOUT: !55 = !DILocation(line: 16, column: 1, scope: !40)
// CHECK:STDOUT: !56 = distinct !DISubprogram(name: "Op", linkageName: "_COp.641010f66e2ea46f:core.SubobjectDestroy.Core", scope: null, file: !20, line: 20, type: !29, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !59)
// CHECK:STDOUT: !57 = !DILocalVariable(arg: 1, scope: !56, type: !23)
// CHECK:STDOUT: !58 = !DILocation(line: 20, column: 16, scope: !56)
// CHECK:STDOUT: !59 = !{!57}
// CHECK:STDOUT: !60 = distinct !DISubprogram(name: "Op", linkageName: "_COp.35f6012581c5aabb:core.SubobjectDestroy.Core", scope: null, file: !20, line: 17, type: !29, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !63)
// CHECK:STDOUT: !61 = !DILocalVariable(arg: 1, scope: !60, type: !23)
// CHECK:STDOUT: !62 = !DILocation(line: 17, column: 26, scope: !60)
// CHECK:STDOUT: !63 = !{!61}
// CHECK:STDOUT: !64 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.a9c4ae3b4c910d5b", scope: null, file: !14, line: 30, type: !29, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !70)
// CHECK:STDOUT: !65 = !DILocalVariable(arg: 1, scope: !64, type: !23)
// CHECK:STDOUT: !68 = !DILocation(line: 32, column: 5, scope: !64)
// CHECK:STDOUT: !69 = !DILocation(line: 30, column: 1, scope: !64)
// CHECK:STDOUT: !70 = !{!65}
// CHECK:STDOUT: !71 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.dc5658e0ac74b2aa", scope: null, file: !14, line: 30, type: !54, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !77)
// CHECK:STDOUT: !72 = !DILocalVariable(arg: 1, scope: !71, type: !22)
// CHECK:STDOUT: !76 = !DILocation(line: 30, column: 1, scope: !71)
// CHECK:STDOUT: !77 = !{!72}
// CHECK:STDOUT: !78 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.eda669d623214d41", scope: null, file: !14, line: 30, type: !29, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !84)
// CHECK:STDOUT: !79 = !DILocalVariable(arg: 1, scope: !78, type: !23)
// CHECK:STDOUT: !82 = !DILocation(line: 32, column: 5, scope: !78)
// CHECK:STDOUT: !83 = !DILocation(line: 30, column: 1, scope: !78)
// CHECK:STDOUT: !84 = !{!79}
// CHECK:STDOUT:
