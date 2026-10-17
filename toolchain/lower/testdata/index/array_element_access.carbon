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
// CHECK:STDOUT: define void @_CA.Main(ptr sret({ i32, i32 }) %return) #0 !dbg !28 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 0, !dbg !31
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 1, !dbg !31
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %return, ptr align 4 @tuple.380.loc12_37, i64 8, i1 false), !dbg !32
// CHECK:STDOUT:   ret void, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CB.Main(ptr sret([2 x i32]) %return) #0 !dbg !33 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc14_39.3.array.index = getelementptr inbounds [2 x i32], ptr %return, i32 0, i64 0, !dbg !36
// CHECK:STDOUT:   %.loc14_39.6.array.index = getelementptr inbounds [2 x i32], ptr %return, i32 0, i64 1, !dbg !36
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %return, ptr align 4 @array.loc14_40, i64 8, i1 false), !dbg !37
// CHECK:STDOUT:   ret void, !dbg !37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @main() #0 !dbg !38 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca [2 x i32], align 4, !dbg !39
// CHECK:STDOUT:   %.loc17_28.1.temp = alloca { i32, i32 }, align 4, !dbg !40
// CHECK:STDOUT:   %b.var = alloca i32, align 4, !dbg !42
// CHECK:STDOUT:   %_.var.loc19 = alloca i32, align 4, !dbg !44
// CHECK:STDOUT:   %_.var.loc20 = alloca i32, align 4, !dbg !48
// CHECK:STDOUT:   %.loc20_18.1.temp = alloca [2 x i32], align 4, !dbg !49
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !39
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc17_28.1.temp), !dbg !40
// CHECK:STDOUT:   call void @_CA.Main(ptr %.loc17_28.1.temp), !dbg !40
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %.loc17_28.1.temp, i32 0, i32 0, !dbg !40
// CHECK:STDOUT:   %.loc17_28.3 = load i32, ptr %tuple.elem0.tuple.elem, align 4, !dbg !40
// CHECK:STDOUT:   %.loc17_28.4.array.index = getelementptr inbounds [2 x i32], ptr %a.var, i32 0, i64 0, !dbg !40
// CHECK:STDOUT:   store i32 %.loc17_28.3, ptr %.loc17_28.4.array.index, align 4, !dbg !40
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %.loc17_28.1.temp, i32 0, i32 1, !dbg !40
// CHECK:STDOUT:   %.loc17_28.6 = load i32, ptr %tuple.elem1.tuple.elem, align 4, !dbg !40
// CHECK:STDOUT:   %.loc17_28.7.array.index = getelementptr inbounds [2 x i32], ptr %a.var, i32 0, i64 1, !dbg !40
// CHECK:STDOUT:   store i32 %.loc17_28.6, ptr %.loc17_28.7.array.index, align 4, !dbg !40
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b.var), !dbg !42
// CHECK:STDOUT:   store i32 1, ptr %b.var, align 4, !dbg !42
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc19), !dbg !44
// CHECK:STDOUT:   %.loc19_18 = load i32, ptr %b.var, align 4, !dbg !46
// CHECK:STDOUT:   %.loc19_19.1.array.index = getelementptr inbounds [2 x i32], ptr %a.var, i32 0, i32 %.loc19_18, !dbg !45
// CHECK:STDOUT:   %.loc19_19.2 = load i32, ptr %.loc19_19.1.array.index, align 4, !dbg !45
// CHECK:STDOUT:   store i32 %.loc19_19.2, ptr %_.var.loc19, align 4, !dbg !44
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc20), !dbg !48
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc20_18.1.temp), !dbg !49
// CHECK:STDOUT:   call void @_CB.Main(ptr %.loc20_18.1.temp), !dbg !49
// CHECK:STDOUT:   %.loc20_21.1.array.index = getelementptr inbounds [2 x i32], ptr %.loc20_18.1.temp, i32 0, i32 1, !dbg !49
// CHECK:STDOUT:   %.loc20_21.2 = load i32, ptr %.loc20_21.1.array.index, align 4, !dbg !49
// CHECK:STDOUT:   store i32 %.loc20_21.2, ptr %_.var.loc20, align 4, !dbg !48
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.a9c4ae3b4c910d5b(ptr %.loc20_18.1.temp), !dbg !49
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.dc5658e0ac74b2aa(ptr %_.var.loc20), !dbg !48
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.dc5658e0ac74b2aa(ptr %_.var.loc19), !dbg !44
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.dc5658e0ac74b2aa(ptr %b.var), !dbg !42
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.eda669d623214d41(ptr %.loc17_28.1.temp), !dbg !40
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.a9c4ae3b4c910d5b(ptr %a.var), !dbg !39
// CHECK:STDOUT:   ret i32 0, !dbg !55
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.a9c4ae3b4c910d5b(ptr %t) #0 !dbg !56 {
// CHECK:STDOUT:   ret void, !dbg !61
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.dc5658e0ac74b2aa(ptr %t) #0 !dbg !63 {
// CHECK:STDOUT:   ret void, !dbg !68
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.eda669d623214d41(ptr %t) #0 !dbg !70 {
// CHECK:STDOUT:   ret void, !dbg !75
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.a9c4ae3b4c910d5b, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.dc5658e0ac74b2aa, { 2, 1, 0 }
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
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "A", linkageName: "_CA.Main", scope: null, file: !20, line: 12, type: !25, spFlags: DISPFlagDefinition, unit: !21)
// CHECK:STDOUT: !31 = !DILocation(line: 12, column: 31, scope: !28)
// CHECK:STDOUT: !32 = !DILocation(line: 12, column: 24, scope: !28)
// CHECK:STDOUT: !33 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main", scope: null, file: !20, line: 14, type: !25, spFlags: DISPFlagDefinition, unit: !21)
// CHECK:STDOUT: !36 = !DILocation(line: 14, column: 34, scope: !33)
// CHECK:STDOUT: !37 = !DILocation(line: 14, column: 27, scope: !33)
// CHECK:STDOUT: !38 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !20, line: 16, type: !27, spFlags: DISPFlagDefinition, unit: !21)
// CHECK:STDOUT: !39 = !DILocation(line: 17, column: 3, scope: !38)
// CHECK:STDOUT: !40 = !DILocation(line: 17, column: 26, scope: !38)
// CHECK:STDOUT: !42 = !DILocation(line: 18, column: 3, scope: !38)
// CHECK:STDOUT: !44 = !DILocation(line: 19, column: 3, scope: !38)
// CHECK:STDOUT: !45 = !DILocation(line: 19, column: 16, scope: !38)
// CHECK:STDOUT: !46 = !DILocation(line: 19, column: 18, scope: !38)
// CHECK:STDOUT: !48 = !DILocation(line: 20, column: 3, scope: !38)
// CHECK:STDOUT: !49 = !DILocation(line: 20, column: 16, scope: !38)
// CHECK:STDOUT: !51 = !{null, !23}
// CHECK:STDOUT: !52 = !DISubroutineType(types: !51)
// CHECK:STDOUT: !53 = !{null, !22}
// CHECK:STDOUT: !54 = !DISubroutineType(types: !53)
// CHECK:STDOUT: !55 = !DILocation(line: 16, column: 1, scope: !38)
// CHECK:STDOUT: !56 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.a9c4ae3b4c910d5b", scope: null, file: !14, line: 30, type: !52, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !62)
// CHECK:STDOUT: !57 = !DILocalVariable(arg: 1, scope: !56, type: !23)
// CHECK:STDOUT: !61 = !DILocation(line: 30, column: 1, scope: !56)
// CHECK:STDOUT: !62 = !{!57}
// CHECK:STDOUT: !63 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.dc5658e0ac74b2aa", scope: null, file: !14, line: 30, type: !54, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !69)
// CHECK:STDOUT: !64 = !DILocalVariable(arg: 1, scope: !63, type: !22)
// CHECK:STDOUT: !68 = !DILocation(line: 30, column: 1, scope: !63)
// CHECK:STDOUT: !69 = !{!64}
// CHECK:STDOUT: !70 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.eda669d623214d41", scope: null, file: !14, line: 30, type: !52, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !76)
// CHECK:STDOUT: !71 = !DILocalVariable(arg: 1, scope: !70, type: !23)
// CHECK:STDOUT: !75 = !DILocation(line: 30, column: 1, scope: !70)
// CHECK:STDOUT: !76 = !{!71}
// CHECK:STDOUT:
