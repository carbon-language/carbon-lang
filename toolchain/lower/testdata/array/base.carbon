// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/array/base.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/array/base.carbon

fn Run() {
  var _: array(i32, 1) = (1,);
  var _: array(f64, 2) = (11.1, 2.2,);
  var _: array((), 5) = ((), (), (), (), (),);
  var d: (i32, i32, i32) = (1, 2, 3);
  var _: array(i32, 3) = d;
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'base.carbon'
// CHECK:STDOUT: source_filename = "base.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @array.5e9.loc14_3 = internal constant [1 x i32] [i32 1]
// CHECK:STDOUT: @array.33e.loc15_3 = internal constant [2 x double] [double 1.110000e+01, double 2.200000e+00]
// CHECK:STDOUT: @array.b6f.loc16_3 = internal constant [5 x {}] zeroinitializer
// CHECK:STDOUT: @tuple.d77.loc17_3 = internal constant { i32, i32, i32 } { i32 1, i32 2, i32 3 }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @main() #0 !dbg !44 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var.loc14 = alloca [1 x i32], align 4, !dbg !45
// CHECK:STDOUT:   %_.var.loc15 = alloca [2 x double], align 8, !dbg !48
// CHECK:STDOUT:   %_.var.loc16 = alloca [5 x {}], align 1, !dbg !51
// CHECK:STDOUT:   %d.var = alloca { i32, i32, i32 }, align 4, !dbg !54
// CHECK:STDOUT:   %_.var.loc18 = alloca [3 x i32], align 4, !dbg !57
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc14), !dbg !45
// CHECK:STDOUT:   %.loc14_29.3.array.index = getelementptr inbounds [1 x i32], ptr %_.var.loc14, i32 0, i64 0, !dbg !46
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %_.var.loc14, ptr align 4 @array.5e9.loc14_3, i64 4, i1 false), !dbg !45
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc15), !dbg !48
// CHECK:STDOUT:   %.loc15_37.3.array.index = getelementptr inbounds [2 x double], ptr %_.var.loc15, i32 0, i64 0, !dbg !49
// CHECK:STDOUT:   %.loc15_37.6.array.index = getelementptr inbounds [2 x double], ptr %_.var.loc15, i32 0, i64 1, !dbg !49
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %_.var.loc15, ptr align 8 @array.33e.loc15_3, i64 16, i1 false), !dbg !48
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc16), !dbg !51
// CHECK:STDOUT:   %.loc16_45.2.array.index = getelementptr inbounds [5 x {}], ptr %_.var.loc16, i32 0, i64 0, !dbg !52
// CHECK:STDOUT:   %.loc16_45.5.array.index = getelementptr inbounds [5 x {}], ptr %_.var.loc16, i32 0, i64 1, !dbg !52
// CHECK:STDOUT:   %.loc16_45.8.array.index = getelementptr inbounds [5 x {}], ptr %_.var.loc16, i32 0, i64 2, !dbg !52
// CHECK:STDOUT:   %.loc16_45.11.array.index = getelementptr inbounds [5 x {}], ptr %_.var.loc16, i32 0, i64 3, !dbg !52
// CHECK:STDOUT:   %.loc16_45.14.array.index = getelementptr inbounds [5 x {}], ptr %_.var.loc16, i32 0, i64 4, !dbg !52
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %_.var.loc16, ptr align 1 @array.b6f.loc16_3, i64 0, i1 false), !dbg !51
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %d.var), !dbg !54
// CHECK:STDOUT:   %tuple.elem0.loc17.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %d.var, i32 0, i32 0, !dbg !55
// CHECK:STDOUT:   %tuple.elem1.loc17.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %d.var, i32 0, i32 1, !dbg !55
// CHECK:STDOUT:   %tuple.elem2.loc17.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %d.var, i32 0, i32 2, !dbg !55
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %d.var, ptr align 4 @tuple.d77.loc17_3, i64 12, i1 false), !dbg !54
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc18), !dbg !57
// CHECK:STDOUT:   %tuple.elem0.loc18.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %d.var, i32 0, i32 0, !dbg !58
// CHECK:STDOUT:   %.loc18_26.1 = load i32, ptr %tuple.elem0.loc18.tuple.elem, align 4, !dbg !58
// CHECK:STDOUT:   %.loc18_26.2.array.index = getelementptr inbounds [3 x i32], ptr %_.var.loc18, i32 0, i64 0, !dbg !58
// CHECK:STDOUT:   store i32 %.loc18_26.1, ptr %.loc18_26.2.array.index, align 4, !dbg !58
// CHECK:STDOUT:   %tuple.elem1.loc18.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %d.var, i32 0, i32 1, !dbg !58
// CHECK:STDOUT:   %.loc18_26.4 = load i32, ptr %tuple.elem1.loc18.tuple.elem, align 4, !dbg !58
// CHECK:STDOUT:   %.loc18_26.5.array.index = getelementptr inbounds [3 x i32], ptr %_.var.loc18, i32 0, i64 1, !dbg !58
// CHECK:STDOUT:   store i32 %.loc18_26.4, ptr %.loc18_26.5.array.index, align 4, !dbg !58
// CHECK:STDOUT:   %tuple.elem2.loc18.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %d.var, i32 0, i32 2, !dbg !58
// CHECK:STDOUT:   %.loc18_26.7 = load i32, ptr %tuple.elem2.loc18.tuple.elem, align 4, !dbg !58
// CHECK:STDOUT:   %.loc18_26.8.array.index = getelementptr inbounds [3 x i32], ptr %_.var.loc18, i32 0, i64 2, !dbg !58
// CHECK:STDOUT:   store i32 %.loc18_26.7, ptr %.loc18_26.8.array.index, align 4, !dbg !58
// CHECK:STDOUT:   call void @"_CSelfDestruct.3c49d12388450843:core.Destroy.Core"(ptr %_.var.loc18), !dbg !57
// CHECK:STDOUT:   call void @"_CSelfDestruct.3ff83804c0fae8cb:core.Destroy.Core"(ptr %d.var), !dbg !54
// CHECK:STDOUT:   call void @"_CSelfDestruct.040a50b2092a2f94:core.Destroy.Core"(ptr %_.var.loc16), !dbg !51
// CHECK:STDOUT:   call void @"_CSelfDestruct.f24946731f398dd1:core.Destroy.Core"(ptr %_.var.loc15), !dbg !48
// CHECK:STDOUT:   call void @"_CSelfDestruct.b0e7c4b991a889bf:core.Destroy.Core"(ptr %_.var.loc14), !dbg !45
// CHECK:STDOUT:   ret i32 0, !dbg !60
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.76db170ab64c05aa:core.Destroy.Core"(ptr %self) #0 !dbg !61 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !63
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.3c49d12388450843:core.Destroy.Core"(ptr %self) #0 !dbg !65 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc18_3.1.array.index = getelementptr inbounds [3 x i32], ptr %self, i32 0, i64 2, !dbg !67
// CHECK:STDOUT:   call void @"_CSelfDestruct.76db170ab64c05aa:core.Destroy.Core"(ptr %.loc18_3.1.array.index), !dbg !67
// CHECK:STDOUT:   %.loc18_3.2.array.index = getelementptr inbounds [3 x i32], ptr %self, i32 0, i64 1, !dbg !67
// CHECK:STDOUT:   call void @"_CSelfDestruct.76db170ab64c05aa:core.Destroy.Core"(ptr %.loc18_3.2.array.index), !dbg !67
// CHECK:STDOUT:   %.loc18_3.3.array.index = getelementptr inbounds [3 x i32], ptr %self, i32 0, i64 0, !dbg !67
// CHECK:STDOUT:   call void @"_CSelfDestruct.76db170ab64c05aa:core.Destroy.Core"(ptr %.loc18_3.3.array.index), !dbg !67
// CHECK:STDOUT:   ret void, !dbg !67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.3c49d12388450843:core.Destroy.Core"(ptr %self) #0 !dbg !69 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.3c49d12388450843:core.Destroy.Core"(ptr %self), !dbg !71
// CHECK:STDOUT:   ret void, !dbg !71
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.3ff83804c0fae8cb:core.Destroy.Core"(ptr %self) #0 !dbg !73 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %tuple.elem2.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %self, i32 0, i32 2, !dbg !75
// CHECK:STDOUT:   call void @"_CSelfDestruct.76db170ab64c05aa:core.Destroy.Core"(ptr %tuple.elem2.tuple.elem), !dbg !75
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %self, i32 0, i32 1, !dbg !75
// CHECK:STDOUT:   call void @"_CSelfDestruct.76db170ab64c05aa:core.Destroy.Core"(ptr %tuple.elem1.tuple.elem), !dbg !75
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %self, i32 0, i32 0, !dbg !75
// CHECK:STDOUT:   call void @"_CSelfDestruct.76db170ab64c05aa:core.Destroy.Core"(ptr %tuple.elem0.tuple.elem), !dbg !75
// CHECK:STDOUT:   ret void, !dbg !75
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.3ff83804c0fae8cb:core.Destroy.Core"(ptr %self) #0 !dbg !77 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.3ff83804c0fae8cb:core.Destroy.Core"(ptr %self), !dbg !79
// CHECK:STDOUT:   ret void, !dbg !79
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.61ea2aba74ab3bf1:core.Destroy.Core"(ptr %self) #0 !dbg !81 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !83
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.040a50b2092a2f94:core.Destroy.Core"(ptr %self) #0 !dbg !85 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc16_3.1.array.index = getelementptr inbounds [5 x {}], ptr %self, i32 0, i64 4, !dbg !87
// CHECK:STDOUT:   call void @"_CSelfDestruct.61ea2aba74ab3bf1:core.Destroy.Core"(ptr %.loc16_3.1.array.index), !dbg !87
// CHECK:STDOUT:   %.loc16_3.2.array.index = getelementptr inbounds [5 x {}], ptr %self, i32 0, i64 3, !dbg !87
// CHECK:STDOUT:   call void @"_CSelfDestruct.61ea2aba74ab3bf1:core.Destroy.Core"(ptr %.loc16_3.2.array.index), !dbg !87
// CHECK:STDOUT:   %.loc16_3.3.array.index = getelementptr inbounds [5 x {}], ptr %self, i32 0, i64 2, !dbg !87
// CHECK:STDOUT:   call void @"_CSelfDestruct.61ea2aba74ab3bf1:core.Destroy.Core"(ptr %.loc16_3.3.array.index), !dbg !87
// CHECK:STDOUT:   %.loc16_3.4.array.index = getelementptr inbounds [5 x {}], ptr %self, i32 0, i64 1, !dbg !87
// CHECK:STDOUT:   call void @"_CSelfDestruct.61ea2aba74ab3bf1:core.Destroy.Core"(ptr %.loc16_3.4.array.index), !dbg !87
// CHECK:STDOUT:   %.loc16_3.5.array.index = getelementptr inbounds [5 x {}], ptr %self, i32 0, i64 0, !dbg !87
// CHECK:STDOUT:   call void @"_CSelfDestruct.61ea2aba74ab3bf1:core.Destroy.Core"(ptr %.loc16_3.5.array.index), !dbg !87
// CHECK:STDOUT:   ret void, !dbg !87
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.040a50b2092a2f94:core.Destroy.Core"(ptr %self) #0 !dbg !89 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.040a50b2092a2f94:core.Destroy.Core"(ptr %self), !dbg !91
// CHECK:STDOUT:   ret void, !dbg !91
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.84c40a423a4f9e08:core.Destroy.Core"(ptr %self) #0 !dbg !93 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !95
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f24946731f398dd1:core.Destroy.Core"(ptr %self) #0 !dbg !97 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc15_3.1.array.index = getelementptr inbounds [2 x double], ptr %self, i32 0, i64 1, !dbg !99
// CHECK:STDOUT:   call void @"_CSelfDestruct.84c40a423a4f9e08:core.Destroy.Core"(ptr %.loc15_3.1.array.index), !dbg !99
// CHECK:STDOUT:   %.loc15_3.2.array.index = getelementptr inbounds [2 x double], ptr %self, i32 0, i64 0, !dbg !99
// CHECK:STDOUT:   call void @"_CSelfDestruct.84c40a423a4f9e08:core.Destroy.Core"(ptr %.loc15_3.2.array.index), !dbg !99
// CHECK:STDOUT:   ret void, !dbg !99
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f24946731f398dd1:core.Destroy.Core"(ptr %self) #0 !dbg !101 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.f24946731f398dd1:core.Destroy.Core"(ptr %self), !dbg !103
// CHECK:STDOUT:   ret void, !dbg !103
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.b0e7c4b991a889bf:core.Destroy.Core"(ptr %self) #0 !dbg !105 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc14.array.index = getelementptr inbounds [1 x i32], ptr %self, i32 0, i64 0, !dbg !107
// CHECK:STDOUT:   call void @"_CSelfDestruct.76db170ab64c05aa:core.Destroy.Core"(ptr %.loc14.array.index), !dbg !107
// CHECK:STDOUT:   ret void, !dbg !107
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.b0e7c4b991a889bf:core.Destroy.Core"(ptr %self) #0 !dbg !109 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.b0e7c4b991a889bf:core.Destroy.Core"(ptr %self), !dbg !111
// CHECK:STDOUT:   ret void, !dbg !111
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!35}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !34 = !DIFile(filename: "base.carbon", directory: "")
// CHECK:STDOUT: !35 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !34, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !36 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !37 = !{!36}
// CHECK:STDOUT: !38 = !DISubroutineType(types: !37)
// CHECK:STDOUT: !39 = !{null, !36}
// CHECK:STDOUT: !40 = !DISubroutineType(types: !39)
// CHECK:STDOUT: !41 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !42 = !{null, !41}
// CHECK:STDOUT: !43 = !DISubroutineType(types: !42)
// CHECK:STDOUT: !44 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !34, line: 13, type: !38, spFlags: DISPFlagDefinition, unit: !35)
// CHECK:STDOUT: !45 = !DILocation(line: 14, column: 3, scope: !44)
// CHECK:STDOUT: !46 = !DILocation(line: 14, column: 26, scope: !44)
// CHECK:STDOUT: !48 = !DILocation(line: 15, column: 3, scope: !44)
// CHECK:STDOUT: !49 = !DILocation(line: 15, column: 26, scope: !44)
// CHECK:STDOUT: !51 = !DILocation(line: 16, column: 3, scope: !44)
// CHECK:STDOUT: !52 = !DILocation(line: 16, column: 25, scope: !44)
// CHECK:STDOUT: !54 = !DILocation(line: 17, column: 3, scope: !44)
// CHECK:STDOUT: !55 = !DILocation(line: 17, column: 28, scope: !44)
// CHECK:STDOUT: !57 = !DILocation(line: 18, column: 3, scope: !44)
// CHECK:STDOUT: !58 = !DILocation(line: 18, column: 26, scope: !44)
// CHECK:STDOUT: !60 = !DILocation(line: 13, column: 1, scope: !44)
// CHECK:STDOUT: !61 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.76db170ab64c05aa:core.Destroy.Core", scope: null, file: !34, line: 18, type: !40, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !64)
// CHECK:STDOUT: !62 = !DILocalVariable(arg: 1, scope: !61, type: !36)
// CHECK:STDOUT: !63 = !DILocation(line: 18, column: 3, scope: !61)
// CHECK:STDOUT: !64 = !{!62}
// CHECK:STDOUT: !65 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.3c49d12388450843:core.Destroy.Core", scope: null, file: !34, line: 18, type: !43, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !68)
// CHECK:STDOUT: !66 = !DILocalVariable(arg: 1, scope: !65, type: !41)
// CHECK:STDOUT: !67 = !DILocation(line: 18, column: 3, scope: !65)
// CHECK:STDOUT: !68 = !{!66}
// CHECK:STDOUT: !69 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.3c49d12388450843:core.Destroy.Core", scope: null, file: !34, line: 18, type: !43, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !72)
// CHECK:STDOUT: !70 = !DILocalVariable(arg: 1, scope: !69, type: !41)
// CHECK:STDOUT: !71 = !DILocation(line: 18, column: 3, scope: !69)
// CHECK:STDOUT: !72 = !{!70}
// CHECK:STDOUT: !73 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.3ff83804c0fae8cb:core.Destroy.Core", scope: null, file: !34, line: 17, type: !43, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !76)
// CHECK:STDOUT: !74 = !DILocalVariable(arg: 1, scope: !73, type: !41)
// CHECK:STDOUT: !75 = !DILocation(line: 17, column: 3, scope: !73)
// CHECK:STDOUT: !76 = !{!74}
// CHECK:STDOUT: !77 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.3ff83804c0fae8cb:core.Destroy.Core", scope: null, file: !34, line: 17, type: !43, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !80)
// CHECK:STDOUT: !78 = !DILocalVariable(arg: 1, scope: !77, type: !41)
// CHECK:STDOUT: !79 = !DILocation(line: 17, column: 3, scope: !77)
// CHECK:STDOUT: !80 = !{!78}
// CHECK:STDOUT: !81 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.61ea2aba74ab3bf1:core.Destroy.Core", scope: null, file: !34, line: 16, type: !43, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !84)
// CHECK:STDOUT: !82 = !DILocalVariable(arg: 1, scope: !81, type: !41)
// CHECK:STDOUT: !83 = !DILocation(line: 16, column: 3, scope: !81)
// CHECK:STDOUT: !84 = !{!82}
// CHECK:STDOUT: !85 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.040a50b2092a2f94:core.Destroy.Core", scope: null, file: !34, line: 16, type: !43, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !88)
// CHECK:STDOUT: !86 = !DILocalVariable(arg: 1, scope: !85, type: !41)
// CHECK:STDOUT: !87 = !DILocation(line: 16, column: 3, scope: !85)
// CHECK:STDOUT: !88 = !{!86}
// CHECK:STDOUT: !89 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.040a50b2092a2f94:core.Destroy.Core", scope: null, file: !34, line: 16, type: !43, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !92)
// CHECK:STDOUT: !90 = !DILocalVariable(arg: 1, scope: !89, type: !41)
// CHECK:STDOUT: !91 = !DILocation(line: 16, column: 3, scope: !89)
// CHECK:STDOUT: !92 = !{!90}
// CHECK:STDOUT: !93 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.84c40a423a4f9e08:core.Destroy.Core", scope: null, file: !34, line: 15, type: !43, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !96)
// CHECK:STDOUT: !94 = !DILocalVariable(arg: 1, scope: !93, type: !41)
// CHECK:STDOUT: !95 = !DILocation(line: 15, column: 3, scope: !93)
// CHECK:STDOUT: !96 = !{!94}
// CHECK:STDOUT: !97 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f24946731f398dd1:core.Destroy.Core", scope: null, file: !34, line: 15, type: !43, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !100)
// CHECK:STDOUT: !98 = !DILocalVariable(arg: 1, scope: !97, type: !41)
// CHECK:STDOUT: !99 = !DILocation(line: 15, column: 3, scope: !97)
// CHECK:STDOUT: !100 = !{!98}
// CHECK:STDOUT: !101 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f24946731f398dd1:core.Destroy.Core", scope: null, file: !34, line: 15, type: !43, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !104)
// CHECK:STDOUT: !102 = !DILocalVariable(arg: 1, scope: !101, type: !41)
// CHECK:STDOUT: !103 = !DILocation(line: 15, column: 3, scope: !101)
// CHECK:STDOUT: !104 = !{!102}
// CHECK:STDOUT: !105 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.b0e7c4b991a889bf:core.Destroy.Core", scope: null, file: !34, line: 14, type: !43, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !108)
// CHECK:STDOUT: !106 = !DILocalVariable(arg: 1, scope: !105, type: !41)
// CHECK:STDOUT: !107 = !DILocation(line: 14, column: 3, scope: !105)
// CHECK:STDOUT: !108 = !{!106}
// CHECK:STDOUT: !109 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.b0e7c4b991a889bf:core.Destroy.Core", scope: null, file: !34, line: 14, type: !43, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !112)
// CHECK:STDOUT: !110 = !DILocalVariable(arg: 1, scope: !109, type: !41)
// CHECK:STDOUT: !111 = !DILocation(line: 14, column: 3, scope: !109)
// CHECK:STDOUT: !112 = !{!110}
// CHECK:STDOUT:
