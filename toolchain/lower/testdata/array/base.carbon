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
// CHECK:STDOUT: @array.a00.loc14_3.1 = internal constant [1 x i32] [i32 1]
// CHECK:STDOUT: @array.541.loc15_3.1 = internal constant [2 x double] [double 1.110000e+01, double 2.200000e+00]
// CHECK:STDOUT: @array.1cb.loc16_3.1 = internal constant [5 x {}] zeroinitializer
// CHECK:STDOUT: @tuple.479.loc17_3.1 = internal constant { i32, i32, i32 } { i32 1, i32 2, i32 3 }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @main() #0 !dbg !42 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var.loc14 = alloca [1 x i32], align 4, !dbg !43
// CHECK:STDOUT:   %_.var.loc15 = alloca [2 x double], align 8, !dbg !46
// CHECK:STDOUT:   %_.var.loc16 = alloca [5 x {}], align 1, !dbg !49
// CHECK:STDOUT:   %d.var = alloca { i32, i32, i32 }, align 4, !dbg !52
// CHECK:STDOUT:   %_.var.loc18 = alloca [3 x i32], align 4, !dbg !55
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc14), !dbg !43
// CHECK:STDOUT:   %.loc14_29.3.array.index = getelementptr inbounds [1 x i32], ptr %_.var.loc14, i32 0, i64 0, !dbg !44
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %_.var.loc14, ptr align 4 @array.a00.loc14_3.1, i64 4, i1 false), !dbg !43
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc15), !dbg !46
// CHECK:STDOUT:   %.loc15_37.3.array.index = getelementptr inbounds [2 x double], ptr %_.var.loc15, i32 0, i64 0, !dbg !47
// CHECK:STDOUT:   %.loc15_37.6.array.index = getelementptr inbounds [2 x double], ptr %_.var.loc15, i32 0, i64 1, !dbg !47
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %_.var.loc15, ptr align 8 @array.541.loc15_3.1, i64 16, i1 false), !dbg !46
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc16), !dbg !49
// CHECK:STDOUT:   %.loc16_45.2.array.index = getelementptr inbounds [5 x {}], ptr %_.var.loc16, i32 0, i64 0, !dbg !50
// CHECK:STDOUT:   %.loc16_45.5.array.index = getelementptr inbounds [5 x {}], ptr %_.var.loc16, i32 0, i64 1, !dbg !50
// CHECK:STDOUT:   %.loc16_45.8.array.index = getelementptr inbounds [5 x {}], ptr %_.var.loc16, i32 0, i64 2, !dbg !50
// CHECK:STDOUT:   %.loc16_45.11.array.index = getelementptr inbounds [5 x {}], ptr %_.var.loc16, i32 0, i64 3, !dbg !50
// CHECK:STDOUT:   %.loc16_45.14.array.index = getelementptr inbounds [5 x {}], ptr %_.var.loc16, i32 0, i64 4, !dbg !50
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %_.var.loc16, ptr align 1 @array.1cb.loc16_3.1, i64 0, i1 false), !dbg !49
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %d.var), !dbg !52
// CHECK:STDOUT:   %tuple.elem0.loc17.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %d.var, i32 0, i32 0, !dbg !53
// CHECK:STDOUT:   %tuple.elem1.loc17.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %d.var, i32 0, i32 1, !dbg !53
// CHECK:STDOUT:   %tuple.elem2.loc17.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %d.var, i32 0, i32 2, !dbg !53
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %d.var, ptr align 4 @tuple.479.loc17_3.1, i64 12, i1 false), !dbg !52
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc18), !dbg !55
// CHECK:STDOUT:   %tuple.elem0.loc18.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %d.var, i32 0, i32 0, !dbg !56
// CHECK:STDOUT:   %.loc18_26.1 = load i32, ptr %tuple.elem0.loc18.tuple.elem, align 4, !dbg !56
// CHECK:STDOUT:   %.loc18_26.2.array.index = getelementptr inbounds [3 x i32], ptr %_.var.loc18, i32 0, i64 0, !dbg !56
// CHECK:STDOUT:   store i32 %.loc18_26.1, ptr %.loc18_26.2.array.index, align 4, !dbg !56
// CHECK:STDOUT:   %tuple.elem1.loc18.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %d.var, i32 0, i32 1, !dbg !56
// CHECK:STDOUT:   %.loc18_26.4 = load i32, ptr %tuple.elem1.loc18.tuple.elem, align 4, !dbg !56
// CHECK:STDOUT:   %.loc18_26.5.array.index = getelementptr inbounds [3 x i32], ptr %_.var.loc18, i32 0, i64 1, !dbg !56
// CHECK:STDOUT:   store i32 %.loc18_26.4, ptr %.loc18_26.5.array.index, align 4, !dbg !56
// CHECK:STDOUT:   %tuple.elem2.loc18.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %d.var, i32 0, i32 2, !dbg !56
// CHECK:STDOUT:   %.loc18_26.7 = load i32, ptr %tuple.elem2.loc18.tuple.elem, align 4, !dbg !56
// CHECK:STDOUT:   %.loc18_26.8.array.index = getelementptr inbounds [3 x i32], ptr %_.var.loc18, i32 0, i64 2, !dbg !56
// CHECK:STDOUT:   store i32 %.loc18_26.7, ptr %.loc18_26.8.array.index, align 4, !dbg !56
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.f699181ca81df557(ptr %_.var.loc18), !dbg !55
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.dd851822db46e4f5(ptr %d.var), !dbg !52
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.95146ba8dd075b9e(ptr %_.var.loc16), !dbg !49
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.f2d786893def15ae(ptr %_.var.loc15), !dbg !46
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.c0323949427aa835(ptr %_.var.loc14), !dbg !43
// CHECK:STDOUT:   ret i32 0, !dbg !58
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !59 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !61
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.dc7dedd104b837b2:core.Destroy.Core"(ptr %self) #0 !dbg !63 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !65
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.555b1314256372b8:core.Destroy.Core"(ptr %self) #0 !dbg !67 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !69
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.c135263f75f7214d:core.Destroy.Core"(ptr %self) #0 !dbg !71 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !73
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.dc14d4dd21801791:core.Destroy.Core"(ptr %self) #0 !dbg !75 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !77
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.8815565a27e3ec8c:core.Destroy.Core"(ptr %self) #0 !dbg !79 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !81
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.d277540f487baeed:core.Destroy.Core"(ptr %self) #0 !dbg !83 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !85
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.f699181ca81df557(ptr %self) #0 !dbg !87 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.dc7dedd104b837b2:core.Destroy.Core"(ptr %self), !dbg !91
// CHECK:STDOUT:   ret void, !dbg !92
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.dd851822db46e4f5(ptr %self) #0 !dbg !94 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.555b1314256372b8:core.Destroy.Core"(ptr %self), !dbg !98
// CHECK:STDOUT:   ret void, !dbg !99
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.95146ba8dd075b9e(ptr %self) #0 !dbg !101 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.c135263f75f7214d:core.Destroy.Core"(ptr %self), !dbg !105
// CHECK:STDOUT:   ret void, !dbg !106
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.f2d786893def15ae(ptr %self) #0 !dbg !108 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.8815565a27e3ec8c:core.Destroy.Core"(ptr %self), !dbg !112
// CHECK:STDOUT:   ret void, !dbg !113
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.c0323949427aa835(ptr %self) #0 !dbg !115 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.d277540f487baeed:core.Destroy.Core"(ptr %self), !dbg !119
// CHECK:STDOUT:   ret void, !dbg !120
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!33}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !28 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !32 = !DIFile(filename: "base.carbon", directory: "")
// CHECK:STDOUT: !33 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !32, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !34 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !35 = !{!34}
// CHECK:STDOUT: !36 = !DISubroutineType(types: !35)
// CHECK:STDOUT: !37 = !{null, !34}
// CHECK:STDOUT: !38 = !DISubroutineType(types: !37)
// CHECK:STDOUT: !39 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !40 = !{null, !39}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !32, line: 13, type: !36, spFlags: DISPFlagDefinition, unit: !33)
// CHECK:STDOUT: !43 = !DILocation(line: 14, column: 3, scope: !42)
// CHECK:STDOUT: !44 = !DILocation(line: 14, column: 26, scope: !42)
// CHECK:STDOUT: !46 = !DILocation(line: 15, column: 3, scope: !42)
// CHECK:STDOUT: !47 = !DILocation(line: 15, column: 26, scope: !42)
// CHECK:STDOUT: !49 = !DILocation(line: 16, column: 3, scope: !42)
// CHECK:STDOUT: !50 = !DILocation(line: 16, column: 25, scope: !42)
// CHECK:STDOUT: !52 = !DILocation(line: 17, column: 3, scope: !42)
// CHECK:STDOUT: !53 = !DILocation(line: 17, column: 28, scope: !42)
// CHECK:STDOUT: !55 = !DILocation(line: 18, column: 3, scope: !42)
// CHECK:STDOUT: !56 = !DILocation(line: 18, column: 26, scope: !42)
// CHECK:STDOUT: !58 = !DILocation(line: 13, column: 1, scope: !42)
// CHECK:STDOUT: !59 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core", scope: null, file: !32, line: 18, type: !38, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !62)
// CHECK:STDOUT: !60 = !DILocalVariable(arg: 1, scope: !59, type: !34)
// CHECK:STDOUT: !61 = !DILocation(line: 18, column: 3, scope: !59)
// CHECK:STDOUT: !62 = !{!60}
// CHECK:STDOUT: !63 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.dc7dedd104b837b2:core.Destroy.Core", scope: null, file: !32, line: 18, type: !41, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !66)
// CHECK:STDOUT: !64 = !DILocalVariable(arg: 1, scope: !63, type: !39)
// CHECK:STDOUT: !65 = !DILocation(line: 18, column: 3, scope: !63)
// CHECK:STDOUT: !66 = !{!64}
// CHECK:STDOUT: !67 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.555b1314256372b8:core.Destroy.Core", scope: null, file: !32, line: 17, type: !41, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !70)
// CHECK:STDOUT: !68 = !DILocalVariable(arg: 1, scope: !67, type: !39)
// CHECK:STDOUT: !69 = !DILocation(line: 17, column: 3, scope: !67)
// CHECK:STDOUT: !70 = !{!68}
// CHECK:STDOUT: !71 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.c135263f75f7214d:core.Destroy.Core", scope: null, file: !32, line: 16, type: !41, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !74)
// CHECK:STDOUT: !72 = !DILocalVariable(arg: 1, scope: !71, type: !39)
// CHECK:STDOUT: !73 = !DILocation(line: 16, column: 3, scope: !71)
// CHECK:STDOUT: !74 = !{!72}
// CHECK:STDOUT: !75 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.dc14d4dd21801791:core.Destroy.Core", scope: null, file: !32, line: 15, type: !41, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !78)
// CHECK:STDOUT: !76 = !DILocalVariable(arg: 1, scope: !75, type: !39)
// CHECK:STDOUT: !77 = !DILocation(line: 15, column: 3, scope: !75)
// CHECK:STDOUT: !78 = !{!76}
// CHECK:STDOUT: !79 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.8815565a27e3ec8c:core.Destroy.Core", scope: null, file: !32, line: 15, type: !41, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !82)
// CHECK:STDOUT: !80 = !DILocalVariable(arg: 1, scope: !79, type: !39)
// CHECK:STDOUT: !81 = !DILocation(line: 15, column: 3, scope: !79)
// CHECK:STDOUT: !82 = !{!80}
// CHECK:STDOUT: !83 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.d277540f487baeed:core.Destroy.Core", scope: null, file: !32, line: 14, type: !41, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !86)
// CHECK:STDOUT: !84 = !DILocalVariable(arg: 1, scope: !83, type: !39)
// CHECK:STDOUT: !85 = !DILocation(line: 14, column: 3, scope: !83)
// CHECK:STDOUT: !86 = !{!84}
// CHECK:STDOUT: !87 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.f699181ca81df557", scope: null, file: !28, line: 24, type: !41, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !93)
// CHECK:STDOUT: !88 = !DILocalVariable(arg: 1, scope: !87, type: !39)
// CHECK:STDOUT: !91 = !DILocation(line: 26, column: 5, scope: !87)
// CHECK:STDOUT: !92 = !DILocation(line: 24, column: 3, scope: !87)
// CHECK:STDOUT: !93 = !{!88}
// CHECK:STDOUT: !94 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.dd851822db46e4f5", scope: null, file: !28, line: 24, type: !41, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !100)
// CHECK:STDOUT: !95 = !DILocalVariable(arg: 1, scope: !94, type: !39)
// CHECK:STDOUT: !98 = !DILocation(line: 26, column: 5, scope: !94)
// CHECK:STDOUT: !99 = !DILocation(line: 24, column: 3, scope: !94)
// CHECK:STDOUT: !100 = !{!95}
// CHECK:STDOUT: !101 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.95146ba8dd075b9e", scope: null, file: !28, line: 24, type: !41, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !107)
// CHECK:STDOUT: !102 = !DILocalVariable(arg: 1, scope: !101, type: !39)
// CHECK:STDOUT: !105 = !DILocation(line: 26, column: 5, scope: !101)
// CHECK:STDOUT: !106 = !DILocation(line: 24, column: 3, scope: !101)
// CHECK:STDOUT: !107 = !{!102}
// CHECK:STDOUT: !108 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.f2d786893def15ae", scope: null, file: !28, line: 24, type: !41, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !114)
// CHECK:STDOUT: !109 = !DILocalVariable(arg: 1, scope: !108, type: !39)
// CHECK:STDOUT: !112 = !DILocation(line: 26, column: 5, scope: !108)
// CHECK:STDOUT: !113 = !DILocation(line: 24, column: 3, scope: !108)
// CHECK:STDOUT: !114 = !{!109}
// CHECK:STDOUT: !115 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.c0323949427aa835", scope: null, file: !28, line: 24, type: !41, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !121)
// CHECK:STDOUT: !116 = !DILocalVariable(arg: 1, scope: !115, type: !39)
// CHECK:STDOUT: !119 = !DILocation(line: 26, column: 5, scope: !115)
// CHECK:STDOUT: !120 = !DILocation(line: 24, column: 3, scope: !115)
// CHECK:STDOUT: !121 = !{!116}
// CHECK:STDOUT:
