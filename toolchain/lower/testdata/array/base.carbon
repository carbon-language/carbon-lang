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
// CHECK:STDOUT: @array.e85.loc14_3.1 = internal constant [1 x i32] [i32 1]
// CHECK:STDOUT: @array.c57.loc15_3.1 = internal constant [2 x double] [double 1.110000e+01, double 2.200000e+00]
// CHECK:STDOUT: @array.432.loc16_3.1 = internal constant [5 x {}] zeroinitializer
// CHECK:STDOUT: @tuple.d9f.loc17_3.1 = internal constant { i32, i32, i32 } { i32 1, i32 2, i32 3 }
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
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %_.var.loc14, ptr align 4 @array.e85.loc14_3.1, i64 4, i1 false), !dbg !43
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc15), !dbg !46
// CHECK:STDOUT:   %.loc15_37.3.array.index = getelementptr inbounds [2 x double], ptr %_.var.loc15, i32 0, i64 0, !dbg !47
// CHECK:STDOUT:   %.loc15_37.6.array.index = getelementptr inbounds [2 x double], ptr %_.var.loc15, i32 0, i64 1, !dbg !47
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %_.var.loc15, ptr align 8 @array.c57.loc15_3.1, i64 16, i1 false), !dbg !46
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc16), !dbg !49
// CHECK:STDOUT:   %.loc16_45.2.array.index = getelementptr inbounds [5 x {}], ptr %_.var.loc16, i32 0, i64 0, !dbg !50
// CHECK:STDOUT:   %.loc16_45.5.array.index = getelementptr inbounds [5 x {}], ptr %_.var.loc16, i32 0, i64 1, !dbg !50
// CHECK:STDOUT:   %.loc16_45.8.array.index = getelementptr inbounds [5 x {}], ptr %_.var.loc16, i32 0, i64 2, !dbg !50
// CHECK:STDOUT:   %.loc16_45.11.array.index = getelementptr inbounds [5 x {}], ptr %_.var.loc16, i32 0, i64 3, !dbg !50
// CHECK:STDOUT:   %.loc16_45.14.array.index = getelementptr inbounds [5 x {}], ptr %_.var.loc16, i32 0, i64 4, !dbg !50
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %_.var.loc16, ptr align 1 @array.432.loc16_3.1, i64 0, i1 false), !dbg !49
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %d.var), !dbg !52
// CHECK:STDOUT:   %tuple.elem0.loc17.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %d.var, i32 0, i32 0, !dbg !53
// CHECK:STDOUT:   %tuple.elem1.loc17.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %d.var, i32 0, i32 1, !dbg !53
// CHECK:STDOUT:   %tuple.elem2.loc17.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %d.var, i32 0, i32 2, !dbg !53
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %d.var, ptr align 4 @tuple.d9f.loc17_3.1, i64 12, i1 false), !dbg !52
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
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.c7bfed00bf5d7cb1(ptr %_.var.loc18), !dbg !55
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.e007740e1faeeffa(ptr %d.var), !dbg !52
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.cd22484ee08e869b(ptr %_.var.loc16), !dbg !49
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.22a0077550b33033(ptr %_.var.loc15), !dbg !46
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.672f5301749a1dff(ptr %_.var.loc14), !dbg !43
// CHECK:STDOUT:   ret i32 0, !dbg !58
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.022f2f6eb6531e95:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !59 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc18_3.1.array.index = getelementptr inbounds [3 x i32], ptr %self, i32 0, i64 2, !dbg !61
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.7f8dce37f94a83bf(ptr %.loc18_3.1.array.index), !dbg !61
// CHECK:STDOUT:   %.loc18_3.4.array.index = getelementptr inbounds [3 x i32], ptr %self, i32 0, i64 1, !dbg !61
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.7f8dce37f94a83bf(ptr %.loc18_3.4.array.index), !dbg !61
// CHECK:STDOUT:   %.loc18_3.7.array.index = getelementptr inbounds [3 x i32], ptr %self, i32 0, i64 0, !dbg !61
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.7f8dce37f94a83bf(ptr %.loc18_3.7.array.index), !dbg !61
// CHECK:STDOUT:   ret void, !dbg !61
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.9f53a29b4522c35d:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !65 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %tuple.elem2.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %self, i32 0, i32 2, !dbg !67
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.7f8dce37f94a83bf(ptr %tuple.elem2.tuple.elem), !dbg !67
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %self, i32 0, i32 1, !dbg !67
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.7f8dce37f94a83bf(ptr %tuple.elem1.tuple.elem), !dbg !67
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %self, i32 0, i32 0, !dbg !67
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.7f8dce37f94a83bf(ptr %tuple.elem0.tuple.elem), !dbg !67
// CHECK:STDOUT:   ret void, !dbg !67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.4649e4dcde68346e:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !69 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc16_3.1.array.index = getelementptr inbounds [5 x {}], ptr %self, i32 0, i64 4, !dbg !71
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.cd6aa98e8d0958d3(ptr %.loc16_3.1.array.index), !dbg !71
// CHECK:STDOUT:   %.loc16_3.4.array.index = getelementptr inbounds [5 x {}], ptr %self, i32 0, i64 3, !dbg !71
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.cd6aa98e8d0958d3(ptr %.loc16_3.4.array.index), !dbg !71
// CHECK:STDOUT:   %.loc16_3.7.array.index = getelementptr inbounds [5 x {}], ptr %self, i32 0, i64 2, !dbg !71
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.cd6aa98e8d0958d3(ptr %.loc16_3.7.array.index), !dbg !71
// CHECK:STDOUT:   %.loc16_3.10.array.index = getelementptr inbounds [5 x {}], ptr %self, i32 0, i64 1, !dbg !71
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.cd6aa98e8d0958d3(ptr %.loc16_3.10.array.index), !dbg !71
// CHECK:STDOUT:   %.loc16_3.13.array.index = getelementptr inbounds [5 x {}], ptr %self, i32 0, i64 0, !dbg !71
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.cd6aa98e8d0958d3(ptr %.loc16_3.13.array.index), !dbg !71
// CHECK:STDOUT:   ret void, !dbg !71
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.2fd803684ed7f369:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !73 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc15_3.1.array.index = getelementptr inbounds [2 x double], ptr %self, i32 0, i64 1, !dbg !75
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.236ebccb4c02fafd(ptr %.loc15_3.1.array.index), !dbg !75
// CHECK:STDOUT:   %.loc15_3.4.array.index = getelementptr inbounds [2 x double], ptr %self, i32 0, i64 0, !dbg !75
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.236ebccb4c02fafd(ptr %.loc15_3.4.array.index), !dbg !75
// CHECK:STDOUT:   ret void, !dbg !75
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.270c99ab48ee2493:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !77 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc14_3.1.array.index = getelementptr inbounds [1 x i32], ptr %self, i32 0, i64 0, !dbg !79
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.7f8dce37f94a83bf(ptr %.loc14_3.1.array.index), !dbg !79
// CHECK:STDOUT:   ret void, !dbg !79
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.c7bfed00bf5d7cb1(ptr %t) #0 !dbg !81 {
// CHECK:STDOUT:   call void @"_COp.022f2f6eb6531e95:core.SubobjectDestroy.Core"(ptr %t), !dbg !85
// CHECK:STDOUT:   ret void, !dbg !86
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.e007740e1faeeffa(ptr %t) #0 !dbg !88 {
// CHECK:STDOUT:   call void @"_COp.9f53a29b4522c35d:core.SubobjectDestroy.Core"(ptr %t), !dbg !92
// CHECK:STDOUT:   ret void, !dbg !93
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.cd22484ee08e869b(ptr %t) #0 !dbg !95 {
// CHECK:STDOUT:   call void @"_COp.4649e4dcde68346e:core.SubobjectDestroy.Core"(ptr %t), !dbg !99
// CHECK:STDOUT:   ret void, !dbg !100
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.22a0077550b33033(ptr %t) #0 !dbg !102 {
// CHECK:STDOUT:   call void @"_COp.2fd803684ed7f369:core.SubobjectDestroy.Core"(ptr %t), !dbg !106
// CHECK:STDOUT:   ret void, !dbg !107
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.672f5301749a1dff(ptr %t) #0 !dbg !109 {
// CHECK:STDOUT:   call void @"_COp.270c99ab48ee2493:core.SubobjectDestroy.Core"(ptr %t), !dbg !113
// CHECK:STDOUT:   ret void, !dbg !114
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.7f8dce37f94a83bf(ptr %t) #0 !dbg !116 {
// CHECK:STDOUT:   ret void, !dbg !121
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.cd6aa98e8d0958d3(ptr %t) #0 !dbg !123 {
// CHECK:STDOUT:   ret void, !dbg !128
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.236ebccb4c02fafd(ptr %t) #0 !dbg !130 {
// CHECK:STDOUT:   ret void, !dbg !135
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.7f8dce37f94a83bf, { 6, 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.cd6aa98e8d0958d3, { 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.236ebccb4c02fafd, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!35}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !28 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !34 = !DIFile(filename: "base.carbon", directory: "")
// CHECK:STDOUT: !35 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !34, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !36 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !37 = !{!36}
// CHECK:STDOUT: !38 = !DISubroutineType(types: !37)
// CHECK:STDOUT: !39 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !40 = !{null, !39}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !34, line: 13, type: !38, spFlags: DISPFlagDefinition, unit: !35)
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
// CHECK:STDOUT: !59 = distinct !DISubprogram(name: "Op", linkageName: "_COp.022f2f6eb6531e95:core.SubobjectDestroy.Core", scope: null, file: !34, line: 18, type: !41, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !64)
// CHECK:STDOUT: !60 = !DILocalVariable(arg: 1, scope: !59, type: !39)
// CHECK:STDOUT: !61 = !DILocation(line: 18, column: 3, scope: !59)
// CHECK:STDOUT: !62 = !{null, !36}
// CHECK:STDOUT: !63 = !DISubroutineType(types: !62)
// CHECK:STDOUT: !64 = !{!60}
// CHECK:STDOUT: !65 = distinct !DISubprogram(name: "Op", linkageName: "_COp.9f53a29b4522c35d:core.SubobjectDestroy.Core", scope: null, file: !34, line: 17, type: !41, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !68)
// CHECK:STDOUT: !66 = !DILocalVariable(arg: 1, scope: !65, type: !39)
// CHECK:STDOUT: !67 = !DILocation(line: 17, column: 3, scope: !65)
// CHECK:STDOUT: !68 = !{!66}
// CHECK:STDOUT: !69 = distinct !DISubprogram(name: "Op", linkageName: "_COp.4649e4dcde68346e:core.SubobjectDestroy.Core", scope: null, file: !34, line: 16, type: !41, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !72)
// CHECK:STDOUT: !70 = !DILocalVariable(arg: 1, scope: !69, type: !39)
// CHECK:STDOUT: !71 = !DILocation(line: 16, column: 3, scope: !69)
// CHECK:STDOUT: !72 = !{!70}
// CHECK:STDOUT: !73 = distinct !DISubprogram(name: "Op", linkageName: "_COp.2fd803684ed7f369:core.SubobjectDestroy.Core", scope: null, file: !34, line: 15, type: !41, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !76)
// CHECK:STDOUT: !74 = !DILocalVariable(arg: 1, scope: !73, type: !39)
// CHECK:STDOUT: !75 = !DILocation(line: 15, column: 3, scope: !73)
// CHECK:STDOUT: !76 = !{!74}
// CHECK:STDOUT: !77 = distinct !DISubprogram(name: "Op", linkageName: "_COp.270c99ab48ee2493:core.SubobjectDestroy.Core", scope: null, file: !34, line: 14, type: !41, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !80)
// CHECK:STDOUT: !78 = !DILocalVariable(arg: 1, scope: !77, type: !39)
// CHECK:STDOUT: !79 = !DILocation(line: 14, column: 3, scope: !77)
// CHECK:STDOUT: !80 = !{!78}
// CHECK:STDOUT: !81 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.c7bfed00bf5d7cb1", scope: null, file: !28, line: 30, type: !41, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !87)
// CHECK:STDOUT: !82 = !DILocalVariable(arg: 1, scope: !81, type: !39)
// CHECK:STDOUT: !85 = !DILocation(line: 32, column: 5, scope: !81)
// CHECK:STDOUT: !86 = !DILocation(line: 30, column: 1, scope: !81)
// CHECK:STDOUT: !87 = !{!82}
// CHECK:STDOUT: !88 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.e007740e1faeeffa", scope: null, file: !28, line: 30, type: !41, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !94)
// CHECK:STDOUT: !89 = !DILocalVariable(arg: 1, scope: !88, type: !39)
// CHECK:STDOUT: !92 = !DILocation(line: 32, column: 5, scope: !88)
// CHECK:STDOUT: !93 = !DILocation(line: 30, column: 1, scope: !88)
// CHECK:STDOUT: !94 = !{!89}
// CHECK:STDOUT: !95 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.cd22484ee08e869b", scope: null, file: !28, line: 30, type: !41, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !101)
// CHECK:STDOUT: !96 = !DILocalVariable(arg: 1, scope: !95, type: !39)
// CHECK:STDOUT: !99 = !DILocation(line: 32, column: 5, scope: !95)
// CHECK:STDOUT: !100 = !DILocation(line: 30, column: 1, scope: !95)
// CHECK:STDOUT: !101 = !{!96}
// CHECK:STDOUT: !102 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.22a0077550b33033", scope: null, file: !28, line: 30, type: !41, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !108)
// CHECK:STDOUT: !103 = !DILocalVariable(arg: 1, scope: !102, type: !39)
// CHECK:STDOUT: !106 = !DILocation(line: 32, column: 5, scope: !102)
// CHECK:STDOUT: !107 = !DILocation(line: 30, column: 1, scope: !102)
// CHECK:STDOUT: !108 = !{!103}
// CHECK:STDOUT: !109 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.672f5301749a1dff", scope: null, file: !28, line: 30, type: !41, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !115)
// CHECK:STDOUT: !110 = !DILocalVariable(arg: 1, scope: !109, type: !39)
// CHECK:STDOUT: !113 = !DILocation(line: 32, column: 5, scope: !109)
// CHECK:STDOUT: !114 = !DILocation(line: 30, column: 1, scope: !109)
// CHECK:STDOUT: !115 = !{!110}
// CHECK:STDOUT: !116 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.7f8dce37f94a83bf", scope: null, file: !28, line: 30, type: !63, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !122)
// CHECK:STDOUT: !117 = !DILocalVariable(arg: 1, scope: !116, type: !36)
// CHECK:STDOUT: !121 = !DILocation(line: 30, column: 1, scope: !116)
// CHECK:STDOUT: !122 = !{!117}
// CHECK:STDOUT: !123 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.cd6aa98e8d0958d3", scope: null, file: !28, line: 30, type: !41, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !129)
// CHECK:STDOUT: !124 = !DILocalVariable(arg: 1, scope: !123, type: !39)
// CHECK:STDOUT: !128 = !DILocation(line: 30, column: 1, scope: !123)
// CHECK:STDOUT: !129 = !{!124}
// CHECK:STDOUT: !130 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.236ebccb4c02fafd", scope: null, file: !28, line: 30, type: !41, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !136)
// CHECK:STDOUT: !131 = !DILocalVariable(arg: 1, scope: !130, type: !39)
// CHECK:STDOUT: !135 = !DILocation(line: 30, column: 1, scope: !130)
// CHECK:STDOUT: !136 = !{!131}
// CHECK:STDOUT:
