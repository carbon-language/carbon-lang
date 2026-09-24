// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/let/tuple.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/let/tuple.carbon

fn F() -> i32 {
  var a: (i32, i32, i32) = (1, 2, 3);
  var b: (i32, i32) = (4, 5);
  let c: ((i32, i32, i32), (i32, i32)) = (a, b);
  return c.1.(1);
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'tuple.carbon'
// CHECK:STDOUT: source_filename = "tuple.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @tuple.479.loc14_3 = internal constant { i32, i32, i32 } { i32 1, i32 2, i32 3 }
// CHECK:STDOUT: @tuple.8a1.loc15_3 = internal constant { i32, i32 } { i32 4, i32 5 }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CF.Main() #0 !dbg !28 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca { i32, i32, i32 }, align 4, !dbg !31
// CHECK:STDOUT:   %b.var = alloca { i32, i32 }, align 4, !dbg !34
// CHECK:STDOUT:   %tuple.loc16_43 = alloca { i32, i32, i32 }, align 4, !dbg !37
// CHECK:STDOUT:   %tuple.loc16_46 = alloca { i32, i32 }, align 4, !dbg !38
// CHECK:STDOUT:   %tuple.loc16_47 = alloca { ptr, ptr }, align 8, !dbg !39
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !31
// CHECK:STDOUT:   %tuple.elem0.loc14.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %a.var, i32 0, i32 0, !dbg !32
// CHECK:STDOUT:   %tuple.elem1.loc14.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %a.var, i32 0, i32 1, !dbg !32
// CHECK:STDOUT:   %tuple.elem2.loc14.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %a.var, i32 0, i32 2, !dbg !32
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %a.var, ptr align 4 @tuple.479.loc14_3, i64 12, i1 false), !dbg !31
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b.var), !dbg !34
// CHECK:STDOUT:   %tuple.elem0.loc15.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %b.var, i32 0, i32 0, !dbg !35
// CHECK:STDOUT:   %tuple.elem1.loc15.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %b.var, i32 0, i32 1, !dbg !35
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %b.var, ptr align 4 @tuple.8a1.loc15_3, i64 8, i1 false), !dbg !34
// CHECK:STDOUT:   %tuple.elem0.loc16_43.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %a.var, i32 0, i32 0, !dbg !37
// CHECK:STDOUT:   %.loc16_43.1 = load i32, ptr %tuple.elem0.loc16_43.tuple.elem, align 4, !dbg !37
// CHECK:STDOUT:   %tuple.elem1.loc16_43.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %a.var, i32 0, i32 1, !dbg !37
// CHECK:STDOUT:   %.loc16_43.2 = load i32, ptr %tuple.elem1.loc16_43.tuple.elem, align 4, !dbg !37
// CHECK:STDOUT:   %tuple.elem2.loc16.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %a.var, i32 0, i32 2, !dbg !37
// CHECK:STDOUT:   %.loc16_43.3 = load i32, ptr %tuple.elem2.loc16.tuple.elem, align 4, !dbg !37
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %tuple.loc16_43), !dbg !37
// CHECK:STDOUT:   %tuple.loc16_431 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %tuple.loc16_43, i32 0, i32 0, !dbg !37
// CHECK:STDOUT:   store i32 %.loc16_43.1, ptr %tuple.loc16_431, align 4, !dbg !37
// CHECK:STDOUT:   %tuple.loc16_432 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %tuple.loc16_43, i32 0, i32 1, !dbg !37
// CHECK:STDOUT:   store i32 %.loc16_43.2, ptr %tuple.loc16_432, align 4, !dbg !37
// CHECK:STDOUT:   %tuple.loc16_433 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %tuple.loc16_43, i32 0, i32 2, !dbg !37
// CHECK:STDOUT:   store i32 %.loc16_43.3, ptr %tuple.loc16_433, align 4, !dbg !37
// CHECK:STDOUT:   %tuple.elem0.loc16_46.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %b.var, i32 0, i32 0, !dbg !38
// CHECK:STDOUT:   %.loc16_46.1 = load i32, ptr %tuple.elem0.loc16_46.tuple.elem, align 4, !dbg !38
// CHECK:STDOUT:   %tuple.elem1.loc16_46.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %b.var, i32 0, i32 1, !dbg !38
// CHECK:STDOUT:   %.loc16_46.2 = load i32, ptr %tuple.elem1.loc16_46.tuple.elem, align 4, !dbg !38
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %tuple.loc16_46), !dbg !38
// CHECK:STDOUT:   %tuple.loc16_464 = getelementptr inbounds nuw { i32, i32 }, ptr %tuple.loc16_46, i32 0, i32 0, !dbg !38
// CHECK:STDOUT:   store i32 %.loc16_46.1, ptr %tuple.loc16_464, align 4, !dbg !38
// CHECK:STDOUT:   %tuple.loc16_465 = getelementptr inbounds nuw { i32, i32 }, ptr %tuple.loc16_46, i32 0, i32 1, !dbg !38
// CHECK:STDOUT:   store i32 %.loc16_46.2, ptr %tuple.loc16_465, align 4, !dbg !38
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %tuple.loc16_47), !dbg !39
// CHECK:STDOUT:   %tuple.loc16_476 = getelementptr inbounds nuw { ptr, ptr }, ptr %tuple.loc16_47, i32 0, i32 0, !dbg !39
// CHECK:STDOUT:   store ptr %tuple.loc16_43, ptr %tuple.loc16_476, align 8, !dbg !39
// CHECK:STDOUT:   %tuple.loc16_477 = getelementptr inbounds nuw { ptr, ptr }, ptr %tuple.loc16_47, i32 0, i32 1, !dbg !39
// CHECK:STDOUT:   store ptr %tuple.loc16_46, ptr %tuple.loc16_477, align 8, !dbg !39
// CHECK:STDOUT:   %tuple.elem1.loc17_11.tuple.elem = getelementptr inbounds nuw { ptr, ptr }, ptr %tuple.loc16_47, i32 0, i32 1, !dbg !42
// CHECK:STDOUT:   %tuple.elem1.loc17_11.tuple.elem.load = load ptr, ptr %tuple.elem1.loc17_11.tuple.elem, align 8, !dbg !42
// CHECK:STDOUT:   %tuple.elem1.loc17_13.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %tuple.elem1.loc17_11.tuple.elem.load, i32 0, i32 1, !dbg !42
// CHECK:STDOUT:   %tuple.elem1.loc17_13.tuple.elem.load = load i32, ptr %tuple.elem1.loc17_13.tuple.elem, align 4, !dbg !42
// CHECK:STDOUT:   call void @"_CSelfDestruct.e554695bf635d9c1:core.Destroy.Core"(ptr %b.var), !dbg !34
// CHECK:STDOUT:   call void @"_CSelfDestruct.555b1314256372b8:core.Destroy.Core"(ptr %a.var), !dbg !31
// CHECK:STDOUT:   ret i32 %tuple.elem1.loc17_13.tuple.elem.load, !dbg !43
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.6dae29ece9a708c4:core.Destroy.Core"(ptr %self) #0 !dbg !44 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !46
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !48 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !50
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !52 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self), !dbg !54
// CHECK:STDOUT:   ret void, !dbg !54
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.e554695bf635d9c1:core.Destroy.Core"(ptr %self) #0 !dbg !56 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !58
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.e554695bf635d9c1:core.Destroy.Core"(ptr %self) #0 !dbg !60 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.e554695bf635d9c1:core.Destroy.Core"(ptr %self), !dbg !62
// CHECK:STDOUT:   ret void, !dbg !62
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.555b1314256372b8:core.Destroy.Core"(ptr %self) #0 !dbg !64 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !66
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.555b1314256372b8:core.Destroy.Core"(ptr %self) #0 !dbg !68 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.555b1314256372b8:core.Destroy.Core"(ptr %self), !dbg !70
// CHECK:STDOUT:   ret void, !dbg !70
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
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !18 = !DIFile(filename: "tuple.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !{!20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = !{null, !20}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !25 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !26 = !{null, !25}
// CHECK:STDOUT: !27 = !DISubroutineType(types: !26)
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !18, line: 13, type: !22, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !31 = !DILocation(line: 14, column: 3, scope: !28)
// CHECK:STDOUT: !32 = !DILocation(line: 14, column: 28, scope: !28)
// CHECK:STDOUT: !34 = !DILocation(line: 15, column: 3, scope: !28)
// CHECK:STDOUT: !35 = !DILocation(line: 15, column: 23, scope: !28)
// CHECK:STDOUT: !37 = !DILocation(line: 16, column: 43, scope: !28)
// CHECK:STDOUT: !38 = !DILocation(line: 16, column: 46, scope: !28)
// CHECK:STDOUT: !39 = !DILocation(line: 16, column: 42, scope: !28)
// CHECK:STDOUT: !42 = !DILocation(line: 17, column: 10, scope: !28)
// CHECK:STDOUT: !43 = !DILocation(line: 17, column: 3, scope: !28)
// CHECK:STDOUT: !44 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.6dae29ece9a708c4:core.Destroy.Core", scope: null, file: !18, line: 15, type: !24, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !47)
// CHECK:STDOUT: !45 = !DILocalVariable(arg: 1, scope: !44, type: !20)
// CHECK:STDOUT: !46 = !DILocation(line: 15, column: 3, scope: !44)
// CHECK:STDOUT: !47 = !{!45}
// CHECK:STDOUT: !48 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core", scope: null, file: !18, line: 15, type: !24, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !51)
// CHECK:STDOUT: !49 = !DILocalVariable(arg: 1, scope: !48, type: !20)
// CHECK:STDOUT: !50 = !DILocation(line: 15, column: 3, scope: !48)
// CHECK:STDOUT: !51 = !{!49}
// CHECK:STDOUT: !52 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f58524831c37807e:core.Destroy.Core", scope: null, file: !18, line: 15, type: !24, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !55)
// CHECK:STDOUT: !53 = !DILocalVariable(arg: 1, scope: !52, type: !20)
// CHECK:STDOUT: !54 = !DILocation(line: 15, column: 3, scope: !52)
// CHECK:STDOUT: !55 = !{!53}
// CHECK:STDOUT: !56 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.e554695bf635d9c1:core.Destroy.Core", scope: null, file: !18, line: 15, type: !27, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !59)
// CHECK:STDOUT: !57 = !DILocalVariable(arg: 1, scope: !56, type: !25)
// CHECK:STDOUT: !58 = !DILocation(line: 15, column: 3, scope: !56)
// CHECK:STDOUT: !59 = !{!57}
// CHECK:STDOUT: !60 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.e554695bf635d9c1:core.Destroy.Core", scope: null, file: !18, line: 15, type: !27, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !63)
// CHECK:STDOUT: !61 = !DILocalVariable(arg: 1, scope: !60, type: !25)
// CHECK:STDOUT: !62 = !DILocation(line: 15, column: 3, scope: !60)
// CHECK:STDOUT: !63 = !{!61}
// CHECK:STDOUT: !64 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.555b1314256372b8:core.Destroy.Core", scope: null, file: !18, line: 14, type: !27, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !67)
// CHECK:STDOUT: !65 = !DILocalVariable(arg: 1, scope: !64, type: !25)
// CHECK:STDOUT: !66 = !DILocation(line: 14, column: 3, scope: !64)
// CHECK:STDOUT: !67 = !{!65}
// CHECK:STDOUT: !68 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.555b1314256372b8:core.Destroy.Core", scope: null, file: !18, line: 14, type: !27, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !71)
// CHECK:STDOUT: !69 = !DILocalVariable(arg: 1, scope: !68, type: !25)
// CHECK:STDOUT: !70 = !DILocation(line: 14, column: 3, scope: !68)
// CHECK:STDOUT: !71 = !{!69}
// CHECK:STDOUT:
