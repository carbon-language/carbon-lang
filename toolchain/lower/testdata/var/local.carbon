// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/var/local.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/var/local.carbon

// --- simple.carbon

library "[[@TEST_NAME]]";

fn Run() -> i32 {
  var x: i32 = 1;
  return x;
}

// --- eval_init_in_place.carbon

library "[[@TEST_NAME]]";

eval fn F() -> (i32, i32, i32) { return (1, 2, 3); }

fn G() -> (i32, i32, i32) {
  var a: (i32, i32, i32) = F();
  return a;
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'simple.carbon'
// CHECK:STDOUT: source_filename = "simple.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @main() #0 !dbg !25 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x.var = alloca i32, align 4, !dbg !28
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %x.var), !dbg !28
// CHECK:STDOUT:   store i32 1, ptr %x.var, align 4, !dbg !28
// CHECK:STDOUT:   %.loc6 = load i32, ptr %x.var, align 4, !dbg !30
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48(ptr %x.var), !dbg !28
// CHECK:STDOUT:   ret i32 %.loc6, !dbg !31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !32 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48(ptr %self) #0 !dbg !36 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self), !dbg !40
// CHECK:STDOUT:   ret void, !dbg !41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !14 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !18 = !DIFile(filename: "simple.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !{!20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = !{null, !20}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !18, line: 4, type: !22, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !28 = !DILocation(line: 5, column: 3, scope: !25)
// CHECK:STDOUT: !30 = !DILocation(line: 6, column: 10, scope: !25)
// CHECK:STDOUT: !31 = !DILocation(line: 6, column: 3, scope: !25)
// CHECK:STDOUT: !32 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core", scope: null, file: !18, line: 5, type: !24, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !35)
// CHECK:STDOUT: !33 = !DILocalVariable(arg: 1, scope: !32, type: !20)
// CHECK:STDOUT: !34 = !DILocation(line: 5, column: 3, scope: !32)
// CHECK:STDOUT: !35 = !{!33}
// CHECK:STDOUT: !36 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48", scope: null, file: !14, line: 24, type: !24, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !42)
// CHECK:STDOUT: !37 = !DILocalVariable(arg: 1, scope: !36, type: !20)
// CHECK:STDOUT: !40 = !DILocation(line: 26, column: 5, scope: !36)
// CHECK:STDOUT: !41 = !DILocation(line: 24, column: 3, scope: !36)
// CHECK:STDOUT: !42 = !{!37}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'eval_init_in_place.carbon'
// CHECK:STDOUT: source_filename = "eval_init_in_place.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @tuple.479.loc4_50 = internal constant { i32, i32, i32 } { i32 1, i32 2, i32 3 }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main(ptr sret({ i32, i32, i32 }) %return) #0 !dbg !50 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %return, i32 0, i32 0, !dbg !53
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %return, i32 0, i32 1, !dbg !53
// CHECK:STDOUT:   %tuple.elem2.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %return, i32 0, i32 2, !dbg !53
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %return, ptr align 4 @tuple.479.loc4_50, i64 12, i1 false), !dbg !54
// CHECK:STDOUT:   ret void, !dbg !54
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main(ptr sret({ i32, i32, i32 }) %return) #0 !dbg !55 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca { i32, i32, i32 }, align 4, !dbg !58
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !58
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %a.var, ptr align 4 @tuple.479.loc4_50, i64 12, i1 false), !dbg !58
// CHECK:STDOUT:   %tuple.elem0.loc8_10.1.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %a.var, i32 0, i32 0, !dbg !60
// CHECK:STDOUT:   %.loc8_10.1 = load i32, ptr %tuple.elem0.loc8_10.1.tuple.elem, align 4, !dbg !60
// CHECK:STDOUT:   %tuple.elem0.loc8_10.2.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %return, i32 0, i32 0, !dbg !60
// CHECK:STDOUT:   store i32 %.loc8_10.1, ptr %tuple.elem0.loc8_10.2.tuple.elem, align 4, !dbg !60
// CHECK:STDOUT:   %tuple.elem1.loc8_10.1.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %a.var, i32 0, i32 1, !dbg !60
// CHECK:STDOUT:   %.loc8_10.3 = load i32, ptr %tuple.elem1.loc8_10.1.tuple.elem, align 4, !dbg !60
// CHECK:STDOUT:   %tuple.elem1.loc8_10.2.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %return, i32 0, i32 1, !dbg !60
// CHECK:STDOUT:   store i32 %.loc8_10.3, ptr %tuple.elem1.loc8_10.2.tuple.elem, align 4, !dbg !60
// CHECK:STDOUT:   %tuple.elem2.loc8_10.1.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %a.var, i32 0, i32 2, !dbg !60
// CHECK:STDOUT:   %.loc8_10.5 = load i32, ptr %tuple.elem2.loc8_10.1.tuple.elem, align 4, !dbg !60
// CHECK:STDOUT:   %tuple.elem2.loc8_10.2.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %return, i32 0, i32 2, !dbg !60
// CHECK:STDOUT:   store i32 %.loc8_10.5, ptr %tuple.elem2.loc8_10.2.tuple.elem, align 4, !dbg !60
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.dd851822db46e4f5(ptr %a.var), !dbg !58
// CHECK:STDOUT:   ret void, !dbg !61
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !62 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !64
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.555b1314256372b8:core.Destroy.Core"(ptr %self) #0 !dbg !66 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !68
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.dd851822db46e4f5(ptr %self) #0 !dbg !70 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.555b1314256372b8:core.Destroy.Core"(ptr %self), !dbg !74
// CHECK:STDOUT:   ret void, !dbg !75
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!44}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !14 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !23 = !{null, !20}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !43 = !DIFile(filename: "eval_init_in_place.carbon", directory: "")
// CHECK:STDOUT: !44 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !43, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !45 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !46 = !{!45}
// CHECK:STDOUT: !47 = !DISubroutineType(types: !46)
// CHECK:STDOUT: !48 = !{null, !45}
// CHECK:STDOUT: !49 = !DISubroutineType(types: !48)
// CHECK:STDOUT: !50 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !43, line: 4, type: !47, spFlags: DISPFlagDefinition, unit: !44)
// CHECK:STDOUT: !53 = !DILocation(line: 4, column: 41, scope: !50)
// CHECK:STDOUT: !54 = !DILocation(line: 4, column: 34, scope: !50)
// CHECK:STDOUT: !55 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !43, line: 6, type: !47, spFlags: DISPFlagDefinition, unit: !44)
// CHECK:STDOUT: !58 = !DILocation(line: 7, column: 3, scope: !55)
// CHECK:STDOUT: !60 = !DILocation(line: 8, column: 10, scope: !55)
// CHECK:STDOUT: !61 = !DILocation(line: 8, column: 3, scope: !55)
// CHECK:STDOUT: !62 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core", scope: null, file: !43, line: 7, type: !24, spFlags: DISPFlagDefinition, unit: !44, retainedNodes: !65)
// CHECK:STDOUT: !63 = !DILocalVariable(arg: 1, scope: !62, type: !20)
// CHECK:STDOUT: !64 = !DILocation(line: 7, column: 3, scope: !62)
// CHECK:STDOUT: !65 = !{!63}
// CHECK:STDOUT: !66 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.555b1314256372b8:core.Destroy.Core", scope: null, file: !43, line: 7, type: !49, spFlags: DISPFlagDefinition, unit: !44, retainedNodes: !69)
// CHECK:STDOUT: !67 = !DILocalVariable(arg: 1, scope: !66, type: !45)
// CHECK:STDOUT: !68 = !DILocation(line: 7, column: 3, scope: !66)
// CHECK:STDOUT: !69 = !{!67}
// CHECK:STDOUT: !70 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.dd851822db46e4f5", scope: null, file: !14, line: 24, type: !49, spFlags: DISPFlagDefinition, unit: !44, retainedNodes: !76)
// CHECK:STDOUT: !71 = !DILocalVariable(arg: 1, scope: !70, type: !45)
// CHECK:STDOUT: !74 = !DILocation(line: 26, column: 5, scope: !70)
// CHECK:STDOUT: !75 = !DILocation(line: 24, column: 3, scope: !70)
// CHECK:STDOUT: !76 = !{!71}
// CHECK:STDOUT:
