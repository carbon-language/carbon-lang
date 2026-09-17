// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/operators/assignment.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/operators/assignment.carbon

fn F() -> i32 {
  var a: i32 = 12;
  a = 9;
  return a;
}

fn G() -> (i32, i32) {
  var b: (i32, i32);
  b = (1, 2);
  return b;
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'assignment.carbon'
// CHECK:STDOUT: source_filename = "assignment.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @tuple.102.loc21_5 = internal constant { i32, i32 } { i32 1, i32 2 }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CF.Main() #0 !dbg !30 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca i32, align 4, !dbg !33
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !33
// CHECK:STDOUT:   store i32 12, ptr %a.var, align 4, !dbg !33
// CHECK:STDOUT:   store i32 9, ptr %a.var, align 4, !dbg !35
// CHECK:STDOUT:   %.loc16 = load i32, ptr %a.var, align 4, !dbg !36
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48(ptr %a.var), !dbg !33
// CHECK:STDOUT:   ret i32 %.loc16, !dbg !37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !38 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main(ptr sret({ i32, i32 }) %return) #0 !dbg !42 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %b.var = alloca { i32, i32 }, align 4, !dbg !45
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b.var), !dbg !45
// CHECK:STDOUT:   %tuple.elem0.loc21.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %b.var, i32 0, i32 0, !dbg !48
// CHECK:STDOUT:   %tuple.elem1.loc21.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %b.var, i32 0, i32 1, !dbg !48
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %b.var, ptr align 4 @tuple.102.loc21_5, i64 8, i1 false), !dbg !47
// CHECK:STDOUT:   %tuple.elem0.loc22_10.1.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %b.var, i32 0, i32 0, !dbg !49
// CHECK:STDOUT:   %.loc22_10.1 = load i32, ptr %tuple.elem0.loc22_10.1.tuple.elem, align 4, !dbg !49
// CHECK:STDOUT:   %tuple.elem0.loc22_10.2.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 0, !dbg !49
// CHECK:STDOUT:   store i32 %.loc22_10.1, ptr %tuple.elem0.loc22_10.2.tuple.elem, align 4, !dbg !49
// CHECK:STDOUT:   %tuple.elem1.loc22_10.1.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %b.var, i32 0, i32 1, !dbg !49
// CHECK:STDOUT:   %.loc22_10.3 = load i32, ptr %tuple.elem1.loc22_10.1.tuple.elem, align 4, !dbg !49
// CHECK:STDOUT:   %tuple.elem1.loc22_10.2.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 1, !dbg !49
// CHECK:STDOUT:   store i32 %.loc22_10.3, ptr %tuple.elem1.loc22_10.2.tuple.elem, align 4, !dbg !49
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.c6bc553020e9306c(ptr %b.var), !dbg !45
// CHECK:STDOUT:   ret void, !dbg !50
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.e554695bf635d9c1:core.Destroy.Core"(ptr %self) #0 !dbg !51 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48(ptr %self) #0 !dbg !55 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self), !dbg !59
// CHECK:STDOUT:   ret void, !dbg !60
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.c6bc553020e9306c(ptr %self) #0 !dbg !62 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.e554695bf635d9c1:core.Destroy.Core"(ptr %self), !dbg !66
// CHECK:STDOUT:   ret void, !dbg !67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
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
// CHECK:STDOUT: !18 = !DIFile(filename: "assignment.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !{!20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = !{null, !20}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !25 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !26 = !{!25}
// CHECK:STDOUT: !27 = !DISubroutineType(types: !26)
// CHECK:STDOUT: !28 = !{null, !25}
// CHECK:STDOUT: !29 = !DISubroutineType(types: !28)
// CHECK:STDOUT: !30 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !18, line: 13, type: !22, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !33 = !DILocation(line: 14, column: 3, scope: !30)
// CHECK:STDOUT: !35 = !DILocation(line: 15, column: 3, scope: !30)
// CHECK:STDOUT: !36 = !DILocation(line: 16, column: 10, scope: !30)
// CHECK:STDOUT: !37 = !DILocation(line: 16, column: 3, scope: !30)
// CHECK:STDOUT: !38 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core", scope: null, file: !18, line: 14, type: !24, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !41)
// CHECK:STDOUT: !39 = !DILocalVariable(arg: 1, scope: !38, type: !20)
// CHECK:STDOUT: !40 = !DILocation(line: 14, column: 3, scope: !38)
// CHECK:STDOUT: !41 = !{!39}
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !18, line: 19, type: !27, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !45 = !DILocation(line: 20, column: 3, scope: !42)
// CHECK:STDOUT: !47 = !DILocation(line: 21, column: 3, scope: !42)
// CHECK:STDOUT: !48 = !DILocation(line: 21, column: 7, scope: !42)
// CHECK:STDOUT: !49 = !DILocation(line: 22, column: 10, scope: !42)
// CHECK:STDOUT: !50 = !DILocation(line: 22, column: 3, scope: !42)
// CHECK:STDOUT: !51 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.e554695bf635d9c1:core.Destroy.Core", scope: null, file: !18, line: 20, type: !29, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !54)
// CHECK:STDOUT: !52 = !DILocalVariable(arg: 1, scope: !51, type: !25)
// CHECK:STDOUT: !53 = !DILocation(line: 20, column: 3, scope: !51)
// CHECK:STDOUT: !54 = !{!52}
// CHECK:STDOUT: !55 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48", scope: null, file: !14, line: 24, type: !24, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !61)
// CHECK:STDOUT: !56 = !DILocalVariable(arg: 1, scope: !55, type: !20)
// CHECK:STDOUT: !59 = !DILocation(line: 26, column: 5, scope: !55)
// CHECK:STDOUT: !60 = !DILocation(line: 24, column: 3, scope: !55)
// CHECK:STDOUT: !61 = !{!56}
// CHECK:STDOUT: !62 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.c6bc553020e9306c", scope: null, file: !14, line: 24, type: !29, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !68)
// CHECK:STDOUT: !63 = !DILocalVariable(arg: 1, scope: !62, type: !25)
// CHECK:STDOUT: !66 = !DILocation(line: 26, column: 5, scope: !62)
// CHECK:STDOUT: !67 = !DILocation(line: 24, column: 3, scope: !62)
// CHECK:STDOUT: !68 = !{!63}
// CHECK:STDOUT:
