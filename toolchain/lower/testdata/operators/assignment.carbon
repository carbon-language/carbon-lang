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
// CHECK:STDOUT: @tuple.380.loc21_5 = internal constant { i32, i32 } { i32 1, i32 2 }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CF.Main() #0 !dbg !26 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca i32, align 4, !dbg !29
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !29
// CHECK:STDOUT:   store i32 12, ptr %a.var, align 4, !dbg !29
// CHECK:STDOUT:   store i32 9, ptr %a.var, align 4, !dbg !31
// CHECK:STDOUT:   %.loc16 = load i32, ptr %a.var, align 4, !dbg !32
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %a.var), !dbg !29
// CHECK:STDOUT:   ret i32 %.loc16, !dbg !35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main(ptr sret({ i32, i32 }) %return) #0 !dbg !36 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %b.var = alloca { i32, i32 }, align 4, !dbg !39
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b.var), !dbg !39
// CHECK:STDOUT:   %tuple.elem0.loc21.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %b.var, i32 0, i32 0, !dbg !42
// CHECK:STDOUT:   %tuple.elem1.loc21.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %b.var, i32 0, i32 1, !dbg !42
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %b.var, ptr align 4 @tuple.380.loc21_5, i64 8, i1 false), !dbg !41
// CHECK:STDOUT:   %tuple.elem0.loc22_10.1.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %b.var, i32 0, i32 0, !dbg !43
// CHECK:STDOUT:   %.loc22_10.1 = load i32, ptr %tuple.elem0.loc22_10.1.tuple.elem, align 4, !dbg !43
// CHECK:STDOUT:   %tuple.elem0.loc22_10.2.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 0, !dbg !43
// CHECK:STDOUT:   store i32 %.loc22_10.1, ptr %tuple.elem0.loc22_10.2.tuple.elem, align 4, !dbg !43
// CHECK:STDOUT:   %tuple.elem1.loc22_10.1.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %b.var, i32 0, i32 1, !dbg !43
// CHECK:STDOUT:   %.loc22_10.3 = load i32, ptr %tuple.elem1.loc22_10.1.tuple.elem, align 4, !dbg !43
// CHECK:STDOUT:   %tuple.elem1.loc22_10.2.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 1, !dbg !43
// CHECK:STDOUT:   store i32 %.loc22_10.3, ptr %tuple.elem1.loc22_10.2.tuple.elem, align 4, !dbg !43
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.ffbc82e0019b7bf2(ptr %b.var), !dbg !39
// CHECK:STDOUT:   ret void, !dbg !44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.f127f30143696d51(ptr %t) #0 !dbg !47 {
// CHECK:STDOUT:   ret void, !dbg !52
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.ffbc82e0019b7bf2(ptr %t) #0 !dbg !54 {
// CHECK:STDOUT:   ret void, !dbg !59
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
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !18 = !DIFile(filename: "assignment.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !{!20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{!23}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !18, line: 13, type: !22, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !29 = !DILocation(line: 14, column: 3, scope: !26)
// CHECK:STDOUT: !31 = !DILocation(line: 15, column: 3, scope: !26)
// CHECK:STDOUT: !32 = !DILocation(line: 16, column: 10, scope: !26)
// CHECK:STDOUT: !33 = !{null, !20}
// CHECK:STDOUT: !34 = !DISubroutineType(types: !33)
// CHECK:STDOUT: !35 = !DILocation(line: 16, column: 3, scope: !26)
// CHECK:STDOUT: !36 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !18, line: 19, type: !25, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !39 = !DILocation(line: 20, column: 3, scope: !36)
// CHECK:STDOUT: !41 = !DILocation(line: 21, column: 3, scope: !36)
// CHECK:STDOUT: !42 = !DILocation(line: 21, column: 7, scope: !36)
// CHECK:STDOUT: !43 = !DILocation(line: 22, column: 10, scope: !36)
// CHECK:STDOUT: !44 = !DILocation(line: 22, column: 3, scope: !36)
// CHECK:STDOUT: !45 = !{null, !23}
// CHECK:STDOUT: !46 = !DISubroutineType(types: !45)
// CHECK:STDOUT: !47 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.f127f30143696d51", scope: null, file: !10, line: 32, type: !34, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !53)
// CHECK:STDOUT: !48 = !DILocalVariable(arg: 1, scope: !47, type: !20)
// CHECK:STDOUT: !52 = !DILocation(line: 32, column: 1, scope: !47)
// CHECK:STDOUT: !53 = !{!48}
// CHECK:STDOUT: !54 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.ffbc82e0019b7bf2", scope: null, file: !10, line: 32, type: !46, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !60)
// CHECK:STDOUT: !55 = !DILocalVariable(arg: 1, scope: !54, type: !23)
// CHECK:STDOUT: !59 = !DILocation(line: 32, column: 1, scope: !54)
// CHECK:STDOUT: !60 = !{!55}
// CHECK:STDOUT:
