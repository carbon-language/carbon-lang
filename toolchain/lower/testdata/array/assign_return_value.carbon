// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/array.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/array/assign_return_value.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/array/assign_return_value.carbon

fn F() -> (i32, i32) { return (12, 24); }

fn Run() {
  var _: array(i32, 2) = F();
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'assign_return_value.carbon'
// CHECK:STDOUT: source_filename = "assign_return_value.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @tuple.311.loc13_39 = internal constant { i32, i32 } { i32 12, i32 24 }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main(ptr sret({ i32, i32 }) %return) #0 !dbg !28 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 0, !dbg !31
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 1, !dbg !31
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %return, ptr align 4 @tuple.311.loc13_39, i64 8, i1 false), !dbg !32
// CHECK:STDOUT:   ret void, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @main() #0 !dbg !33 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca [2 x i32], align 4, !dbg !34
// CHECK:STDOUT:   %.loc16_28.1.temp = alloca { i32, i32 }, align 4, !dbg !35
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !34
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc16_28.1.temp), !dbg !35
// CHECK:STDOUT:   call void @_CF.Main(ptr %.loc16_28.1.temp), !dbg !35
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %.loc16_28.1.temp, i32 0, i32 0, !dbg !35
// CHECK:STDOUT:   %.loc16_28.3 = load i32, ptr %tuple.elem0.tuple.elem, align 4, !dbg !35
// CHECK:STDOUT:   %.loc16_28.4.array.index = getelementptr inbounds [2 x i32], ptr %_.var, i32 0, i64 0, !dbg !35
// CHECK:STDOUT:   store i32 %.loc16_28.3, ptr %.loc16_28.4.array.index, align 4, !dbg !35
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %.loc16_28.1.temp, i32 0, i32 1, !dbg !35
// CHECK:STDOUT:   %.loc16_28.6 = load i32, ptr %tuple.elem1.tuple.elem, align 4, !dbg !35
// CHECK:STDOUT:   %.loc16_28.7.array.index = getelementptr inbounds [2 x i32], ptr %_.var, i32 0, i64 1, !dbg !35
// CHECK:STDOUT:   store i32 %.loc16_28.6, ptr %.loc16_28.7.array.index, align 4, !dbg !35
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.ffbc82e0019b7bf2(ptr %.loc16_28.1.temp), !dbg !35
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.4f2cee5f6b4c30a4(ptr %_.var), !dbg !34
// CHECK:STDOUT:   ret i32 0, !dbg !39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.ffbc82e0019b7bf2(ptr %t) #0 !dbg !40 {
// CHECK:STDOUT:   ret void, !dbg !45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.4f2cee5f6b4c30a4(ptr %t) #0 !dbg !47 {
// CHECK:STDOUT:   ret void, !dbg !52
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
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
// CHECK:STDOUT: !20 = !DIFile(filename: "assign_return_value.carbon", directory: "")
// CHECK:STDOUT: !21 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !20, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !22 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{!23}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = !{!22}
// CHECK:STDOUT: !27 = !DISubroutineType(types: !26)
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !20, line: 13, type: !25, spFlags: DISPFlagDefinition, unit: !21)
// CHECK:STDOUT: !31 = !DILocation(line: 13, column: 31, scope: !28)
// CHECK:STDOUT: !32 = !DILocation(line: 13, column: 24, scope: !28)
// CHECK:STDOUT: !33 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !20, line: 15, type: !27, spFlags: DISPFlagDefinition, unit: !21)
// CHECK:STDOUT: !34 = !DILocation(line: 16, column: 3, scope: !33)
// CHECK:STDOUT: !35 = !DILocation(line: 16, column: 26, scope: !33)
// CHECK:STDOUT: !37 = !{null, !23}
// CHECK:STDOUT: !38 = !DISubroutineType(types: !37)
// CHECK:STDOUT: !39 = !DILocation(line: 15, column: 1, scope: !33)
// CHECK:STDOUT: !40 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.ffbc82e0019b7bf2", scope: null, file: !14, line: 32, type: !38, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !46)
// CHECK:STDOUT: !41 = !DILocalVariable(arg: 1, scope: !40, type: !23)
// CHECK:STDOUT: !45 = !DILocation(line: 32, column: 1, scope: !40)
// CHECK:STDOUT: !46 = !{!41}
// CHECK:STDOUT: !47 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.4f2cee5f6b4c30a4", scope: null, file: !14, line: 32, type: !38, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !53)
// CHECK:STDOUT: !48 = !DILocalVariable(arg: 1, scope: !47, type: !23)
// CHECK:STDOUT: !52 = !DILocation(line: 32, column: 1, scope: !47)
// CHECK:STDOUT: !53 = !{!48}
// CHECK:STDOUT:
