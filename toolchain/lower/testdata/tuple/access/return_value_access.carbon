// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/tuple/access/return_value_access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/tuple/access/return_value_access.carbon

fn F() -> (i32, i32) { return (12, 24); }

fn Run() {
  var _: i32 = F().1;
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'return_value_access.carbon'
// CHECK:STDOUT: source_filename = "return_value_access.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @tuple.311.loc13_39 = internal constant { i32, i32 } { i32 12, i32 24 }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main(ptr sret({ i32, i32 }) %return) #0 !dbg !26 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 0, !dbg !29
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 1, !dbg !29
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %return, ptr align 4 @tuple.311.loc13_39, i64 8, i1 false), !dbg !30
// CHECK:STDOUT:   ret void, !dbg !30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @main() #0 !dbg !31 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca i32, align 4, !dbg !32
// CHECK:STDOUT:   %.loc16_18.1.temp = alloca { i32, i32 }, align 4, !dbg !33
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !32
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc16_18.1.temp), !dbg !33
// CHECK:STDOUT:   call void @_CF.Main(ptr %.loc16_18.1.temp), !dbg !33
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %.loc16_18.1.temp, i32 0, i32 1, !dbg !33
// CHECK:STDOUT:   %.loc16_19 = load i32, ptr %tuple.elem1.tuple.elem, align 4, !dbg !33
// CHECK:STDOUT:   store i32 %.loc16_19, ptr %_.var, align 4, !dbg !32
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.ffbc82e0019b7bf2(ptr %.loc16_18.1.temp), !dbg !33
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %_.var), !dbg !32
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
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.f127f30143696d51(ptr %t) #0 !dbg !47 {
// CHECK:STDOUT:   ret void, !dbg !52
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
// CHECK:STDOUT: !18 = !DIFile(filename: "return_value_access.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{!21}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !24 = !{!20}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !18, line: 13, type: !23, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !29 = !DILocation(line: 13, column: 31, scope: !26)
// CHECK:STDOUT: !30 = !DILocation(line: 13, column: 24, scope: !26)
// CHECK:STDOUT: !31 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !18, line: 15, type: !25, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !32 = !DILocation(line: 16, column: 3, scope: !31)
// CHECK:STDOUT: !33 = !DILocation(line: 16, column: 16, scope: !31)
// CHECK:STDOUT: !35 = !{null, !21}
// CHECK:STDOUT: !36 = !DISubroutineType(types: !35)
// CHECK:STDOUT: !37 = !{null, !20}
// CHECK:STDOUT: !38 = !DISubroutineType(types: !37)
// CHECK:STDOUT: !39 = !DILocation(line: 15, column: 1, scope: !31)
// CHECK:STDOUT: !40 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.ffbc82e0019b7bf2", scope: null, file: !10, line: 32, type: !36, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !46)
// CHECK:STDOUT: !41 = !DILocalVariable(arg: 1, scope: !40, type: !21)
// CHECK:STDOUT: !45 = !DILocation(line: 32, column: 1, scope: !40)
// CHECK:STDOUT: !46 = !{!41}
// CHECK:STDOUT: !47 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.f127f30143696d51", scope: null, file: !10, line: 32, type: !38, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !53)
// CHECK:STDOUT: !48 = !DILocalVariable(arg: 1, scope: !47, type: !20)
// CHECK:STDOUT: !52 = !DILocation(line: 32, column: 1, scope: !47)
// CHECK:STDOUT: !53 = !{!48}
// CHECK:STDOUT:
