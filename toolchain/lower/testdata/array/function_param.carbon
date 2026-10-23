// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/array.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/array/function_param.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/array/function_param.carbon

fn F(arr: array(i32, 3), i: i32) -> i32 {
  return arr[i];
}

fn G() -> i32 {
  return F((1, 2, 3), 1);
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'function_param.carbon'
// CHECK:STDOUT: source_filename = "function_param.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @array = internal constant [3 x i32] [i32 1, i32 2, i32 3]
// CHECK:STDOUT: @array.loc18_20.15 = internal constant [3 x i32] [i32 1, i32 2, i32 3]
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CF.Main(ptr %arr, i32 %i) #0 !dbg !28 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc14_15.2.array.index = getelementptr inbounds [3 x i32], ptr %arr, i32 0, i32 %i, !dbg !35
// CHECK:STDOUT:   %.loc14_15.3 = load i32, ptr %.loc14_15.2.array.index, align 4, !dbg !35
// CHECK:STDOUT:   ret i32 %.loc14_15.3, !dbg !37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CG.Main() #0 !dbg !39 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc18_20.3.temp = alloca [3 x i32], align 4, !dbg !42
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc18_20.3.temp), !dbg !42
// CHECK:STDOUT:   %.loc18_20.4.array.index = getelementptr inbounds [3 x i32], ptr %.loc18_20.3.temp, i32 0, i64 0, !dbg !42
// CHECK:STDOUT:   %.loc18_20.7.array.index = getelementptr inbounds [3 x i32], ptr %.loc18_20.3.temp, i32 0, i64 1, !dbg !42
// CHECK:STDOUT:   %.loc18_20.10.array.index = getelementptr inbounds [3 x i32], ptr %.loc18_20.3.temp, i32 0, i64 2, !dbg !42
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main(ptr @array.loc18_20.15, i32 1), !dbg !43
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.3c9880f178514995(ptr @array), !dbg !42
// CHECK:STDOUT:   ret i32 %F.call, !dbg !46
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.3c9880f178514995(ptr %t) #0 !dbg !47 {
// CHECK:STDOUT:   ret void, !dbg !52
// CHECK:STDOUT: }
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
// CHECK:STDOUT: !20 = !DIFile(filename: "function_param.carbon", directory: "")
// CHECK:STDOUT: !21 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !20, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !22 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{!22, !23, !22}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = !{!22}
// CHECK:STDOUT: !27 = !DISubroutineType(types: !26)
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !20, line: 13, type: !25, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !38)
// CHECK:STDOUT: !29 = !DILocalVariable(arg: 1, scope: !28, type: !23)
// CHECK:STDOUT: !30 = !DILocalVariable(arg: 2, scope: !28, type: !22)
// CHECK:STDOUT: !35 = !DILocation(line: 14, column: 10, scope: !28)
// CHECK:STDOUT: !37 = !DILocation(line: 14, column: 3, scope: !28)
// CHECK:STDOUT: !38 = !{!29, !30}
// CHECK:STDOUT: !39 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !20, line: 17, type: !27, spFlags: DISPFlagDefinition, unit: !21)
// CHECK:STDOUT: !42 = !DILocation(line: 18, column: 12, scope: !39)
// CHECK:STDOUT: !43 = !DILocation(line: 18, column: 10, scope: !39)
// CHECK:STDOUT: !44 = !{null, !23}
// CHECK:STDOUT: !45 = !DISubroutineType(types: !44)
// CHECK:STDOUT: !46 = !DILocation(line: 18, column: 3, scope: !39)
// CHECK:STDOUT: !47 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.3c9880f178514995", scope: null, file: !14, line: 32, type: !45, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !53)
// CHECK:STDOUT: !48 = !DILocalVariable(arg: 1, scope: !47, type: !23)
// CHECK:STDOUT: !52 = !DILocation(line: 32, column: 1, scope: !47)
// CHECK:STDOUT: !53 = !{!48}
// CHECK:STDOUT:
