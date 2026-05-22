// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
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

// CHECK:STDOUT: ; ModuleID = 'function_param.carbon'
// CHECK:STDOUT: source_filename = "function_param.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @array = internal constant [3 x i32] [i32 1, i32 2, i32 3]
// CHECK:STDOUT: @array.loc18_20.15 = internal constant [3 x i32] [i32 1, i32 2, i32 3]
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CF.Main(ptr %arr, i32 %i) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc14_15.2.array.index = getelementptr inbounds [3 x i32], ptr %arr, i32 0, i32 %i, !dbg !12
// CHECK:STDOUT:   %.loc14_15.3 = load i32, ptr %.loc14_15.2.array.index, align 4, !dbg !12
// CHECK:STDOUT:   ret i32 %.loc14_15.3, !dbg !13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CG.Main() #0 !dbg !14 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc18_20.3.temp = alloca [3 x i32], align 4, !dbg !17
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc18_20.3.temp), !dbg !17
// CHECK:STDOUT:   %.loc18_20.4.array.index = getelementptr inbounds [3 x i32], ptr %.loc18_20.3.temp, i32 0, i64 0, !dbg !17
// CHECK:STDOUT:   %.loc18_20.7.array.index = getelementptr inbounds [3 x i32], ptr %.loc18_20.3.temp, i32 0, i64 1, !dbg !17
// CHECK:STDOUT:   %.loc18_20.10.array.index = getelementptr inbounds [3 x i32], ptr %.loc18_20.3.temp, i32 0, i64 2, !dbg !17
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main(ptr @array.loc18_20.15, i32 1), !dbg !18
// CHECK:STDOUT:   call void @"_COp.1c95d180afb5f529:core.Destroy.Core"(ptr @array), !dbg !17
// CHECK:STDOUT:   ret i32 %F.call, !dbg !19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.94e3ca86614aba5d:core.Destroy.Core"(ptr %self) #0 !dbg !20 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.1c95d180afb5f529:core.Destroy.Core"(ptr %self) #0 !dbg !26 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "function_param.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !3, line: 13, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !9)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7, !8, !7}
// CHECK:STDOUT: !7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !9 = !{!10, !11}
// CHECK:STDOUT: !10 = !DILocalVariable(arg: 1, scope: !4, type: !8)
// CHECK:STDOUT: !11 = !DILocalVariable(arg: 2, scope: !4, type: !7)
// CHECK:STDOUT: !12 = !DILocation(line: 14, column: 10, scope: !4)
// CHECK:STDOUT: !13 = !DILocation(line: 14, column: 3, scope: !4)
// CHECK:STDOUT: !14 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !3, line: 17, type: !15, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !15 = !DISubroutineType(types: !16)
// CHECK:STDOUT: !16 = !{!7}
// CHECK:STDOUT: !17 = !DILocation(line: 18, column: 12, scope: !14)
// CHECK:STDOUT: !18 = !DILocation(line: 18, column: 10, scope: !14)
// CHECK:STDOUT: !19 = !DILocation(line: 18, column: 3, scope: !14)
// CHECK:STDOUT: !20 = distinct !DISubprogram(name: "Op", linkageName: "_COp.94e3ca86614aba5d:core.Destroy.Core", scope: null, file: !3, line: 18, type: !21, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !23)
// CHECK:STDOUT: !21 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !22 = !{null, !7}
// CHECK:STDOUT: !23 = !{!24}
// CHECK:STDOUT: !24 = !DILocalVariable(arg: 1, scope: !20, type: !7)
// CHECK:STDOUT: !25 = !DILocation(line: 18, column: 12, scope: !20)
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "Op", linkageName: "_COp.1c95d180afb5f529:core.Destroy.Core", scope: null, file: !3, line: 18, type: !27, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !29)
// CHECK:STDOUT: !27 = !DISubroutineType(types: !28)
// CHECK:STDOUT: !28 = !{null, !8}
// CHECK:STDOUT: !29 = !{!30}
// CHECK:STDOUT: !30 = !DILocalVariable(arg: 1, scope: !26, type: !8)
// CHECK:STDOUT: !31 = !DILocation(line: 18, column: 12, scope: !26)
