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

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'function_param.carbon'
// CHECK:STDOUT: source_filename = "function_param.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @array = internal constant [3 x i32] [i32 1, i32 2, i32 3]
// CHECK:STDOUT: @array.loc18_20.15 = internal constant [3 x i32] [i32 1, i32 2, i32 3]
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CF.Main(ptr %arr, i32 %i) #0 !dbg !30 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc14_15.2.array.index = getelementptr inbounds [3 x i32], ptr %arr, i32 0, i32 %i, !dbg !37
// CHECK:STDOUT:   %.loc14_15.3 = load i32, ptr %.loc14_15.2.array.index, align 4, !dbg !37
// CHECK:STDOUT:   ret i32 %.loc14_15.3, !dbg !39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CG.Main() #0 !dbg !41 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc18_20.3.temp = alloca [3 x i32], align 4, !dbg !44
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc18_20.3.temp), !dbg !44
// CHECK:STDOUT:   %.loc18_20.4.array.index = getelementptr inbounds [3 x i32], ptr %.loc18_20.3.temp, i32 0, i64 0, !dbg !44
// CHECK:STDOUT:   %.loc18_20.7.array.index = getelementptr inbounds [3 x i32], ptr %.loc18_20.3.temp, i32 0, i64 1, !dbg !44
// CHECK:STDOUT:   %.loc18_20.10.array.index = getelementptr inbounds [3 x i32], ptr %.loc18_20.3.temp, i32 0, i64 2, !dbg !44
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main(ptr @array.loc18_20.15, i32 1), !dbg !45
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.f699181ca81df557(ptr @array), !dbg !44
// CHECK:STDOUT:   ret i32 %F.call, !dbg !46
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !47 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !49
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.dc7dedd104b837b2:core.Destroy.Core"(ptr %self) #0 !dbg !51 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.f699181ca81df557(ptr %self) #0 !dbg !55 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.dc7dedd104b837b2:core.Destroy.Core"(ptr %self), !dbg !59
// CHECK:STDOUT:   ret void, !dbg !60
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
// CHECK:STDOUT: !18 = !DIFile(filename: "function_param.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{!20, !21, !20}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !24 = !{!20}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = !{null, !20}
// CHECK:STDOUT: !27 = !DISubroutineType(types: !26)
// CHECK:STDOUT: !28 = !{null, !21}
// CHECK:STDOUT: !29 = !DISubroutineType(types: !28)
// CHECK:STDOUT: !30 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !18, line: 13, type: !23, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !40)
// CHECK:STDOUT: !31 = !DILocalVariable(arg: 1, scope: !30, type: !21)
// CHECK:STDOUT: !32 = !DILocalVariable(arg: 2, scope: !30, type: !20)
// CHECK:STDOUT: !37 = !DILocation(line: 14, column: 10, scope: !30)
// CHECK:STDOUT: !39 = !DILocation(line: 14, column: 3, scope: !30)
// CHECK:STDOUT: !40 = !{!31, !32}
// CHECK:STDOUT: !41 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !18, line: 17, type: !25, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !44 = !DILocation(line: 18, column: 12, scope: !41)
// CHECK:STDOUT: !45 = !DILocation(line: 18, column: 10, scope: !41)
// CHECK:STDOUT: !46 = !DILocation(line: 18, column: 3, scope: !41)
// CHECK:STDOUT: !47 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core", scope: null, file: !18, line: 18, type: !27, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !50)
// CHECK:STDOUT: !48 = !DILocalVariable(arg: 1, scope: !47, type: !20)
// CHECK:STDOUT: !49 = !DILocation(line: 18, column: 12, scope: !47)
// CHECK:STDOUT: !50 = !{!48}
// CHECK:STDOUT: !51 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.dc7dedd104b837b2:core.Destroy.Core", scope: null, file: !18, line: 18, type: !29, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !54)
// CHECK:STDOUT: !52 = !DILocalVariable(arg: 1, scope: !51, type: !21)
// CHECK:STDOUT: !53 = !DILocation(line: 18, column: 12, scope: !51)
// CHECK:STDOUT: !54 = !{!52}
// CHECK:STDOUT: !55 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.f699181ca81df557", scope: null, file: !14, line: 24, type: !29, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !61)
// CHECK:STDOUT: !56 = !DILocalVariable(arg: 1, scope: !55, type: !21)
// CHECK:STDOUT: !59 = !DILocation(line: 26, column: 5, scope: !55)
// CHECK:STDOUT: !60 = !DILocation(line: 24, column: 3, scope: !55)
// CHECK:STDOUT: !61 = !{!56}
// CHECK:STDOUT:
