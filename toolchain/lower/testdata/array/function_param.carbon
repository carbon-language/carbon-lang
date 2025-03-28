// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
// CHECK:STDOUT: @array.loc16_20.13 = internal constant [3 x i32] [i32 1, i32 2, i32 3]
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_CF.Main(ptr %arr, i32 %i) !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc12_15.2.array.index = getelementptr inbounds [3 x i32], ptr %arr, i32 0, i32 %i, !dbg !7
// CHECK:STDOUT:   %.loc12_15.3 = load i32, ptr %.loc12_15.2.array.index, align 4, !dbg !7
// CHECK:STDOUT:   ret i32 %.loc12_15.3, !dbg !8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_CG.Main() !dbg !9 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc16_20.3.temp = alloca [3 x i32], align 4, !dbg !10
// CHECK:STDOUT:   %.loc16_20.4.array.index = getelementptr inbounds [3 x i32], ptr %.loc16_20.3.temp, i32 0, i64 0, !dbg !10
// CHECK:STDOUT:   %.loc16_20.7.array.index = getelementptr inbounds [3 x i32], ptr %.loc16_20.3.temp, i32 0, i64 1, !dbg !10
// CHECK:STDOUT:   %.loc16_20.10.array.index = getelementptr inbounds [3 x i32], ptr %.loc16_20.3.temp, i32 0, i64 2, !dbg !10
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %.loc16_20.3.temp, ptr align 4 @array.loc16_20.13, i64 12, i1 false), !dbg !10
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main(ptr %.loc16_20.3.temp, i32 1), !dbg !11
// CHECK:STDOUT:   ret i32 %F.call, !dbg !12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #0
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "function_param.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !3, line: 11, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 12, column: 10, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 12, column: 3, scope: !4)
// CHECK:STDOUT: !9 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !3, line: 15, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !10 = !DILocation(line: 16, column: 12, scope: !9)
// CHECK:STDOUT: !11 = !DILocation(line: 16, column: 10, scope: !9)
// CHECK:STDOUT: !12 = !DILocation(line: 16, column: 3, scope: !9)
