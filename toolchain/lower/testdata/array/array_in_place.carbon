// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/array/array_in_place.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/array/array_in_place.carbon

fn F() -> (i32, i32, i32);

fn G() {
  var v: array((i32, i32, i32), 2) = (F(), F());
}

// CHECK:STDOUT: ; ModuleID = 'array_in_place.carbon'
// CHECK:STDOUT: source_filename = "array_in_place.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CF.Main(ptr sret({ i32, i32, i32 }))
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CG.Main() !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %v.var = alloca [2 x { i32, i32, i32 }], align 8, !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 24, ptr %v.var), !dbg !7
// CHECK:STDOUT:   %.loc14_47.2.array.index = getelementptr inbounds [2 x { i32, i32, i32 }], ptr %v.var, i32 0, i64 0, !dbg !8
// CHECK:STDOUT:   call void @_CF.Main(ptr %.loc14_47.2.array.index), !dbg !9
// CHECK:STDOUT:   %.loc14_47.4.array.index = getelementptr inbounds [2 x { i32, i32, i32 }], ptr %v.var, i32 0, i64 1, !dbg !8
// CHECK:STDOUT:   call void @_CF.Main(ptr %.loc14_47.4.array.index), !dbg !10
// CHECK:STDOUT:   ret void, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(i64 immarg, ptr captures(none)) #0
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "array_in_place.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !3, line: 13, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 14, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 14, column: 38, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 14, column: 39, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 14, column: 44, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 13, column: 1, scope: !4)
