// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/struct/nested_struct_in_place.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/struct/nested_struct_in_place.carbon

fn F() -> (i32, i32, i32);

fn G() {
  var v: {.a: (i32, i32, i32), .b: (i32, i32, i32)} = {.a = F(), .b = F()};
}

// CHECK:STDOUT: ; ModuleID = 'nested_struct_in_place.carbon'
// CHECK:STDOUT: source_filename = "nested_struct_in_place.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CF.Main(ptr sret({ i32, i32, i32 }))
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CG.Main() !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %v.var = alloca { { i32, i32, i32 }, { i32, i32, i32 } }, align 8, !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 24, ptr %v.var), !dbg !7
// CHECK:STDOUT:   %.loc16_74.1.a = getelementptr inbounds nuw { { i32, i32, i32 }, { i32, i32, i32 } }, ptr %v.var, i32 0, i32 0, !dbg !8
// CHECK:STDOUT:   call void @_CF.Main(ptr %.loc16_74.1.a), !dbg !9
// CHECK:STDOUT:   %.loc16_74.2.b = getelementptr inbounds nuw { { i32, i32, i32 }, { i32, i32, i32 } }, ptr %v.var, i32 0, i32 1, !dbg !8
// CHECK:STDOUT:   call void @_CF.Main(ptr %.loc16_74.2.b), !dbg !10
// CHECK:STDOUT:   %tuple.elem0.loc16_74.1.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %.loc16_74.2.b, i32 0, i32 0, !dbg !8
// CHECK:STDOUT:   %.loc16_74.5 = load i32, ptr %tuple.elem0.loc16_74.1.tuple.elem, align 4, !dbg !8
// CHECK:STDOUT:   %tuple.elem1.loc16_74.1.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %.loc16_74.2.b, i32 0, i32 1, !dbg !8
// CHECK:STDOUT:   %.loc16_74.6 = load i32, ptr %tuple.elem1.loc16_74.1.tuple.elem, align 4, !dbg !8
// CHECK:STDOUT:   %tuple.elem2.loc16_74.1.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %.loc16_74.2.b, i32 0, i32 2, !dbg !8
// CHECK:STDOUT:   %.loc16_74.7 = load i32, ptr %tuple.elem2.loc16_74.1.tuple.elem, align 4, !dbg !8
// CHECK:STDOUT:   %tuple.loc16_74.1 = alloca { i32, i32, i32 }, align 8, !dbg !8
// CHECK:STDOUT:   %tuple.loc16_74.11 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %tuple.loc16_74.1, i32 0, i32 0, !dbg !8
// CHECK:STDOUT:   store i32 %.loc16_74.5, ptr %tuple.loc16_74.11, align 4, !dbg !8
// CHECK:STDOUT:   %tuple.loc16_74.12 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %tuple.loc16_74.1, i32 0, i32 1, !dbg !8
// CHECK:STDOUT:   store i32 %.loc16_74.6, ptr %tuple.loc16_74.12, align 4, !dbg !8
// CHECK:STDOUT:   %tuple.loc16_74.13 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %tuple.loc16_74.1, i32 0, i32 2, !dbg !8
// CHECK:STDOUT:   store i32 %.loc16_74.7, ptr %tuple.loc16_74.13, align 4, !dbg !8
// CHECK:STDOUT:   %tuple.elem0.loc16_74.2.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %.loc16_74.1.a, i32 0, i32 0, !dbg !8
// CHECK:STDOUT:   %.loc16_74.9 = load i32, ptr %tuple.elem0.loc16_74.2.tuple.elem, align 4, !dbg !8
// CHECK:STDOUT:   %tuple.elem1.loc16_74.2.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %.loc16_74.1.a, i32 0, i32 1, !dbg !8
// CHECK:STDOUT:   %.loc16_74.10 = load i32, ptr %tuple.elem1.loc16_74.2.tuple.elem, align 4, !dbg !8
// CHECK:STDOUT:   %tuple.elem2.loc16_74.2.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %.loc16_74.1.a, i32 0, i32 2, !dbg !8
// CHECK:STDOUT:   %.loc16_74.11 = load i32, ptr %tuple.elem2.loc16_74.2.tuple.elem, align 4, !dbg !8
// CHECK:STDOUT:   %tuple.loc16_74.2 = alloca { i32, i32, i32 }, align 8, !dbg !8
// CHECK:STDOUT:   %tuple.loc16_74.24 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %tuple.loc16_74.2, i32 0, i32 0, !dbg !8
// CHECK:STDOUT:   store i32 %.loc16_74.9, ptr %tuple.loc16_74.24, align 4, !dbg !8
// CHECK:STDOUT:   %tuple.loc16_74.25 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %tuple.loc16_74.2, i32 0, i32 1, !dbg !8
// CHECK:STDOUT:   store i32 %.loc16_74.10, ptr %tuple.loc16_74.25, align 4, !dbg !8
// CHECK:STDOUT:   %tuple.loc16_74.26 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %tuple.loc16_74.2, i32 0, i32 2, !dbg !8
// CHECK:STDOUT:   store i32 %.loc16_74.11, ptr %tuple.loc16_74.26, align 4, !dbg !8
// CHECK:STDOUT:   %.loc16_3.2.a = getelementptr inbounds nuw { { i32, i32, i32 }, { i32, i32, i32 } }, ptr %v.var, i32 0, i32 0, !dbg !7
// CHECK:STDOUT:   %tuple.elem0.loc16_3.1.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %.loc16_3.2.a, i32 0, i32 0, !dbg !7
// CHECK:STDOUT:   %.loc16_3.3 = load i32, ptr %tuple.elem0.loc16_3.1.tuple.elem, align 4, !dbg !7
// CHECK:STDOUT:   %tuple.elem1.loc16_3.1.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %.loc16_3.2.a, i32 0, i32 1, !dbg !7
// CHECK:STDOUT:   %.loc16_3.4 = load i32, ptr %tuple.elem1.loc16_3.1.tuple.elem, align 4, !dbg !7
// CHECK:STDOUT:   %tuple.elem2.loc16_3.1.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %.loc16_3.2.a, i32 0, i32 2, !dbg !7
// CHECK:STDOUT:   %.loc16_3.5 = load i32, ptr %tuple.elem2.loc16_3.1.tuple.elem, align 4, !dbg !7
// CHECK:STDOUT:   %tuple.loc16_3.1 = alloca { i32, i32, i32 }, align 8, !dbg !7
// CHECK:STDOUT:   %tuple.loc16_3.17 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %tuple.loc16_3.1, i32 0, i32 0, !dbg !7
// CHECK:STDOUT:   store i32 %.loc16_3.3, ptr %tuple.loc16_3.17, align 4, !dbg !7
// CHECK:STDOUT:   %tuple.loc16_3.18 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %tuple.loc16_3.1, i32 0, i32 1, !dbg !7
// CHECK:STDOUT:   store i32 %.loc16_3.4, ptr %tuple.loc16_3.18, align 4, !dbg !7
// CHECK:STDOUT:   %tuple.loc16_3.19 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %tuple.loc16_3.1, i32 0, i32 2, !dbg !7
// CHECK:STDOUT:   store i32 %.loc16_3.5, ptr %tuple.loc16_3.19, align 4, !dbg !7
// CHECK:STDOUT:   %.loc16_3.7.b = getelementptr inbounds nuw { { i32, i32, i32 }, { i32, i32, i32 } }, ptr %v.var, i32 0, i32 1, !dbg !7
// CHECK:STDOUT:   %tuple.elem0.loc16_3.2.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %.loc16_3.7.b, i32 0, i32 0, !dbg !7
// CHECK:STDOUT:   %.loc16_3.8 = load i32, ptr %tuple.elem0.loc16_3.2.tuple.elem, align 4, !dbg !7
// CHECK:STDOUT:   %tuple.elem1.loc16_3.2.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %.loc16_3.7.b, i32 0, i32 1, !dbg !7
// CHECK:STDOUT:   %.loc16_3.9 = load i32, ptr %tuple.elem1.loc16_3.2.tuple.elem, align 4, !dbg !7
// CHECK:STDOUT:   %tuple.elem2.loc16_3.2.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %.loc16_3.7.b, i32 0, i32 2, !dbg !7
// CHECK:STDOUT:   %.loc16_3.10 = load i32, ptr %tuple.elem2.loc16_3.2.tuple.elem, align 4, !dbg !7
// CHECK:STDOUT:   %tuple.loc16_3.2 = alloca { i32, i32, i32 }, align 8, !dbg !7
// CHECK:STDOUT:   %tuple.loc16_3.210 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %tuple.loc16_3.2, i32 0, i32 0, !dbg !7
// CHECK:STDOUT:   store i32 %.loc16_3.8, ptr %tuple.loc16_3.210, align 4, !dbg !7
// CHECK:STDOUT:   %tuple.loc16_3.211 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %tuple.loc16_3.2, i32 0, i32 1, !dbg !7
// CHECK:STDOUT:   store i32 %.loc16_3.9, ptr %tuple.loc16_3.211, align 4, !dbg !7
// CHECK:STDOUT:   %tuple.loc16_3.212 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %tuple.loc16_3.2, i32 0, i32 2, !dbg !7
// CHECK:STDOUT:   store i32 %.loc16_3.10, ptr %tuple.loc16_3.212, align 4, !dbg !7
// CHECK:STDOUT:   %struct = alloca { ptr, ptr }, align 8, !dbg !7
// CHECK:STDOUT:   %struct13 = getelementptr inbounds nuw { ptr, ptr }, ptr %struct, i32 0, i32 0, !dbg !7
// CHECK:STDOUT:   store ptr %tuple.loc16_3.1, ptr %struct13, align 8, !dbg !7
// CHECK:STDOUT:   %struct14 = getelementptr inbounds nuw { ptr, ptr }, ptr %struct, i32 0, i32 1, !dbg !7
// CHECK:STDOUT:   store ptr %tuple.loc16_3.2, ptr %struct14, align 8, !dbg !7
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
// CHECK:STDOUT: !3 = !DIFile(filename: "nested_struct_in_place.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !3, line: 15, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 16, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 16, column: 55, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 16, column: 61, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 16, column: 71, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 15, column: 1, scope: !4)
