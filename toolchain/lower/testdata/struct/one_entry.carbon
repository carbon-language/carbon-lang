// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/struct/one_entry.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/struct/one_entry.carbon

fn Run() -> i32 {
  var x: {.a: i32} = {.a = 4};
  var _: {.a: i32} = x;
  return 0;
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'one_entry.carbon'
// CHECK:STDOUT: source_filename = "one_entry.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @main() #0 !dbg !25 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x.var = alloca { i32 }, align 4, !dbg !28
// CHECK:STDOUT:   %_.var = alloca { i32 }, align 4, !dbg !30
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %x.var), !dbg !28
// CHECK:STDOUT:   store { i32 } { i32 4 }, ptr %x.var, align 4, !dbg !28
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !30
// CHECK:STDOUT:   %.loc15_22.1.a = getelementptr inbounds nuw { i32 }, ptr %x.var, i32 0, i32 0, !dbg !31
// CHECK:STDOUT:   %.loc15_22.2 = load i32, ptr %.loc15_22.1.a, align 4, !dbg !31
// CHECK:STDOUT:   %.loc15_22.3.struct.init = insertvalue { i32 } poison, i32 %.loc15_22.2, 0, !dbg !31
// CHECK:STDOUT:   store { i32 } %.loc15_22.3.struct.init, ptr %_.var, align 4, !dbg !30
// CHECK:STDOUT:   ret i32 0, !dbg !33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
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
// CHECK:STDOUT: !20 = !DIFile(filename: "one_entry.carbon", directory: "")
// CHECK:STDOUT: !21 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !20, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !22 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !23 = !{!22}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !20, line: 13, type: !24, spFlags: DISPFlagDefinition, unit: !21)
// CHECK:STDOUT: !28 = !DILocation(line: 14, column: 3, scope: !25)
// CHECK:STDOUT: !30 = !DILocation(line: 15, column: 3, scope: !25)
// CHECK:STDOUT: !31 = !DILocation(line: 15, column: 22, scope: !25)
// CHECK:STDOUT: !33 = !DILocation(line: 16, column: 3, scope: !25)
// CHECK:STDOUT:
