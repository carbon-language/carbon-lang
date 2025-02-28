// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/struct/two_entries.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/struct/two_entries.carbon

fn Run() -> i32 {
  var x: {.a: i32, .b: i32} = {.a = 1, .b = 2};
  var y: {.a: i32, .b: i32} = x;
  return 0;
}

// CHECK:STDOUT: ; ModuleID = 'two_entries.carbon'
// CHECK:STDOUT: source_filename = "two_entries.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @struct.loc12_3.2 = internal constant { i32, i32 } { i32 1, i32 2 }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @main() !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x.var = alloca { i32, i32 }, align 8, !dbg !7
// CHECK:STDOUT:   %y.var = alloca { i32, i32 }, align 8, !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 8, ptr %x.var), !dbg !7
// CHECK:STDOUT:   %.loc12_46.3.a = getelementptr inbounds nuw { i32, i32 }, ptr %x.var, i32 0, i32 0, !dbg !8
// CHECK:STDOUT:   %.loc12_46.6.b = getelementptr inbounds nuw { i32, i32 }, ptr %x.var, i32 0, i32 1, !dbg !8
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %x.var, ptr align 4 @struct.loc12_3.2, i64 8, i1 false), !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 8, ptr %y.var), !dbg !7
// CHECK:STDOUT:   %.loc13_31.1.a = getelementptr inbounds nuw { i32, i32 }, ptr %x.var, i32 0, i32 0, !dbg !9
// CHECK:STDOUT:   %.loc13_31.2 = load i32, ptr %.loc13_31.1.a, align 4, !dbg !9
// CHECK:STDOUT:   %.loc13_31.3.a = getelementptr inbounds nuw { i32, i32 }, ptr %y.var, i32 0, i32 0, !dbg !9
// CHECK:STDOUT:   store i32 %.loc13_31.2, ptr %.loc13_31.3.a, align 4, !dbg !9
// CHECK:STDOUT:   %.loc13_31.5.b = getelementptr inbounds nuw { i32, i32 }, ptr %x.var, i32 0, i32 1, !dbg !9
// CHECK:STDOUT:   %.loc13_31.6 = load i32, ptr %.loc13_31.5.b, align 4, !dbg !9
// CHECK:STDOUT:   %.loc13_31.7.b = getelementptr inbounds nuw { i32, i32 }, ptr %y.var, i32 0, i32 1, !dbg !9
// CHECK:STDOUT:   store i32 %.loc13_31.6, ptr %.loc13_31.7.b, align 4, !dbg !9
// CHECK:STDOUT:   ret i32 0, !dbg !10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(i64 immarg, ptr captures(none)) #0
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder i32 1, { 0, 1, 3, 2, 4, 5 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "two_entries.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !3, line: 11, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 12, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 12, column: 31, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 13, column: 31, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 14, column: 3, scope: !4)
