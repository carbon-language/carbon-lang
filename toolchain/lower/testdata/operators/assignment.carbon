// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/operators/assignment.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/operators/assignment.carbon

fn Main() {
  var a: i32 = 12;
  a = 9;
  var b: (i32, i32);
  b = (1, 2);
}

// CHECK:STDOUT: ; ModuleID = 'assignment.carbon'
// CHECK:STDOUT: source_filename = "assignment.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @tuple.loc15_5 = internal constant { i32, i32 } { i32 1, i32 2 }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CMain.Main() !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca i32, align 4, !dbg !7
// CHECK:STDOUT:   %b.var = alloca { i32, i32 }, align 8, !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 4, ptr %a.var), !dbg !7
// CHECK:STDOUT:   store i32 12, ptr %a.var, align 4, !dbg !7
// CHECK:STDOUT:   store i32 9, ptr %a.var, align 4, !dbg !8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 8, ptr %b.var), !dbg !7
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %b.var, i32 0, i32 0, !dbg !9
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %b.var, i32 0, i32 1, !dbg !9
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %b.var, ptr align 4 @tuple.loc15_5, i64 8, i1 false), !dbg !10
// CHECK:STDOUT:   ret void, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(i64 immarg, ptr captures(none)) #0
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
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
// CHECK:STDOUT: !3 = !DIFile(filename: "assignment.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Main", linkageName: "_CMain.Main", scope: null, file: !3, line: 11, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 12, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 13, column: 3, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 15, column: 7, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 15, column: 3, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 11, column: 1, scope: !4)
