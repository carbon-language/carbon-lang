// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/call/var_param.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/call/var_param.carbon

fn DoNothing(a: i32) {}

fn Main() {
  var a: i32 = 0;
  DoNothing(a);
}

// CHECK:STDOUT: ; ModuleID = 'var_param.carbon'
// CHECK:STDOUT: source_filename = "var_param.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CDoNothing.Main(i32 %a) !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CMain.Main() !dbg !8 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca i32, align 4, !dbg !9
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 4, ptr %a.var), !dbg !9
// CHECK:STDOUT:   store i32 0, ptr %a.var, align 4, !dbg !9
// CHECK:STDOUT:   %.loc15 = load i32, ptr %a.var, align 4, !dbg !10
// CHECK:STDOUT:   call void @_CDoNothing.Main(i32 %.loc15), !dbg !11
// CHECK:STDOUT:   ret void, !dbg !12
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
// CHECK:STDOUT: !3 = !DIFile(filename: "var_param.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "DoNothing", linkageName: "_CDoNothing.Main", scope: null, file: !3, line: 11, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 11, column: 1, scope: !4)
// CHECK:STDOUT: !8 = distinct !DISubprogram(name: "Main", linkageName: "_CMain.Main", scope: null, file: !3, line: 13, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !9 = !DILocation(line: 14, column: 3, scope: !8)
// CHECK:STDOUT: !10 = !DILocation(line: 15, column: 13, scope: !8)
// CHECK:STDOUT: !11 = !DILocation(line: 15, column: 3, scope: !8)
// CHECK:STDOUT: !12 = !DILocation(line: 13, column: 1, scope: !8)
