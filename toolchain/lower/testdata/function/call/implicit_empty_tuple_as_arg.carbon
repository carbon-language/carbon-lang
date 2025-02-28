// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/call/implicit_empty_tuple_as_arg.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/call/implicit_empty_tuple_as_arg.carbon


fn Foo() {}

fn Bar(a: ()) -> () { return a; }

fn Main() {
  var x: () = Bar(Foo());
}

// CHECK:STDOUT: ; ModuleID = 'implicit_empty_tuple_as_arg.carbon'
// CHECK:STDOUT: source_filename = "implicit_empty_tuple_as_arg.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CFoo.Main() !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CBar.Main() !dbg !8 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CMain.Main() !dbg !10 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x.var = alloca {}, align 8, !dbg !11
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 0, ptr %x.var), !dbg !11
// CHECK:STDOUT:   call void @_CFoo.Main(), !dbg !12
// CHECK:STDOUT:   %.loc17_23.1.temp = alloca {}, align 8, !dbg !12
// CHECK:STDOUT:   call void @_CBar.Main(), !dbg !13
// CHECK:STDOUT:   ret void, !dbg !14
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
// CHECK:STDOUT: !3 = !DIFile(filename: "implicit_empty_tuple_as_arg.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Foo", linkageName: "_CFoo.Main", scope: null, file: !3, line: 12, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 12, column: 1, scope: !4)
// CHECK:STDOUT: !8 = distinct !DISubprogram(name: "Bar", linkageName: "_CBar.Main", scope: null, file: !3, line: 14, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !9 = !DILocation(line: 14, column: 23, scope: !8)
// CHECK:STDOUT: !10 = distinct !DISubprogram(name: "Main", linkageName: "_CMain.Main", scope: null, file: !3, line: 16, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !11 = !DILocation(line: 17, column: 3, scope: !10)
// CHECK:STDOUT: !12 = !DILocation(line: 17, column: 19, scope: !10)
// CHECK:STDOUT: !13 = !DILocation(line: 17, column: 15, scope: !10)
// CHECK:STDOUT: !14 = !DILocation(line: 16, column: 1, scope: !10)
