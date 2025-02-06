// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/pointer/pointer_to_pointer.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/pointer/pointer_to_pointer.carbon

fn F(p: i32**) -> i32 {
  var a: i32** = p;
  var b: i32* = *p;
  var c: i32** = &b;
  return **c;
}

// CHECK:STDOUT: ; ModuleID = 'pointer_to_pointer.carbon'
// CHECK:STDOUT: source_filename = "pointer_to_pointer.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_CF.Main(ptr %p) !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca ptr, align 8, !dbg !7
// CHECK:STDOUT:   %b.var = alloca ptr, align 8, !dbg !7
// CHECK:STDOUT:   %c.var = alloca ptr, align 8, !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 8, ptr %a.var), !dbg !7
// CHECK:STDOUT:   store ptr %p, ptr %a.var, align 8, !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 8, ptr %b.var), !dbg !7
// CHECK:STDOUT:   %.loc13_17.2 = load ptr, ptr %p, align 8, !dbg !8
// CHECK:STDOUT:   store ptr %.loc13_17.2, ptr %b.var, align 8, !dbg !9
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 8, ptr %c.var), !dbg !7
// CHECK:STDOUT:   store ptr %b.var, ptr %c.var, align 8, !dbg !10
// CHECK:STDOUT:   %.loc15_12 = load ptr, ptr %c.var, align 8, !dbg !11
// CHECK:STDOUT:   %.loc15_11.2 = load ptr, ptr %.loc15_12, align 8, !dbg !12
// CHECK:STDOUT:   %.loc15_10.2 = load i32, ptr %.loc15_11.2, align 4, !dbg !13
// CHECK:STDOUT:   ret i32 %.loc15_10.2, !dbg !14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(i64 immarg, ptr captures(none)) #0
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "pointer_to_pointer.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !3, line: 11, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 12, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 13, column: 17, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 13, column: 3, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 14, column: 3, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 15, column: 12, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 15, column: 11, scope: !4)
// CHECK:STDOUT: !13 = !DILocation(line: 15, column: 10, scope: !4)
// CHECK:STDOUT: !14 = !DILocation(line: 15, column: 3, scope: !4)
