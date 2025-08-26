// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/builtins/maybe_unformed.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/builtins/maybe_unformed.carbon

fn MaybeUnformed(t: type) -> type = "maybe_unformed.make_type";

fn Take(x: MaybeUnformed(i32)) {}
fn Make() -> MaybeUnformed(i32);

fn Pass() {
  // This is passed by pointer even though `i32` would generally be passed in
  // registers.
  Take(Make());
}

// CHECK:STDOUT: ; ModuleID = 'maybe_unformed.carbon'
// CHECK:STDOUT: source_filename = "maybe_unformed.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CTake.Main(ptr %x) !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CMake.Main(ptr sret(i32))
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CPass.Main() !dbg !8 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc21_13.1.temp = alloca i32, align 4, !dbg !9
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc21_13.1.temp), !dbg !9
// CHECK:STDOUT:   call void @_CMake.Main(ptr %.loc21_13.1.temp), !dbg !9
// CHECK:STDOUT:   call void @_CTake.Main(ptr %.loc21_13.1.temp), !dbg !10
// CHECK:STDOUT:   ret void, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #0
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "maybe_unformed.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Take", linkageName: "_CTake.Main", scope: null, file: !3, line: 15, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 15, column: 1, scope: !4)
// CHECK:STDOUT: !8 = distinct !DISubprogram(name: "Pass", linkageName: "_CPass.Main", scope: null, file: !3, line: 18, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !9 = !DILocation(line: 21, column: 8, scope: !8)
// CHECK:STDOUT: !10 = !DILocation(line: 21, column: 3, scope: !8)
// CHECK:STDOUT: !11 = !DILocation(line: 18, column: 1, scope: !8)
