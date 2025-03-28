// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/generic/call_basic_depth.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/generic/call_basic_depth.carbon

fn F[T:! type](x: T) {
}

fn H[T:! type](x: T) -> T {
  // This call crashes because check/eval does not emit a definition for H.
  // In G below, the call to H can be typechecked without needing a definition.
  // TODO: this needs to be resolved in check/eval.
  // F(x);
  return x;
}

fn G[T:! type](x: T) -> T {
  H(x);
  F(x);
  return x;
}

fn M() {
  var n: i32 = 0;
  var m: i32;

  F(n);
  m = G(n);
  // Adding something like this makes check/eval emit a definition for H, hence no crash.
  // m = H(n);
}

// CHECK:STDOUT: ; ModuleID = 'call_basic_depth.carbon'
// CHECK:STDOUT: source_filename = "call_basic_depth.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CM.Main() !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %n.var = alloca i32, align 4, !dbg !7
// CHECK:STDOUT:   %m.var = alloca i32, align 4, !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 4, ptr %n.var), !dbg !7
// CHECK:STDOUT:   store i32 0, ptr %n.var, align 4, !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 4, ptr %m.var), !dbg !7
// CHECK:STDOUT:   %.loc32 = load i32, ptr %n.var, align 4, !dbg !8
// CHECK:STDOUT:   call void @_CF.Main.b88d1103f417c6d4(i32 %.loc32), !dbg !9
// CHECK:STDOUT:   %.loc33 = load i32, ptr %n.var, align 4, !dbg !10
// CHECK:STDOUT:   %G.call = call i32 @_CG.Main.b88d1103f417c6d4(i32 %.loc33), !dbg !11
// CHECK:STDOUT:   store i32 %G.call, ptr %m.var, align 4, !dbg !12
// CHECK:STDOUT:   ret void, !dbg !13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(i64 immarg, ptr captures(none)) #0
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CF.Main.b88d1103f417c6d4(i32 %x) !dbg !14 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_CG.Main.b88d1103f417c6d4(i32 %x) !dbg !16 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %H.call = call i32 @_CH.Main.b88d1103f417c6d4(i32 %x), !dbg !17
// CHECK:STDOUT:   call void @_CF.Main.b88d1103f417c6d4(i32 %x), !dbg !18
// CHECK:STDOUT:   ret i32 %x, !dbg !19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_CH.Main.b88d1103f417c6d4(i32 %x) !dbg !20 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %x, !dbg !21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "call_basic_depth.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "M", linkageName: "_CM.Main", scope: null, file: !3, line: 28, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 29, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 32, column: 5, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 32, column: 3, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 33, column: 9, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 33, column: 7, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 33, column: 3, scope: !4)
// CHECK:STDOUT: !13 = !DILocation(line: 28, column: 1, scope: !4)
// CHECK:STDOUT: !14 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.b88d1103f417c6d4", scope: null, file: !3, line: 11, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !15 = !DILocation(line: 11, column: 1, scope: !14)
// CHECK:STDOUT: !16 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.b88d1103f417c6d4", scope: null, file: !3, line: 22, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !17 = !DILocation(line: 23, column: 3, scope: !16)
// CHECK:STDOUT: !18 = !DILocation(line: 24, column: 3, scope: !16)
// CHECK:STDOUT: !19 = !DILocation(line: 25, column: 3, scope: !16)
// CHECK:STDOUT: !20 = distinct !DISubprogram(name: "H", linkageName: "_CH.Main.b88d1103f417c6d4", scope: null, file: !3, line: 14, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !21 = !DILocation(line: 19, column: 3, scope: !20)
