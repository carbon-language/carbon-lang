// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/generic/call_basic_depth.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/generic/call_basic_depth.carbon

// Builds on `call_basic.carbon`. Checks definitions are emitted with deeper
// call stack, i.e., when functions can be type checked without looking at the
// specific calling context.
fn F[T:! type](_: T) {
}

fn H[T:! Core.Copy](x: T) -> T {
  F(x);
  return x;
}

fn G[T:! Core.Copy & Core.Destroy](x: T) -> T {
  H(x);
  F(x);
  return x;
}

fn M() -> i32 {
  var n: i32 = 0;
  var m: i32;

  F(n);
  m = G(n);
  return m;
}

// CHECK:STDOUT: ; ModuleID = 'call_basic_depth.carbon'
// CHECK:STDOUT: source_filename = "call_basic_depth.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CM.Main() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %n.var = alloca i32, align 4, !dbg !8
// CHECK:STDOUT:   %m.var = alloca i32, align 4, !dbg !9
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %n.var), !dbg !8
// CHECK:STDOUT:   store i32 0, ptr %n.var, align 4, !dbg !8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %m.var), !dbg !9
// CHECK:STDOUT:   store i32 poison, ptr %m.var, align 4, !dbg !9
// CHECK:STDOUT:   %.loc34 = load i32, ptr %n.var, align 4, !dbg !10
// CHECK:STDOUT:   call void @_CF.Main.b88d1103f417c6d4(i32 %.loc34), !dbg !11
// CHECK:STDOUT:   %.loc35_9 = load i32, ptr %n.var, align 4, !dbg !12
// CHECK:STDOUT:   %G.call = call i32 @_CG.Main.bfb441135f07cbe1(i32 %.loc35_9), !dbg !13
// CHECK:STDOUT:   store i32 %G.call, ptr %m.var, align 4, !dbg !14
// CHECK:STDOUT:   %.loc36 = load i32, ptr %m.var, align 4, !dbg !15
// CHECK:STDOUT:   call void @"_COp.7e389eab4a7e5487:core.Destroy.Core"(ptr %m.var), !dbg !9
// CHECK:STDOUT:   call void @"_COp.7e389eab4a7e5487:core.Destroy.Core"(ptr %n.var), !dbg !8
// CHECK:STDOUT:   ret i32 %.loc36, !dbg !16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.7e389eab4a7e5487:core.Destroy.Core"(ptr %self) #0 !dbg !17 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.b88d1103f417c6d4(i32 %_) #0 !dbg !23 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CG.Main.bfb441135f07cbe1(i32 %x) #0 !dbg !27 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc25_6.2.temp = alloca i32, align 4, !dbg !32
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc25_6.2.temp), !dbg !32
// CHECK:STDOUT:   %H.call = call i32 @_CH.Main.64ccbb8e5d9a0b8e(i32 %x), !dbg !32
// CHECK:STDOUT:   store i32 %H.call, ptr %.loc25_6.2.temp, align 4, !dbg !32
// CHECK:STDOUT:   call void @_CF.Main.b88d1103f417c6d4(i32 %x), !dbg !33
// CHECK:STDOUT:   call void @"_COp.7e389eab4a7e5487:core.Destroy.Core"(ptr %.loc25_6.2.temp), !dbg !32
// CHECK:STDOUT:   ret i32 %x, !dbg !34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CH.Main.64ccbb8e5d9a0b8e(i32 %x) #0 !dbg !35 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main.b88d1103f417c6d4(i32 %x), !dbg !38
// CHECK:STDOUT:   ret i32 %x, !dbg !39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @"_COp.7e389eab4a7e5487:core.Destroy.Core", { 0, 2, 1 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 2, 1 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "call_basic_depth.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "M", linkageName: "_CM.Main", scope: null, file: !3, line: 30, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7}
// CHECK:STDOUT: !7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !8 = !DILocation(line: 31, column: 3, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 32, column: 3, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 34, column: 5, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 34, column: 3, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 35, column: 9, scope: !4)
// CHECK:STDOUT: !13 = !DILocation(line: 35, column: 7, scope: !4)
// CHECK:STDOUT: !14 = !DILocation(line: 35, column: 3, scope: !4)
// CHECK:STDOUT: !15 = !DILocation(line: 36, column: 10, scope: !4)
// CHECK:STDOUT: !16 = !DILocation(line: 36, column: 3, scope: !4)
// CHECK:STDOUT: !17 = distinct !DISubprogram(name: "Op", linkageName: "_COp.7e389eab4a7e5487:core.Destroy.Core", scope: null, file: !3, line: 35, type: !18, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !20)
// CHECK:STDOUT: !18 = !DISubroutineType(types: !19)
// CHECK:STDOUT: !19 = !{null, !7}
// CHECK:STDOUT: !20 = !{!21}
// CHECK:STDOUT: !21 = !DILocalVariable(arg: 1, scope: !17, type: !7)
// CHECK:STDOUT: !22 = !DILocation(line: 35, column: 7, scope: !17)
// CHECK:STDOUT: !23 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.b88d1103f417c6d4", scope: null, file: !3, line: 16, type: !18, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !24)
// CHECK:STDOUT: !24 = !{!25}
// CHECK:STDOUT: !25 = !DILocalVariable(arg: 1, scope: !23, type: !7)
// CHECK:STDOUT: !26 = !DILocation(line: 16, column: 1, scope: !23)
// CHECK:STDOUT: !27 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.bfb441135f07cbe1", scope: null, file: !3, line: 24, type: !28, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !30)
// CHECK:STDOUT: !28 = !DISubroutineType(types: !29)
// CHECK:STDOUT: !29 = !{!7, !7}
// CHECK:STDOUT: !30 = !{!31}
// CHECK:STDOUT: !31 = !DILocalVariable(arg: 1, scope: !27, type: !7)
// CHECK:STDOUT: !32 = !DILocation(line: 25, column: 3, scope: !27)
// CHECK:STDOUT: !33 = !DILocation(line: 26, column: 3, scope: !27)
// CHECK:STDOUT: !34 = !DILocation(line: 27, column: 3, scope: !27)
// CHECK:STDOUT: !35 = distinct !DISubprogram(name: "H", linkageName: "_CH.Main.64ccbb8e5d9a0b8e", scope: null, file: !3, line: 19, type: !28, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !36)
// CHECK:STDOUT: !36 = !{!37}
// CHECK:STDOUT: !37 = !DILocalVariable(arg: 1, scope: !35, type: !7)
// CHECK:STDOUT: !38 = !DILocation(line: 20, column: 3, scope: !35)
// CHECK:STDOUT: !39 = !DILocation(line: 21, column: 3, scope: !35)
