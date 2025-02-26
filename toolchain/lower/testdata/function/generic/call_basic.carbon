// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/generic/call_basic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/generic/call_basic.carbon

fn F[T:! type](x: T) {
}

fn H[T:! type](x: T) {
}

fn G[T:! type](x: T) -> T {
  // TODO: the call below is crashing because proper type resolution to
  // use the G specific, not the G generic is not done yet.
  // H(T);
  return x;
}


fn M() {
  var n: i32 = 0;
  var m: i32;
  var p: f64 = 1.0;
  var q: f64;

  F(n);
  m = G(n);
  F(p);
  q = G(p);
}

// CHECK:STDOUT: ; ModuleID = 'call_basic.carbon'
// CHECK:STDOUT: source_filename = "call_basic.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CM.Main() !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %n.var = alloca i32, align 4, !dbg !7
// CHECK:STDOUT:   %m.var = alloca i32, align 4, !dbg !7
// CHECK:STDOUT:   %p.var = alloca double, align 8, !dbg !7
// CHECK:STDOUT:   %q.var = alloca double, align 8, !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 4, ptr %n.var), !dbg !7
// CHECK:STDOUT:   store i32 0, ptr %n.var, align 4, !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 4, ptr %m.var), !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 8, ptr %p.var), !dbg !7
// CHECK:STDOUT:   store double 1.000000e+00, ptr %p.var, align 8, !dbg !8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 8, ptr %q.var), !dbg !7
// CHECK:STDOUT:   %.loc31 = load i32, ptr %n.var, align 4, !dbg !9
// CHECK:STDOUT:   call void @_CF.Main.b88d1103f417c6d4(i32 %.loc31), !dbg !10
// CHECK:STDOUT:   %.loc32 = load i32, ptr %n.var, align 4, !dbg !11
// CHECK:STDOUT:   %G.call.loc32 = call i32 @_CG.Main.b88d1103f417c6d4(i32 %.loc32), !dbg !12
// CHECK:STDOUT:   store i32 %G.call.loc32, ptr %m.var, align 4, !dbg !13
// CHECK:STDOUT:   %.loc33 = load double, ptr %p.var, align 8, !dbg !14
// CHECK:STDOUT:   call void @_CF.Main.66be507887ceee78(double %.loc33), !dbg !15
// CHECK:STDOUT:   %.loc34 = load double, ptr %p.var, align 8, !dbg !16
// CHECK:STDOUT:   %G.call.loc34 = call double @_CG.Main.66be507887ceee78(double %.loc34), !dbg !17
// CHECK:STDOUT:   store double %G.call.loc34, ptr %q.var, align 8, !dbg !18
// CHECK:STDOUT:   ret void, !dbg !19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(i64 immarg, ptr captures(none)) #0
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CF.Main.b88d1103f417c6d4(i32 %x) !dbg !20 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_CG.Main.b88d1103f417c6d4(i32 %x) !dbg !22 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %x, !dbg !23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CF.Main.66be507887ceee78(double %x) !dbg !24 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define double @_CG.Main.66be507887ceee78(double %x) !dbg !26 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret double %x, !dbg !27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "call_basic.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "M", linkageName: "_CM.Main", scope: null, file: !3, line: 25, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 26, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 28, column: 3, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 31, column: 5, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 31, column: 3, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 32, column: 9, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 32, column: 7, scope: !4)
// CHECK:STDOUT: !13 = !DILocation(line: 32, column: 3, scope: !4)
// CHECK:STDOUT: !14 = !DILocation(line: 33, column: 5, scope: !4)
// CHECK:STDOUT: !15 = !DILocation(line: 33, column: 3, scope: !4)
// CHECK:STDOUT: !16 = !DILocation(line: 34, column: 9, scope: !4)
// CHECK:STDOUT: !17 = !DILocation(line: 34, column: 7, scope: !4)
// CHECK:STDOUT: !18 = !DILocation(line: 34, column: 3, scope: !4)
// CHECK:STDOUT: !19 = !DILocation(line: 25, column: 1, scope: !4)
// CHECK:STDOUT: !20 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.b88d1103f417c6d4", scope: null, file: !3, line: 11, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !21 = !DILocation(line: 11, column: 1, scope: !20)
// CHECK:STDOUT: !22 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.b88d1103f417c6d4", scope: null, file: !3, line: 17, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !23 = !DILocation(line: 21, column: 3, scope: !22)
// CHECK:STDOUT: !24 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.66be507887ceee78", scope: null, file: !3, line: 11, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !25 = !DILocation(line: 11, column: 1, scope: !24)
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.66be507887ceee78", scope: null, file: !3, line: 17, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !27 = !DILocation(line: 21, column: 3, scope: !26)
