// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/generic/call_basic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/generic/call_basic.carbon

class C {
  fn Cfn[self: Self, T:! type](x: T) {
  }
}

fn F[T:! type](x: T) {
}

fn H[T:! type](x: T) -> T {
  return x;
}

fn G[T:! type](x: T) -> T {
  // small stress test for single depth of calls with change in types
  H(x);
  H(T);
  H(i32);
  H(G(x));
  G(T);

  var a: f64;
  G(a);
  H(a);

  var c: C;
  c.Cfn(x);

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
// CHECK:STDOUT: %type = type {}
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
// CHECK:STDOUT:   %.loc47 = load i32, ptr %n.var, align 4, !dbg !9
// CHECK:STDOUT:   call void @_CF.Main.b88d1103f417c6d4(i32 %.loc47), !dbg !10
// CHECK:STDOUT:   %.loc48 = load i32, ptr %n.var, align 4, !dbg !11
// CHECK:STDOUT:   %G.call.loc48 = call i32 @_CG.Main.b88d1103f417c6d4(i32 %.loc48), !dbg !12
// CHECK:STDOUT:   store i32 %G.call.loc48, ptr %m.var, align 4, !dbg !13
// CHECK:STDOUT:   %.loc49 = load double, ptr %p.var, align 8, !dbg !14
// CHECK:STDOUT:   call void @_CF.Main.66be507887ceee78(double %.loc49), !dbg !15
// CHECK:STDOUT:   %.loc50 = load double, ptr %p.var, align 8, !dbg !16
// CHECK:STDOUT:   %G.call.loc50 = call double @_CG.Main.66be507887ceee78(double %.loc50), !dbg !17
// CHECK:STDOUT:   store double %G.call.loc50, ptr %q.var, align 8, !dbg !18
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
// CHECK:STDOUT:   %a.var = alloca double, align 8, !dbg !23
// CHECK:STDOUT:   %c.var = alloca {}, align 8, !dbg !24
// CHECK:STDOUT:   %H.call.loc25 = call i32 @_CH.Main.b88d1103f417c6d4(i32 %x), !dbg !24
// CHECK:STDOUT:   %H.call.loc26 = call %type @_CH.Main.5754c7a55c7cbe4a(%type zeroinitializer), !dbg !25
// CHECK:STDOUT:   %H.call.loc27 = call %type @_CH.Main.5754c7a55c7cbe4a(%type zeroinitializer), !dbg !26
// CHECK:STDOUT:   %G.call.loc28 = call i32 @_CG.Main.b88d1103f417c6d4(i32 %x), !dbg !27
// CHECK:STDOUT:   %H.call.loc28 = call i32 @_CH.Main.b88d1103f417c6d4(i32 %G.call.loc28), !dbg !28
// CHECK:STDOUT:   %G.call.loc29 = call %type @_CG.Main.5754c7a55c7cbe4a(%type zeroinitializer), !dbg !29
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 8, ptr %a.var), !dbg !23
// CHECK:STDOUT:   %.loc32 = load double, ptr %a.var, align 8, !dbg !30
// CHECK:STDOUT:   %G.call.loc32 = call double @_CG.Main.66be507887ceee78(double %.loc32), !dbg !31
// CHECK:STDOUT:   %.loc33 = load double, ptr %a.var, align 8, !dbg !32
// CHECK:STDOUT:   %H.call.loc33 = call double @_CH.Main.66be507887ceee78(double %.loc33), !dbg !33
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 0, ptr %c.var), !dbg !24
// CHECK:STDOUT:   call void @_CCfn.C.Main.b88d1103f417c6d4(ptr %c.var, i32 %x), !dbg !34
// CHECK:STDOUT:   ret i32 %x, !dbg !35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CF.Main.66be507887ceee78(double %x) !dbg !36 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define double @_CG.Main.66be507887ceee78(double %x) !dbg !38 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca double, align 8, !dbg !39
// CHECK:STDOUT:   %c.var = alloca {}, align 8, !dbg !40
// CHECK:STDOUT:   %H.call.loc25 = call double @_CH.Main.66be507887ceee78(double %x), !dbg !40
// CHECK:STDOUT:   %H.call.loc26 = call %type @_CH.Main.5754c7a55c7cbe4a(%type zeroinitializer), !dbg !41
// CHECK:STDOUT:   %H.call.loc27 = call %type @_CH.Main.5754c7a55c7cbe4a(%type zeroinitializer), !dbg !42
// CHECK:STDOUT:   %G.call.loc28 = call double @_CG.Main.66be507887ceee78(double %x), !dbg !43
// CHECK:STDOUT:   %H.call.loc28 = call double @_CH.Main.66be507887ceee78(double %G.call.loc28), !dbg !44
// CHECK:STDOUT:   %G.call.loc29 = call %type @_CG.Main.5754c7a55c7cbe4a(%type zeroinitializer), !dbg !45
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 8, ptr %a.var), !dbg !39
// CHECK:STDOUT:   %.loc32 = load double, ptr %a.var, align 8, !dbg !46
// CHECK:STDOUT:   %G.call.loc32 = call double @_CG.Main.66be507887ceee78(double %.loc32), !dbg !47
// CHECK:STDOUT:   %.loc33 = load double, ptr %a.var, align 8, !dbg !48
// CHECK:STDOUT:   %H.call.loc33 = call double @_CH.Main.66be507887ceee78(double %.loc33), !dbg !49
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 0, ptr %c.var), !dbg !40
// CHECK:STDOUT:   call void @_CCfn.C.Main.66be507887ceee78(ptr %c.var, double %x), !dbg !50
// CHECK:STDOUT:   ret double %x, !dbg !51
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_CH.Main.b88d1103f417c6d4(i32 %x) !dbg !52 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %x, !dbg !53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define %type @_CH.Main.5754c7a55c7cbe4a(%type %x) !dbg !54 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret %type %x, !dbg !55
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define %type @_CG.Main.5754c7a55c7cbe4a(%type %x) !dbg !56 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca double, align 8, !dbg !57
// CHECK:STDOUT:   %c.var = alloca {}, align 8, !dbg !58
// CHECK:STDOUT:   %H.call.loc25 = call %type @_CH.Main.5754c7a55c7cbe4a(%type %x), !dbg !58
// CHECK:STDOUT:   %H.call.loc26 = call %type @_CH.Main.5754c7a55c7cbe4a(%type zeroinitializer), !dbg !59
// CHECK:STDOUT:   %H.call.loc27 = call %type @_CH.Main.5754c7a55c7cbe4a(%type zeroinitializer), !dbg !60
// CHECK:STDOUT:   %G.call.loc28 = call %type @_CG.Main.5754c7a55c7cbe4a(%type %x), !dbg !61
// CHECK:STDOUT:   %H.call.loc28 = call %type @_CH.Main.5754c7a55c7cbe4a(%type %G.call.loc28), !dbg !62
// CHECK:STDOUT:   %G.call.loc29 = call %type @_CG.Main.5754c7a55c7cbe4a(%type zeroinitializer), !dbg !63
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 8, ptr %a.var), !dbg !57
// CHECK:STDOUT:   %.loc32 = load double, ptr %a.var, align 8, !dbg !64
// CHECK:STDOUT:   %G.call.loc32 = call double @_CG.Main.66be507887ceee78(double %.loc32), !dbg !65
// CHECK:STDOUT:   %.loc33 = load double, ptr %a.var, align 8, !dbg !66
// CHECK:STDOUT:   %H.call.loc33 = call double @_CH.Main.66be507887ceee78(double %.loc33), !dbg !67
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 0, ptr %c.var), !dbg !58
// CHECK:STDOUT:   call void @_CCfn.C.Main.5754c7a55c7cbe4a(ptr %c.var, %type %x), !dbg !68
// CHECK:STDOUT:   ret %type %x, !dbg !69
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define double @_CH.Main.66be507887ceee78(double %x) !dbg !70 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret double %x, !dbg !71
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CCfn.C.Main.b88d1103f417c6d4(ptr %self, i32 %x) !dbg !72 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !73
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CCfn.C.Main.66be507887ceee78(ptr %self, double %x) !dbg !74 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !75
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CCfn.C.Main.5754c7a55c7cbe4a(ptr %self, %type %x) !dbg !76 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !77
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 1, 2, 3, 4, 5, 9, 8, 7, 6 }
// CHECK:STDOUT: uselistorder ptr @_CG.Main.66be507887ceee78, { 0, 1, 2, 4, 3 }
// CHECK:STDOUT: uselistorder ptr @_CH.Main.b88d1103f417c6d4, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CH.Main.5754c7a55c7cbe4a, { 0, 1, 2, 3, 7, 6, 5, 4 }
// CHECK:STDOUT: uselistorder ptr @_CG.Main.5754c7a55c7cbe4a, { 0, 1, 3, 2 }
// CHECK:STDOUT: uselistorder ptr @_CH.Main.66be507887ceee78, { 4, 3, 2, 1, 0 }
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
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "M", linkageName: "_CM.Main", scope: null, file: !3, line: 41, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 42, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 44, column: 3, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 47, column: 5, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 47, column: 3, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 48, column: 9, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 48, column: 7, scope: !4)
// CHECK:STDOUT: !13 = !DILocation(line: 48, column: 3, scope: !4)
// CHECK:STDOUT: !14 = !DILocation(line: 49, column: 5, scope: !4)
// CHECK:STDOUT: !15 = !DILocation(line: 49, column: 3, scope: !4)
// CHECK:STDOUT: !16 = !DILocation(line: 50, column: 9, scope: !4)
// CHECK:STDOUT: !17 = !DILocation(line: 50, column: 7, scope: !4)
// CHECK:STDOUT: !18 = !DILocation(line: 50, column: 3, scope: !4)
// CHECK:STDOUT: !19 = !DILocation(line: 41, column: 1, scope: !4)
// CHECK:STDOUT: !20 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.b88d1103f417c6d4", scope: null, file: !3, line: 16, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !21 = !DILocation(line: 16, column: 1, scope: !20)
// CHECK:STDOUT: !22 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.b88d1103f417c6d4", scope: null, file: !3, line: 23, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !23 = !DILocation(line: 31, column: 3, scope: !22)
// CHECK:STDOUT: !24 = !DILocation(line: 25, column: 3, scope: !22)
// CHECK:STDOUT: !25 = !DILocation(line: 26, column: 3, scope: !22)
// CHECK:STDOUT: !26 = !DILocation(line: 27, column: 3, scope: !22)
// CHECK:STDOUT: !27 = !DILocation(line: 28, column: 5, scope: !22)
// CHECK:STDOUT: !28 = !DILocation(line: 28, column: 3, scope: !22)
// CHECK:STDOUT: !29 = !DILocation(line: 29, column: 3, scope: !22)
// CHECK:STDOUT: !30 = !DILocation(line: 32, column: 5, scope: !22)
// CHECK:STDOUT: !31 = !DILocation(line: 32, column: 3, scope: !22)
// CHECK:STDOUT: !32 = !DILocation(line: 33, column: 5, scope: !22)
// CHECK:STDOUT: !33 = !DILocation(line: 33, column: 3, scope: !22)
// CHECK:STDOUT: !34 = !DILocation(line: 36, column: 3, scope: !22)
// CHECK:STDOUT: !35 = !DILocation(line: 38, column: 3, scope: !22)
// CHECK:STDOUT: !36 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.66be507887ceee78", scope: null, file: !3, line: 16, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !37 = !DILocation(line: 16, column: 1, scope: !36)
// CHECK:STDOUT: !38 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.66be507887ceee78", scope: null, file: !3, line: 23, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !39 = !DILocation(line: 31, column: 3, scope: !38)
// CHECK:STDOUT: !40 = !DILocation(line: 25, column: 3, scope: !38)
// CHECK:STDOUT: !41 = !DILocation(line: 26, column: 3, scope: !38)
// CHECK:STDOUT: !42 = !DILocation(line: 27, column: 3, scope: !38)
// CHECK:STDOUT: !43 = !DILocation(line: 28, column: 5, scope: !38)
// CHECK:STDOUT: !44 = !DILocation(line: 28, column: 3, scope: !38)
// CHECK:STDOUT: !45 = !DILocation(line: 29, column: 3, scope: !38)
// CHECK:STDOUT: !46 = !DILocation(line: 32, column: 5, scope: !38)
// CHECK:STDOUT: !47 = !DILocation(line: 32, column: 3, scope: !38)
// CHECK:STDOUT: !48 = !DILocation(line: 33, column: 5, scope: !38)
// CHECK:STDOUT: !49 = !DILocation(line: 33, column: 3, scope: !38)
// CHECK:STDOUT: !50 = !DILocation(line: 36, column: 3, scope: !38)
// CHECK:STDOUT: !51 = !DILocation(line: 38, column: 3, scope: !38)
// CHECK:STDOUT: !52 = distinct !DISubprogram(name: "H", linkageName: "_CH.Main.b88d1103f417c6d4", scope: null, file: !3, line: 19, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !53 = !DILocation(line: 20, column: 3, scope: !52)
// CHECK:STDOUT: !54 = distinct !DISubprogram(name: "H", linkageName: "_CH.Main.5754c7a55c7cbe4a", scope: null, file: !3, line: 19, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !55 = !DILocation(line: 20, column: 3, scope: !54)
// CHECK:STDOUT: !56 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.5754c7a55c7cbe4a", scope: null, file: !3, line: 23, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !57 = !DILocation(line: 31, column: 3, scope: !56)
// CHECK:STDOUT: !58 = !DILocation(line: 25, column: 3, scope: !56)
// CHECK:STDOUT: !59 = !DILocation(line: 26, column: 3, scope: !56)
// CHECK:STDOUT: !60 = !DILocation(line: 27, column: 3, scope: !56)
// CHECK:STDOUT: !61 = !DILocation(line: 28, column: 5, scope: !56)
// CHECK:STDOUT: !62 = !DILocation(line: 28, column: 3, scope: !56)
// CHECK:STDOUT: !63 = !DILocation(line: 29, column: 3, scope: !56)
// CHECK:STDOUT: !64 = !DILocation(line: 32, column: 5, scope: !56)
// CHECK:STDOUT: !65 = !DILocation(line: 32, column: 3, scope: !56)
// CHECK:STDOUT: !66 = !DILocation(line: 33, column: 5, scope: !56)
// CHECK:STDOUT: !67 = !DILocation(line: 33, column: 3, scope: !56)
// CHECK:STDOUT: !68 = !DILocation(line: 36, column: 3, scope: !56)
// CHECK:STDOUT: !69 = !DILocation(line: 38, column: 3, scope: !56)
// CHECK:STDOUT: !70 = distinct !DISubprogram(name: "H", linkageName: "_CH.Main.66be507887ceee78", scope: null, file: !3, line: 19, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !71 = !DILocation(line: 20, column: 3, scope: !70)
// CHECK:STDOUT: !72 = distinct !DISubprogram(name: "Cfn", linkageName: "_CCfn.C.Main.b88d1103f417c6d4", scope: null, file: !3, line: 12, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !73 = !DILocation(line: 12, column: 3, scope: !72)
// CHECK:STDOUT: !74 = distinct !DISubprogram(name: "Cfn", linkageName: "_CCfn.C.Main.66be507887ceee78", scope: null, file: !3, line: 12, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !75 = !DILocation(line: 12, column: 3, scope: !74)
// CHECK:STDOUT: !76 = distinct !DISubprogram(name: "Cfn", linkageName: "_CCfn.C.Main.5754c7a55c7cbe4a", scope: null, file: !3, line: 12, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !77 = !DILocation(line: 12, column: 3, scope: !76)
