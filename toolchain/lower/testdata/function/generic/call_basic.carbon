// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/generic/call_basic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/generic/call_basic.carbon

class C {}

fn F[T:! type](x: T) {
}

fn H[T:! type](x: T) -> T {
  return x;
}

// Simple stress test for single depth of calls with change in types.
// Check definitions are emitted for each specific.
fn G[T:! type](x: T) -> T {
  H(x);
  H(T);
  H(i32);
  H(G(x));

  // Calls to H with types not dependent on T.
  var var_f64: f64;
  H(var_f64);
  var ptr_i32 : i32*;
  H(ptr_i32);
  var ptr_f64 : f64*;
  H(ptr_f64);
  var ptr_i8 : i8*;
  H(ptr_i8);
  var c: C;
  H(c);

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
// CHECK:STDOUT:   %m.var = alloca i32, align 4, !dbg !8
// CHECK:STDOUT:   %p.var = alloca double, align 8, !dbg !9
// CHECK:STDOUT:   %q.var = alloca double, align 8, !dbg !10
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 4, ptr %n.var), !dbg !7
// CHECK:STDOUT:   store i32 0, ptr %n.var, align 4, !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 4, ptr %m.var), !dbg !8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 8, ptr %p.var), !dbg !9
// CHECK:STDOUT:   store double 1.000000e+00, ptr %p.var, align 8, !dbg !9
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 8, ptr %q.var), !dbg !10
// CHECK:STDOUT:   %.loc49 = load i32, ptr %n.var, align 4, !dbg !11
// CHECK:STDOUT:   call void @_CF.Main.b88d1103f417c6d4(i32 %.loc49), !dbg !12
// CHECK:STDOUT:   %.loc50 = load i32, ptr %n.var, align 4, !dbg !13
// CHECK:STDOUT:   %G.call.loc50 = call i32 @_CG.Main.b88d1103f417c6d4(i32 %.loc50), !dbg !14
// CHECK:STDOUT:   store i32 %G.call.loc50, ptr %m.var, align 4, !dbg !15
// CHECK:STDOUT:   %.loc51 = load double, ptr %p.var, align 8, !dbg !16
// CHECK:STDOUT:   call void @_CF.Main.66be507887ceee78(double %.loc51), !dbg !17
// CHECK:STDOUT:   %.loc52 = load double, ptr %p.var, align 8, !dbg !18
// CHECK:STDOUT:   %G.call.loc52 = call double @_CG.Main.66be507887ceee78(double %.loc52), !dbg !19
// CHECK:STDOUT:   store double %G.call.loc52, ptr %q.var, align 8, !dbg !20
// CHECK:STDOUT:   ret void, !dbg !21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(i64 immarg, ptr captures(none)) #0
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CF.Main.b88d1103f417c6d4(i32 %x) !dbg !22 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_CG.Main.b88d1103f417c6d4(i32 %x) !dbg !24 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %var_f64.var = alloca double, align 8, !dbg !25
// CHECK:STDOUT:   %ptr_i32.var = alloca ptr, align 8, !dbg !26
// CHECK:STDOUT:   %ptr_f64.var = alloca ptr, align 8, !dbg !27
// CHECK:STDOUT:   %ptr_i8.var = alloca ptr, align 8, !dbg !28
// CHECK:STDOUT:   %c.var = alloca {}, align 8, !dbg !29
// CHECK:STDOUT:   %H.call.loc23 = call i32 @_CH.Main.b88d1103f417c6d4(i32 %x), !dbg !30
// CHECK:STDOUT:   %H.call.loc24 = call %type @_CH.Main.5754c7a55c7cbe4a(%type zeroinitializer), !dbg !31
// CHECK:STDOUT:   %H.call.loc25 = call %type @_CH.Main.5754c7a55c7cbe4a(%type zeroinitializer), !dbg !32
// CHECK:STDOUT:   %G.call = call i32 @_CG.Main.b88d1103f417c6d4(i32 %x), !dbg !33
// CHECK:STDOUT:   %H.call.loc26 = call i32 @_CH.Main.b88d1103f417c6d4(i32 %G.call), !dbg !34
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 8, ptr %var_f64.var), !dbg !25
// CHECK:STDOUT:   %.loc30 = load double, ptr %var_f64.var, align 8, !dbg !35
// CHECK:STDOUT:   %H.call.loc30 = call double @_CH.Main.66be507887ceee78(double %.loc30), !dbg !36
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 8, ptr %ptr_i32.var), !dbg !26
// CHECK:STDOUT:   %.loc32 = load ptr, ptr %ptr_i32.var, align 8, !dbg !37
// CHECK:STDOUT:   %H.call.loc32 = call ptr @_CH.Main.e8193710fd35b608(ptr %.loc32), !dbg !38
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 8, ptr %ptr_f64.var), !dbg !27
// CHECK:STDOUT:   %.loc34 = load ptr, ptr %ptr_f64.var, align 8, !dbg !39
// CHECK:STDOUT:   %H.call.loc34 = call ptr @_CH.Main.04bf2edaaa84aa22(ptr %.loc34), !dbg !40
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 8, ptr %ptr_i8.var), !dbg !28
// CHECK:STDOUT:   %.loc36 = load ptr, ptr %ptr_i8.var, align 8, !dbg !41
// CHECK:STDOUT:   %H.call.loc36 = call ptr @_CH.Main.bda010de15e6a5ad(ptr %.loc36), !dbg !42
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 0, ptr %c.var), !dbg !29
// CHECK:STDOUT:   %.loc38_6.1.temp = alloca {}, align 8, !dbg !43
// CHECK:STDOUT:   call void @_CH.Main.15b1f98bd9cc0c5b(ptr %.loc38_6.1.temp, ptr %c.var), !dbg !43
// CHECK:STDOUT:   ret i32 %x, !dbg !44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CF.Main.66be507887ceee78(double %x) !dbg !45 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !46
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define double @_CG.Main.66be507887ceee78(double %x) !dbg !47 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %var_f64.var = alloca double, align 8, !dbg !48
// CHECK:STDOUT:   %ptr_i32.var = alloca ptr, align 8, !dbg !49
// CHECK:STDOUT:   %ptr_f64.var = alloca ptr, align 8, !dbg !50
// CHECK:STDOUT:   %ptr_i8.var = alloca ptr, align 8, !dbg !51
// CHECK:STDOUT:   %c.var = alloca {}, align 8, !dbg !52
// CHECK:STDOUT:   %H.call.loc23 = call double @_CH.Main.66be507887ceee78(double %x), !dbg !53
// CHECK:STDOUT:   %H.call.loc24 = call %type @_CH.Main.5754c7a55c7cbe4a(%type zeroinitializer), !dbg !54
// CHECK:STDOUT:   %H.call.loc25 = call %type @_CH.Main.5754c7a55c7cbe4a(%type zeroinitializer), !dbg !55
// CHECK:STDOUT:   %G.call = call double @_CG.Main.66be507887ceee78(double %x), !dbg !56
// CHECK:STDOUT:   %H.call.loc26 = call double @_CH.Main.66be507887ceee78(double %G.call), !dbg !57
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 8, ptr %var_f64.var), !dbg !48
// CHECK:STDOUT:   %.loc30 = load double, ptr %var_f64.var, align 8, !dbg !58
// CHECK:STDOUT:   %H.call.loc30 = call double @_CH.Main.66be507887ceee78(double %.loc30), !dbg !59
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 8, ptr %ptr_i32.var), !dbg !49
// CHECK:STDOUT:   %.loc32 = load ptr, ptr %ptr_i32.var, align 8, !dbg !60
// CHECK:STDOUT:   %H.call.loc32 = call ptr @_CH.Main.e8193710fd35b608(ptr %.loc32), !dbg !61
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 8, ptr %ptr_f64.var), !dbg !50
// CHECK:STDOUT:   %.loc34 = load ptr, ptr %ptr_f64.var, align 8, !dbg !62
// CHECK:STDOUT:   %H.call.loc34 = call ptr @_CH.Main.04bf2edaaa84aa22(ptr %.loc34), !dbg !63
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 8, ptr %ptr_i8.var), !dbg !51
// CHECK:STDOUT:   %.loc36 = load ptr, ptr %ptr_i8.var, align 8, !dbg !64
// CHECK:STDOUT:   %H.call.loc36 = call ptr @_CH.Main.bda010de15e6a5ad(ptr %.loc36), !dbg !65
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 0, ptr %c.var), !dbg !52
// CHECK:STDOUT:   %.loc38_6.1.temp = alloca {}, align 8, !dbg !66
// CHECK:STDOUT:   call void @_CH.Main.15b1f98bd9cc0c5b(ptr %.loc38_6.1.temp, ptr %c.var), !dbg !66
// CHECK:STDOUT:   ret double %x, !dbg !67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_CH.Main.b88d1103f417c6d4(i32 %x) !dbg !68 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %x, !dbg !69
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define %type @_CH.Main.5754c7a55c7cbe4a(%type %x) !dbg !70 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret %type %x, !dbg !71
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define double @_CH.Main.66be507887ceee78(double %x) !dbg !72 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret double %x, !dbg !73
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define ptr @_CH.Main.e8193710fd35b608(ptr %x) !dbg !74 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret ptr %x, !dbg !75
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define ptr @_CH.Main.04bf2edaaa84aa22(ptr %x) !dbg !76 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret ptr %x, !dbg !77
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define ptr @_CH.Main.bda010de15e6a5ad(ptr %x) !dbg !78 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret ptr %x, !dbg !79
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CH.Main.15b1f98bd9cc0c5b(ptr sret({}) %return, ptr %x) !dbg !80 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret ptr %x, !dbg !81
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 12, 11, 10 }
// CHECK:STDOUT: uselistorder ptr @_CH.Main.b88d1103f417c6d4, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CH.Main.5754c7a55c7cbe4a, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CH.Main.66be507887ceee78, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CH.Main.e8193710fd35b608, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CH.Main.04bf2edaaa84aa22, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CH.Main.bda010de15e6a5ad, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CH.Main.15b1f98bd9cc0c5b, { 1, 0 }
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
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "M", linkageName: "_CM.Main", scope: null, file: !3, line: 43, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 44, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 45, column: 3, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 46, column: 3, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 47, column: 3, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 49, column: 5, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 49, column: 3, scope: !4)
// CHECK:STDOUT: !13 = !DILocation(line: 50, column: 9, scope: !4)
// CHECK:STDOUT: !14 = !DILocation(line: 50, column: 7, scope: !4)
// CHECK:STDOUT: !15 = !DILocation(line: 50, column: 3, scope: !4)
// CHECK:STDOUT: !16 = !DILocation(line: 51, column: 5, scope: !4)
// CHECK:STDOUT: !17 = !DILocation(line: 51, column: 3, scope: !4)
// CHECK:STDOUT: !18 = !DILocation(line: 52, column: 9, scope: !4)
// CHECK:STDOUT: !19 = !DILocation(line: 52, column: 7, scope: !4)
// CHECK:STDOUT: !20 = !DILocation(line: 52, column: 3, scope: !4)
// CHECK:STDOUT: !21 = !DILocation(line: 43, column: 1, scope: !4)
// CHECK:STDOUT: !22 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.b88d1103f417c6d4", scope: null, file: !3, line: 13, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !23 = !DILocation(line: 13, column: 1, scope: !22)
// CHECK:STDOUT: !24 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.b88d1103f417c6d4", scope: null, file: !3, line: 22, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !25 = !DILocation(line: 29, column: 3, scope: !24)
// CHECK:STDOUT: !26 = !DILocation(line: 31, column: 3, scope: !24)
// CHECK:STDOUT: !27 = !DILocation(line: 33, column: 3, scope: !24)
// CHECK:STDOUT: !28 = !DILocation(line: 35, column: 3, scope: !24)
// CHECK:STDOUT: !29 = !DILocation(line: 37, column: 3, scope: !24)
// CHECK:STDOUT: !30 = !DILocation(line: 23, column: 3, scope: !24)
// CHECK:STDOUT: !31 = !DILocation(line: 24, column: 3, scope: !24)
// CHECK:STDOUT: !32 = !DILocation(line: 25, column: 3, scope: !24)
// CHECK:STDOUT: !33 = !DILocation(line: 26, column: 5, scope: !24)
// CHECK:STDOUT: !34 = !DILocation(line: 26, column: 3, scope: !24)
// CHECK:STDOUT: !35 = !DILocation(line: 30, column: 5, scope: !24)
// CHECK:STDOUT: !36 = !DILocation(line: 30, column: 3, scope: !24)
// CHECK:STDOUT: !37 = !DILocation(line: 32, column: 5, scope: !24)
// CHECK:STDOUT: !38 = !DILocation(line: 32, column: 3, scope: !24)
// CHECK:STDOUT: !39 = !DILocation(line: 34, column: 5, scope: !24)
// CHECK:STDOUT: !40 = !DILocation(line: 34, column: 3, scope: !24)
// CHECK:STDOUT: !41 = !DILocation(line: 36, column: 5, scope: !24)
// CHECK:STDOUT: !42 = !DILocation(line: 36, column: 3, scope: !24)
// CHECK:STDOUT: !43 = !DILocation(line: 38, column: 3, scope: !24)
// CHECK:STDOUT: !44 = !DILocation(line: 40, column: 3, scope: !24)
// CHECK:STDOUT: !45 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.66be507887ceee78", scope: null, file: !3, line: 13, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !46 = !DILocation(line: 13, column: 1, scope: !45)
// CHECK:STDOUT: !47 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.66be507887ceee78", scope: null, file: !3, line: 22, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !48 = !DILocation(line: 29, column: 3, scope: !47)
// CHECK:STDOUT: !49 = !DILocation(line: 31, column: 3, scope: !47)
// CHECK:STDOUT: !50 = !DILocation(line: 33, column: 3, scope: !47)
// CHECK:STDOUT: !51 = !DILocation(line: 35, column: 3, scope: !47)
// CHECK:STDOUT: !52 = !DILocation(line: 37, column: 3, scope: !47)
// CHECK:STDOUT: !53 = !DILocation(line: 23, column: 3, scope: !47)
// CHECK:STDOUT: !54 = !DILocation(line: 24, column: 3, scope: !47)
// CHECK:STDOUT: !55 = !DILocation(line: 25, column: 3, scope: !47)
// CHECK:STDOUT: !56 = !DILocation(line: 26, column: 5, scope: !47)
// CHECK:STDOUT: !57 = !DILocation(line: 26, column: 3, scope: !47)
// CHECK:STDOUT: !58 = !DILocation(line: 30, column: 5, scope: !47)
// CHECK:STDOUT: !59 = !DILocation(line: 30, column: 3, scope: !47)
// CHECK:STDOUT: !60 = !DILocation(line: 32, column: 5, scope: !47)
// CHECK:STDOUT: !61 = !DILocation(line: 32, column: 3, scope: !47)
// CHECK:STDOUT: !62 = !DILocation(line: 34, column: 5, scope: !47)
// CHECK:STDOUT: !63 = !DILocation(line: 34, column: 3, scope: !47)
// CHECK:STDOUT: !64 = !DILocation(line: 36, column: 5, scope: !47)
// CHECK:STDOUT: !65 = !DILocation(line: 36, column: 3, scope: !47)
// CHECK:STDOUT: !66 = !DILocation(line: 38, column: 3, scope: !47)
// CHECK:STDOUT: !67 = !DILocation(line: 40, column: 3, scope: !47)
// CHECK:STDOUT: !68 = distinct !DISubprogram(name: "H", linkageName: "_CH.Main.b88d1103f417c6d4", scope: null, file: !3, line: 16, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !69 = !DILocation(line: 17, column: 3, scope: !68)
// CHECK:STDOUT: !70 = distinct !DISubprogram(name: "H", linkageName: "_CH.Main.5754c7a55c7cbe4a", scope: null, file: !3, line: 16, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !71 = !DILocation(line: 17, column: 3, scope: !70)
// CHECK:STDOUT: !72 = distinct !DISubprogram(name: "H", linkageName: "_CH.Main.66be507887ceee78", scope: null, file: !3, line: 16, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !73 = !DILocation(line: 17, column: 3, scope: !72)
// CHECK:STDOUT: !74 = distinct !DISubprogram(name: "H", linkageName: "_CH.Main.e8193710fd35b608", scope: null, file: !3, line: 16, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !75 = !DILocation(line: 17, column: 3, scope: !74)
// CHECK:STDOUT: !76 = distinct !DISubprogram(name: "H", linkageName: "_CH.Main.04bf2edaaa84aa22", scope: null, file: !3, line: 16, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !77 = !DILocation(line: 17, column: 3, scope: !76)
// CHECK:STDOUT: !78 = distinct !DISubprogram(name: "H", linkageName: "_CH.Main.bda010de15e6a5ad", scope: null, file: !3, line: 16, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !79 = !DILocation(line: 17, column: 3, scope: !78)
// CHECK:STDOUT: !80 = distinct !DISubprogram(name: "H", linkageName: "_CH.Main.15b1f98bd9cc0c5b", scope: null, file: !3, line: 16, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !81 = !DILocation(line: 17, column: 3, scope: !80)
