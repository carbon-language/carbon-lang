// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/generic/call_basic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/generic/call_basic.carbon

class C {
  impl as Core.Copy {
    fn Op[self: Self]() -> Self { return {}; }
  }
}

fn F[T:! type](x: T) {
}

fn H[T:! Core.Copy](x: T) -> T {
  return x;
}

// Simple stress test for single depth of calls with change in types.
// Check definitions are emitted for each specific.
fn G[T:! Core.Copy & Core.Destroy](x: T) -> T {
  H(x);
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
// CHECK:STDOUT: @C.val.loc15_44 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: define void @"_COp.C.Main:Copy.Core"(ptr sret({}) %return, ptr %self) !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %return, ptr align 1 @C.val.loc15_44, i64 0, i1 false), !dbg !7
// CHECK:STDOUT:   ret void, !dbg !7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CM.Main() !dbg !8 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %n.var = alloca i32, align 4, !dbg !9
// CHECK:STDOUT:   %m.var = alloca i32, align 4, !dbg !10
// CHECK:STDOUT:   %p.var = alloca double, align 8, !dbg !11
// CHECK:STDOUT:   %q.var = alloca double, align 8, !dbg !12
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %n.var), !dbg !9
// CHECK:STDOUT:   store i32 0, ptr %n.var, align 4, !dbg !9
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %m.var), !dbg !10
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %p.var), !dbg !11
// CHECK:STDOUT:   store double 1.000000e+00, ptr %p.var, align 8, !dbg !11
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %q.var), !dbg !12
// CHECK:STDOUT:   %.loc54 = load i32, ptr %n.var, align 4, !dbg !13
// CHECK:STDOUT:   call void @_CF.Main.b88d1103f417c6d4(i32 %.loc54), !dbg !14
// CHECK:STDOUT:   %.loc55_9 = load i32, ptr %n.var, align 4, !dbg !15
// CHECK:STDOUT:   %G.call.loc55 = call i32 @_CG.Main.c831c62a6babb214(i32 %.loc55_9), !dbg !16
// CHECK:STDOUT:   store i32 %G.call.loc55, ptr %m.var, align 4, !dbg !17
// CHECK:STDOUT:   %.loc56 = load double, ptr %p.var, align 8, !dbg !18
// CHECK:STDOUT:   call void @_CF.Main.d4b5665541d5d7a8(double %.loc56), !dbg !19
// CHECK:STDOUT:   %.loc57_9 = load double, ptr %p.var, align 8, !dbg !20
// CHECK:STDOUT:   %G.call.loc57 = call double @_CG.Main.54ba0c5f7a576a29(double %.loc57_9), !dbg !21
// CHECK:STDOUT:   store double %G.call.loc57, ptr %q.var, align 8, !dbg !22
// CHECK:STDOUT:   ret void, !dbg !23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #0
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.b88d1103f417c6d4(i32 %x) !dbg !24 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr i32 @_CG.Main.c831c62a6babb214(i32 %x) !dbg !26 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc29_6.3.temp = alloca i32, align 4, !dbg !27
// CHECK:STDOUT:   %.loc31_8.3.temp = alloca i32, align 4, !dbg !28
// CHECK:STDOUT:   %.loc31_9.3.temp = alloca i32, align 4, !dbg !29
// CHECK:STDOUT:   %var_f64.var = alloca double, align 8, !dbg !30
// CHECK:STDOUT:   %ptr_i32.var = alloca ptr, align 8, !dbg !31
// CHECK:STDOUT:   %ptr_f64.var = alloca ptr, align 8, !dbg !32
// CHECK:STDOUT:   %ptr_i8.var = alloca ptr, align 8, !dbg !33
// CHECK:STDOUT:   %c.var = alloca {}, align 8, !dbg !34
// CHECK:STDOUT:   %.loc43_6.3.temp = alloca {}, align 8, !dbg !35
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc29_6.3.temp), !dbg !27
// CHECK:STDOUT:   %H.call.loc29 = call i32 @_CH.Main.de631560529e9861(i32 %x), !dbg !27
// CHECK:STDOUT:   store i32 %H.call.loc29, ptr %.loc29_6.3.temp, align 4, !dbg !27
// CHECK:STDOUT:   %H.call.loc30 = call %type @_CH.Main.402deed6b8733082(%type zeroinitializer), !dbg !36
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc31_8.3.temp), !dbg !28
// CHECK:STDOUT:   %G.call = call i32 @_CG.Main.c831c62a6babb214(i32 %x), !dbg !28
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc31_9.3.temp), !dbg !29
// CHECK:STDOUT:   store i32 %G.call, ptr %.loc31_8.3.temp, align 4, !dbg !28
// CHECK:STDOUT:   %.loc31_8.5 = load i32, ptr %.loc31_8.3.temp, align 4, !dbg !28
// CHECK:STDOUT:   %H.call.loc31 = call i32 @_CH.Main.de631560529e9861(i32 %.loc31_8.5), !dbg !29
// CHECK:STDOUT:   store i32 %H.call.loc31, ptr %.loc31_9.3.temp, align 4, !dbg !29
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %var_f64.var), !dbg !30
// CHECK:STDOUT:   %.loc35_5 = load double, ptr %var_f64.var, align 8, !dbg !37
// CHECK:STDOUT:   %H.call.loc35 = call double @_CH.Main.3836ccc537683c19(double %.loc35_5), !dbg !38
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_i32.var), !dbg !31
// CHECK:STDOUT:   %.loc37_5 = load ptr, ptr %ptr_i32.var, align 8, !dbg !39
// CHECK:STDOUT:   %H.call.loc37 = call ptr @_CH.Main.a194de4a3b8dbd0c(ptr %.loc37_5), !dbg !40
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_f64.var), !dbg !32
// CHECK:STDOUT:   %.loc39_5 = load ptr, ptr %ptr_f64.var, align 8, !dbg !41
// CHECK:STDOUT:   %H.call.loc39 = call ptr @_CH.Main.a194de4a3b8dbd0c(ptr %.loc39_5), !dbg !42
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_i8.var), !dbg !33
// CHECK:STDOUT:   %.loc41_5 = load ptr, ptr %ptr_i8.var, align 8, !dbg !43
// CHECK:STDOUT:   %H.call.loc41 = call ptr @_CH.Main.a194de4a3b8dbd0c(ptr %.loc41_5), !dbg !44
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !34
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc43_6.3.temp), !dbg !35
// CHECK:STDOUT:   call void @_CH.Main.936d996ea935415c(ptr %.loc43_6.3.temp, ptr %c.var), !dbg !35
// CHECK:STDOUT:   ret i32 %x, !dbg !45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.d4b5665541d5d7a8(double %x) !dbg !46 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !47
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr double @_CG.Main.54ba0c5f7a576a29(double %x) !dbg !48 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc29_6.3.temp = alloca double, align 8, !dbg !49
// CHECK:STDOUT:   %.loc31_8.3.temp = alloca double, align 8, !dbg !50
// CHECK:STDOUT:   %.loc31_9.3.temp = alloca double, align 8, !dbg !51
// CHECK:STDOUT:   %var_f64.var = alloca double, align 8, !dbg !52
// CHECK:STDOUT:   %ptr_i32.var = alloca ptr, align 8, !dbg !53
// CHECK:STDOUT:   %ptr_f64.var = alloca ptr, align 8, !dbg !54
// CHECK:STDOUT:   %ptr_i8.var = alloca ptr, align 8, !dbg !55
// CHECK:STDOUT:   %c.var = alloca {}, align 8, !dbg !56
// CHECK:STDOUT:   %.loc43_6.3.temp = alloca {}, align 8, !dbg !57
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc29_6.3.temp), !dbg !49
// CHECK:STDOUT:   %H.call.loc29 = call double @_CH.Main.3836ccc537683c19(double %x), !dbg !49
// CHECK:STDOUT:   store double %H.call.loc29, ptr %.loc29_6.3.temp, align 8, !dbg !49
// CHECK:STDOUT:   %H.call.loc30 = call %type @_CH.Main.402deed6b8733082(%type zeroinitializer), !dbg !58
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc31_8.3.temp), !dbg !50
// CHECK:STDOUT:   %G.call = call double @_CG.Main.54ba0c5f7a576a29(double %x), !dbg !50
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc31_9.3.temp), !dbg !51
// CHECK:STDOUT:   store double %G.call, ptr %.loc31_8.3.temp, align 8, !dbg !50
// CHECK:STDOUT:   %.loc31_8.5 = load double, ptr %.loc31_8.3.temp, align 8, !dbg !50
// CHECK:STDOUT:   %H.call.loc31 = call double @_CH.Main.3836ccc537683c19(double %.loc31_8.5), !dbg !51
// CHECK:STDOUT:   store double %H.call.loc31, ptr %.loc31_9.3.temp, align 8, !dbg !51
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %var_f64.var), !dbg !52
// CHECK:STDOUT:   %.loc35_5 = load double, ptr %var_f64.var, align 8, !dbg !59
// CHECK:STDOUT:   %H.call.loc35 = call double @_CH.Main.3836ccc537683c19(double %.loc35_5), !dbg !60
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_i32.var), !dbg !53
// CHECK:STDOUT:   %.loc37_5 = load ptr, ptr %ptr_i32.var, align 8, !dbg !61
// CHECK:STDOUT:   %H.call.loc37 = call ptr @_CH.Main.a194de4a3b8dbd0c(ptr %.loc37_5), !dbg !62
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_f64.var), !dbg !54
// CHECK:STDOUT:   %.loc39_5 = load ptr, ptr %ptr_f64.var, align 8, !dbg !63
// CHECK:STDOUT:   %H.call.loc39 = call ptr @_CH.Main.a194de4a3b8dbd0c(ptr %.loc39_5), !dbg !64
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_i8.var), !dbg !55
// CHECK:STDOUT:   %.loc41_5 = load ptr, ptr %ptr_i8.var, align 8, !dbg !65
// CHECK:STDOUT:   %H.call.loc41 = call ptr @_CH.Main.a194de4a3b8dbd0c(ptr %.loc41_5), !dbg !66
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !56
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc43_6.3.temp), !dbg !57
// CHECK:STDOUT:   call void @_CH.Main.936d996ea935415c(ptr %.loc43_6.3.temp, ptr %c.var), !dbg !57
// CHECK:STDOUT:   ret double %x, !dbg !67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr i32 @_CH.Main.de631560529e9861(i32 %x) !dbg !68 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %x, !dbg !69
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr %type @_CH.Main.402deed6b8733082(%type %x) !dbg !70 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret %type %x, !dbg !71
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr double @_CH.Main.3836ccc537683c19(double %x) !dbg !72 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret double %x, !dbg !73
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr ptr @_CH.Main.a194de4a3b8dbd0c(ptr %x) !dbg !74 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret ptr %x, !dbg !75
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr void @_CH.Main.936d996ea935415c(ptr sret({}) %return, ptr %x) !dbg !76 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_COp.C.Main:Copy.Core"(ptr %return, ptr %x), !dbg !77
// CHECK:STDOUT:   ret void, !dbg !78
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 20, 19, 18 }
// CHECK:STDOUT: uselistorder ptr @_CH.Main.de631560529e9861, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CH.Main.402deed6b8733082, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CH.Main.3836ccc537683c19, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CH.Main.a194de4a3b8dbd0c, { 5, 2, 0, 4, 3, 1 }
// CHECK:STDOUT: uselistorder ptr @_CH.Main.936d996ea935415c, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "call_basic.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Op", linkageName: "_COp.C.Main:Copy.Core", scope: null, file: !3, line: 15, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 15, column: 35, scope: !4)
// CHECK:STDOUT: !8 = distinct !DISubprogram(name: "M", linkageName: "_CM.Main", scope: null, file: !3, line: 48, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !9 = !DILocation(line: 49, column: 3, scope: !8)
// CHECK:STDOUT: !10 = !DILocation(line: 50, column: 3, scope: !8)
// CHECK:STDOUT: !11 = !DILocation(line: 51, column: 3, scope: !8)
// CHECK:STDOUT: !12 = !DILocation(line: 52, column: 3, scope: !8)
// CHECK:STDOUT: !13 = !DILocation(line: 54, column: 5, scope: !8)
// CHECK:STDOUT: !14 = !DILocation(line: 54, column: 3, scope: !8)
// CHECK:STDOUT: !15 = !DILocation(line: 55, column: 9, scope: !8)
// CHECK:STDOUT: !16 = !DILocation(line: 55, column: 7, scope: !8)
// CHECK:STDOUT: !17 = !DILocation(line: 55, column: 3, scope: !8)
// CHECK:STDOUT: !18 = !DILocation(line: 56, column: 5, scope: !8)
// CHECK:STDOUT: !19 = !DILocation(line: 56, column: 3, scope: !8)
// CHECK:STDOUT: !20 = !DILocation(line: 57, column: 9, scope: !8)
// CHECK:STDOUT: !21 = !DILocation(line: 57, column: 7, scope: !8)
// CHECK:STDOUT: !22 = !DILocation(line: 57, column: 3, scope: !8)
// CHECK:STDOUT: !23 = !DILocation(line: 48, column: 1, scope: !8)
// CHECK:STDOUT: !24 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.b88d1103f417c6d4", scope: null, file: !3, line: 19, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !25 = !DILocation(line: 19, column: 1, scope: !24)
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.c831c62a6babb214", scope: null, file: !3, line: 28, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !27 = !DILocation(line: 29, column: 3, scope: !26)
// CHECK:STDOUT: !28 = !DILocation(line: 31, column: 5, scope: !26)
// CHECK:STDOUT: !29 = !DILocation(line: 31, column: 3, scope: !26)
// CHECK:STDOUT: !30 = !DILocation(line: 34, column: 3, scope: !26)
// CHECK:STDOUT: !31 = !DILocation(line: 36, column: 3, scope: !26)
// CHECK:STDOUT: !32 = !DILocation(line: 38, column: 3, scope: !26)
// CHECK:STDOUT: !33 = !DILocation(line: 40, column: 3, scope: !26)
// CHECK:STDOUT: !34 = !DILocation(line: 42, column: 3, scope: !26)
// CHECK:STDOUT: !35 = !DILocation(line: 43, column: 3, scope: !26)
// CHECK:STDOUT: !36 = !DILocation(line: 30, column: 3, scope: !26)
// CHECK:STDOUT: !37 = !DILocation(line: 35, column: 5, scope: !26)
// CHECK:STDOUT: !38 = !DILocation(line: 35, column: 3, scope: !26)
// CHECK:STDOUT: !39 = !DILocation(line: 37, column: 5, scope: !26)
// CHECK:STDOUT: !40 = !DILocation(line: 37, column: 3, scope: !26)
// CHECK:STDOUT: !41 = !DILocation(line: 39, column: 5, scope: !26)
// CHECK:STDOUT: !42 = !DILocation(line: 39, column: 3, scope: !26)
// CHECK:STDOUT: !43 = !DILocation(line: 41, column: 5, scope: !26)
// CHECK:STDOUT: !44 = !DILocation(line: 41, column: 3, scope: !26)
// CHECK:STDOUT: !45 = !DILocation(line: 45, column: 3, scope: !26)
// CHECK:STDOUT: !46 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.d4b5665541d5d7a8", scope: null, file: !3, line: 19, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !47 = !DILocation(line: 19, column: 1, scope: !46)
// CHECK:STDOUT: !48 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.54ba0c5f7a576a29", scope: null, file: !3, line: 28, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !49 = !DILocation(line: 29, column: 3, scope: !48)
// CHECK:STDOUT: !50 = !DILocation(line: 31, column: 5, scope: !48)
// CHECK:STDOUT: !51 = !DILocation(line: 31, column: 3, scope: !48)
// CHECK:STDOUT: !52 = !DILocation(line: 34, column: 3, scope: !48)
// CHECK:STDOUT: !53 = !DILocation(line: 36, column: 3, scope: !48)
// CHECK:STDOUT: !54 = !DILocation(line: 38, column: 3, scope: !48)
// CHECK:STDOUT: !55 = !DILocation(line: 40, column: 3, scope: !48)
// CHECK:STDOUT: !56 = !DILocation(line: 42, column: 3, scope: !48)
// CHECK:STDOUT: !57 = !DILocation(line: 43, column: 3, scope: !48)
// CHECK:STDOUT: !58 = !DILocation(line: 30, column: 3, scope: !48)
// CHECK:STDOUT: !59 = !DILocation(line: 35, column: 5, scope: !48)
// CHECK:STDOUT: !60 = !DILocation(line: 35, column: 3, scope: !48)
// CHECK:STDOUT: !61 = !DILocation(line: 37, column: 5, scope: !48)
// CHECK:STDOUT: !62 = !DILocation(line: 37, column: 3, scope: !48)
// CHECK:STDOUT: !63 = !DILocation(line: 39, column: 5, scope: !48)
// CHECK:STDOUT: !64 = !DILocation(line: 39, column: 3, scope: !48)
// CHECK:STDOUT: !65 = !DILocation(line: 41, column: 5, scope: !48)
// CHECK:STDOUT: !66 = !DILocation(line: 41, column: 3, scope: !48)
// CHECK:STDOUT: !67 = !DILocation(line: 45, column: 3, scope: !48)
// CHECK:STDOUT: !68 = distinct !DISubprogram(name: "H", linkageName: "_CH.Main.de631560529e9861", scope: null, file: !3, line: 22, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !69 = !DILocation(line: 23, column: 3, scope: !68)
// CHECK:STDOUT: !70 = distinct !DISubprogram(name: "H", linkageName: "_CH.Main.402deed6b8733082", scope: null, file: !3, line: 22, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !71 = !DILocation(line: 23, column: 3, scope: !70)
// CHECK:STDOUT: !72 = distinct !DISubprogram(name: "H", linkageName: "_CH.Main.3836ccc537683c19", scope: null, file: !3, line: 22, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !73 = !DILocation(line: 23, column: 3, scope: !72)
// CHECK:STDOUT: !74 = distinct !DISubprogram(name: "H", linkageName: "_CH.Main.a194de4a3b8dbd0c", scope: null, file: !3, line: 22, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !75 = !DILocation(line: 23, column: 3, scope: !74)
// CHECK:STDOUT: !76 = distinct !DISubprogram(name: "H", linkageName: "_CH.Main.936d996ea935415c", scope: null, file: !3, line: 22, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !77 = !DILocation(line: 23, column: 10, scope: !76)
// CHECK:STDOUT: !78 = !DILocation(line: 23, column: 3, scope: !76)
