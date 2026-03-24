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
    fn Op[unused self: Self]() -> Self { return {}; }
  }
}

fn F[T:! type](_: T) {
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
  var p: f64 = 1.0;

  F(n);
  let _: i32 = G(n);
  F(p);
  let _: f64 = G(p);
}

// CHECK:STDOUT: ; ModuleID = 'call_basic.carbon'
// CHECK:STDOUT: source_filename = "call_basic.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: %type = type {}
// CHECK:STDOUT:
// CHECK:STDOUT: @C.val.loc15_51 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_COp.C.Main:Copy.Core"(ptr sret({}) %return, ptr %self) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %return, ptr align 1 @C.val.loc15_51, i64 0, i1 false), !dbg !10
// CHECK:STDOUT:   ret void, !dbg !10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CM.Main() #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %n.var = alloca i32, align 4, !dbg !14
// CHECK:STDOUT:   %p.var = alloca double, align 8, !dbg !15
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %n.var), !dbg !14
// CHECK:STDOUT:   store i32 0, ptr %n.var, align 4, !dbg !14
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %p.var), !dbg !15
// CHECK:STDOUT:   store double 1.000000e+00, ptr %p.var, align 8, !dbg !15
// CHECK:STDOUT:   %.loc52 = load i32, ptr %n.var, align 4, !dbg !16
// CHECK:STDOUT:   call void @_CF.Main.b88d1103f417c6d4(i32 %.loc52), !dbg !17
// CHECK:STDOUT:   %.loc53_18 = load i32, ptr %n.var, align 4, !dbg !18
// CHECK:STDOUT:   %G.call.loc53 = call i32 @_CG.Main.bfb441135f07cbe1(i32 %.loc53_18), !dbg !19
// CHECK:STDOUT:   %.loc54 = load double, ptr %p.var, align 8, !dbg !20
// CHECK:STDOUT:   call void @_CF.Main.d4b5665541d5d7a8(double %.loc54), !dbg !21
// CHECK:STDOUT:   %.loc55_18 = load double, ptr %p.var, align 8, !dbg !22
// CHECK:STDOUT:   %G.call.loc55 = call double @_CG.Main.29cf1dfeccef7249(double %.loc55_18), !dbg !23
// CHECK:STDOUT:   ret void, !dbg !24
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.b88d1103f417c6d4(i32 %_) #0 !dbg !25 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CG.Main.bfb441135f07cbe1(i32 %x) #0 !dbg !32 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc29_6.2.temp = alloca i32, align 4, !dbg !37
// CHECK:STDOUT:   %.loc31_8.2.temp = alloca i32, align 4, !dbg !38
// CHECK:STDOUT:   %.loc31_9.2.temp = alloca i32, align 4, !dbg !39
// CHECK:STDOUT:   %var_f64.var = alloca double, align 8, !dbg !40
// CHECK:STDOUT:   %ptr_i32.var = alloca ptr, align 8, !dbg !41
// CHECK:STDOUT:   %ptr_f64.var = alloca ptr, align 8, !dbg !42
// CHECK:STDOUT:   %ptr_i8.var = alloca ptr, align 8, !dbg !43
// CHECK:STDOUT:   %c.var = alloca {}, align 8, !dbg !44
// CHECK:STDOUT:   %.loc43_6.2.temp = alloca {}, align 8, !dbg !45
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc29_6.2.temp), !dbg !37
// CHECK:STDOUT:   %H.call.loc29 = call i32 @_CH.Main.64ccbb8e5d9a0b8e(i32 %x), !dbg !37
// CHECK:STDOUT:   store i32 %H.call.loc29, ptr %.loc29_6.2.temp, align 4, !dbg !37
// CHECK:STDOUT:   %H.call.loc30 = call %type @_CH.Main.582d60b54baf6b63(%type zeroinitializer), !dbg !46
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc31_8.2.temp), !dbg !38
// CHECK:STDOUT:   %G.call = call i32 @_CG.Main.bfb441135f07cbe1(i32 %x), !dbg !38
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc31_9.2.temp), !dbg !39
// CHECK:STDOUT:   store i32 %G.call, ptr %.loc31_8.2.temp, align 4, !dbg !38
// CHECK:STDOUT:   %.loc31_8.4 = load i32, ptr %.loc31_8.2.temp, align 4, !dbg !38
// CHECK:STDOUT:   %H.call.loc31 = call i32 @_CH.Main.64ccbb8e5d9a0b8e(i32 %.loc31_8.4), !dbg !39
// CHECK:STDOUT:   store i32 %H.call.loc31, ptr %.loc31_9.2.temp, align 4, !dbg !39
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %var_f64.var), !dbg !40
// CHECK:STDOUT:   store double poison, ptr %var_f64.var, align 8, !dbg !40
// CHECK:STDOUT:   %.loc35_5 = load double, ptr %var_f64.var, align 8, !dbg !47
// CHECK:STDOUT:   %H.call.loc35 = call double @_CH.Main.de5b400758c39a65(double %.loc35_5), !dbg !48
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_i32.var), !dbg !41
// CHECK:STDOUT:   store ptr poison, ptr %ptr_i32.var, align 8, !dbg !41
// CHECK:STDOUT:   %.loc37_5 = load ptr, ptr %ptr_i32.var, align 8, !dbg !49
// CHECK:STDOUT:   %H.call.loc37 = call ptr @_CH.Main.779b9c0f3b54a7f8(ptr %.loc37_5), !dbg !50
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_f64.var), !dbg !42
// CHECK:STDOUT:   store ptr poison, ptr %ptr_f64.var, align 8, !dbg !42
// CHECK:STDOUT:   %.loc39_5 = load ptr, ptr %ptr_f64.var, align 8, !dbg !51
// CHECK:STDOUT:   %H.call.loc39 = call ptr @_CH.Main.779b9c0f3b54a7f8(ptr %.loc39_5), !dbg !52
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_i8.var), !dbg !43
// CHECK:STDOUT:   store ptr poison, ptr %ptr_i8.var, align 8, !dbg !43
// CHECK:STDOUT:   %.loc41_5 = load ptr, ptr %ptr_i8.var, align 8, !dbg !53
// CHECK:STDOUT:   %H.call.loc41 = call ptr @_CH.Main.779b9c0f3b54a7f8(ptr %.loc41_5), !dbg !54
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !44
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc43_6.2.temp), !dbg !45
// CHECK:STDOUT:   call void @_CH.Main.706e9f413f1bae3d(ptr %.loc43_6.2.temp, ptr %c.var), !dbg !45
// CHECK:STDOUT:   ret i32 %x, !dbg !55
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.d4b5665541d5d7a8(double %_) #0 !dbg !56 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !61
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CG.Main.29cf1dfeccef7249(double %x) #0 !dbg !62 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc29_6.2.temp = alloca double, align 8, !dbg !65
// CHECK:STDOUT:   %.loc31_8.2.temp = alloca double, align 8, !dbg !66
// CHECK:STDOUT:   %.loc31_9.2.temp = alloca double, align 8, !dbg !67
// CHECK:STDOUT:   %var_f64.var = alloca double, align 8, !dbg !68
// CHECK:STDOUT:   %ptr_i32.var = alloca ptr, align 8, !dbg !69
// CHECK:STDOUT:   %ptr_f64.var = alloca ptr, align 8, !dbg !70
// CHECK:STDOUT:   %ptr_i8.var = alloca ptr, align 8, !dbg !71
// CHECK:STDOUT:   %c.var = alloca {}, align 8, !dbg !72
// CHECK:STDOUT:   %.loc43_6.2.temp = alloca {}, align 8, !dbg !73
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc29_6.2.temp), !dbg !65
// CHECK:STDOUT:   %H.call.loc29 = call double @_CH.Main.de5b400758c39a65(double %x), !dbg !65
// CHECK:STDOUT:   store double %H.call.loc29, ptr %.loc29_6.2.temp, align 8, !dbg !65
// CHECK:STDOUT:   %H.call.loc30 = call %type @_CH.Main.582d60b54baf6b63(%type zeroinitializer), !dbg !74
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc31_8.2.temp), !dbg !66
// CHECK:STDOUT:   %G.call = call double @_CG.Main.29cf1dfeccef7249(double %x), !dbg !66
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc31_9.2.temp), !dbg !67
// CHECK:STDOUT:   store double %G.call, ptr %.loc31_8.2.temp, align 8, !dbg !66
// CHECK:STDOUT:   %.loc31_8.4 = load double, ptr %.loc31_8.2.temp, align 8, !dbg !66
// CHECK:STDOUT:   %H.call.loc31 = call double @_CH.Main.de5b400758c39a65(double %.loc31_8.4), !dbg !67
// CHECK:STDOUT:   store double %H.call.loc31, ptr %.loc31_9.2.temp, align 8, !dbg !67
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %var_f64.var), !dbg !68
// CHECK:STDOUT:   store double poison, ptr %var_f64.var, align 8, !dbg !68
// CHECK:STDOUT:   %.loc35_5 = load double, ptr %var_f64.var, align 8, !dbg !75
// CHECK:STDOUT:   %H.call.loc35 = call double @_CH.Main.de5b400758c39a65(double %.loc35_5), !dbg !76
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_i32.var), !dbg !69
// CHECK:STDOUT:   store ptr poison, ptr %ptr_i32.var, align 8, !dbg !69
// CHECK:STDOUT:   %.loc37_5 = load ptr, ptr %ptr_i32.var, align 8, !dbg !77
// CHECK:STDOUT:   %H.call.loc37 = call ptr @_CH.Main.779b9c0f3b54a7f8(ptr %.loc37_5), !dbg !78
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_f64.var), !dbg !70
// CHECK:STDOUT:   store ptr poison, ptr %ptr_f64.var, align 8, !dbg !70
// CHECK:STDOUT:   %.loc39_5 = load ptr, ptr %ptr_f64.var, align 8, !dbg !79
// CHECK:STDOUT:   %H.call.loc39 = call ptr @_CH.Main.779b9c0f3b54a7f8(ptr %.loc39_5), !dbg !80
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_i8.var), !dbg !71
// CHECK:STDOUT:   store ptr poison, ptr %ptr_i8.var, align 8, !dbg !71
// CHECK:STDOUT:   %.loc41_5 = load ptr, ptr %ptr_i8.var, align 8, !dbg !81
// CHECK:STDOUT:   %H.call.loc41 = call ptr @_CH.Main.779b9c0f3b54a7f8(ptr %.loc41_5), !dbg !82
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !72
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc43_6.2.temp), !dbg !73
// CHECK:STDOUT:   call void @_CH.Main.706e9f413f1bae3d(ptr %.loc43_6.2.temp, ptr %c.var), !dbg !73
// CHECK:STDOUT:   ret double %x, !dbg !83
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CH.Main.64ccbb8e5d9a0b8e(i32 %x) #0 !dbg !84 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %x, !dbg !87
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr %type @_CH.Main.582d60b54baf6b63(%type %x) #0 !dbg !88 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret %type %x, !dbg !91
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CH.Main.de5b400758c39a65(double %x) #0 !dbg !92 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret double %x, !dbg !95
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CH.Main.779b9c0f3b54a7f8(ptr %x) #0 !dbg !96 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret ptr %x, !dbg !99
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CH.Main.706e9f413f1bae3d(ptr sret({}) %return, ptr %x) #0 !dbg !100 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_COp.C.Main:Copy.Core"(ptr %return, ptr %x), !dbg !103
// CHECK:STDOUT:   ret void, !dbg !104
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 18 }
// CHECK:STDOUT: uselistorder ptr @_CH.Main.64ccbb8e5d9a0b8e, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CH.Main.582d60b54baf6b63, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CH.Main.de5b400758c39a65, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CH.Main.779b9c0f3b54a7f8, { 5, 2, 0, 4, 3, 1 }
// CHECK:STDOUT: uselistorder ptr @_CH.Main.706e9f413f1bae3d, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "call_basic.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Op", linkageName: "_COp.C.Main:Copy.Core", scope: null, file: !3, line: 15, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !8)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7, !7}
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 8)
// CHECK:STDOUT: !8 = !{!9}
// CHECK:STDOUT: !9 = !DILocalVariable(arg: 1, scope: !4, type: !7)
// CHECK:STDOUT: !10 = !DILocation(line: 15, column: 42, scope: !4)
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "M", linkageName: "_CM.Main", scope: null, file: !3, line: 48, type: !12, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !12 = !DISubroutineType(types: !13)
// CHECK:STDOUT: !13 = !{null}
// CHECK:STDOUT: !14 = !DILocation(line: 49, column: 3, scope: !11)
// CHECK:STDOUT: !15 = !DILocation(line: 50, column: 3, scope: !11)
// CHECK:STDOUT: !16 = !DILocation(line: 52, column: 5, scope: !11)
// CHECK:STDOUT: !17 = !DILocation(line: 52, column: 3, scope: !11)
// CHECK:STDOUT: !18 = !DILocation(line: 53, column: 18, scope: !11)
// CHECK:STDOUT: !19 = !DILocation(line: 53, column: 16, scope: !11)
// CHECK:STDOUT: !20 = !DILocation(line: 54, column: 5, scope: !11)
// CHECK:STDOUT: !21 = !DILocation(line: 54, column: 3, scope: !11)
// CHECK:STDOUT: !22 = !DILocation(line: 55, column: 18, scope: !11)
// CHECK:STDOUT: !23 = !DILocation(line: 55, column: 16, scope: !11)
// CHECK:STDOUT: !24 = !DILocation(line: 48, column: 1, scope: !11)
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.b88d1103f417c6d4", scope: null, file: !3, line: 19, type: !26, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !29)
// CHECK:STDOUT: !26 = !DISubroutineType(types: !27)
// CHECK:STDOUT: !27 = !{null, !28}
// CHECK:STDOUT: !28 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !29 = !{!30}
// CHECK:STDOUT: !30 = !DILocalVariable(arg: 1, scope: !25, type: !28)
// CHECK:STDOUT: !31 = !DILocation(line: 19, column: 1, scope: !25)
// CHECK:STDOUT: !32 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.bfb441135f07cbe1", scope: null, file: !3, line: 28, type: !33, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !35)
// CHECK:STDOUT: !33 = !DISubroutineType(types: !34)
// CHECK:STDOUT: !34 = !{!28, !28}
// CHECK:STDOUT: !35 = !{!36}
// CHECK:STDOUT: !36 = !DILocalVariable(arg: 1, scope: !32, type: !28)
// CHECK:STDOUT: !37 = !DILocation(line: 29, column: 3, scope: !32)
// CHECK:STDOUT: !38 = !DILocation(line: 31, column: 5, scope: !32)
// CHECK:STDOUT: !39 = !DILocation(line: 31, column: 3, scope: !32)
// CHECK:STDOUT: !40 = !DILocation(line: 34, column: 3, scope: !32)
// CHECK:STDOUT: !41 = !DILocation(line: 36, column: 3, scope: !32)
// CHECK:STDOUT: !42 = !DILocation(line: 38, column: 3, scope: !32)
// CHECK:STDOUT: !43 = !DILocation(line: 40, column: 3, scope: !32)
// CHECK:STDOUT: !44 = !DILocation(line: 42, column: 3, scope: !32)
// CHECK:STDOUT: !45 = !DILocation(line: 43, column: 3, scope: !32)
// CHECK:STDOUT: !46 = !DILocation(line: 30, column: 3, scope: !32)
// CHECK:STDOUT: !47 = !DILocation(line: 35, column: 5, scope: !32)
// CHECK:STDOUT: !48 = !DILocation(line: 35, column: 3, scope: !32)
// CHECK:STDOUT: !49 = !DILocation(line: 37, column: 5, scope: !32)
// CHECK:STDOUT: !50 = !DILocation(line: 37, column: 3, scope: !32)
// CHECK:STDOUT: !51 = !DILocation(line: 39, column: 5, scope: !32)
// CHECK:STDOUT: !52 = !DILocation(line: 39, column: 3, scope: !32)
// CHECK:STDOUT: !53 = !DILocation(line: 41, column: 5, scope: !32)
// CHECK:STDOUT: !54 = !DILocation(line: 41, column: 3, scope: !32)
// CHECK:STDOUT: !55 = !DILocation(line: 45, column: 3, scope: !32)
// CHECK:STDOUT: !56 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.d4b5665541d5d7a8", scope: null, file: !3, line: 19, type: !57, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !59)
// CHECK:STDOUT: !57 = !DISubroutineType(types: !58)
// CHECK:STDOUT: !58 = !{null, !7}
// CHECK:STDOUT: !59 = !{!60}
// CHECK:STDOUT: !60 = !DILocalVariable(arg: 1, scope: !56, type: !7)
// CHECK:STDOUT: !61 = !DILocation(line: 19, column: 1, scope: !56)
// CHECK:STDOUT: !62 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.29cf1dfeccef7249", scope: null, file: !3, line: 28, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !63)
// CHECK:STDOUT: !63 = !{!64}
// CHECK:STDOUT: !64 = !DILocalVariable(arg: 1, scope: !62, type: !7)
// CHECK:STDOUT: !65 = !DILocation(line: 29, column: 3, scope: !62)
// CHECK:STDOUT: !66 = !DILocation(line: 31, column: 5, scope: !62)
// CHECK:STDOUT: !67 = !DILocation(line: 31, column: 3, scope: !62)
// CHECK:STDOUT: !68 = !DILocation(line: 34, column: 3, scope: !62)
// CHECK:STDOUT: !69 = !DILocation(line: 36, column: 3, scope: !62)
// CHECK:STDOUT: !70 = !DILocation(line: 38, column: 3, scope: !62)
// CHECK:STDOUT: !71 = !DILocation(line: 40, column: 3, scope: !62)
// CHECK:STDOUT: !72 = !DILocation(line: 42, column: 3, scope: !62)
// CHECK:STDOUT: !73 = !DILocation(line: 43, column: 3, scope: !62)
// CHECK:STDOUT: !74 = !DILocation(line: 30, column: 3, scope: !62)
// CHECK:STDOUT: !75 = !DILocation(line: 35, column: 5, scope: !62)
// CHECK:STDOUT: !76 = !DILocation(line: 35, column: 3, scope: !62)
// CHECK:STDOUT: !77 = !DILocation(line: 37, column: 5, scope: !62)
// CHECK:STDOUT: !78 = !DILocation(line: 37, column: 3, scope: !62)
// CHECK:STDOUT: !79 = !DILocation(line: 39, column: 5, scope: !62)
// CHECK:STDOUT: !80 = !DILocation(line: 39, column: 3, scope: !62)
// CHECK:STDOUT: !81 = !DILocation(line: 41, column: 5, scope: !62)
// CHECK:STDOUT: !82 = !DILocation(line: 41, column: 3, scope: !62)
// CHECK:STDOUT: !83 = !DILocation(line: 45, column: 3, scope: !62)
// CHECK:STDOUT: !84 = distinct !DISubprogram(name: "H", linkageName: "_CH.Main.64ccbb8e5d9a0b8e", scope: null, file: !3, line: 22, type: !33, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !85)
// CHECK:STDOUT: !85 = !{!86}
// CHECK:STDOUT: !86 = !DILocalVariable(arg: 1, scope: !84, type: !28)
// CHECK:STDOUT: !87 = !DILocation(line: 23, column: 3, scope: !84)
// CHECK:STDOUT: !88 = distinct !DISubprogram(name: "H", linkageName: "_CH.Main.582d60b54baf6b63", scope: null, file: !3, line: 22, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !89)
// CHECK:STDOUT: !89 = !{!90}
// CHECK:STDOUT: !90 = !DILocalVariable(arg: 1, scope: !88, type: !7)
// CHECK:STDOUT: !91 = !DILocation(line: 23, column: 3, scope: !88)
// CHECK:STDOUT: !92 = distinct !DISubprogram(name: "H", linkageName: "_CH.Main.de5b400758c39a65", scope: null, file: !3, line: 22, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !93)
// CHECK:STDOUT: !93 = !{!94}
// CHECK:STDOUT: !94 = !DILocalVariable(arg: 1, scope: !92, type: !7)
// CHECK:STDOUT: !95 = !DILocation(line: 23, column: 3, scope: !92)
// CHECK:STDOUT: !96 = distinct !DISubprogram(name: "H", linkageName: "_CH.Main.779b9c0f3b54a7f8", scope: null, file: !3, line: 22, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !97)
// CHECK:STDOUT: !97 = !{!98}
// CHECK:STDOUT: !98 = !DILocalVariable(arg: 1, scope: !96, type: !7)
// CHECK:STDOUT: !99 = !DILocation(line: 23, column: 3, scope: !96)
// CHECK:STDOUT: !100 = distinct !DISubprogram(name: "H", linkageName: "_CH.Main.706e9f413f1bae3d", scope: null, file: !3, line: 22, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !101)
// CHECK:STDOUT: !101 = !{!102}
// CHECK:STDOUT: !102 = !DILocalVariable(arg: 1, scope: !100, type: !7)
// CHECK:STDOUT: !103 = !DILocation(line: 23, column: 10, scope: !100)
// CHECK:STDOUT: !104 = !DILocation(line: 23, column: 3, scope: !100)
