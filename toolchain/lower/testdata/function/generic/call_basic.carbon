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
    fn Op(unused self) -> Self { return {}; }
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
// CHECK:STDOUT: @C.val.loc15_43 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_COp.C.Main:Copy.Core"(ptr sret({}) %return, ptr %self) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %return, ptr align 1 @C.val.loc15_43, i64 0, i1 false), !dbg !10
// CHECK:STDOUT:   ret void, !dbg !10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.807eb4f2d3ed7e54:core.Destroy.Core"(ptr %self) #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.30d8007fdcc57f83:core.Destroy.Core"(ptr %self) #0 !dbg !17 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CM.Main() #0 !dbg !21 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %n.var = alloca i32, align 4, !dbg !24
// CHECK:STDOUT:   %p.var = alloca double, align 8, !dbg !25
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %n.var), !dbg !24
// CHECK:STDOUT:   store i32 0, ptr %n.var, align 4, !dbg !24
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %p.var), !dbg !25
// CHECK:STDOUT:   store double 1.000000e+00, ptr %p.var, align 8, !dbg !25
// CHECK:STDOUT:   %.loc52 = load i32, ptr %n.var, align 4, !dbg !26
// CHECK:STDOUT:   call void @_CF.Main.84588f41d61dafba(i32 %.loc52), !dbg !27
// CHECK:STDOUT:   %.loc53_18 = load i32, ptr %n.var, align 4, !dbg !28
// CHECK:STDOUT:   %G.call.loc53 = call i32 @_CG.Main.58016a73bff04416(i32 %.loc53_18), !dbg !29
// CHECK:STDOUT:   %.loc54 = load double, ptr %p.var, align 8, !dbg !30
// CHECK:STDOUT:   call void @_CF.Main.073fbce68be6103e(double %.loc54), !dbg !31
// CHECK:STDOUT:   %.loc55_18 = load double, ptr %p.var, align 8, !dbg !32
// CHECK:STDOUT:   %G.call.loc55 = call double @_CG.Main.fa1f0912a5bbe922(double %.loc55_18), !dbg !33
// CHECK:STDOUT:   call void @"_COp.30d8007fdcc57f83:core.Destroy.Core"(ptr %p.var), !dbg !25
// CHECK:STDOUT:   call void @"_COp.5e27612b9dd31a14:core.Destroy.Core"(ptr %n.var), !dbg !24
// CHECK:STDOUT:   ret void, !dbg !34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.5e27612b9dd31a14:core.Destroy.Core"(ptr %self) #0 !dbg !35 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.84588f41d61dafba(i32 %_) #0 !dbg !42 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CG.Main.58016a73bff04416(i32 %x) #0 !dbg !46 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc29_6.3.temp = alloca i32, align 4, !dbg !51
// CHECK:STDOUT:   %.loc31_8.3.temp = alloca i32, align 4, !dbg !52
// CHECK:STDOUT:   %.loc31_9.3.temp = alloca i32, align 4, !dbg !53
// CHECK:STDOUT:   %var_f64.var = alloca double, align 8, !dbg !54
// CHECK:STDOUT:   %ptr_i32.var = alloca ptr, align 8, !dbg !55
// CHECK:STDOUT:   %ptr_f64.var = alloca ptr, align 8, !dbg !56
// CHECK:STDOUT:   %ptr_i8.var = alloca ptr, align 8, !dbg !57
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !58
// CHECK:STDOUT:   %.loc43_6.3.temp = alloca {}, align 1, !dbg !59
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc29_6.3.temp), !dbg !51
// CHECK:STDOUT:   %H.call.loc29 = call i32 @_CH.Main.5bd8d5767580997a(i32 %x), !dbg !51
// CHECK:STDOUT:   store i32 %H.call.loc29, ptr %.loc29_6.3.temp, align 4, !dbg !51
// CHECK:STDOUT:   %H.call.loc30 = call %type @_CH.Main.ee509b43cf4eb3c5(%type zeroinitializer), !dbg !60
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc31_8.3.temp), !dbg !52
// CHECK:STDOUT:   %G.call = call i32 @_CG.Main.58016a73bff04416(i32 %x), !dbg !52
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc31_9.3.temp), !dbg !53
// CHECK:STDOUT:   store i32 %G.call, ptr %.loc31_8.3.temp, align 4, !dbg !52
// CHECK:STDOUT:   %.loc31_8.5 = load i32, ptr %.loc31_8.3.temp, align 4, !dbg !52
// CHECK:STDOUT:   %H.call.loc31 = call i32 @_CH.Main.5bd8d5767580997a(i32 %.loc31_8.5), !dbg !53
// CHECK:STDOUT:   store i32 %H.call.loc31, ptr %.loc31_9.3.temp, align 4, !dbg !53
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %var_f64.var), !dbg !54
// CHECK:STDOUT:   store double poison, ptr %var_f64.var, align 8, !dbg !54
// CHECK:STDOUT:   %.loc35_5 = load double, ptr %var_f64.var, align 8, !dbg !61
// CHECK:STDOUT:   %H.call.loc35 = call double @_CH.Main.c27b765f2159c9a0(double %.loc35_5), !dbg !62
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_i32.var), !dbg !55
// CHECK:STDOUT:   store ptr poison, ptr %ptr_i32.var, align 8, !dbg !55
// CHECK:STDOUT:   %.loc37_5 = load ptr, ptr %ptr_i32.var, align 8, !dbg !63
// CHECK:STDOUT:   %H.call.loc37 = call ptr @_CH.Main.2621742a01881e29(ptr %.loc37_5), !dbg !64
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_f64.var), !dbg !56
// CHECK:STDOUT:   store ptr poison, ptr %ptr_f64.var, align 8, !dbg !56
// CHECK:STDOUT:   %.loc39_5 = load ptr, ptr %ptr_f64.var, align 8, !dbg !65
// CHECK:STDOUT:   %H.call.loc39 = call ptr @_CH.Main.2621742a01881e29(ptr %.loc39_5), !dbg !66
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_i8.var), !dbg !57
// CHECK:STDOUT:   store ptr poison, ptr %ptr_i8.var, align 8, !dbg !57
// CHECK:STDOUT:   %.loc41_5 = load ptr, ptr %ptr_i8.var, align 8, !dbg !67
// CHECK:STDOUT:   %H.call.loc41 = call ptr @_CH.Main.2621742a01881e29(ptr %.loc41_5), !dbg !68
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !58
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc43_6.3.temp), !dbg !59
// CHECK:STDOUT:   call void @_CH.Main.4d597ec621e4ba00(ptr %.loc43_6.3.temp, ptr %c.var), !dbg !59
// CHECK:STDOUT:   call void @"_COp.807eb4f2d3ed7e54:core.Destroy.Core"(ptr %.loc43_6.3.temp), !dbg !59
// CHECK:STDOUT:   call void @"_COp.807eb4f2d3ed7e54:core.Destroy.Core"(ptr %c.var), !dbg !58
// CHECK:STDOUT:   call void @"_COp.30d8007fdcc57f83:core.Destroy.Core"(ptr %var_f64.var), !dbg !54
// CHECK:STDOUT:   call void @"_COp.5e27612b9dd31a14:core.Destroy.Core"(ptr %.loc31_9.3.temp), !dbg !53
// CHECK:STDOUT:   call void @"_COp.5e27612b9dd31a14:core.Destroy.Core"(ptr %.loc31_8.3.temp), !dbg !52
// CHECK:STDOUT:   call void @"_COp.5e27612b9dd31a14:core.Destroy.Core"(ptr %.loc29_6.3.temp), !dbg !51
// CHECK:STDOUT:   ret i32 %x, !dbg !69
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.073fbce68be6103e(double %_) #0 !dbg !70 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !73
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CG.Main.fa1f0912a5bbe922(double %x) #0 !dbg !74 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc29_6.3.temp = alloca double, align 8, !dbg !77
// CHECK:STDOUT:   %.loc31_8.3.temp = alloca double, align 8, !dbg !78
// CHECK:STDOUT:   %.loc31_9.3.temp = alloca double, align 8, !dbg !79
// CHECK:STDOUT:   %var_f64.var = alloca double, align 8, !dbg !80
// CHECK:STDOUT:   %ptr_i32.var = alloca ptr, align 8, !dbg !81
// CHECK:STDOUT:   %ptr_f64.var = alloca ptr, align 8, !dbg !82
// CHECK:STDOUT:   %ptr_i8.var = alloca ptr, align 8, !dbg !83
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !84
// CHECK:STDOUT:   %.loc43_6.3.temp = alloca {}, align 1, !dbg !85
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc29_6.3.temp), !dbg !77
// CHECK:STDOUT:   %H.call.loc29 = call double @_CH.Main.c27b765f2159c9a0(double %x), !dbg !77
// CHECK:STDOUT:   store double %H.call.loc29, ptr %.loc29_6.3.temp, align 8, !dbg !77
// CHECK:STDOUT:   %H.call.loc30 = call %type @_CH.Main.ee509b43cf4eb3c5(%type zeroinitializer), !dbg !86
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc31_8.3.temp), !dbg !78
// CHECK:STDOUT:   %G.call = call double @_CG.Main.fa1f0912a5bbe922(double %x), !dbg !78
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc31_9.3.temp), !dbg !79
// CHECK:STDOUT:   store double %G.call, ptr %.loc31_8.3.temp, align 8, !dbg !78
// CHECK:STDOUT:   %.loc31_8.5 = load double, ptr %.loc31_8.3.temp, align 8, !dbg !78
// CHECK:STDOUT:   %H.call.loc31 = call double @_CH.Main.c27b765f2159c9a0(double %.loc31_8.5), !dbg !79
// CHECK:STDOUT:   store double %H.call.loc31, ptr %.loc31_9.3.temp, align 8, !dbg !79
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %var_f64.var), !dbg !80
// CHECK:STDOUT:   store double poison, ptr %var_f64.var, align 8, !dbg !80
// CHECK:STDOUT:   %.loc35_5 = load double, ptr %var_f64.var, align 8, !dbg !87
// CHECK:STDOUT:   %H.call.loc35 = call double @_CH.Main.c27b765f2159c9a0(double %.loc35_5), !dbg !88
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_i32.var), !dbg !81
// CHECK:STDOUT:   store ptr poison, ptr %ptr_i32.var, align 8, !dbg !81
// CHECK:STDOUT:   %.loc37_5 = load ptr, ptr %ptr_i32.var, align 8, !dbg !89
// CHECK:STDOUT:   %H.call.loc37 = call ptr @_CH.Main.2621742a01881e29(ptr %.loc37_5), !dbg !90
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_f64.var), !dbg !82
// CHECK:STDOUT:   store ptr poison, ptr %ptr_f64.var, align 8, !dbg !82
// CHECK:STDOUT:   %.loc39_5 = load ptr, ptr %ptr_f64.var, align 8, !dbg !91
// CHECK:STDOUT:   %H.call.loc39 = call ptr @_CH.Main.2621742a01881e29(ptr %.loc39_5), !dbg !92
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_i8.var), !dbg !83
// CHECK:STDOUT:   store ptr poison, ptr %ptr_i8.var, align 8, !dbg !83
// CHECK:STDOUT:   %.loc41_5 = load ptr, ptr %ptr_i8.var, align 8, !dbg !93
// CHECK:STDOUT:   %H.call.loc41 = call ptr @_CH.Main.2621742a01881e29(ptr %.loc41_5), !dbg !94
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !84
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc43_6.3.temp), !dbg !85
// CHECK:STDOUT:   call void @_CH.Main.4d597ec621e4ba00(ptr %.loc43_6.3.temp, ptr %c.var), !dbg !85
// CHECK:STDOUT:   call void @"_COp.807eb4f2d3ed7e54:core.Destroy.Core"(ptr %.loc43_6.3.temp), !dbg !85
// CHECK:STDOUT:   call void @"_COp.807eb4f2d3ed7e54:core.Destroy.Core"(ptr %c.var), !dbg !84
// CHECK:STDOUT:   call void @"_COp.30d8007fdcc57f83:core.Destroy.Core"(ptr %var_f64.var), !dbg !80
// CHECK:STDOUT:   call void @"_COp.30d8007fdcc57f83:core.Destroy.Core"(ptr %.loc31_9.3.temp), !dbg !79
// CHECK:STDOUT:   call void @"_COp.30d8007fdcc57f83:core.Destroy.Core"(ptr %.loc31_8.3.temp), !dbg !78
// CHECK:STDOUT:   call void @"_COp.30d8007fdcc57f83:core.Destroy.Core"(ptr %.loc29_6.3.temp), !dbg !77
// CHECK:STDOUT:   ret double %x, !dbg !95
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CH.Main.5bd8d5767580997a(i32 %x) #0 !dbg !96 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %x, !dbg !99
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr %type @_CH.Main.ee509b43cf4eb3c5(%type %x) #0 !dbg !100 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret %type %x, !dbg !103
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CH.Main.c27b765f2159c9a0(double %x) #0 !dbg !104 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret double %x, !dbg !107
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CH.Main.2621742a01881e29(ptr %x) #0 !dbg !108 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret ptr %x, !dbg !111
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CH.Main.4d597ec621e4ba00(ptr sret({}) %return, ptr %x) #0 !dbg !112 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_COp.C.Main:Copy.Core"(ptr %return, ptr %x), !dbg !115
// CHECK:STDOUT:   ret void, !dbg !116
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 18 }
// CHECK:STDOUT: uselistorder ptr @_CH.Main.5bd8d5767580997a, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CH.Main.ee509b43cf4eb3c5, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CH.Main.c27b765f2159c9a0, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CH.Main.2621742a01881e29, { 5, 2, 0, 4, 3, 1 }
// CHECK:STDOUT: uselistorder ptr @_CH.Main.4d597ec621e4ba00, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "call_basic.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Op", linkageName: "_COp.C.Main:Copy.Core", scope: null, file: !1, line: 15, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !8)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7, !7}
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{!9}
// CHECK:STDOUT: !9 = !DILocalVariable(arg: 1, scope: !4, type: !7)
// CHECK:STDOUT: !10 = !DILocation(line: 15, column: 34, scope: !4)
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "Op", linkageName: "_COp.807eb4f2d3ed7e54:core.Destroy.Core", scope: null, file: !1, line: 43, type: !12, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !14)
// CHECK:STDOUT: !12 = !DISubroutineType(types: !13)
// CHECK:STDOUT: !13 = !{null, !7}
// CHECK:STDOUT: !14 = !{!15}
// CHECK:STDOUT: !15 = !DILocalVariable(arg: 1, scope: !11, type: !7)
// CHECK:STDOUT: !16 = !DILocation(line: 43, column: 3, scope: !11)
// CHECK:STDOUT: !17 = distinct !DISubprogram(name: "Op", linkageName: "_COp.30d8007fdcc57f83:core.Destroy.Core", scope: null, file: !1, line: 34, type: !12, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !18)
// CHECK:STDOUT: !18 = !{!19}
// CHECK:STDOUT: !19 = !DILocalVariable(arg: 1, scope: !17, type: !7)
// CHECK:STDOUT: !20 = !DILocation(line: 34, column: 3, scope: !17)
// CHECK:STDOUT: !21 = distinct !DISubprogram(name: "M", linkageName: "_CM.Main", scope: null, file: !1, line: 48, type: !22, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !22 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !23 = !{null}
// CHECK:STDOUT: !24 = !DILocation(line: 49, column: 3, scope: !21)
// CHECK:STDOUT: !25 = !DILocation(line: 50, column: 3, scope: !21)
// CHECK:STDOUT: !26 = !DILocation(line: 52, column: 5, scope: !21)
// CHECK:STDOUT: !27 = !DILocation(line: 52, column: 3, scope: !21)
// CHECK:STDOUT: !28 = !DILocation(line: 53, column: 18, scope: !21)
// CHECK:STDOUT: !29 = !DILocation(line: 53, column: 16, scope: !21)
// CHECK:STDOUT: !30 = !DILocation(line: 54, column: 5, scope: !21)
// CHECK:STDOUT: !31 = !DILocation(line: 54, column: 3, scope: !21)
// CHECK:STDOUT: !32 = !DILocation(line: 55, column: 18, scope: !21)
// CHECK:STDOUT: !33 = !DILocation(line: 55, column: 16, scope: !21)
// CHECK:STDOUT: !34 = !DILocation(line: 48, column: 1, scope: !21)
// CHECK:STDOUT: !35 = distinct !DISubprogram(name: "Op", linkageName: "_COp.5e27612b9dd31a14:core.Destroy.Core", scope: null, file: !1, line: 53, type: !36, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !39)
// CHECK:STDOUT: !36 = !DISubroutineType(types: !37)
// CHECK:STDOUT: !37 = !{null, !38}
// CHECK:STDOUT: !38 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !39 = !{!40}
// CHECK:STDOUT: !40 = !DILocalVariable(arg: 1, scope: !35, type: !38)
// CHECK:STDOUT: !41 = !DILocation(line: 53, column: 16, scope: !35)
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.84588f41d61dafba", scope: null, file: !1, line: 19, type: !36, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !43)
// CHECK:STDOUT: !43 = !{!44}
// CHECK:STDOUT: !44 = !DILocalVariable(arg: 1, scope: !42, type: !38)
// CHECK:STDOUT: !45 = !DILocation(line: 19, column: 1, scope: !42)
// CHECK:STDOUT: !46 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.58016a73bff04416", scope: null, file: !1, line: 28, type: !47, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !49)
// CHECK:STDOUT: !47 = !DISubroutineType(types: !48)
// CHECK:STDOUT: !48 = !{!38, !38}
// CHECK:STDOUT: !49 = !{!50}
// CHECK:STDOUT: !50 = !DILocalVariable(arg: 1, scope: !46, type: !38)
// CHECK:STDOUT: !51 = !DILocation(line: 29, column: 3, scope: !46)
// CHECK:STDOUT: !52 = !DILocation(line: 31, column: 5, scope: !46)
// CHECK:STDOUT: !53 = !DILocation(line: 31, column: 3, scope: !46)
// CHECK:STDOUT: !54 = !DILocation(line: 34, column: 3, scope: !46)
// CHECK:STDOUT: !55 = !DILocation(line: 36, column: 3, scope: !46)
// CHECK:STDOUT: !56 = !DILocation(line: 38, column: 3, scope: !46)
// CHECK:STDOUT: !57 = !DILocation(line: 40, column: 3, scope: !46)
// CHECK:STDOUT: !58 = !DILocation(line: 42, column: 3, scope: !46)
// CHECK:STDOUT: !59 = !DILocation(line: 43, column: 3, scope: !46)
// CHECK:STDOUT: !60 = !DILocation(line: 30, column: 3, scope: !46)
// CHECK:STDOUT: !61 = !DILocation(line: 35, column: 5, scope: !46)
// CHECK:STDOUT: !62 = !DILocation(line: 35, column: 3, scope: !46)
// CHECK:STDOUT: !63 = !DILocation(line: 37, column: 5, scope: !46)
// CHECK:STDOUT: !64 = !DILocation(line: 37, column: 3, scope: !46)
// CHECK:STDOUT: !65 = !DILocation(line: 39, column: 5, scope: !46)
// CHECK:STDOUT: !66 = !DILocation(line: 39, column: 3, scope: !46)
// CHECK:STDOUT: !67 = !DILocation(line: 41, column: 5, scope: !46)
// CHECK:STDOUT: !68 = !DILocation(line: 41, column: 3, scope: !46)
// CHECK:STDOUT: !69 = !DILocation(line: 45, column: 3, scope: !46)
// CHECK:STDOUT: !70 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.073fbce68be6103e", scope: null, file: !1, line: 19, type: !12, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !71)
// CHECK:STDOUT: !71 = !{!72}
// CHECK:STDOUT: !72 = !DILocalVariable(arg: 1, scope: !70, type: !7)
// CHECK:STDOUT: !73 = !DILocation(line: 19, column: 1, scope: !70)
// CHECK:STDOUT: !74 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.fa1f0912a5bbe922", scope: null, file: !1, line: 28, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !75)
// CHECK:STDOUT: !75 = !{!76}
// CHECK:STDOUT: !76 = !DILocalVariable(arg: 1, scope: !74, type: !7)
// CHECK:STDOUT: !77 = !DILocation(line: 29, column: 3, scope: !74)
// CHECK:STDOUT: !78 = !DILocation(line: 31, column: 5, scope: !74)
// CHECK:STDOUT: !79 = !DILocation(line: 31, column: 3, scope: !74)
// CHECK:STDOUT: !80 = !DILocation(line: 34, column: 3, scope: !74)
// CHECK:STDOUT: !81 = !DILocation(line: 36, column: 3, scope: !74)
// CHECK:STDOUT: !82 = !DILocation(line: 38, column: 3, scope: !74)
// CHECK:STDOUT: !83 = !DILocation(line: 40, column: 3, scope: !74)
// CHECK:STDOUT: !84 = !DILocation(line: 42, column: 3, scope: !74)
// CHECK:STDOUT: !85 = !DILocation(line: 43, column: 3, scope: !74)
// CHECK:STDOUT: !86 = !DILocation(line: 30, column: 3, scope: !74)
// CHECK:STDOUT: !87 = !DILocation(line: 35, column: 5, scope: !74)
// CHECK:STDOUT: !88 = !DILocation(line: 35, column: 3, scope: !74)
// CHECK:STDOUT: !89 = !DILocation(line: 37, column: 5, scope: !74)
// CHECK:STDOUT: !90 = !DILocation(line: 37, column: 3, scope: !74)
// CHECK:STDOUT: !91 = !DILocation(line: 39, column: 5, scope: !74)
// CHECK:STDOUT: !92 = !DILocation(line: 39, column: 3, scope: !74)
// CHECK:STDOUT: !93 = !DILocation(line: 41, column: 5, scope: !74)
// CHECK:STDOUT: !94 = !DILocation(line: 41, column: 3, scope: !74)
// CHECK:STDOUT: !95 = !DILocation(line: 45, column: 3, scope: !74)
// CHECK:STDOUT: !96 = distinct !DISubprogram(name: "H", linkageName: "_CH.Main.5bd8d5767580997a", scope: null, file: !1, line: 22, type: !47, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !97)
// CHECK:STDOUT: !97 = !{!98}
// CHECK:STDOUT: !98 = !DILocalVariable(arg: 1, scope: !96, type: !38)
// CHECK:STDOUT: !99 = !DILocation(line: 23, column: 3, scope: !96)
// CHECK:STDOUT: !100 = distinct !DISubprogram(name: "H", linkageName: "_CH.Main.ee509b43cf4eb3c5", scope: null, file: !1, line: 22, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !101)
// CHECK:STDOUT: !101 = !{!102}
// CHECK:STDOUT: !102 = !DILocalVariable(arg: 1, scope: !100, type: !7)
// CHECK:STDOUT: !103 = !DILocation(line: 23, column: 3, scope: !100)
// CHECK:STDOUT: !104 = distinct !DISubprogram(name: "H", linkageName: "_CH.Main.c27b765f2159c9a0", scope: null, file: !1, line: 22, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !105)
// CHECK:STDOUT: !105 = !{!106}
// CHECK:STDOUT: !106 = !DILocalVariable(arg: 1, scope: !104, type: !7)
// CHECK:STDOUT: !107 = !DILocation(line: 23, column: 3, scope: !104)
// CHECK:STDOUT: !108 = distinct !DISubprogram(name: "H", linkageName: "_CH.Main.2621742a01881e29", scope: null, file: !1, line: 22, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !109)
// CHECK:STDOUT: !109 = !{!110}
// CHECK:STDOUT: !110 = !DILocalVariable(arg: 1, scope: !108, type: !7)
// CHECK:STDOUT: !111 = !DILocation(line: 23, column: 3, scope: !108)
// CHECK:STDOUT: !112 = distinct !DISubprogram(name: "H", linkageName: "_CH.Main.4d597ec621e4ba00", scope: null, file: !1, line: 22, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !113)
// CHECK:STDOUT: !113 = !{!114}
// CHECK:STDOUT: !114 = !DILocalVariable(arg: 1, scope: !112, type: !7)
// CHECK:STDOUT: !115 = !DILocation(line: 23, column: 10, scope: !112)
// CHECK:STDOUT: !116 = !DILocation(line: 23, column: 3, scope: !112)
