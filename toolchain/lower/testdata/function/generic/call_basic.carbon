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

fn F[T: type](_: T) {
}

fn H[T: Core.Copy](x: T) -> T {
  return x;
}

// Simple stress test for single depth of calls with change in types.
// Check definitions are emitted for each specific.
fn G[T: Core.Copy & Core.Destroy](x: T) -> T {
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

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'call_basic.carbon'
// CHECK:STDOUT: source_filename = "call_basic.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: %type.26 = type {}
// CHECK:STDOUT:
// CHECK:STDOUT: @C.val.loc15_43 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_COp.C.Main:Copy.Core"(ptr sret({}) %return, ptr %self) #0 !dbg !44 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %return, ptr align 1 @C.val.loc15_43, i64 0, i1 false), !dbg !49
// CHECK:STDOUT:   ret void, !dbg !49
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.cb8adced19392b99:core.Destroy.Core"(ptr %self) #0 !dbg !51 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core"(ptr %self) #0 !dbg !55 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !57
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.de57ff04fd84f9c8:core.Destroy.Core"(ptr %self) #0 !dbg !59 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !61
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.749b51755634ad17:core.Destroy.Core"(ptr %self) #0 !dbg !63 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !65
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.d088c7eb746488be:core.Destroy.Core"(ptr %self) #0 !dbg !67 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !69
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.dc14d4dd21801791:core.Destroy.Core"(ptr %self) #0 !dbg !71 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !73
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CM.Main() #0 !dbg !75 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %n.var = alloca i32, align 4, !dbg !76
// CHECK:STDOUT:   %p.var = alloca double, align 8, !dbg !78
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %n.var), !dbg !76
// CHECK:STDOUT:   store i32 0, ptr %n.var, align 4, !dbg !76
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %p.var), !dbg !78
// CHECK:STDOUT:   store double 1.000000e+00, ptr %p.var, align 8, !dbg !78
// CHECK:STDOUT:   %.loc52 = load i32, ptr %n.var, align 4, !dbg !80
// CHECK:STDOUT:   call void @_CF.Main.84588f41d61dafba(i32 %.loc52), !dbg !81
// CHECK:STDOUT:   %.loc53_18 = load i32, ptr %n.var, align 4, !dbg !82
// CHECK:STDOUT:   %G.call.loc53 = call i32 @_CG.Main.4d9f279dbc2b9f6f(i32 %.loc53_18), !dbg !83
// CHECK:STDOUT:   %.loc54 = load double, ptr %p.var, align 8, !dbg !88
// CHECK:STDOUT:   call void @_CF.Main.073fbce68be6103e(double %.loc54), !dbg !89
// CHECK:STDOUT:   %.loc55_18 = load double, ptr %p.var, align 8, !dbg !90
// CHECK:STDOUT:   %G.call.loc55 = call double @_CG.Main.cfa50e0c7743522e(double %.loc55_18), !dbg !91
// CHECK:STDOUT:   call void @"_CSelfDestruct.dc14d4dd21801791:core.Destroy.Core"(ptr %p.var), !dbg !78
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %n.var), !dbg !76
// CHECK:STDOUT:   ret void, !dbg !94
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !95 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !97
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.84588f41d61dafba(i32 %_) #0 !dbg !99 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !102
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CG.Main.4d9f279dbc2b9f6f(i32 %x) #0 !dbg !104 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc29_6.3.temp = alloca i32, align 4, !dbg !110
// CHECK:STDOUT:   %.loc31_8.3.temp = alloca i32, align 4, !dbg !113
// CHECK:STDOUT:   %.loc31_9.3.temp = alloca i32, align 4, !dbg !114
// CHECK:STDOUT:   %var_f64.var = alloca double, align 8, !dbg !115
// CHECK:STDOUT:   %ptr_i32.var = alloca ptr, align 8, !dbg !119
// CHECK:STDOUT:   %ptr_f64.var = alloca ptr, align 8, !dbg !123
// CHECK:STDOUT:   %ptr_i8.var = alloca ptr, align 8, !dbg !127
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !131
// CHECK:STDOUT:   %.loc43_6.3.temp = alloca {}, align 1, !dbg !134
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc29_6.3.temp), !dbg !110
// CHECK:STDOUT:   %H.call.loc29 = call i32 @_CH.Main.4d6ceefeb7f772f4(i32 %x), !dbg !110
// CHECK:STDOUT:   store i32 %H.call.loc29, ptr %.loc29_6.3.temp, align 4, !dbg !110
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %.loc29_6.3.temp), !dbg !110
// CHECK:STDOUT:   %H.call.loc30 = call %type.26 @_CH.Main.c3359c2c217ad593(%type.26 zeroinitializer), !dbg !111
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc31_8.3.temp), !dbg !113
// CHECK:STDOUT:   %G.call = call i32 @_CG.Main.4d9f279dbc2b9f6f(i32 %x), !dbg !113
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc31_9.3.temp), !dbg !114
// CHECK:STDOUT:   store i32 %G.call, ptr %.loc31_8.3.temp, align 4, !dbg !113
// CHECK:STDOUT:   %.loc31_8.5 = load i32, ptr %.loc31_8.3.temp, align 4, !dbg !113
// CHECK:STDOUT:   %H.call.loc31 = call i32 @_CH.Main.4d6ceefeb7f772f4(i32 %.loc31_8.5), !dbg !114
// CHECK:STDOUT:   store i32 %H.call.loc31, ptr %.loc31_9.3.temp, align 4, !dbg !114
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %.loc31_9.3.temp), !dbg !114
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %.loc31_8.3.temp), !dbg !113
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %var_f64.var), !dbg !115
// CHECK:STDOUT:   store double poison, ptr %var_f64.var, align 8, !dbg !115
// CHECK:STDOUT:   %.loc35_5 = load double, ptr %var_f64.var, align 8, !dbg !117
// CHECK:STDOUT:   %H.call.loc35 = call double @_CH.Main.c5e4ba542a56b71f(double %.loc35_5), !dbg !118
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_i32.var), !dbg !119
// CHECK:STDOUT:   store ptr poison, ptr %ptr_i32.var, align 8, !dbg !119
// CHECK:STDOUT:   %.loc37_5 = load ptr, ptr %ptr_i32.var, align 8, !dbg !121
// CHECK:STDOUT:   %H.call.loc37 = call ptr @_CH.Main.a76b78461416b4a8(ptr %.loc37_5), !dbg !122
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_f64.var), !dbg !123
// CHECK:STDOUT:   store ptr poison, ptr %ptr_f64.var, align 8, !dbg !123
// CHECK:STDOUT:   %.loc39_5 = load ptr, ptr %ptr_f64.var, align 8, !dbg !125
// CHECK:STDOUT:   %H.call.loc39 = call ptr @_CH.Main.a76b78461416b4a8(ptr %.loc39_5), !dbg !126
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_i8.var), !dbg !127
// CHECK:STDOUT:   store ptr poison, ptr %ptr_i8.var, align 8, !dbg !127
// CHECK:STDOUT:   %.loc41_5 = load ptr, ptr %ptr_i8.var, align 8, !dbg !129
// CHECK:STDOUT:   %H.call.loc41 = call ptr @_CH.Main.a76b78461416b4a8(ptr %.loc41_5), !dbg !130
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !131
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc43_6.3.temp), !dbg !134
// CHECK:STDOUT:   call void @_CH.Main.42513763e01cba2c(ptr %.loc43_6.3.temp, ptr %c.var), !dbg !134
// CHECK:STDOUT:   call void @"_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core"(ptr %.loc43_6.3.temp), !dbg !134
// CHECK:STDOUT:   call void @"_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core"(ptr %c.var), !dbg !131
// CHECK:STDOUT:   call void @"_CSelfDestruct.de57ff04fd84f9c8:core.Destroy.Core"(ptr %ptr_i8.var), !dbg !127
// CHECK:STDOUT:   call void @"_CSelfDestruct.749b51755634ad17:core.Destroy.Core"(ptr %ptr_f64.var), !dbg !123
// CHECK:STDOUT:   call void @"_CSelfDestruct.d088c7eb746488be:core.Destroy.Core"(ptr %ptr_i32.var), !dbg !119
// CHECK:STDOUT:   call void @"_CSelfDestruct.dc14d4dd21801791:core.Destroy.Core"(ptr %var_f64.var), !dbg !115
// CHECK:STDOUT:   ret i32 %x, !dbg !136
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.073fbce68be6103e(double %_) #0 !dbg !138 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !141
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CG.Main.cfa50e0c7743522e(double %x) #0 !dbg !143 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc29_6.3.temp = alloca double, align 8, !dbg !149
// CHECK:STDOUT:   %.loc31_8.3.temp = alloca double, align 8, !dbg !152
// CHECK:STDOUT:   %.loc31_9.3.temp = alloca double, align 8, !dbg !153
// CHECK:STDOUT:   %var_f64.var = alloca double, align 8, !dbg !154
// CHECK:STDOUT:   %ptr_i32.var = alloca ptr, align 8, !dbg !158
// CHECK:STDOUT:   %ptr_f64.var = alloca ptr, align 8, !dbg !162
// CHECK:STDOUT:   %ptr_i8.var = alloca ptr, align 8, !dbg !166
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !170
// CHECK:STDOUT:   %.loc43_6.3.temp = alloca {}, align 1, !dbg !173
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc29_6.3.temp), !dbg !149
// CHECK:STDOUT:   %H.call.loc29 = call double @_CH.Main.c5e4ba542a56b71f(double %x), !dbg !149
// CHECK:STDOUT:   store double %H.call.loc29, ptr %.loc29_6.3.temp, align 8, !dbg !149
// CHECK:STDOUT:   call void @"_CSelfDestruct.dc14d4dd21801791:core.Destroy.Core"(ptr %.loc29_6.3.temp), !dbg !149
// CHECK:STDOUT:   %H.call.loc30 = call %type.26 @_CH.Main.c3359c2c217ad593(%type.26 zeroinitializer), !dbg !150
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc31_8.3.temp), !dbg !152
// CHECK:STDOUT:   %G.call = call double @_CG.Main.cfa50e0c7743522e(double %x), !dbg !152
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc31_9.3.temp), !dbg !153
// CHECK:STDOUT:   store double %G.call, ptr %.loc31_8.3.temp, align 8, !dbg !152
// CHECK:STDOUT:   %.loc31_8.5 = load double, ptr %.loc31_8.3.temp, align 8, !dbg !152
// CHECK:STDOUT:   %H.call.loc31 = call double @_CH.Main.c5e4ba542a56b71f(double %.loc31_8.5), !dbg !153
// CHECK:STDOUT:   store double %H.call.loc31, ptr %.loc31_9.3.temp, align 8, !dbg !153
// CHECK:STDOUT:   call void @"_CSelfDestruct.dc14d4dd21801791:core.Destroy.Core"(ptr %.loc31_9.3.temp), !dbg !153
// CHECK:STDOUT:   call void @"_CSelfDestruct.dc14d4dd21801791:core.Destroy.Core"(ptr %.loc31_8.3.temp), !dbg !152
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %var_f64.var), !dbg !154
// CHECK:STDOUT:   store double poison, ptr %var_f64.var, align 8, !dbg !154
// CHECK:STDOUT:   %.loc35_5 = load double, ptr %var_f64.var, align 8, !dbg !156
// CHECK:STDOUT:   %H.call.loc35 = call double @_CH.Main.c5e4ba542a56b71f(double %.loc35_5), !dbg !157
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_i32.var), !dbg !158
// CHECK:STDOUT:   store ptr poison, ptr %ptr_i32.var, align 8, !dbg !158
// CHECK:STDOUT:   %.loc37_5 = load ptr, ptr %ptr_i32.var, align 8, !dbg !160
// CHECK:STDOUT:   %H.call.loc37 = call ptr @_CH.Main.a76b78461416b4a8(ptr %.loc37_5), !dbg !161
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_f64.var), !dbg !162
// CHECK:STDOUT:   store ptr poison, ptr %ptr_f64.var, align 8, !dbg !162
// CHECK:STDOUT:   %.loc39_5 = load ptr, ptr %ptr_f64.var, align 8, !dbg !164
// CHECK:STDOUT:   %H.call.loc39 = call ptr @_CH.Main.a76b78461416b4a8(ptr %.loc39_5), !dbg !165
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_i8.var), !dbg !166
// CHECK:STDOUT:   store ptr poison, ptr %ptr_i8.var, align 8, !dbg !166
// CHECK:STDOUT:   %.loc41_5 = load ptr, ptr %ptr_i8.var, align 8, !dbg !168
// CHECK:STDOUT:   %H.call.loc41 = call ptr @_CH.Main.a76b78461416b4a8(ptr %.loc41_5), !dbg !169
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !170
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc43_6.3.temp), !dbg !173
// CHECK:STDOUT:   call void @_CH.Main.42513763e01cba2c(ptr %.loc43_6.3.temp, ptr %c.var), !dbg !173
// CHECK:STDOUT:   call void @"_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core"(ptr %.loc43_6.3.temp), !dbg !173
// CHECK:STDOUT:   call void @"_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core"(ptr %c.var), !dbg !170
// CHECK:STDOUT:   call void @"_CSelfDestruct.de57ff04fd84f9c8:core.Destroy.Core"(ptr %ptr_i8.var), !dbg !166
// CHECK:STDOUT:   call void @"_CSelfDestruct.749b51755634ad17:core.Destroy.Core"(ptr %ptr_f64.var), !dbg !162
// CHECK:STDOUT:   call void @"_CSelfDestruct.d088c7eb746488be:core.Destroy.Core"(ptr %ptr_i32.var), !dbg !158
// CHECK:STDOUT:   call void @"_CSelfDestruct.dc14d4dd21801791:core.Destroy.Core"(ptr %var_f64.var), !dbg !154
// CHECK:STDOUT:   ret double %x, !dbg !175
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CH.Main.4d6ceefeb7f772f4(i32 %x) #0 !dbg !177 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %x, !dbg !183
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr %type.26 @_CH.Main.c3359c2c217ad593(%type.26 %x) #0 !dbg !185 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret %type.26 %x, !dbg !191
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CH.Main.c5e4ba542a56b71f(double %x) #0 !dbg !193 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret double %x, !dbg !199
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CH.Main.a76b78461416b4a8(ptr %x) #0 !dbg !201 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret ptr %x, !dbg !207
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CH.Main.42513763e01cba2c(ptr sret({}) %return, ptr %x) #0 !dbg !225 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_COp.C.Main:Copy.Core"(ptr %return, ptr %x), !dbg !230
// CHECK:STDOUT:   ret void, !dbg !231
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 18 }
// CHECK:STDOUT: uselistorder ptr @_CH.Main.4d6ceefeb7f772f4, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CH.Main.c3359c2c217ad593, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CH.Main.c5e4ba542a56b71f, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CH.Main.a76b78461416b4a8, { 5, 2, 0, 4, 3, 1 }
// CHECK:STDOUT: uselistorder ptr @_CH.Main.42513763e01cba2c, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!33}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !32 = !DIFile(filename: "call_basic.carbon", directory: "")
// CHECK:STDOUT: !33 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !32, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !34 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !35 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !36 = !{!35, !35}
// CHECK:STDOUT: !37 = !DISubroutineType(types: !36)
// CHECK:STDOUT: !38 = !{null, !35}
// CHECK:STDOUT: !39 = !DISubroutineType(types: !38)
// CHECK:STDOUT: !40 = !{null}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !42 = !{null, !34}
// CHECK:STDOUT: !43 = !DISubroutineType(types: !42)
// CHECK:STDOUT: !44 = distinct !DISubprogram(name: "Op", linkageName: "_COp.C.Main:Copy.Core", scope: null, file: !32, line: 15, type: !37, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !50)
// CHECK:STDOUT: !45 = !DILocalVariable(arg: 1, scope: !44, type: !35)
// CHECK:STDOUT: !49 = !DILocation(line: 15, column: 34, scope: !44)
// CHECK:STDOUT: !50 = !{!45}
// CHECK:STDOUT: !51 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.cb8adced19392b99:core.Destroy.Core", scope: null, file: !32, line: 43, type: !39, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !54)
// CHECK:STDOUT: !52 = !DILocalVariable(arg: 1, scope: !51, type: !35)
// CHECK:STDOUT: !53 = !DILocation(line: 43, column: 3, scope: !51)
// CHECK:STDOUT: !54 = !{!52}
// CHECK:STDOUT: !55 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core", scope: null, file: !32, line: 43, type: !39, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !58)
// CHECK:STDOUT: !56 = !DILocalVariable(arg: 1, scope: !55, type: !35)
// CHECK:STDOUT: !57 = !DILocation(line: 43, column: 3, scope: !55)
// CHECK:STDOUT: !58 = !{!56}
// CHECK:STDOUT: !59 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.de57ff04fd84f9c8:core.Destroy.Core", scope: null, file: !32, line: 40, type: !39, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !62)
// CHECK:STDOUT: !60 = !DILocalVariable(arg: 1, scope: !59, type: !35)
// CHECK:STDOUT: !61 = !DILocation(line: 40, column: 3, scope: !59)
// CHECK:STDOUT: !62 = !{!60}
// CHECK:STDOUT: !63 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.749b51755634ad17:core.Destroy.Core", scope: null, file: !32, line: 38, type: !39, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !66)
// CHECK:STDOUT: !64 = !DILocalVariable(arg: 1, scope: !63, type: !35)
// CHECK:STDOUT: !65 = !DILocation(line: 38, column: 3, scope: !63)
// CHECK:STDOUT: !66 = !{!64}
// CHECK:STDOUT: !67 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.d088c7eb746488be:core.Destroy.Core", scope: null, file: !32, line: 36, type: !39, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !70)
// CHECK:STDOUT: !68 = !DILocalVariable(arg: 1, scope: !67, type: !35)
// CHECK:STDOUT: !69 = !DILocation(line: 36, column: 3, scope: !67)
// CHECK:STDOUT: !70 = !{!68}
// CHECK:STDOUT: !71 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.dc14d4dd21801791:core.Destroy.Core", scope: null, file: !32, line: 34, type: !39, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !74)
// CHECK:STDOUT: !72 = !DILocalVariable(arg: 1, scope: !71, type: !35)
// CHECK:STDOUT: !73 = !DILocation(line: 34, column: 3, scope: !71)
// CHECK:STDOUT: !74 = !{!72}
// CHECK:STDOUT: !75 = distinct !DISubprogram(name: "M", linkageName: "_CM.Main", scope: null, file: !32, line: 48, type: !41, spFlags: DISPFlagDefinition, unit: !33)
// CHECK:STDOUT: !76 = !DILocation(line: 49, column: 3, scope: !75)
// CHECK:STDOUT: !78 = !DILocation(line: 50, column: 3, scope: !75)
// CHECK:STDOUT: !80 = !DILocation(line: 52, column: 5, scope: !75)
// CHECK:STDOUT: !81 = !DILocation(line: 52, column: 3, scope: !75)
// CHECK:STDOUT: !82 = !DILocation(line: 53, column: 18, scope: !75)
// CHECK:STDOUT: !83 = !DILocation(line: 53, column: 16, scope: !75)
// CHECK:STDOUT: !84 = !{!34, !34}
// CHECK:STDOUT: !85 = !DISubroutineType(types: !84)
// CHECK:STDOUT: !88 = !DILocation(line: 54, column: 5, scope: !75)
// CHECK:STDOUT: !89 = !DILocation(line: 54, column: 3, scope: !75)
// CHECK:STDOUT: !90 = !DILocation(line: 55, column: 18, scope: !75)
// CHECK:STDOUT: !91 = !DILocation(line: 55, column: 16, scope: !75)
// CHECK:STDOUT: !94 = !DILocation(line: 48, column: 1, scope: !75)
// CHECK:STDOUT: !95 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f58524831c37807e:core.Destroy.Core", scope: null, file: !32, line: 53, type: !43, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !98)
// CHECK:STDOUT: !96 = !DILocalVariable(arg: 1, scope: !95, type: !34)
// CHECK:STDOUT: !97 = !DILocation(line: 53, column: 16, scope: !95)
// CHECK:STDOUT: !98 = !{!96}
// CHECK:STDOUT: !99 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.84588f41d61dafba", scope: null, file: !32, line: 19, type: !43, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !103)
// CHECK:STDOUT: !100 = !DILocalVariable(arg: 1, scope: !99, type: !34)
// CHECK:STDOUT: !102 = !DILocation(line: 19, column: 1, scope: !99)
// CHECK:STDOUT: !103 = !{!100}
// CHECK:STDOUT: !104 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.4d9f279dbc2b9f6f", scope: null, file: !32, line: 28, type: !85, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !137)
// CHECK:STDOUT: !105 = !DILocalVariable(arg: 1, scope: !104, type: !34)
// CHECK:STDOUT: !110 = !DILocation(line: 29, column: 3, scope: !104)
// CHECK:STDOUT: !111 = !DILocation(line: 30, column: 3, scope: !104)
// CHECK:STDOUT: !113 = !DILocation(line: 31, column: 5, scope: !104)
// CHECK:STDOUT: !114 = !DILocation(line: 31, column: 3, scope: !104)
// CHECK:STDOUT: !115 = !DILocation(line: 34, column: 3, scope: !104)
// CHECK:STDOUT: !117 = !DILocation(line: 35, column: 5, scope: !104)
// CHECK:STDOUT: !118 = !DILocation(line: 35, column: 3, scope: !104)
// CHECK:STDOUT: !119 = !DILocation(line: 36, column: 3, scope: !104)
// CHECK:STDOUT: !121 = !DILocation(line: 37, column: 5, scope: !104)
// CHECK:STDOUT: !122 = !DILocation(line: 37, column: 3, scope: !104)
// CHECK:STDOUT: !123 = !DILocation(line: 38, column: 3, scope: !104)
// CHECK:STDOUT: !125 = !DILocation(line: 39, column: 5, scope: !104)
// CHECK:STDOUT: !126 = !DILocation(line: 39, column: 3, scope: !104)
// CHECK:STDOUT: !127 = !DILocation(line: 40, column: 3, scope: !104)
// CHECK:STDOUT: !129 = !DILocation(line: 41, column: 5, scope: !104)
// CHECK:STDOUT: !130 = !DILocation(line: 41, column: 3, scope: !104)
// CHECK:STDOUT: !131 = !DILocation(line: 42, column: 3, scope: !104)
// CHECK:STDOUT: !134 = !DILocation(line: 43, column: 3, scope: !104)
// CHECK:STDOUT: !136 = !DILocation(line: 45, column: 3, scope: !104)
// CHECK:STDOUT: !137 = !{!105}
// CHECK:STDOUT: !138 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.073fbce68be6103e", scope: null, file: !32, line: 19, type: !39, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !142)
// CHECK:STDOUT: !139 = !DILocalVariable(arg: 1, scope: !138, type: !35)
// CHECK:STDOUT: !141 = !DILocation(line: 19, column: 1, scope: !138)
// CHECK:STDOUT: !142 = !{!139}
// CHECK:STDOUT: !143 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.cfa50e0c7743522e", scope: null, file: !32, line: 28, type: !37, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !176)
// CHECK:STDOUT: !144 = !DILocalVariable(arg: 1, scope: !143, type: !35)
// CHECK:STDOUT: !149 = !DILocation(line: 29, column: 3, scope: !143)
// CHECK:STDOUT: !150 = !DILocation(line: 30, column: 3, scope: !143)
// CHECK:STDOUT: !152 = !DILocation(line: 31, column: 5, scope: !143)
// CHECK:STDOUT: !153 = !DILocation(line: 31, column: 3, scope: !143)
// CHECK:STDOUT: !154 = !DILocation(line: 34, column: 3, scope: !143)
// CHECK:STDOUT: !156 = !DILocation(line: 35, column: 5, scope: !143)
// CHECK:STDOUT: !157 = !DILocation(line: 35, column: 3, scope: !143)
// CHECK:STDOUT: !158 = !DILocation(line: 36, column: 3, scope: !143)
// CHECK:STDOUT: !160 = !DILocation(line: 37, column: 5, scope: !143)
// CHECK:STDOUT: !161 = !DILocation(line: 37, column: 3, scope: !143)
// CHECK:STDOUT: !162 = !DILocation(line: 38, column: 3, scope: !143)
// CHECK:STDOUT: !164 = !DILocation(line: 39, column: 5, scope: !143)
// CHECK:STDOUT: !165 = !DILocation(line: 39, column: 3, scope: !143)
// CHECK:STDOUT: !166 = !DILocation(line: 40, column: 3, scope: !143)
// CHECK:STDOUT: !168 = !DILocation(line: 41, column: 5, scope: !143)
// CHECK:STDOUT: !169 = !DILocation(line: 41, column: 3, scope: !143)
// CHECK:STDOUT: !170 = !DILocation(line: 42, column: 3, scope: !143)
// CHECK:STDOUT: !173 = !DILocation(line: 43, column: 3, scope: !143)
// CHECK:STDOUT: !175 = !DILocation(line: 45, column: 3, scope: !143)
// CHECK:STDOUT: !176 = !{!144}
// CHECK:STDOUT: !177 = distinct !DISubprogram(name: "H", linkageName: "_CH.Main.4d6ceefeb7f772f4", scope: null, file: !32, line: 22, type: !85, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !184)
// CHECK:STDOUT: !178 = !DILocalVariable(arg: 1, scope: !177, type: !34)
// CHECK:STDOUT: !183 = !DILocation(line: 23, column: 3, scope: !177)
// CHECK:STDOUT: !184 = !{!178}
// CHECK:STDOUT: !185 = distinct !DISubprogram(name: "H", linkageName: "_CH.Main.c3359c2c217ad593", scope: null, file: !32, line: 22, type: !37, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !192)
// CHECK:STDOUT: !186 = !DILocalVariable(arg: 1, scope: !185, type: !35)
// CHECK:STDOUT: !191 = !DILocation(line: 23, column: 3, scope: !185)
// CHECK:STDOUT: !192 = !{!186}
// CHECK:STDOUT: !193 = distinct !DISubprogram(name: "H", linkageName: "_CH.Main.c5e4ba542a56b71f", scope: null, file: !32, line: 22, type: !37, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !200)
// CHECK:STDOUT: !194 = !DILocalVariable(arg: 1, scope: !193, type: !35)
// CHECK:STDOUT: !199 = !DILocation(line: 23, column: 3, scope: !193)
// CHECK:STDOUT: !200 = !{!194}
// CHECK:STDOUT: !201 = distinct !DISubprogram(name: "H", linkageName: "_CH.Main.a76b78461416b4a8", scope: null, file: !32, line: 22, type: !37, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !208)
// CHECK:STDOUT: !202 = !DILocalVariable(arg: 1, scope: !201, type: !35)
// CHECK:STDOUT: !207 = !DILocation(line: 23, column: 3, scope: !201)
// CHECK:STDOUT: !208 = !{!202}
// CHECK:STDOUT: !225 = distinct !DISubprogram(name: "H", linkageName: "_CH.Main.42513763e01cba2c", scope: null, file: !32, line: 22, type: !37, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !232)
// CHECK:STDOUT: !226 = !DILocalVariable(arg: 1, scope: !225, type: !35)
// CHECK:STDOUT: !230 = !DILocation(line: 23, column: 10, scope: !225)
// CHECK:STDOUT: !231 = !DILocation(line: 23, column: 3, scope: !225)
// CHECK:STDOUT: !232 = !{!226}
// CHECK:STDOUT:
