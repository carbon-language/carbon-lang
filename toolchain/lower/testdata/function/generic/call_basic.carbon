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
// CHECK:STDOUT: %type.28 = type {}
// CHECK:STDOUT:
// CHECK:STDOUT: @C.val.loc15_43 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_COp.C.Main:Copy.Core"(ptr sret({}) %return, ptr %self) #0 !dbg !46 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %return, ptr align 1 @C.val.loc15_43, i64 0, i1 false), !dbg !51
// CHECK:STDOUT:   ret void, !dbg !51
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.cb8adced19392b99:core.Destroy.Core"(ptr %self) #0 !dbg !53 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !55
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core"(ptr %self) #0 !dbg !57 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !59
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.64a2114a2cb70470:core.Destroy.Core"(ptr %self) #0 !dbg !61 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !63
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.3a7a2e5beb0d0fa7:core.Destroy.Core"(ptr %self) #0 !dbg !65 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.1feff81df232a20a:core.Destroy.Core"(ptr %self) #0 !dbg !69 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !71
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.84c40a423a4f9e08:core.Destroy.Core"(ptr %self) #0 !dbg !73 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !75
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CM.Main() #0 !dbg !77 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %n.var = alloca i32, align 4, !dbg !78
// CHECK:STDOUT:   %p.var = alloca double, align 8, !dbg !80
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %n.var), !dbg !78
// CHECK:STDOUT:   store i32 0, ptr %n.var, align 4, !dbg !78
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %p.var), !dbg !80
// CHECK:STDOUT:   store double 1.000000e+00, ptr %p.var, align 8, !dbg !80
// CHECK:STDOUT:   %.loc52 = load i32, ptr %n.var, align 4, !dbg !82
// CHECK:STDOUT:   call void @_CF.Main.6d93ae89cb2e7b0f(i32 %.loc52), !dbg !83
// CHECK:STDOUT:   %.loc53_18 = load i32, ptr %n.var, align 4, !dbg !84
// CHECK:STDOUT:   %G.call.loc53 = call i32 @_CG.Main.64609c610134ef75(i32 %.loc53_18), !dbg !85
// CHECK:STDOUT:   %.loc54 = load double, ptr %p.var, align 8, !dbg !90
// CHECK:STDOUT:   call void @_CF.Main.3219c8c671376a1e(double %.loc54), !dbg !91
// CHECK:STDOUT:   %.loc55_18 = load double, ptr %p.var, align 8, !dbg !92
// CHECK:STDOUT:   %G.call.loc55 = call double @_CG.Main.862050cdacea1015(double %.loc55_18), !dbg !93
// CHECK:STDOUT:   call void @"_CSelfDestruct.84c40a423a4f9e08:core.Destroy.Core"(ptr %p.var), !dbg !80
// CHECK:STDOUT:   call void @"_CSelfDestruct.76db170ab64c05aa:core.Destroy.Core"(ptr %n.var), !dbg !78
// CHECK:STDOUT:   ret void, !dbg !96
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.76db170ab64c05aa:core.Destroy.Core"(ptr %self) #0 !dbg !97 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !99
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.6d93ae89cb2e7b0f(i32 %_) #0 !dbg !101 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !104
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CG.Main.64609c610134ef75(i32 %x) #0 !dbg !106 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc29_6.3.temp = alloca i32, align 4, !dbg !112
// CHECK:STDOUT:   %.loc31_8.3.temp = alloca i32, align 4, !dbg !115
// CHECK:STDOUT:   %.loc31_9.3.temp = alloca i32, align 4, !dbg !116
// CHECK:STDOUT:   %var_f64.var = alloca double, align 8, !dbg !117
// CHECK:STDOUT:   %ptr_i32.var = alloca ptr, align 8, !dbg !121
// CHECK:STDOUT:   %ptr_f64.var = alloca ptr, align 8, !dbg !125
// CHECK:STDOUT:   %ptr_i8.var = alloca ptr, align 8, !dbg !129
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !133
// CHECK:STDOUT:   %.loc43_6.3.temp = alloca {}, align 1, !dbg !136
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc29_6.3.temp), !dbg !112
// CHECK:STDOUT:   %H.call.loc29 = call i32 @_CH.Main.3094c9a3a4d84642(i32 %x), !dbg !112
// CHECK:STDOUT:   store i32 %H.call.loc29, ptr %.loc29_6.3.temp, align 4, !dbg !112
// CHECK:STDOUT:   call void @"_CSelfDestruct.76db170ab64c05aa:core.Destroy.Core"(ptr %.loc29_6.3.temp), !dbg !112
// CHECK:STDOUT:   %H.call.loc30 = call %type.28 @_CH.Main.c3359c2c217ad593(%type.28 zeroinitializer), !dbg !113
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc31_8.3.temp), !dbg !115
// CHECK:STDOUT:   %G.call = call i32 @_CG.Main.64609c610134ef75(i32 %x), !dbg !115
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc31_9.3.temp), !dbg !116
// CHECK:STDOUT:   store i32 %G.call, ptr %.loc31_8.3.temp, align 4, !dbg !115
// CHECK:STDOUT:   %.loc31_8.5 = load i32, ptr %.loc31_8.3.temp, align 4, !dbg !115
// CHECK:STDOUT:   %H.call.loc31 = call i32 @_CH.Main.3094c9a3a4d84642(i32 %.loc31_8.5), !dbg !116
// CHECK:STDOUT:   store i32 %H.call.loc31, ptr %.loc31_9.3.temp, align 4, !dbg !116
// CHECK:STDOUT:   call void @"_CSelfDestruct.76db170ab64c05aa:core.Destroy.Core"(ptr %.loc31_9.3.temp), !dbg !116
// CHECK:STDOUT:   call void @"_CSelfDestruct.76db170ab64c05aa:core.Destroy.Core"(ptr %.loc31_8.3.temp), !dbg !115
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %var_f64.var), !dbg !117
// CHECK:STDOUT:   store double poison, ptr %var_f64.var, align 8, !dbg !117
// CHECK:STDOUT:   %.loc35_5 = load double, ptr %var_f64.var, align 8, !dbg !119
// CHECK:STDOUT:   %H.call.loc35 = call double @_CH.Main.d30a9b8c414024b6(double %.loc35_5), !dbg !120
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_i32.var), !dbg !121
// CHECK:STDOUT:   store ptr poison, ptr %ptr_i32.var, align 8, !dbg !121
// CHECK:STDOUT:   %.loc37_5 = load ptr, ptr %ptr_i32.var, align 8, !dbg !123
// CHECK:STDOUT:   %H.call.loc37 = call ptr @_CH.Main.01c402613389278e(ptr %.loc37_5), !dbg !124
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_f64.var), !dbg !125
// CHECK:STDOUT:   store ptr poison, ptr %ptr_f64.var, align 8, !dbg !125
// CHECK:STDOUT:   %.loc39_5 = load ptr, ptr %ptr_f64.var, align 8, !dbg !127
// CHECK:STDOUT:   %H.call.loc39 = call ptr @_CH.Main.01c402613389278e(ptr %.loc39_5), !dbg !128
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_i8.var), !dbg !129
// CHECK:STDOUT:   store ptr poison, ptr %ptr_i8.var, align 8, !dbg !129
// CHECK:STDOUT:   %.loc41_5 = load ptr, ptr %ptr_i8.var, align 8, !dbg !131
// CHECK:STDOUT:   %H.call.loc41 = call ptr @_CH.Main.01c402613389278e(ptr %.loc41_5), !dbg !132
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !133
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc43_6.3.temp), !dbg !136
// CHECK:STDOUT:   call void @_CH.Main.42513763e01cba2c(ptr %.loc43_6.3.temp, ptr %c.var), !dbg !136
// CHECK:STDOUT:   call void @"_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core"(ptr %.loc43_6.3.temp), !dbg !136
// CHECK:STDOUT:   call void @"_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core"(ptr %c.var), !dbg !133
// CHECK:STDOUT:   call void @"_CSelfDestruct.64a2114a2cb70470:core.Destroy.Core"(ptr %ptr_i8.var), !dbg !129
// CHECK:STDOUT:   call void @"_CSelfDestruct.3a7a2e5beb0d0fa7:core.Destroy.Core"(ptr %ptr_f64.var), !dbg !125
// CHECK:STDOUT:   call void @"_CSelfDestruct.1feff81df232a20a:core.Destroy.Core"(ptr %ptr_i32.var), !dbg !121
// CHECK:STDOUT:   call void @"_CSelfDestruct.84c40a423a4f9e08:core.Destroy.Core"(ptr %var_f64.var), !dbg !117
// CHECK:STDOUT:   ret i32 %x, !dbg !138
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.3219c8c671376a1e(double %_) #0 !dbg !140 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !143
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CG.Main.862050cdacea1015(double %x) #0 !dbg !145 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc29_6.3.temp = alloca double, align 8, !dbg !151
// CHECK:STDOUT:   %.loc31_8.3.temp = alloca double, align 8, !dbg !154
// CHECK:STDOUT:   %.loc31_9.3.temp = alloca double, align 8, !dbg !155
// CHECK:STDOUT:   %var_f64.var = alloca double, align 8, !dbg !156
// CHECK:STDOUT:   %ptr_i32.var = alloca ptr, align 8, !dbg !160
// CHECK:STDOUT:   %ptr_f64.var = alloca ptr, align 8, !dbg !164
// CHECK:STDOUT:   %ptr_i8.var = alloca ptr, align 8, !dbg !168
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !172
// CHECK:STDOUT:   %.loc43_6.3.temp = alloca {}, align 1, !dbg !175
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc29_6.3.temp), !dbg !151
// CHECK:STDOUT:   %H.call.loc29 = call double @_CH.Main.d30a9b8c414024b6(double %x), !dbg !151
// CHECK:STDOUT:   store double %H.call.loc29, ptr %.loc29_6.3.temp, align 8, !dbg !151
// CHECK:STDOUT:   call void @"_CSelfDestruct.84c40a423a4f9e08:core.Destroy.Core"(ptr %.loc29_6.3.temp), !dbg !151
// CHECK:STDOUT:   %H.call.loc30 = call %type.28 @_CH.Main.c3359c2c217ad593(%type.28 zeroinitializer), !dbg !152
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc31_8.3.temp), !dbg !154
// CHECK:STDOUT:   %G.call = call double @_CG.Main.862050cdacea1015(double %x), !dbg !154
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc31_9.3.temp), !dbg !155
// CHECK:STDOUT:   store double %G.call, ptr %.loc31_8.3.temp, align 8, !dbg !154
// CHECK:STDOUT:   %.loc31_8.5 = load double, ptr %.loc31_8.3.temp, align 8, !dbg !154
// CHECK:STDOUT:   %H.call.loc31 = call double @_CH.Main.d30a9b8c414024b6(double %.loc31_8.5), !dbg !155
// CHECK:STDOUT:   store double %H.call.loc31, ptr %.loc31_9.3.temp, align 8, !dbg !155
// CHECK:STDOUT:   call void @"_CSelfDestruct.84c40a423a4f9e08:core.Destroy.Core"(ptr %.loc31_9.3.temp), !dbg !155
// CHECK:STDOUT:   call void @"_CSelfDestruct.84c40a423a4f9e08:core.Destroy.Core"(ptr %.loc31_8.3.temp), !dbg !154
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %var_f64.var), !dbg !156
// CHECK:STDOUT:   store double poison, ptr %var_f64.var, align 8, !dbg !156
// CHECK:STDOUT:   %.loc35_5 = load double, ptr %var_f64.var, align 8, !dbg !158
// CHECK:STDOUT:   %H.call.loc35 = call double @_CH.Main.d30a9b8c414024b6(double %.loc35_5), !dbg !159
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_i32.var), !dbg !160
// CHECK:STDOUT:   store ptr poison, ptr %ptr_i32.var, align 8, !dbg !160
// CHECK:STDOUT:   %.loc37_5 = load ptr, ptr %ptr_i32.var, align 8, !dbg !162
// CHECK:STDOUT:   %H.call.loc37 = call ptr @_CH.Main.01c402613389278e(ptr %.loc37_5), !dbg !163
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_f64.var), !dbg !164
// CHECK:STDOUT:   store ptr poison, ptr %ptr_f64.var, align 8, !dbg !164
// CHECK:STDOUT:   %.loc39_5 = load ptr, ptr %ptr_f64.var, align 8, !dbg !166
// CHECK:STDOUT:   %H.call.loc39 = call ptr @_CH.Main.01c402613389278e(ptr %.loc39_5), !dbg !167
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_i8.var), !dbg !168
// CHECK:STDOUT:   store ptr poison, ptr %ptr_i8.var, align 8, !dbg !168
// CHECK:STDOUT:   %.loc41_5 = load ptr, ptr %ptr_i8.var, align 8, !dbg !170
// CHECK:STDOUT:   %H.call.loc41 = call ptr @_CH.Main.01c402613389278e(ptr %.loc41_5), !dbg !171
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !172
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc43_6.3.temp), !dbg !175
// CHECK:STDOUT:   call void @_CH.Main.42513763e01cba2c(ptr %.loc43_6.3.temp, ptr %c.var), !dbg !175
// CHECK:STDOUT:   call void @"_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core"(ptr %.loc43_6.3.temp), !dbg !175
// CHECK:STDOUT:   call void @"_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core"(ptr %c.var), !dbg !172
// CHECK:STDOUT:   call void @"_CSelfDestruct.64a2114a2cb70470:core.Destroy.Core"(ptr %ptr_i8.var), !dbg !168
// CHECK:STDOUT:   call void @"_CSelfDestruct.3a7a2e5beb0d0fa7:core.Destroy.Core"(ptr %ptr_f64.var), !dbg !164
// CHECK:STDOUT:   call void @"_CSelfDestruct.1feff81df232a20a:core.Destroy.Core"(ptr %ptr_i32.var), !dbg !160
// CHECK:STDOUT:   call void @"_CSelfDestruct.84c40a423a4f9e08:core.Destroy.Core"(ptr %var_f64.var), !dbg !156
// CHECK:STDOUT:   ret double %x, !dbg !177
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CH.Main.3094c9a3a4d84642(i32 %x) #0 !dbg !179 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %x, !dbg !185
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr %type.28 @_CH.Main.c3359c2c217ad593(%type.28 %x) #0 !dbg !187 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret %type.28 %x, !dbg !193
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CH.Main.d30a9b8c414024b6(double %x) #0 !dbg !195 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret double %x, !dbg !201
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CH.Main.01c402613389278e(ptr %x) #0 !dbg !203 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret ptr %x, !dbg !209
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CH.Main.42513763e01cba2c(ptr sret({}) %return, ptr %x) #0 !dbg !227 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_COp.C.Main:Copy.Core"(ptr %return, ptr %x), !dbg !232
// CHECK:STDOUT:   ret void, !dbg !233
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 18 }
// CHECK:STDOUT: uselistorder ptr @_CH.Main.3094c9a3a4d84642, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CH.Main.c3359c2c217ad593, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CH.Main.d30a9b8c414024b6, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CH.Main.01c402613389278e, { 5, 2, 0, 4, 3, 1 }
// CHECK:STDOUT: uselistorder ptr @_CH.Main.42513763e01cba2c, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!35}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !34 = !DIFile(filename: "call_basic.carbon", directory: "")
// CHECK:STDOUT: !35 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !34, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !36 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !37 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !38 = !{!37, !37}
// CHECK:STDOUT: !39 = !DISubroutineType(types: !38)
// CHECK:STDOUT: !40 = !{null, !37}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !42 = !{null}
// CHECK:STDOUT: !43 = !DISubroutineType(types: !42)
// CHECK:STDOUT: !44 = !{null, !36}
// CHECK:STDOUT: !45 = !DISubroutineType(types: !44)
// CHECK:STDOUT: !46 = distinct !DISubprogram(name: "Op", linkageName: "_COp.C.Main:Copy.Core", scope: null, file: !34, line: 15, type: !39, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !52)
// CHECK:STDOUT: !47 = !DILocalVariable(arg: 1, scope: !46, type: !37)
// CHECK:STDOUT: !51 = !DILocation(line: 15, column: 34, scope: !46)
// CHECK:STDOUT: !52 = !{!47}
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.cb8adced19392b99:core.Destroy.Core", scope: null, file: !34, line: 43, type: !41, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !56)
// CHECK:STDOUT: !54 = !DILocalVariable(arg: 1, scope: !53, type: !37)
// CHECK:STDOUT: !55 = !DILocation(line: 43, column: 3, scope: !53)
// CHECK:STDOUT: !56 = !{!54}
// CHECK:STDOUT: !57 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core", scope: null, file: !34, line: 43, type: !41, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !60)
// CHECK:STDOUT: !58 = !DILocalVariable(arg: 1, scope: !57, type: !37)
// CHECK:STDOUT: !59 = !DILocation(line: 43, column: 3, scope: !57)
// CHECK:STDOUT: !60 = !{!58}
// CHECK:STDOUT: !61 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.64a2114a2cb70470:core.Destroy.Core", scope: null, file: !34, line: 40, type: !41, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !64)
// CHECK:STDOUT: !62 = !DILocalVariable(arg: 1, scope: !61, type: !37)
// CHECK:STDOUT: !63 = !DILocation(line: 40, column: 3, scope: !61)
// CHECK:STDOUT: !64 = !{!62}
// CHECK:STDOUT: !65 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.3a7a2e5beb0d0fa7:core.Destroy.Core", scope: null, file: !34, line: 38, type: !41, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !68)
// CHECK:STDOUT: !66 = !DILocalVariable(arg: 1, scope: !65, type: !37)
// CHECK:STDOUT: !67 = !DILocation(line: 38, column: 3, scope: !65)
// CHECK:STDOUT: !68 = !{!66}
// CHECK:STDOUT: !69 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.1feff81df232a20a:core.Destroy.Core", scope: null, file: !34, line: 36, type: !41, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !72)
// CHECK:STDOUT: !70 = !DILocalVariable(arg: 1, scope: !69, type: !37)
// CHECK:STDOUT: !71 = !DILocation(line: 36, column: 3, scope: !69)
// CHECK:STDOUT: !72 = !{!70}
// CHECK:STDOUT: !73 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.84c40a423a4f9e08:core.Destroy.Core", scope: null, file: !34, line: 34, type: !41, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !76)
// CHECK:STDOUT: !74 = !DILocalVariable(arg: 1, scope: !73, type: !37)
// CHECK:STDOUT: !75 = !DILocation(line: 34, column: 3, scope: !73)
// CHECK:STDOUT: !76 = !{!74}
// CHECK:STDOUT: !77 = distinct !DISubprogram(name: "M", linkageName: "_CM.Main", scope: null, file: !34, line: 48, type: !43, spFlags: DISPFlagDefinition, unit: !35)
// CHECK:STDOUT: !78 = !DILocation(line: 49, column: 3, scope: !77)
// CHECK:STDOUT: !80 = !DILocation(line: 50, column: 3, scope: !77)
// CHECK:STDOUT: !82 = !DILocation(line: 52, column: 5, scope: !77)
// CHECK:STDOUT: !83 = !DILocation(line: 52, column: 3, scope: !77)
// CHECK:STDOUT: !84 = !DILocation(line: 53, column: 18, scope: !77)
// CHECK:STDOUT: !85 = !DILocation(line: 53, column: 16, scope: !77)
// CHECK:STDOUT: !86 = !{!36, !36}
// CHECK:STDOUT: !87 = !DISubroutineType(types: !86)
// CHECK:STDOUT: !90 = !DILocation(line: 54, column: 5, scope: !77)
// CHECK:STDOUT: !91 = !DILocation(line: 54, column: 3, scope: !77)
// CHECK:STDOUT: !92 = !DILocation(line: 55, column: 18, scope: !77)
// CHECK:STDOUT: !93 = !DILocation(line: 55, column: 16, scope: !77)
// CHECK:STDOUT: !96 = !DILocation(line: 48, column: 1, scope: !77)
// CHECK:STDOUT: !97 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.76db170ab64c05aa:core.Destroy.Core", scope: null, file: !34, line: 53, type: !45, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !100)
// CHECK:STDOUT: !98 = !DILocalVariable(arg: 1, scope: !97, type: !36)
// CHECK:STDOUT: !99 = !DILocation(line: 53, column: 16, scope: !97)
// CHECK:STDOUT: !100 = !{!98}
// CHECK:STDOUT: !101 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.6d93ae89cb2e7b0f", scope: null, file: !34, line: 19, type: !45, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !105)
// CHECK:STDOUT: !102 = !DILocalVariable(arg: 1, scope: !101, type: !36)
// CHECK:STDOUT: !104 = !DILocation(line: 19, column: 1, scope: !101)
// CHECK:STDOUT: !105 = !{!102}
// CHECK:STDOUT: !106 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.64609c610134ef75", scope: null, file: !34, line: 28, type: !87, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !139)
// CHECK:STDOUT: !107 = !DILocalVariable(arg: 1, scope: !106, type: !36)
// CHECK:STDOUT: !112 = !DILocation(line: 29, column: 3, scope: !106)
// CHECK:STDOUT: !113 = !DILocation(line: 30, column: 3, scope: !106)
// CHECK:STDOUT: !115 = !DILocation(line: 31, column: 5, scope: !106)
// CHECK:STDOUT: !116 = !DILocation(line: 31, column: 3, scope: !106)
// CHECK:STDOUT: !117 = !DILocation(line: 34, column: 3, scope: !106)
// CHECK:STDOUT: !119 = !DILocation(line: 35, column: 5, scope: !106)
// CHECK:STDOUT: !120 = !DILocation(line: 35, column: 3, scope: !106)
// CHECK:STDOUT: !121 = !DILocation(line: 36, column: 3, scope: !106)
// CHECK:STDOUT: !123 = !DILocation(line: 37, column: 5, scope: !106)
// CHECK:STDOUT: !124 = !DILocation(line: 37, column: 3, scope: !106)
// CHECK:STDOUT: !125 = !DILocation(line: 38, column: 3, scope: !106)
// CHECK:STDOUT: !127 = !DILocation(line: 39, column: 5, scope: !106)
// CHECK:STDOUT: !128 = !DILocation(line: 39, column: 3, scope: !106)
// CHECK:STDOUT: !129 = !DILocation(line: 40, column: 3, scope: !106)
// CHECK:STDOUT: !131 = !DILocation(line: 41, column: 5, scope: !106)
// CHECK:STDOUT: !132 = !DILocation(line: 41, column: 3, scope: !106)
// CHECK:STDOUT: !133 = !DILocation(line: 42, column: 3, scope: !106)
// CHECK:STDOUT: !136 = !DILocation(line: 43, column: 3, scope: !106)
// CHECK:STDOUT: !138 = !DILocation(line: 45, column: 3, scope: !106)
// CHECK:STDOUT: !139 = !{!107}
// CHECK:STDOUT: !140 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.3219c8c671376a1e", scope: null, file: !34, line: 19, type: !41, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !144)
// CHECK:STDOUT: !141 = !DILocalVariable(arg: 1, scope: !140, type: !37)
// CHECK:STDOUT: !143 = !DILocation(line: 19, column: 1, scope: !140)
// CHECK:STDOUT: !144 = !{!141}
// CHECK:STDOUT: !145 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.862050cdacea1015", scope: null, file: !34, line: 28, type: !39, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !178)
// CHECK:STDOUT: !146 = !DILocalVariable(arg: 1, scope: !145, type: !37)
// CHECK:STDOUT: !151 = !DILocation(line: 29, column: 3, scope: !145)
// CHECK:STDOUT: !152 = !DILocation(line: 30, column: 3, scope: !145)
// CHECK:STDOUT: !154 = !DILocation(line: 31, column: 5, scope: !145)
// CHECK:STDOUT: !155 = !DILocation(line: 31, column: 3, scope: !145)
// CHECK:STDOUT: !156 = !DILocation(line: 34, column: 3, scope: !145)
// CHECK:STDOUT: !158 = !DILocation(line: 35, column: 5, scope: !145)
// CHECK:STDOUT: !159 = !DILocation(line: 35, column: 3, scope: !145)
// CHECK:STDOUT: !160 = !DILocation(line: 36, column: 3, scope: !145)
// CHECK:STDOUT: !162 = !DILocation(line: 37, column: 5, scope: !145)
// CHECK:STDOUT: !163 = !DILocation(line: 37, column: 3, scope: !145)
// CHECK:STDOUT: !164 = !DILocation(line: 38, column: 3, scope: !145)
// CHECK:STDOUT: !166 = !DILocation(line: 39, column: 5, scope: !145)
// CHECK:STDOUT: !167 = !DILocation(line: 39, column: 3, scope: !145)
// CHECK:STDOUT: !168 = !DILocation(line: 40, column: 3, scope: !145)
// CHECK:STDOUT: !170 = !DILocation(line: 41, column: 5, scope: !145)
// CHECK:STDOUT: !171 = !DILocation(line: 41, column: 3, scope: !145)
// CHECK:STDOUT: !172 = !DILocation(line: 42, column: 3, scope: !145)
// CHECK:STDOUT: !175 = !DILocation(line: 43, column: 3, scope: !145)
// CHECK:STDOUT: !177 = !DILocation(line: 45, column: 3, scope: !145)
// CHECK:STDOUT: !178 = !{!146}
// CHECK:STDOUT: !179 = distinct !DISubprogram(name: "H", linkageName: "_CH.Main.3094c9a3a4d84642", scope: null, file: !34, line: 22, type: !87, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !186)
// CHECK:STDOUT: !180 = !DILocalVariable(arg: 1, scope: !179, type: !36)
// CHECK:STDOUT: !185 = !DILocation(line: 23, column: 3, scope: !179)
// CHECK:STDOUT: !186 = !{!180}
// CHECK:STDOUT: !187 = distinct !DISubprogram(name: "H", linkageName: "_CH.Main.c3359c2c217ad593", scope: null, file: !34, line: 22, type: !39, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !194)
// CHECK:STDOUT: !188 = !DILocalVariable(arg: 1, scope: !187, type: !37)
// CHECK:STDOUT: !193 = !DILocation(line: 23, column: 3, scope: !187)
// CHECK:STDOUT: !194 = !{!188}
// CHECK:STDOUT: !195 = distinct !DISubprogram(name: "H", linkageName: "_CH.Main.d30a9b8c414024b6", scope: null, file: !34, line: 22, type: !39, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !202)
// CHECK:STDOUT: !196 = !DILocalVariable(arg: 1, scope: !195, type: !37)
// CHECK:STDOUT: !201 = !DILocation(line: 23, column: 3, scope: !195)
// CHECK:STDOUT: !202 = !{!196}
// CHECK:STDOUT: !203 = distinct !DISubprogram(name: "H", linkageName: "_CH.Main.01c402613389278e", scope: null, file: !34, line: 22, type: !39, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !210)
// CHECK:STDOUT: !204 = !DILocalVariable(arg: 1, scope: !203, type: !37)
// CHECK:STDOUT: !209 = !DILocation(line: 23, column: 3, scope: !203)
// CHECK:STDOUT: !210 = !{!204}
// CHECK:STDOUT: !227 = distinct !DISubprogram(name: "H", linkageName: "_CH.Main.42513763e01cba2c", scope: null, file: !34, line: 22, type: !39, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !234)
// CHECK:STDOUT: !228 = !DILocalVariable(arg: 1, scope: !227, type: !37)
// CHECK:STDOUT: !232 = !DILocation(line: 23, column: 10, scope: !227)
// CHECK:STDOUT: !233 = !DILocation(line: 23, column: 3, scope: !227)
// CHECK:STDOUT: !234 = !{!228}
// CHECK:STDOUT:
