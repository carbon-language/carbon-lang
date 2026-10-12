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
// CHECK:STDOUT: define void @"_COp.C.Main:Copy.Core"(ptr sret({}) %return, ptr %self) #0 !dbg !42 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %return, ptr align 1 @C.val.loc15_43, i64 0, i1 false), !dbg !47
// CHECK:STDOUT:   ret void, !dbg !47
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CM.Main() #0 !dbg !49 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %n.var = alloca i32, align 4, !dbg !50
// CHECK:STDOUT:   %p.var = alloca double, align 8, !dbg !52
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %n.var), !dbg !50
// CHECK:STDOUT:   store i32 0, ptr %n.var, align 4, !dbg !50
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %p.var), !dbg !52
// CHECK:STDOUT:   store double 1.000000e+00, ptr %p.var, align 8, !dbg !52
// CHECK:STDOUT:   %.loc52 = load i32, ptr %n.var, align 4, !dbg !54
// CHECK:STDOUT:   call void @_CF.Main.d727653b20be66f9(i32 %.loc52), !dbg !55
// CHECK:STDOUT:   %.loc53_18 = load i32, ptr %n.var, align 4, !dbg !58
// CHECK:STDOUT:   %G.call.loc53 = call i32 @_CG.Main.13bcdd223d74356e(i32 %.loc53_18), !dbg !59
// CHECK:STDOUT:   %.loc54 = load double, ptr %p.var, align 8, !dbg !64
// CHECK:STDOUT:   call void @_CF.Main.e0b820d0bf6546cf(double %.loc54), !dbg !65
// CHECK:STDOUT:   %.loc55_18 = load double, ptr %p.var, align 8, !dbg !68
// CHECK:STDOUT:   %G.call.loc55 = call double @_CG.Main.eebe1ab4c839fb72(double %.loc55_18), !dbg !69
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.25cb13bcac5af6bc(ptr %p.var), !dbg !52
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.0aefc5da911f5051(ptr %n.var), !dbg !50
// CHECK:STDOUT:   ret void, !dbg !72
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.d727653b20be66f9(i32 %_) #0 !dbg !73 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !76
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CG.Main.13bcdd223d74356e(i32 %x) #0 !dbg !78 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc29_6.3.temp = alloca i32, align 4, !dbg !84
// CHECK:STDOUT:   %.loc31_8.3.temp = alloca i32, align 4, !dbg !87
// CHECK:STDOUT:   %.loc31_9.3.temp = alloca i32, align 4, !dbg !88
// CHECK:STDOUT:   %var_f64.var = alloca double, align 8, !dbg !89
// CHECK:STDOUT:   %ptr_i32.var = alloca ptr, align 8, !dbg !93
// CHECK:STDOUT:   %ptr_f64.var = alloca ptr, align 8, !dbg !97
// CHECK:STDOUT:   %ptr_i8.var = alloca ptr, align 8, !dbg !101
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !105
// CHECK:STDOUT:   %.loc43_6.3.temp = alloca {}, align 1, !dbg !108
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc29_6.3.temp), !dbg !84
// CHECK:STDOUT:   %H.call.loc29 = call i32 @_CH.Main.2aef1d52a6e1f957(i32 %x), !dbg !84
// CHECK:STDOUT:   store i32 %H.call.loc29, ptr %.loc29_6.3.temp, align 4, !dbg !84
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.0aefc5da911f5051(ptr %.loc29_6.3.temp), !dbg !84
// CHECK:STDOUT:   %H.call.loc30 = call %type.28 @_CH.Main.c3359c2c217ad593(%type.28 zeroinitializer), !dbg !85
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc31_8.3.temp), !dbg !87
// CHECK:STDOUT:   %G.call = call i32 @_CG.Main.13bcdd223d74356e(i32 %x), !dbg !87
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc31_9.3.temp), !dbg !88
// CHECK:STDOUT:   store i32 %G.call, ptr %.loc31_8.3.temp, align 4, !dbg !87
// CHECK:STDOUT:   %.loc31_8.5 = load i32, ptr %.loc31_8.3.temp, align 4, !dbg !87
// CHECK:STDOUT:   %H.call.loc31 = call i32 @_CH.Main.2aef1d52a6e1f957(i32 %.loc31_8.5), !dbg !88
// CHECK:STDOUT:   store i32 %H.call.loc31, ptr %.loc31_9.3.temp, align 4, !dbg !88
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.0aefc5da911f5051(ptr %.loc31_9.3.temp), !dbg !88
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.0aefc5da911f5051(ptr %.loc31_8.3.temp), !dbg !87
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %var_f64.var), !dbg !89
// CHECK:STDOUT:   store double poison, ptr %var_f64.var, align 8, !dbg !89
// CHECK:STDOUT:   %.loc35_5 = load double, ptr %var_f64.var, align 8, !dbg !91
// CHECK:STDOUT:   %H.call.loc35 = call double @_CH.Main.cff5c31fb9f8fd40(double %.loc35_5), !dbg !92
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_i32.var), !dbg !93
// CHECK:STDOUT:   store ptr poison, ptr %ptr_i32.var, align 8, !dbg !93
// CHECK:STDOUT:   %.loc37_5 = load ptr, ptr %ptr_i32.var, align 8, !dbg !95
// CHECK:STDOUT:   %H.call.loc37 = call ptr @_CH.Main.c0480b580588c73f(ptr %.loc37_5), !dbg !96
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_f64.var), !dbg !97
// CHECK:STDOUT:   store ptr poison, ptr %ptr_f64.var, align 8, !dbg !97
// CHECK:STDOUT:   %.loc39_5 = load ptr, ptr %ptr_f64.var, align 8, !dbg !99
// CHECK:STDOUT:   %H.call.loc39 = call ptr @_CH.Main.c0480b580588c73f(ptr %.loc39_5), !dbg !100
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_i8.var), !dbg !101
// CHECK:STDOUT:   store ptr poison, ptr %ptr_i8.var, align 8, !dbg !101
// CHECK:STDOUT:   %.loc41_5 = load ptr, ptr %ptr_i8.var, align 8, !dbg !103
// CHECK:STDOUT:   %H.call.loc41 = call ptr @_CH.Main.c0480b580588c73f(ptr %.loc41_5), !dbg !104
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !105
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc43_6.3.temp), !dbg !108
// CHECK:STDOUT:   call void @_CH.Main.42513763e01cba2c(ptr %.loc43_6.3.temp, ptr %c.var), !dbg !108
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.166956a092acbab5(ptr %.loc43_6.3.temp), !dbg !108
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.166956a092acbab5(ptr %c.var), !dbg !105
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.48914ea11ffa7b46(ptr %ptr_i8.var), !dbg !101
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.373ee65cc5e0fcdf(ptr %ptr_f64.var), !dbg !97
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.008e0b30cf994e40(ptr %ptr_i32.var), !dbg !93
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.25cb13bcac5af6bc(ptr %var_f64.var), !dbg !89
// CHECK:STDOUT:   ret i32 %x, !dbg !110
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.e0b820d0bf6546cf(double %_) #0 !dbg !112 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !115
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CG.Main.eebe1ab4c839fb72(double %x) #0 !dbg !117 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc29_6.3.temp = alloca double, align 8, !dbg !123
// CHECK:STDOUT:   %.loc31_8.3.temp = alloca double, align 8, !dbg !126
// CHECK:STDOUT:   %.loc31_9.3.temp = alloca double, align 8, !dbg !127
// CHECK:STDOUT:   %var_f64.var = alloca double, align 8, !dbg !128
// CHECK:STDOUT:   %ptr_i32.var = alloca ptr, align 8, !dbg !132
// CHECK:STDOUT:   %ptr_f64.var = alloca ptr, align 8, !dbg !136
// CHECK:STDOUT:   %ptr_i8.var = alloca ptr, align 8, !dbg !140
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !144
// CHECK:STDOUT:   %.loc43_6.3.temp = alloca {}, align 1, !dbg !147
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc29_6.3.temp), !dbg !123
// CHECK:STDOUT:   %H.call.loc29 = call double @_CH.Main.cff5c31fb9f8fd40(double %x), !dbg !123
// CHECK:STDOUT:   store double %H.call.loc29, ptr %.loc29_6.3.temp, align 8, !dbg !123
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.25cb13bcac5af6bc(ptr %.loc29_6.3.temp), !dbg !123
// CHECK:STDOUT:   %H.call.loc30 = call %type.28 @_CH.Main.c3359c2c217ad593(%type.28 zeroinitializer), !dbg !124
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc31_8.3.temp), !dbg !126
// CHECK:STDOUT:   %G.call = call double @_CG.Main.eebe1ab4c839fb72(double %x), !dbg !126
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc31_9.3.temp), !dbg !127
// CHECK:STDOUT:   store double %G.call, ptr %.loc31_8.3.temp, align 8, !dbg !126
// CHECK:STDOUT:   %.loc31_8.5 = load double, ptr %.loc31_8.3.temp, align 8, !dbg !126
// CHECK:STDOUT:   %H.call.loc31 = call double @_CH.Main.cff5c31fb9f8fd40(double %.loc31_8.5), !dbg !127
// CHECK:STDOUT:   store double %H.call.loc31, ptr %.loc31_9.3.temp, align 8, !dbg !127
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.25cb13bcac5af6bc(ptr %.loc31_9.3.temp), !dbg !127
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.25cb13bcac5af6bc(ptr %.loc31_8.3.temp), !dbg !126
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %var_f64.var), !dbg !128
// CHECK:STDOUT:   store double poison, ptr %var_f64.var, align 8, !dbg !128
// CHECK:STDOUT:   %.loc35_5 = load double, ptr %var_f64.var, align 8, !dbg !130
// CHECK:STDOUT:   %H.call.loc35 = call double @_CH.Main.cff5c31fb9f8fd40(double %.loc35_5), !dbg !131
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_i32.var), !dbg !132
// CHECK:STDOUT:   store ptr poison, ptr %ptr_i32.var, align 8, !dbg !132
// CHECK:STDOUT:   %.loc37_5 = load ptr, ptr %ptr_i32.var, align 8, !dbg !134
// CHECK:STDOUT:   %H.call.loc37 = call ptr @_CH.Main.c0480b580588c73f(ptr %.loc37_5), !dbg !135
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_f64.var), !dbg !136
// CHECK:STDOUT:   store ptr poison, ptr %ptr_f64.var, align 8, !dbg !136
// CHECK:STDOUT:   %.loc39_5 = load ptr, ptr %ptr_f64.var, align 8, !dbg !138
// CHECK:STDOUT:   %H.call.loc39 = call ptr @_CH.Main.c0480b580588c73f(ptr %.loc39_5), !dbg !139
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_i8.var), !dbg !140
// CHECK:STDOUT:   store ptr poison, ptr %ptr_i8.var, align 8, !dbg !140
// CHECK:STDOUT:   %.loc41_5 = load ptr, ptr %ptr_i8.var, align 8, !dbg !142
// CHECK:STDOUT:   %H.call.loc41 = call ptr @_CH.Main.c0480b580588c73f(ptr %.loc41_5), !dbg !143
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !144
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc43_6.3.temp), !dbg !147
// CHECK:STDOUT:   call void @_CH.Main.42513763e01cba2c(ptr %.loc43_6.3.temp, ptr %c.var), !dbg !147
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.166956a092acbab5(ptr %.loc43_6.3.temp), !dbg !147
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.166956a092acbab5(ptr %c.var), !dbg !144
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.48914ea11ffa7b46(ptr %ptr_i8.var), !dbg !140
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.373ee65cc5e0fcdf(ptr %ptr_f64.var), !dbg !136
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.008e0b30cf994e40(ptr %ptr_i32.var), !dbg !132
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.25cb13bcac5af6bc(ptr %var_f64.var), !dbg !128
// CHECK:STDOUT:   ret double %x, !dbg !149
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.25cb13bcac5af6bc(ptr %t) #0 !dbg !151 {
// CHECK:STDOUT:   ret void, !dbg !156
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.0aefc5da911f5051(ptr %t) #0 !dbg !158 {
// CHECK:STDOUT:   ret void, !dbg !163
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CH.Main.2aef1d52a6e1f957(i32 %x) #0 !dbg !165 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %x, !dbg !171
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr %type.28 @_CH.Main.c3359c2c217ad593(%type.28 %x) #0 !dbg !173 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret %type.28 %x, !dbg !179
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CH.Main.cff5c31fb9f8fd40(double %x) #0 !dbg !181 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret double %x, !dbg !187
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CH.Main.c0480b580588c73f(ptr %x) #0 !dbg !189 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret ptr %x, !dbg !195
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CH.Main.42513763e01cba2c(ptr sret({}) %return, ptr %x) #0 !dbg !213 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_COp.C.Main:Copy.Core"(ptr %return, ptr %x), !dbg !218
// CHECK:STDOUT:   ret void, !dbg !219
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.166956a092acbab5(ptr %t) #0 !dbg !221 {
// CHECK:STDOUT:   ret void, !dbg !226
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.48914ea11ffa7b46(ptr %t) #0 !dbg !228 {
// CHECK:STDOUT:   ret void, !dbg !233
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.373ee65cc5e0fcdf(ptr %t) #0 !dbg !235 {
// CHECK:STDOUT:   ret void, !dbg !240
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.008e0b30cf994e40(ptr %t) #0 !dbg !242 {
// CHECK:STDOUT:   ret void, !dbg !247
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 18 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.25cb13bcac5af6bc, { 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.0aefc5da911f5051, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CH.Main.2aef1d52a6e1f957, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CH.Main.c3359c2c217ad593, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CH.Main.cff5c31fb9f8fd40, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CH.Main.c0480b580588c73f, { 5, 2, 0, 4, 3, 1 }
// CHECK:STDOUT: uselistorder ptr @_CH.Main.42513763e01cba2c, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.166956a092acbab5, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.48914ea11ffa7b46, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.373ee65cc5e0fcdf, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.008e0b30cf994e40, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!35}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !28 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !34 = !DIFile(filename: "call_basic.carbon", directory: "")
// CHECK:STDOUT: !35 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !34, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !36 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !37 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !38 = !{!37, !37}
// CHECK:STDOUT: !39 = !DISubroutineType(types: !38)
// CHECK:STDOUT: !40 = !{null}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "Op", linkageName: "_COp.C.Main:Copy.Core", scope: null, file: !34, line: 15, type: !39, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !48)
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 1, scope: !42, type: !37)
// CHECK:STDOUT: !47 = !DILocation(line: 15, column: 34, scope: !42)
// CHECK:STDOUT: !48 = !{!43}
// CHECK:STDOUT: !49 = distinct !DISubprogram(name: "M", linkageName: "_CM.Main", scope: null, file: !34, line: 48, type: !41, spFlags: DISPFlagDefinition, unit: !35)
// CHECK:STDOUT: !50 = !DILocation(line: 49, column: 3, scope: !49)
// CHECK:STDOUT: !52 = !DILocation(line: 50, column: 3, scope: !49)
// CHECK:STDOUT: !54 = !DILocation(line: 52, column: 5, scope: !49)
// CHECK:STDOUT: !55 = !DILocation(line: 52, column: 3, scope: !49)
// CHECK:STDOUT: !56 = !{null, !36}
// CHECK:STDOUT: !57 = !DISubroutineType(types: !56)
// CHECK:STDOUT: !58 = !DILocation(line: 53, column: 18, scope: !49)
// CHECK:STDOUT: !59 = !DILocation(line: 53, column: 16, scope: !49)
// CHECK:STDOUT: !60 = !{!36, !36}
// CHECK:STDOUT: !61 = !DISubroutineType(types: !60)
// CHECK:STDOUT: !64 = !DILocation(line: 54, column: 5, scope: !49)
// CHECK:STDOUT: !65 = !DILocation(line: 54, column: 3, scope: !49)
// CHECK:STDOUT: !66 = !{null, !37}
// CHECK:STDOUT: !67 = !DISubroutineType(types: !66)
// CHECK:STDOUT: !68 = !DILocation(line: 55, column: 18, scope: !49)
// CHECK:STDOUT: !69 = !DILocation(line: 55, column: 16, scope: !49)
// CHECK:STDOUT: !72 = !DILocation(line: 48, column: 1, scope: !49)
// CHECK:STDOUT: !73 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.d727653b20be66f9", scope: null, file: !34, line: 19, type: !57, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !77)
// CHECK:STDOUT: !74 = !DILocalVariable(arg: 1, scope: !73, type: !36)
// CHECK:STDOUT: !76 = !DILocation(line: 19, column: 1, scope: !73)
// CHECK:STDOUT: !77 = !{!74}
// CHECK:STDOUT: !78 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.13bcdd223d74356e", scope: null, file: !34, line: 28, type: !61, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !111)
// CHECK:STDOUT: !79 = !DILocalVariable(arg: 1, scope: !78, type: !36)
// CHECK:STDOUT: !84 = !DILocation(line: 29, column: 3, scope: !78)
// CHECK:STDOUT: !85 = !DILocation(line: 30, column: 3, scope: !78)
// CHECK:STDOUT: !87 = !DILocation(line: 31, column: 5, scope: !78)
// CHECK:STDOUT: !88 = !DILocation(line: 31, column: 3, scope: !78)
// CHECK:STDOUT: !89 = !DILocation(line: 34, column: 3, scope: !78)
// CHECK:STDOUT: !91 = !DILocation(line: 35, column: 5, scope: !78)
// CHECK:STDOUT: !92 = !DILocation(line: 35, column: 3, scope: !78)
// CHECK:STDOUT: !93 = !DILocation(line: 36, column: 3, scope: !78)
// CHECK:STDOUT: !95 = !DILocation(line: 37, column: 5, scope: !78)
// CHECK:STDOUT: !96 = !DILocation(line: 37, column: 3, scope: !78)
// CHECK:STDOUT: !97 = !DILocation(line: 38, column: 3, scope: !78)
// CHECK:STDOUT: !99 = !DILocation(line: 39, column: 5, scope: !78)
// CHECK:STDOUT: !100 = !DILocation(line: 39, column: 3, scope: !78)
// CHECK:STDOUT: !101 = !DILocation(line: 40, column: 3, scope: !78)
// CHECK:STDOUT: !103 = !DILocation(line: 41, column: 5, scope: !78)
// CHECK:STDOUT: !104 = !DILocation(line: 41, column: 3, scope: !78)
// CHECK:STDOUT: !105 = !DILocation(line: 42, column: 3, scope: !78)
// CHECK:STDOUT: !108 = !DILocation(line: 43, column: 3, scope: !78)
// CHECK:STDOUT: !110 = !DILocation(line: 45, column: 3, scope: !78)
// CHECK:STDOUT: !111 = !{!79}
// CHECK:STDOUT: !112 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.e0b820d0bf6546cf", scope: null, file: !34, line: 19, type: !67, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !116)
// CHECK:STDOUT: !113 = !DILocalVariable(arg: 1, scope: !112, type: !37)
// CHECK:STDOUT: !115 = !DILocation(line: 19, column: 1, scope: !112)
// CHECK:STDOUT: !116 = !{!113}
// CHECK:STDOUT: !117 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.eebe1ab4c839fb72", scope: null, file: !34, line: 28, type: !39, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !150)
// CHECK:STDOUT: !118 = !DILocalVariable(arg: 1, scope: !117, type: !37)
// CHECK:STDOUT: !123 = !DILocation(line: 29, column: 3, scope: !117)
// CHECK:STDOUT: !124 = !DILocation(line: 30, column: 3, scope: !117)
// CHECK:STDOUT: !126 = !DILocation(line: 31, column: 5, scope: !117)
// CHECK:STDOUT: !127 = !DILocation(line: 31, column: 3, scope: !117)
// CHECK:STDOUT: !128 = !DILocation(line: 34, column: 3, scope: !117)
// CHECK:STDOUT: !130 = !DILocation(line: 35, column: 5, scope: !117)
// CHECK:STDOUT: !131 = !DILocation(line: 35, column: 3, scope: !117)
// CHECK:STDOUT: !132 = !DILocation(line: 36, column: 3, scope: !117)
// CHECK:STDOUT: !134 = !DILocation(line: 37, column: 5, scope: !117)
// CHECK:STDOUT: !135 = !DILocation(line: 37, column: 3, scope: !117)
// CHECK:STDOUT: !136 = !DILocation(line: 38, column: 3, scope: !117)
// CHECK:STDOUT: !138 = !DILocation(line: 39, column: 5, scope: !117)
// CHECK:STDOUT: !139 = !DILocation(line: 39, column: 3, scope: !117)
// CHECK:STDOUT: !140 = !DILocation(line: 40, column: 3, scope: !117)
// CHECK:STDOUT: !142 = !DILocation(line: 41, column: 5, scope: !117)
// CHECK:STDOUT: !143 = !DILocation(line: 41, column: 3, scope: !117)
// CHECK:STDOUT: !144 = !DILocation(line: 42, column: 3, scope: !117)
// CHECK:STDOUT: !147 = !DILocation(line: 43, column: 3, scope: !117)
// CHECK:STDOUT: !149 = !DILocation(line: 45, column: 3, scope: !117)
// CHECK:STDOUT: !150 = !{!118}
// CHECK:STDOUT: !151 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.25cb13bcac5af6bc", scope: null, file: !28, line: 26, type: !67, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !157)
// CHECK:STDOUT: !152 = !DILocalVariable(arg: 1, scope: !151, type: !37)
// CHECK:STDOUT: !156 = !DILocation(line: 26, column: 1, scope: !151)
// CHECK:STDOUT: !157 = !{!152}
// CHECK:STDOUT: !158 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.0aefc5da911f5051", scope: null, file: !28, line: 26, type: !57, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !164)
// CHECK:STDOUT: !159 = !DILocalVariable(arg: 1, scope: !158, type: !36)
// CHECK:STDOUT: !163 = !DILocation(line: 26, column: 1, scope: !158)
// CHECK:STDOUT: !164 = !{!159}
// CHECK:STDOUT: !165 = distinct !DISubprogram(name: "H", linkageName: "_CH.Main.2aef1d52a6e1f957", scope: null, file: !34, line: 22, type: !61, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !172)
// CHECK:STDOUT: !166 = !DILocalVariable(arg: 1, scope: !165, type: !36)
// CHECK:STDOUT: !171 = !DILocation(line: 23, column: 3, scope: !165)
// CHECK:STDOUT: !172 = !{!166}
// CHECK:STDOUT: !173 = distinct !DISubprogram(name: "H", linkageName: "_CH.Main.c3359c2c217ad593", scope: null, file: !34, line: 22, type: !39, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !180)
// CHECK:STDOUT: !174 = !DILocalVariable(arg: 1, scope: !173, type: !37)
// CHECK:STDOUT: !179 = !DILocation(line: 23, column: 3, scope: !173)
// CHECK:STDOUT: !180 = !{!174}
// CHECK:STDOUT: !181 = distinct !DISubprogram(name: "H", linkageName: "_CH.Main.cff5c31fb9f8fd40", scope: null, file: !34, line: 22, type: !39, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !188)
// CHECK:STDOUT: !182 = !DILocalVariable(arg: 1, scope: !181, type: !37)
// CHECK:STDOUT: !187 = !DILocation(line: 23, column: 3, scope: !181)
// CHECK:STDOUT: !188 = !{!182}
// CHECK:STDOUT: !189 = distinct !DISubprogram(name: "H", linkageName: "_CH.Main.c0480b580588c73f", scope: null, file: !34, line: 22, type: !39, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !196)
// CHECK:STDOUT: !190 = !DILocalVariable(arg: 1, scope: !189, type: !37)
// CHECK:STDOUT: !195 = !DILocation(line: 23, column: 3, scope: !189)
// CHECK:STDOUT: !196 = !{!190}
// CHECK:STDOUT: !213 = distinct !DISubprogram(name: "H", linkageName: "_CH.Main.42513763e01cba2c", scope: null, file: !34, line: 22, type: !39, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !220)
// CHECK:STDOUT: !214 = !DILocalVariable(arg: 1, scope: !213, type: !37)
// CHECK:STDOUT: !218 = !DILocation(line: 23, column: 10, scope: !213)
// CHECK:STDOUT: !219 = !DILocation(line: 23, column: 3, scope: !213)
// CHECK:STDOUT: !220 = !{!214}
// CHECK:STDOUT: !221 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.166956a092acbab5", scope: null, file: !28, line: 26, type: !67, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !227)
// CHECK:STDOUT: !222 = !DILocalVariable(arg: 1, scope: !221, type: !37)
// CHECK:STDOUT: !226 = !DILocation(line: 26, column: 1, scope: !221)
// CHECK:STDOUT: !227 = !{!222}
// CHECK:STDOUT: !228 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.48914ea11ffa7b46", scope: null, file: !28, line: 26, type: !67, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !234)
// CHECK:STDOUT: !229 = !DILocalVariable(arg: 1, scope: !228, type: !37)
// CHECK:STDOUT: !233 = !DILocation(line: 26, column: 1, scope: !228)
// CHECK:STDOUT: !234 = !{!229}
// CHECK:STDOUT: !235 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.373ee65cc5e0fcdf", scope: null, file: !28, line: 26, type: !67, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !241)
// CHECK:STDOUT: !236 = !DILocalVariable(arg: 1, scope: !235, type: !37)
// CHECK:STDOUT: !240 = !DILocation(line: 26, column: 1, scope: !235)
// CHECK:STDOUT: !241 = !{!236}
// CHECK:STDOUT: !242 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.008e0b30cf994e40", scope: null, file: !28, line: 26, type: !67, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !248)
// CHECK:STDOUT: !243 = !DILocalVariable(arg: 1, scope: !242, type: !37)
// CHECK:STDOUT: !247 = !DILocation(line: 26, column: 1, scope: !242)
// CHECK:STDOUT: !248 = !{!243}
// CHECK:STDOUT:
