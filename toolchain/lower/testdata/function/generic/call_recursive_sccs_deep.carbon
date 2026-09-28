// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --include-carbon-core
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/generic/call_recursive_sccs_deep.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/generic/call_recursive_sccs_deep.carbon

import Core library "io";
fn A[T: Core.Copy](x: T, count: i32) -> T;
fn B[T: Core.Copy](x: T, count: i32);
fn C[T: Core.Copy](x: T, count: i32);
fn D[T: Core.Copy](x: T, count: i32) -> T;
fn E[T: Core.Copy](x: T, count: i32) -> T;
fn F[T: Core.Copy](x: T, count: i32) -> T;

// Builds on `call_recursive_mutual.carbon` and `call_recursive_diamond.carbon`
// B-C form a mutually recursive SCC (strongly connected component), D-E-F-G
// form a diamond SCC.
// The two specifics for each of A, B, C, D, E, F, G with a pointer type could
// be deduplicated.
// E and F are also equivalent in the diamond, even though they are different
// generics. There is potential front-end ICF (identical code folding)
// optimization here, if the function fingerprint may infer deduplication
// when not including the two different generic_ids.

fn A[T: Core.Copy](x: T, count: i32) -> T {
  B(x, count);
  return D(x, count);
}

fn B[T: Core.Copy](x: T, count: i32) {
  C(x, count);
}

fn C[T: Core.Copy](x: T, count: i32) {
  if (count <= 2) {
    Core.Print(count);
    B(x, count + 1);
  }
}

fn D[T: Core.Copy](x: T, count: i32) -> T {
  if (count > 4) {
    return x;
  }
  if (count % 2 == 0) {
    return E(x, count);
  } else {
    return F(x, count);
  }
}

fn G[T: Core.Copy](x: T, count: i32) -> T;

fn E[T: Core.Copy](x: T, count: i32) -> T {
  return G(x, count);
}

fn F[T: Core.Copy](x: T, count: i32) -> T {
  return G(x, count);
}

fn G[T: Core.Copy](x: T, count: i32) -> T {
  return D(x, count + 1);
}


fn M() {
  var n: i32 = 0;
  var m: f64 = 1.0;
  var ptr_i32 : i32* = &n;
  var ptr_f64 : f64* = &m;

  A(n, 0);
  A(m, 0);
  A(ptr_i32, 0);
  A(ptr_f64, 0);
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'call_recursive_sccs_deep.carbon'
// CHECK:STDOUT: source_filename = "call_recursive_sccs_deep.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @printf.int.format = private unnamed_addr constant [4 x i8] c"%d\0A\00", align 1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CM.Main() #0 !dbg !14 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %n.var = alloca i32, align 4, !dbg !15
// CHECK:STDOUT:   %m.var = alloca double, align 8, !dbg !17
// CHECK:STDOUT:   %ptr_i32.var = alloca ptr, align 8, !dbg !19
// CHECK:STDOUT:   %ptr_f64.var = alloca ptr, align 8, !dbg !23
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %n.var), !dbg !15
// CHECK:STDOUT:   store i32 0, ptr %n.var, align 4, !dbg !15
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %m.var), !dbg !17
// CHECK:STDOUT:   store double 1.000000e+00, ptr %m.var, align 8, !dbg !17
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_i32.var), !dbg !19
// CHECK:STDOUT:   store ptr %n.var, ptr %ptr_i32.var, align 8, !dbg !19
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_f64.var), !dbg !23
// CHECK:STDOUT:   store ptr %m.var, ptr %ptr_f64.var, align 8, !dbg !23
// CHECK:STDOUT:   %.loc80_5 = load i32, ptr %n.var, align 4, !dbg !27
// CHECK:STDOUT:   %A.call.loc80 = call i32 @_CA.Main.4d6ceefeb7f772f4(i32 %.loc80_5, i32 0), !dbg !28
// CHECK:STDOUT:   %.loc81_5 = load double, ptr %m.var, align 8, !dbg !31
// CHECK:STDOUT:   %A.call.loc81 = call double @_CA.Main.c5e4ba542a56b71f(double %.loc81_5, i32 0), !dbg !32
// CHECK:STDOUT:   %.loc82_5 = load ptr, ptr %ptr_i32.var, align 8, !dbg !35
// CHECK:STDOUT:   %A.call.loc82 = call ptr @_CA.Main.a76b78461416b4a8(ptr %.loc82_5, i32 0), !dbg !36
// CHECK:STDOUT:   %.loc83_5 = load ptr, ptr %ptr_f64.var, align 8, !dbg !37
// CHECK:STDOUT:   %A.call.loc83 = call ptr @_CA.Main.a76b78461416b4a8(ptr %.loc83_5, i32 0), !dbg !38
// CHECK:STDOUT:   call void @"_CSelfDestruct.749b51755634ad17:core.Destroy.Core"(ptr %ptr_f64.var), !dbg !23
// CHECK:STDOUT:   call void @"_CSelfDestruct.d088c7eb746488be:core.Destroy.Core"(ptr %ptr_i32.var), !dbg !19
// CHECK:STDOUT:   call void @"_CSelfDestruct.dc14d4dd21801791:core.Destroy.Core"(ptr %m.var), !dbg !17
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %n.var), !dbg !15
// CHECK:STDOUT:   ret void, !dbg !39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.749b51755634ad17:core.Destroy.Core"(ptr %self) #0 !dbg !40 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !42
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.d088c7eb746488be:core.Destroy.Core"(ptr %self) #0 !dbg !44 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !46
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.7945ecba5c505fc0:core.Destroy.Core"(ptr %self) #0 !dbg !48 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !50
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.dc14d4dd21801791:core.Destroy.Core"(ptr %self) #0 !dbg !52 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !54
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.6dae29ece9a708c4:core.Destroy.Core"(ptr %self) #0 !dbg !56 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !58
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !60 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !62
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CA.Main.4d6ceefeb7f772f4(i32 %x, i32 %count) #0 !dbg !64 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CB.Main.4d6ceefeb7f772f4(i32 %x, i32 %count), !dbg !73
// CHECK:STDOUT:   %D.call = call i32 @_CD.Main.4d6ceefeb7f772f4(i32 %x, i32 %count), !dbg !78
// CHECK:STDOUT:   ret i32 %D.call, !dbg !79
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CA.Main.c5e4ba542a56b71f(double %x, i32 %count) #0 !dbg !81 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CB.Main.c5e4ba542a56b71f(double %x, i32 %count), !dbg !90
// CHECK:STDOUT:   %D.call = call double @_CD.Main.c5e4ba542a56b71f(double %x, i32 %count), !dbg !95
// CHECK:STDOUT:   ret double %D.call, !dbg !96
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CA.Main.a76b78461416b4a8(ptr %x, i32 %count) #0 !dbg !98 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CB.Main.a76b78461416b4a8(ptr %x, i32 %count), !dbg !107
// CHECK:STDOUT:   %D.call = call ptr @_CD.Main.a76b78461416b4a8(ptr %x, i32 %count), !dbg !110
// CHECK:STDOUT:   ret ptr %D.call, !dbg !111
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CB.Main.4d6ceefeb7f772f4(i32 %x, i32 %count) #0 !dbg !128 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CC.Main.4d6ceefeb7f772f4(i32 %x, i32 %count), !dbg !135
// CHECK:STDOUT:   ret void, !dbg !136
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CD.Main.4d6ceefeb7f772f4(i32 %x, i32 %count) #0 !dbg !138 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 4, !dbg !145
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then.loc49, label %if.else.loc49, !dbg !146
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc49:                                    ; preds = %entry
// CHECK:STDOUT:   ret i32 %x, !dbg !148
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc49:                                    ; preds = %entry
// CHECK:STDOUT:   %Int.as.ModWith.impl.Op.call = srem i32 %count, 2, !dbg !149
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.call = icmp eq i32 %Int.as.ModWith.impl.Op.call, 0, !dbg !149
// CHECK:STDOUT:   br i1 %Int.as.EqWith.impl.Equal.call, label %if.then.loc52, label %if.else.loc52, !dbg !150
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc52:                                    ; preds = %if.else.loc49
// CHECK:STDOUT:   %E.call = call i32 @_CE.Main.4d6ceefeb7f772f4(i32 %x, i32 %count), !dbg !153
// CHECK:STDOUT:   ret i32 %E.call, !dbg !154
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc52:                                    ; preds = %if.else.loc49
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main.4d6ceefeb7f772f4(i32 %x, i32 %count), !dbg !157
// CHECK:STDOUT:   ret i32 %F.call, !dbg !158
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CB.Main.c5e4ba542a56b71f(double %x, i32 %count) #0 !dbg !160 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CC.Main.c5e4ba542a56b71f(double %x, i32 %count), !dbg !167
// CHECK:STDOUT:   ret void, !dbg !168
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CD.Main.c5e4ba542a56b71f(double %x, i32 %count) #0 !dbg !170 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 4, !dbg !177
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then.loc49, label %if.else.loc49, !dbg !178
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc49:                                    ; preds = %entry
// CHECK:STDOUT:   ret double %x, !dbg !180
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc49:                                    ; preds = %entry
// CHECK:STDOUT:   %Int.as.ModWith.impl.Op.call = srem i32 %count, 2, !dbg !181
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.call = icmp eq i32 %Int.as.ModWith.impl.Op.call, 0, !dbg !181
// CHECK:STDOUT:   br i1 %Int.as.EqWith.impl.Equal.call, label %if.then.loc52, label %if.else.loc52, !dbg !182
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc52:                                    ; preds = %if.else.loc49
// CHECK:STDOUT:   %E.call = call double @_CE.Main.c5e4ba542a56b71f(double %x, i32 %count), !dbg !185
// CHECK:STDOUT:   ret double %E.call, !dbg !186
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc52:                                    ; preds = %if.else.loc49
// CHECK:STDOUT:   %F.call = call double @_CF.Main.c5e4ba542a56b71f(double %x, i32 %count), !dbg !189
// CHECK:STDOUT:   ret double %F.call, !dbg !190
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CB.Main.a76b78461416b4a8(ptr %x, i32 %count) #0 !dbg !192 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CC.Main.a76b78461416b4a8(ptr %x, i32 %count), !dbg !199
// CHECK:STDOUT:   ret void, !dbg !200
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CD.Main.a76b78461416b4a8(ptr %x, i32 %count) #0 !dbg !202 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 4, !dbg !209
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then.loc49, label %if.else.loc49, !dbg !210
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc49:                                    ; preds = %entry
// CHECK:STDOUT:   ret ptr %x, !dbg !212
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc49:                                    ; preds = %entry
// CHECK:STDOUT:   %Int.as.ModWith.impl.Op.call = srem i32 %count, 2, !dbg !213
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.call = icmp eq i32 %Int.as.ModWith.impl.Op.call, 0, !dbg !213
// CHECK:STDOUT:   br i1 %Int.as.EqWith.impl.Equal.call, label %if.then.loc52, label %if.else.loc52, !dbg !214
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc52:                                    ; preds = %if.else.loc49
// CHECK:STDOUT:   %E.call = call ptr @_CE.Main.a76b78461416b4a8(ptr %x, i32 %count), !dbg !217
// CHECK:STDOUT:   ret ptr %E.call, !dbg !218
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc52:                                    ; preds = %if.else.loc49
// CHECK:STDOUT:   %F.call = call ptr @_CF.Main.a76b78461416b4a8(ptr %x, i32 %count), !dbg !221
// CHECK:STDOUT:   ret ptr %F.call, !dbg !222
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CC.Main.4d6ceefeb7f772f4(i32 %x, i32 %count) #0 !dbg !256 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.call = icmp sle i32 %count, 2, !dbg !261
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.LessOrEquivalent.call, label %if.then, label %if.else, !dbg !262
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 %count), !dbg !264
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !266
// CHECK:STDOUT:   call void @_CB.Main.4d6ceefeb7f772f4(i32 %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !267
// CHECK:STDOUT:   br label %if.else, !dbg !268
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %if.then, %entry
// CHECK:STDOUT:   ret void, !dbg !269
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CE.Main.4d6ceefeb7f772f4(i32 %x, i32 %count) #0 !dbg !271 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call i32 @_CG.Main.4d6ceefeb7f772f4(i32 %x, i32 %count), !dbg !280
// CHECK:STDOUT:   ret i32 %G.call, !dbg !281
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.4d6ceefeb7f772f4(i32 %x, i32 %count) #0 !dbg !283 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call i32 @_CG.Main.4d6ceefeb7f772f4(i32 %x, i32 %count), !dbg !292
// CHECK:STDOUT:   ret i32 %G.call, !dbg !293
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CC.Main.c5e4ba542a56b71f(double %x, i32 %count) #0 !dbg !295 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.call = icmp sle i32 %count, 2, !dbg !300
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.LessOrEquivalent.call, label %if.then, label %if.else, !dbg !301
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 %count), !dbg !303
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !305
// CHECK:STDOUT:   call void @_CB.Main.c5e4ba542a56b71f(double %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !306
// CHECK:STDOUT:   br label %if.else, !dbg !307
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %if.then, %entry
// CHECK:STDOUT:   ret void, !dbg !308
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CE.Main.c5e4ba542a56b71f(double %x, i32 %count) #0 !dbg !310 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call double @_CG.Main.c5e4ba542a56b71f(double %x, i32 %count), !dbg !319
// CHECK:STDOUT:   ret double %G.call, !dbg !320
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CF.Main.c5e4ba542a56b71f(double %x, i32 %count) #0 !dbg !322 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call double @_CG.Main.c5e4ba542a56b71f(double %x, i32 %count), !dbg !331
// CHECK:STDOUT:   ret double %G.call, !dbg !332
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CC.Main.a76b78461416b4a8(ptr %x, i32 %count) #0 !dbg !334 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.call = icmp sle i32 %count, 2, !dbg !339
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.LessOrEquivalent.call, label %if.then, label %if.else, !dbg !340
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 %count), !dbg !342
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !344
// CHECK:STDOUT:   call void @_CB.Main.a76b78461416b4a8(ptr %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !345
// CHECK:STDOUT:   br label %if.else, !dbg !346
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %if.then, %entry
// CHECK:STDOUT:   ret void, !dbg !347
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CE.Main.a76b78461416b4a8(ptr %x, i32 %count) #0 !dbg !349 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call ptr @_CG.Main.a76b78461416b4a8(ptr %x, i32 %count), !dbg !358
// CHECK:STDOUT:   ret ptr %G.call, !dbg !359
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CF.Main.a76b78461416b4a8(ptr %x, i32 %count) #0 !dbg !361 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call ptr @_CG.Main.a76b78461416b4a8(ptr %x, i32 %count), !dbg !370
// CHECK:STDOUT:   ret ptr %G.call, !dbg !371
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @printf(ptr, ...)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CG.Main.4d6ceefeb7f772f4(i32 %x, i32 %count) #0 !dbg !412 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !420
// CHECK:STDOUT:   %D.call = call i32 @_CD.Main.4d6ceefeb7f772f4(i32 %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !421
// CHECK:STDOUT:   ret i32 %D.call, !dbg !422
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CG.Main.c5e4ba542a56b71f(double %x, i32 %count) #0 !dbg !424 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !432
// CHECK:STDOUT:   %D.call = call double @_CD.Main.c5e4ba542a56b71f(double %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !433
// CHECK:STDOUT:   ret double %D.call, !dbg !434
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CG.Main.a76b78461416b4a8(ptr %x, i32 %count) #0 !dbg !436 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !444
// CHECK:STDOUT:   %D.call = call ptr @_CD.Main.a76b78461416b4a8(ptr %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !445
// CHECK:STDOUT:   ret ptr %D.call, !dbg !446
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CA.Main.a76b78461416b4a8, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @printf, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CG.Main.4d6ceefeb7f772f4, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CG.Main.c5e4ba542a56b71f, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CG.Main.a76b78461416b4a8, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "call_recursive_sccs_deep.carbon", directory: "")
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !7 = !{null}
// CHECK:STDOUT: !8 = !DISubroutineType(types: !7)
// CHECK:STDOUT: !9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !10 = !{null, !9}
// CHECK:STDOUT: !11 = !DISubroutineType(types: !10)
// CHECK:STDOUT: !12 = !{null, !6}
// CHECK:STDOUT: !13 = !DISubroutineType(types: !12)
// CHECK:STDOUT: !14 = distinct !DISubprogram(name: "M", linkageName: "_CM.Main", scope: null, file: !4, line: 74, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !15 = !DILocation(line: 75, column: 3, scope: !14)
// CHECK:STDOUT: !17 = !DILocation(line: 76, column: 3, scope: !14)
// CHECK:STDOUT: !19 = !DILocation(line: 77, column: 3, scope: !14)
// CHECK:STDOUT: !23 = !DILocation(line: 78, column: 3, scope: !14)
// CHECK:STDOUT: !27 = !DILocation(line: 80, column: 5, scope: !14)
// CHECK:STDOUT: !28 = !DILocation(line: 80, column: 3, scope: !14)
// CHECK:STDOUT: !29 = !{!6, !6, !6}
// CHECK:STDOUT: !30 = !DISubroutineType(types: !29)
// CHECK:STDOUT: !31 = !DILocation(line: 81, column: 5, scope: !14)
// CHECK:STDOUT: !32 = !DILocation(line: 81, column: 3, scope: !14)
// CHECK:STDOUT: !33 = !{!9, !9, !6}
// CHECK:STDOUT: !34 = !DISubroutineType(types: !33)
// CHECK:STDOUT: !35 = !DILocation(line: 82, column: 5, scope: !14)
// CHECK:STDOUT: !36 = !DILocation(line: 82, column: 3, scope: !14)
// CHECK:STDOUT: !37 = !DILocation(line: 83, column: 5, scope: !14)
// CHECK:STDOUT: !38 = !DILocation(line: 83, column: 3, scope: !14)
// CHECK:STDOUT: !39 = !DILocation(line: 74, column: 1, scope: !14)
// CHECK:STDOUT: !40 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.749b51755634ad17:core.Destroy.Core", scope: null, file: !4, line: 78, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !43)
// CHECK:STDOUT: !41 = !DILocalVariable(arg: 1, scope: !40, type: !9)
// CHECK:STDOUT: !42 = !DILocation(line: 78, column: 3, scope: !40)
// CHECK:STDOUT: !43 = !{!41}
// CHECK:STDOUT: !44 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.d088c7eb746488be:core.Destroy.Core", scope: null, file: !4, line: 77, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !47)
// CHECK:STDOUT: !45 = !DILocalVariable(arg: 1, scope: !44, type: !9)
// CHECK:STDOUT: !46 = !DILocation(line: 77, column: 3, scope: !44)
// CHECK:STDOUT: !47 = !{!45}
// CHECK:STDOUT: !48 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.7945ecba5c505fc0:core.Destroy.Core", scope: null, file: !4, line: 76, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !51)
// CHECK:STDOUT: !49 = !DILocalVariable(arg: 1, scope: !48, type: !9)
// CHECK:STDOUT: !50 = !DILocation(line: 76, column: 3, scope: !48)
// CHECK:STDOUT: !51 = !{!49}
// CHECK:STDOUT: !52 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.dc14d4dd21801791:core.Destroy.Core", scope: null, file: !4, line: 76, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !55)
// CHECK:STDOUT: !53 = !DILocalVariable(arg: 1, scope: !52, type: !9)
// CHECK:STDOUT: !54 = !DILocation(line: 76, column: 3, scope: !52)
// CHECK:STDOUT: !55 = !{!53}
// CHECK:STDOUT: !56 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.6dae29ece9a708c4:core.Destroy.Core", scope: null, file: !4, line: 75, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !59)
// CHECK:STDOUT: !57 = !DILocalVariable(arg: 1, scope: !56, type: !6)
// CHECK:STDOUT: !58 = !DILocation(line: 75, column: 3, scope: !56)
// CHECK:STDOUT: !59 = !{!57}
// CHECK:STDOUT: !60 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f58524831c37807e:core.Destroy.Core", scope: null, file: !4, line: 75, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !63)
// CHECK:STDOUT: !61 = !DILocalVariable(arg: 1, scope: !60, type: !6)
// CHECK:STDOUT: !62 = !DILocation(line: 75, column: 3, scope: !60)
// CHECK:STDOUT: !63 = !{!61}
// CHECK:STDOUT: !64 = distinct !DISubprogram(name: "A", linkageName: "_CA.Main.4d6ceefeb7f772f4", scope: null, file: !4, line: 32, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !80)
// CHECK:STDOUT: !65 = !DILocalVariable(arg: 1, scope: !64, type: !6)
// CHECK:STDOUT: !66 = !DILocalVariable(arg: 2, scope: !64, type: !6)
// CHECK:STDOUT: !73 = !DILocation(line: 33, column: 3, scope: !64)
// CHECK:STDOUT: !74 = !{null, !6, !6}
// CHECK:STDOUT: !75 = !DISubroutineType(types: !74)
// CHECK:STDOUT: !78 = !DILocation(line: 34, column: 10, scope: !64)
// CHECK:STDOUT: !79 = !DILocation(line: 34, column: 3, scope: !64)
// CHECK:STDOUT: !80 = !{!65, !66}
// CHECK:STDOUT: !81 = distinct !DISubprogram(name: "A", linkageName: "_CA.Main.c5e4ba542a56b71f", scope: null, file: !4, line: 32, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !97)
// CHECK:STDOUT: !82 = !DILocalVariable(arg: 1, scope: !81, type: !9)
// CHECK:STDOUT: !83 = !DILocalVariable(arg: 2, scope: !81, type: !6)
// CHECK:STDOUT: !90 = !DILocation(line: 33, column: 3, scope: !81)
// CHECK:STDOUT: !91 = !{null, !9, !6}
// CHECK:STDOUT: !92 = !DISubroutineType(types: !91)
// CHECK:STDOUT: !95 = !DILocation(line: 34, column: 10, scope: !81)
// CHECK:STDOUT: !96 = !DILocation(line: 34, column: 3, scope: !81)
// CHECK:STDOUT: !97 = !{!82, !83}
// CHECK:STDOUT: !98 = distinct !DISubprogram(name: "A", linkageName: "_CA.Main.a76b78461416b4a8", scope: null, file: !4, line: 32, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !112)
// CHECK:STDOUT: !99 = !DILocalVariable(arg: 1, scope: !98, type: !9)
// CHECK:STDOUT: !100 = !DILocalVariable(arg: 2, scope: !98, type: !6)
// CHECK:STDOUT: !107 = !DILocation(line: 33, column: 3, scope: !98)
// CHECK:STDOUT: !110 = !DILocation(line: 34, column: 10, scope: !98)
// CHECK:STDOUT: !111 = !DILocation(line: 34, column: 3, scope: !98)
// CHECK:STDOUT: !112 = !{!99, !100}
// CHECK:STDOUT: !128 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.4d6ceefeb7f772f4", scope: null, file: !4, line: 37, type: !75, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !137)
// CHECK:STDOUT: !129 = !DILocalVariable(arg: 1, scope: !128, type: !6)
// CHECK:STDOUT: !130 = !DILocalVariable(arg: 2, scope: !128, type: !6)
// CHECK:STDOUT: !135 = !DILocation(line: 38, column: 3, scope: !128)
// CHECK:STDOUT: !136 = !DILocation(line: 37, column: 1, scope: !128)
// CHECK:STDOUT: !137 = !{!129, !130}
// CHECK:STDOUT: !138 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.4d6ceefeb7f772f4", scope: null, file: !4, line: 48, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !159)
// CHECK:STDOUT: !139 = !DILocalVariable(arg: 1, scope: !138, type: !6)
// CHECK:STDOUT: !140 = !DILocalVariable(arg: 2, scope: !138, type: !6)
// CHECK:STDOUT: !145 = !DILocation(line: 49, column: 7, scope: !138)
// CHECK:STDOUT: !146 = !DILocation(line: 49, column: 6, scope: !138)
// CHECK:STDOUT: !148 = !DILocation(line: 50, column: 5, scope: !138)
// CHECK:STDOUT: !149 = !DILocation(line: 52, column: 7, scope: !138)
// CHECK:STDOUT: !150 = !DILocation(line: 52, column: 6, scope: !138)
// CHECK:STDOUT: !153 = !DILocation(line: 53, column: 12, scope: !138)
// CHECK:STDOUT: !154 = !DILocation(line: 53, column: 5, scope: !138)
// CHECK:STDOUT: !157 = !DILocation(line: 55, column: 12, scope: !138)
// CHECK:STDOUT: !158 = !DILocation(line: 55, column: 5, scope: !138)
// CHECK:STDOUT: !159 = !{!139, !140}
// CHECK:STDOUT: !160 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.c5e4ba542a56b71f", scope: null, file: !4, line: 37, type: !92, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !169)
// CHECK:STDOUT: !161 = !DILocalVariable(arg: 1, scope: !160, type: !9)
// CHECK:STDOUT: !162 = !DILocalVariable(arg: 2, scope: !160, type: !6)
// CHECK:STDOUT: !167 = !DILocation(line: 38, column: 3, scope: !160)
// CHECK:STDOUT: !168 = !DILocation(line: 37, column: 1, scope: !160)
// CHECK:STDOUT: !169 = !{!161, !162}
// CHECK:STDOUT: !170 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.c5e4ba542a56b71f", scope: null, file: !4, line: 48, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !191)
// CHECK:STDOUT: !171 = !DILocalVariable(arg: 1, scope: !170, type: !9)
// CHECK:STDOUT: !172 = !DILocalVariable(arg: 2, scope: !170, type: !6)
// CHECK:STDOUT: !177 = !DILocation(line: 49, column: 7, scope: !170)
// CHECK:STDOUT: !178 = !DILocation(line: 49, column: 6, scope: !170)
// CHECK:STDOUT: !180 = !DILocation(line: 50, column: 5, scope: !170)
// CHECK:STDOUT: !181 = !DILocation(line: 52, column: 7, scope: !170)
// CHECK:STDOUT: !182 = !DILocation(line: 52, column: 6, scope: !170)
// CHECK:STDOUT: !185 = !DILocation(line: 53, column: 12, scope: !170)
// CHECK:STDOUT: !186 = !DILocation(line: 53, column: 5, scope: !170)
// CHECK:STDOUT: !189 = !DILocation(line: 55, column: 12, scope: !170)
// CHECK:STDOUT: !190 = !DILocation(line: 55, column: 5, scope: !170)
// CHECK:STDOUT: !191 = !{!171, !172}
// CHECK:STDOUT: !192 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.a76b78461416b4a8", scope: null, file: !4, line: 37, type: !92, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !201)
// CHECK:STDOUT: !193 = !DILocalVariable(arg: 1, scope: !192, type: !9)
// CHECK:STDOUT: !194 = !DILocalVariable(arg: 2, scope: !192, type: !6)
// CHECK:STDOUT: !199 = !DILocation(line: 38, column: 3, scope: !192)
// CHECK:STDOUT: !200 = !DILocation(line: 37, column: 1, scope: !192)
// CHECK:STDOUT: !201 = !{!193, !194}
// CHECK:STDOUT: !202 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.a76b78461416b4a8", scope: null, file: !4, line: 48, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !223)
// CHECK:STDOUT: !203 = !DILocalVariable(arg: 1, scope: !202, type: !9)
// CHECK:STDOUT: !204 = !DILocalVariable(arg: 2, scope: !202, type: !6)
// CHECK:STDOUT: !209 = !DILocation(line: 49, column: 7, scope: !202)
// CHECK:STDOUT: !210 = !DILocation(line: 49, column: 6, scope: !202)
// CHECK:STDOUT: !212 = !DILocation(line: 50, column: 5, scope: !202)
// CHECK:STDOUT: !213 = !DILocation(line: 52, column: 7, scope: !202)
// CHECK:STDOUT: !214 = !DILocation(line: 52, column: 6, scope: !202)
// CHECK:STDOUT: !217 = !DILocation(line: 53, column: 12, scope: !202)
// CHECK:STDOUT: !218 = !DILocation(line: 53, column: 5, scope: !202)
// CHECK:STDOUT: !221 = !DILocation(line: 55, column: 12, scope: !202)
// CHECK:STDOUT: !222 = !DILocation(line: 55, column: 5, scope: !202)
// CHECK:STDOUT: !223 = !{!203, !204}
// CHECK:STDOUT: !256 = distinct !DISubprogram(name: "C", linkageName: "_CC.Main.4d6ceefeb7f772f4", scope: null, file: !4, line: 41, type: !75, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !270)
// CHECK:STDOUT: !257 = !DILocalVariable(arg: 1, scope: !256, type: !6)
// CHECK:STDOUT: !258 = !DILocalVariable(arg: 2, scope: !256, type: !6)
// CHECK:STDOUT: !261 = !DILocation(line: 42, column: 7, scope: !256)
// CHECK:STDOUT: !262 = !DILocation(line: 42, column: 6, scope: !256)
// CHECK:STDOUT: !264 = !DILocation(line: 43, column: 5, scope: !256)
// CHECK:STDOUT: !266 = !DILocation(line: 44, column: 10, scope: !256)
// CHECK:STDOUT: !267 = !DILocation(line: 44, column: 5, scope: !256)
// CHECK:STDOUT: !268 = !DILocation(line: 42, column: 3, scope: !256)
// CHECK:STDOUT: !269 = !DILocation(line: 41, column: 1, scope: !256)
// CHECK:STDOUT: !270 = !{!257, !258}
// CHECK:STDOUT: !271 = distinct !DISubprogram(name: "E", linkageName: "_CE.Main.4d6ceefeb7f772f4", scope: null, file: !4, line: 61, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !282)
// CHECK:STDOUT: !272 = !DILocalVariable(arg: 1, scope: !271, type: !6)
// CHECK:STDOUT: !273 = !DILocalVariable(arg: 2, scope: !271, type: !6)
// CHECK:STDOUT: !280 = !DILocation(line: 62, column: 10, scope: !271)
// CHECK:STDOUT: !281 = !DILocation(line: 62, column: 3, scope: !271)
// CHECK:STDOUT: !282 = !{!272, !273}
// CHECK:STDOUT: !283 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.4d6ceefeb7f772f4", scope: null, file: !4, line: 65, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !294)
// CHECK:STDOUT: !284 = !DILocalVariable(arg: 1, scope: !283, type: !6)
// CHECK:STDOUT: !285 = !DILocalVariable(arg: 2, scope: !283, type: !6)
// CHECK:STDOUT: !292 = !DILocation(line: 66, column: 10, scope: !283)
// CHECK:STDOUT: !293 = !DILocation(line: 66, column: 3, scope: !283)
// CHECK:STDOUT: !294 = !{!284, !285}
// CHECK:STDOUT: !295 = distinct !DISubprogram(name: "C", linkageName: "_CC.Main.c5e4ba542a56b71f", scope: null, file: !4, line: 41, type: !92, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !309)
// CHECK:STDOUT: !296 = !DILocalVariable(arg: 1, scope: !295, type: !9)
// CHECK:STDOUT: !297 = !DILocalVariable(arg: 2, scope: !295, type: !6)
// CHECK:STDOUT: !300 = !DILocation(line: 42, column: 7, scope: !295)
// CHECK:STDOUT: !301 = !DILocation(line: 42, column: 6, scope: !295)
// CHECK:STDOUT: !303 = !DILocation(line: 43, column: 5, scope: !295)
// CHECK:STDOUT: !305 = !DILocation(line: 44, column: 10, scope: !295)
// CHECK:STDOUT: !306 = !DILocation(line: 44, column: 5, scope: !295)
// CHECK:STDOUT: !307 = !DILocation(line: 42, column: 3, scope: !295)
// CHECK:STDOUT: !308 = !DILocation(line: 41, column: 1, scope: !295)
// CHECK:STDOUT: !309 = !{!296, !297}
// CHECK:STDOUT: !310 = distinct !DISubprogram(name: "E", linkageName: "_CE.Main.c5e4ba542a56b71f", scope: null, file: !4, line: 61, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !321)
// CHECK:STDOUT: !311 = !DILocalVariable(arg: 1, scope: !310, type: !9)
// CHECK:STDOUT: !312 = !DILocalVariable(arg: 2, scope: !310, type: !6)
// CHECK:STDOUT: !319 = !DILocation(line: 62, column: 10, scope: !310)
// CHECK:STDOUT: !320 = !DILocation(line: 62, column: 3, scope: !310)
// CHECK:STDOUT: !321 = !{!311, !312}
// CHECK:STDOUT: !322 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.c5e4ba542a56b71f", scope: null, file: !4, line: 65, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !333)
// CHECK:STDOUT: !323 = !DILocalVariable(arg: 1, scope: !322, type: !9)
// CHECK:STDOUT: !324 = !DILocalVariable(arg: 2, scope: !322, type: !6)
// CHECK:STDOUT: !331 = !DILocation(line: 66, column: 10, scope: !322)
// CHECK:STDOUT: !332 = !DILocation(line: 66, column: 3, scope: !322)
// CHECK:STDOUT: !333 = !{!323, !324}
// CHECK:STDOUT: !334 = distinct !DISubprogram(name: "C", linkageName: "_CC.Main.a76b78461416b4a8", scope: null, file: !4, line: 41, type: !92, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !348)
// CHECK:STDOUT: !335 = !DILocalVariable(arg: 1, scope: !334, type: !9)
// CHECK:STDOUT: !336 = !DILocalVariable(arg: 2, scope: !334, type: !6)
// CHECK:STDOUT: !339 = !DILocation(line: 42, column: 7, scope: !334)
// CHECK:STDOUT: !340 = !DILocation(line: 42, column: 6, scope: !334)
// CHECK:STDOUT: !342 = !DILocation(line: 43, column: 5, scope: !334)
// CHECK:STDOUT: !344 = !DILocation(line: 44, column: 10, scope: !334)
// CHECK:STDOUT: !345 = !DILocation(line: 44, column: 5, scope: !334)
// CHECK:STDOUT: !346 = !DILocation(line: 42, column: 3, scope: !334)
// CHECK:STDOUT: !347 = !DILocation(line: 41, column: 1, scope: !334)
// CHECK:STDOUT: !348 = !{!335, !336}
// CHECK:STDOUT: !349 = distinct !DISubprogram(name: "E", linkageName: "_CE.Main.a76b78461416b4a8", scope: null, file: !4, line: 61, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !360)
// CHECK:STDOUT: !350 = !DILocalVariable(arg: 1, scope: !349, type: !9)
// CHECK:STDOUT: !351 = !DILocalVariable(arg: 2, scope: !349, type: !6)
// CHECK:STDOUT: !358 = !DILocation(line: 62, column: 10, scope: !349)
// CHECK:STDOUT: !359 = !DILocation(line: 62, column: 3, scope: !349)
// CHECK:STDOUT: !360 = !{!350, !351}
// CHECK:STDOUT: !361 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.a76b78461416b4a8", scope: null, file: !4, line: 65, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !372)
// CHECK:STDOUT: !362 = !DILocalVariable(arg: 1, scope: !361, type: !9)
// CHECK:STDOUT: !363 = !DILocalVariable(arg: 2, scope: !361, type: !6)
// CHECK:STDOUT: !370 = !DILocation(line: 66, column: 10, scope: !361)
// CHECK:STDOUT: !371 = !DILocation(line: 66, column: 3, scope: !361)
// CHECK:STDOUT: !372 = !{!362, !363}
// CHECK:STDOUT: !412 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.4d6ceefeb7f772f4", scope: null, file: !4, line: 69, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !423)
// CHECK:STDOUT: !413 = !DILocalVariable(arg: 1, scope: !412, type: !6)
// CHECK:STDOUT: !414 = !DILocalVariable(arg: 2, scope: !412, type: !6)
// CHECK:STDOUT: !420 = !DILocation(line: 70, column: 15, scope: !412)
// CHECK:STDOUT: !421 = !DILocation(line: 70, column: 10, scope: !412)
// CHECK:STDOUT: !422 = !DILocation(line: 70, column: 3, scope: !412)
// CHECK:STDOUT: !423 = !{!413, !414}
// CHECK:STDOUT: !424 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.c5e4ba542a56b71f", scope: null, file: !4, line: 69, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !435)
// CHECK:STDOUT: !425 = !DILocalVariable(arg: 1, scope: !424, type: !9)
// CHECK:STDOUT: !426 = !DILocalVariable(arg: 2, scope: !424, type: !6)
// CHECK:STDOUT: !432 = !DILocation(line: 70, column: 15, scope: !424)
// CHECK:STDOUT: !433 = !DILocation(line: 70, column: 10, scope: !424)
// CHECK:STDOUT: !434 = !DILocation(line: 70, column: 3, scope: !424)
// CHECK:STDOUT: !435 = !{!425, !426}
// CHECK:STDOUT: !436 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.a76b78461416b4a8", scope: null, file: !4, line: 69, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !447)
// CHECK:STDOUT: !437 = !DILocalVariable(arg: 1, scope: !436, type: !9)
// CHECK:STDOUT: !438 = !DILocalVariable(arg: 2, scope: !436, type: !6)
// CHECK:STDOUT: !444 = !DILocation(line: 70, column: 15, scope: !436)
// CHECK:STDOUT: !445 = !DILocation(line: 70, column: 10, scope: !436)
// CHECK:STDOUT: !446 = !DILocation(line: 70, column: 3, scope: !436)
// CHECK:STDOUT: !447 = !{!437, !438}
// CHECK:STDOUT:
