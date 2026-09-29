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
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.dc14d4dd21801791:core.Destroy.Core"(ptr %self) #0 !dbg !48 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !50
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !52 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !54
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CA.Main.4d6ceefeb7f772f4(i32 %x, i32 %count) #0 !dbg !56 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CB.Main.4d6ceefeb7f772f4(i32 %x, i32 %count), !dbg !65
// CHECK:STDOUT:   %D.call = call i32 @_CD.Main.4d6ceefeb7f772f4(i32 %x, i32 %count), !dbg !70
// CHECK:STDOUT:   ret i32 %D.call, !dbg !71
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CA.Main.c5e4ba542a56b71f(double %x, i32 %count) #0 !dbg !73 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CB.Main.c5e4ba542a56b71f(double %x, i32 %count), !dbg !82
// CHECK:STDOUT:   %D.call = call double @_CD.Main.c5e4ba542a56b71f(double %x, i32 %count), !dbg !87
// CHECK:STDOUT:   ret double %D.call, !dbg !88
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CA.Main.a76b78461416b4a8(ptr %x, i32 %count) #0 !dbg !90 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CB.Main.a76b78461416b4a8(ptr %x, i32 %count), !dbg !99
// CHECK:STDOUT:   %D.call = call ptr @_CD.Main.a76b78461416b4a8(ptr %x, i32 %count), !dbg !102
// CHECK:STDOUT:   ret ptr %D.call, !dbg !103
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CB.Main.4d6ceefeb7f772f4(i32 %x, i32 %count) #0 !dbg !120 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CC.Main.4d6ceefeb7f772f4(i32 %x, i32 %count), !dbg !127
// CHECK:STDOUT:   ret void, !dbg !128
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CD.Main.4d6ceefeb7f772f4(i32 %x, i32 %count) #0 !dbg !130 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 4, !dbg !137
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then.loc49, label %if.else.loc49, !dbg !138
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc49:                                    ; preds = %entry
// CHECK:STDOUT:   ret i32 %x, !dbg !140
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc49:                                    ; preds = %entry
// CHECK:STDOUT:   %Int.as.ModWith.impl.Op.call = srem i32 %count, 2, !dbg !141
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.call = icmp eq i32 %Int.as.ModWith.impl.Op.call, 0, !dbg !141
// CHECK:STDOUT:   br i1 %Int.as.EqWith.impl.Equal.call, label %if.then.loc52, label %if.else.loc52, !dbg !142
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc52:                                    ; preds = %if.else.loc49
// CHECK:STDOUT:   %E.call = call i32 @_CE.Main.4d6ceefeb7f772f4(i32 %x, i32 %count), !dbg !145
// CHECK:STDOUT:   ret i32 %E.call, !dbg !146
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc52:                                    ; preds = %if.else.loc49
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main.4d6ceefeb7f772f4(i32 %x, i32 %count), !dbg !149
// CHECK:STDOUT:   ret i32 %F.call, !dbg !150
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CB.Main.c5e4ba542a56b71f(double %x, i32 %count) #0 !dbg !152 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CC.Main.c5e4ba542a56b71f(double %x, i32 %count), !dbg !159
// CHECK:STDOUT:   ret void, !dbg !160
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CD.Main.c5e4ba542a56b71f(double %x, i32 %count) #0 !dbg !162 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 4, !dbg !169
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then.loc49, label %if.else.loc49, !dbg !170
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc49:                                    ; preds = %entry
// CHECK:STDOUT:   ret double %x, !dbg !172
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc49:                                    ; preds = %entry
// CHECK:STDOUT:   %Int.as.ModWith.impl.Op.call = srem i32 %count, 2, !dbg !173
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.call = icmp eq i32 %Int.as.ModWith.impl.Op.call, 0, !dbg !173
// CHECK:STDOUT:   br i1 %Int.as.EqWith.impl.Equal.call, label %if.then.loc52, label %if.else.loc52, !dbg !174
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc52:                                    ; preds = %if.else.loc49
// CHECK:STDOUT:   %E.call = call double @_CE.Main.c5e4ba542a56b71f(double %x, i32 %count), !dbg !177
// CHECK:STDOUT:   ret double %E.call, !dbg !178
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc52:                                    ; preds = %if.else.loc49
// CHECK:STDOUT:   %F.call = call double @_CF.Main.c5e4ba542a56b71f(double %x, i32 %count), !dbg !181
// CHECK:STDOUT:   ret double %F.call, !dbg !182
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CB.Main.a76b78461416b4a8(ptr %x, i32 %count) #0 !dbg !184 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CC.Main.a76b78461416b4a8(ptr %x, i32 %count), !dbg !191
// CHECK:STDOUT:   ret void, !dbg !192
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CD.Main.a76b78461416b4a8(ptr %x, i32 %count) #0 !dbg !194 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 4, !dbg !201
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then.loc49, label %if.else.loc49, !dbg !202
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc49:                                    ; preds = %entry
// CHECK:STDOUT:   ret ptr %x, !dbg !204
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc49:                                    ; preds = %entry
// CHECK:STDOUT:   %Int.as.ModWith.impl.Op.call = srem i32 %count, 2, !dbg !205
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.call = icmp eq i32 %Int.as.ModWith.impl.Op.call, 0, !dbg !205
// CHECK:STDOUT:   br i1 %Int.as.EqWith.impl.Equal.call, label %if.then.loc52, label %if.else.loc52, !dbg !206
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc52:                                    ; preds = %if.else.loc49
// CHECK:STDOUT:   %E.call = call ptr @_CE.Main.a76b78461416b4a8(ptr %x, i32 %count), !dbg !209
// CHECK:STDOUT:   ret ptr %E.call, !dbg !210
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc52:                                    ; preds = %if.else.loc49
// CHECK:STDOUT:   %F.call = call ptr @_CF.Main.a76b78461416b4a8(ptr %x, i32 %count), !dbg !213
// CHECK:STDOUT:   ret ptr %F.call, !dbg !214
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CC.Main.4d6ceefeb7f772f4(i32 %x, i32 %count) #0 !dbg !248 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.call = icmp sle i32 %count, 2, !dbg !253
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.LessOrEquivalent.call, label %if.then, label %if.else, !dbg !254
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 %count), !dbg !256
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !258
// CHECK:STDOUT:   call void @_CB.Main.4d6ceefeb7f772f4(i32 %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !259
// CHECK:STDOUT:   br label %if.else, !dbg !260
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %if.then, %entry
// CHECK:STDOUT:   ret void, !dbg !261
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CE.Main.4d6ceefeb7f772f4(i32 %x, i32 %count) #0 !dbg !263 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call i32 @_CG.Main.4d6ceefeb7f772f4(i32 %x, i32 %count), !dbg !272
// CHECK:STDOUT:   ret i32 %G.call, !dbg !273
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.4d6ceefeb7f772f4(i32 %x, i32 %count) #0 !dbg !275 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call i32 @_CG.Main.4d6ceefeb7f772f4(i32 %x, i32 %count), !dbg !284
// CHECK:STDOUT:   ret i32 %G.call, !dbg !285
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CC.Main.c5e4ba542a56b71f(double %x, i32 %count) #0 !dbg !287 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.call = icmp sle i32 %count, 2, !dbg !292
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.LessOrEquivalent.call, label %if.then, label %if.else, !dbg !293
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 %count), !dbg !295
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !297
// CHECK:STDOUT:   call void @_CB.Main.c5e4ba542a56b71f(double %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !298
// CHECK:STDOUT:   br label %if.else, !dbg !299
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %if.then, %entry
// CHECK:STDOUT:   ret void, !dbg !300
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CE.Main.c5e4ba542a56b71f(double %x, i32 %count) #0 !dbg !302 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call double @_CG.Main.c5e4ba542a56b71f(double %x, i32 %count), !dbg !311
// CHECK:STDOUT:   ret double %G.call, !dbg !312
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CF.Main.c5e4ba542a56b71f(double %x, i32 %count) #0 !dbg !314 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call double @_CG.Main.c5e4ba542a56b71f(double %x, i32 %count), !dbg !323
// CHECK:STDOUT:   ret double %G.call, !dbg !324
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CC.Main.a76b78461416b4a8(ptr %x, i32 %count) #0 !dbg !326 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.call = icmp sle i32 %count, 2, !dbg !331
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.LessOrEquivalent.call, label %if.then, label %if.else, !dbg !332
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 %count), !dbg !334
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !336
// CHECK:STDOUT:   call void @_CB.Main.a76b78461416b4a8(ptr %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !337
// CHECK:STDOUT:   br label %if.else, !dbg !338
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %if.then, %entry
// CHECK:STDOUT:   ret void, !dbg !339
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CE.Main.a76b78461416b4a8(ptr %x, i32 %count) #0 !dbg !341 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call ptr @_CG.Main.a76b78461416b4a8(ptr %x, i32 %count), !dbg !350
// CHECK:STDOUT:   ret ptr %G.call, !dbg !351
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CF.Main.a76b78461416b4a8(ptr %x, i32 %count) #0 !dbg !353 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call ptr @_CG.Main.a76b78461416b4a8(ptr %x, i32 %count), !dbg !362
// CHECK:STDOUT:   ret ptr %G.call, !dbg !363
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @printf(ptr, ...)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CG.Main.4d6ceefeb7f772f4(i32 %x, i32 %count) #0 !dbg !404 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !412
// CHECK:STDOUT:   %D.call = call i32 @_CD.Main.4d6ceefeb7f772f4(i32 %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !413
// CHECK:STDOUT:   ret i32 %D.call, !dbg !414
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CG.Main.c5e4ba542a56b71f(double %x, i32 %count) #0 !dbg !416 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !424
// CHECK:STDOUT:   %D.call = call double @_CD.Main.c5e4ba542a56b71f(double %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !425
// CHECK:STDOUT:   ret double %D.call, !dbg !426
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CG.Main.a76b78461416b4a8(ptr %x, i32 %count) #0 !dbg !428 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !436
// CHECK:STDOUT:   %D.call = call ptr @_CD.Main.a76b78461416b4a8(ptr %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !437
// CHECK:STDOUT:   ret ptr %D.call, !dbg !438
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
// CHECK:STDOUT: !48 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.dc14d4dd21801791:core.Destroy.Core", scope: null, file: !4, line: 76, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !51)
// CHECK:STDOUT: !49 = !DILocalVariable(arg: 1, scope: !48, type: !9)
// CHECK:STDOUT: !50 = !DILocation(line: 76, column: 3, scope: !48)
// CHECK:STDOUT: !51 = !{!49}
// CHECK:STDOUT: !52 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f58524831c37807e:core.Destroy.Core", scope: null, file: !4, line: 75, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !55)
// CHECK:STDOUT: !53 = !DILocalVariable(arg: 1, scope: !52, type: !6)
// CHECK:STDOUT: !54 = !DILocation(line: 75, column: 3, scope: !52)
// CHECK:STDOUT: !55 = !{!53}
// CHECK:STDOUT: !56 = distinct !DISubprogram(name: "A", linkageName: "_CA.Main.4d6ceefeb7f772f4", scope: null, file: !4, line: 32, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !72)
// CHECK:STDOUT: !57 = !DILocalVariable(arg: 1, scope: !56, type: !6)
// CHECK:STDOUT: !58 = !DILocalVariable(arg: 2, scope: !56, type: !6)
// CHECK:STDOUT: !65 = !DILocation(line: 33, column: 3, scope: !56)
// CHECK:STDOUT: !66 = !{null, !6, !6}
// CHECK:STDOUT: !67 = !DISubroutineType(types: !66)
// CHECK:STDOUT: !70 = !DILocation(line: 34, column: 10, scope: !56)
// CHECK:STDOUT: !71 = !DILocation(line: 34, column: 3, scope: !56)
// CHECK:STDOUT: !72 = !{!57, !58}
// CHECK:STDOUT: !73 = distinct !DISubprogram(name: "A", linkageName: "_CA.Main.c5e4ba542a56b71f", scope: null, file: !4, line: 32, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !89)
// CHECK:STDOUT: !74 = !DILocalVariable(arg: 1, scope: !73, type: !9)
// CHECK:STDOUT: !75 = !DILocalVariable(arg: 2, scope: !73, type: !6)
// CHECK:STDOUT: !82 = !DILocation(line: 33, column: 3, scope: !73)
// CHECK:STDOUT: !83 = !{null, !9, !6}
// CHECK:STDOUT: !84 = !DISubroutineType(types: !83)
// CHECK:STDOUT: !87 = !DILocation(line: 34, column: 10, scope: !73)
// CHECK:STDOUT: !88 = !DILocation(line: 34, column: 3, scope: !73)
// CHECK:STDOUT: !89 = !{!74, !75}
// CHECK:STDOUT: !90 = distinct !DISubprogram(name: "A", linkageName: "_CA.Main.a76b78461416b4a8", scope: null, file: !4, line: 32, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !104)
// CHECK:STDOUT: !91 = !DILocalVariable(arg: 1, scope: !90, type: !9)
// CHECK:STDOUT: !92 = !DILocalVariable(arg: 2, scope: !90, type: !6)
// CHECK:STDOUT: !99 = !DILocation(line: 33, column: 3, scope: !90)
// CHECK:STDOUT: !102 = !DILocation(line: 34, column: 10, scope: !90)
// CHECK:STDOUT: !103 = !DILocation(line: 34, column: 3, scope: !90)
// CHECK:STDOUT: !104 = !{!91, !92}
// CHECK:STDOUT: !120 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.4d6ceefeb7f772f4", scope: null, file: !4, line: 37, type: !67, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !129)
// CHECK:STDOUT: !121 = !DILocalVariable(arg: 1, scope: !120, type: !6)
// CHECK:STDOUT: !122 = !DILocalVariable(arg: 2, scope: !120, type: !6)
// CHECK:STDOUT: !127 = !DILocation(line: 38, column: 3, scope: !120)
// CHECK:STDOUT: !128 = !DILocation(line: 37, column: 1, scope: !120)
// CHECK:STDOUT: !129 = !{!121, !122}
// CHECK:STDOUT: !130 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.4d6ceefeb7f772f4", scope: null, file: !4, line: 48, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !151)
// CHECK:STDOUT: !131 = !DILocalVariable(arg: 1, scope: !130, type: !6)
// CHECK:STDOUT: !132 = !DILocalVariable(arg: 2, scope: !130, type: !6)
// CHECK:STDOUT: !137 = !DILocation(line: 49, column: 7, scope: !130)
// CHECK:STDOUT: !138 = !DILocation(line: 49, column: 6, scope: !130)
// CHECK:STDOUT: !140 = !DILocation(line: 50, column: 5, scope: !130)
// CHECK:STDOUT: !141 = !DILocation(line: 52, column: 7, scope: !130)
// CHECK:STDOUT: !142 = !DILocation(line: 52, column: 6, scope: !130)
// CHECK:STDOUT: !145 = !DILocation(line: 53, column: 12, scope: !130)
// CHECK:STDOUT: !146 = !DILocation(line: 53, column: 5, scope: !130)
// CHECK:STDOUT: !149 = !DILocation(line: 55, column: 12, scope: !130)
// CHECK:STDOUT: !150 = !DILocation(line: 55, column: 5, scope: !130)
// CHECK:STDOUT: !151 = !{!131, !132}
// CHECK:STDOUT: !152 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.c5e4ba542a56b71f", scope: null, file: !4, line: 37, type: !84, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !161)
// CHECK:STDOUT: !153 = !DILocalVariable(arg: 1, scope: !152, type: !9)
// CHECK:STDOUT: !154 = !DILocalVariable(arg: 2, scope: !152, type: !6)
// CHECK:STDOUT: !159 = !DILocation(line: 38, column: 3, scope: !152)
// CHECK:STDOUT: !160 = !DILocation(line: 37, column: 1, scope: !152)
// CHECK:STDOUT: !161 = !{!153, !154}
// CHECK:STDOUT: !162 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.c5e4ba542a56b71f", scope: null, file: !4, line: 48, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !183)
// CHECK:STDOUT: !163 = !DILocalVariable(arg: 1, scope: !162, type: !9)
// CHECK:STDOUT: !164 = !DILocalVariable(arg: 2, scope: !162, type: !6)
// CHECK:STDOUT: !169 = !DILocation(line: 49, column: 7, scope: !162)
// CHECK:STDOUT: !170 = !DILocation(line: 49, column: 6, scope: !162)
// CHECK:STDOUT: !172 = !DILocation(line: 50, column: 5, scope: !162)
// CHECK:STDOUT: !173 = !DILocation(line: 52, column: 7, scope: !162)
// CHECK:STDOUT: !174 = !DILocation(line: 52, column: 6, scope: !162)
// CHECK:STDOUT: !177 = !DILocation(line: 53, column: 12, scope: !162)
// CHECK:STDOUT: !178 = !DILocation(line: 53, column: 5, scope: !162)
// CHECK:STDOUT: !181 = !DILocation(line: 55, column: 12, scope: !162)
// CHECK:STDOUT: !182 = !DILocation(line: 55, column: 5, scope: !162)
// CHECK:STDOUT: !183 = !{!163, !164}
// CHECK:STDOUT: !184 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.a76b78461416b4a8", scope: null, file: !4, line: 37, type: !84, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !193)
// CHECK:STDOUT: !185 = !DILocalVariable(arg: 1, scope: !184, type: !9)
// CHECK:STDOUT: !186 = !DILocalVariable(arg: 2, scope: !184, type: !6)
// CHECK:STDOUT: !191 = !DILocation(line: 38, column: 3, scope: !184)
// CHECK:STDOUT: !192 = !DILocation(line: 37, column: 1, scope: !184)
// CHECK:STDOUT: !193 = !{!185, !186}
// CHECK:STDOUT: !194 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.a76b78461416b4a8", scope: null, file: !4, line: 48, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !215)
// CHECK:STDOUT: !195 = !DILocalVariable(arg: 1, scope: !194, type: !9)
// CHECK:STDOUT: !196 = !DILocalVariable(arg: 2, scope: !194, type: !6)
// CHECK:STDOUT: !201 = !DILocation(line: 49, column: 7, scope: !194)
// CHECK:STDOUT: !202 = !DILocation(line: 49, column: 6, scope: !194)
// CHECK:STDOUT: !204 = !DILocation(line: 50, column: 5, scope: !194)
// CHECK:STDOUT: !205 = !DILocation(line: 52, column: 7, scope: !194)
// CHECK:STDOUT: !206 = !DILocation(line: 52, column: 6, scope: !194)
// CHECK:STDOUT: !209 = !DILocation(line: 53, column: 12, scope: !194)
// CHECK:STDOUT: !210 = !DILocation(line: 53, column: 5, scope: !194)
// CHECK:STDOUT: !213 = !DILocation(line: 55, column: 12, scope: !194)
// CHECK:STDOUT: !214 = !DILocation(line: 55, column: 5, scope: !194)
// CHECK:STDOUT: !215 = !{!195, !196}
// CHECK:STDOUT: !248 = distinct !DISubprogram(name: "C", linkageName: "_CC.Main.4d6ceefeb7f772f4", scope: null, file: !4, line: 41, type: !67, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !262)
// CHECK:STDOUT: !249 = !DILocalVariable(arg: 1, scope: !248, type: !6)
// CHECK:STDOUT: !250 = !DILocalVariable(arg: 2, scope: !248, type: !6)
// CHECK:STDOUT: !253 = !DILocation(line: 42, column: 7, scope: !248)
// CHECK:STDOUT: !254 = !DILocation(line: 42, column: 6, scope: !248)
// CHECK:STDOUT: !256 = !DILocation(line: 43, column: 5, scope: !248)
// CHECK:STDOUT: !258 = !DILocation(line: 44, column: 10, scope: !248)
// CHECK:STDOUT: !259 = !DILocation(line: 44, column: 5, scope: !248)
// CHECK:STDOUT: !260 = !DILocation(line: 42, column: 3, scope: !248)
// CHECK:STDOUT: !261 = !DILocation(line: 41, column: 1, scope: !248)
// CHECK:STDOUT: !262 = !{!249, !250}
// CHECK:STDOUT: !263 = distinct !DISubprogram(name: "E", linkageName: "_CE.Main.4d6ceefeb7f772f4", scope: null, file: !4, line: 61, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !274)
// CHECK:STDOUT: !264 = !DILocalVariable(arg: 1, scope: !263, type: !6)
// CHECK:STDOUT: !265 = !DILocalVariable(arg: 2, scope: !263, type: !6)
// CHECK:STDOUT: !272 = !DILocation(line: 62, column: 10, scope: !263)
// CHECK:STDOUT: !273 = !DILocation(line: 62, column: 3, scope: !263)
// CHECK:STDOUT: !274 = !{!264, !265}
// CHECK:STDOUT: !275 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.4d6ceefeb7f772f4", scope: null, file: !4, line: 65, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !286)
// CHECK:STDOUT: !276 = !DILocalVariable(arg: 1, scope: !275, type: !6)
// CHECK:STDOUT: !277 = !DILocalVariable(arg: 2, scope: !275, type: !6)
// CHECK:STDOUT: !284 = !DILocation(line: 66, column: 10, scope: !275)
// CHECK:STDOUT: !285 = !DILocation(line: 66, column: 3, scope: !275)
// CHECK:STDOUT: !286 = !{!276, !277}
// CHECK:STDOUT: !287 = distinct !DISubprogram(name: "C", linkageName: "_CC.Main.c5e4ba542a56b71f", scope: null, file: !4, line: 41, type: !84, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !301)
// CHECK:STDOUT: !288 = !DILocalVariable(arg: 1, scope: !287, type: !9)
// CHECK:STDOUT: !289 = !DILocalVariable(arg: 2, scope: !287, type: !6)
// CHECK:STDOUT: !292 = !DILocation(line: 42, column: 7, scope: !287)
// CHECK:STDOUT: !293 = !DILocation(line: 42, column: 6, scope: !287)
// CHECK:STDOUT: !295 = !DILocation(line: 43, column: 5, scope: !287)
// CHECK:STDOUT: !297 = !DILocation(line: 44, column: 10, scope: !287)
// CHECK:STDOUT: !298 = !DILocation(line: 44, column: 5, scope: !287)
// CHECK:STDOUT: !299 = !DILocation(line: 42, column: 3, scope: !287)
// CHECK:STDOUT: !300 = !DILocation(line: 41, column: 1, scope: !287)
// CHECK:STDOUT: !301 = !{!288, !289}
// CHECK:STDOUT: !302 = distinct !DISubprogram(name: "E", linkageName: "_CE.Main.c5e4ba542a56b71f", scope: null, file: !4, line: 61, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !313)
// CHECK:STDOUT: !303 = !DILocalVariable(arg: 1, scope: !302, type: !9)
// CHECK:STDOUT: !304 = !DILocalVariable(arg: 2, scope: !302, type: !6)
// CHECK:STDOUT: !311 = !DILocation(line: 62, column: 10, scope: !302)
// CHECK:STDOUT: !312 = !DILocation(line: 62, column: 3, scope: !302)
// CHECK:STDOUT: !313 = !{!303, !304}
// CHECK:STDOUT: !314 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.c5e4ba542a56b71f", scope: null, file: !4, line: 65, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !325)
// CHECK:STDOUT: !315 = !DILocalVariable(arg: 1, scope: !314, type: !9)
// CHECK:STDOUT: !316 = !DILocalVariable(arg: 2, scope: !314, type: !6)
// CHECK:STDOUT: !323 = !DILocation(line: 66, column: 10, scope: !314)
// CHECK:STDOUT: !324 = !DILocation(line: 66, column: 3, scope: !314)
// CHECK:STDOUT: !325 = !{!315, !316}
// CHECK:STDOUT: !326 = distinct !DISubprogram(name: "C", linkageName: "_CC.Main.a76b78461416b4a8", scope: null, file: !4, line: 41, type: !84, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !340)
// CHECK:STDOUT: !327 = !DILocalVariable(arg: 1, scope: !326, type: !9)
// CHECK:STDOUT: !328 = !DILocalVariable(arg: 2, scope: !326, type: !6)
// CHECK:STDOUT: !331 = !DILocation(line: 42, column: 7, scope: !326)
// CHECK:STDOUT: !332 = !DILocation(line: 42, column: 6, scope: !326)
// CHECK:STDOUT: !334 = !DILocation(line: 43, column: 5, scope: !326)
// CHECK:STDOUT: !336 = !DILocation(line: 44, column: 10, scope: !326)
// CHECK:STDOUT: !337 = !DILocation(line: 44, column: 5, scope: !326)
// CHECK:STDOUT: !338 = !DILocation(line: 42, column: 3, scope: !326)
// CHECK:STDOUT: !339 = !DILocation(line: 41, column: 1, scope: !326)
// CHECK:STDOUT: !340 = !{!327, !328}
// CHECK:STDOUT: !341 = distinct !DISubprogram(name: "E", linkageName: "_CE.Main.a76b78461416b4a8", scope: null, file: !4, line: 61, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !352)
// CHECK:STDOUT: !342 = !DILocalVariable(arg: 1, scope: !341, type: !9)
// CHECK:STDOUT: !343 = !DILocalVariable(arg: 2, scope: !341, type: !6)
// CHECK:STDOUT: !350 = !DILocation(line: 62, column: 10, scope: !341)
// CHECK:STDOUT: !351 = !DILocation(line: 62, column: 3, scope: !341)
// CHECK:STDOUT: !352 = !{!342, !343}
// CHECK:STDOUT: !353 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.a76b78461416b4a8", scope: null, file: !4, line: 65, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !364)
// CHECK:STDOUT: !354 = !DILocalVariable(arg: 1, scope: !353, type: !9)
// CHECK:STDOUT: !355 = !DILocalVariable(arg: 2, scope: !353, type: !6)
// CHECK:STDOUT: !362 = !DILocation(line: 66, column: 10, scope: !353)
// CHECK:STDOUT: !363 = !DILocation(line: 66, column: 3, scope: !353)
// CHECK:STDOUT: !364 = !{!354, !355}
// CHECK:STDOUT: !404 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.4d6ceefeb7f772f4", scope: null, file: !4, line: 69, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !415)
// CHECK:STDOUT: !405 = !DILocalVariable(arg: 1, scope: !404, type: !6)
// CHECK:STDOUT: !406 = !DILocalVariable(arg: 2, scope: !404, type: !6)
// CHECK:STDOUT: !412 = !DILocation(line: 70, column: 15, scope: !404)
// CHECK:STDOUT: !413 = !DILocation(line: 70, column: 10, scope: !404)
// CHECK:STDOUT: !414 = !DILocation(line: 70, column: 3, scope: !404)
// CHECK:STDOUT: !415 = !{!405, !406}
// CHECK:STDOUT: !416 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.c5e4ba542a56b71f", scope: null, file: !4, line: 69, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !427)
// CHECK:STDOUT: !417 = !DILocalVariable(arg: 1, scope: !416, type: !9)
// CHECK:STDOUT: !418 = !DILocalVariable(arg: 2, scope: !416, type: !6)
// CHECK:STDOUT: !424 = !DILocation(line: 70, column: 15, scope: !416)
// CHECK:STDOUT: !425 = !DILocation(line: 70, column: 10, scope: !416)
// CHECK:STDOUT: !426 = !DILocation(line: 70, column: 3, scope: !416)
// CHECK:STDOUT: !427 = !{!417, !418}
// CHECK:STDOUT: !428 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.a76b78461416b4a8", scope: null, file: !4, line: 69, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !439)
// CHECK:STDOUT: !429 = !DILocalVariable(arg: 1, scope: !428, type: !9)
// CHECK:STDOUT: !430 = !DILocalVariable(arg: 2, scope: !428, type: !6)
// CHECK:STDOUT: !436 = !DILocation(line: 70, column: 15, scope: !428)
// CHECK:STDOUT: !437 = !DILocation(line: 70, column: 10, scope: !428)
// CHECK:STDOUT: !438 = !DILocation(line: 70, column: 3, scope: !428)
// CHECK:STDOUT: !439 = !{!429, !430}
// CHECK:STDOUT:
