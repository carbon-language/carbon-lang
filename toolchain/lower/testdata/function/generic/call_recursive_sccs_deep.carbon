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
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.dc14d4dd21801791:core.Destroy.Core"(ptr %self) #0 !dbg !52 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !54
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.dc14d4dd21801791:core.Destroy.Core"(ptr %self) #0 !dbg !56 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.dc14d4dd21801791:core.Destroy.Core"(ptr %self), !dbg !58
// CHECK:STDOUT:   ret void, !dbg !58
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.6dae29ece9a708c4:core.Destroy.Core"(ptr %self) #0 !dbg !60 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !62
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !64 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !66
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !68 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self), !dbg !70
// CHECK:STDOUT:   ret void, !dbg !70
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CA.Main.4d6ceefeb7f772f4(i32 %x, i32 %count) #0 !dbg !72 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CB.Main.4d6ceefeb7f772f4(i32 %x, i32 %count), !dbg !81
// CHECK:STDOUT:   %D.call = call i32 @_CD.Main.4d6ceefeb7f772f4(i32 %x, i32 %count), !dbg !86
// CHECK:STDOUT:   ret i32 %D.call, !dbg !87
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CA.Main.c5e4ba542a56b71f(double %x, i32 %count) #0 !dbg !89 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CB.Main.c5e4ba542a56b71f(double %x, i32 %count), !dbg !98
// CHECK:STDOUT:   %D.call = call double @_CD.Main.c5e4ba542a56b71f(double %x, i32 %count), !dbg !103
// CHECK:STDOUT:   ret double %D.call, !dbg !104
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CA.Main.a76b78461416b4a8(ptr %x, i32 %count) #0 !dbg !106 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CB.Main.a76b78461416b4a8(ptr %x, i32 %count), !dbg !115
// CHECK:STDOUT:   %D.call = call ptr @_CD.Main.a76b78461416b4a8(ptr %x, i32 %count), !dbg !118
// CHECK:STDOUT:   ret ptr %D.call, !dbg !119
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CB.Main.4d6ceefeb7f772f4(i32 %x, i32 %count) #0 !dbg !136 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CC.Main.4d6ceefeb7f772f4(i32 %x, i32 %count), !dbg !143
// CHECK:STDOUT:   ret void, !dbg !144
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CD.Main.4d6ceefeb7f772f4(i32 %x, i32 %count) #0 !dbg !146 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 4, !dbg !153
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then.loc49, label %if.else.loc49, !dbg !154
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc49:                                    ; preds = %entry
// CHECK:STDOUT:   ret i32 %x, !dbg !156
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc49:                                    ; preds = %entry
// CHECK:STDOUT:   %Int.as.ModWith.impl.Op.call = srem i32 %count, 2, !dbg !157
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.call = icmp eq i32 %Int.as.ModWith.impl.Op.call, 0, !dbg !157
// CHECK:STDOUT:   br i1 %Int.as.EqWith.impl.Equal.call, label %if.then.loc52, label %if.else.loc52, !dbg !158
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc52:                                    ; preds = %if.else.loc49
// CHECK:STDOUT:   %E.call = call i32 @_CE.Main.4d6ceefeb7f772f4(i32 %x, i32 %count), !dbg !161
// CHECK:STDOUT:   ret i32 %E.call, !dbg !162
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc52:                                    ; preds = %if.else.loc49
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main.4d6ceefeb7f772f4(i32 %x, i32 %count), !dbg !165
// CHECK:STDOUT:   ret i32 %F.call, !dbg !166
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CB.Main.c5e4ba542a56b71f(double %x, i32 %count) #0 !dbg !168 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CC.Main.c5e4ba542a56b71f(double %x, i32 %count), !dbg !175
// CHECK:STDOUT:   ret void, !dbg !176
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CD.Main.c5e4ba542a56b71f(double %x, i32 %count) #0 !dbg !178 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 4, !dbg !185
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then.loc49, label %if.else.loc49, !dbg !186
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc49:                                    ; preds = %entry
// CHECK:STDOUT:   ret double %x, !dbg !188
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc49:                                    ; preds = %entry
// CHECK:STDOUT:   %Int.as.ModWith.impl.Op.call = srem i32 %count, 2, !dbg !189
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.call = icmp eq i32 %Int.as.ModWith.impl.Op.call, 0, !dbg !189
// CHECK:STDOUT:   br i1 %Int.as.EqWith.impl.Equal.call, label %if.then.loc52, label %if.else.loc52, !dbg !190
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc52:                                    ; preds = %if.else.loc49
// CHECK:STDOUT:   %E.call = call double @_CE.Main.c5e4ba542a56b71f(double %x, i32 %count), !dbg !193
// CHECK:STDOUT:   ret double %E.call, !dbg !194
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc52:                                    ; preds = %if.else.loc49
// CHECK:STDOUT:   %F.call = call double @_CF.Main.c5e4ba542a56b71f(double %x, i32 %count), !dbg !197
// CHECK:STDOUT:   ret double %F.call, !dbg !198
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CB.Main.a76b78461416b4a8(ptr %x, i32 %count) #0 !dbg !200 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CC.Main.a76b78461416b4a8(ptr %x, i32 %count), !dbg !207
// CHECK:STDOUT:   ret void, !dbg !208
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CD.Main.a76b78461416b4a8(ptr %x, i32 %count) #0 !dbg !210 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 4, !dbg !217
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then.loc49, label %if.else.loc49, !dbg !218
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc49:                                    ; preds = %entry
// CHECK:STDOUT:   ret ptr %x, !dbg !220
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc49:                                    ; preds = %entry
// CHECK:STDOUT:   %Int.as.ModWith.impl.Op.call = srem i32 %count, 2, !dbg !221
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.call = icmp eq i32 %Int.as.ModWith.impl.Op.call, 0, !dbg !221
// CHECK:STDOUT:   br i1 %Int.as.EqWith.impl.Equal.call, label %if.then.loc52, label %if.else.loc52, !dbg !222
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc52:                                    ; preds = %if.else.loc49
// CHECK:STDOUT:   %E.call = call ptr @_CE.Main.a76b78461416b4a8(ptr %x, i32 %count), !dbg !225
// CHECK:STDOUT:   ret ptr %E.call, !dbg !226
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc52:                                    ; preds = %if.else.loc49
// CHECK:STDOUT:   %F.call = call ptr @_CF.Main.a76b78461416b4a8(ptr %x, i32 %count), !dbg !229
// CHECK:STDOUT:   ret ptr %F.call, !dbg !230
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CC.Main.4d6ceefeb7f772f4(i32 %x, i32 %count) #0 !dbg !264 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.call = icmp sle i32 %count, 2, !dbg !269
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.LessOrEquivalent.call, label %if.then, label %if.else, !dbg !270
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 %count), !dbg !272
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !274
// CHECK:STDOUT:   call void @_CB.Main.4d6ceefeb7f772f4(i32 %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !275
// CHECK:STDOUT:   br label %if.else, !dbg !276
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %if.then, %entry
// CHECK:STDOUT:   ret void, !dbg !277
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CE.Main.4d6ceefeb7f772f4(i32 %x, i32 %count) #0 !dbg !279 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call i32 @_CG.Main.4d6ceefeb7f772f4(i32 %x, i32 %count), !dbg !288
// CHECK:STDOUT:   ret i32 %G.call, !dbg !289
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.4d6ceefeb7f772f4(i32 %x, i32 %count) #0 !dbg !291 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call i32 @_CG.Main.4d6ceefeb7f772f4(i32 %x, i32 %count), !dbg !300
// CHECK:STDOUT:   ret i32 %G.call, !dbg !301
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CC.Main.c5e4ba542a56b71f(double %x, i32 %count) #0 !dbg !303 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.call = icmp sle i32 %count, 2, !dbg !308
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.LessOrEquivalent.call, label %if.then, label %if.else, !dbg !309
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 %count), !dbg !311
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !313
// CHECK:STDOUT:   call void @_CB.Main.c5e4ba542a56b71f(double %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !314
// CHECK:STDOUT:   br label %if.else, !dbg !315
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %if.then, %entry
// CHECK:STDOUT:   ret void, !dbg !316
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CE.Main.c5e4ba542a56b71f(double %x, i32 %count) #0 !dbg !318 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call double @_CG.Main.c5e4ba542a56b71f(double %x, i32 %count), !dbg !327
// CHECK:STDOUT:   ret double %G.call, !dbg !328
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CF.Main.c5e4ba542a56b71f(double %x, i32 %count) #0 !dbg !330 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call double @_CG.Main.c5e4ba542a56b71f(double %x, i32 %count), !dbg !339
// CHECK:STDOUT:   ret double %G.call, !dbg !340
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CC.Main.a76b78461416b4a8(ptr %x, i32 %count) #0 !dbg !342 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.call = icmp sle i32 %count, 2, !dbg !347
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.LessOrEquivalent.call, label %if.then, label %if.else, !dbg !348
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 %count), !dbg !350
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !352
// CHECK:STDOUT:   call void @_CB.Main.a76b78461416b4a8(ptr %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !353
// CHECK:STDOUT:   br label %if.else, !dbg !354
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %if.then, %entry
// CHECK:STDOUT:   ret void, !dbg !355
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CE.Main.a76b78461416b4a8(ptr %x, i32 %count) #0 !dbg !357 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call ptr @_CG.Main.a76b78461416b4a8(ptr %x, i32 %count), !dbg !366
// CHECK:STDOUT:   ret ptr %G.call, !dbg !367
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CF.Main.a76b78461416b4a8(ptr %x, i32 %count) #0 !dbg !369 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call ptr @_CG.Main.a76b78461416b4a8(ptr %x, i32 %count), !dbg !378
// CHECK:STDOUT:   ret ptr %G.call, !dbg !379
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @printf(ptr, ...)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CG.Main.4d6ceefeb7f772f4(i32 %x, i32 %count) #0 !dbg !420 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !428
// CHECK:STDOUT:   %D.call = call i32 @_CD.Main.4d6ceefeb7f772f4(i32 %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !429
// CHECK:STDOUT:   ret i32 %D.call, !dbg !430
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CG.Main.c5e4ba542a56b71f(double %x, i32 %count) #0 !dbg !432 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !440
// CHECK:STDOUT:   %D.call = call double @_CD.Main.c5e4ba542a56b71f(double %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !441
// CHECK:STDOUT:   ret double %D.call, !dbg !442
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CG.Main.a76b78461416b4a8(ptr %x, i32 %count) #0 !dbg !444 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !452
// CHECK:STDOUT:   %D.call = call ptr @_CD.Main.a76b78461416b4a8(ptr %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !453
// CHECK:STDOUT:   ret ptr %D.call, !dbg !454
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
// CHECK:STDOUT: !52 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.dc14d4dd21801791:core.Destroy.Core", scope: null, file: !4, line: 76, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !55)
// CHECK:STDOUT: !53 = !DILocalVariable(arg: 1, scope: !52, type: !9)
// CHECK:STDOUT: !54 = !DILocation(line: 76, column: 3, scope: !52)
// CHECK:STDOUT: !55 = !{!53}
// CHECK:STDOUT: !56 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.dc14d4dd21801791:core.Destroy.Core", scope: null, file: !4, line: 76, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !59)
// CHECK:STDOUT: !57 = !DILocalVariable(arg: 1, scope: !56, type: !9)
// CHECK:STDOUT: !58 = !DILocation(line: 76, column: 3, scope: !56)
// CHECK:STDOUT: !59 = !{!57}
// CHECK:STDOUT: !60 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.6dae29ece9a708c4:core.Destroy.Core", scope: null, file: !4, line: 75, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !63)
// CHECK:STDOUT: !61 = !DILocalVariable(arg: 1, scope: !60, type: !6)
// CHECK:STDOUT: !62 = !DILocation(line: 75, column: 3, scope: !60)
// CHECK:STDOUT: !63 = !{!61}
// CHECK:STDOUT: !64 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core", scope: null, file: !4, line: 75, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !67)
// CHECK:STDOUT: !65 = !DILocalVariable(arg: 1, scope: !64, type: !6)
// CHECK:STDOUT: !66 = !DILocation(line: 75, column: 3, scope: !64)
// CHECK:STDOUT: !67 = !{!65}
// CHECK:STDOUT: !68 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f58524831c37807e:core.Destroy.Core", scope: null, file: !4, line: 75, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !71)
// CHECK:STDOUT: !69 = !DILocalVariable(arg: 1, scope: !68, type: !6)
// CHECK:STDOUT: !70 = !DILocation(line: 75, column: 3, scope: !68)
// CHECK:STDOUT: !71 = !{!69}
// CHECK:STDOUT: !72 = distinct !DISubprogram(name: "A", linkageName: "_CA.Main.4d6ceefeb7f772f4", scope: null, file: !4, line: 32, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !88)
// CHECK:STDOUT: !73 = !DILocalVariable(arg: 1, scope: !72, type: !6)
// CHECK:STDOUT: !74 = !DILocalVariable(arg: 2, scope: !72, type: !6)
// CHECK:STDOUT: !81 = !DILocation(line: 33, column: 3, scope: !72)
// CHECK:STDOUT: !82 = !{null, !6, !6}
// CHECK:STDOUT: !83 = !DISubroutineType(types: !82)
// CHECK:STDOUT: !86 = !DILocation(line: 34, column: 10, scope: !72)
// CHECK:STDOUT: !87 = !DILocation(line: 34, column: 3, scope: !72)
// CHECK:STDOUT: !88 = !{!73, !74}
// CHECK:STDOUT: !89 = distinct !DISubprogram(name: "A", linkageName: "_CA.Main.c5e4ba542a56b71f", scope: null, file: !4, line: 32, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !105)
// CHECK:STDOUT: !90 = !DILocalVariable(arg: 1, scope: !89, type: !9)
// CHECK:STDOUT: !91 = !DILocalVariable(arg: 2, scope: !89, type: !6)
// CHECK:STDOUT: !98 = !DILocation(line: 33, column: 3, scope: !89)
// CHECK:STDOUT: !99 = !{null, !9, !6}
// CHECK:STDOUT: !100 = !DISubroutineType(types: !99)
// CHECK:STDOUT: !103 = !DILocation(line: 34, column: 10, scope: !89)
// CHECK:STDOUT: !104 = !DILocation(line: 34, column: 3, scope: !89)
// CHECK:STDOUT: !105 = !{!90, !91}
// CHECK:STDOUT: !106 = distinct !DISubprogram(name: "A", linkageName: "_CA.Main.a76b78461416b4a8", scope: null, file: !4, line: 32, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !120)
// CHECK:STDOUT: !107 = !DILocalVariable(arg: 1, scope: !106, type: !9)
// CHECK:STDOUT: !108 = !DILocalVariable(arg: 2, scope: !106, type: !6)
// CHECK:STDOUT: !115 = !DILocation(line: 33, column: 3, scope: !106)
// CHECK:STDOUT: !118 = !DILocation(line: 34, column: 10, scope: !106)
// CHECK:STDOUT: !119 = !DILocation(line: 34, column: 3, scope: !106)
// CHECK:STDOUT: !120 = !{!107, !108}
// CHECK:STDOUT: !136 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.4d6ceefeb7f772f4", scope: null, file: !4, line: 37, type: !83, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !145)
// CHECK:STDOUT: !137 = !DILocalVariable(arg: 1, scope: !136, type: !6)
// CHECK:STDOUT: !138 = !DILocalVariable(arg: 2, scope: !136, type: !6)
// CHECK:STDOUT: !143 = !DILocation(line: 38, column: 3, scope: !136)
// CHECK:STDOUT: !144 = !DILocation(line: 37, column: 1, scope: !136)
// CHECK:STDOUT: !145 = !{!137, !138}
// CHECK:STDOUT: !146 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.4d6ceefeb7f772f4", scope: null, file: !4, line: 48, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !167)
// CHECK:STDOUT: !147 = !DILocalVariable(arg: 1, scope: !146, type: !6)
// CHECK:STDOUT: !148 = !DILocalVariable(arg: 2, scope: !146, type: !6)
// CHECK:STDOUT: !153 = !DILocation(line: 49, column: 7, scope: !146)
// CHECK:STDOUT: !154 = !DILocation(line: 49, column: 6, scope: !146)
// CHECK:STDOUT: !156 = !DILocation(line: 50, column: 5, scope: !146)
// CHECK:STDOUT: !157 = !DILocation(line: 52, column: 7, scope: !146)
// CHECK:STDOUT: !158 = !DILocation(line: 52, column: 6, scope: !146)
// CHECK:STDOUT: !161 = !DILocation(line: 53, column: 12, scope: !146)
// CHECK:STDOUT: !162 = !DILocation(line: 53, column: 5, scope: !146)
// CHECK:STDOUT: !165 = !DILocation(line: 55, column: 12, scope: !146)
// CHECK:STDOUT: !166 = !DILocation(line: 55, column: 5, scope: !146)
// CHECK:STDOUT: !167 = !{!147, !148}
// CHECK:STDOUT: !168 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.c5e4ba542a56b71f", scope: null, file: !4, line: 37, type: !100, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !177)
// CHECK:STDOUT: !169 = !DILocalVariable(arg: 1, scope: !168, type: !9)
// CHECK:STDOUT: !170 = !DILocalVariable(arg: 2, scope: !168, type: !6)
// CHECK:STDOUT: !175 = !DILocation(line: 38, column: 3, scope: !168)
// CHECK:STDOUT: !176 = !DILocation(line: 37, column: 1, scope: !168)
// CHECK:STDOUT: !177 = !{!169, !170}
// CHECK:STDOUT: !178 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.c5e4ba542a56b71f", scope: null, file: !4, line: 48, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !199)
// CHECK:STDOUT: !179 = !DILocalVariable(arg: 1, scope: !178, type: !9)
// CHECK:STDOUT: !180 = !DILocalVariable(arg: 2, scope: !178, type: !6)
// CHECK:STDOUT: !185 = !DILocation(line: 49, column: 7, scope: !178)
// CHECK:STDOUT: !186 = !DILocation(line: 49, column: 6, scope: !178)
// CHECK:STDOUT: !188 = !DILocation(line: 50, column: 5, scope: !178)
// CHECK:STDOUT: !189 = !DILocation(line: 52, column: 7, scope: !178)
// CHECK:STDOUT: !190 = !DILocation(line: 52, column: 6, scope: !178)
// CHECK:STDOUT: !193 = !DILocation(line: 53, column: 12, scope: !178)
// CHECK:STDOUT: !194 = !DILocation(line: 53, column: 5, scope: !178)
// CHECK:STDOUT: !197 = !DILocation(line: 55, column: 12, scope: !178)
// CHECK:STDOUT: !198 = !DILocation(line: 55, column: 5, scope: !178)
// CHECK:STDOUT: !199 = !{!179, !180}
// CHECK:STDOUT: !200 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.a76b78461416b4a8", scope: null, file: !4, line: 37, type: !100, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !209)
// CHECK:STDOUT: !201 = !DILocalVariable(arg: 1, scope: !200, type: !9)
// CHECK:STDOUT: !202 = !DILocalVariable(arg: 2, scope: !200, type: !6)
// CHECK:STDOUT: !207 = !DILocation(line: 38, column: 3, scope: !200)
// CHECK:STDOUT: !208 = !DILocation(line: 37, column: 1, scope: !200)
// CHECK:STDOUT: !209 = !{!201, !202}
// CHECK:STDOUT: !210 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.a76b78461416b4a8", scope: null, file: !4, line: 48, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !231)
// CHECK:STDOUT: !211 = !DILocalVariable(arg: 1, scope: !210, type: !9)
// CHECK:STDOUT: !212 = !DILocalVariable(arg: 2, scope: !210, type: !6)
// CHECK:STDOUT: !217 = !DILocation(line: 49, column: 7, scope: !210)
// CHECK:STDOUT: !218 = !DILocation(line: 49, column: 6, scope: !210)
// CHECK:STDOUT: !220 = !DILocation(line: 50, column: 5, scope: !210)
// CHECK:STDOUT: !221 = !DILocation(line: 52, column: 7, scope: !210)
// CHECK:STDOUT: !222 = !DILocation(line: 52, column: 6, scope: !210)
// CHECK:STDOUT: !225 = !DILocation(line: 53, column: 12, scope: !210)
// CHECK:STDOUT: !226 = !DILocation(line: 53, column: 5, scope: !210)
// CHECK:STDOUT: !229 = !DILocation(line: 55, column: 12, scope: !210)
// CHECK:STDOUT: !230 = !DILocation(line: 55, column: 5, scope: !210)
// CHECK:STDOUT: !231 = !{!211, !212}
// CHECK:STDOUT: !264 = distinct !DISubprogram(name: "C", linkageName: "_CC.Main.4d6ceefeb7f772f4", scope: null, file: !4, line: 41, type: !83, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !278)
// CHECK:STDOUT: !265 = !DILocalVariable(arg: 1, scope: !264, type: !6)
// CHECK:STDOUT: !266 = !DILocalVariable(arg: 2, scope: !264, type: !6)
// CHECK:STDOUT: !269 = !DILocation(line: 42, column: 7, scope: !264)
// CHECK:STDOUT: !270 = !DILocation(line: 42, column: 6, scope: !264)
// CHECK:STDOUT: !272 = !DILocation(line: 43, column: 5, scope: !264)
// CHECK:STDOUT: !274 = !DILocation(line: 44, column: 10, scope: !264)
// CHECK:STDOUT: !275 = !DILocation(line: 44, column: 5, scope: !264)
// CHECK:STDOUT: !276 = !DILocation(line: 42, column: 3, scope: !264)
// CHECK:STDOUT: !277 = !DILocation(line: 41, column: 1, scope: !264)
// CHECK:STDOUT: !278 = !{!265, !266}
// CHECK:STDOUT: !279 = distinct !DISubprogram(name: "E", linkageName: "_CE.Main.4d6ceefeb7f772f4", scope: null, file: !4, line: 61, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !290)
// CHECK:STDOUT: !280 = !DILocalVariable(arg: 1, scope: !279, type: !6)
// CHECK:STDOUT: !281 = !DILocalVariable(arg: 2, scope: !279, type: !6)
// CHECK:STDOUT: !288 = !DILocation(line: 62, column: 10, scope: !279)
// CHECK:STDOUT: !289 = !DILocation(line: 62, column: 3, scope: !279)
// CHECK:STDOUT: !290 = !{!280, !281}
// CHECK:STDOUT: !291 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.4d6ceefeb7f772f4", scope: null, file: !4, line: 65, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !302)
// CHECK:STDOUT: !292 = !DILocalVariable(arg: 1, scope: !291, type: !6)
// CHECK:STDOUT: !293 = !DILocalVariable(arg: 2, scope: !291, type: !6)
// CHECK:STDOUT: !300 = !DILocation(line: 66, column: 10, scope: !291)
// CHECK:STDOUT: !301 = !DILocation(line: 66, column: 3, scope: !291)
// CHECK:STDOUT: !302 = !{!292, !293}
// CHECK:STDOUT: !303 = distinct !DISubprogram(name: "C", linkageName: "_CC.Main.c5e4ba542a56b71f", scope: null, file: !4, line: 41, type: !100, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !317)
// CHECK:STDOUT: !304 = !DILocalVariable(arg: 1, scope: !303, type: !9)
// CHECK:STDOUT: !305 = !DILocalVariable(arg: 2, scope: !303, type: !6)
// CHECK:STDOUT: !308 = !DILocation(line: 42, column: 7, scope: !303)
// CHECK:STDOUT: !309 = !DILocation(line: 42, column: 6, scope: !303)
// CHECK:STDOUT: !311 = !DILocation(line: 43, column: 5, scope: !303)
// CHECK:STDOUT: !313 = !DILocation(line: 44, column: 10, scope: !303)
// CHECK:STDOUT: !314 = !DILocation(line: 44, column: 5, scope: !303)
// CHECK:STDOUT: !315 = !DILocation(line: 42, column: 3, scope: !303)
// CHECK:STDOUT: !316 = !DILocation(line: 41, column: 1, scope: !303)
// CHECK:STDOUT: !317 = !{!304, !305}
// CHECK:STDOUT: !318 = distinct !DISubprogram(name: "E", linkageName: "_CE.Main.c5e4ba542a56b71f", scope: null, file: !4, line: 61, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !329)
// CHECK:STDOUT: !319 = !DILocalVariable(arg: 1, scope: !318, type: !9)
// CHECK:STDOUT: !320 = !DILocalVariable(arg: 2, scope: !318, type: !6)
// CHECK:STDOUT: !327 = !DILocation(line: 62, column: 10, scope: !318)
// CHECK:STDOUT: !328 = !DILocation(line: 62, column: 3, scope: !318)
// CHECK:STDOUT: !329 = !{!319, !320}
// CHECK:STDOUT: !330 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.c5e4ba542a56b71f", scope: null, file: !4, line: 65, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !341)
// CHECK:STDOUT: !331 = !DILocalVariable(arg: 1, scope: !330, type: !9)
// CHECK:STDOUT: !332 = !DILocalVariable(arg: 2, scope: !330, type: !6)
// CHECK:STDOUT: !339 = !DILocation(line: 66, column: 10, scope: !330)
// CHECK:STDOUT: !340 = !DILocation(line: 66, column: 3, scope: !330)
// CHECK:STDOUT: !341 = !{!331, !332}
// CHECK:STDOUT: !342 = distinct !DISubprogram(name: "C", linkageName: "_CC.Main.a76b78461416b4a8", scope: null, file: !4, line: 41, type: !100, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !356)
// CHECK:STDOUT: !343 = !DILocalVariable(arg: 1, scope: !342, type: !9)
// CHECK:STDOUT: !344 = !DILocalVariable(arg: 2, scope: !342, type: !6)
// CHECK:STDOUT: !347 = !DILocation(line: 42, column: 7, scope: !342)
// CHECK:STDOUT: !348 = !DILocation(line: 42, column: 6, scope: !342)
// CHECK:STDOUT: !350 = !DILocation(line: 43, column: 5, scope: !342)
// CHECK:STDOUT: !352 = !DILocation(line: 44, column: 10, scope: !342)
// CHECK:STDOUT: !353 = !DILocation(line: 44, column: 5, scope: !342)
// CHECK:STDOUT: !354 = !DILocation(line: 42, column: 3, scope: !342)
// CHECK:STDOUT: !355 = !DILocation(line: 41, column: 1, scope: !342)
// CHECK:STDOUT: !356 = !{!343, !344}
// CHECK:STDOUT: !357 = distinct !DISubprogram(name: "E", linkageName: "_CE.Main.a76b78461416b4a8", scope: null, file: !4, line: 61, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !368)
// CHECK:STDOUT: !358 = !DILocalVariable(arg: 1, scope: !357, type: !9)
// CHECK:STDOUT: !359 = !DILocalVariable(arg: 2, scope: !357, type: !6)
// CHECK:STDOUT: !366 = !DILocation(line: 62, column: 10, scope: !357)
// CHECK:STDOUT: !367 = !DILocation(line: 62, column: 3, scope: !357)
// CHECK:STDOUT: !368 = !{!358, !359}
// CHECK:STDOUT: !369 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.a76b78461416b4a8", scope: null, file: !4, line: 65, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !380)
// CHECK:STDOUT: !370 = !DILocalVariable(arg: 1, scope: !369, type: !9)
// CHECK:STDOUT: !371 = !DILocalVariable(arg: 2, scope: !369, type: !6)
// CHECK:STDOUT: !378 = !DILocation(line: 66, column: 10, scope: !369)
// CHECK:STDOUT: !379 = !DILocation(line: 66, column: 3, scope: !369)
// CHECK:STDOUT: !380 = !{!370, !371}
// CHECK:STDOUT: !420 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.4d6ceefeb7f772f4", scope: null, file: !4, line: 69, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !431)
// CHECK:STDOUT: !421 = !DILocalVariable(arg: 1, scope: !420, type: !6)
// CHECK:STDOUT: !422 = !DILocalVariable(arg: 2, scope: !420, type: !6)
// CHECK:STDOUT: !428 = !DILocation(line: 70, column: 15, scope: !420)
// CHECK:STDOUT: !429 = !DILocation(line: 70, column: 10, scope: !420)
// CHECK:STDOUT: !430 = !DILocation(line: 70, column: 3, scope: !420)
// CHECK:STDOUT: !431 = !{!421, !422}
// CHECK:STDOUT: !432 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.c5e4ba542a56b71f", scope: null, file: !4, line: 69, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !443)
// CHECK:STDOUT: !433 = !DILocalVariable(arg: 1, scope: !432, type: !9)
// CHECK:STDOUT: !434 = !DILocalVariable(arg: 2, scope: !432, type: !6)
// CHECK:STDOUT: !440 = !DILocation(line: 70, column: 15, scope: !432)
// CHECK:STDOUT: !441 = !DILocation(line: 70, column: 10, scope: !432)
// CHECK:STDOUT: !442 = !DILocation(line: 70, column: 3, scope: !432)
// CHECK:STDOUT: !443 = !{!433, !434}
// CHECK:STDOUT: !444 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.a76b78461416b4a8", scope: null, file: !4, line: 69, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !455)
// CHECK:STDOUT: !445 = !DILocalVariable(arg: 1, scope: !444, type: !9)
// CHECK:STDOUT: !446 = !DILocalVariable(arg: 2, scope: !444, type: !6)
// CHECK:STDOUT: !452 = !DILocation(line: 70, column: 15, scope: !444)
// CHECK:STDOUT: !453 = !DILocation(line: 70, column: 10, scope: !444)
// CHECK:STDOUT: !454 = !DILocation(line: 70, column: 3, scope: !444)
// CHECK:STDOUT: !455 = !{!445, !446}
// CHECK:STDOUT:
