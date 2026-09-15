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
// CHECK:STDOUT:   call void @"_COp.dc14d4dd21801791:core.Destroy.Core"(ptr %m.var), !dbg !17
// CHECK:STDOUT:   call void @"_COp.f58524831c37807e:core.Destroy.Core"(ptr %n.var), !dbg !15
// CHECK:STDOUT:   ret void, !dbg !39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.dc14d4dd21801791:core.Destroy.Core"(ptr %self) #0 !dbg !40 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !42
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !44 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !46
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CA.Main.4d6ceefeb7f772f4(i32 %x, i32 %count) #0 !dbg !48 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CB.Main.4d6ceefeb7f772f4(i32 %x, i32 %count), !dbg !57
// CHECK:STDOUT:   %D.call = call i32 @_CD.Main.4d6ceefeb7f772f4(i32 %x, i32 %count), !dbg !62
// CHECK:STDOUT:   ret i32 %D.call, !dbg !63
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CA.Main.c5e4ba542a56b71f(double %x, i32 %count) #0 !dbg !65 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CB.Main.c5e4ba542a56b71f(double %x, i32 %count), !dbg !74
// CHECK:STDOUT:   %D.call = call double @_CD.Main.c5e4ba542a56b71f(double %x, i32 %count), !dbg !79
// CHECK:STDOUT:   ret double %D.call, !dbg !80
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CA.Main.a76b78461416b4a8(ptr %x, i32 %count) #0 !dbg !82 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CB.Main.a76b78461416b4a8(ptr %x, i32 %count), !dbg !91
// CHECK:STDOUT:   %D.call = call ptr @_CD.Main.a76b78461416b4a8(ptr %x, i32 %count), !dbg !94
// CHECK:STDOUT:   ret ptr %D.call, !dbg !95
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CB.Main.4d6ceefeb7f772f4(i32 %x, i32 %count) #0 !dbg !112 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CC.Main.4d6ceefeb7f772f4(i32 %x, i32 %count), !dbg !119
// CHECK:STDOUT:   ret void, !dbg !120
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CD.Main.4d6ceefeb7f772f4(i32 %x, i32 %count) #0 !dbg !122 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 4, !dbg !129
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then.loc49, label %if.else.loc49, !dbg !130
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc49:                                    ; preds = %entry
// CHECK:STDOUT:   ret i32 %x, !dbg !132
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc49:                                    ; preds = %entry
// CHECK:STDOUT:   %Int.as.ModWith.impl.Op.call = srem i32 %count, 2, !dbg !133
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.call = icmp eq i32 %Int.as.ModWith.impl.Op.call, 0, !dbg !133
// CHECK:STDOUT:   br i1 %Int.as.EqWith.impl.Equal.call, label %if.then.loc52, label %if.else.loc52, !dbg !134
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc52:                                    ; preds = %if.else.loc49
// CHECK:STDOUT:   %E.call = call i32 @_CE.Main.4d6ceefeb7f772f4(i32 %x, i32 %count), !dbg !137
// CHECK:STDOUT:   ret i32 %E.call, !dbg !138
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc52:                                    ; preds = %if.else.loc49
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main.4d6ceefeb7f772f4(i32 %x, i32 %count), !dbg !141
// CHECK:STDOUT:   ret i32 %F.call, !dbg !142
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CB.Main.c5e4ba542a56b71f(double %x, i32 %count) #0 !dbg !144 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CC.Main.c5e4ba542a56b71f(double %x, i32 %count), !dbg !151
// CHECK:STDOUT:   ret void, !dbg !152
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CD.Main.c5e4ba542a56b71f(double %x, i32 %count) #0 !dbg !154 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 4, !dbg !161
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then.loc49, label %if.else.loc49, !dbg !162
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc49:                                    ; preds = %entry
// CHECK:STDOUT:   ret double %x, !dbg !164
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc49:                                    ; preds = %entry
// CHECK:STDOUT:   %Int.as.ModWith.impl.Op.call = srem i32 %count, 2, !dbg !165
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.call = icmp eq i32 %Int.as.ModWith.impl.Op.call, 0, !dbg !165
// CHECK:STDOUT:   br i1 %Int.as.EqWith.impl.Equal.call, label %if.then.loc52, label %if.else.loc52, !dbg !166
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc52:                                    ; preds = %if.else.loc49
// CHECK:STDOUT:   %E.call = call double @_CE.Main.c5e4ba542a56b71f(double %x, i32 %count), !dbg !169
// CHECK:STDOUT:   ret double %E.call, !dbg !170
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc52:                                    ; preds = %if.else.loc49
// CHECK:STDOUT:   %F.call = call double @_CF.Main.c5e4ba542a56b71f(double %x, i32 %count), !dbg !173
// CHECK:STDOUT:   ret double %F.call, !dbg !174
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CB.Main.a76b78461416b4a8(ptr %x, i32 %count) #0 !dbg !176 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CC.Main.a76b78461416b4a8(ptr %x, i32 %count), !dbg !183
// CHECK:STDOUT:   ret void, !dbg !184
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CD.Main.a76b78461416b4a8(ptr %x, i32 %count) #0 !dbg !186 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 4, !dbg !193
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then.loc49, label %if.else.loc49, !dbg !194
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc49:                                    ; preds = %entry
// CHECK:STDOUT:   ret ptr %x, !dbg !196
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc49:                                    ; preds = %entry
// CHECK:STDOUT:   %Int.as.ModWith.impl.Op.call = srem i32 %count, 2, !dbg !197
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.call = icmp eq i32 %Int.as.ModWith.impl.Op.call, 0, !dbg !197
// CHECK:STDOUT:   br i1 %Int.as.EqWith.impl.Equal.call, label %if.then.loc52, label %if.else.loc52, !dbg !198
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc52:                                    ; preds = %if.else.loc49
// CHECK:STDOUT:   %E.call = call ptr @_CE.Main.a76b78461416b4a8(ptr %x, i32 %count), !dbg !201
// CHECK:STDOUT:   ret ptr %E.call, !dbg !202
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc52:                                    ; preds = %if.else.loc49
// CHECK:STDOUT:   %F.call = call ptr @_CF.Main.a76b78461416b4a8(ptr %x, i32 %count), !dbg !205
// CHECK:STDOUT:   ret ptr %F.call, !dbg !206
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CC.Main.4d6ceefeb7f772f4(i32 %x, i32 %count) #0 !dbg !240 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.call = icmp sle i32 %count, 2, !dbg !245
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.LessOrEquivalent.call, label %if.then, label %if.else, !dbg !246
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 %count), !dbg !248
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !250
// CHECK:STDOUT:   call void @_CB.Main.4d6ceefeb7f772f4(i32 %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !251
// CHECK:STDOUT:   br label %if.else, !dbg !252
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %if.then, %entry
// CHECK:STDOUT:   ret void, !dbg !253
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CE.Main.4d6ceefeb7f772f4(i32 %x, i32 %count) #0 !dbg !255 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call i32 @_CG.Main.4d6ceefeb7f772f4(i32 %x, i32 %count), !dbg !264
// CHECK:STDOUT:   ret i32 %G.call, !dbg !265
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.4d6ceefeb7f772f4(i32 %x, i32 %count) #0 !dbg !267 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call i32 @_CG.Main.4d6ceefeb7f772f4(i32 %x, i32 %count), !dbg !276
// CHECK:STDOUT:   ret i32 %G.call, !dbg !277
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CC.Main.c5e4ba542a56b71f(double %x, i32 %count) #0 !dbg !279 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.call = icmp sle i32 %count, 2, !dbg !284
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.LessOrEquivalent.call, label %if.then, label %if.else, !dbg !285
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 %count), !dbg !287
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !289
// CHECK:STDOUT:   call void @_CB.Main.c5e4ba542a56b71f(double %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !290
// CHECK:STDOUT:   br label %if.else, !dbg !291
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %if.then, %entry
// CHECK:STDOUT:   ret void, !dbg !292
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CE.Main.c5e4ba542a56b71f(double %x, i32 %count) #0 !dbg !294 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call double @_CG.Main.c5e4ba542a56b71f(double %x, i32 %count), !dbg !303
// CHECK:STDOUT:   ret double %G.call, !dbg !304
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CF.Main.c5e4ba542a56b71f(double %x, i32 %count) #0 !dbg !306 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call double @_CG.Main.c5e4ba542a56b71f(double %x, i32 %count), !dbg !315
// CHECK:STDOUT:   ret double %G.call, !dbg !316
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CC.Main.a76b78461416b4a8(ptr %x, i32 %count) #0 !dbg !318 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.call = icmp sle i32 %count, 2, !dbg !323
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.LessOrEquivalent.call, label %if.then, label %if.else, !dbg !324
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 %count), !dbg !326
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !328
// CHECK:STDOUT:   call void @_CB.Main.a76b78461416b4a8(ptr %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !329
// CHECK:STDOUT:   br label %if.else, !dbg !330
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %if.then, %entry
// CHECK:STDOUT:   ret void, !dbg !331
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CE.Main.a76b78461416b4a8(ptr %x, i32 %count) #0 !dbg !333 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call ptr @_CG.Main.a76b78461416b4a8(ptr %x, i32 %count), !dbg !342
// CHECK:STDOUT:   ret ptr %G.call, !dbg !343
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CF.Main.a76b78461416b4a8(ptr %x, i32 %count) #0 !dbg !345 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call ptr @_CG.Main.a76b78461416b4a8(ptr %x, i32 %count), !dbg !354
// CHECK:STDOUT:   ret ptr %G.call, !dbg !355
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @printf(ptr, ...)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CG.Main.4d6ceefeb7f772f4(i32 %x, i32 %count) #0 !dbg !396 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !404
// CHECK:STDOUT:   %D.call = call i32 @_CD.Main.4d6ceefeb7f772f4(i32 %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !405
// CHECK:STDOUT:   ret i32 %D.call, !dbg !406
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CG.Main.c5e4ba542a56b71f(double %x, i32 %count) #0 !dbg !408 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !416
// CHECK:STDOUT:   %D.call = call double @_CD.Main.c5e4ba542a56b71f(double %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !417
// CHECK:STDOUT:   ret double %D.call, !dbg !418
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CG.Main.a76b78461416b4a8(ptr %x, i32 %count) #0 !dbg !420 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !428
// CHECK:STDOUT:   %D.call = call ptr @_CD.Main.a76b78461416b4a8(ptr %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !429
// CHECK:STDOUT:   ret ptr %D.call, !dbg !430
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
// CHECK:STDOUT: !40 = distinct !DISubprogram(name: "Op", linkageName: "_COp.dc14d4dd21801791:core.Destroy.Core", scope: null, file: !4, line: 76, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !43)
// CHECK:STDOUT: !41 = !DILocalVariable(arg: 1, scope: !40, type: !9)
// CHECK:STDOUT: !42 = !DILocation(line: 76, column: 3, scope: !40)
// CHECK:STDOUT: !43 = !{!41}
// CHECK:STDOUT: !44 = distinct !DISubprogram(name: "Op", linkageName: "_COp.f58524831c37807e:core.Destroy.Core", scope: null, file: !4, line: 75, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !47)
// CHECK:STDOUT: !45 = !DILocalVariable(arg: 1, scope: !44, type: !6)
// CHECK:STDOUT: !46 = !DILocation(line: 75, column: 3, scope: !44)
// CHECK:STDOUT: !47 = !{!45}
// CHECK:STDOUT: !48 = distinct !DISubprogram(name: "A", linkageName: "_CA.Main.4d6ceefeb7f772f4", scope: null, file: !4, line: 32, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !64)
// CHECK:STDOUT: !49 = !DILocalVariable(arg: 1, scope: !48, type: !6)
// CHECK:STDOUT: !50 = !DILocalVariable(arg: 2, scope: !48, type: !6)
// CHECK:STDOUT: !57 = !DILocation(line: 33, column: 3, scope: !48)
// CHECK:STDOUT: !58 = !{null, !6, !6}
// CHECK:STDOUT: !59 = !DISubroutineType(types: !58)
// CHECK:STDOUT: !62 = !DILocation(line: 34, column: 10, scope: !48)
// CHECK:STDOUT: !63 = !DILocation(line: 34, column: 3, scope: !48)
// CHECK:STDOUT: !64 = !{!49, !50}
// CHECK:STDOUT: !65 = distinct !DISubprogram(name: "A", linkageName: "_CA.Main.c5e4ba542a56b71f", scope: null, file: !4, line: 32, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !81)
// CHECK:STDOUT: !66 = !DILocalVariable(arg: 1, scope: !65, type: !9)
// CHECK:STDOUT: !67 = !DILocalVariable(arg: 2, scope: !65, type: !6)
// CHECK:STDOUT: !74 = !DILocation(line: 33, column: 3, scope: !65)
// CHECK:STDOUT: !75 = !{null, !9, !6}
// CHECK:STDOUT: !76 = !DISubroutineType(types: !75)
// CHECK:STDOUT: !79 = !DILocation(line: 34, column: 10, scope: !65)
// CHECK:STDOUT: !80 = !DILocation(line: 34, column: 3, scope: !65)
// CHECK:STDOUT: !81 = !{!66, !67}
// CHECK:STDOUT: !82 = distinct !DISubprogram(name: "A", linkageName: "_CA.Main.a76b78461416b4a8", scope: null, file: !4, line: 32, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !96)
// CHECK:STDOUT: !83 = !DILocalVariable(arg: 1, scope: !82, type: !9)
// CHECK:STDOUT: !84 = !DILocalVariable(arg: 2, scope: !82, type: !6)
// CHECK:STDOUT: !91 = !DILocation(line: 33, column: 3, scope: !82)
// CHECK:STDOUT: !94 = !DILocation(line: 34, column: 10, scope: !82)
// CHECK:STDOUT: !95 = !DILocation(line: 34, column: 3, scope: !82)
// CHECK:STDOUT: !96 = !{!83, !84}
// CHECK:STDOUT: !112 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.4d6ceefeb7f772f4", scope: null, file: !4, line: 37, type: !59, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !121)
// CHECK:STDOUT: !113 = !DILocalVariable(arg: 1, scope: !112, type: !6)
// CHECK:STDOUT: !114 = !DILocalVariable(arg: 2, scope: !112, type: !6)
// CHECK:STDOUT: !119 = !DILocation(line: 38, column: 3, scope: !112)
// CHECK:STDOUT: !120 = !DILocation(line: 37, column: 1, scope: !112)
// CHECK:STDOUT: !121 = !{!113, !114}
// CHECK:STDOUT: !122 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.4d6ceefeb7f772f4", scope: null, file: !4, line: 48, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !143)
// CHECK:STDOUT: !123 = !DILocalVariable(arg: 1, scope: !122, type: !6)
// CHECK:STDOUT: !124 = !DILocalVariable(arg: 2, scope: !122, type: !6)
// CHECK:STDOUT: !129 = !DILocation(line: 49, column: 7, scope: !122)
// CHECK:STDOUT: !130 = !DILocation(line: 49, column: 6, scope: !122)
// CHECK:STDOUT: !132 = !DILocation(line: 50, column: 5, scope: !122)
// CHECK:STDOUT: !133 = !DILocation(line: 52, column: 7, scope: !122)
// CHECK:STDOUT: !134 = !DILocation(line: 52, column: 6, scope: !122)
// CHECK:STDOUT: !137 = !DILocation(line: 53, column: 12, scope: !122)
// CHECK:STDOUT: !138 = !DILocation(line: 53, column: 5, scope: !122)
// CHECK:STDOUT: !141 = !DILocation(line: 55, column: 12, scope: !122)
// CHECK:STDOUT: !142 = !DILocation(line: 55, column: 5, scope: !122)
// CHECK:STDOUT: !143 = !{!123, !124}
// CHECK:STDOUT: !144 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.c5e4ba542a56b71f", scope: null, file: !4, line: 37, type: !76, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !153)
// CHECK:STDOUT: !145 = !DILocalVariable(arg: 1, scope: !144, type: !9)
// CHECK:STDOUT: !146 = !DILocalVariable(arg: 2, scope: !144, type: !6)
// CHECK:STDOUT: !151 = !DILocation(line: 38, column: 3, scope: !144)
// CHECK:STDOUT: !152 = !DILocation(line: 37, column: 1, scope: !144)
// CHECK:STDOUT: !153 = !{!145, !146}
// CHECK:STDOUT: !154 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.c5e4ba542a56b71f", scope: null, file: !4, line: 48, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !175)
// CHECK:STDOUT: !155 = !DILocalVariable(arg: 1, scope: !154, type: !9)
// CHECK:STDOUT: !156 = !DILocalVariable(arg: 2, scope: !154, type: !6)
// CHECK:STDOUT: !161 = !DILocation(line: 49, column: 7, scope: !154)
// CHECK:STDOUT: !162 = !DILocation(line: 49, column: 6, scope: !154)
// CHECK:STDOUT: !164 = !DILocation(line: 50, column: 5, scope: !154)
// CHECK:STDOUT: !165 = !DILocation(line: 52, column: 7, scope: !154)
// CHECK:STDOUT: !166 = !DILocation(line: 52, column: 6, scope: !154)
// CHECK:STDOUT: !169 = !DILocation(line: 53, column: 12, scope: !154)
// CHECK:STDOUT: !170 = !DILocation(line: 53, column: 5, scope: !154)
// CHECK:STDOUT: !173 = !DILocation(line: 55, column: 12, scope: !154)
// CHECK:STDOUT: !174 = !DILocation(line: 55, column: 5, scope: !154)
// CHECK:STDOUT: !175 = !{!155, !156}
// CHECK:STDOUT: !176 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.a76b78461416b4a8", scope: null, file: !4, line: 37, type: !76, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !185)
// CHECK:STDOUT: !177 = !DILocalVariable(arg: 1, scope: !176, type: !9)
// CHECK:STDOUT: !178 = !DILocalVariable(arg: 2, scope: !176, type: !6)
// CHECK:STDOUT: !183 = !DILocation(line: 38, column: 3, scope: !176)
// CHECK:STDOUT: !184 = !DILocation(line: 37, column: 1, scope: !176)
// CHECK:STDOUT: !185 = !{!177, !178}
// CHECK:STDOUT: !186 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.a76b78461416b4a8", scope: null, file: !4, line: 48, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !207)
// CHECK:STDOUT: !187 = !DILocalVariable(arg: 1, scope: !186, type: !9)
// CHECK:STDOUT: !188 = !DILocalVariable(arg: 2, scope: !186, type: !6)
// CHECK:STDOUT: !193 = !DILocation(line: 49, column: 7, scope: !186)
// CHECK:STDOUT: !194 = !DILocation(line: 49, column: 6, scope: !186)
// CHECK:STDOUT: !196 = !DILocation(line: 50, column: 5, scope: !186)
// CHECK:STDOUT: !197 = !DILocation(line: 52, column: 7, scope: !186)
// CHECK:STDOUT: !198 = !DILocation(line: 52, column: 6, scope: !186)
// CHECK:STDOUT: !201 = !DILocation(line: 53, column: 12, scope: !186)
// CHECK:STDOUT: !202 = !DILocation(line: 53, column: 5, scope: !186)
// CHECK:STDOUT: !205 = !DILocation(line: 55, column: 12, scope: !186)
// CHECK:STDOUT: !206 = !DILocation(line: 55, column: 5, scope: !186)
// CHECK:STDOUT: !207 = !{!187, !188}
// CHECK:STDOUT: !240 = distinct !DISubprogram(name: "C", linkageName: "_CC.Main.4d6ceefeb7f772f4", scope: null, file: !4, line: 41, type: !59, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !254)
// CHECK:STDOUT: !241 = !DILocalVariable(arg: 1, scope: !240, type: !6)
// CHECK:STDOUT: !242 = !DILocalVariable(arg: 2, scope: !240, type: !6)
// CHECK:STDOUT: !245 = !DILocation(line: 42, column: 7, scope: !240)
// CHECK:STDOUT: !246 = !DILocation(line: 42, column: 6, scope: !240)
// CHECK:STDOUT: !248 = !DILocation(line: 43, column: 5, scope: !240)
// CHECK:STDOUT: !250 = !DILocation(line: 44, column: 10, scope: !240)
// CHECK:STDOUT: !251 = !DILocation(line: 44, column: 5, scope: !240)
// CHECK:STDOUT: !252 = !DILocation(line: 42, column: 3, scope: !240)
// CHECK:STDOUT: !253 = !DILocation(line: 41, column: 1, scope: !240)
// CHECK:STDOUT: !254 = !{!241, !242}
// CHECK:STDOUT: !255 = distinct !DISubprogram(name: "E", linkageName: "_CE.Main.4d6ceefeb7f772f4", scope: null, file: !4, line: 61, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !266)
// CHECK:STDOUT: !256 = !DILocalVariable(arg: 1, scope: !255, type: !6)
// CHECK:STDOUT: !257 = !DILocalVariable(arg: 2, scope: !255, type: !6)
// CHECK:STDOUT: !264 = !DILocation(line: 62, column: 10, scope: !255)
// CHECK:STDOUT: !265 = !DILocation(line: 62, column: 3, scope: !255)
// CHECK:STDOUT: !266 = !{!256, !257}
// CHECK:STDOUT: !267 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.4d6ceefeb7f772f4", scope: null, file: !4, line: 65, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !278)
// CHECK:STDOUT: !268 = !DILocalVariable(arg: 1, scope: !267, type: !6)
// CHECK:STDOUT: !269 = !DILocalVariable(arg: 2, scope: !267, type: !6)
// CHECK:STDOUT: !276 = !DILocation(line: 66, column: 10, scope: !267)
// CHECK:STDOUT: !277 = !DILocation(line: 66, column: 3, scope: !267)
// CHECK:STDOUT: !278 = !{!268, !269}
// CHECK:STDOUT: !279 = distinct !DISubprogram(name: "C", linkageName: "_CC.Main.c5e4ba542a56b71f", scope: null, file: !4, line: 41, type: !76, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !293)
// CHECK:STDOUT: !280 = !DILocalVariable(arg: 1, scope: !279, type: !9)
// CHECK:STDOUT: !281 = !DILocalVariable(arg: 2, scope: !279, type: !6)
// CHECK:STDOUT: !284 = !DILocation(line: 42, column: 7, scope: !279)
// CHECK:STDOUT: !285 = !DILocation(line: 42, column: 6, scope: !279)
// CHECK:STDOUT: !287 = !DILocation(line: 43, column: 5, scope: !279)
// CHECK:STDOUT: !289 = !DILocation(line: 44, column: 10, scope: !279)
// CHECK:STDOUT: !290 = !DILocation(line: 44, column: 5, scope: !279)
// CHECK:STDOUT: !291 = !DILocation(line: 42, column: 3, scope: !279)
// CHECK:STDOUT: !292 = !DILocation(line: 41, column: 1, scope: !279)
// CHECK:STDOUT: !293 = !{!280, !281}
// CHECK:STDOUT: !294 = distinct !DISubprogram(name: "E", linkageName: "_CE.Main.c5e4ba542a56b71f", scope: null, file: !4, line: 61, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !305)
// CHECK:STDOUT: !295 = !DILocalVariable(arg: 1, scope: !294, type: !9)
// CHECK:STDOUT: !296 = !DILocalVariable(arg: 2, scope: !294, type: !6)
// CHECK:STDOUT: !303 = !DILocation(line: 62, column: 10, scope: !294)
// CHECK:STDOUT: !304 = !DILocation(line: 62, column: 3, scope: !294)
// CHECK:STDOUT: !305 = !{!295, !296}
// CHECK:STDOUT: !306 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.c5e4ba542a56b71f", scope: null, file: !4, line: 65, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !317)
// CHECK:STDOUT: !307 = !DILocalVariable(arg: 1, scope: !306, type: !9)
// CHECK:STDOUT: !308 = !DILocalVariable(arg: 2, scope: !306, type: !6)
// CHECK:STDOUT: !315 = !DILocation(line: 66, column: 10, scope: !306)
// CHECK:STDOUT: !316 = !DILocation(line: 66, column: 3, scope: !306)
// CHECK:STDOUT: !317 = !{!307, !308}
// CHECK:STDOUT: !318 = distinct !DISubprogram(name: "C", linkageName: "_CC.Main.a76b78461416b4a8", scope: null, file: !4, line: 41, type: !76, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !332)
// CHECK:STDOUT: !319 = !DILocalVariable(arg: 1, scope: !318, type: !9)
// CHECK:STDOUT: !320 = !DILocalVariable(arg: 2, scope: !318, type: !6)
// CHECK:STDOUT: !323 = !DILocation(line: 42, column: 7, scope: !318)
// CHECK:STDOUT: !324 = !DILocation(line: 42, column: 6, scope: !318)
// CHECK:STDOUT: !326 = !DILocation(line: 43, column: 5, scope: !318)
// CHECK:STDOUT: !328 = !DILocation(line: 44, column: 10, scope: !318)
// CHECK:STDOUT: !329 = !DILocation(line: 44, column: 5, scope: !318)
// CHECK:STDOUT: !330 = !DILocation(line: 42, column: 3, scope: !318)
// CHECK:STDOUT: !331 = !DILocation(line: 41, column: 1, scope: !318)
// CHECK:STDOUT: !332 = !{!319, !320}
// CHECK:STDOUT: !333 = distinct !DISubprogram(name: "E", linkageName: "_CE.Main.a76b78461416b4a8", scope: null, file: !4, line: 61, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !344)
// CHECK:STDOUT: !334 = !DILocalVariable(arg: 1, scope: !333, type: !9)
// CHECK:STDOUT: !335 = !DILocalVariable(arg: 2, scope: !333, type: !6)
// CHECK:STDOUT: !342 = !DILocation(line: 62, column: 10, scope: !333)
// CHECK:STDOUT: !343 = !DILocation(line: 62, column: 3, scope: !333)
// CHECK:STDOUT: !344 = !{!334, !335}
// CHECK:STDOUT: !345 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.a76b78461416b4a8", scope: null, file: !4, line: 65, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !356)
// CHECK:STDOUT: !346 = !DILocalVariable(arg: 1, scope: !345, type: !9)
// CHECK:STDOUT: !347 = !DILocalVariable(arg: 2, scope: !345, type: !6)
// CHECK:STDOUT: !354 = !DILocation(line: 66, column: 10, scope: !345)
// CHECK:STDOUT: !355 = !DILocation(line: 66, column: 3, scope: !345)
// CHECK:STDOUT: !356 = !{!346, !347}
// CHECK:STDOUT: !396 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.4d6ceefeb7f772f4", scope: null, file: !4, line: 69, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !407)
// CHECK:STDOUT: !397 = !DILocalVariable(arg: 1, scope: !396, type: !6)
// CHECK:STDOUT: !398 = !DILocalVariable(arg: 2, scope: !396, type: !6)
// CHECK:STDOUT: !404 = !DILocation(line: 70, column: 15, scope: !396)
// CHECK:STDOUT: !405 = !DILocation(line: 70, column: 10, scope: !396)
// CHECK:STDOUT: !406 = !DILocation(line: 70, column: 3, scope: !396)
// CHECK:STDOUT: !407 = !{!397, !398}
// CHECK:STDOUT: !408 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.c5e4ba542a56b71f", scope: null, file: !4, line: 69, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !419)
// CHECK:STDOUT: !409 = !DILocalVariable(arg: 1, scope: !408, type: !9)
// CHECK:STDOUT: !410 = !DILocalVariable(arg: 2, scope: !408, type: !6)
// CHECK:STDOUT: !416 = !DILocation(line: 70, column: 15, scope: !408)
// CHECK:STDOUT: !417 = !DILocation(line: 70, column: 10, scope: !408)
// CHECK:STDOUT: !418 = !DILocation(line: 70, column: 3, scope: !408)
// CHECK:STDOUT: !419 = !{!409, !410}
// CHECK:STDOUT: !420 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.a76b78461416b4a8", scope: null, file: !4, line: 69, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !431)
// CHECK:STDOUT: !421 = !DILocalVariable(arg: 1, scope: !420, type: !9)
// CHECK:STDOUT: !422 = !DILocalVariable(arg: 2, scope: !420, type: !6)
// CHECK:STDOUT: !428 = !DILocation(line: 70, column: 15, scope: !420)
// CHECK:STDOUT: !429 = !DILocation(line: 70, column: 10, scope: !420)
// CHECK:STDOUT: !430 = !DILocation(line: 70, column: 3, scope: !420)
// CHECK:STDOUT: !431 = !{!421, !422}
// CHECK:STDOUT:
