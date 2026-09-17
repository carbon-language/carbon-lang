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
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.9f1044688d586372(ptr %ptr_f64.var), !dbg !23
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.f427c19f24d378eb(ptr %ptr_i32.var), !dbg !19
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.23895ff6626cffe1(ptr %m.var), !dbg !17
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48(ptr %n.var), !dbg !15
// CHECK:STDOUT:   ret void, !dbg !39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.dc14d4dd21801791:core.Destroy.Core"(ptr %self) #0 !dbg !40 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !42
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !44 {
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
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.9f1044688d586372(ptr %self) #0 !dbg !113 {
// CHECK:STDOUT:   ret void, !dbg !118
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.f427c19f24d378eb(ptr %self) #0 !dbg !120 {
// CHECK:STDOUT:   ret void, !dbg !125
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.23895ff6626cffe1(ptr %self) #0 !dbg !127 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.dc14d4dd21801791:core.Destroy.Core"(ptr %self), !dbg !131
// CHECK:STDOUT:   ret void, !dbg !132
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48(ptr %self) #0 !dbg !134 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self), !dbg !138
// CHECK:STDOUT:   ret void, !dbg !139
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CB.Main.4d6ceefeb7f772f4(i32 %x, i32 %count) #0 !dbg !141 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CC.Main.4d6ceefeb7f772f4(i32 %x, i32 %count), !dbg !148
// CHECK:STDOUT:   ret void, !dbg !149
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CD.Main.4d6ceefeb7f772f4(i32 %x, i32 %count) #0 !dbg !151 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 4, !dbg !158
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then.loc49, label %if.else.loc49, !dbg !159
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc49:                                    ; preds = %entry
// CHECK:STDOUT:   ret i32 %x, !dbg !161
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc49:                                    ; preds = %entry
// CHECK:STDOUT:   %Int.as.ModWith.impl.Op.call = srem i32 %count, 2, !dbg !162
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.call = icmp eq i32 %Int.as.ModWith.impl.Op.call, 0, !dbg !162
// CHECK:STDOUT:   br i1 %Int.as.EqWith.impl.Equal.call, label %if.then.loc52, label %if.else.loc52, !dbg !163
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc52:                                    ; preds = %if.else.loc49
// CHECK:STDOUT:   %E.call = call i32 @_CE.Main.4d6ceefeb7f772f4(i32 %x, i32 %count), !dbg !166
// CHECK:STDOUT:   ret i32 %E.call, !dbg !167
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc52:                                    ; preds = %if.else.loc49
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main.4d6ceefeb7f772f4(i32 %x, i32 %count), !dbg !170
// CHECK:STDOUT:   ret i32 %F.call, !dbg !171
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CB.Main.c5e4ba542a56b71f(double %x, i32 %count) #0 !dbg !173 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CC.Main.c5e4ba542a56b71f(double %x, i32 %count), !dbg !180
// CHECK:STDOUT:   ret void, !dbg !181
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CD.Main.c5e4ba542a56b71f(double %x, i32 %count) #0 !dbg !183 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 4, !dbg !190
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then.loc49, label %if.else.loc49, !dbg !191
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc49:                                    ; preds = %entry
// CHECK:STDOUT:   ret double %x, !dbg !193
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc49:                                    ; preds = %entry
// CHECK:STDOUT:   %Int.as.ModWith.impl.Op.call = srem i32 %count, 2, !dbg !194
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.call = icmp eq i32 %Int.as.ModWith.impl.Op.call, 0, !dbg !194
// CHECK:STDOUT:   br i1 %Int.as.EqWith.impl.Equal.call, label %if.then.loc52, label %if.else.loc52, !dbg !195
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc52:                                    ; preds = %if.else.loc49
// CHECK:STDOUT:   %E.call = call double @_CE.Main.c5e4ba542a56b71f(double %x, i32 %count), !dbg !198
// CHECK:STDOUT:   ret double %E.call, !dbg !199
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc52:                                    ; preds = %if.else.loc49
// CHECK:STDOUT:   %F.call = call double @_CF.Main.c5e4ba542a56b71f(double %x, i32 %count), !dbg !202
// CHECK:STDOUT:   ret double %F.call, !dbg !203
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CB.Main.a76b78461416b4a8(ptr %x, i32 %count) #0 !dbg !205 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CC.Main.a76b78461416b4a8(ptr %x, i32 %count), !dbg !212
// CHECK:STDOUT:   ret void, !dbg !213
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CD.Main.a76b78461416b4a8(ptr %x, i32 %count) #0 !dbg !215 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 4, !dbg !222
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then.loc49, label %if.else.loc49, !dbg !223
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc49:                                    ; preds = %entry
// CHECK:STDOUT:   ret ptr %x, !dbg !225
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc49:                                    ; preds = %entry
// CHECK:STDOUT:   %Int.as.ModWith.impl.Op.call = srem i32 %count, 2, !dbg !226
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.call = icmp eq i32 %Int.as.ModWith.impl.Op.call, 0, !dbg !226
// CHECK:STDOUT:   br i1 %Int.as.EqWith.impl.Equal.call, label %if.then.loc52, label %if.else.loc52, !dbg !227
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc52:                                    ; preds = %if.else.loc49
// CHECK:STDOUT:   %E.call = call ptr @_CE.Main.a76b78461416b4a8(ptr %x, i32 %count), !dbg !230
// CHECK:STDOUT:   ret ptr %E.call, !dbg !231
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc52:                                    ; preds = %if.else.loc49
// CHECK:STDOUT:   %F.call = call ptr @_CF.Main.a76b78461416b4a8(ptr %x, i32 %count), !dbg !234
// CHECK:STDOUT:   ret ptr %F.call, !dbg !235
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CC.Main.4d6ceefeb7f772f4(i32 %x, i32 %count) #0 !dbg !269 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.call = icmp sle i32 %count, 2, !dbg !274
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.LessOrEquivalent.call, label %if.then, label %if.else, !dbg !275
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 %count), !dbg !277
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !279
// CHECK:STDOUT:   call void @_CB.Main.4d6ceefeb7f772f4(i32 %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !280
// CHECK:STDOUT:   br label %if.else, !dbg !281
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %if.then, %entry
// CHECK:STDOUT:   ret void, !dbg !282
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CE.Main.4d6ceefeb7f772f4(i32 %x, i32 %count) #0 !dbg !284 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call i32 @_CG.Main.4d6ceefeb7f772f4(i32 %x, i32 %count), !dbg !293
// CHECK:STDOUT:   ret i32 %G.call, !dbg !294
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.4d6ceefeb7f772f4(i32 %x, i32 %count) #0 !dbg !296 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call i32 @_CG.Main.4d6ceefeb7f772f4(i32 %x, i32 %count), !dbg !305
// CHECK:STDOUT:   ret i32 %G.call, !dbg !306
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CC.Main.c5e4ba542a56b71f(double %x, i32 %count) #0 !dbg !308 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.call = icmp sle i32 %count, 2, !dbg !313
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.LessOrEquivalent.call, label %if.then, label %if.else, !dbg !314
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 %count), !dbg !316
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !318
// CHECK:STDOUT:   call void @_CB.Main.c5e4ba542a56b71f(double %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !319
// CHECK:STDOUT:   br label %if.else, !dbg !320
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %if.then, %entry
// CHECK:STDOUT:   ret void, !dbg !321
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CE.Main.c5e4ba542a56b71f(double %x, i32 %count) #0 !dbg !323 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call double @_CG.Main.c5e4ba542a56b71f(double %x, i32 %count), !dbg !332
// CHECK:STDOUT:   ret double %G.call, !dbg !333
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CF.Main.c5e4ba542a56b71f(double %x, i32 %count) #0 !dbg !335 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call double @_CG.Main.c5e4ba542a56b71f(double %x, i32 %count), !dbg !344
// CHECK:STDOUT:   ret double %G.call, !dbg !345
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CC.Main.a76b78461416b4a8(ptr %x, i32 %count) #0 !dbg !347 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.call = icmp sle i32 %count, 2, !dbg !352
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.LessOrEquivalent.call, label %if.then, label %if.else, !dbg !353
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 %count), !dbg !355
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !357
// CHECK:STDOUT:   call void @_CB.Main.a76b78461416b4a8(ptr %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !358
// CHECK:STDOUT:   br label %if.else, !dbg !359
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %if.then, %entry
// CHECK:STDOUT:   ret void, !dbg !360
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CE.Main.a76b78461416b4a8(ptr %x, i32 %count) #0 !dbg !362 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call ptr @_CG.Main.a76b78461416b4a8(ptr %x, i32 %count), !dbg !371
// CHECK:STDOUT:   ret ptr %G.call, !dbg !372
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CF.Main.a76b78461416b4a8(ptr %x, i32 %count) #0 !dbg !374 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call ptr @_CG.Main.a76b78461416b4a8(ptr %x, i32 %count), !dbg !383
// CHECK:STDOUT:   ret ptr %G.call, !dbg !384
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @printf(ptr, ...)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CG.Main.4d6ceefeb7f772f4(i32 %x, i32 %count) #0 !dbg !425 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !433
// CHECK:STDOUT:   %D.call = call i32 @_CD.Main.4d6ceefeb7f772f4(i32 %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !434
// CHECK:STDOUT:   ret i32 %D.call, !dbg !435
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CG.Main.c5e4ba542a56b71f(double %x, i32 %count) #0 !dbg !437 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !445
// CHECK:STDOUT:   %D.call = call double @_CD.Main.c5e4ba542a56b71f(double %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !446
// CHECK:STDOUT:   ret double %D.call, !dbg !447
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CG.Main.a76b78461416b4a8(ptr %x, i32 %count) #0 !dbg !449 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !457
// CHECK:STDOUT:   %D.call = call ptr @_CD.Main.a76b78461416b4a8(ptr %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !458
// CHECK:STDOUT:   ret ptr %D.call, !dbg !459
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
// CHECK:STDOUT: !40 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.dc14d4dd21801791:core.Destroy.Core", scope: null, file: !4, line: 76, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !43)
// CHECK:STDOUT: !41 = !DILocalVariable(arg: 1, scope: !40, type: !9)
// CHECK:STDOUT: !42 = !DILocation(line: 76, column: 3, scope: !40)
// CHECK:STDOUT: !43 = !{!41}
// CHECK:STDOUT: !44 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core", scope: null, file: !4, line: 75, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !47)
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
// CHECK:STDOUT: !112 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !113 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.9f1044688d586372", scope: null, file: !112, line: 26, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !119)
// CHECK:STDOUT: !114 = !DILocalVariable(arg: 1, scope: !113, type: !9)
// CHECK:STDOUT: !118 = !DILocation(line: 26, column: 3, scope: !113)
// CHECK:STDOUT: !119 = !{!114}
// CHECK:STDOUT: !120 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.f427c19f24d378eb", scope: null, file: !112, line: 26, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !126)
// CHECK:STDOUT: !121 = !DILocalVariable(arg: 1, scope: !120, type: !9)
// CHECK:STDOUT: !125 = !DILocation(line: 26, column: 3, scope: !120)
// CHECK:STDOUT: !126 = !{!121}
// CHECK:STDOUT: !127 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.23895ff6626cffe1", scope: null, file: !112, line: 26, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !133)
// CHECK:STDOUT: !128 = !DILocalVariable(arg: 1, scope: !127, type: !9)
// CHECK:STDOUT: !131 = !DILocation(line: 28, column: 5, scope: !127)
// CHECK:STDOUT: !132 = !DILocation(line: 26, column: 3, scope: !127)
// CHECK:STDOUT: !133 = !{!128}
// CHECK:STDOUT: !134 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48", scope: null, file: !112, line: 26, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !140)
// CHECK:STDOUT: !135 = !DILocalVariable(arg: 1, scope: !134, type: !6)
// CHECK:STDOUT: !138 = !DILocation(line: 28, column: 5, scope: !134)
// CHECK:STDOUT: !139 = !DILocation(line: 26, column: 3, scope: !134)
// CHECK:STDOUT: !140 = !{!135}
// CHECK:STDOUT: !141 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.4d6ceefeb7f772f4", scope: null, file: !4, line: 37, type: !59, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !150)
// CHECK:STDOUT: !142 = !DILocalVariable(arg: 1, scope: !141, type: !6)
// CHECK:STDOUT: !143 = !DILocalVariable(arg: 2, scope: !141, type: !6)
// CHECK:STDOUT: !148 = !DILocation(line: 38, column: 3, scope: !141)
// CHECK:STDOUT: !149 = !DILocation(line: 37, column: 1, scope: !141)
// CHECK:STDOUT: !150 = !{!142, !143}
// CHECK:STDOUT: !151 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.4d6ceefeb7f772f4", scope: null, file: !4, line: 48, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !172)
// CHECK:STDOUT: !152 = !DILocalVariable(arg: 1, scope: !151, type: !6)
// CHECK:STDOUT: !153 = !DILocalVariable(arg: 2, scope: !151, type: !6)
// CHECK:STDOUT: !158 = !DILocation(line: 49, column: 7, scope: !151)
// CHECK:STDOUT: !159 = !DILocation(line: 49, column: 6, scope: !151)
// CHECK:STDOUT: !161 = !DILocation(line: 50, column: 5, scope: !151)
// CHECK:STDOUT: !162 = !DILocation(line: 52, column: 7, scope: !151)
// CHECK:STDOUT: !163 = !DILocation(line: 52, column: 6, scope: !151)
// CHECK:STDOUT: !166 = !DILocation(line: 53, column: 12, scope: !151)
// CHECK:STDOUT: !167 = !DILocation(line: 53, column: 5, scope: !151)
// CHECK:STDOUT: !170 = !DILocation(line: 55, column: 12, scope: !151)
// CHECK:STDOUT: !171 = !DILocation(line: 55, column: 5, scope: !151)
// CHECK:STDOUT: !172 = !{!152, !153}
// CHECK:STDOUT: !173 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.c5e4ba542a56b71f", scope: null, file: !4, line: 37, type: !76, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !182)
// CHECK:STDOUT: !174 = !DILocalVariable(arg: 1, scope: !173, type: !9)
// CHECK:STDOUT: !175 = !DILocalVariable(arg: 2, scope: !173, type: !6)
// CHECK:STDOUT: !180 = !DILocation(line: 38, column: 3, scope: !173)
// CHECK:STDOUT: !181 = !DILocation(line: 37, column: 1, scope: !173)
// CHECK:STDOUT: !182 = !{!174, !175}
// CHECK:STDOUT: !183 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.c5e4ba542a56b71f", scope: null, file: !4, line: 48, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !204)
// CHECK:STDOUT: !184 = !DILocalVariable(arg: 1, scope: !183, type: !9)
// CHECK:STDOUT: !185 = !DILocalVariable(arg: 2, scope: !183, type: !6)
// CHECK:STDOUT: !190 = !DILocation(line: 49, column: 7, scope: !183)
// CHECK:STDOUT: !191 = !DILocation(line: 49, column: 6, scope: !183)
// CHECK:STDOUT: !193 = !DILocation(line: 50, column: 5, scope: !183)
// CHECK:STDOUT: !194 = !DILocation(line: 52, column: 7, scope: !183)
// CHECK:STDOUT: !195 = !DILocation(line: 52, column: 6, scope: !183)
// CHECK:STDOUT: !198 = !DILocation(line: 53, column: 12, scope: !183)
// CHECK:STDOUT: !199 = !DILocation(line: 53, column: 5, scope: !183)
// CHECK:STDOUT: !202 = !DILocation(line: 55, column: 12, scope: !183)
// CHECK:STDOUT: !203 = !DILocation(line: 55, column: 5, scope: !183)
// CHECK:STDOUT: !204 = !{!184, !185}
// CHECK:STDOUT: !205 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.a76b78461416b4a8", scope: null, file: !4, line: 37, type: !76, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !214)
// CHECK:STDOUT: !206 = !DILocalVariable(arg: 1, scope: !205, type: !9)
// CHECK:STDOUT: !207 = !DILocalVariable(arg: 2, scope: !205, type: !6)
// CHECK:STDOUT: !212 = !DILocation(line: 38, column: 3, scope: !205)
// CHECK:STDOUT: !213 = !DILocation(line: 37, column: 1, scope: !205)
// CHECK:STDOUT: !214 = !{!206, !207}
// CHECK:STDOUT: !215 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.a76b78461416b4a8", scope: null, file: !4, line: 48, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !236)
// CHECK:STDOUT: !216 = !DILocalVariable(arg: 1, scope: !215, type: !9)
// CHECK:STDOUT: !217 = !DILocalVariable(arg: 2, scope: !215, type: !6)
// CHECK:STDOUT: !222 = !DILocation(line: 49, column: 7, scope: !215)
// CHECK:STDOUT: !223 = !DILocation(line: 49, column: 6, scope: !215)
// CHECK:STDOUT: !225 = !DILocation(line: 50, column: 5, scope: !215)
// CHECK:STDOUT: !226 = !DILocation(line: 52, column: 7, scope: !215)
// CHECK:STDOUT: !227 = !DILocation(line: 52, column: 6, scope: !215)
// CHECK:STDOUT: !230 = !DILocation(line: 53, column: 12, scope: !215)
// CHECK:STDOUT: !231 = !DILocation(line: 53, column: 5, scope: !215)
// CHECK:STDOUT: !234 = !DILocation(line: 55, column: 12, scope: !215)
// CHECK:STDOUT: !235 = !DILocation(line: 55, column: 5, scope: !215)
// CHECK:STDOUT: !236 = !{!216, !217}
// CHECK:STDOUT: !269 = distinct !DISubprogram(name: "C", linkageName: "_CC.Main.4d6ceefeb7f772f4", scope: null, file: !4, line: 41, type: !59, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !283)
// CHECK:STDOUT: !270 = !DILocalVariable(arg: 1, scope: !269, type: !6)
// CHECK:STDOUT: !271 = !DILocalVariable(arg: 2, scope: !269, type: !6)
// CHECK:STDOUT: !274 = !DILocation(line: 42, column: 7, scope: !269)
// CHECK:STDOUT: !275 = !DILocation(line: 42, column: 6, scope: !269)
// CHECK:STDOUT: !277 = !DILocation(line: 43, column: 5, scope: !269)
// CHECK:STDOUT: !279 = !DILocation(line: 44, column: 10, scope: !269)
// CHECK:STDOUT: !280 = !DILocation(line: 44, column: 5, scope: !269)
// CHECK:STDOUT: !281 = !DILocation(line: 42, column: 3, scope: !269)
// CHECK:STDOUT: !282 = !DILocation(line: 41, column: 1, scope: !269)
// CHECK:STDOUT: !283 = !{!270, !271}
// CHECK:STDOUT: !284 = distinct !DISubprogram(name: "E", linkageName: "_CE.Main.4d6ceefeb7f772f4", scope: null, file: !4, line: 61, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !295)
// CHECK:STDOUT: !285 = !DILocalVariable(arg: 1, scope: !284, type: !6)
// CHECK:STDOUT: !286 = !DILocalVariable(arg: 2, scope: !284, type: !6)
// CHECK:STDOUT: !293 = !DILocation(line: 62, column: 10, scope: !284)
// CHECK:STDOUT: !294 = !DILocation(line: 62, column: 3, scope: !284)
// CHECK:STDOUT: !295 = !{!285, !286}
// CHECK:STDOUT: !296 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.4d6ceefeb7f772f4", scope: null, file: !4, line: 65, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !307)
// CHECK:STDOUT: !297 = !DILocalVariable(arg: 1, scope: !296, type: !6)
// CHECK:STDOUT: !298 = !DILocalVariable(arg: 2, scope: !296, type: !6)
// CHECK:STDOUT: !305 = !DILocation(line: 66, column: 10, scope: !296)
// CHECK:STDOUT: !306 = !DILocation(line: 66, column: 3, scope: !296)
// CHECK:STDOUT: !307 = !{!297, !298}
// CHECK:STDOUT: !308 = distinct !DISubprogram(name: "C", linkageName: "_CC.Main.c5e4ba542a56b71f", scope: null, file: !4, line: 41, type: !76, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !322)
// CHECK:STDOUT: !309 = !DILocalVariable(arg: 1, scope: !308, type: !9)
// CHECK:STDOUT: !310 = !DILocalVariable(arg: 2, scope: !308, type: !6)
// CHECK:STDOUT: !313 = !DILocation(line: 42, column: 7, scope: !308)
// CHECK:STDOUT: !314 = !DILocation(line: 42, column: 6, scope: !308)
// CHECK:STDOUT: !316 = !DILocation(line: 43, column: 5, scope: !308)
// CHECK:STDOUT: !318 = !DILocation(line: 44, column: 10, scope: !308)
// CHECK:STDOUT: !319 = !DILocation(line: 44, column: 5, scope: !308)
// CHECK:STDOUT: !320 = !DILocation(line: 42, column: 3, scope: !308)
// CHECK:STDOUT: !321 = !DILocation(line: 41, column: 1, scope: !308)
// CHECK:STDOUT: !322 = !{!309, !310}
// CHECK:STDOUT: !323 = distinct !DISubprogram(name: "E", linkageName: "_CE.Main.c5e4ba542a56b71f", scope: null, file: !4, line: 61, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !334)
// CHECK:STDOUT: !324 = !DILocalVariable(arg: 1, scope: !323, type: !9)
// CHECK:STDOUT: !325 = !DILocalVariable(arg: 2, scope: !323, type: !6)
// CHECK:STDOUT: !332 = !DILocation(line: 62, column: 10, scope: !323)
// CHECK:STDOUT: !333 = !DILocation(line: 62, column: 3, scope: !323)
// CHECK:STDOUT: !334 = !{!324, !325}
// CHECK:STDOUT: !335 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.c5e4ba542a56b71f", scope: null, file: !4, line: 65, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !346)
// CHECK:STDOUT: !336 = !DILocalVariable(arg: 1, scope: !335, type: !9)
// CHECK:STDOUT: !337 = !DILocalVariable(arg: 2, scope: !335, type: !6)
// CHECK:STDOUT: !344 = !DILocation(line: 66, column: 10, scope: !335)
// CHECK:STDOUT: !345 = !DILocation(line: 66, column: 3, scope: !335)
// CHECK:STDOUT: !346 = !{!336, !337}
// CHECK:STDOUT: !347 = distinct !DISubprogram(name: "C", linkageName: "_CC.Main.a76b78461416b4a8", scope: null, file: !4, line: 41, type: !76, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !361)
// CHECK:STDOUT: !348 = !DILocalVariable(arg: 1, scope: !347, type: !9)
// CHECK:STDOUT: !349 = !DILocalVariable(arg: 2, scope: !347, type: !6)
// CHECK:STDOUT: !352 = !DILocation(line: 42, column: 7, scope: !347)
// CHECK:STDOUT: !353 = !DILocation(line: 42, column: 6, scope: !347)
// CHECK:STDOUT: !355 = !DILocation(line: 43, column: 5, scope: !347)
// CHECK:STDOUT: !357 = !DILocation(line: 44, column: 10, scope: !347)
// CHECK:STDOUT: !358 = !DILocation(line: 44, column: 5, scope: !347)
// CHECK:STDOUT: !359 = !DILocation(line: 42, column: 3, scope: !347)
// CHECK:STDOUT: !360 = !DILocation(line: 41, column: 1, scope: !347)
// CHECK:STDOUT: !361 = !{!348, !349}
// CHECK:STDOUT: !362 = distinct !DISubprogram(name: "E", linkageName: "_CE.Main.a76b78461416b4a8", scope: null, file: !4, line: 61, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !373)
// CHECK:STDOUT: !363 = !DILocalVariable(arg: 1, scope: !362, type: !9)
// CHECK:STDOUT: !364 = !DILocalVariable(arg: 2, scope: !362, type: !6)
// CHECK:STDOUT: !371 = !DILocation(line: 62, column: 10, scope: !362)
// CHECK:STDOUT: !372 = !DILocation(line: 62, column: 3, scope: !362)
// CHECK:STDOUT: !373 = !{!363, !364}
// CHECK:STDOUT: !374 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.a76b78461416b4a8", scope: null, file: !4, line: 65, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !385)
// CHECK:STDOUT: !375 = !DILocalVariable(arg: 1, scope: !374, type: !9)
// CHECK:STDOUT: !376 = !DILocalVariable(arg: 2, scope: !374, type: !6)
// CHECK:STDOUT: !383 = !DILocation(line: 66, column: 10, scope: !374)
// CHECK:STDOUT: !384 = !DILocation(line: 66, column: 3, scope: !374)
// CHECK:STDOUT: !385 = !{!375, !376}
// CHECK:STDOUT: !425 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.4d6ceefeb7f772f4", scope: null, file: !4, line: 69, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !436)
// CHECK:STDOUT: !426 = !DILocalVariable(arg: 1, scope: !425, type: !6)
// CHECK:STDOUT: !427 = !DILocalVariable(arg: 2, scope: !425, type: !6)
// CHECK:STDOUT: !433 = !DILocation(line: 70, column: 15, scope: !425)
// CHECK:STDOUT: !434 = !DILocation(line: 70, column: 10, scope: !425)
// CHECK:STDOUT: !435 = !DILocation(line: 70, column: 3, scope: !425)
// CHECK:STDOUT: !436 = !{!426, !427}
// CHECK:STDOUT: !437 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.c5e4ba542a56b71f", scope: null, file: !4, line: 69, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !448)
// CHECK:STDOUT: !438 = !DILocalVariable(arg: 1, scope: !437, type: !9)
// CHECK:STDOUT: !439 = !DILocalVariable(arg: 2, scope: !437, type: !6)
// CHECK:STDOUT: !445 = !DILocation(line: 70, column: 15, scope: !437)
// CHECK:STDOUT: !446 = !DILocation(line: 70, column: 10, scope: !437)
// CHECK:STDOUT: !447 = !DILocation(line: 70, column: 3, scope: !437)
// CHECK:STDOUT: !448 = !{!438, !439}
// CHECK:STDOUT: !449 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.a76b78461416b4a8", scope: null, file: !4, line: 69, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !460)
// CHECK:STDOUT: !450 = !DILocalVariable(arg: 1, scope: !449, type: !9)
// CHECK:STDOUT: !451 = !DILocalVariable(arg: 2, scope: !449, type: !6)
// CHECK:STDOUT: !457 = !DILocation(line: 70, column: 15, scope: !449)
// CHECK:STDOUT: !458 = !DILocation(line: 70, column: 10, scope: !449)
// CHECK:STDOUT: !459 = !DILocation(line: 70, column: 3, scope: !449)
// CHECK:STDOUT: !460 = !{!450, !451}
// CHECK:STDOUT:
