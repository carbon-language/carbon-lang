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
// CHECK:STDOUT: define void @_CM.Main() #0 !dbg !9 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %n.var = alloca i32, align 4, !dbg !10
// CHECK:STDOUT:   %m.var = alloca double, align 8, !dbg !12
// CHECK:STDOUT:   %ptr_i32.var = alloca ptr, align 8, !dbg !14
// CHECK:STDOUT:   %ptr_f64.var = alloca ptr, align 8, !dbg !18
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %n.var), !dbg !10
// CHECK:STDOUT:   store i32 0, ptr %n.var, align 4, !dbg !10
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %m.var), !dbg !12
// CHECK:STDOUT:   store double 1.000000e+00, ptr %m.var, align 8, !dbg !12
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_i32.var), !dbg !14
// CHECK:STDOUT:   store ptr %n.var, ptr %ptr_i32.var, align 8, !dbg !14
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_f64.var), !dbg !18
// CHECK:STDOUT:   store ptr %m.var, ptr %ptr_f64.var, align 8, !dbg !18
// CHECK:STDOUT:   %.loc80_5 = load i32, ptr %n.var, align 4, !dbg !22
// CHECK:STDOUT:   %A.call.loc80 = call i32 @_CA.Main.2fad9574e050607e(i32 %.loc80_5, i32 0), !dbg !23
// CHECK:STDOUT:   %.loc81_5 = load double, ptr %m.var, align 8, !dbg !26
// CHECK:STDOUT:   %A.call.loc81 = call double @_CA.Main.21aefb7dfdebb8e4(double %.loc81_5, i32 0), !dbg !27
// CHECK:STDOUT:   %.loc82_5 = load ptr, ptr %ptr_i32.var, align 8, !dbg !31
// CHECK:STDOUT:   %A.call.loc82 = call ptr @_CA.Main.417c9c4a651ab31b(ptr %.loc82_5, i32 0), !dbg !32
// CHECK:STDOUT:   %.loc83_5 = load ptr, ptr %ptr_f64.var, align 8, !dbg !33
// CHECK:STDOUT:   %A.call.loc83 = call ptr @_CA.Main.417c9c4a651ab31b(ptr %.loc83_5, i32 0), !dbg !34
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.b18855c76072615b(ptr %ptr_f64.var), !dbg !18
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.4f29d15a20894e9b(ptr %ptr_i32.var), !dbg !14
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.2b38c2ca9f7f1a94(ptr %m.var), !dbg !12
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %n.var), !dbg !10
// CHECK:STDOUT:   ret void, !dbg !39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CA.Main.2fad9574e050607e(i32 %x, i32 %count) #0 !dbg !40 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CB.Main.2fad9574e050607e(i32 %x, i32 %count), !dbg !49
// CHECK:STDOUT:   %D.call = call i32 @_CD.Main.2fad9574e050607e(i32 %x, i32 %count), !dbg !54
// CHECK:STDOUT:   ret i32 %D.call, !dbg !55
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CA.Main.21aefb7dfdebb8e4(double %x, i32 %count) #0 !dbg !57 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CB.Main.21aefb7dfdebb8e4(double %x, i32 %count), !dbg !66
// CHECK:STDOUT:   %D.call = call double @_CD.Main.21aefb7dfdebb8e4(double %x, i32 %count), !dbg !71
// CHECK:STDOUT:   ret double %D.call, !dbg !72
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CA.Main.417c9c4a651ab31b(ptr %x, i32 %count) #0 !dbg !74 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CB.Main.417c9c4a651ab31b(ptr %x, i32 %count), !dbg !83
// CHECK:STDOUT:   %D.call = call ptr @_CD.Main.417c9c4a651ab31b(ptr %x, i32 %count), !dbg !86
// CHECK:STDOUT:   ret ptr %D.call, !dbg !87
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.b18855c76072615b(ptr %t) #0 !dbg !105 {
// CHECK:STDOUT:   ret void, !dbg !110
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.4f29d15a20894e9b(ptr %t) #0 !dbg !112 {
// CHECK:STDOUT:   ret void, !dbg !117
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.2b38c2ca9f7f1a94(ptr %t) #0 !dbg !119 {
// CHECK:STDOUT:   ret void, !dbg !124
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.f127f30143696d51(ptr %t) #0 !dbg !126 {
// CHECK:STDOUT:   ret void, !dbg !131
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CB.Main.2fad9574e050607e(i32 %x, i32 %count) #0 !dbg !133 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CC.Main.2fad9574e050607e(i32 %x, i32 %count), !dbg !140
// CHECK:STDOUT:   ret void, !dbg !141
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CD.Main.2fad9574e050607e(i32 %x, i32 %count) #0 !dbg !143 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 4, !dbg !150
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then.loc49, label %if.else.loc49, !dbg !151
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc49:                                    ; preds = %entry
// CHECK:STDOUT:   ret i32 %x, !dbg !153
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc49:                                    ; preds = %entry
// CHECK:STDOUT:   %Int.as.ModWith.impl.Op.call = srem i32 %count, 2, !dbg !154
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.call = icmp eq i32 %Int.as.ModWith.impl.Op.call, 0, !dbg !154
// CHECK:STDOUT:   br i1 %Int.as.EqWith.impl.Equal.call, label %if.then.loc52, label %if.else.loc52, !dbg !155
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc52:                                    ; preds = %if.else.loc49
// CHECK:STDOUT:   %E.call = call i32 @_CE.Main.2fad9574e050607e(i32 %x, i32 %count), !dbg !158
// CHECK:STDOUT:   ret i32 %E.call, !dbg !159
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc52:                                    ; preds = %if.else.loc49
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main.2fad9574e050607e(i32 %x, i32 %count), !dbg !162
// CHECK:STDOUT:   ret i32 %F.call, !dbg !163
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CB.Main.21aefb7dfdebb8e4(double %x, i32 %count) #0 !dbg !165 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CC.Main.21aefb7dfdebb8e4(double %x, i32 %count), !dbg !172
// CHECK:STDOUT:   ret void, !dbg !173
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CD.Main.21aefb7dfdebb8e4(double %x, i32 %count) #0 !dbg !175 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 4, !dbg !182
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then.loc49, label %if.else.loc49, !dbg !183
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc49:                                    ; preds = %entry
// CHECK:STDOUT:   ret double %x, !dbg !185
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc49:                                    ; preds = %entry
// CHECK:STDOUT:   %Int.as.ModWith.impl.Op.call = srem i32 %count, 2, !dbg !186
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.call = icmp eq i32 %Int.as.ModWith.impl.Op.call, 0, !dbg !186
// CHECK:STDOUT:   br i1 %Int.as.EqWith.impl.Equal.call, label %if.then.loc52, label %if.else.loc52, !dbg !187
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc52:                                    ; preds = %if.else.loc49
// CHECK:STDOUT:   %E.call = call double @_CE.Main.21aefb7dfdebb8e4(double %x, i32 %count), !dbg !190
// CHECK:STDOUT:   ret double %E.call, !dbg !191
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc52:                                    ; preds = %if.else.loc49
// CHECK:STDOUT:   %F.call = call double @_CF.Main.21aefb7dfdebb8e4(double %x, i32 %count), !dbg !194
// CHECK:STDOUT:   ret double %F.call, !dbg !195
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CB.Main.417c9c4a651ab31b(ptr %x, i32 %count) #0 !dbg !197 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CC.Main.417c9c4a651ab31b(ptr %x, i32 %count), !dbg !204
// CHECK:STDOUT:   ret void, !dbg !205
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CD.Main.417c9c4a651ab31b(ptr %x, i32 %count) #0 !dbg !207 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 4, !dbg !214
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then.loc49, label %if.else.loc49, !dbg !215
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc49:                                    ; preds = %entry
// CHECK:STDOUT:   ret ptr %x, !dbg !217
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc49:                                    ; preds = %entry
// CHECK:STDOUT:   %Int.as.ModWith.impl.Op.call = srem i32 %count, 2, !dbg !218
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.call = icmp eq i32 %Int.as.ModWith.impl.Op.call, 0, !dbg !218
// CHECK:STDOUT:   br i1 %Int.as.EqWith.impl.Equal.call, label %if.then.loc52, label %if.else.loc52, !dbg !219
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc52:                                    ; preds = %if.else.loc49
// CHECK:STDOUT:   %E.call = call ptr @_CE.Main.417c9c4a651ab31b(ptr %x, i32 %count), !dbg !222
// CHECK:STDOUT:   ret ptr %E.call, !dbg !223
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc52:                                    ; preds = %if.else.loc49
// CHECK:STDOUT:   %F.call = call ptr @_CF.Main.417c9c4a651ab31b(ptr %x, i32 %count), !dbg !226
// CHECK:STDOUT:   ret ptr %F.call, !dbg !227
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CC.Main.2fad9574e050607e(i32 %x, i32 %count) #0 !dbg !261 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.call = icmp sle i32 %count, 2, !dbg !266
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.LessOrEquivalent.call, label %if.then, label %if.else, !dbg !267
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 %count), !dbg !269
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !271
// CHECK:STDOUT:   call void @_CB.Main.2fad9574e050607e(i32 %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !272
// CHECK:STDOUT:   br label %if.else, !dbg !273
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %if.then, %entry
// CHECK:STDOUT:   ret void, !dbg !274
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CE.Main.2fad9574e050607e(i32 %x, i32 %count) #0 !dbg !276 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call i32 @_CG.Main.2fad9574e050607e(i32 %x, i32 %count), !dbg !285
// CHECK:STDOUT:   ret i32 %G.call, !dbg !286
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.2fad9574e050607e(i32 %x, i32 %count) #0 !dbg !288 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call i32 @_CG.Main.2fad9574e050607e(i32 %x, i32 %count), !dbg !297
// CHECK:STDOUT:   ret i32 %G.call, !dbg !298
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CC.Main.21aefb7dfdebb8e4(double %x, i32 %count) #0 !dbg !300 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.call = icmp sle i32 %count, 2, !dbg !305
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.LessOrEquivalent.call, label %if.then, label %if.else, !dbg !306
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 %count), !dbg !308
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !310
// CHECK:STDOUT:   call void @_CB.Main.21aefb7dfdebb8e4(double %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !311
// CHECK:STDOUT:   br label %if.else, !dbg !312
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %if.then, %entry
// CHECK:STDOUT:   ret void, !dbg !313
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CE.Main.21aefb7dfdebb8e4(double %x, i32 %count) #0 !dbg !315 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call double @_CG.Main.21aefb7dfdebb8e4(double %x, i32 %count), !dbg !324
// CHECK:STDOUT:   ret double %G.call, !dbg !325
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CF.Main.21aefb7dfdebb8e4(double %x, i32 %count) #0 !dbg !327 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call double @_CG.Main.21aefb7dfdebb8e4(double %x, i32 %count), !dbg !336
// CHECK:STDOUT:   ret double %G.call, !dbg !337
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CC.Main.417c9c4a651ab31b(ptr %x, i32 %count) #0 !dbg !339 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.call = icmp sle i32 %count, 2, !dbg !344
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.LessOrEquivalent.call, label %if.then, label %if.else, !dbg !345
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 %count), !dbg !347
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !349
// CHECK:STDOUT:   call void @_CB.Main.417c9c4a651ab31b(ptr %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !350
// CHECK:STDOUT:   br label %if.else, !dbg !351
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %if.then, %entry
// CHECK:STDOUT:   ret void, !dbg !352
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CE.Main.417c9c4a651ab31b(ptr %x, i32 %count) #0 !dbg !354 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call ptr @_CG.Main.417c9c4a651ab31b(ptr %x, i32 %count), !dbg !363
// CHECK:STDOUT:   ret ptr %G.call, !dbg !364
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CF.Main.417c9c4a651ab31b(ptr %x, i32 %count) #0 !dbg !366 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call ptr @_CG.Main.417c9c4a651ab31b(ptr %x, i32 %count), !dbg !375
// CHECK:STDOUT:   ret ptr %G.call, !dbg !376
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @printf(ptr, ...)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CG.Main.2fad9574e050607e(i32 %x, i32 %count) #0 !dbg !417 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !425
// CHECK:STDOUT:   %D.call = call i32 @_CD.Main.2fad9574e050607e(i32 %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !426
// CHECK:STDOUT:   ret i32 %D.call, !dbg !427
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CG.Main.21aefb7dfdebb8e4(double %x, i32 %count) #0 !dbg !429 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !437
// CHECK:STDOUT:   %D.call = call double @_CD.Main.21aefb7dfdebb8e4(double %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !438
// CHECK:STDOUT:   ret double %D.call, !dbg !439
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CG.Main.417c9c4a651ab31b(ptr %x, i32 %count) #0 !dbg !441 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !449
// CHECK:STDOUT:   %D.call = call ptr @_CD.Main.417c9c4a651ab31b(ptr %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !450
// CHECK:STDOUT:   ret ptr %D.call, !dbg !451
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CA.Main.417c9c4a651ab31b, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @printf, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CG.Main.2fad9574e050607e, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CG.Main.21aefb7dfdebb8e4, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CG.Main.417c9c4a651ab31b, { 1, 0 }
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
// CHECK:STDOUT: !9 = distinct !DISubprogram(name: "M", linkageName: "_CM.Main", scope: null, file: !4, line: 74, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !10 = !DILocation(line: 75, column: 3, scope: !9)
// CHECK:STDOUT: !12 = !DILocation(line: 76, column: 3, scope: !9)
// CHECK:STDOUT: !14 = !DILocation(line: 77, column: 3, scope: !9)
// CHECK:STDOUT: !18 = !DILocation(line: 78, column: 3, scope: !9)
// CHECK:STDOUT: !22 = !DILocation(line: 80, column: 5, scope: !9)
// CHECK:STDOUT: !23 = !DILocation(line: 80, column: 3, scope: !9)
// CHECK:STDOUT: !24 = !{!6, !6, !6}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = !DILocation(line: 81, column: 5, scope: !9)
// CHECK:STDOUT: !27 = !DILocation(line: 81, column: 3, scope: !9)
// CHECK:STDOUT: !28 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !29 = !{!28, !28, !6}
// CHECK:STDOUT: !30 = !DISubroutineType(types: !29)
// CHECK:STDOUT: !31 = !DILocation(line: 82, column: 5, scope: !9)
// CHECK:STDOUT: !32 = !DILocation(line: 82, column: 3, scope: !9)
// CHECK:STDOUT: !33 = !DILocation(line: 83, column: 5, scope: !9)
// CHECK:STDOUT: !34 = !DILocation(line: 83, column: 3, scope: !9)
// CHECK:STDOUT: !35 = !{null, !28}
// CHECK:STDOUT: !36 = !DISubroutineType(types: !35)
// CHECK:STDOUT: !37 = !{null, !6}
// CHECK:STDOUT: !38 = !DISubroutineType(types: !37)
// CHECK:STDOUT: !39 = !DILocation(line: 74, column: 1, scope: !9)
// CHECK:STDOUT: !40 = distinct !DISubprogram(name: "A", linkageName: "_CA.Main.2fad9574e050607e", scope: null, file: !4, line: 32, type: !25, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !56)
// CHECK:STDOUT: !41 = !DILocalVariable(arg: 1, scope: !40, type: !6)
// CHECK:STDOUT: !42 = !DILocalVariable(arg: 2, scope: !40, type: !6)
// CHECK:STDOUT: !49 = !DILocation(line: 33, column: 3, scope: !40)
// CHECK:STDOUT: !50 = !{null, !6, !6}
// CHECK:STDOUT: !51 = !DISubroutineType(types: !50)
// CHECK:STDOUT: !54 = !DILocation(line: 34, column: 10, scope: !40)
// CHECK:STDOUT: !55 = !DILocation(line: 34, column: 3, scope: !40)
// CHECK:STDOUT: !56 = !{!41, !42}
// CHECK:STDOUT: !57 = distinct !DISubprogram(name: "A", linkageName: "_CA.Main.21aefb7dfdebb8e4", scope: null, file: !4, line: 32, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !73)
// CHECK:STDOUT: !58 = !DILocalVariable(arg: 1, scope: !57, type: !28)
// CHECK:STDOUT: !59 = !DILocalVariable(arg: 2, scope: !57, type: !6)
// CHECK:STDOUT: !66 = !DILocation(line: 33, column: 3, scope: !57)
// CHECK:STDOUT: !67 = !{null, !28, !6}
// CHECK:STDOUT: !68 = !DISubroutineType(types: !67)
// CHECK:STDOUT: !71 = !DILocation(line: 34, column: 10, scope: !57)
// CHECK:STDOUT: !72 = !DILocation(line: 34, column: 3, scope: !57)
// CHECK:STDOUT: !73 = !{!58, !59}
// CHECK:STDOUT: !74 = distinct !DISubprogram(name: "A", linkageName: "_CA.Main.417c9c4a651ab31b", scope: null, file: !4, line: 32, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !88)
// CHECK:STDOUT: !75 = !DILocalVariable(arg: 1, scope: !74, type: !28)
// CHECK:STDOUT: !76 = !DILocalVariable(arg: 2, scope: !74, type: !6)
// CHECK:STDOUT: !83 = !DILocation(line: 33, column: 3, scope: !74)
// CHECK:STDOUT: !86 = !DILocation(line: 34, column: 10, scope: !74)
// CHECK:STDOUT: !87 = !DILocation(line: 34, column: 3, scope: !74)
// CHECK:STDOUT: !88 = !{!75, !76}
// CHECK:STDOUT: !104 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !105 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.b18855c76072615b", scope: null, file: !104, line: 35, type: !36, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !111)
// CHECK:STDOUT: !106 = !DILocalVariable(arg: 1, scope: !105, type: !28)
// CHECK:STDOUT: !110 = !DILocation(line: 35, column: 1, scope: !105)
// CHECK:STDOUT: !111 = !{!106}
// CHECK:STDOUT: !112 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.4f29d15a20894e9b", scope: null, file: !104, line: 35, type: !36, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !118)
// CHECK:STDOUT: !113 = !DILocalVariable(arg: 1, scope: !112, type: !28)
// CHECK:STDOUT: !117 = !DILocation(line: 35, column: 1, scope: !112)
// CHECK:STDOUT: !118 = !{!113}
// CHECK:STDOUT: !119 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.2b38c2ca9f7f1a94", scope: null, file: !104, line: 35, type: !36, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !125)
// CHECK:STDOUT: !120 = !DILocalVariable(arg: 1, scope: !119, type: !28)
// CHECK:STDOUT: !124 = !DILocation(line: 35, column: 1, scope: !119)
// CHECK:STDOUT: !125 = !{!120}
// CHECK:STDOUT: !126 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.f127f30143696d51", scope: null, file: !104, line: 35, type: !38, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !132)
// CHECK:STDOUT: !127 = !DILocalVariable(arg: 1, scope: !126, type: !6)
// CHECK:STDOUT: !131 = !DILocation(line: 35, column: 1, scope: !126)
// CHECK:STDOUT: !132 = !{!127}
// CHECK:STDOUT: !133 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.2fad9574e050607e", scope: null, file: !4, line: 37, type: !51, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !142)
// CHECK:STDOUT: !134 = !DILocalVariable(arg: 1, scope: !133, type: !6)
// CHECK:STDOUT: !135 = !DILocalVariable(arg: 2, scope: !133, type: !6)
// CHECK:STDOUT: !140 = !DILocation(line: 38, column: 3, scope: !133)
// CHECK:STDOUT: !141 = !DILocation(line: 37, column: 1, scope: !133)
// CHECK:STDOUT: !142 = !{!134, !135}
// CHECK:STDOUT: !143 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.2fad9574e050607e", scope: null, file: !4, line: 48, type: !25, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !164)
// CHECK:STDOUT: !144 = !DILocalVariable(arg: 1, scope: !143, type: !6)
// CHECK:STDOUT: !145 = !DILocalVariable(arg: 2, scope: !143, type: !6)
// CHECK:STDOUT: !150 = !DILocation(line: 49, column: 7, scope: !143)
// CHECK:STDOUT: !151 = !DILocation(line: 49, column: 6, scope: !143)
// CHECK:STDOUT: !153 = !DILocation(line: 50, column: 5, scope: !143)
// CHECK:STDOUT: !154 = !DILocation(line: 52, column: 7, scope: !143)
// CHECK:STDOUT: !155 = !DILocation(line: 52, column: 6, scope: !143)
// CHECK:STDOUT: !158 = !DILocation(line: 53, column: 12, scope: !143)
// CHECK:STDOUT: !159 = !DILocation(line: 53, column: 5, scope: !143)
// CHECK:STDOUT: !162 = !DILocation(line: 55, column: 12, scope: !143)
// CHECK:STDOUT: !163 = !DILocation(line: 55, column: 5, scope: !143)
// CHECK:STDOUT: !164 = !{!144, !145}
// CHECK:STDOUT: !165 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.21aefb7dfdebb8e4", scope: null, file: !4, line: 37, type: !68, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !174)
// CHECK:STDOUT: !166 = !DILocalVariable(arg: 1, scope: !165, type: !28)
// CHECK:STDOUT: !167 = !DILocalVariable(arg: 2, scope: !165, type: !6)
// CHECK:STDOUT: !172 = !DILocation(line: 38, column: 3, scope: !165)
// CHECK:STDOUT: !173 = !DILocation(line: 37, column: 1, scope: !165)
// CHECK:STDOUT: !174 = !{!166, !167}
// CHECK:STDOUT: !175 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.21aefb7dfdebb8e4", scope: null, file: !4, line: 48, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !196)
// CHECK:STDOUT: !176 = !DILocalVariable(arg: 1, scope: !175, type: !28)
// CHECK:STDOUT: !177 = !DILocalVariable(arg: 2, scope: !175, type: !6)
// CHECK:STDOUT: !182 = !DILocation(line: 49, column: 7, scope: !175)
// CHECK:STDOUT: !183 = !DILocation(line: 49, column: 6, scope: !175)
// CHECK:STDOUT: !185 = !DILocation(line: 50, column: 5, scope: !175)
// CHECK:STDOUT: !186 = !DILocation(line: 52, column: 7, scope: !175)
// CHECK:STDOUT: !187 = !DILocation(line: 52, column: 6, scope: !175)
// CHECK:STDOUT: !190 = !DILocation(line: 53, column: 12, scope: !175)
// CHECK:STDOUT: !191 = !DILocation(line: 53, column: 5, scope: !175)
// CHECK:STDOUT: !194 = !DILocation(line: 55, column: 12, scope: !175)
// CHECK:STDOUT: !195 = !DILocation(line: 55, column: 5, scope: !175)
// CHECK:STDOUT: !196 = !{!176, !177}
// CHECK:STDOUT: !197 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.417c9c4a651ab31b", scope: null, file: !4, line: 37, type: !68, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !206)
// CHECK:STDOUT: !198 = !DILocalVariable(arg: 1, scope: !197, type: !28)
// CHECK:STDOUT: !199 = !DILocalVariable(arg: 2, scope: !197, type: !6)
// CHECK:STDOUT: !204 = !DILocation(line: 38, column: 3, scope: !197)
// CHECK:STDOUT: !205 = !DILocation(line: 37, column: 1, scope: !197)
// CHECK:STDOUT: !206 = !{!198, !199}
// CHECK:STDOUT: !207 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.417c9c4a651ab31b", scope: null, file: !4, line: 48, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !228)
// CHECK:STDOUT: !208 = !DILocalVariable(arg: 1, scope: !207, type: !28)
// CHECK:STDOUT: !209 = !DILocalVariable(arg: 2, scope: !207, type: !6)
// CHECK:STDOUT: !214 = !DILocation(line: 49, column: 7, scope: !207)
// CHECK:STDOUT: !215 = !DILocation(line: 49, column: 6, scope: !207)
// CHECK:STDOUT: !217 = !DILocation(line: 50, column: 5, scope: !207)
// CHECK:STDOUT: !218 = !DILocation(line: 52, column: 7, scope: !207)
// CHECK:STDOUT: !219 = !DILocation(line: 52, column: 6, scope: !207)
// CHECK:STDOUT: !222 = !DILocation(line: 53, column: 12, scope: !207)
// CHECK:STDOUT: !223 = !DILocation(line: 53, column: 5, scope: !207)
// CHECK:STDOUT: !226 = !DILocation(line: 55, column: 12, scope: !207)
// CHECK:STDOUT: !227 = !DILocation(line: 55, column: 5, scope: !207)
// CHECK:STDOUT: !228 = !{!208, !209}
// CHECK:STDOUT: !261 = distinct !DISubprogram(name: "C", linkageName: "_CC.Main.2fad9574e050607e", scope: null, file: !4, line: 41, type: !51, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !275)
// CHECK:STDOUT: !262 = !DILocalVariable(arg: 1, scope: !261, type: !6)
// CHECK:STDOUT: !263 = !DILocalVariable(arg: 2, scope: !261, type: !6)
// CHECK:STDOUT: !266 = !DILocation(line: 42, column: 7, scope: !261)
// CHECK:STDOUT: !267 = !DILocation(line: 42, column: 6, scope: !261)
// CHECK:STDOUT: !269 = !DILocation(line: 43, column: 5, scope: !261)
// CHECK:STDOUT: !271 = !DILocation(line: 44, column: 10, scope: !261)
// CHECK:STDOUT: !272 = !DILocation(line: 44, column: 5, scope: !261)
// CHECK:STDOUT: !273 = !DILocation(line: 42, column: 3, scope: !261)
// CHECK:STDOUT: !274 = !DILocation(line: 41, column: 1, scope: !261)
// CHECK:STDOUT: !275 = !{!262, !263}
// CHECK:STDOUT: !276 = distinct !DISubprogram(name: "E", linkageName: "_CE.Main.2fad9574e050607e", scope: null, file: !4, line: 61, type: !25, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !287)
// CHECK:STDOUT: !277 = !DILocalVariable(arg: 1, scope: !276, type: !6)
// CHECK:STDOUT: !278 = !DILocalVariable(arg: 2, scope: !276, type: !6)
// CHECK:STDOUT: !285 = !DILocation(line: 62, column: 10, scope: !276)
// CHECK:STDOUT: !286 = !DILocation(line: 62, column: 3, scope: !276)
// CHECK:STDOUT: !287 = !{!277, !278}
// CHECK:STDOUT: !288 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.2fad9574e050607e", scope: null, file: !4, line: 65, type: !25, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !299)
// CHECK:STDOUT: !289 = !DILocalVariable(arg: 1, scope: !288, type: !6)
// CHECK:STDOUT: !290 = !DILocalVariable(arg: 2, scope: !288, type: !6)
// CHECK:STDOUT: !297 = !DILocation(line: 66, column: 10, scope: !288)
// CHECK:STDOUT: !298 = !DILocation(line: 66, column: 3, scope: !288)
// CHECK:STDOUT: !299 = !{!289, !290}
// CHECK:STDOUT: !300 = distinct !DISubprogram(name: "C", linkageName: "_CC.Main.21aefb7dfdebb8e4", scope: null, file: !4, line: 41, type: !68, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !314)
// CHECK:STDOUT: !301 = !DILocalVariable(arg: 1, scope: !300, type: !28)
// CHECK:STDOUT: !302 = !DILocalVariable(arg: 2, scope: !300, type: !6)
// CHECK:STDOUT: !305 = !DILocation(line: 42, column: 7, scope: !300)
// CHECK:STDOUT: !306 = !DILocation(line: 42, column: 6, scope: !300)
// CHECK:STDOUT: !308 = !DILocation(line: 43, column: 5, scope: !300)
// CHECK:STDOUT: !310 = !DILocation(line: 44, column: 10, scope: !300)
// CHECK:STDOUT: !311 = !DILocation(line: 44, column: 5, scope: !300)
// CHECK:STDOUT: !312 = !DILocation(line: 42, column: 3, scope: !300)
// CHECK:STDOUT: !313 = !DILocation(line: 41, column: 1, scope: !300)
// CHECK:STDOUT: !314 = !{!301, !302}
// CHECK:STDOUT: !315 = distinct !DISubprogram(name: "E", linkageName: "_CE.Main.21aefb7dfdebb8e4", scope: null, file: !4, line: 61, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !326)
// CHECK:STDOUT: !316 = !DILocalVariable(arg: 1, scope: !315, type: !28)
// CHECK:STDOUT: !317 = !DILocalVariable(arg: 2, scope: !315, type: !6)
// CHECK:STDOUT: !324 = !DILocation(line: 62, column: 10, scope: !315)
// CHECK:STDOUT: !325 = !DILocation(line: 62, column: 3, scope: !315)
// CHECK:STDOUT: !326 = !{!316, !317}
// CHECK:STDOUT: !327 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.21aefb7dfdebb8e4", scope: null, file: !4, line: 65, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !338)
// CHECK:STDOUT: !328 = !DILocalVariable(arg: 1, scope: !327, type: !28)
// CHECK:STDOUT: !329 = !DILocalVariable(arg: 2, scope: !327, type: !6)
// CHECK:STDOUT: !336 = !DILocation(line: 66, column: 10, scope: !327)
// CHECK:STDOUT: !337 = !DILocation(line: 66, column: 3, scope: !327)
// CHECK:STDOUT: !338 = !{!328, !329}
// CHECK:STDOUT: !339 = distinct !DISubprogram(name: "C", linkageName: "_CC.Main.417c9c4a651ab31b", scope: null, file: !4, line: 41, type: !68, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !353)
// CHECK:STDOUT: !340 = !DILocalVariable(arg: 1, scope: !339, type: !28)
// CHECK:STDOUT: !341 = !DILocalVariable(arg: 2, scope: !339, type: !6)
// CHECK:STDOUT: !344 = !DILocation(line: 42, column: 7, scope: !339)
// CHECK:STDOUT: !345 = !DILocation(line: 42, column: 6, scope: !339)
// CHECK:STDOUT: !347 = !DILocation(line: 43, column: 5, scope: !339)
// CHECK:STDOUT: !349 = !DILocation(line: 44, column: 10, scope: !339)
// CHECK:STDOUT: !350 = !DILocation(line: 44, column: 5, scope: !339)
// CHECK:STDOUT: !351 = !DILocation(line: 42, column: 3, scope: !339)
// CHECK:STDOUT: !352 = !DILocation(line: 41, column: 1, scope: !339)
// CHECK:STDOUT: !353 = !{!340, !341}
// CHECK:STDOUT: !354 = distinct !DISubprogram(name: "E", linkageName: "_CE.Main.417c9c4a651ab31b", scope: null, file: !4, line: 61, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !365)
// CHECK:STDOUT: !355 = !DILocalVariable(arg: 1, scope: !354, type: !28)
// CHECK:STDOUT: !356 = !DILocalVariable(arg: 2, scope: !354, type: !6)
// CHECK:STDOUT: !363 = !DILocation(line: 62, column: 10, scope: !354)
// CHECK:STDOUT: !364 = !DILocation(line: 62, column: 3, scope: !354)
// CHECK:STDOUT: !365 = !{!355, !356}
// CHECK:STDOUT: !366 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.417c9c4a651ab31b", scope: null, file: !4, line: 65, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !377)
// CHECK:STDOUT: !367 = !DILocalVariable(arg: 1, scope: !366, type: !28)
// CHECK:STDOUT: !368 = !DILocalVariable(arg: 2, scope: !366, type: !6)
// CHECK:STDOUT: !375 = !DILocation(line: 66, column: 10, scope: !366)
// CHECK:STDOUT: !376 = !DILocation(line: 66, column: 3, scope: !366)
// CHECK:STDOUT: !377 = !{!367, !368}
// CHECK:STDOUT: !417 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.2fad9574e050607e", scope: null, file: !4, line: 69, type: !25, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !428)
// CHECK:STDOUT: !418 = !DILocalVariable(arg: 1, scope: !417, type: !6)
// CHECK:STDOUT: !419 = !DILocalVariable(arg: 2, scope: !417, type: !6)
// CHECK:STDOUT: !425 = !DILocation(line: 70, column: 15, scope: !417)
// CHECK:STDOUT: !426 = !DILocation(line: 70, column: 10, scope: !417)
// CHECK:STDOUT: !427 = !DILocation(line: 70, column: 3, scope: !417)
// CHECK:STDOUT: !428 = !{!418, !419}
// CHECK:STDOUT: !429 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.21aefb7dfdebb8e4", scope: null, file: !4, line: 69, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !440)
// CHECK:STDOUT: !430 = !DILocalVariable(arg: 1, scope: !429, type: !28)
// CHECK:STDOUT: !431 = !DILocalVariable(arg: 2, scope: !429, type: !6)
// CHECK:STDOUT: !437 = !DILocation(line: 70, column: 15, scope: !429)
// CHECK:STDOUT: !438 = !DILocation(line: 70, column: 10, scope: !429)
// CHECK:STDOUT: !439 = !DILocation(line: 70, column: 3, scope: !429)
// CHECK:STDOUT: !440 = !{!430, !431}
// CHECK:STDOUT: !441 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.417c9c4a651ab31b", scope: null, file: !4, line: 69, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !452)
// CHECK:STDOUT: !442 = !DILocalVariable(arg: 1, scope: !441, type: !28)
// CHECK:STDOUT: !443 = !DILocalVariable(arg: 2, scope: !441, type: !6)
// CHECK:STDOUT: !449 = !DILocation(line: 70, column: 15, scope: !441)
// CHECK:STDOUT: !450 = !DILocation(line: 70, column: 10, scope: !441)
// CHECK:STDOUT: !451 = !DILocation(line: 70, column: 3, scope: !441)
// CHECK:STDOUT: !452 = !{!442, !443}
// CHECK:STDOUT:
