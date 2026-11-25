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
// CHECK:STDOUT:   ret void, !dbg !35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CA.Main.2fad9574e050607e(i32 %x, i32 %count) #0 !dbg !36 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CB.Main.2fad9574e050607e(i32 %x, i32 %count), !dbg !45
// CHECK:STDOUT:   %D.call = call i32 @_CD.Main.2fad9574e050607e(i32 %x, i32 %count), !dbg !50
// CHECK:STDOUT:   ret i32 %D.call, !dbg !51
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CA.Main.21aefb7dfdebb8e4(double %x, i32 %count) #0 !dbg !53 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CB.Main.21aefb7dfdebb8e4(double %x, i32 %count), !dbg !62
// CHECK:STDOUT:   %D.call = call double @_CD.Main.21aefb7dfdebb8e4(double %x, i32 %count), !dbg !67
// CHECK:STDOUT:   ret double %D.call, !dbg !68
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CA.Main.417c9c4a651ab31b(ptr %x, i32 %count) #0 !dbg !70 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CB.Main.417c9c4a651ab31b(ptr %x, i32 %count), !dbg !79
// CHECK:STDOUT:   %D.call = call ptr @_CD.Main.417c9c4a651ab31b(ptr %x, i32 %count), !dbg !82
// CHECK:STDOUT:   ret ptr %D.call, !dbg !83
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CB.Main.2fad9574e050607e(i32 %x, i32 %count) #0 !dbg !100 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CC.Main.2fad9574e050607e(i32 %x, i32 %count), !dbg !107
// CHECK:STDOUT:   ret void, !dbg !108
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CD.Main.2fad9574e050607e(i32 %x, i32 %count) #0 !dbg !110 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 4, !dbg !117
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then.loc49, label %if.else.loc49, !dbg !118
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc49:                                    ; preds = %entry
// CHECK:STDOUT:   ret i32 %x, !dbg !120
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc49:                                    ; preds = %entry
// CHECK:STDOUT:   %Int.as.ModWith.impl.Op.call = srem i32 %count, 2, !dbg !121
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.call = icmp eq i32 %Int.as.ModWith.impl.Op.call, 0, !dbg !121
// CHECK:STDOUT:   br i1 %Int.as.EqWith.impl.Equal.call, label %if.then.loc52, label %if.else.loc52, !dbg !122
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc52:                                    ; preds = %if.else.loc49
// CHECK:STDOUT:   %E.call = call i32 @_CE.Main.2fad9574e050607e(i32 %x, i32 %count), !dbg !125
// CHECK:STDOUT:   ret i32 %E.call, !dbg !126
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc52:                                    ; preds = %if.else.loc49
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main.2fad9574e050607e(i32 %x, i32 %count), !dbg !129
// CHECK:STDOUT:   ret i32 %F.call, !dbg !130
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CB.Main.21aefb7dfdebb8e4(double %x, i32 %count) #0 !dbg !132 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CC.Main.21aefb7dfdebb8e4(double %x, i32 %count), !dbg !139
// CHECK:STDOUT:   ret void, !dbg !140
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CD.Main.21aefb7dfdebb8e4(double %x, i32 %count) #0 !dbg !142 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 4, !dbg !149
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then.loc49, label %if.else.loc49, !dbg !150
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc49:                                    ; preds = %entry
// CHECK:STDOUT:   ret double %x, !dbg !152
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc49:                                    ; preds = %entry
// CHECK:STDOUT:   %Int.as.ModWith.impl.Op.call = srem i32 %count, 2, !dbg !153
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.call = icmp eq i32 %Int.as.ModWith.impl.Op.call, 0, !dbg !153
// CHECK:STDOUT:   br i1 %Int.as.EqWith.impl.Equal.call, label %if.then.loc52, label %if.else.loc52, !dbg !154
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc52:                                    ; preds = %if.else.loc49
// CHECK:STDOUT:   %E.call = call double @_CE.Main.21aefb7dfdebb8e4(double %x, i32 %count), !dbg !157
// CHECK:STDOUT:   ret double %E.call, !dbg !158
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc52:                                    ; preds = %if.else.loc49
// CHECK:STDOUT:   %F.call = call double @_CF.Main.21aefb7dfdebb8e4(double %x, i32 %count), !dbg !161
// CHECK:STDOUT:   ret double %F.call, !dbg !162
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CB.Main.417c9c4a651ab31b(ptr %x, i32 %count) #0 !dbg !164 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CC.Main.417c9c4a651ab31b(ptr %x, i32 %count), !dbg !171
// CHECK:STDOUT:   ret void, !dbg !172
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CD.Main.417c9c4a651ab31b(ptr %x, i32 %count) #0 !dbg !174 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 4, !dbg !181
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then.loc49, label %if.else.loc49, !dbg !182
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc49:                                    ; preds = %entry
// CHECK:STDOUT:   ret ptr %x, !dbg !184
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc49:                                    ; preds = %entry
// CHECK:STDOUT:   %Int.as.ModWith.impl.Op.call = srem i32 %count, 2, !dbg !185
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.call = icmp eq i32 %Int.as.ModWith.impl.Op.call, 0, !dbg !185
// CHECK:STDOUT:   br i1 %Int.as.EqWith.impl.Equal.call, label %if.then.loc52, label %if.else.loc52, !dbg !186
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc52:                                    ; preds = %if.else.loc49
// CHECK:STDOUT:   %E.call = call ptr @_CE.Main.417c9c4a651ab31b(ptr %x, i32 %count), !dbg !189
// CHECK:STDOUT:   ret ptr %E.call, !dbg !190
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc52:                                    ; preds = %if.else.loc49
// CHECK:STDOUT:   %F.call = call ptr @_CF.Main.417c9c4a651ab31b(ptr %x, i32 %count), !dbg !193
// CHECK:STDOUT:   ret ptr %F.call, !dbg !194
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CC.Main.2fad9574e050607e(i32 %x, i32 %count) #0 !dbg !228 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.call = icmp sle i32 %count, 2, !dbg !233
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.LessOrEquivalent.call, label %if.then, label %if.else, !dbg !234
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 %count), !dbg !236
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !238
// CHECK:STDOUT:   call void @_CB.Main.2fad9574e050607e(i32 %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !239
// CHECK:STDOUT:   br label %if.else, !dbg !240
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %if.then, %entry
// CHECK:STDOUT:   ret void, !dbg !241
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CE.Main.2fad9574e050607e(i32 %x, i32 %count) #0 !dbg !243 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call i32 @_CG.Main.2fad9574e050607e(i32 %x, i32 %count), !dbg !252
// CHECK:STDOUT:   ret i32 %G.call, !dbg !253
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.2fad9574e050607e(i32 %x, i32 %count) #0 !dbg !255 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call i32 @_CG.Main.2fad9574e050607e(i32 %x, i32 %count), !dbg !264
// CHECK:STDOUT:   ret i32 %G.call, !dbg !265
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CC.Main.21aefb7dfdebb8e4(double %x, i32 %count) #0 !dbg !267 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.call = icmp sle i32 %count, 2, !dbg !272
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.LessOrEquivalent.call, label %if.then, label %if.else, !dbg !273
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 %count), !dbg !275
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !277
// CHECK:STDOUT:   call void @_CB.Main.21aefb7dfdebb8e4(double %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !278
// CHECK:STDOUT:   br label %if.else, !dbg !279
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %if.then, %entry
// CHECK:STDOUT:   ret void, !dbg !280
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CE.Main.21aefb7dfdebb8e4(double %x, i32 %count) #0 !dbg !282 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call double @_CG.Main.21aefb7dfdebb8e4(double %x, i32 %count), !dbg !291
// CHECK:STDOUT:   ret double %G.call, !dbg !292
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CF.Main.21aefb7dfdebb8e4(double %x, i32 %count) #0 !dbg !294 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call double @_CG.Main.21aefb7dfdebb8e4(double %x, i32 %count), !dbg !303
// CHECK:STDOUT:   ret double %G.call, !dbg !304
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CC.Main.417c9c4a651ab31b(ptr %x, i32 %count) #0 !dbg !306 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.call = icmp sle i32 %count, 2, !dbg !311
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.LessOrEquivalent.call, label %if.then, label %if.else, !dbg !312
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 %count), !dbg !314
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !316
// CHECK:STDOUT:   call void @_CB.Main.417c9c4a651ab31b(ptr %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !317
// CHECK:STDOUT:   br label %if.else, !dbg !318
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %if.then, %entry
// CHECK:STDOUT:   ret void, !dbg !319
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CE.Main.417c9c4a651ab31b(ptr %x, i32 %count) #0 !dbg !321 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call ptr @_CG.Main.417c9c4a651ab31b(ptr %x, i32 %count), !dbg !330
// CHECK:STDOUT:   ret ptr %G.call, !dbg !331
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CF.Main.417c9c4a651ab31b(ptr %x, i32 %count) #0 !dbg !333 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call ptr @_CG.Main.417c9c4a651ab31b(ptr %x, i32 %count), !dbg !342
// CHECK:STDOUT:   ret ptr %G.call, !dbg !343
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @printf(ptr, ...)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CG.Main.2fad9574e050607e(i32 %x, i32 %count) #0 !dbg !384 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !392
// CHECK:STDOUT:   %D.call = call i32 @_CD.Main.2fad9574e050607e(i32 %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !393
// CHECK:STDOUT:   ret i32 %D.call, !dbg !394
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CG.Main.21aefb7dfdebb8e4(double %x, i32 %count) #0 !dbg !396 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !404
// CHECK:STDOUT:   %D.call = call double @_CD.Main.21aefb7dfdebb8e4(double %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !405
// CHECK:STDOUT:   ret double %D.call, !dbg !406
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CG.Main.417c9c4a651ab31b(ptr %x, i32 %count) #0 !dbg !408 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !416
// CHECK:STDOUT:   %D.call = call ptr @_CD.Main.417c9c4a651ab31b(ptr %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !417
// CHECK:STDOUT:   ret ptr %D.call, !dbg !418
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
// CHECK:STDOUT: !35 = !DILocation(line: 74, column: 1, scope: !9)
// CHECK:STDOUT: !36 = distinct !DISubprogram(name: "A", linkageName: "_CA.Main.2fad9574e050607e", scope: null, file: !4, line: 32, type: !25, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !52)
// CHECK:STDOUT: !37 = !DILocalVariable(arg: 1, scope: !36, type: !6)
// CHECK:STDOUT: !38 = !DILocalVariable(arg: 2, scope: !36, type: !6)
// CHECK:STDOUT: !45 = !DILocation(line: 33, column: 3, scope: !36)
// CHECK:STDOUT: !46 = !{null, !6, !6}
// CHECK:STDOUT: !47 = !DISubroutineType(types: !46)
// CHECK:STDOUT: !50 = !DILocation(line: 34, column: 10, scope: !36)
// CHECK:STDOUT: !51 = !DILocation(line: 34, column: 3, scope: !36)
// CHECK:STDOUT: !52 = !{!37, !38}
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "A", linkageName: "_CA.Main.21aefb7dfdebb8e4", scope: null, file: !4, line: 32, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !69)
// CHECK:STDOUT: !54 = !DILocalVariable(arg: 1, scope: !53, type: !28)
// CHECK:STDOUT: !55 = !DILocalVariable(arg: 2, scope: !53, type: !6)
// CHECK:STDOUT: !62 = !DILocation(line: 33, column: 3, scope: !53)
// CHECK:STDOUT: !63 = !{null, !28, !6}
// CHECK:STDOUT: !64 = !DISubroutineType(types: !63)
// CHECK:STDOUT: !67 = !DILocation(line: 34, column: 10, scope: !53)
// CHECK:STDOUT: !68 = !DILocation(line: 34, column: 3, scope: !53)
// CHECK:STDOUT: !69 = !{!54, !55}
// CHECK:STDOUT: !70 = distinct !DISubprogram(name: "A", linkageName: "_CA.Main.417c9c4a651ab31b", scope: null, file: !4, line: 32, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !84)
// CHECK:STDOUT: !71 = !DILocalVariable(arg: 1, scope: !70, type: !28)
// CHECK:STDOUT: !72 = !DILocalVariable(arg: 2, scope: !70, type: !6)
// CHECK:STDOUT: !79 = !DILocation(line: 33, column: 3, scope: !70)
// CHECK:STDOUT: !82 = !DILocation(line: 34, column: 10, scope: !70)
// CHECK:STDOUT: !83 = !DILocation(line: 34, column: 3, scope: !70)
// CHECK:STDOUT: !84 = !{!71, !72}
// CHECK:STDOUT: !100 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.2fad9574e050607e", scope: null, file: !4, line: 37, type: !47, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !109)
// CHECK:STDOUT: !101 = !DILocalVariable(arg: 1, scope: !100, type: !6)
// CHECK:STDOUT: !102 = !DILocalVariable(arg: 2, scope: !100, type: !6)
// CHECK:STDOUT: !107 = !DILocation(line: 38, column: 3, scope: !100)
// CHECK:STDOUT: !108 = !DILocation(line: 37, column: 1, scope: !100)
// CHECK:STDOUT: !109 = !{!101, !102}
// CHECK:STDOUT: !110 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.2fad9574e050607e", scope: null, file: !4, line: 48, type: !25, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !131)
// CHECK:STDOUT: !111 = !DILocalVariable(arg: 1, scope: !110, type: !6)
// CHECK:STDOUT: !112 = !DILocalVariable(arg: 2, scope: !110, type: !6)
// CHECK:STDOUT: !117 = !DILocation(line: 49, column: 7, scope: !110)
// CHECK:STDOUT: !118 = !DILocation(line: 49, column: 6, scope: !110)
// CHECK:STDOUT: !120 = !DILocation(line: 50, column: 5, scope: !110)
// CHECK:STDOUT: !121 = !DILocation(line: 52, column: 7, scope: !110)
// CHECK:STDOUT: !122 = !DILocation(line: 52, column: 6, scope: !110)
// CHECK:STDOUT: !125 = !DILocation(line: 53, column: 12, scope: !110)
// CHECK:STDOUT: !126 = !DILocation(line: 53, column: 5, scope: !110)
// CHECK:STDOUT: !129 = !DILocation(line: 55, column: 12, scope: !110)
// CHECK:STDOUT: !130 = !DILocation(line: 55, column: 5, scope: !110)
// CHECK:STDOUT: !131 = !{!111, !112}
// CHECK:STDOUT: !132 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.21aefb7dfdebb8e4", scope: null, file: !4, line: 37, type: !64, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !141)
// CHECK:STDOUT: !133 = !DILocalVariable(arg: 1, scope: !132, type: !28)
// CHECK:STDOUT: !134 = !DILocalVariable(arg: 2, scope: !132, type: !6)
// CHECK:STDOUT: !139 = !DILocation(line: 38, column: 3, scope: !132)
// CHECK:STDOUT: !140 = !DILocation(line: 37, column: 1, scope: !132)
// CHECK:STDOUT: !141 = !{!133, !134}
// CHECK:STDOUT: !142 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.21aefb7dfdebb8e4", scope: null, file: !4, line: 48, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !163)
// CHECK:STDOUT: !143 = !DILocalVariable(arg: 1, scope: !142, type: !28)
// CHECK:STDOUT: !144 = !DILocalVariable(arg: 2, scope: !142, type: !6)
// CHECK:STDOUT: !149 = !DILocation(line: 49, column: 7, scope: !142)
// CHECK:STDOUT: !150 = !DILocation(line: 49, column: 6, scope: !142)
// CHECK:STDOUT: !152 = !DILocation(line: 50, column: 5, scope: !142)
// CHECK:STDOUT: !153 = !DILocation(line: 52, column: 7, scope: !142)
// CHECK:STDOUT: !154 = !DILocation(line: 52, column: 6, scope: !142)
// CHECK:STDOUT: !157 = !DILocation(line: 53, column: 12, scope: !142)
// CHECK:STDOUT: !158 = !DILocation(line: 53, column: 5, scope: !142)
// CHECK:STDOUT: !161 = !DILocation(line: 55, column: 12, scope: !142)
// CHECK:STDOUT: !162 = !DILocation(line: 55, column: 5, scope: !142)
// CHECK:STDOUT: !163 = !{!143, !144}
// CHECK:STDOUT: !164 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.417c9c4a651ab31b", scope: null, file: !4, line: 37, type: !64, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !173)
// CHECK:STDOUT: !165 = !DILocalVariable(arg: 1, scope: !164, type: !28)
// CHECK:STDOUT: !166 = !DILocalVariable(arg: 2, scope: !164, type: !6)
// CHECK:STDOUT: !171 = !DILocation(line: 38, column: 3, scope: !164)
// CHECK:STDOUT: !172 = !DILocation(line: 37, column: 1, scope: !164)
// CHECK:STDOUT: !173 = !{!165, !166}
// CHECK:STDOUT: !174 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.417c9c4a651ab31b", scope: null, file: !4, line: 48, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !195)
// CHECK:STDOUT: !175 = !DILocalVariable(arg: 1, scope: !174, type: !28)
// CHECK:STDOUT: !176 = !DILocalVariable(arg: 2, scope: !174, type: !6)
// CHECK:STDOUT: !181 = !DILocation(line: 49, column: 7, scope: !174)
// CHECK:STDOUT: !182 = !DILocation(line: 49, column: 6, scope: !174)
// CHECK:STDOUT: !184 = !DILocation(line: 50, column: 5, scope: !174)
// CHECK:STDOUT: !185 = !DILocation(line: 52, column: 7, scope: !174)
// CHECK:STDOUT: !186 = !DILocation(line: 52, column: 6, scope: !174)
// CHECK:STDOUT: !189 = !DILocation(line: 53, column: 12, scope: !174)
// CHECK:STDOUT: !190 = !DILocation(line: 53, column: 5, scope: !174)
// CHECK:STDOUT: !193 = !DILocation(line: 55, column: 12, scope: !174)
// CHECK:STDOUT: !194 = !DILocation(line: 55, column: 5, scope: !174)
// CHECK:STDOUT: !195 = !{!175, !176}
// CHECK:STDOUT: !228 = distinct !DISubprogram(name: "C", linkageName: "_CC.Main.2fad9574e050607e", scope: null, file: !4, line: 41, type: !47, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !242)
// CHECK:STDOUT: !229 = !DILocalVariable(arg: 1, scope: !228, type: !6)
// CHECK:STDOUT: !230 = !DILocalVariable(arg: 2, scope: !228, type: !6)
// CHECK:STDOUT: !233 = !DILocation(line: 42, column: 7, scope: !228)
// CHECK:STDOUT: !234 = !DILocation(line: 42, column: 6, scope: !228)
// CHECK:STDOUT: !236 = !DILocation(line: 43, column: 5, scope: !228)
// CHECK:STDOUT: !238 = !DILocation(line: 44, column: 10, scope: !228)
// CHECK:STDOUT: !239 = !DILocation(line: 44, column: 5, scope: !228)
// CHECK:STDOUT: !240 = !DILocation(line: 42, column: 3, scope: !228)
// CHECK:STDOUT: !241 = !DILocation(line: 41, column: 1, scope: !228)
// CHECK:STDOUT: !242 = !{!229, !230}
// CHECK:STDOUT: !243 = distinct !DISubprogram(name: "E", linkageName: "_CE.Main.2fad9574e050607e", scope: null, file: !4, line: 61, type: !25, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !254)
// CHECK:STDOUT: !244 = !DILocalVariable(arg: 1, scope: !243, type: !6)
// CHECK:STDOUT: !245 = !DILocalVariable(arg: 2, scope: !243, type: !6)
// CHECK:STDOUT: !252 = !DILocation(line: 62, column: 10, scope: !243)
// CHECK:STDOUT: !253 = !DILocation(line: 62, column: 3, scope: !243)
// CHECK:STDOUT: !254 = !{!244, !245}
// CHECK:STDOUT: !255 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.2fad9574e050607e", scope: null, file: !4, line: 65, type: !25, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !266)
// CHECK:STDOUT: !256 = !DILocalVariable(arg: 1, scope: !255, type: !6)
// CHECK:STDOUT: !257 = !DILocalVariable(arg: 2, scope: !255, type: !6)
// CHECK:STDOUT: !264 = !DILocation(line: 66, column: 10, scope: !255)
// CHECK:STDOUT: !265 = !DILocation(line: 66, column: 3, scope: !255)
// CHECK:STDOUT: !266 = !{!256, !257}
// CHECK:STDOUT: !267 = distinct !DISubprogram(name: "C", linkageName: "_CC.Main.21aefb7dfdebb8e4", scope: null, file: !4, line: 41, type: !64, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !281)
// CHECK:STDOUT: !268 = !DILocalVariable(arg: 1, scope: !267, type: !28)
// CHECK:STDOUT: !269 = !DILocalVariable(arg: 2, scope: !267, type: !6)
// CHECK:STDOUT: !272 = !DILocation(line: 42, column: 7, scope: !267)
// CHECK:STDOUT: !273 = !DILocation(line: 42, column: 6, scope: !267)
// CHECK:STDOUT: !275 = !DILocation(line: 43, column: 5, scope: !267)
// CHECK:STDOUT: !277 = !DILocation(line: 44, column: 10, scope: !267)
// CHECK:STDOUT: !278 = !DILocation(line: 44, column: 5, scope: !267)
// CHECK:STDOUT: !279 = !DILocation(line: 42, column: 3, scope: !267)
// CHECK:STDOUT: !280 = !DILocation(line: 41, column: 1, scope: !267)
// CHECK:STDOUT: !281 = !{!268, !269}
// CHECK:STDOUT: !282 = distinct !DISubprogram(name: "E", linkageName: "_CE.Main.21aefb7dfdebb8e4", scope: null, file: !4, line: 61, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !293)
// CHECK:STDOUT: !283 = !DILocalVariable(arg: 1, scope: !282, type: !28)
// CHECK:STDOUT: !284 = !DILocalVariable(arg: 2, scope: !282, type: !6)
// CHECK:STDOUT: !291 = !DILocation(line: 62, column: 10, scope: !282)
// CHECK:STDOUT: !292 = !DILocation(line: 62, column: 3, scope: !282)
// CHECK:STDOUT: !293 = !{!283, !284}
// CHECK:STDOUT: !294 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.21aefb7dfdebb8e4", scope: null, file: !4, line: 65, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !305)
// CHECK:STDOUT: !295 = !DILocalVariable(arg: 1, scope: !294, type: !28)
// CHECK:STDOUT: !296 = !DILocalVariable(arg: 2, scope: !294, type: !6)
// CHECK:STDOUT: !303 = !DILocation(line: 66, column: 10, scope: !294)
// CHECK:STDOUT: !304 = !DILocation(line: 66, column: 3, scope: !294)
// CHECK:STDOUT: !305 = !{!295, !296}
// CHECK:STDOUT: !306 = distinct !DISubprogram(name: "C", linkageName: "_CC.Main.417c9c4a651ab31b", scope: null, file: !4, line: 41, type: !64, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !320)
// CHECK:STDOUT: !307 = !DILocalVariable(arg: 1, scope: !306, type: !28)
// CHECK:STDOUT: !308 = !DILocalVariable(arg: 2, scope: !306, type: !6)
// CHECK:STDOUT: !311 = !DILocation(line: 42, column: 7, scope: !306)
// CHECK:STDOUT: !312 = !DILocation(line: 42, column: 6, scope: !306)
// CHECK:STDOUT: !314 = !DILocation(line: 43, column: 5, scope: !306)
// CHECK:STDOUT: !316 = !DILocation(line: 44, column: 10, scope: !306)
// CHECK:STDOUT: !317 = !DILocation(line: 44, column: 5, scope: !306)
// CHECK:STDOUT: !318 = !DILocation(line: 42, column: 3, scope: !306)
// CHECK:STDOUT: !319 = !DILocation(line: 41, column: 1, scope: !306)
// CHECK:STDOUT: !320 = !{!307, !308}
// CHECK:STDOUT: !321 = distinct !DISubprogram(name: "E", linkageName: "_CE.Main.417c9c4a651ab31b", scope: null, file: !4, line: 61, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !332)
// CHECK:STDOUT: !322 = !DILocalVariable(arg: 1, scope: !321, type: !28)
// CHECK:STDOUT: !323 = !DILocalVariable(arg: 2, scope: !321, type: !6)
// CHECK:STDOUT: !330 = !DILocation(line: 62, column: 10, scope: !321)
// CHECK:STDOUT: !331 = !DILocation(line: 62, column: 3, scope: !321)
// CHECK:STDOUT: !332 = !{!322, !323}
// CHECK:STDOUT: !333 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.417c9c4a651ab31b", scope: null, file: !4, line: 65, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !344)
// CHECK:STDOUT: !334 = !DILocalVariable(arg: 1, scope: !333, type: !28)
// CHECK:STDOUT: !335 = !DILocalVariable(arg: 2, scope: !333, type: !6)
// CHECK:STDOUT: !342 = !DILocation(line: 66, column: 10, scope: !333)
// CHECK:STDOUT: !343 = !DILocation(line: 66, column: 3, scope: !333)
// CHECK:STDOUT: !344 = !{!334, !335}
// CHECK:STDOUT: !384 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.2fad9574e050607e", scope: null, file: !4, line: 69, type: !25, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !395)
// CHECK:STDOUT: !385 = !DILocalVariable(arg: 1, scope: !384, type: !6)
// CHECK:STDOUT: !386 = !DILocalVariable(arg: 2, scope: !384, type: !6)
// CHECK:STDOUT: !392 = !DILocation(line: 70, column: 15, scope: !384)
// CHECK:STDOUT: !393 = !DILocation(line: 70, column: 10, scope: !384)
// CHECK:STDOUT: !394 = !DILocation(line: 70, column: 3, scope: !384)
// CHECK:STDOUT: !395 = !{!385, !386}
// CHECK:STDOUT: !396 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.21aefb7dfdebb8e4", scope: null, file: !4, line: 69, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !407)
// CHECK:STDOUT: !397 = !DILocalVariable(arg: 1, scope: !396, type: !28)
// CHECK:STDOUT: !398 = !DILocalVariable(arg: 2, scope: !396, type: !6)
// CHECK:STDOUT: !404 = !DILocation(line: 70, column: 15, scope: !396)
// CHECK:STDOUT: !405 = !DILocation(line: 70, column: 10, scope: !396)
// CHECK:STDOUT: !406 = !DILocation(line: 70, column: 3, scope: !396)
// CHECK:STDOUT: !407 = !{!397, !398}
// CHECK:STDOUT: !408 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.417c9c4a651ab31b", scope: null, file: !4, line: 69, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !419)
// CHECK:STDOUT: !409 = !DILocalVariable(arg: 1, scope: !408, type: !28)
// CHECK:STDOUT: !410 = !DILocalVariable(arg: 2, scope: !408, type: !6)
// CHECK:STDOUT: !416 = !DILocation(line: 70, column: 15, scope: !408)
// CHECK:STDOUT: !417 = !DILocation(line: 70, column: 10, scope: !408)
// CHECK:STDOUT: !418 = !DILocation(line: 70, column: 3, scope: !408)
// CHECK:STDOUT: !419 = !{!409, !410}
// CHECK:STDOUT:
