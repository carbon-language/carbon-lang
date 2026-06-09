// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/generic/call_recursive_sccs_deep.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/generic/call_recursive_sccs_deep.carbon

import Core library "io";
fn A[T:! Core.Copy](x: T, count: i32) -> T;
fn B[T:! Core.Copy](x: T, count: i32);
fn C[T:! Core.Copy](x: T, count: i32);
fn D[T:! Core.Copy](x: T, count: i32) -> T;
fn E[T:! Core.Copy](x: T, count: i32) -> T;
fn F[T:! Core.Copy](x: T, count: i32) -> T;

// Builds on `call_recursive_mutual.carbon` and `call_recursive_diamond.carbon`
// B-C form a mutually recursive SCC (strongly connected component), D-E-F-G
// form a diamond SCC.
// The two specifics for each of A, B, C, D, E, F, G with a pointer type could
// be deduplicated.
// E and F are also equivalent in the diamond, even though they are different
// generics. There is potential front-end ICF (identical code folding)
// optimization here, if the function fingerprint may infer deduplication
// when not including the two different generic_ids.

fn A[T:! Core.Copy](x: T, count: i32) -> T {
  B(x, count);
  return D(x, count);
}

fn B[T:! Core.Copy](x: T, count: i32) {
  C(x, count);
}

fn C[T:! Core.Copy](x: T, count: i32) {
  if (count <= 2) {
    Core.Print(count);
    B(x, count + 1);
  }
}

fn D[T:! Core.Copy](x: T, count: i32) -> T {
  if (count > 4) {
    return x;
  }
  if (count % 2 == 0) {
    return E(x, count);
  } else {
    return F(x, count);
  }
}

fn G[T:! Core.Copy](x: T, count: i32) -> T;

fn E[T:! Core.Copy](x: T, count: i32) -> T {
  return G(x, count);
}

fn F[T:! Core.Copy](x: T, count: i32) -> T {
  return G(x, count);
}

fn G[T:! Core.Copy](x: T, count: i32) -> T {
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

// CHECK:STDOUT: ; ModuleID = 'call_recursive_sccs_deep.carbon'
// CHECK:STDOUT: source_filename = "call_recursive_sccs_deep.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @printf.int.format = private unnamed_addr constant [4 x i8] c"%d\0A\00", align 1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CM.Main() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %n.var = alloca i32, align 4, !dbg !7
// CHECK:STDOUT:   %m.var = alloca double, align 8, !dbg !8
// CHECK:STDOUT:   %ptr_i32.var = alloca ptr, align 8, !dbg !9
// CHECK:STDOUT:   %ptr_f64.var = alloca ptr, align 8, !dbg !10
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %n.var), !dbg !7
// CHECK:STDOUT:   store i32 0, ptr %n.var, align 4, !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %m.var), !dbg !8
// CHECK:STDOUT:   store double 1.000000e+00, ptr %m.var, align 8, !dbg !8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_i32.var), !dbg !9
// CHECK:STDOUT:   store ptr %n.var, ptr %ptr_i32.var, align 8, !dbg !9
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_f64.var), !dbg !10
// CHECK:STDOUT:   store ptr %m.var, ptr %ptr_f64.var, align 8, !dbg !10
// CHECK:STDOUT:   %.loc79_5 = load i32, ptr %n.var, align 4, !dbg !11
// CHECK:STDOUT:   %A.call.loc79 = call i32 @_CA.Main.5bd8d5767580997a(i32 %.loc79_5, i32 0), !dbg !12
// CHECK:STDOUT:   %.loc80_5 = load double, ptr %m.var, align 8, !dbg !13
// CHECK:STDOUT:   %A.call.loc80 = call double @_CA.Main.c27b765f2159c9a0(double %.loc80_5, i32 0), !dbg !14
// CHECK:STDOUT:   %.loc81_5 = load ptr, ptr %ptr_i32.var, align 8, !dbg !15
// CHECK:STDOUT:   %A.call.loc81 = call ptr @_CA.Main.2621742a01881e29(ptr %.loc81_5, i32 0), !dbg !16
// CHECK:STDOUT:   %.loc82_5 = load ptr, ptr %ptr_f64.var, align 8, !dbg !17
// CHECK:STDOUT:   %A.call.loc82 = call ptr @_CA.Main.2621742a01881e29(ptr %.loc82_5, i32 0), !dbg !18
// CHECK:STDOUT:   call void @"_COp.30d8007fdcc57f83:core.Destroy.Core"(ptr %m.var), !dbg !8
// CHECK:STDOUT:   call void @"_COp.5e27612b9dd31a14:core.Destroy.Core"(ptr %n.var), !dbg !7
// CHECK:STDOUT:   ret void, !dbg !19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.30d8007fdcc57f83:core.Destroy.Core"(ptr %self) #0 !dbg !20 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.5e27612b9dd31a14:core.Destroy.Core"(ptr %self) #0 !dbg !27 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CA.Main.5bd8d5767580997a(i32 %x, i32 %count) #0 !dbg !34 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CB.Main.5bd8d5767580997a(i32 %x, i32 %count), !dbg !40
// CHECK:STDOUT:   %D.call = call i32 @_CD.Main.5bd8d5767580997a(i32 %x, i32 %count), !dbg !41
// CHECK:STDOUT:   ret i32 %D.call, !dbg !42
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CA.Main.c27b765f2159c9a0(double %x, i32 %count) #0 !dbg !43 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CB.Main.c27b765f2159c9a0(double %x, i32 %count), !dbg !49
// CHECK:STDOUT:   %D.call = call double @_CD.Main.c27b765f2159c9a0(double %x, i32 %count), !dbg !50
// CHECK:STDOUT:   ret double %D.call, !dbg !51
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CA.Main.2621742a01881e29(ptr %x, i32 %count) #0 !dbg !52 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CB.Main.2621742a01881e29(ptr %x, i32 %count), !dbg !56
// CHECK:STDOUT:   %D.call = call ptr @_CD.Main.2621742a01881e29(ptr %x, i32 %count), !dbg !57
// CHECK:STDOUT:   ret ptr %D.call, !dbg !58
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CB.Main.5bd8d5767580997a(i32 %x, i32 %count) #0 !dbg !59 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CC.Main.5bd8d5767580997a(i32 %x, i32 %count), !dbg !65
// CHECK:STDOUT:   ret void, !dbg !66
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CD.Main.5bd8d5767580997a(i32 %x, i32 %count) #0 !dbg !67 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 4, !dbg !71
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then.loc48, label %if.else.loc48, !dbg !72
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc48:                                    ; preds = %entry
// CHECK:STDOUT:   ret i32 %x, !dbg !73
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc48:                                    ; preds = %entry
// CHECK:STDOUT:   %Int.as.ModWith.impl.Op.call = srem i32 %count, 2, !dbg !74
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.call = icmp eq i32 %Int.as.ModWith.impl.Op.call, 0, !dbg !74
// CHECK:STDOUT:   br i1 %Int.as.EqWith.impl.Equal.call, label %if.then.loc51, label %if.else.loc51, !dbg !75
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc51:                                    ; preds = %if.else.loc48
// CHECK:STDOUT:   %E.call = call i32 @_CE.Main.5bd8d5767580997a(i32 %x, i32 %count), !dbg !76
// CHECK:STDOUT:   ret i32 %E.call, !dbg !77
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc51:                                    ; preds = %if.else.loc48
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main.5bd8d5767580997a(i32 %x, i32 %count), !dbg !78
// CHECK:STDOUT:   ret i32 %F.call, !dbg !79
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CB.Main.c27b765f2159c9a0(double %x, i32 %count) #0 !dbg !80 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CC.Main.c27b765f2159c9a0(double %x, i32 %count), !dbg !86
// CHECK:STDOUT:   ret void, !dbg !87
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CD.Main.c27b765f2159c9a0(double %x, i32 %count) #0 !dbg !88 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 4, !dbg !92
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then.loc48, label %if.else.loc48, !dbg !93
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc48:                                    ; preds = %entry
// CHECK:STDOUT:   ret double %x, !dbg !94
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc48:                                    ; preds = %entry
// CHECK:STDOUT:   %Int.as.ModWith.impl.Op.call = srem i32 %count, 2, !dbg !95
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.call = icmp eq i32 %Int.as.ModWith.impl.Op.call, 0, !dbg !95
// CHECK:STDOUT:   br i1 %Int.as.EqWith.impl.Equal.call, label %if.then.loc51, label %if.else.loc51, !dbg !96
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc51:                                    ; preds = %if.else.loc48
// CHECK:STDOUT:   %E.call = call double @_CE.Main.c27b765f2159c9a0(double %x, i32 %count), !dbg !97
// CHECK:STDOUT:   ret double %E.call, !dbg !98
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc51:                                    ; preds = %if.else.loc48
// CHECK:STDOUT:   %F.call = call double @_CF.Main.c27b765f2159c9a0(double %x, i32 %count), !dbg !99
// CHECK:STDOUT:   ret double %F.call, !dbg !100
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CB.Main.2621742a01881e29(ptr %x, i32 %count) #0 !dbg !101 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CC.Main.2621742a01881e29(ptr %x, i32 %count), !dbg !105
// CHECK:STDOUT:   ret void, !dbg !106
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CD.Main.2621742a01881e29(ptr %x, i32 %count) #0 !dbg !107 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 4, !dbg !111
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then.loc48, label %if.else.loc48, !dbg !112
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc48:                                    ; preds = %entry
// CHECK:STDOUT:   ret ptr %x, !dbg !113
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc48:                                    ; preds = %entry
// CHECK:STDOUT:   %Int.as.ModWith.impl.Op.call = srem i32 %count, 2, !dbg !114
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.call = icmp eq i32 %Int.as.ModWith.impl.Op.call, 0, !dbg !114
// CHECK:STDOUT:   br i1 %Int.as.EqWith.impl.Equal.call, label %if.then.loc51, label %if.else.loc51, !dbg !115
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc51:                                    ; preds = %if.else.loc48
// CHECK:STDOUT:   %E.call = call ptr @_CE.Main.2621742a01881e29(ptr %x, i32 %count), !dbg !116
// CHECK:STDOUT:   ret ptr %E.call, !dbg !117
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc51:                                    ; preds = %if.else.loc48
// CHECK:STDOUT:   %F.call = call ptr @_CF.Main.2621742a01881e29(ptr %x, i32 %count), !dbg !118
// CHECK:STDOUT:   ret ptr %F.call, !dbg !119
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CC.Main.5bd8d5767580997a(i32 %x, i32 %count) #0 !dbg !120 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.call = icmp sle i32 %count, 2, !dbg !124
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.LessOrEquivalent.call, label %if.then, label %if.else, !dbg !125
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 %count), !dbg !126
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !127
// CHECK:STDOUT:   call void @_CB.Main.5bd8d5767580997a(i32 %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !128
// CHECK:STDOUT:   br label %if.else, !dbg !129
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %if.then, %entry
// CHECK:STDOUT:   ret void, !dbg !130
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CE.Main.5bd8d5767580997a(i32 %x, i32 %count) #0 !dbg !131 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call i32 @_CG.Main.5bd8d5767580997a(i32 %x, i32 %count), !dbg !135
// CHECK:STDOUT:   ret i32 %G.call, !dbg !136
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.5bd8d5767580997a(i32 %x, i32 %count) #0 !dbg !137 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call i32 @_CG.Main.5bd8d5767580997a(i32 %x, i32 %count), !dbg !141
// CHECK:STDOUT:   ret i32 %G.call, !dbg !142
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CC.Main.c27b765f2159c9a0(double %x, i32 %count) #0 !dbg !143 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.call = icmp sle i32 %count, 2, !dbg !147
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.LessOrEquivalent.call, label %if.then, label %if.else, !dbg !148
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 %count), !dbg !149
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !150
// CHECK:STDOUT:   call void @_CB.Main.c27b765f2159c9a0(double %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !151
// CHECK:STDOUT:   br label %if.else, !dbg !152
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %if.then, %entry
// CHECK:STDOUT:   ret void, !dbg !153
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CE.Main.c27b765f2159c9a0(double %x, i32 %count) #0 !dbg !154 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call double @_CG.Main.c27b765f2159c9a0(double %x, i32 %count), !dbg !158
// CHECK:STDOUT:   ret double %G.call, !dbg !159
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CF.Main.c27b765f2159c9a0(double %x, i32 %count) #0 !dbg !160 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call double @_CG.Main.c27b765f2159c9a0(double %x, i32 %count), !dbg !164
// CHECK:STDOUT:   ret double %G.call, !dbg !165
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CC.Main.2621742a01881e29(ptr %x, i32 %count) #0 !dbg !166 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.call = icmp sle i32 %count, 2, !dbg !170
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.LessOrEquivalent.call, label %if.then, label %if.else, !dbg !171
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 %count), !dbg !172
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !173
// CHECK:STDOUT:   call void @_CB.Main.2621742a01881e29(ptr %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !174
// CHECK:STDOUT:   br label %if.else, !dbg !175
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %if.then, %entry
// CHECK:STDOUT:   ret void, !dbg !176
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CE.Main.2621742a01881e29(ptr %x, i32 %count) #0 !dbg !177 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call ptr @_CG.Main.2621742a01881e29(ptr %x, i32 %count), !dbg !181
// CHECK:STDOUT:   ret ptr %G.call, !dbg !182
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CF.Main.2621742a01881e29(ptr %x, i32 %count) #0 !dbg !183 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call ptr @_CG.Main.2621742a01881e29(ptr %x, i32 %count), !dbg !187
// CHECK:STDOUT:   ret ptr %G.call, !dbg !188
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @printf(ptr, ...)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CG.Main.5bd8d5767580997a(i32 %x, i32 %count) #0 !dbg !189 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !193
// CHECK:STDOUT:   %D.call = call i32 @_CD.Main.5bd8d5767580997a(i32 %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !194
// CHECK:STDOUT:   ret i32 %D.call, !dbg !195
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CG.Main.c27b765f2159c9a0(double %x, i32 %count) #0 !dbg !196 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !200
// CHECK:STDOUT:   %D.call = call double @_CD.Main.c27b765f2159c9a0(double %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !201
// CHECK:STDOUT:   ret double %D.call, !dbg !202
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CG.Main.2621742a01881e29(ptr %x, i32 %count) #0 !dbg !203 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !207
// CHECK:STDOUT:   %D.call = call ptr @_CD.Main.2621742a01881e29(ptr %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !208
// CHECK:STDOUT:   ret ptr %D.call, !dbg !209
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CA.Main.2621742a01881e29, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @printf, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CG.Main.5bd8d5767580997a, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CG.Main.c27b765f2159c9a0, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CG.Main.2621742a01881e29, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "call_recursive_sccs_deep.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "M", linkageName: "_CM.Main", scope: null, file: !3, line: 73, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DILocation(line: 74, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 75, column: 3, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 76, column: 3, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 77, column: 3, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 79, column: 5, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 79, column: 3, scope: !4)
// CHECK:STDOUT: !13 = !DILocation(line: 80, column: 5, scope: !4)
// CHECK:STDOUT: !14 = !DILocation(line: 80, column: 3, scope: !4)
// CHECK:STDOUT: !15 = !DILocation(line: 81, column: 5, scope: !4)
// CHECK:STDOUT: !16 = !DILocation(line: 81, column: 3, scope: !4)
// CHECK:STDOUT: !17 = !DILocation(line: 82, column: 5, scope: !4)
// CHECK:STDOUT: !18 = !DILocation(line: 82, column: 3, scope: !4)
// CHECK:STDOUT: !19 = !DILocation(line: 73, column: 1, scope: !4)
// CHECK:STDOUT: !20 = distinct !DISubprogram(name: "Op", linkageName: "_COp.30d8007fdcc57f83:core.Destroy.Core", scope: null, file: !3, line: 75, type: !21, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !24)
// CHECK:STDOUT: !21 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !22 = !{null, !23}
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{!25}
// CHECK:STDOUT: !25 = !DILocalVariable(arg: 1, scope: !20, type: !23)
// CHECK:STDOUT: !26 = !DILocation(line: 75, column: 3, scope: !20)
// CHECK:STDOUT: !27 = distinct !DISubprogram(name: "Op", linkageName: "_COp.5e27612b9dd31a14:core.Destroy.Core", scope: null, file: !3, line: 74, type: !28, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !31)
// CHECK:STDOUT: !28 = !DISubroutineType(types: !29)
// CHECK:STDOUT: !29 = !{null, !30}
// CHECK:STDOUT: !30 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !31 = !{!32}
// CHECK:STDOUT: !32 = !DILocalVariable(arg: 1, scope: !27, type: !30)
// CHECK:STDOUT: !33 = !DILocation(line: 74, column: 3, scope: !27)
// CHECK:STDOUT: !34 = distinct !DISubprogram(name: "A", linkageName: "_CA.Main.5bd8d5767580997a", scope: null, file: !3, line: 31, type: !35, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !37)
// CHECK:STDOUT: !35 = !DISubroutineType(types: !36)
// CHECK:STDOUT: !36 = !{!30, !30, !30}
// CHECK:STDOUT: !37 = !{!38, !39}
// CHECK:STDOUT: !38 = !DILocalVariable(arg: 1, scope: !34, type: !30)
// CHECK:STDOUT: !39 = !DILocalVariable(arg: 2, scope: !34, type: !30)
// CHECK:STDOUT: !40 = !DILocation(line: 32, column: 3, scope: !34)
// CHECK:STDOUT: !41 = !DILocation(line: 33, column: 10, scope: !34)
// CHECK:STDOUT: !42 = !DILocation(line: 33, column: 3, scope: !34)
// CHECK:STDOUT: !43 = distinct !DISubprogram(name: "A", linkageName: "_CA.Main.c27b765f2159c9a0", scope: null, file: !3, line: 31, type: !44, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !46)
// CHECK:STDOUT: !44 = !DISubroutineType(types: !45)
// CHECK:STDOUT: !45 = !{!23, !23, !30}
// CHECK:STDOUT: !46 = !{!47, !48}
// CHECK:STDOUT: !47 = !DILocalVariable(arg: 1, scope: !43, type: !23)
// CHECK:STDOUT: !48 = !DILocalVariable(arg: 2, scope: !43, type: !30)
// CHECK:STDOUT: !49 = !DILocation(line: 32, column: 3, scope: !43)
// CHECK:STDOUT: !50 = !DILocation(line: 33, column: 10, scope: !43)
// CHECK:STDOUT: !51 = !DILocation(line: 33, column: 3, scope: !43)
// CHECK:STDOUT: !52 = distinct !DISubprogram(name: "A", linkageName: "_CA.Main.2621742a01881e29", scope: null, file: !3, line: 31, type: !44, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !53)
// CHECK:STDOUT: !53 = !{!54, !55}
// CHECK:STDOUT: !54 = !DILocalVariable(arg: 1, scope: !52, type: !23)
// CHECK:STDOUT: !55 = !DILocalVariable(arg: 2, scope: !52, type: !30)
// CHECK:STDOUT: !56 = !DILocation(line: 32, column: 3, scope: !52)
// CHECK:STDOUT: !57 = !DILocation(line: 33, column: 10, scope: !52)
// CHECK:STDOUT: !58 = !DILocation(line: 33, column: 3, scope: !52)
// CHECK:STDOUT: !59 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.5bd8d5767580997a", scope: null, file: !3, line: 36, type: !60, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !62)
// CHECK:STDOUT: !60 = !DISubroutineType(types: !61)
// CHECK:STDOUT: !61 = !{null, !30, !30}
// CHECK:STDOUT: !62 = !{!63, !64}
// CHECK:STDOUT: !63 = !DILocalVariable(arg: 1, scope: !59, type: !30)
// CHECK:STDOUT: !64 = !DILocalVariable(arg: 2, scope: !59, type: !30)
// CHECK:STDOUT: !65 = !DILocation(line: 37, column: 3, scope: !59)
// CHECK:STDOUT: !66 = !DILocation(line: 36, column: 1, scope: !59)
// CHECK:STDOUT: !67 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.5bd8d5767580997a", scope: null, file: !3, line: 47, type: !35, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !68)
// CHECK:STDOUT: !68 = !{!69, !70}
// CHECK:STDOUT: !69 = !DILocalVariable(arg: 1, scope: !67, type: !30)
// CHECK:STDOUT: !70 = !DILocalVariable(arg: 2, scope: !67, type: !30)
// CHECK:STDOUT: !71 = !DILocation(line: 48, column: 7, scope: !67)
// CHECK:STDOUT: !72 = !DILocation(line: 48, column: 6, scope: !67)
// CHECK:STDOUT: !73 = !DILocation(line: 49, column: 5, scope: !67)
// CHECK:STDOUT: !74 = !DILocation(line: 51, column: 7, scope: !67)
// CHECK:STDOUT: !75 = !DILocation(line: 51, column: 6, scope: !67)
// CHECK:STDOUT: !76 = !DILocation(line: 52, column: 12, scope: !67)
// CHECK:STDOUT: !77 = !DILocation(line: 52, column: 5, scope: !67)
// CHECK:STDOUT: !78 = !DILocation(line: 54, column: 12, scope: !67)
// CHECK:STDOUT: !79 = !DILocation(line: 54, column: 5, scope: !67)
// CHECK:STDOUT: !80 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.c27b765f2159c9a0", scope: null, file: !3, line: 36, type: !81, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !83)
// CHECK:STDOUT: !81 = !DISubroutineType(types: !82)
// CHECK:STDOUT: !82 = !{null, !23, !30}
// CHECK:STDOUT: !83 = !{!84, !85}
// CHECK:STDOUT: !84 = !DILocalVariable(arg: 1, scope: !80, type: !23)
// CHECK:STDOUT: !85 = !DILocalVariable(arg: 2, scope: !80, type: !30)
// CHECK:STDOUT: !86 = !DILocation(line: 37, column: 3, scope: !80)
// CHECK:STDOUT: !87 = !DILocation(line: 36, column: 1, scope: !80)
// CHECK:STDOUT: !88 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.c27b765f2159c9a0", scope: null, file: !3, line: 47, type: !44, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !89)
// CHECK:STDOUT: !89 = !{!90, !91}
// CHECK:STDOUT: !90 = !DILocalVariable(arg: 1, scope: !88, type: !23)
// CHECK:STDOUT: !91 = !DILocalVariable(arg: 2, scope: !88, type: !30)
// CHECK:STDOUT: !92 = !DILocation(line: 48, column: 7, scope: !88)
// CHECK:STDOUT: !93 = !DILocation(line: 48, column: 6, scope: !88)
// CHECK:STDOUT: !94 = !DILocation(line: 49, column: 5, scope: !88)
// CHECK:STDOUT: !95 = !DILocation(line: 51, column: 7, scope: !88)
// CHECK:STDOUT: !96 = !DILocation(line: 51, column: 6, scope: !88)
// CHECK:STDOUT: !97 = !DILocation(line: 52, column: 12, scope: !88)
// CHECK:STDOUT: !98 = !DILocation(line: 52, column: 5, scope: !88)
// CHECK:STDOUT: !99 = !DILocation(line: 54, column: 12, scope: !88)
// CHECK:STDOUT: !100 = !DILocation(line: 54, column: 5, scope: !88)
// CHECK:STDOUT: !101 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.2621742a01881e29", scope: null, file: !3, line: 36, type: !81, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !102)
// CHECK:STDOUT: !102 = !{!103, !104}
// CHECK:STDOUT: !103 = !DILocalVariable(arg: 1, scope: !101, type: !23)
// CHECK:STDOUT: !104 = !DILocalVariable(arg: 2, scope: !101, type: !30)
// CHECK:STDOUT: !105 = !DILocation(line: 37, column: 3, scope: !101)
// CHECK:STDOUT: !106 = !DILocation(line: 36, column: 1, scope: !101)
// CHECK:STDOUT: !107 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.2621742a01881e29", scope: null, file: !3, line: 47, type: !44, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !108)
// CHECK:STDOUT: !108 = !{!109, !110}
// CHECK:STDOUT: !109 = !DILocalVariable(arg: 1, scope: !107, type: !23)
// CHECK:STDOUT: !110 = !DILocalVariable(arg: 2, scope: !107, type: !30)
// CHECK:STDOUT: !111 = !DILocation(line: 48, column: 7, scope: !107)
// CHECK:STDOUT: !112 = !DILocation(line: 48, column: 6, scope: !107)
// CHECK:STDOUT: !113 = !DILocation(line: 49, column: 5, scope: !107)
// CHECK:STDOUT: !114 = !DILocation(line: 51, column: 7, scope: !107)
// CHECK:STDOUT: !115 = !DILocation(line: 51, column: 6, scope: !107)
// CHECK:STDOUT: !116 = !DILocation(line: 52, column: 12, scope: !107)
// CHECK:STDOUT: !117 = !DILocation(line: 52, column: 5, scope: !107)
// CHECK:STDOUT: !118 = !DILocation(line: 54, column: 12, scope: !107)
// CHECK:STDOUT: !119 = !DILocation(line: 54, column: 5, scope: !107)
// CHECK:STDOUT: !120 = distinct !DISubprogram(name: "C", linkageName: "_CC.Main.5bd8d5767580997a", scope: null, file: !3, line: 40, type: !60, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !121)
// CHECK:STDOUT: !121 = !{!122, !123}
// CHECK:STDOUT: !122 = !DILocalVariable(arg: 1, scope: !120, type: !30)
// CHECK:STDOUT: !123 = !DILocalVariable(arg: 2, scope: !120, type: !30)
// CHECK:STDOUT: !124 = !DILocation(line: 41, column: 7, scope: !120)
// CHECK:STDOUT: !125 = !DILocation(line: 41, column: 6, scope: !120)
// CHECK:STDOUT: !126 = !DILocation(line: 42, column: 5, scope: !120)
// CHECK:STDOUT: !127 = !DILocation(line: 43, column: 10, scope: !120)
// CHECK:STDOUT: !128 = !DILocation(line: 43, column: 5, scope: !120)
// CHECK:STDOUT: !129 = !DILocation(line: 41, column: 3, scope: !120)
// CHECK:STDOUT: !130 = !DILocation(line: 40, column: 1, scope: !120)
// CHECK:STDOUT: !131 = distinct !DISubprogram(name: "E", linkageName: "_CE.Main.5bd8d5767580997a", scope: null, file: !3, line: 60, type: !35, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !132)
// CHECK:STDOUT: !132 = !{!133, !134}
// CHECK:STDOUT: !133 = !DILocalVariable(arg: 1, scope: !131, type: !30)
// CHECK:STDOUT: !134 = !DILocalVariable(arg: 2, scope: !131, type: !30)
// CHECK:STDOUT: !135 = !DILocation(line: 61, column: 10, scope: !131)
// CHECK:STDOUT: !136 = !DILocation(line: 61, column: 3, scope: !131)
// CHECK:STDOUT: !137 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.5bd8d5767580997a", scope: null, file: !3, line: 64, type: !35, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !138)
// CHECK:STDOUT: !138 = !{!139, !140}
// CHECK:STDOUT: !139 = !DILocalVariable(arg: 1, scope: !137, type: !30)
// CHECK:STDOUT: !140 = !DILocalVariable(arg: 2, scope: !137, type: !30)
// CHECK:STDOUT: !141 = !DILocation(line: 65, column: 10, scope: !137)
// CHECK:STDOUT: !142 = !DILocation(line: 65, column: 3, scope: !137)
// CHECK:STDOUT: !143 = distinct !DISubprogram(name: "C", linkageName: "_CC.Main.c27b765f2159c9a0", scope: null, file: !3, line: 40, type: !81, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !144)
// CHECK:STDOUT: !144 = !{!145, !146}
// CHECK:STDOUT: !145 = !DILocalVariable(arg: 1, scope: !143, type: !23)
// CHECK:STDOUT: !146 = !DILocalVariable(arg: 2, scope: !143, type: !30)
// CHECK:STDOUT: !147 = !DILocation(line: 41, column: 7, scope: !143)
// CHECK:STDOUT: !148 = !DILocation(line: 41, column: 6, scope: !143)
// CHECK:STDOUT: !149 = !DILocation(line: 42, column: 5, scope: !143)
// CHECK:STDOUT: !150 = !DILocation(line: 43, column: 10, scope: !143)
// CHECK:STDOUT: !151 = !DILocation(line: 43, column: 5, scope: !143)
// CHECK:STDOUT: !152 = !DILocation(line: 41, column: 3, scope: !143)
// CHECK:STDOUT: !153 = !DILocation(line: 40, column: 1, scope: !143)
// CHECK:STDOUT: !154 = distinct !DISubprogram(name: "E", linkageName: "_CE.Main.c27b765f2159c9a0", scope: null, file: !3, line: 60, type: !44, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !155)
// CHECK:STDOUT: !155 = !{!156, !157}
// CHECK:STDOUT: !156 = !DILocalVariable(arg: 1, scope: !154, type: !23)
// CHECK:STDOUT: !157 = !DILocalVariable(arg: 2, scope: !154, type: !30)
// CHECK:STDOUT: !158 = !DILocation(line: 61, column: 10, scope: !154)
// CHECK:STDOUT: !159 = !DILocation(line: 61, column: 3, scope: !154)
// CHECK:STDOUT: !160 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.c27b765f2159c9a0", scope: null, file: !3, line: 64, type: !44, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !161)
// CHECK:STDOUT: !161 = !{!162, !163}
// CHECK:STDOUT: !162 = !DILocalVariable(arg: 1, scope: !160, type: !23)
// CHECK:STDOUT: !163 = !DILocalVariable(arg: 2, scope: !160, type: !30)
// CHECK:STDOUT: !164 = !DILocation(line: 65, column: 10, scope: !160)
// CHECK:STDOUT: !165 = !DILocation(line: 65, column: 3, scope: !160)
// CHECK:STDOUT: !166 = distinct !DISubprogram(name: "C", linkageName: "_CC.Main.2621742a01881e29", scope: null, file: !3, line: 40, type: !81, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !167)
// CHECK:STDOUT: !167 = !{!168, !169}
// CHECK:STDOUT: !168 = !DILocalVariable(arg: 1, scope: !166, type: !23)
// CHECK:STDOUT: !169 = !DILocalVariable(arg: 2, scope: !166, type: !30)
// CHECK:STDOUT: !170 = !DILocation(line: 41, column: 7, scope: !166)
// CHECK:STDOUT: !171 = !DILocation(line: 41, column: 6, scope: !166)
// CHECK:STDOUT: !172 = !DILocation(line: 42, column: 5, scope: !166)
// CHECK:STDOUT: !173 = !DILocation(line: 43, column: 10, scope: !166)
// CHECK:STDOUT: !174 = !DILocation(line: 43, column: 5, scope: !166)
// CHECK:STDOUT: !175 = !DILocation(line: 41, column: 3, scope: !166)
// CHECK:STDOUT: !176 = !DILocation(line: 40, column: 1, scope: !166)
// CHECK:STDOUT: !177 = distinct !DISubprogram(name: "E", linkageName: "_CE.Main.2621742a01881e29", scope: null, file: !3, line: 60, type: !44, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !178)
// CHECK:STDOUT: !178 = !{!179, !180}
// CHECK:STDOUT: !179 = !DILocalVariable(arg: 1, scope: !177, type: !23)
// CHECK:STDOUT: !180 = !DILocalVariable(arg: 2, scope: !177, type: !30)
// CHECK:STDOUT: !181 = !DILocation(line: 61, column: 10, scope: !177)
// CHECK:STDOUT: !182 = !DILocation(line: 61, column: 3, scope: !177)
// CHECK:STDOUT: !183 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.2621742a01881e29", scope: null, file: !3, line: 64, type: !44, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !184)
// CHECK:STDOUT: !184 = !{!185, !186}
// CHECK:STDOUT: !185 = !DILocalVariable(arg: 1, scope: !183, type: !23)
// CHECK:STDOUT: !186 = !DILocalVariable(arg: 2, scope: !183, type: !30)
// CHECK:STDOUT: !187 = !DILocation(line: 65, column: 10, scope: !183)
// CHECK:STDOUT: !188 = !DILocation(line: 65, column: 3, scope: !183)
// CHECK:STDOUT: !189 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.5bd8d5767580997a", scope: null, file: !3, line: 68, type: !35, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !190)
// CHECK:STDOUT: !190 = !{!191, !192}
// CHECK:STDOUT: !191 = !DILocalVariable(arg: 1, scope: !189, type: !30)
// CHECK:STDOUT: !192 = !DILocalVariable(arg: 2, scope: !189, type: !30)
// CHECK:STDOUT: !193 = !DILocation(line: 69, column: 15, scope: !189)
// CHECK:STDOUT: !194 = !DILocation(line: 69, column: 10, scope: !189)
// CHECK:STDOUT: !195 = !DILocation(line: 69, column: 3, scope: !189)
// CHECK:STDOUT: !196 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.c27b765f2159c9a0", scope: null, file: !3, line: 68, type: !44, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !197)
// CHECK:STDOUT: !197 = !{!198, !199}
// CHECK:STDOUT: !198 = !DILocalVariable(arg: 1, scope: !196, type: !23)
// CHECK:STDOUT: !199 = !DILocalVariable(arg: 2, scope: !196, type: !30)
// CHECK:STDOUT: !200 = !DILocation(line: 69, column: 15, scope: !196)
// CHECK:STDOUT: !201 = !DILocation(line: 69, column: 10, scope: !196)
// CHECK:STDOUT: !202 = !DILocation(line: 69, column: 3, scope: !196)
// CHECK:STDOUT: !203 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.2621742a01881e29", scope: null, file: !3, line: 68, type: !44, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !204)
// CHECK:STDOUT: !204 = !{!205, !206}
// CHECK:STDOUT: !205 = !DILocalVariable(arg: 1, scope: !203, type: !23)
// CHECK:STDOUT: !206 = !DILocalVariable(arg: 2, scope: !203, type: !30)
// CHECK:STDOUT: !207 = !DILocation(line: 69, column: 15, scope: !203)
// CHECK:STDOUT: !208 = !DILocation(line: 69, column: 10, scope: !203)
// CHECK:STDOUT: !209 = !DILocation(line: 69, column: 3, scope: !203)
