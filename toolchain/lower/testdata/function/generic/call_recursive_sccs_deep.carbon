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
// CHECK:STDOUT:   %.loc80_5 = load i32, ptr %n.var, align 4, !dbg !11
// CHECK:STDOUT:   %A.call.loc80 = call i32 @_CA.Main.4d6ceefeb7f772f4(i32 %.loc80_5, i32 0), !dbg !12
// CHECK:STDOUT:   %.loc81_5 = load double, ptr %m.var, align 8, !dbg !13
// CHECK:STDOUT:   %A.call.loc81 = call double @_CA.Main.c5e4ba542a56b71f(double %.loc81_5, i32 0), !dbg !14
// CHECK:STDOUT:   %.loc82_5 = load ptr, ptr %ptr_i32.var, align 8, !dbg !15
// CHECK:STDOUT:   %A.call.loc82 = call ptr @_CA.Main.a76b78461416b4a8(ptr %.loc82_5, i32 0), !dbg !16
// CHECK:STDOUT:   %.loc83_5 = load ptr, ptr %ptr_f64.var, align 8, !dbg !17
// CHECK:STDOUT:   %A.call.loc83 = call ptr @_CA.Main.a76b78461416b4a8(ptr %.loc83_5, i32 0), !dbg !18
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.9f1044688d586372(ptr %ptr_f64.var), !dbg !10
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.f427c19f24d378eb(ptr %ptr_i32.var), !dbg !9
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.23895ff6626cffe1(ptr %m.var), !dbg !8
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48(ptr %n.var), !dbg !7
// CHECK:STDOUT:   ret void, !dbg !19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.a1ae3d31ab69a7f8:core.Destroy.Core"(ptr %self) #0 !dbg !20 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.70db2d92ffddbef7:core.Destroy.Core"(ptr %self) #0 !dbg !27 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CA.Main.4d6ceefeb7f772f4(i32 %x, i32 %count) #0 !dbg !34 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CB.Main.4d6ceefeb7f772f4(i32 %x, i32 %count), !dbg !40
// CHECK:STDOUT:   %D.call = call i32 @_CD.Main.4d6ceefeb7f772f4(i32 %x, i32 %count), !dbg !41
// CHECK:STDOUT:   ret i32 %D.call, !dbg !42
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CA.Main.c5e4ba542a56b71f(double %x, i32 %count) #0 !dbg !43 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CB.Main.c5e4ba542a56b71f(double %x, i32 %count), !dbg !49
// CHECK:STDOUT:   %D.call = call double @_CD.Main.c5e4ba542a56b71f(double %x, i32 %count), !dbg !50
// CHECK:STDOUT:   ret double %D.call, !dbg !51
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CA.Main.a76b78461416b4a8(ptr %x, i32 %count) #0 !dbg !52 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CB.Main.a76b78461416b4a8(ptr %x, i32 %count), !dbg !56
// CHECK:STDOUT:   %D.call = call ptr @_CD.Main.a76b78461416b4a8(ptr %x, i32 %count), !dbg !57
// CHECK:STDOUT:   ret ptr %D.call, !dbg !58
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.9f1044688d586372(ptr %self) #0 !dbg !59 {
// CHECK:STDOUT:   ret void, !dbg !63
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.f427c19f24d378eb(ptr %self) #0 !dbg !64 {
// CHECK:STDOUT:   ret void, !dbg !67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.23895ff6626cffe1(ptr %self) #0 !dbg !68 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.a1ae3d31ab69a7f8:core.Destroy.Core"(ptr %self), !dbg !71
// CHECK:STDOUT:   ret void, !dbg !72
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48(ptr %self) #0 !dbg !73 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.70db2d92ffddbef7:core.Destroy.Core"(ptr %self), !dbg !76
// CHECK:STDOUT:   ret void, !dbg !77
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CB.Main.4d6ceefeb7f772f4(i32 %x, i32 %count) #0 !dbg !78 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CC.Main.4d6ceefeb7f772f4(i32 %x, i32 %count), !dbg !84
// CHECK:STDOUT:   ret void, !dbg !85
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CD.Main.4d6ceefeb7f772f4(i32 %x, i32 %count) #0 !dbg !86 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 4, !dbg !90
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then.loc49, label %if.else.loc49, !dbg !91
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc49:                                    ; preds = %entry
// CHECK:STDOUT:   ret i32 %x, !dbg !92
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc49:                                    ; preds = %entry
// CHECK:STDOUT:   %Int.as.ModWith.impl.Op.call = srem i32 %count, 2, !dbg !93
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.call = icmp eq i32 %Int.as.ModWith.impl.Op.call, 0, !dbg !93
// CHECK:STDOUT:   br i1 %Int.as.EqWith.impl.Equal.call, label %if.then.loc52, label %if.else.loc52, !dbg !94
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc52:                                    ; preds = %if.else.loc49
// CHECK:STDOUT:   %E.call = call i32 @_CE.Main.4d6ceefeb7f772f4(i32 %x, i32 %count), !dbg !95
// CHECK:STDOUT:   ret i32 %E.call, !dbg !96
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc52:                                    ; preds = %if.else.loc49
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main.4d6ceefeb7f772f4(i32 %x, i32 %count), !dbg !97
// CHECK:STDOUT:   ret i32 %F.call, !dbg !98
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CB.Main.c5e4ba542a56b71f(double %x, i32 %count) #0 !dbg !99 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CC.Main.c5e4ba542a56b71f(double %x, i32 %count), !dbg !105
// CHECK:STDOUT:   ret void, !dbg !106
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CD.Main.c5e4ba542a56b71f(double %x, i32 %count) #0 !dbg !107 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 4, !dbg !111
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then.loc49, label %if.else.loc49, !dbg !112
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc49:                                    ; preds = %entry
// CHECK:STDOUT:   ret double %x, !dbg !113
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc49:                                    ; preds = %entry
// CHECK:STDOUT:   %Int.as.ModWith.impl.Op.call = srem i32 %count, 2, !dbg !114
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.call = icmp eq i32 %Int.as.ModWith.impl.Op.call, 0, !dbg !114
// CHECK:STDOUT:   br i1 %Int.as.EqWith.impl.Equal.call, label %if.then.loc52, label %if.else.loc52, !dbg !115
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc52:                                    ; preds = %if.else.loc49
// CHECK:STDOUT:   %E.call = call double @_CE.Main.c5e4ba542a56b71f(double %x, i32 %count), !dbg !116
// CHECK:STDOUT:   ret double %E.call, !dbg !117
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc52:                                    ; preds = %if.else.loc49
// CHECK:STDOUT:   %F.call = call double @_CF.Main.c5e4ba542a56b71f(double %x, i32 %count), !dbg !118
// CHECK:STDOUT:   ret double %F.call, !dbg !119
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CB.Main.a76b78461416b4a8(ptr %x, i32 %count) #0 !dbg !120 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CC.Main.a76b78461416b4a8(ptr %x, i32 %count), !dbg !124
// CHECK:STDOUT:   ret void, !dbg !125
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CD.Main.a76b78461416b4a8(ptr %x, i32 %count) #0 !dbg !126 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 4, !dbg !130
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then.loc49, label %if.else.loc49, !dbg !131
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc49:                                    ; preds = %entry
// CHECK:STDOUT:   ret ptr %x, !dbg !132
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc49:                                    ; preds = %entry
// CHECK:STDOUT:   %Int.as.ModWith.impl.Op.call = srem i32 %count, 2, !dbg !133
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.call = icmp eq i32 %Int.as.ModWith.impl.Op.call, 0, !dbg !133
// CHECK:STDOUT:   br i1 %Int.as.EqWith.impl.Equal.call, label %if.then.loc52, label %if.else.loc52, !dbg !134
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc52:                                    ; preds = %if.else.loc49
// CHECK:STDOUT:   %E.call = call ptr @_CE.Main.a76b78461416b4a8(ptr %x, i32 %count), !dbg !135
// CHECK:STDOUT:   ret ptr %E.call, !dbg !136
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc52:                                    ; preds = %if.else.loc49
// CHECK:STDOUT:   %F.call = call ptr @_CF.Main.a76b78461416b4a8(ptr %x, i32 %count), !dbg !137
// CHECK:STDOUT:   ret ptr %F.call, !dbg !138
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CC.Main.4d6ceefeb7f772f4(i32 %x, i32 %count) #0 !dbg !139 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.call = icmp sle i32 %count, 2, !dbg !143
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.LessOrEquivalent.call, label %if.then, label %if.else, !dbg !144
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 %count), !dbg !145
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !146
// CHECK:STDOUT:   call void @_CB.Main.4d6ceefeb7f772f4(i32 %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !147
// CHECK:STDOUT:   br label %if.else, !dbg !148
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %if.then, %entry
// CHECK:STDOUT:   ret void, !dbg !149
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CE.Main.4d6ceefeb7f772f4(i32 %x, i32 %count) #0 !dbg !150 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call i32 @_CG.Main.4d6ceefeb7f772f4(i32 %x, i32 %count), !dbg !154
// CHECK:STDOUT:   ret i32 %G.call, !dbg !155
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.4d6ceefeb7f772f4(i32 %x, i32 %count) #0 !dbg !156 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call i32 @_CG.Main.4d6ceefeb7f772f4(i32 %x, i32 %count), !dbg !160
// CHECK:STDOUT:   ret i32 %G.call, !dbg !161
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CC.Main.c5e4ba542a56b71f(double %x, i32 %count) #0 !dbg !162 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.call = icmp sle i32 %count, 2, !dbg !166
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.LessOrEquivalent.call, label %if.then, label %if.else, !dbg !167
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 %count), !dbg !168
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !169
// CHECK:STDOUT:   call void @_CB.Main.c5e4ba542a56b71f(double %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !170
// CHECK:STDOUT:   br label %if.else, !dbg !171
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %if.then, %entry
// CHECK:STDOUT:   ret void, !dbg !172
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CE.Main.c5e4ba542a56b71f(double %x, i32 %count) #0 !dbg !173 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call double @_CG.Main.c5e4ba542a56b71f(double %x, i32 %count), !dbg !177
// CHECK:STDOUT:   ret double %G.call, !dbg !178
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CF.Main.c5e4ba542a56b71f(double %x, i32 %count) #0 !dbg !179 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call double @_CG.Main.c5e4ba542a56b71f(double %x, i32 %count), !dbg !183
// CHECK:STDOUT:   ret double %G.call, !dbg !184
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CC.Main.a76b78461416b4a8(ptr %x, i32 %count) #0 !dbg !185 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.call = icmp sle i32 %count, 2, !dbg !189
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.LessOrEquivalent.call, label %if.then, label %if.else, !dbg !190
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 %count), !dbg !191
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !192
// CHECK:STDOUT:   call void @_CB.Main.a76b78461416b4a8(ptr %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !193
// CHECK:STDOUT:   br label %if.else, !dbg !194
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %if.then, %entry
// CHECK:STDOUT:   ret void, !dbg !195
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CE.Main.a76b78461416b4a8(ptr %x, i32 %count) #0 !dbg !196 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call ptr @_CG.Main.a76b78461416b4a8(ptr %x, i32 %count), !dbg !200
// CHECK:STDOUT:   ret ptr %G.call, !dbg !201
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CF.Main.a76b78461416b4a8(ptr %x, i32 %count) #0 !dbg !202 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call ptr @_CG.Main.a76b78461416b4a8(ptr %x, i32 %count), !dbg !206
// CHECK:STDOUT:   ret ptr %G.call, !dbg !207
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @printf(ptr, ...)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CG.Main.4d6ceefeb7f772f4(i32 %x, i32 %count) #0 !dbg !208 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !212
// CHECK:STDOUT:   %D.call = call i32 @_CD.Main.4d6ceefeb7f772f4(i32 %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !213
// CHECK:STDOUT:   ret i32 %D.call, !dbg !214
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CG.Main.c5e4ba542a56b71f(double %x, i32 %count) #0 !dbg !215 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !219
// CHECK:STDOUT:   %D.call = call double @_CD.Main.c5e4ba542a56b71f(double %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !220
// CHECK:STDOUT:   ret double %D.call, !dbg !221
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CG.Main.a76b78461416b4a8(ptr %x, i32 %count) #0 !dbg !222 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !226
// CHECK:STDOUT:   %D.call = call ptr @_CD.Main.a76b78461416b4a8(ptr %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !227
// CHECK:STDOUT:   ret ptr %D.call, !dbg !228
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "call_recursive_sccs_deep.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "M", linkageName: "_CM.Main", scope: null, file: !1, line: 74, type: !5, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DILocation(line: 75, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 76, column: 3, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 77, column: 3, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 78, column: 3, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 80, column: 5, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 80, column: 3, scope: !4)
// CHECK:STDOUT: !13 = !DILocation(line: 81, column: 5, scope: !4)
// CHECK:STDOUT: !14 = !DILocation(line: 81, column: 3, scope: !4)
// CHECK:STDOUT: !15 = !DILocation(line: 82, column: 5, scope: !4)
// CHECK:STDOUT: !16 = !DILocation(line: 82, column: 3, scope: !4)
// CHECK:STDOUT: !17 = !DILocation(line: 83, column: 5, scope: !4)
// CHECK:STDOUT: !18 = !DILocation(line: 83, column: 3, scope: !4)
// CHECK:STDOUT: !19 = !DILocation(line: 74, column: 1, scope: !4)
// CHECK:STDOUT: !20 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.a1ae3d31ab69a7f8:core.Destroy.Core", scope: null, file: !1, line: 76, type: !21, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !24)
// CHECK:STDOUT: !21 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !22 = !{null, !23}
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{!25}
// CHECK:STDOUT: !25 = !DILocalVariable(arg: 1, scope: !20, type: !23)
// CHECK:STDOUT: !26 = !DILocation(line: 76, column: 3, scope: !20)
// CHECK:STDOUT: !27 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.70db2d92ffddbef7:core.Destroy.Core", scope: null, file: !1, line: 75, type: !28, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !31)
// CHECK:STDOUT: !28 = !DISubroutineType(types: !29)
// CHECK:STDOUT: !29 = !{null, !30}
// CHECK:STDOUT: !30 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !31 = !{!32}
// CHECK:STDOUT: !32 = !DILocalVariable(arg: 1, scope: !27, type: !30)
// CHECK:STDOUT: !33 = !DILocation(line: 75, column: 3, scope: !27)
// CHECK:STDOUT: !34 = distinct !DISubprogram(name: "A", linkageName: "_CA.Main.4d6ceefeb7f772f4", scope: null, file: !1, line: 32, type: !35, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !37)
// CHECK:STDOUT: !35 = !DISubroutineType(types: !36)
// CHECK:STDOUT: !36 = !{!30, !30, !30}
// CHECK:STDOUT: !37 = !{!38, !39}
// CHECK:STDOUT: !38 = !DILocalVariable(arg: 1, scope: !34, type: !30)
// CHECK:STDOUT: !39 = !DILocalVariable(arg: 2, scope: !34, type: !30)
// CHECK:STDOUT: !40 = !DILocation(line: 33, column: 3, scope: !34)
// CHECK:STDOUT: !41 = !DILocation(line: 34, column: 10, scope: !34)
// CHECK:STDOUT: !42 = !DILocation(line: 34, column: 3, scope: !34)
// CHECK:STDOUT: !43 = distinct !DISubprogram(name: "A", linkageName: "_CA.Main.c5e4ba542a56b71f", scope: null, file: !1, line: 32, type: !44, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !46)
// CHECK:STDOUT: !44 = !DISubroutineType(types: !45)
// CHECK:STDOUT: !45 = !{!23, !23, !30}
// CHECK:STDOUT: !46 = !{!47, !48}
// CHECK:STDOUT: !47 = !DILocalVariable(arg: 1, scope: !43, type: !23)
// CHECK:STDOUT: !48 = !DILocalVariable(arg: 2, scope: !43, type: !30)
// CHECK:STDOUT: !49 = !DILocation(line: 33, column: 3, scope: !43)
// CHECK:STDOUT: !50 = !DILocation(line: 34, column: 10, scope: !43)
// CHECK:STDOUT: !51 = !DILocation(line: 34, column: 3, scope: !43)
// CHECK:STDOUT: !52 = distinct !DISubprogram(name: "A", linkageName: "_CA.Main.a76b78461416b4a8", scope: null, file: !1, line: 32, type: !44, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !53)
// CHECK:STDOUT: !53 = !{!54, !55}
// CHECK:STDOUT: !54 = !DILocalVariable(arg: 1, scope: !52, type: !23)
// CHECK:STDOUT: !55 = !DILocalVariable(arg: 2, scope: !52, type: !30)
// CHECK:STDOUT: !56 = !DILocation(line: 33, column: 3, scope: !52)
// CHECK:STDOUT: !57 = !DILocation(line: 34, column: 10, scope: !52)
// CHECK:STDOUT: !58 = !DILocation(line: 34, column: 3, scope: !52)
// CHECK:STDOUT: !59 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.9f1044688d586372", scope: null, file: !60, line: 26, type: !21, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !61)
// CHECK:STDOUT: !60 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !61 = !{!62}
// CHECK:STDOUT: !62 = !DILocalVariable(arg: 1, scope: !59, type: !23)
// CHECK:STDOUT: !63 = !DILocation(line: 26, column: 3, scope: !59)
// CHECK:STDOUT: !64 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.f427c19f24d378eb", scope: null, file: !60, line: 26, type: !21, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !65)
// CHECK:STDOUT: !65 = !{!66}
// CHECK:STDOUT: !66 = !DILocalVariable(arg: 1, scope: !64, type: !23)
// CHECK:STDOUT: !67 = !DILocation(line: 26, column: 3, scope: !64)
// CHECK:STDOUT: !68 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.23895ff6626cffe1", scope: null, file: !60, line: 26, type: !21, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !69)
// CHECK:STDOUT: !69 = !{!70}
// CHECK:STDOUT: !70 = !DILocalVariable(arg: 1, scope: !68, type: !23)
// CHECK:STDOUT: !71 = !DILocation(line: 28, column: 5, scope: !68)
// CHECK:STDOUT: !72 = !DILocation(line: 26, column: 3, scope: !68)
// CHECK:STDOUT: !73 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48", scope: null, file: !60, line: 26, type: !28, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !74)
// CHECK:STDOUT: !74 = !{!75}
// CHECK:STDOUT: !75 = !DILocalVariable(arg: 1, scope: !73, type: !30)
// CHECK:STDOUT: !76 = !DILocation(line: 28, column: 5, scope: !73)
// CHECK:STDOUT: !77 = !DILocation(line: 26, column: 3, scope: !73)
// CHECK:STDOUT: !78 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.4d6ceefeb7f772f4", scope: null, file: !1, line: 37, type: !79, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !81)
// CHECK:STDOUT: !79 = !DISubroutineType(types: !80)
// CHECK:STDOUT: !80 = !{null, !30, !30}
// CHECK:STDOUT: !81 = !{!82, !83}
// CHECK:STDOUT: !82 = !DILocalVariable(arg: 1, scope: !78, type: !30)
// CHECK:STDOUT: !83 = !DILocalVariable(arg: 2, scope: !78, type: !30)
// CHECK:STDOUT: !84 = !DILocation(line: 38, column: 3, scope: !78)
// CHECK:STDOUT: !85 = !DILocation(line: 37, column: 1, scope: !78)
// CHECK:STDOUT: !86 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.4d6ceefeb7f772f4", scope: null, file: !1, line: 48, type: !35, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !87)
// CHECK:STDOUT: !87 = !{!88, !89}
// CHECK:STDOUT: !88 = !DILocalVariable(arg: 1, scope: !86, type: !30)
// CHECK:STDOUT: !89 = !DILocalVariable(arg: 2, scope: !86, type: !30)
// CHECK:STDOUT: !90 = !DILocation(line: 49, column: 7, scope: !86)
// CHECK:STDOUT: !91 = !DILocation(line: 49, column: 6, scope: !86)
// CHECK:STDOUT: !92 = !DILocation(line: 50, column: 5, scope: !86)
// CHECK:STDOUT: !93 = !DILocation(line: 52, column: 7, scope: !86)
// CHECK:STDOUT: !94 = !DILocation(line: 52, column: 6, scope: !86)
// CHECK:STDOUT: !95 = !DILocation(line: 53, column: 12, scope: !86)
// CHECK:STDOUT: !96 = !DILocation(line: 53, column: 5, scope: !86)
// CHECK:STDOUT: !97 = !DILocation(line: 55, column: 12, scope: !86)
// CHECK:STDOUT: !98 = !DILocation(line: 55, column: 5, scope: !86)
// CHECK:STDOUT: !99 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.c5e4ba542a56b71f", scope: null, file: !1, line: 37, type: !100, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !102)
// CHECK:STDOUT: !100 = !DISubroutineType(types: !101)
// CHECK:STDOUT: !101 = !{null, !23, !30}
// CHECK:STDOUT: !102 = !{!103, !104}
// CHECK:STDOUT: !103 = !DILocalVariable(arg: 1, scope: !99, type: !23)
// CHECK:STDOUT: !104 = !DILocalVariable(arg: 2, scope: !99, type: !30)
// CHECK:STDOUT: !105 = !DILocation(line: 38, column: 3, scope: !99)
// CHECK:STDOUT: !106 = !DILocation(line: 37, column: 1, scope: !99)
// CHECK:STDOUT: !107 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.c5e4ba542a56b71f", scope: null, file: !1, line: 48, type: !44, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !108)
// CHECK:STDOUT: !108 = !{!109, !110}
// CHECK:STDOUT: !109 = !DILocalVariable(arg: 1, scope: !107, type: !23)
// CHECK:STDOUT: !110 = !DILocalVariable(arg: 2, scope: !107, type: !30)
// CHECK:STDOUT: !111 = !DILocation(line: 49, column: 7, scope: !107)
// CHECK:STDOUT: !112 = !DILocation(line: 49, column: 6, scope: !107)
// CHECK:STDOUT: !113 = !DILocation(line: 50, column: 5, scope: !107)
// CHECK:STDOUT: !114 = !DILocation(line: 52, column: 7, scope: !107)
// CHECK:STDOUT: !115 = !DILocation(line: 52, column: 6, scope: !107)
// CHECK:STDOUT: !116 = !DILocation(line: 53, column: 12, scope: !107)
// CHECK:STDOUT: !117 = !DILocation(line: 53, column: 5, scope: !107)
// CHECK:STDOUT: !118 = !DILocation(line: 55, column: 12, scope: !107)
// CHECK:STDOUT: !119 = !DILocation(line: 55, column: 5, scope: !107)
// CHECK:STDOUT: !120 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.a76b78461416b4a8", scope: null, file: !1, line: 37, type: !100, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !121)
// CHECK:STDOUT: !121 = !{!122, !123}
// CHECK:STDOUT: !122 = !DILocalVariable(arg: 1, scope: !120, type: !23)
// CHECK:STDOUT: !123 = !DILocalVariable(arg: 2, scope: !120, type: !30)
// CHECK:STDOUT: !124 = !DILocation(line: 38, column: 3, scope: !120)
// CHECK:STDOUT: !125 = !DILocation(line: 37, column: 1, scope: !120)
// CHECK:STDOUT: !126 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.a76b78461416b4a8", scope: null, file: !1, line: 48, type: !44, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !127)
// CHECK:STDOUT: !127 = !{!128, !129}
// CHECK:STDOUT: !128 = !DILocalVariable(arg: 1, scope: !126, type: !23)
// CHECK:STDOUT: !129 = !DILocalVariable(arg: 2, scope: !126, type: !30)
// CHECK:STDOUT: !130 = !DILocation(line: 49, column: 7, scope: !126)
// CHECK:STDOUT: !131 = !DILocation(line: 49, column: 6, scope: !126)
// CHECK:STDOUT: !132 = !DILocation(line: 50, column: 5, scope: !126)
// CHECK:STDOUT: !133 = !DILocation(line: 52, column: 7, scope: !126)
// CHECK:STDOUT: !134 = !DILocation(line: 52, column: 6, scope: !126)
// CHECK:STDOUT: !135 = !DILocation(line: 53, column: 12, scope: !126)
// CHECK:STDOUT: !136 = !DILocation(line: 53, column: 5, scope: !126)
// CHECK:STDOUT: !137 = !DILocation(line: 55, column: 12, scope: !126)
// CHECK:STDOUT: !138 = !DILocation(line: 55, column: 5, scope: !126)
// CHECK:STDOUT: !139 = distinct !DISubprogram(name: "C", linkageName: "_CC.Main.4d6ceefeb7f772f4", scope: null, file: !1, line: 41, type: !79, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !140)
// CHECK:STDOUT: !140 = !{!141, !142}
// CHECK:STDOUT: !141 = !DILocalVariable(arg: 1, scope: !139, type: !30)
// CHECK:STDOUT: !142 = !DILocalVariable(arg: 2, scope: !139, type: !30)
// CHECK:STDOUT: !143 = !DILocation(line: 42, column: 7, scope: !139)
// CHECK:STDOUT: !144 = !DILocation(line: 42, column: 6, scope: !139)
// CHECK:STDOUT: !145 = !DILocation(line: 43, column: 5, scope: !139)
// CHECK:STDOUT: !146 = !DILocation(line: 44, column: 10, scope: !139)
// CHECK:STDOUT: !147 = !DILocation(line: 44, column: 5, scope: !139)
// CHECK:STDOUT: !148 = !DILocation(line: 42, column: 3, scope: !139)
// CHECK:STDOUT: !149 = !DILocation(line: 41, column: 1, scope: !139)
// CHECK:STDOUT: !150 = distinct !DISubprogram(name: "E", linkageName: "_CE.Main.4d6ceefeb7f772f4", scope: null, file: !1, line: 61, type: !35, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !151)
// CHECK:STDOUT: !151 = !{!152, !153}
// CHECK:STDOUT: !152 = !DILocalVariable(arg: 1, scope: !150, type: !30)
// CHECK:STDOUT: !153 = !DILocalVariable(arg: 2, scope: !150, type: !30)
// CHECK:STDOUT: !154 = !DILocation(line: 62, column: 10, scope: !150)
// CHECK:STDOUT: !155 = !DILocation(line: 62, column: 3, scope: !150)
// CHECK:STDOUT: !156 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.4d6ceefeb7f772f4", scope: null, file: !1, line: 65, type: !35, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !157)
// CHECK:STDOUT: !157 = !{!158, !159}
// CHECK:STDOUT: !158 = !DILocalVariable(arg: 1, scope: !156, type: !30)
// CHECK:STDOUT: !159 = !DILocalVariable(arg: 2, scope: !156, type: !30)
// CHECK:STDOUT: !160 = !DILocation(line: 66, column: 10, scope: !156)
// CHECK:STDOUT: !161 = !DILocation(line: 66, column: 3, scope: !156)
// CHECK:STDOUT: !162 = distinct !DISubprogram(name: "C", linkageName: "_CC.Main.c5e4ba542a56b71f", scope: null, file: !1, line: 41, type: !100, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !163)
// CHECK:STDOUT: !163 = !{!164, !165}
// CHECK:STDOUT: !164 = !DILocalVariable(arg: 1, scope: !162, type: !23)
// CHECK:STDOUT: !165 = !DILocalVariable(arg: 2, scope: !162, type: !30)
// CHECK:STDOUT: !166 = !DILocation(line: 42, column: 7, scope: !162)
// CHECK:STDOUT: !167 = !DILocation(line: 42, column: 6, scope: !162)
// CHECK:STDOUT: !168 = !DILocation(line: 43, column: 5, scope: !162)
// CHECK:STDOUT: !169 = !DILocation(line: 44, column: 10, scope: !162)
// CHECK:STDOUT: !170 = !DILocation(line: 44, column: 5, scope: !162)
// CHECK:STDOUT: !171 = !DILocation(line: 42, column: 3, scope: !162)
// CHECK:STDOUT: !172 = !DILocation(line: 41, column: 1, scope: !162)
// CHECK:STDOUT: !173 = distinct !DISubprogram(name: "E", linkageName: "_CE.Main.c5e4ba542a56b71f", scope: null, file: !1, line: 61, type: !44, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !174)
// CHECK:STDOUT: !174 = !{!175, !176}
// CHECK:STDOUT: !175 = !DILocalVariable(arg: 1, scope: !173, type: !23)
// CHECK:STDOUT: !176 = !DILocalVariable(arg: 2, scope: !173, type: !30)
// CHECK:STDOUT: !177 = !DILocation(line: 62, column: 10, scope: !173)
// CHECK:STDOUT: !178 = !DILocation(line: 62, column: 3, scope: !173)
// CHECK:STDOUT: !179 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.c5e4ba542a56b71f", scope: null, file: !1, line: 65, type: !44, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !180)
// CHECK:STDOUT: !180 = !{!181, !182}
// CHECK:STDOUT: !181 = !DILocalVariable(arg: 1, scope: !179, type: !23)
// CHECK:STDOUT: !182 = !DILocalVariable(arg: 2, scope: !179, type: !30)
// CHECK:STDOUT: !183 = !DILocation(line: 66, column: 10, scope: !179)
// CHECK:STDOUT: !184 = !DILocation(line: 66, column: 3, scope: !179)
// CHECK:STDOUT: !185 = distinct !DISubprogram(name: "C", linkageName: "_CC.Main.a76b78461416b4a8", scope: null, file: !1, line: 41, type: !100, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !186)
// CHECK:STDOUT: !186 = !{!187, !188}
// CHECK:STDOUT: !187 = !DILocalVariable(arg: 1, scope: !185, type: !23)
// CHECK:STDOUT: !188 = !DILocalVariable(arg: 2, scope: !185, type: !30)
// CHECK:STDOUT: !189 = !DILocation(line: 42, column: 7, scope: !185)
// CHECK:STDOUT: !190 = !DILocation(line: 42, column: 6, scope: !185)
// CHECK:STDOUT: !191 = !DILocation(line: 43, column: 5, scope: !185)
// CHECK:STDOUT: !192 = !DILocation(line: 44, column: 10, scope: !185)
// CHECK:STDOUT: !193 = !DILocation(line: 44, column: 5, scope: !185)
// CHECK:STDOUT: !194 = !DILocation(line: 42, column: 3, scope: !185)
// CHECK:STDOUT: !195 = !DILocation(line: 41, column: 1, scope: !185)
// CHECK:STDOUT: !196 = distinct !DISubprogram(name: "E", linkageName: "_CE.Main.a76b78461416b4a8", scope: null, file: !1, line: 61, type: !44, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !197)
// CHECK:STDOUT: !197 = !{!198, !199}
// CHECK:STDOUT: !198 = !DILocalVariable(arg: 1, scope: !196, type: !23)
// CHECK:STDOUT: !199 = !DILocalVariable(arg: 2, scope: !196, type: !30)
// CHECK:STDOUT: !200 = !DILocation(line: 62, column: 10, scope: !196)
// CHECK:STDOUT: !201 = !DILocation(line: 62, column: 3, scope: !196)
// CHECK:STDOUT: !202 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.a76b78461416b4a8", scope: null, file: !1, line: 65, type: !44, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !203)
// CHECK:STDOUT: !203 = !{!204, !205}
// CHECK:STDOUT: !204 = !DILocalVariable(arg: 1, scope: !202, type: !23)
// CHECK:STDOUT: !205 = !DILocalVariable(arg: 2, scope: !202, type: !30)
// CHECK:STDOUT: !206 = !DILocation(line: 66, column: 10, scope: !202)
// CHECK:STDOUT: !207 = !DILocation(line: 66, column: 3, scope: !202)
// CHECK:STDOUT: !208 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.4d6ceefeb7f772f4", scope: null, file: !1, line: 69, type: !35, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !209)
// CHECK:STDOUT: !209 = !{!210, !211}
// CHECK:STDOUT: !210 = !DILocalVariable(arg: 1, scope: !208, type: !30)
// CHECK:STDOUT: !211 = !DILocalVariable(arg: 2, scope: !208, type: !30)
// CHECK:STDOUT: !212 = !DILocation(line: 70, column: 15, scope: !208)
// CHECK:STDOUT: !213 = !DILocation(line: 70, column: 10, scope: !208)
// CHECK:STDOUT: !214 = !DILocation(line: 70, column: 3, scope: !208)
// CHECK:STDOUT: !215 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.c5e4ba542a56b71f", scope: null, file: !1, line: 69, type: !44, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !216)
// CHECK:STDOUT: !216 = !{!217, !218}
// CHECK:STDOUT: !217 = !DILocalVariable(arg: 1, scope: !215, type: !23)
// CHECK:STDOUT: !218 = !DILocalVariable(arg: 2, scope: !215, type: !30)
// CHECK:STDOUT: !219 = !DILocation(line: 70, column: 15, scope: !215)
// CHECK:STDOUT: !220 = !DILocation(line: 70, column: 10, scope: !215)
// CHECK:STDOUT: !221 = !DILocation(line: 70, column: 3, scope: !215)
// CHECK:STDOUT: !222 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.a76b78461416b4a8", scope: null, file: !1, line: 69, type: !44, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !223)
// CHECK:STDOUT: !223 = !{!224, !225}
// CHECK:STDOUT: !224 = !DILocalVariable(arg: 1, scope: !222, type: !23)
// CHECK:STDOUT: !225 = !DILocalVariable(arg: 2, scope: !222, type: !30)
// CHECK:STDOUT: !226 = !DILocation(line: 70, column: 15, scope: !222)
// CHECK:STDOUT: !227 = !DILocation(line: 70, column: 10, scope: !222)
// CHECK:STDOUT: !228 = !DILocation(line: 70, column: 3, scope: !222)
// CHECK:STDOUT:
