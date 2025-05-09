// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/generic/call_recursive_sccs_deep.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/generic/call_recursive_sccs_deep.carbon

import Core library "io";
fn A[T:! type](x: T, count: i32) -> T;
fn B[T:! type](x: T, count: i32);
fn C[T:! type](x: T, count: i32);
fn D[T:! type](x: T, count: i32) -> T;
fn E[T:! type](x: T, count: i32) -> T;
fn F[T:! type](x: T, count: i32) -> T;

// Builds on `call_recursive_mutual.carbon` and `call_recursive_diamond.carbon`
// B-C form a mutually recursive SCC (strongly connected component), D-E-F-G
// form a diamond SCC.
// The two specifics for each of A, B, C, D, E, F, G with a pointer type could
// be deduplicated.
// E and F are also equivalent in the diamond, even though they are different
// generics. There is potential front-end ICF (identical code folding)
// optimization here, if the function fingerprint may infer deduplication
// when not including the two different generic_ids.

fn A[T:! type](x: T, count: i32) -> T {
  B(x, count);
  return D(x, count);
}

fn B[T:! type](x: T, count: i32) {
  C(x, count);
}

fn C[T:! type](x: T, count: i32) {
  if (count <= 2) {
    Core.Print(count);
    B(x, count + 1);
  }
}

fn D[T:! type](x: T, count: i32) -> T {
  if (count > 4) {
    return x;
  }
  if (count % 2 == 0) {
    return E(x, count);
  } else {
    return F(x, count);
  }
}

fn G[T:! type](x: T, count: i32) -> T;

fn E[T:! type](x: T, count: i32) -> T {
  return G(x, count);
}

fn F[T:! type](x: T, count: i32) -> T {
  return G(x, count);
}

fn G[T:! type](x: T, count: i32) -> T {
  return D(x, count + 1);
}


fn M() {
  var n: i32 = 0;
  var m: f64 = 1.0;
  var ptr_i32 : i32*;
  var ptr_f64 : f64*;

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
// CHECK:STDOUT: define void @_CM.Main() !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %n.var = alloca i32, align 4, !dbg !7
// CHECK:STDOUT:   %m.var = alloca double, align 8, !dbg !8
// CHECK:STDOUT:   %ptr_i32.var = alloca ptr, align 8, !dbg !9
// CHECK:STDOUT:   %ptr_f64.var = alloca ptr, align 8, !dbg !10
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 4, ptr %n.var), !dbg !7
// CHECK:STDOUT:   store i32 0, ptr %n.var, align 4, !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 8, ptr %m.var), !dbg !8
// CHECK:STDOUT:   store double 1.000000e+00, ptr %m.var, align 8, !dbg !8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 8, ptr %ptr_i32.var), !dbg !9
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 8, ptr %ptr_f64.var), !dbg !10
// CHECK:STDOUT:   %.loc77_5 = load i32, ptr %n.var, align 4, !dbg !11
// CHECK:STDOUT:   %A.call.loc77 = call i32 @_CA.Main.b88d1103f417c6d4(i32 %.loc77_5, i32 0), !dbg !12
// CHECK:STDOUT:   %.loc78_5 = load double, ptr %m.var, align 8, !dbg !13
// CHECK:STDOUT:   %A.call.loc78 = call double @_CA.Main.66be507887ceee78(double %.loc78_5, i32 0), !dbg !14
// CHECK:STDOUT:   %.loc79_5 = load ptr, ptr %ptr_i32.var, align 8, !dbg !15
// CHECK:STDOUT:   %A.call.loc79 = call ptr @_CA.Main.e8193710fd35b608(ptr %.loc79_5, i32 0), !dbg !16
// CHECK:STDOUT:   %.loc80_5 = load ptr, ptr %ptr_f64.var, align 8, !dbg !17
// CHECK:STDOUT:   %A.call.loc80 = call ptr @_CA.Main.04bf2edaaa84aa22(ptr %.loc80_5, i32 0), !dbg !18
// CHECK:STDOUT:   ret void, !dbg !19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(i64 immarg, ptr captures(none)) #0
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_CA.Main.b88d1103f417c6d4(i32 %x, i32 %count) !dbg !20 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CB.Main.b88d1103f417c6d4(i32 %x, i32 %count), !dbg !21
// CHECK:STDOUT:   %D.call = call i32 @_CD.Main.b88d1103f417c6d4(i32 %x, i32 %count), !dbg !22
// CHECK:STDOUT:   ret i32 %D.call, !dbg !23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define double @_CA.Main.66be507887ceee78(double %x, i32 %count) !dbg !24 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CB.Main.66be507887ceee78(double %x, i32 %count), !dbg !25
// CHECK:STDOUT:   %D.call = call double @_CD.Main.66be507887ceee78(double %x, i32 %count), !dbg !26
// CHECK:STDOUT:   ret double %D.call, !dbg !27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define ptr @_CA.Main.e8193710fd35b608(ptr %x, i32 %count) !dbg !28 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CB.Main.e8193710fd35b608(ptr %x, i32 %count), !dbg !29
// CHECK:STDOUT:   %D.call = call ptr @_CD.Main.e8193710fd35b608(ptr %x, i32 %count), !dbg !30
// CHECK:STDOUT:   ret ptr %D.call, !dbg !31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define ptr @_CA.Main.04bf2edaaa84aa22(ptr %x, i32 %count) !dbg !32 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CB.Main.04bf2edaaa84aa22(ptr %x, i32 %count), !dbg !33
// CHECK:STDOUT:   %D.call = call ptr @_CD.Main.04bf2edaaa84aa22(ptr %x, i32 %count), !dbg !34
// CHECK:STDOUT:   ret ptr %D.call, !dbg !35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CB.Main.b88d1103f417c6d4(i32 %x, i32 %count) !dbg !36 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CC.Main.b88d1103f417c6d4(i32 %x, i32 %count), !dbg !37
// CHECK:STDOUT:   ret void, !dbg !38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_CD.Main.b88d1103f417c6d4(i32 %x, i32 %count) !dbg !39 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.greater = icmp sgt i32 %count, 4, !dbg !40
// CHECK:STDOUT:   br i1 %int.greater, label %if.then.loc46, label %if.else.loc46, !dbg !41
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc46:                                    ; preds = %entry
// CHECK:STDOUT:   ret i32 %x, !dbg !42
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc46:                                    ; preds = %entry
// CHECK:STDOUT:   %int.smod = srem i32 %count, 2, !dbg !43
// CHECK:STDOUT:   %int.eq = icmp eq i32 %int.smod, 0, !dbg !43
// CHECK:STDOUT:   br i1 %int.eq, label %if.then.loc49, label %if.else.loc49, !dbg !44
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc49:                                    ; preds = %if.else.loc46
// CHECK:STDOUT:   %E.call = call i32 @_CE.Main.b88d1103f417c6d4(i32 %x, i32 %count), !dbg !45
// CHECK:STDOUT:   ret i32 %E.call, !dbg !46
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc49:                                    ; preds = %if.else.loc46
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main.b88d1103f417c6d4(i32 %x, i32 %count), !dbg !47
// CHECK:STDOUT:   ret i32 %F.call, !dbg !48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CB.Main.66be507887ceee78(double %x, i32 %count) !dbg !49 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CC.Main.66be507887ceee78(double %x, i32 %count), !dbg !50
// CHECK:STDOUT:   ret void, !dbg !51
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define double @_CD.Main.66be507887ceee78(double %x, i32 %count) !dbg !52 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.greater = icmp sgt i32 %count, 4, !dbg !53
// CHECK:STDOUT:   br i1 %int.greater, label %if.then.loc46, label %if.else.loc46, !dbg !54
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc46:                                    ; preds = %entry
// CHECK:STDOUT:   ret double %x, !dbg !55
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc46:                                    ; preds = %entry
// CHECK:STDOUT:   %int.smod = srem i32 %count, 2, !dbg !56
// CHECK:STDOUT:   %int.eq = icmp eq i32 %int.smod, 0, !dbg !56
// CHECK:STDOUT:   br i1 %int.eq, label %if.then.loc49, label %if.else.loc49, !dbg !57
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc49:                                    ; preds = %if.else.loc46
// CHECK:STDOUT:   %E.call = call double @_CE.Main.66be507887ceee78(double %x, i32 %count), !dbg !58
// CHECK:STDOUT:   ret double %E.call, !dbg !59
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc49:                                    ; preds = %if.else.loc46
// CHECK:STDOUT:   %F.call = call double @_CF.Main.66be507887ceee78(double %x, i32 %count), !dbg !60
// CHECK:STDOUT:   ret double %F.call, !dbg !61
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CB.Main.e8193710fd35b608(ptr %x, i32 %count) !dbg !62 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CC.Main.e8193710fd35b608(ptr %x, i32 %count), !dbg !63
// CHECK:STDOUT:   ret void, !dbg !64
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define ptr @_CD.Main.e8193710fd35b608(ptr %x, i32 %count) !dbg !65 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.greater = icmp sgt i32 %count, 4, !dbg !66
// CHECK:STDOUT:   br i1 %int.greater, label %if.then.loc46, label %if.else.loc46, !dbg !67
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc46:                                    ; preds = %entry
// CHECK:STDOUT:   ret ptr %x, !dbg !68
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc46:                                    ; preds = %entry
// CHECK:STDOUT:   %int.smod = srem i32 %count, 2, !dbg !69
// CHECK:STDOUT:   %int.eq = icmp eq i32 %int.smod, 0, !dbg !69
// CHECK:STDOUT:   br i1 %int.eq, label %if.then.loc49, label %if.else.loc49, !dbg !70
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc49:                                    ; preds = %if.else.loc46
// CHECK:STDOUT:   %E.call = call ptr @_CE.Main.e8193710fd35b608(ptr %x, i32 %count), !dbg !71
// CHECK:STDOUT:   ret ptr %E.call, !dbg !72
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc49:                                    ; preds = %if.else.loc46
// CHECK:STDOUT:   %F.call = call ptr @_CF.Main.e8193710fd35b608(ptr %x, i32 %count), !dbg !73
// CHECK:STDOUT:   ret ptr %F.call, !dbg !74
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CB.Main.04bf2edaaa84aa22(ptr %x, i32 %count) !dbg !75 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CC.Main.04bf2edaaa84aa22(ptr %x, i32 %count), !dbg !76
// CHECK:STDOUT:   ret void, !dbg !77
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define ptr @_CD.Main.04bf2edaaa84aa22(ptr %x, i32 %count) !dbg !78 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.greater = icmp sgt i32 %count, 4, !dbg !79
// CHECK:STDOUT:   br i1 %int.greater, label %if.then.loc46, label %if.else.loc46, !dbg !80
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc46:                                    ; preds = %entry
// CHECK:STDOUT:   ret ptr %x, !dbg !81
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc46:                                    ; preds = %entry
// CHECK:STDOUT:   %int.smod = srem i32 %count, 2, !dbg !82
// CHECK:STDOUT:   %int.eq = icmp eq i32 %int.smod, 0, !dbg !82
// CHECK:STDOUT:   br i1 %int.eq, label %if.then.loc49, label %if.else.loc49, !dbg !83
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc49:                                    ; preds = %if.else.loc46
// CHECK:STDOUT:   %E.call = call ptr @_CE.Main.04bf2edaaa84aa22(ptr %x, i32 %count), !dbg !84
// CHECK:STDOUT:   ret ptr %E.call, !dbg !85
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc49:                                    ; preds = %if.else.loc46
// CHECK:STDOUT:   %F.call = call ptr @_CF.Main.04bf2edaaa84aa22(ptr %x, i32 %count), !dbg !86
// CHECK:STDOUT:   ret ptr %F.call, !dbg !87
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CC.Main.b88d1103f417c6d4(i32 %x, i32 %count) !dbg !88 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.less_eq = icmp sle i32 %count, 2, !dbg !89
// CHECK:STDOUT:   br i1 %int.less_eq, label %if.then, label %if.else, !dbg !90
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   %print.int = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 %count), !dbg !91
// CHECK:STDOUT:   %int.sadd = add i32 %count, 1, !dbg !92
// CHECK:STDOUT:   call void @_CB.Main.b88d1103f417c6d4(i32 %x, i32 %int.sadd), !dbg !93
// CHECK:STDOUT:   br label %if.else, !dbg !94
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %if.then, %entry
// CHECK:STDOUT:   ret void, !dbg !95
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_CE.Main.b88d1103f417c6d4(i32 %x, i32 %count) !dbg !96 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call i32 @_CG.Main.b88d1103f417c6d4(i32 %x, i32 %count), !dbg !97
// CHECK:STDOUT:   ret i32 %G.call, !dbg !98
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_CF.Main.b88d1103f417c6d4(i32 %x, i32 %count) !dbg !99 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call i32 @_CG.Main.b88d1103f417c6d4(i32 %x, i32 %count), !dbg !100
// CHECK:STDOUT:   ret i32 %G.call, !dbg !101
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CC.Main.66be507887ceee78(double %x, i32 %count) !dbg !102 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.less_eq = icmp sle i32 %count, 2, !dbg !103
// CHECK:STDOUT:   br i1 %int.less_eq, label %if.then, label %if.else, !dbg !104
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   %print.int = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 %count), !dbg !105
// CHECK:STDOUT:   %int.sadd = add i32 %count, 1, !dbg !106
// CHECK:STDOUT:   call void @_CB.Main.66be507887ceee78(double %x, i32 %int.sadd), !dbg !107
// CHECK:STDOUT:   br label %if.else, !dbg !108
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %if.then, %entry
// CHECK:STDOUT:   ret void, !dbg !109
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define double @_CE.Main.66be507887ceee78(double %x, i32 %count) !dbg !110 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call double @_CG.Main.66be507887ceee78(double %x, i32 %count), !dbg !111
// CHECK:STDOUT:   ret double %G.call, !dbg !112
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define double @_CF.Main.66be507887ceee78(double %x, i32 %count) !dbg !113 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call double @_CG.Main.66be507887ceee78(double %x, i32 %count), !dbg !114
// CHECK:STDOUT:   ret double %G.call, !dbg !115
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CC.Main.e8193710fd35b608(ptr %x, i32 %count) !dbg !116 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.less_eq = icmp sle i32 %count, 2, !dbg !117
// CHECK:STDOUT:   br i1 %int.less_eq, label %if.then, label %if.else, !dbg !118
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   %print.int = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 %count), !dbg !119
// CHECK:STDOUT:   %int.sadd = add i32 %count, 1, !dbg !120
// CHECK:STDOUT:   call void @_CB.Main.e8193710fd35b608(ptr %x, i32 %int.sadd), !dbg !121
// CHECK:STDOUT:   br label %if.else, !dbg !122
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %if.then, %entry
// CHECK:STDOUT:   ret void, !dbg !123
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define ptr @_CE.Main.e8193710fd35b608(ptr %x, i32 %count) !dbg !124 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call ptr @_CG.Main.e8193710fd35b608(ptr %x, i32 %count), !dbg !125
// CHECK:STDOUT:   ret ptr %G.call, !dbg !126
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define ptr @_CF.Main.e8193710fd35b608(ptr %x, i32 %count) !dbg !127 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call ptr @_CG.Main.e8193710fd35b608(ptr %x, i32 %count), !dbg !128
// CHECK:STDOUT:   ret ptr %G.call, !dbg !129
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CC.Main.04bf2edaaa84aa22(ptr %x, i32 %count) !dbg !130 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.less_eq = icmp sle i32 %count, 2, !dbg !131
// CHECK:STDOUT:   br i1 %int.less_eq, label %if.then, label %if.else, !dbg !132
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   %print.int = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 %count), !dbg !133
// CHECK:STDOUT:   %int.sadd = add i32 %count, 1, !dbg !134
// CHECK:STDOUT:   call void @_CB.Main.04bf2edaaa84aa22(ptr %x, i32 %int.sadd), !dbg !135
// CHECK:STDOUT:   br label %if.else, !dbg !136
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %if.then, %entry
// CHECK:STDOUT:   ret void, !dbg !137
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define ptr @_CE.Main.04bf2edaaa84aa22(ptr %x, i32 %count) !dbg !138 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call ptr @_CG.Main.04bf2edaaa84aa22(ptr %x, i32 %count), !dbg !139
// CHECK:STDOUT:   ret ptr %G.call, !dbg !140
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define ptr @_CF.Main.04bf2edaaa84aa22(ptr %x, i32 %count) !dbg !141 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call ptr @_CG.Main.04bf2edaaa84aa22(ptr %x, i32 %count), !dbg !142
// CHECK:STDOUT:   ret ptr %G.call, !dbg !143
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @printf(ptr, ...)
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_CG.Main.b88d1103f417c6d4(i32 %x, i32 %count) !dbg !144 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.sadd = add i32 %count, 1, !dbg !145
// CHECK:STDOUT:   %D.call = call i32 @_CD.Main.b88d1103f417c6d4(i32 %x, i32 %int.sadd), !dbg !146
// CHECK:STDOUT:   ret i32 %D.call, !dbg !147
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define double @_CG.Main.66be507887ceee78(double %x, i32 %count) !dbg !148 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.sadd = add i32 %count, 1, !dbg !149
// CHECK:STDOUT:   %D.call = call double @_CD.Main.66be507887ceee78(double %x, i32 %int.sadd), !dbg !150
// CHECK:STDOUT:   ret double %D.call, !dbg !151
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define ptr @_CG.Main.e8193710fd35b608(ptr %x, i32 %count) !dbg !152 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.sadd = add i32 %count, 1, !dbg !153
// CHECK:STDOUT:   %D.call = call ptr @_CD.Main.e8193710fd35b608(ptr %x, i32 %int.sadd), !dbg !154
// CHECK:STDOUT:   ret ptr %D.call, !dbg !155
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define ptr @_CG.Main.04bf2edaaa84aa22(ptr %x, i32 %count) !dbg !156 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.sadd = add i32 %count, 1, !dbg !157
// CHECK:STDOUT:   %D.call = call ptr @_CD.Main.04bf2edaaa84aa22(ptr %x, i32 %int.sadd), !dbg !158
// CHECK:STDOUT:   ret ptr %D.call, !dbg !159
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @printf, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CG.Main.b88d1103f417c6d4, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CG.Main.66be507887ceee78, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CG.Main.e8193710fd35b608, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CG.Main.04bf2edaaa84aa22, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "call_recursive_sccs_deep.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "M", linkageName: "_CM.Main", scope: null, file: !3, line: 71, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 72, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 73, column: 3, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 74, column: 3, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 75, column: 3, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 77, column: 5, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 77, column: 3, scope: !4)
// CHECK:STDOUT: !13 = !DILocation(line: 78, column: 5, scope: !4)
// CHECK:STDOUT: !14 = !DILocation(line: 78, column: 3, scope: !4)
// CHECK:STDOUT: !15 = !DILocation(line: 79, column: 5, scope: !4)
// CHECK:STDOUT: !16 = !DILocation(line: 79, column: 3, scope: !4)
// CHECK:STDOUT: !17 = !DILocation(line: 80, column: 5, scope: !4)
// CHECK:STDOUT: !18 = !DILocation(line: 80, column: 3, scope: !4)
// CHECK:STDOUT: !19 = !DILocation(line: 71, column: 1, scope: !4)
// CHECK:STDOUT: !20 = distinct !DISubprogram(name: "A", linkageName: "_CA.Main.b88d1103f417c6d4", scope: null, file: !3, line: 29, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !21 = !DILocation(line: 30, column: 3, scope: !20)
// CHECK:STDOUT: !22 = !DILocation(line: 31, column: 10, scope: !20)
// CHECK:STDOUT: !23 = !DILocation(line: 31, column: 3, scope: !20)
// CHECK:STDOUT: !24 = distinct !DISubprogram(name: "A", linkageName: "_CA.Main.66be507887ceee78", scope: null, file: !3, line: 29, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !25 = !DILocation(line: 30, column: 3, scope: !24)
// CHECK:STDOUT: !26 = !DILocation(line: 31, column: 10, scope: !24)
// CHECK:STDOUT: !27 = !DILocation(line: 31, column: 3, scope: !24)
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "A", linkageName: "_CA.Main.e8193710fd35b608", scope: null, file: !3, line: 29, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !29 = !DILocation(line: 30, column: 3, scope: !28)
// CHECK:STDOUT: !30 = !DILocation(line: 31, column: 10, scope: !28)
// CHECK:STDOUT: !31 = !DILocation(line: 31, column: 3, scope: !28)
// CHECK:STDOUT: !32 = distinct !DISubprogram(name: "A", linkageName: "_CA.Main.04bf2edaaa84aa22", scope: null, file: !3, line: 29, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !33 = !DILocation(line: 30, column: 3, scope: !32)
// CHECK:STDOUT: !34 = !DILocation(line: 31, column: 10, scope: !32)
// CHECK:STDOUT: !35 = !DILocation(line: 31, column: 3, scope: !32)
// CHECK:STDOUT: !36 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.b88d1103f417c6d4", scope: null, file: !3, line: 34, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !37 = !DILocation(line: 35, column: 3, scope: !36)
// CHECK:STDOUT: !38 = !DILocation(line: 34, column: 1, scope: !36)
// CHECK:STDOUT: !39 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.b88d1103f417c6d4", scope: null, file: !3, line: 45, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !40 = !DILocation(line: 46, column: 7, scope: !39)
// CHECK:STDOUT: !41 = !DILocation(line: 46, column: 6, scope: !39)
// CHECK:STDOUT: !42 = !DILocation(line: 47, column: 5, scope: !39)
// CHECK:STDOUT: !43 = !DILocation(line: 49, column: 7, scope: !39)
// CHECK:STDOUT: !44 = !DILocation(line: 49, column: 6, scope: !39)
// CHECK:STDOUT: !45 = !DILocation(line: 50, column: 12, scope: !39)
// CHECK:STDOUT: !46 = !DILocation(line: 50, column: 5, scope: !39)
// CHECK:STDOUT: !47 = !DILocation(line: 52, column: 12, scope: !39)
// CHECK:STDOUT: !48 = !DILocation(line: 52, column: 5, scope: !39)
// CHECK:STDOUT: !49 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.66be507887ceee78", scope: null, file: !3, line: 34, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !50 = !DILocation(line: 35, column: 3, scope: !49)
// CHECK:STDOUT: !51 = !DILocation(line: 34, column: 1, scope: !49)
// CHECK:STDOUT: !52 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.66be507887ceee78", scope: null, file: !3, line: 45, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !53 = !DILocation(line: 46, column: 7, scope: !52)
// CHECK:STDOUT: !54 = !DILocation(line: 46, column: 6, scope: !52)
// CHECK:STDOUT: !55 = !DILocation(line: 47, column: 5, scope: !52)
// CHECK:STDOUT: !56 = !DILocation(line: 49, column: 7, scope: !52)
// CHECK:STDOUT: !57 = !DILocation(line: 49, column: 6, scope: !52)
// CHECK:STDOUT: !58 = !DILocation(line: 50, column: 12, scope: !52)
// CHECK:STDOUT: !59 = !DILocation(line: 50, column: 5, scope: !52)
// CHECK:STDOUT: !60 = !DILocation(line: 52, column: 12, scope: !52)
// CHECK:STDOUT: !61 = !DILocation(line: 52, column: 5, scope: !52)
// CHECK:STDOUT: !62 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.e8193710fd35b608", scope: null, file: !3, line: 34, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !63 = !DILocation(line: 35, column: 3, scope: !62)
// CHECK:STDOUT: !64 = !DILocation(line: 34, column: 1, scope: !62)
// CHECK:STDOUT: !65 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.e8193710fd35b608", scope: null, file: !3, line: 45, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !66 = !DILocation(line: 46, column: 7, scope: !65)
// CHECK:STDOUT: !67 = !DILocation(line: 46, column: 6, scope: !65)
// CHECK:STDOUT: !68 = !DILocation(line: 47, column: 5, scope: !65)
// CHECK:STDOUT: !69 = !DILocation(line: 49, column: 7, scope: !65)
// CHECK:STDOUT: !70 = !DILocation(line: 49, column: 6, scope: !65)
// CHECK:STDOUT: !71 = !DILocation(line: 50, column: 12, scope: !65)
// CHECK:STDOUT: !72 = !DILocation(line: 50, column: 5, scope: !65)
// CHECK:STDOUT: !73 = !DILocation(line: 52, column: 12, scope: !65)
// CHECK:STDOUT: !74 = !DILocation(line: 52, column: 5, scope: !65)
// CHECK:STDOUT: !75 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.04bf2edaaa84aa22", scope: null, file: !3, line: 34, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !76 = !DILocation(line: 35, column: 3, scope: !75)
// CHECK:STDOUT: !77 = !DILocation(line: 34, column: 1, scope: !75)
// CHECK:STDOUT: !78 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.04bf2edaaa84aa22", scope: null, file: !3, line: 45, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !79 = !DILocation(line: 46, column: 7, scope: !78)
// CHECK:STDOUT: !80 = !DILocation(line: 46, column: 6, scope: !78)
// CHECK:STDOUT: !81 = !DILocation(line: 47, column: 5, scope: !78)
// CHECK:STDOUT: !82 = !DILocation(line: 49, column: 7, scope: !78)
// CHECK:STDOUT: !83 = !DILocation(line: 49, column: 6, scope: !78)
// CHECK:STDOUT: !84 = !DILocation(line: 50, column: 12, scope: !78)
// CHECK:STDOUT: !85 = !DILocation(line: 50, column: 5, scope: !78)
// CHECK:STDOUT: !86 = !DILocation(line: 52, column: 12, scope: !78)
// CHECK:STDOUT: !87 = !DILocation(line: 52, column: 5, scope: !78)
// CHECK:STDOUT: !88 = distinct !DISubprogram(name: "C", linkageName: "_CC.Main.b88d1103f417c6d4", scope: null, file: !3, line: 38, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !89 = !DILocation(line: 39, column: 7, scope: !88)
// CHECK:STDOUT: !90 = !DILocation(line: 39, column: 6, scope: !88)
// CHECK:STDOUT: !91 = !DILocation(line: 40, column: 5, scope: !88)
// CHECK:STDOUT: !92 = !DILocation(line: 41, column: 10, scope: !88)
// CHECK:STDOUT: !93 = !DILocation(line: 41, column: 5, scope: !88)
// CHECK:STDOUT: !94 = !DILocation(line: 39, column: 3, scope: !88)
// CHECK:STDOUT: !95 = !DILocation(line: 38, column: 1, scope: !88)
// CHECK:STDOUT: !96 = distinct !DISubprogram(name: "E", linkageName: "_CE.Main.b88d1103f417c6d4", scope: null, file: !3, line: 58, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !97 = !DILocation(line: 59, column: 10, scope: !96)
// CHECK:STDOUT: !98 = !DILocation(line: 59, column: 3, scope: !96)
// CHECK:STDOUT: !99 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.b88d1103f417c6d4", scope: null, file: !3, line: 62, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !100 = !DILocation(line: 63, column: 10, scope: !99)
// CHECK:STDOUT: !101 = !DILocation(line: 63, column: 3, scope: !99)
// CHECK:STDOUT: !102 = distinct !DISubprogram(name: "C", linkageName: "_CC.Main.66be507887ceee78", scope: null, file: !3, line: 38, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !103 = !DILocation(line: 39, column: 7, scope: !102)
// CHECK:STDOUT: !104 = !DILocation(line: 39, column: 6, scope: !102)
// CHECK:STDOUT: !105 = !DILocation(line: 40, column: 5, scope: !102)
// CHECK:STDOUT: !106 = !DILocation(line: 41, column: 10, scope: !102)
// CHECK:STDOUT: !107 = !DILocation(line: 41, column: 5, scope: !102)
// CHECK:STDOUT: !108 = !DILocation(line: 39, column: 3, scope: !102)
// CHECK:STDOUT: !109 = !DILocation(line: 38, column: 1, scope: !102)
// CHECK:STDOUT: !110 = distinct !DISubprogram(name: "E", linkageName: "_CE.Main.66be507887ceee78", scope: null, file: !3, line: 58, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !111 = !DILocation(line: 59, column: 10, scope: !110)
// CHECK:STDOUT: !112 = !DILocation(line: 59, column: 3, scope: !110)
// CHECK:STDOUT: !113 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.66be507887ceee78", scope: null, file: !3, line: 62, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !114 = !DILocation(line: 63, column: 10, scope: !113)
// CHECK:STDOUT: !115 = !DILocation(line: 63, column: 3, scope: !113)
// CHECK:STDOUT: !116 = distinct !DISubprogram(name: "C", linkageName: "_CC.Main.e8193710fd35b608", scope: null, file: !3, line: 38, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !117 = !DILocation(line: 39, column: 7, scope: !116)
// CHECK:STDOUT: !118 = !DILocation(line: 39, column: 6, scope: !116)
// CHECK:STDOUT: !119 = !DILocation(line: 40, column: 5, scope: !116)
// CHECK:STDOUT: !120 = !DILocation(line: 41, column: 10, scope: !116)
// CHECK:STDOUT: !121 = !DILocation(line: 41, column: 5, scope: !116)
// CHECK:STDOUT: !122 = !DILocation(line: 39, column: 3, scope: !116)
// CHECK:STDOUT: !123 = !DILocation(line: 38, column: 1, scope: !116)
// CHECK:STDOUT: !124 = distinct !DISubprogram(name: "E", linkageName: "_CE.Main.e8193710fd35b608", scope: null, file: !3, line: 58, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !125 = !DILocation(line: 59, column: 10, scope: !124)
// CHECK:STDOUT: !126 = !DILocation(line: 59, column: 3, scope: !124)
// CHECK:STDOUT: !127 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.e8193710fd35b608", scope: null, file: !3, line: 62, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !128 = !DILocation(line: 63, column: 10, scope: !127)
// CHECK:STDOUT: !129 = !DILocation(line: 63, column: 3, scope: !127)
// CHECK:STDOUT: !130 = distinct !DISubprogram(name: "C", linkageName: "_CC.Main.04bf2edaaa84aa22", scope: null, file: !3, line: 38, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !131 = !DILocation(line: 39, column: 7, scope: !130)
// CHECK:STDOUT: !132 = !DILocation(line: 39, column: 6, scope: !130)
// CHECK:STDOUT: !133 = !DILocation(line: 40, column: 5, scope: !130)
// CHECK:STDOUT: !134 = !DILocation(line: 41, column: 10, scope: !130)
// CHECK:STDOUT: !135 = !DILocation(line: 41, column: 5, scope: !130)
// CHECK:STDOUT: !136 = !DILocation(line: 39, column: 3, scope: !130)
// CHECK:STDOUT: !137 = !DILocation(line: 38, column: 1, scope: !130)
// CHECK:STDOUT: !138 = distinct !DISubprogram(name: "E", linkageName: "_CE.Main.04bf2edaaa84aa22", scope: null, file: !3, line: 58, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !139 = !DILocation(line: 59, column: 10, scope: !138)
// CHECK:STDOUT: !140 = !DILocation(line: 59, column: 3, scope: !138)
// CHECK:STDOUT: !141 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.04bf2edaaa84aa22", scope: null, file: !3, line: 62, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !142 = !DILocation(line: 63, column: 10, scope: !141)
// CHECK:STDOUT: !143 = !DILocation(line: 63, column: 3, scope: !141)
// CHECK:STDOUT: !144 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.b88d1103f417c6d4", scope: null, file: !3, line: 66, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !145 = !DILocation(line: 67, column: 15, scope: !144)
// CHECK:STDOUT: !146 = !DILocation(line: 67, column: 10, scope: !144)
// CHECK:STDOUT: !147 = !DILocation(line: 67, column: 3, scope: !144)
// CHECK:STDOUT: !148 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.66be507887ceee78", scope: null, file: !3, line: 66, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !149 = !DILocation(line: 67, column: 15, scope: !148)
// CHECK:STDOUT: !150 = !DILocation(line: 67, column: 10, scope: !148)
// CHECK:STDOUT: !151 = !DILocation(line: 67, column: 3, scope: !148)
// CHECK:STDOUT: !152 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.e8193710fd35b608", scope: null, file: !3, line: 66, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !153 = !DILocation(line: 67, column: 15, scope: !152)
// CHECK:STDOUT: !154 = !DILocation(line: 67, column: 10, scope: !152)
// CHECK:STDOUT: !155 = !DILocation(line: 67, column: 3, scope: !152)
// CHECK:STDOUT: !156 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.04bf2edaaa84aa22", scope: null, file: !3, line: 66, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !157 = !DILocation(line: 67, column: 15, scope: !156)
// CHECK:STDOUT: !158 = !DILocation(line: 67, column: 10, scope: !156)
// CHECK:STDOUT: !159 = !DILocation(line: 67, column: 3, scope: !156)
