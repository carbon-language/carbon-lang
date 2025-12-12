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
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_f64.var), !dbg !10
// CHECK:STDOUT:   %.loc79_5 = load i32, ptr %n.var, align 4, !dbg !11
// CHECK:STDOUT:   %A.call.loc79 = call i32 @_CA.Main.3440082c84c3c17b(i32 %.loc79_5, i32 0), !dbg !12
// CHECK:STDOUT:   %.loc80_5 = load double, ptr %m.var, align 8, !dbg !13
// CHECK:STDOUT:   %A.call.loc80 = call double @_CA.Main.286b0f7890e5304c(double %.loc80_5, i32 0), !dbg !14
// CHECK:STDOUT:   %.loc81_5 = load ptr, ptr %ptr_i32.var, align 8, !dbg !15
// CHECK:STDOUT:   %A.call.loc81 = call ptr @_CA.Main.4b30cda44e16b9ec(ptr %.loc81_5, i32 0), !dbg !16
// CHECK:STDOUT:   %.loc82_5 = load ptr, ptr %ptr_f64.var, align 8, !dbg !17
// CHECK:STDOUT:   %A.call.loc82 = call ptr @_CA.Main.4b30cda44e16b9ec(ptr %.loc82_5, i32 0), !dbg !18
// CHECK:STDOUT:   ret void, !dbg !19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CA.Main.3440082c84c3c17b(i32 %x, i32 %count) #0 !dbg !20 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CB.Main.3440082c84c3c17b(i32 %x, i32 %count), !dbg !27
// CHECK:STDOUT:   %D.call = call i32 @_CD.Main.3440082c84c3c17b(i32 %x, i32 %count), !dbg !28
// CHECK:STDOUT:   ret i32 %D.call, !dbg !29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CA.Main.286b0f7890e5304c(double %x, i32 %count) #0 !dbg !30 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CB.Main.286b0f7890e5304c(double %x, i32 %count), !dbg !37
// CHECK:STDOUT:   %D.call = call double @_CD.Main.286b0f7890e5304c(double %x, i32 %count), !dbg !38
// CHECK:STDOUT:   ret double %D.call, !dbg !39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CA.Main.4b30cda44e16b9ec(ptr %x, i32 %count) #0 !dbg !40 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CB.Main.4b30cda44e16b9ec(ptr %x, i32 %count), !dbg !44
// CHECK:STDOUT:   %D.call = call ptr @_CD.Main.4b30cda44e16b9ec(ptr %x, i32 %count), !dbg !45
// CHECK:STDOUT:   ret ptr %D.call, !dbg !46
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CB.Main.3440082c84c3c17b(i32 %x, i32 %count) #0 !dbg !47 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CC.Main.3440082c84c3c17b(i32 %x, i32 %count), !dbg !53
// CHECK:STDOUT:   ret void, !dbg !54
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CD.Main.3440082c84c3c17b(i32 %x, i32 %count) #0 !dbg !55 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 4, !dbg !59
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then.loc48, label %if.else.loc48, !dbg !60
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc48:                                    ; preds = %entry
// CHECK:STDOUT:   ret i32 %x, !dbg !61
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc48:                                    ; preds = %entry
// CHECK:STDOUT:   %Int.as.ModWith.impl.Op.call = srem i32 %count, 2, !dbg !62
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.call = icmp eq i32 %Int.as.ModWith.impl.Op.call, 0, !dbg !62
// CHECK:STDOUT:   br i1 %Int.as.EqWith.impl.Equal.call, label %if.then.loc51, label %if.else.loc51, !dbg !63
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc51:                                    ; preds = %if.else.loc48
// CHECK:STDOUT:   %E.call = call i32 @_CE.Main.3440082c84c3c17b(i32 %x, i32 %count), !dbg !64
// CHECK:STDOUT:   ret i32 %E.call, !dbg !65
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc51:                                    ; preds = %if.else.loc48
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main.3440082c84c3c17b(i32 %x, i32 %count), !dbg !66
// CHECK:STDOUT:   ret i32 %F.call, !dbg !67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CB.Main.286b0f7890e5304c(double %x, i32 %count) #0 !dbg !68 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CC.Main.286b0f7890e5304c(double %x, i32 %count), !dbg !74
// CHECK:STDOUT:   ret void, !dbg !75
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CD.Main.286b0f7890e5304c(double %x, i32 %count) #0 !dbg !76 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 4, !dbg !80
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then.loc48, label %if.else.loc48, !dbg !81
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc48:                                    ; preds = %entry
// CHECK:STDOUT:   ret double %x, !dbg !82
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc48:                                    ; preds = %entry
// CHECK:STDOUT:   %Int.as.ModWith.impl.Op.call = srem i32 %count, 2, !dbg !83
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.call = icmp eq i32 %Int.as.ModWith.impl.Op.call, 0, !dbg !83
// CHECK:STDOUT:   br i1 %Int.as.EqWith.impl.Equal.call, label %if.then.loc51, label %if.else.loc51, !dbg !84
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc51:                                    ; preds = %if.else.loc48
// CHECK:STDOUT:   %E.call = call double @_CE.Main.286b0f7890e5304c(double %x, i32 %count), !dbg !85
// CHECK:STDOUT:   ret double %E.call, !dbg !86
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc51:                                    ; preds = %if.else.loc48
// CHECK:STDOUT:   %F.call = call double @_CF.Main.286b0f7890e5304c(double %x, i32 %count), !dbg !87
// CHECK:STDOUT:   ret double %F.call, !dbg !88
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CB.Main.4b30cda44e16b9ec(ptr %x, i32 %count) #0 !dbg !89 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CC.Main.4b30cda44e16b9ec(ptr %x, i32 %count), !dbg !93
// CHECK:STDOUT:   ret void, !dbg !94
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CD.Main.4b30cda44e16b9ec(ptr %x, i32 %count) #0 !dbg !95 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 4, !dbg !99
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then.loc48, label %if.else.loc48, !dbg !100
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc48:                                    ; preds = %entry
// CHECK:STDOUT:   ret ptr %x, !dbg !101
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc48:                                    ; preds = %entry
// CHECK:STDOUT:   %Int.as.ModWith.impl.Op.call = srem i32 %count, 2, !dbg !102
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.call = icmp eq i32 %Int.as.ModWith.impl.Op.call, 0, !dbg !102
// CHECK:STDOUT:   br i1 %Int.as.EqWith.impl.Equal.call, label %if.then.loc51, label %if.else.loc51, !dbg !103
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc51:                                    ; preds = %if.else.loc48
// CHECK:STDOUT:   %E.call = call ptr @_CE.Main.4b30cda44e16b9ec(ptr %x, i32 %count), !dbg !104
// CHECK:STDOUT:   ret ptr %E.call, !dbg !105
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc51:                                    ; preds = %if.else.loc48
// CHECK:STDOUT:   %F.call = call ptr @_CF.Main.4b30cda44e16b9ec(ptr %x, i32 %count), !dbg !106
// CHECK:STDOUT:   ret ptr %F.call, !dbg !107
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CC.Main.3440082c84c3c17b(i32 %x, i32 %count) #0 !dbg !108 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.call = icmp sle i32 %count, 2, !dbg !112
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.LessOrEquivalent.call, label %if.then, label %if.else, !dbg !113
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 %count), !dbg !114
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !115
// CHECK:STDOUT:   call void @_CB.Main.3440082c84c3c17b(i32 %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !116
// CHECK:STDOUT:   br label %if.else, !dbg !117
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %if.then, %entry
// CHECK:STDOUT:   ret void, !dbg !118
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CE.Main.3440082c84c3c17b(i32 %x, i32 %count) #0 !dbg !119 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call i32 @_CG.Main.3440082c84c3c17b(i32 %x, i32 %count), !dbg !123
// CHECK:STDOUT:   ret i32 %G.call, !dbg !124
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.3440082c84c3c17b(i32 %x, i32 %count) #0 !dbg !125 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call i32 @_CG.Main.3440082c84c3c17b(i32 %x, i32 %count), !dbg !129
// CHECK:STDOUT:   ret i32 %G.call, !dbg !130
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CC.Main.286b0f7890e5304c(double %x, i32 %count) #0 !dbg !131 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.call = icmp sle i32 %count, 2, !dbg !135
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.LessOrEquivalent.call, label %if.then, label %if.else, !dbg !136
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 %count), !dbg !137
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !138
// CHECK:STDOUT:   call void @_CB.Main.286b0f7890e5304c(double %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !139
// CHECK:STDOUT:   br label %if.else, !dbg !140
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %if.then, %entry
// CHECK:STDOUT:   ret void, !dbg !141
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CE.Main.286b0f7890e5304c(double %x, i32 %count) #0 !dbg !142 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call double @_CG.Main.286b0f7890e5304c(double %x, i32 %count), !dbg !146
// CHECK:STDOUT:   ret double %G.call, !dbg !147
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CF.Main.286b0f7890e5304c(double %x, i32 %count) #0 !dbg !148 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call double @_CG.Main.286b0f7890e5304c(double %x, i32 %count), !dbg !152
// CHECK:STDOUT:   ret double %G.call, !dbg !153
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CC.Main.4b30cda44e16b9ec(ptr %x, i32 %count) #0 !dbg !154 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.call = icmp sle i32 %count, 2, !dbg !158
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.LessOrEquivalent.call, label %if.then, label %if.else, !dbg !159
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 %count), !dbg !160
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !161
// CHECK:STDOUT:   call void @_CB.Main.4b30cda44e16b9ec(ptr %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !162
// CHECK:STDOUT:   br label %if.else, !dbg !163
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %if.then, %entry
// CHECK:STDOUT:   ret void, !dbg !164
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CE.Main.4b30cda44e16b9ec(ptr %x, i32 %count) #0 !dbg !165 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call ptr @_CG.Main.4b30cda44e16b9ec(ptr %x, i32 %count), !dbg !169
// CHECK:STDOUT:   ret ptr %G.call, !dbg !170
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CF.Main.4b30cda44e16b9ec(ptr %x, i32 %count) #0 !dbg !171 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call ptr @_CG.Main.4b30cda44e16b9ec(ptr %x, i32 %count), !dbg !175
// CHECK:STDOUT:   ret ptr %G.call, !dbg !176
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @printf(ptr, ...)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CG.Main.3440082c84c3c17b(i32 %x, i32 %count) #0 !dbg !177 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !181
// CHECK:STDOUT:   %D.call = call i32 @_CD.Main.3440082c84c3c17b(i32 %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !182
// CHECK:STDOUT:   ret i32 %D.call, !dbg !183
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CG.Main.286b0f7890e5304c(double %x, i32 %count) #0 !dbg !184 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !188
// CHECK:STDOUT:   %D.call = call double @_CD.Main.286b0f7890e5304c(double %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !189
// CHECK:STDOUT:   ret double %D.call, !dbg !190
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CG.Main.4b30cda44e16b9ec(ptr %x, i32 %count) #0 !dbg !191 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !195
// CHECK:STDOUT:   %D.call = call ptr @_CD.Main.4b30cda44e16b9ec(ptr %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !196
// CHECK:STDOUT:   ret ptr %D.call, !dbg !197
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CA.Main.4b30cda44e16b9ec, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @printf, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CG.Main.3440082c84c3c17b, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CG.Main.286b0f7890e5304c, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CG.Main.4b30cda44e16b9ec, { 1, 0 }
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
// CHECK:STDOUT: !20 = distinct !DISubprogram(name: "A", linkageName: "_CA.Main.3440082c84c3c17b", scope: null, file: !3, line: 31, type: !21, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !24)
// CHECK:STDOUT: !21 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !22 = !{!23, !23, !23}
// CHECK:STDOUT: !23 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !24 = !{!25, !26}
// CHECK:STDOUT: !25 = !DILocalVariable(arg: 1, scope: !20, type: !23)
// CHECK:STDOUT: !26 = !DILocalVariable(arg: 2, scope: !20, type: !23)
// CHECK:STDOUT: !27 = !DILocation(line: 32, column: 3, scope: !20)
// CHECK:STDOUT: !28 = !DILocation(line: 33, column: 10, scope: !20)
// CHECK:STDOUT: !29 = !DILocation(line: 33, column: 3, scope: !20)
// CHECK:STDOUT: !30 = distinct !DISubprogram(name: "A", linkageName: "_CA.Main.286b0f7890e5304c", scope: null, file: !3, line: 31, type: !31, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !34)
// CHECK:STDOUT: !31 = !DISubroutineType(types: !32)
// CHECK:STDOUT: !32 = !{!33, !33, !23}
// CHECK:STDOUT: !33 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 8)
// CHECK:STDOUT: !34 = !{!35, !36}
// CHECK:STDOUT: !35 = !DILocalVariable(arg: 1, scope: !30, type: !33)
// CHECK:STDOUT: !36 = !DILocalVariable(arg: 2, scope: !30, type: !23)
// CHECK:STDOUT: !37 = !DILocation(line: 32, column: 3, scope: !30)
// CHECK:STDOUT: !38 = !DILocation(line: 33, column: 10, scope: !30)
// CHECK:STDOUT: !39 = !DILocation(line: 33, column: 3, scope: !30)
// CHECK:STDOUT: !40 = distinct !DISubprogram(name: "A", linkageName: "_CA.Main.4b30cda44e16b9ec", scope: null, file: !3, line: 31, type: !31, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !41)
// CHECK:STDOUT: !41 = !{!42, !43}
// CHECK:STDOUT: !42 = !DILocalVariable(arg: 1, scope: !40, type: !33)
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 2, scope: !40, type: !23)
// CHECK:STDOUT: !44 = !DILocation(line: 32, column: 3, scope: !40)
// CHECK:STDOUT: !45 = !DILocation(line: 33, column: 10, scope: !40)
// CHECK:STDOUT: !46 = !DILocation(line: 33, column: 3, scope: !40)
// CHECK:STDOUT: !47 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.3440082c84c3c17b", scope: null, file: !3, line: 36, type: !48, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !50)
// CHECK:STDOUT: !48 = !DISubroutineType(types: !49)
// CHECK:STDOUT: !49 = !{null, !23, !23}
// CHECK:STDOUT: !50 = !{!51, !52}
// CHECK:STDOUT: !51 = !DILocalVariable(arg: 1, scope: !47, type: !23)
// CHECK:STDOUT: !52 = !DILocalVariable(arg: 2, scope: !47, type: !23)
// CHECK:STDOUT: !53 = !DILocation(line: 37, column: 3, scope: !47)
// CHECK:STDOUT: !54 = !DILocation(line: 36, column: 1, scope: !47)
// CHECK:STDOUT: !55 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.3440082c84c3c17b", scope: null, file: !3, line: 47, type: !21, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !56)
// CHECK:STDOUT: !56 = !{!57, !58}
// CHECK:STDOUT: !57 = !DILocalVariable(arg: 1, scope: !55, type: !23)
// CHECK:STDOUT: !58 = !DILocalVariable(arg: 2, scope: !55, type: !23)
// CHECK:STDOUT: !59 = !DILocation(line: 48, column: 7, scope: !55)
// CHECK:STDOUT: !60 = !DILocation(line: 48, column: 6, scope: !55)
// CHECK:STDOUT: !61 = !DILocation(line: 49, column: 5, scope: !55)
// CHECK:STDOUT: !62 = !DILocation(line: 51, column: 7, scope: !55)
// CHECK:STDOUT: !63 = !DILocation(line: 51, column: 6, scope: !55)
// CHECK:STDOUT: !64 = !DILocation(line: 52, column: 12, scope: !55)
// CHECK:STDOUT: !65 = !DILocation(line: 52, column: 5, scope: !55)
// CHECK:STDOUT: !66 = !DILocation(line: 54, column: 12, scope: !55)
// CHECK:STDOUT: !67 = !DILocation(line: 54, column: 5, scope: !55)
// CHECK:STDOUT: !68 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.286b0f7890e5304c", scope: null, file: !3, line: 36, type: !69, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !71)
// CHECK:STDOUT: !69 = !DISubroutineType(types: !70)
// CHECK:STDOUT: !70 = !{null, !33, !23}
// CHECK:STDOUT: !71 = !{!72, !73}
// CHECK:STDOUT: !72 = !DILocalVariable(arg: 1, scope: !68, type: !33)
// CHECK:STDOUT: !73 = !DILocalVariable(arg: 2, scope: !68, type: !23)
// CHECK:STDOUT: !74 = !DILocation(line: 37, column: 3, scope: !68)
// CHECK:STDOUT: !75 = !DILocation(line: 36, column: 1, scope: !68)
// CHECK:STDOUT: !76 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.286b0f7890e5304c", scope: null, file: !3, line: 47, type: !31, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !77)
// CHECK:STDOUT: !77 = !{!78, !79}
// CHECK:STDOUT: !78 = !DILocalVariable(arg: 1, scope: !76, type: !33)
// CHECK:STDOUT: !79 = !DILocalVariable(arg: 2, scope: !76, type: !23)
// CHECK:STDOUT: !80 = !DILocation(line: 48, column: 7, scope: !76)
// CHECK:STDOUT: !81 = !DILocation(line: 48, column: 6, scope: !76)
// CHECK:STDOUT: !82 = !DILocation(line: 49, column: 5, scope: !76)
// CHECK:STDOUT: !83 = !DILocation(line: 51, column: 7, scope: !76)
// CHECK:STDOUT: !84 = !DILocation(line: 51, column: 6, scope: !76)
// CHECK:STDOUT: !85 = !DILocation(line: 52, column: 12, scope: !76)
// CHECK:STDOUT: !86 = !DILocation(line: 52, column: 5, scope: !76)
// CHECK:STDOUT: !87 = !DILocation(line: 54, column: 12, scope: !76)
// CHECK:STDOUT: !88 = !DILocation(line: 54, column: 5, scope: !76)
// CHECK:STDOUT: !89 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.4b30cda44e16b9ec", scope: null, file: !3, line: 36, type: !69, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !90)
// CHECK:STDOUT: !90 = !{!91, !92}
// CHECK:STDOUT: !91 = !DILocalVariable(arg: 1, scope: !89, type: !33)
// CHECK:STDOUT: !92 = !DILocalVariable(arg: 2, scope: !89, type: !23)
// CHECK:STDOUT: !93 = !DILocation(line: 37, column: 3, scope: !89)
// CHECK:STDOUT: !94 = !DILocation(line: 36, column: 1, scope: !89)
// CHECK:STDOUT: !95 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.4b30cda44e16b9ec", scope: null, file: !3, line: 47, type: !31, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !96)
// CHECK:STDOUT: !96 = !{!97, !98}
// CHECK:STDOUT: !97 = !DILocalVariable(arg: 1, scope: !95, type: !33)
// CHECK:STDOUT: !98 = !DILocalVariable(arg: 2, scope: !95, type: !23)
// CHECK:STDOUT: !99 = !DILocation(line: 48, column: 7, scope: !95)
// CHECK:STDOUT: !100 = !DILocation(line: 48, column: 6, scope: !95)
// CHECK:STDOUT: !101 = !DILocation(line: 49, column: 5, scope: !95)
// CHECK:STDOUT: !102 = !DILocation(line: 51, column: 7, scope: !95)
// CHECK:STDOUT: !103 = !DILocation(line: 51, column: 6, scope: !95)
// CHECK:STDOUT: !104 = !DILocation(line: 52, column: 12, scope: !95)
// CHECK:STDOUT: !105 = !DILocation(line: 52, column: 5, scope: !95)
// CHECK:STDOUT: !106 = !DILocation(line: 54, column: 12, scope: !95)
// CHECK:STDOUT: !107 = !DILocation(line: 54, column: 5, scope: !95)
// CHECK:STDOUT: !108 = distinct !DISubprogram(name: "C", linkageName: "_CC.Main.3440082c84c3c17b", scope: null, file: !3, line: 40, type: !48, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !109)
// CHECK:STDOUT: !109 = !{!110, !111}
// CHECK:STDOUT: !110 = !DILocalVariable(arg: 1, scope: !108, type: !23)
// CHECK:STDOUT: !111 = !DILocalVariable(arg: 2, scope: !108, type: !23)
// CHECK:STDOUT: !112 = !DILocation(line: 41, column: 7, scope: !108)
// CHECK:STDOUT: !113 = !DILocation(line: 41, column: 6, scope: !108)
// CHECK:STDOUT: !114 = !DILocation(line: 42, column: 5, scope: !108)
// CHECK:STDOUT: !115 = !DILocation(line: 43, column: 10, scope: !108)
// CHECK:STDOUT: !116 = !DILocation(line: 43, column: 5, scope: !108)
// CHECK:STDOUT: !117 = !DILocation(line: 41, column: 3, scope: !108)
// CHECK:STDOUT: !118 = !DILocation(line: 40, column: 1, scope: !108)
// CHECK:STDOUT: !119 = distinct !DISubprogram(name: "E", linkageName: "_CE.Main.3440082c84c3c17b", scope: null, file: !3, line: 60, type: !21, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !120)
// CHECK:STDOUT: !120 = !{!121, !122}
// CHECK:STDOUT: !121 = !DILocalVariable(arg: 1, scope: !119, type: !23)
// CHECK:STDOUT: !122 = !DILocalVariable(arg: 2, scope: !119, type: !23)
// CHECK:STDOUT: !123 = !DILocation(line: 61, column: 10, scope: !119)
// CHECK:STDOUT: !124 = !DILocation(line: 61, column: 3, scope: !119)
// CHECK:STDOUT: !125 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.3440082c84c3c17b", scope: null, file: !3, line: 64, type: !21, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !126)
// CHECK:STDOUT: !126 = !{!127, !128}
// CHECK:STDOUT: !127 = !DILocalVariable(arg: 1, scope: !125, type: !23)
// CHECK:STDOUT: !128 = !DILocalVariable(arg: 2, scope: !125, type: !23)
// CHECK:STDOUT: !129 = !DILocation(line: 65, column: 10, scope: !125)
// CHECK:STDOUT: !130 = !DILocation(line: 65, column: 3, scope: !125)
// CHECK:STDOUT: !131 = distinct !DISubprogram(name: "C", linkageName: "_CC.Main.286b0f7890e5304c", scope: null, file: !3, line: 40, type: !69, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !132)
// CHECK:STDOUT: !132 = !{!133, !134}
// CHECK:STDOUT: !133 = !DILocalVariable(arg: 1, scope: !131, type: !33)
// CHECK:STDOUT: !134 = !DILocalVariable(arg: 2, scope: !131, type: !23)
// CHECK:STDOUT: !135 = !DILocation(line: 41, column: 7, scope: !131)
// CHECK:STDOUT: !136 = !DILocation(line: 41, column: 6, scope: !131)
// CHECK:STDOUT: !137 = !DILocation(line: 42, column: 5, scope: !131)
// CHECK:STDOUT: !138 = !DILocation(line: 43, column: 10, scope: !131)
// CHECK:STDOUT: !139 = !DILocation(line: 43, column: 5, scope: !131)
// CHECK:STDOUT: !140 = !DILocation(line: 41, column: 3, scope: !131)
// CHECK:STDOUT: !141 = !DILocation(line: 40, column: 1, scope: !131)
// CHECK:STDOUT: !142 = distinct !DISubprogram(name: "E", linkageName: "_CE.Main.286b0f7890e5304c", scope: null, file: !3, line: 60, type: !31, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !143)
// CHECK:STDOUT: !143 = !{!144, !145}
// CHECK:STDOUT: !144 = !DILocalVariable(arg: 1, scope: !142, type: !33)
// CHECK:STDOUT: !145 = !DILocalVariable(arg: 2, scope: !142, type: !23)
// CHECK:STDOUT: !146 = !DILocation(line: 61, column: 10, scope: !142)
// CHECK:STDOUT: !147 = !DILocation(line: 61, column: 3, scope: !142)
// CHECK:STDOUT: !148 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.286b0f7890e5304c", scope: null, file: !3, line: 64, type: !31, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !149)
// CHECK:STDOUT: !149 = !{!150, !151}
// CHECK:STDOUT: !150 = !DILocalVariable(arg: 1, scope: !148, type: !33)
// CHECK:STDOUT: !151 = !DILocalVariable(arg: 2, scope: !148, type: !23)
// CHECK:STDOUT: !152 = !DILocation(line: 65, column: 10, scope: !148)
// CHECK:STDOUT: !153 = !DILocation(line: 65, column: 3, scope: !148)
// CHECK:STDOUT: !154 = distinct !DISubprogram(name: "C", linkageName: "_CC.Main.4b30cda44e16b9ec", scope: null, file: !3, line: 40, type: !69, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !155)
// CHECK:STDOUT: !155 = !{!156, !157}
// CHECK:STDOUT: !156 = !DILocalVariable(arg: 1, scope: !154, type: !33)
// CHECK:STDOUT: !157 = !DILocalVariable(arg: 2, scope: !154, type: !23)
// CHECK:STDOUT: !158 = !DILocation(line: 41, column: 7, scope: !154)
// CHECK:STDOUT: !159 = !DILocation(line: 41, column: 6, scope: !154)
// CHECK:STDOUT: !160 = !DILocation(line: 42, column: 5, scope: !154)
// CHECK:STDOUT: !161 = !DILocation(line: 43, column: 10, scope: !154)
// CHECK:STDOUT: !162 = !DILocation(line: 43, column: 5, scope: !154)
// CHECK:STDOUT: !163 = !DILocation(line: 41, column: 3, scope: !154)
// CHECK:STDOUT: !164 = !DILocation(line: 40, column: 1, scope: !154)
// CHECK:STDOUT: !165 = distinct !DISubprogram(name: "E", linkageName: "_CE.Main.4b30cda44e16b9ec", scope: null, file: !3, line: 60, type: !31, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !166)
// CHECK:STDOUT: !166 = !{!167, !168}
// CHECK:STDOUT: !167 = !DILocalVariable(arg: 1, scope: !165, type: !33)
// CHECK:STDOUT: !168 = !DILocalVariable(arg: 2, scope: !165, type: !23)
// CHECK:STDOUT: !169 = !DILocation(line: 61, column: 10, scope: !165)
// CHECK:STDOUT: !170 = !DILocation(line: 61, column: 3, scope: !165)
// CHECK:STDOUT: !171 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.4b30cda44e16b9ec", scope: null, file: !3, line: 64, type: !31, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !172)
// CHECK:STDOUT: !172 = !{!173, !174}
// CHECK:STDOUT: !173 = !DILocalVariable(arg: 1, scope: !171, type: !33)
// CHECK:STDOUT: !174 = !DILocalVariable(arg: 2, scope: !171, type: !23)
// CHECK:STDOUT: !175 = !DILocation(line: 65, column: 10, scope: !171)
// CHECK:STDOUT: !176 = !DILocation(line: 65, column: 3, scope: !171)
// CHECK:STDOUT: !177 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.3440082c84c3c17b", scope: null, file: !3, line: 68, type: !21, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !178)
// CHECK:STDOUT: !178 = !{!179, !180}
// CHECK:STDOUT: !179 = !DILocalVariable(arg: 1, scope: !177, type: !23)
// CHECK:STDOUT: !180 = !DILocalVariable(arg: 2, scope: !177, type: !23)
// CHECK:STDOUT: !181 = !DILocation(line: 69, column: 15, scope: !177)
// CHECK:STDOUT: !182 = !DILocation(line: 69, column: 10, scope: !177)
// CHECK:STDOUT: !183 = !DILocation(line: 69, column: 3, scope: !177)
// CHECK:STDOUT: !184 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.286b0f7890e5304c", scope: null, file: !3, line: 68, type: !31, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !185)
// CHECK:STDOUT: !185 = !{!186, !187}
// CHECK:STDOUT: !186 = !DILocalVariable(arg: 1, scope: !184, type: !33)
// CHECK:STDOUT: !187 = !DILocalVariable(arg: 2, scope: !184, type: !23)
// CHECK:STDOUT: !188 = !DILocation(line: 69, column: 15, scope: !184)
// CHECK:STDOUT: !189 = !DILocation(line: 69, column: 10, scope: !184)
// CHECK:STDOUT: !190 = !DILocation(line: 69, column: 3, scope: !184)
// CHECK:STDOUT: !191 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.4b30cda44e16b9ec", scope: null, file: !3, line: 68, type: !31, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !192)
// CHECK:STDOUT: !192 = !{!193, !194}
// CHECK:STDOUT: !193 = !DILocalVariable(arg: 1, scope: !191, type: !33)
// CHECK:STDOUT: !194 = !DILocalVariable(arg: 2, scope: !191, type: !23)
// CHECK:STDOUT: !195 = !DILocation(line: 69, column: 15, scope: !191)
// CHECK:STDOUT: !196 = !DILocation(line: 69, column: 10, scope: !191)
// CHECK:STDOUT: !197 = !DILocation(line: 69, column: 3, scope: !191)
