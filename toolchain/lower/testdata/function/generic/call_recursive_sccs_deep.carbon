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
// CHECK:STDOUT: define void @_CM.Main() !dbg !4 {
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
// CHECK:STDOUT:   %A.call.loc79 = call i32 @_CA.Main.de631560529e9861(i32 %.loc79_5, i32 0), !dbg !12
// CHECK:STDOUT:   %.loc80_5 = load double, ptr %m.var, align 8, !dbg !13
// CHECK:STDOUT:   %A.call.loc80 = call double @_CA.Main.3836ccc537683c19(double %.loc80_5, i32 0), !dbg !14
// CHECK:STDOUT:   %.loc81_5 = load ptr, ptr %ptr_i32.var, align 8, !dbg !15
// CHECK:STDOUT:   %A.call.loc81 = call ptr @_CA.Main.a194de4a3b8dbd0c(ptr %.loc81_5, i32 0), !dbg !16
// CHECK:STDOUT:   %.loc82_5 = load ptr, ptr %ptr_f64.var, align 8, !dbg !17
// CHECK:STDOUT:   %A.call.loc82 = call ptr @_CA.Main.a194de4a3b8dbd0c(ptr %.loc82_5, i32 0), !dbg !18
// CHECK:STDOUT:   ret void, !dbg !19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #0
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr i32 @_CA.Main.de631560529e9861(i32 %x, i32 %count) !dbg !20 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CB.Main.de631560529e9861(i32 %x, i32 %count), !dbg !21
// CHECK:STDOUT:   %D.call = call i32 @_CD.Main.de631560529e9861(i32 %x, i32 %count), !dbg !22
// CHECK:STDOUT:   ret i32 %D.call, !dbg !23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr double @_CA.Main.3836ccc537683c19(double %x, i32 %count) !dbg !24 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CB.Main.3836ccc537683c19(double %x, i32 %count), !dbg !25
// CHECK:STDOUT:   %D.call = call double @_CD.Main.3836ccc537683c19(double %x, i32 %count), !dbg !26
// CHECK:STDOUT:   ret double %D.call, !dbg !27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr ptr @_CA.Main.a194de4a3b8dbd0c(ptr %x, i32 %count) !dbg !28 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CB.Main.a194de4a3b8dbd0c(ptr %x, i32 %count), !dbg !29
// CHECK:STDOUT:   %D.call = call ptr @_CD.Main.a194de4a3b8dbd0c(ptr %x, i32 %count), !dbg !30
// CHECK:STDOUT:   ret ptr %D.call, !dbg !31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr void @_CB.Main.de631560529e9861(i32 %x, i32 %count) !dbg !32 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CC.Main.de631560529e9861(i32 %x, i32 %count), !dbg !33
// CHECK:STDOUT:   ret void, !dbg !34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr i32 @_CD.Main.de631560529e9861(i32 %x, i32 %count) !dbg !35 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 4, !dbg !36
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then.loc48, label %if.else.loc48, !dbg !37
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc48:                                    ; preds = %entry
// CHECK:STDOUT:   ret i32 %x, !dbg !38
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc48:                                    ; preds = %entry
// CHECK:STDOUT:   %Int.as.ModWith.impl.Op.call = srem i32 %count, 2, !dbg !39
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.call = icmp eq i32 %Int.as.ModWith.impl.Op.call, 0, !dbg !39
// CHECK:STDOUT:   br i1 %Int.as.EqWith.impl.Equal.call, label %if.then.loc51, label %if.else.loc51, !dbg !40
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc51:                                    ; preds = %if.else.loc48
// CHECK:STDOUT:   %E.call = call i32 @_CE.Main.de631560529e9861(i32 %x, i32 %count), !dbg !41
// CHECK:STDOUT:   ret i32 %E.call, !dbg !42
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc51:                                    ; preds = %if.else.loc48
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main.de631560529e9861(i32 %x, i32 %count), !dbg !43
// CHECK:STDOUT:   ret i32 %F.call, !dbg !44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr void @_CB.Main.3836ccc537683c19(double %x, i32 %count) !dbg !45 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CC.Main.3836ccc537683c19(double %x, i32 %count), !dbg !46
// CHECK:STDOUT:   ret void, !dbg !47
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr double @_CD.Main.3836ccc537683c19(double %x, i32 %count) !dbg !48 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 4, !dbg !49
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then.loc48, label %if.else.loc48, !dbg !50
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc48:                                    ; preds = %entry
// CHECK:STDOUT:   ret double %x, !dbg !51
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc48:                                    ; preds = %entry
// CHECK:STDOUT:   %Int.as.ModWith.impl.Op.call = srem i32 %count, 2, !dbg !52
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.call = icmp eq i32 %Int.as.ModWith.impl.Op.call, 0, !dbg !52
// CHECK:STDOUT:   br i1 %Int.as.EqWith.impl.Equal.call, label %if.then.loc51, label %if.else.loc51, !dbg !53
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc51:                                    ; preds = %if.else.loc48
// CHECK:STDOUT:   %E.call = call double @_CE.Main.3836ccc537683c19(double %x, i32 %count), !dbg !54
// CHECK:STDOUT:   ret double %E.call, !dbg !55
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc51:                                    ; preds = %if.else.loc48
// CHECK:STDOUT:   %F.call = call double @_CF.Main.3836ccc537683c19(double %x, i32 %count), !dbg !56
// CHECK:STDOUT:   ret double %F.call, !dbg !57
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr void @_CB.Main.a194de4a3b8dbd0c(ptr %x, i32 %count) !dbg !58 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CC.Main.a194de4a3b8dbd0c(ptr %x, i32 %count), !dbg !59
// CHECK:STDOUT:   ret void, !dbg !60
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr ptr @_CD.Main.a194de4a3b8dbd0c(ptr %x, i32 %count) !dbg !61 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 4, !dbg !62
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then.loc48, label %if.else.loc48, !dbg !63
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc48:                                    ; preds = %entry
// CHECK:STDOUT:   ret ptr %x, !dbg !64
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc48:                                    ; preds = %entry
// CHECK:STDOUT:   %Int.as.ModWith.impl.Op.call = srem i32 %count, 2, !dbg !65
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.call = icmp eq i32 %Int.as.ModWith.impl.Op.call, 0, !dbg !65
// CHECK:STDOUT:   br i1 %Int.as.EqWith.impl.Equal.call, label %if.then.loc51, label %if.else.loc51, !dbg !66
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc51:                                    ; preds = %if.else.loc48
// CHECK:STDOUT:   %E.call = call ptr @_CE.Main.a194de4a3b8dbd0c(ptr %x, i32 %count), !dbg !67
// CHECK:STDOUT:   ret ptr %E.call, !dbg !68
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc51:                                    ; preds = %if.else.loc48
// CHECK:STDOUT:   %F.call = call ptr @_CF.Main.a194de4a3b8dbd0c(ptr %x, i32 %count), !dbg !69
// CHECK:STDOUT:   ret ptr %F.call, !dbg !70
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr void @_CC.Main.de631560529e9861(i32 %x, i32 %count) !dbg !71 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.call = icmp sle i32 %count, 2, !dbg !72
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.LessOrEquivalent.call, label %if.then, label %if.else, !dbg !73
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 %count), !dbg !74
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !75
// CHECK:STDOUT:   call void @_CB.Main.de631560529e9861(i32 %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !76
// CHECK:STDOUT:   br label %if.else, !dbg !77
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %if.then, %entry
// CHECK:STDOUT:   ret void, !dbg !78
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr i32 @_CE.Main.de631560529e9861(i32 %x, i32 %count) !dbg !79 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call i32 @_CG.Main.de631560529e9861(i32 %x, i32 %count), !dbg !80
// CHECK:STDOUT:   ret i32 %G.call, !dbg !81
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.de631560529e9861(i32 %x, i32 %count) !dbg !82 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call i32 @_CG.Main.de631560529e9861(i32 %x, i32 %count), !dbg !83
// CHECK:STDOUT:   ret i32 %G.call, !dbg !84
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr void @_CC.Main.3836ccc537683c19(double %x, i32 %count) !dbg !85 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.call = icmp sle i32 %count, 2, !dbg !86
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.LessOrEquivalent.call, label %if.then, label %if.else, !dbg !87
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 %count), !dbg !88
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !89
// CHECK:STDOUT:   call void @_CB.Main.3836ccc537683c19(double %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !90
// CHECK:STDOUT:   br label %if.else, !dbg !91
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %if.then, %entry
// CHECK:STDOUT:   ret void, !dbg !92
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr double @_CE.Main.3836ccc537683c19(double %x, i32 %count) !dbg !93 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call double @_CG.Main.3836ccc537683c19(double %x, i32 %count), !dbg !94
// CHECK:STDOUT:   ret double %G.call, !dbg !95
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr double @_CF.Main.3836ccc537683c19(double %x, i32 %count) !dbg !96 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call double @_CG.Main.3836ccc537683c19(double %x, i32 %count), !dbg !97
// CHECK:STDOUT:   ret double %G.call, !dbg !98
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr void @_CC.Main.a194de4a3b8dbd0c(ptr %x, i32 %count) !dbg !99 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.LessOrEquivalent.call = icmp sle i32 %count, 2, !dbg !100
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.LessOrEquivalent.call, label %if.then, label %if.else, !dbg !101
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 %count), !dbg !102
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !103
// CHECK:STDOUT:   call void @_CB.Main.a194de4a3b8dbd0c(ptr %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !104
// CHECK:STDOUT:   br label %if.else, !dbg !105
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %if.then, %entry
// CHECK:STDOUT:   ret void, !dbg !106
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr ptr @_CE.Main.a194de4a3b8dbd0c(ptr %x, i32 %count) !dbg !107 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call ptr @_CG.Main.a194de4a3b8dbd0c(ptr %x, i32 %count), !dbg !108
// CHECK:STDOUT:   ret ptr %G.call, !dbg !109
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr ptr @_CF.Main.a194de4a3b8dbd0c(ptr %x, i32 %count) !dbg !110 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call ptr @_CG.Main.a194de4a3b8dbd0c(ptr %x, i32 %count), !dbg !111
// CHECK:STDOUT:   ret ptr %G.call, !dbg !112
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @printf(ptr, ...)
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr i32 @_CG.Main.de631560529e9861(i32 %x, i32 %count) !dbg !113 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !114
// CHECK:STDOUT:   %D.call = call i32 @_CD.Main.de631560529e9861(i32 %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !115
// CHECK:STDOUT:   ret i32 %D.call, !dbg !116
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr double @_CG.Main.3836ccc537683c19(double %x, i32 %count) !dbg !117 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !118
// CHECK:STDOUT:   %D.call = call double @_CD.Main.3836ccc537683c19(double %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !119
// CHECK:STDOUT:   ret double %D.call, !dbg !120
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr ptr @_CG.Main.a194de4a3b8dbd0c(ptr %x, i32 %count) !dbg !121 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !122
// CHECK:STDOUT:   %D.call = call ptr @_CD.Main.a194de4a3b8dbd0c(ptr %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !123
// CHECK:STDOUT:   ret ptr %D.call, !dbg !124
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CA.Main.a194de4a3b8dbd0c, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @printf, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CG.Main.de631560529e9861, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CG.Main.3836ccc537683c19, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CG.Main.a194de4a3b8dbd0c, { 1, 0 }
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
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "M", linkageName: "_CM.Main", scope: null, file: !3, line: 73, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
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
// CHECK:STDOUT: !20 = distinct !DISubprogram(name: "A", linkageName: "_CA.Main.de631560529e9861", scope: null, file: !3, line: 31, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !21 = !DILocation(line: 32, column: 3, scope: !20)
// CHECK:STDOUT: !22 = !DILocation(line: 33, column: 10, scope: !20)
// CHECK:STDOUT: !23 = !DILocation(line: 33, column: 3, scope: !20)
// CHECK:STDOUT: !24 = distinct !DISubprogram(name: "A", linkageName: "_CA.Main.3836ccc537683c19", scope: null, file: !3, line: 31, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !25 = !DILocation(line: 32, column: 3, scope: !24)
// CHECK:STDOUT: !26 = !DILocation(line: 33, column: 10, scope: !24)
// CHECK:STDOUT: !27 = !DILocation(line: 33, column: 3, scope: !24)
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "A", linkageName: "_CA.Main.a194de4a3b8dbd0c", scope: null, file: !3, line: 31, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !29 = !DILocation(line: 32, column: 3, scope: !28)
// CHECK:STDOUT: !30 = !DILocation(line: 33, column: 10, scope: !28)
// CHECK:STDOUT: !31 = !DILocation(line: 33, column: 3, scope: !28)
// CHECK:STDOUT: !32 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.de631560529e9861", scope: null, file: !3, line: 36, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !33 = !DILocation(line: 37, column: 3, scope: !32)
// CHECK:STDOUT: !34 = !DILocation(line: 36, column: 1, scope: !32)
// CHECK:STDOUT: !35 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.de631560529e9861", scope: null, file: !3, line: 47, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !36 = !DILocation(line: 48, column: 7, scope: !35)
// CHECK:STDOUT: !37 = !DILocation(line: 48, column: 6, scope: !35)
// CHECK:STDOUT: !38 = !DILocation(line: 49, column: 5, scope: !35)
// CHECK:STDOUT: !39 = !DILocation(line: 51, column: 7, scope: !35)
// CHECK:STDOUT: !40 = !DILocation(line: 51, column: 6, scope: !35)
// CHECK:STDOUT: !41 = !DILocation(line: 52, column: 12, scope: !35)
// CHECK:STDOUT: !42 = !DILocation(line: 52, column: 5, scope: !35)
// CHECK:STDOUT: !43 = !DILocation(line: 54, column: 12, scope: !35)
// CHECK:STDOUT: !44 = !DILocation(line: 54, column: 5, scope: !35)
// CHECK:STDOUT: !45 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.3836ccc537683c19", scope: null, file: !3, line: 36, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !46 = !DILocation(line: 37, column: 3, scope: !45)
// CHECK:STDOUT: !47 = !DILocation(line: 36, column: 1, scope: !45)
// CHECK:STDOUT: !48 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.3836ccc537683c19", scope: null, file: !3, line: 47, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !49 = !DILocation(line: 48, column: 7, scope: !48)
// CHECK:STDOUT: !50 = !DILocation(line: 48, column: 6, scope: !48)
// CHECK:STDOUT: !51 = !DILocation(line: 49, column: 5, scope: !48)
// CHECK:STDOUT: !52 = !DILocation(line: 51, column: 7, scope: !48)
// CHECK:STDOUT: !53 = !DILocation(line: 51, column: 6, scope: !48)
// CHECK:STDOUT: !54 = !DILocation(line: 52, column: 12, scope: !48)
// CHECK:STDOUT: !55 = !DILocation(line: 52, column: 5, scope: !48)
// CHECK:STDOUT: !56 = !DILocation(line: 54, column: 12, scope: !48)
// CHECK:STDOUT: !57 = !DILocation(line: 54, column: 5, scope: !48)
// CHECK:STDOUT: !58 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.a194de4a3b8dbd0c", scope: null, file: !3, line: 36, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !59 = !DILocation(line: 37, column: 3, scope: !58)
// CHECK:STDOUT: !60 = !DILocation(line: 36, column: 1, scope: !58)
// CHECK:STDOUT: !61 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.a194de4a3b8dbd0c", scope: null, file: !3, line: 47, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !62 = !DILocation(line: 48, column: 7, scope: !61)
// CHECK:STDOUT: !63 = !DILocation(line: 48, column: 6, scope: !61)
// CHECK:STDOUT: !64 = !DILocation(line: 49, column: 5, scope: !61)
// CHECK:STDOUT: !65 = !DILocation(line: 51, column: 7, scope: !61)
// CHECK:STDOUT: !66 = !DILocation(line: 51, column: 6, scope: !61)
// CHECK:STDOUT: !67 = !DILocation(line: 52, column: 12, scope: !61)
// CHECK:STDOUT: !68 = !DILocation(line: 52, column: 5, scope: !61)
// CHECK:STDOUT: !69 = !DILocation(line: 54, column: 12, scope: !61)
// CHECK:STDOUT: !70 = !DILocation(line: 54, column: 5, scope: !61)
// CHECK:STDOUT: !71 = distinct !DISubprogram(name: "C", linkageName: "_CC.Main.de631560529e9861", scope: null, file: !3, line: 40, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !72 = !DILocation(line: 41, column: 7, scope: !71)
// CHECK:STDOUT: !73 = !DILocation(line: 41, column: 6, scope: !71)
// CHECK:STDOUT: !74 = !DILocation(line: 42, column: 5, scope: !71)
// CHECK:STDOUT: !75 = !DILocation(line: 43, column: 10, scope: !71)
// CHECK:STDOUT: !76 = !DILocation(line: 43, column: 5, scope: !71)
// CHECK:STDOUT: !77 = !DILocation(line: 41, column: 3, scope: !71)
// CHECK:STDOUT: !78 = !DILocation(line: 40, column: 1, scope: !71)
// CHECK:STDOUT: !79 = distinct !DISubprogram(name: "E", linkageName: "_CE.Main.de631560529e9861", scope: null, file: !3, line: 60, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !80 = !DILocation(line: 61, column: 10, scope: !79)
// CHECK:STDOUT: !81 = !DILocation(line: 61, column: 3, scope: !79)
// CHECK:STDOUT: !82 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.de631560529e9861", scope: null, file: !3, line: 64, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !83 = !DILocation(line: 65, column: 10, scope: !82)
// CHECK:STDOUT: !84 = !DILocation(line: 65, column: 3, scope: !82)
// CHECK:STDOUT: !85 = distinct !DISubprogram(name: "C", linkageName: "_CC.Main.3836ccc537683c19", scope: null, file: !3, line: 40, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !86 = !DILocation(line: 41, column: 7, scope: !85)
// CHECK:STDOUT: !87 = !DILocation(line: 41, column: 6, scope: !85)
// CHECK:STDOUT: !88 = !DILocation(line: 42, column: 5, scope: !85)
// CHECK:STDOUT: !89 = !DILocation(line: 43, column: 10, scope: !85)
// CHECK:STDOUT: !90 = !DILocation(line: 43, column: 5, scope: !85)
// CHECK:STDOUT: !91 = !DILocation(line: 41, column: 3, scope: !85)
// CHECK:STDOUT: !92 = !DILocation(line: 40, column: 1, scope: !85)
// CHECK:STDOUT: !93 = distinct !DISubprogram(name: "E", linkageName: "_CE.Main.3836ccc537683c19", scope: null, file: !3, line: 60, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !94 = !DILocation(line: 61, column: 10, scope: !93)
// CHECK:STDOUT: !95 = !DILocation(line: 61, column: 3, scope: !93)
// CHECK:STDOUT: !96 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.3836ccc537683c19", scope: null, file: !3, line: 64, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !97 = !DILocation(line: 65, column: 10, scope: !96)
// CHECK:STDOUT: !98 = !DILocation(line: 65, column: 3, scope: !96)
// CHECK:STDOUT: !99 = distinct !DISubprogram(name: "C", linkageName: "_CC.Main.a194de4a3b8dbd0c", scope: null, file: !3, line: 40, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !100 = !DILocation(line: 41, column: 7, scope: !99)
// CHECK:STDOUT: !101 = !DILocation(line: 41, column: 6, scope: !99)
// CHECK:STDOUT: !102 = !DILocation(line: 42, column: 5, scope: !99)
// CHECK:STDOUT: !103 = !DILocation(line: 43, column: 10, scope: !99)
// CHECK:STDOUT: !104 = !DILocation(line: 43, column: 5, scope: !99)
// CHECK:STDOUT: !105 = !DILocation(line: 41, column: 3, scope: !99)
// CHECK:STDOUT: !106 = !DILocation(line: 40, column: 1, scope: !99)
// CHECK:STDOUT: !107 = distinct !DISubprogram(name: "E", linkageName: "_CE.Main.a194de4a3b8dbd0c", scope: null, file: !3, line: 60, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !108 = !DILocation(line: 61, column: 10, scope: !107)
// CHECK:STDOUT: !109 = !DILocation(line: 61, column: 3, scope: !107)
// CHECK:STDOUT: !110 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.a194de4a3b8dbd0c", scope: null, file: !3, line: 64, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !111 = !DILocation(line: 65, column: 10, scope: !110)
// CHECK:STDOUT: !112 = !DILocation(line: 65, column: 3, scope: !110)
// CHECK:STDOUT: !113 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.de631560529e9861", scope: null, file: !3, line: 68, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !114 = !DILocation(line: 69, column: 15, scope: !113)
// CHECK:STDOUT: !115 = !DILocation(line: 69, column: 10, scope: !113)
// CHECK:STDOUT: !116 = !DILocation(line: 69, column: 3, scope: !113)
// CHECK:STDOUT: !117 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.3836ccc537683c19", scope: null, file: !3, line: 68, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !118 = !DILocation(line: 69, column: 15, scope: !117)
// CHECK:STDOUT: !119 = !DILocation(line: 69, column: 10, scope: !117)
// CHECK:STDOUT: !120 = !DILocation(line: 69, column: 3, scope: !117)
// CHECK:STDOUT: !121 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.a194de4a3b8dbd0c", scope: null, file: !3, line: 68, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !122 = !DILocation(line: 69, column: 15, scope: !121)
// CHECK:STDOUT: !123 = !DILocation(line: 69, column: 10, scope: !121)
// CHECK:STDOUT: !124 = !DILocation(line: 69, column: 3, scope: !121)
