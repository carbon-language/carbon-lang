// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/generic/call_recursive_diamond.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/generic/call_recursive_diamond.carbon

import Core library "io";
fn A[T:! type](x: T, count: i32) -> T;
fn B[T:! type](x: T, count: i32) -> T;
fn C[T:! type](x: T, count: i32) -> T;
fn D[T:! type](x: T, count: i32) -> T;

// Builds on `call_recursive_mutual.carbon`. The two specifics for each of:
// A, B, C, D with a pointer type could be deduplicated.
fn A[T:! type](x: T, count: i32) -> T {
  if (count > 4) {
    return x;
  }
  if (count % 2 == 0) {
    return B(x, count);
  } else {
    return C(x, count);
  }
}

// B and C are not equivalent in the diamond: different generics and different function bodies.
fn B[T:! type](x: T, count: i32) -> T {
  Core.Print(1);
  return D(x, count);
}

fn C[T:! type](x: T, count: i32) -> T {
  Core.Print(2);
  return D(x, count);
}

fn D[T:! type](x: T, count: i32) -> T {
  return A(x, count + 1);
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

// CHECK:STDOUT: ; ModuleID = 'call_recursive_diamond.carbon'
// CHECK:STDOUT: source_filename = "call_recursive_diamond.carbon"
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
// CHECK:STDOUT:   %.loc51_5 = load i32, ptr %n.var, align 4, !dbg !11
// CHECK:STDOUT:   %A.call.loc51 = call i32 @_CA.Main.b88d1103f417c6d4(i32 %.loc51_5, i32 0), !dbg !12
// CHECK:STDOUT:   %.loc52_5 = load double, ptr %m.var, align 8, !dbg !13
// CHECK:STDOUT:   %A.call.loc52 = call double @_CA.Main.66be507887ceee78(double %.loc52_5, i32 0), !dbg !14
// CHECK:STDOUT:   %.loc53_5 = load ptr, ptr %ptr_i32.var, align 8, !dbg !15
// CHECK:STDOUT:   %A.call.loc53 = call ptr @_CA.Main.e8193710fd35b608(ptr %.loc53_5, i32 0), !dbg !16
// CHECK:STDOUT:   %.loc54_5 = load ptr, ptr %ptr_f64.var, align 8, !dbg !17
// CHECK:STDOUT:   %A.call.loc54 = call ptr @_CA.Main.04bf2edaaa84aa22(ptr %.loc54_5, i32 0), !dbg !18
// CHECK:STDOUT:   ret void, !dbg !19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(i64 immarg, ptr captures(none)) #0
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr i32 @_CA.Main.b88d1103f417c6d4(i32 %x, i32 %count) !dbg !20 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.greater = icmp sgt i32 %count, 4, !dbg !21
// CHECK:STDOUT:   br i1 %int.greater, label %if.then.loc20, label %if.else.loc20, !dbg !22
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc20:                                    ; preds = %entry
// CHECK:STDOUT:   ret i32 %x, !dbg !23
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc20:                                    ; preds = %entry
// CHECK:STDOUT:   %int.smod = srem i32 %count, 2, !dbg !24
// CHECK:STDOUT:   %int.eq = icmp eq i32 %int.smod, 0, !dbg !24
// CHECK:STDOUT:   br i1 %int.eq, label %if.then.loc23, label %if.else.loc23, !dbg !25
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc23:                                    ; preds = %if.else.loc20
// CHECK:STDOUT:   %B.call = call i32 @_CB.Main.b88d1103f417c6d4(i32 %x, i32 %count), !dbg !26
// CHECK:STDOUT:   ret i32 %B.call, !dbg !27
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc23:                                    ; preds = %if.else.loc20
// CHECK:STDOUT:   %C.call = call i32 @_CC.Main.b88d1103f417c6d4(i32 %x, i32 %count), !dbg !28
// CHECK:STDOUT:   ret i32 %C.call, !dbg !29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr double @_CA.Main.66be507887ceee78(double %x, i32 %count) !dbg !30 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.greater = icmp sgt i32 %count, 4, !dbg !31
// CHECK:STDOUT:   br i1 %int.greater, label %if.then.loc20, label %if.else.loc20, !dbg !32
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc20:                                    ; preds = %entry
// CHECK:STDOUT:   ret double %x, !dbg !33
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc20:                                    ; preds = %entry
// CHECK:STDOUT:   %int.smod = srem i32 %count, 2, !dbg !34
// CHECK:STDOUT:   %int.eq = icmp eq i32 %int.smod, 0, !dbg !34
// CHECK:STDOUT:   br i1 %int.eq, label %if.then.loc23, label %if.else.loc23, !dbg !35
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc23:                                    ; preds = %if.else.loc20
// CHECK:STDOUT:   %B.call = call double @_CB.Main.66be507887ceee78(double %x, i32 %count), !dbg !36
// CHECK:STDOUT:   ret double %B.call, !dbg !37
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc23:                                    ; preds = %if.else.loc20
// CHECK:STDOUT:   %C.call = call double @_CC.Main.66be507887ceee78(double %x, i32 %count), !dbg !38
// CHECK:STDOUT:   ret double %C.call, !dbg !39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr ptr @_CA.Main.e8193710fd35b608(ptr %x, i32 %count) !dbg !40 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.greater = icmp sgt i32 %count, 4, !dbg !41
// CHECK:STDOUT:   br i1 %int.greater, label %if.then.loc20, label %if.else.loc20, !dbg !42
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc20:                                    ; preds = %entry
// CHECK:STDOUT:   ret ptr %x, !dbg !43
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc20:                                    ; preds = %entry
// CHECK:STDOUT:   %int.smod = srem i32 %count, 2, !dbg !44
// CHECK:STDOUT:   %int.eq = icmp eq i32 %int.smod, 0, !dbg !44
// CHECK:STDOUT:   br i1 %int.eq, label %if.then.loc23, label %if.else.loc23, !dbg !45
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc23:                                    ; preds = %if.else.loc20
// CHECK:STDOUT:   %B.call = call ptr @_CB.Main.e8193710fd35b608(ptr %x, i32 %count), !dbg !46
// CHECK:STDOUT:   ret ptr %B.call, !dbg !47
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc23:                                    ; preds = %if.else.loc20
// CHECK:STDOUT:   %C.call = call ptr @_CC.Main.e8193710fd35b608(ptr %x, i32 %count), !dbg !48
// CHECK:STDOUT:   ret ptr %C.call, !dbg !49
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr ptr @_CA.Main.04bf2edaaa84aa22(ptr %x, i32 %count) !dbg !50 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.greater = icmp sgt i32 %count, 4, !dbg !51
// CHECK:STDOUT:   br i1 %int.greater, label %if.then.loc20, label %if.else.loc20, !dbg !52
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc20:                                    ; preds = %entry
// CHECK:STDOUT:   ret ptr %x, !dbg !53
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc20:                                    ; preds = %entry
// CHECK:STDOUT:   %int.smod = srem i32 %count, 2, !dbg !54
// CHECK:STDOUT:   %int.eq = icmp eq i32 %int.smod, 0, !dbg !54
// CHECK:STDOUT:   br i1 %int.eq, label %if.then.loc23, label %if.else.loc23, !dbg !55
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc23:                                    ; preds = %if.else.loc20
// CHECK:STDOUT:   %B.call = call ptr @_CB.Main.04bf2edaaa84aa22(ptr %x, i32 %count), !dbg !56
// CHECK:STDOUT:   ret ptr %B.call, !dbg !57
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc23:                                    ; preds = %if.else.loc20
// CHECK:STDOUT:   %C.call = call ptr @_CC.Main.04bf2edaaa84aa22(ptr %x, i32 %count), !dbg !58
// CHECK:STDOUT:   ret ptr %C.call, !dbg !59
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr i32 @_CB.Main.b88d1103f417c6d4(i32 %x, i32 %count) !dbg !60 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %print.int = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 1), !dbg !61
// CHECK:STDOUT:   %D.call = call i32 @_CD.Main.b88d1103f417c6d4(i32 %x, i32 %count), !dbg !62
// CHECK:STDOUT:   ret i32 %D.call, !dbg !63
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr i32 @_CC.Main.b88d1103f417c6d4(i32 %x, i32 %count) !dbg !64 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %print.int = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 2), !dbg !65
// CHECK:STDOUT:   %D.call = call i32 @_CD.Main.b88d1103f417c6d4(i32 %x, i32 %count), !dbg !66
// CHECK:STDOUT:   ret i32 %D.call, !dbg !67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr double @_CB.Main.66be507887ceee78(double %x, i32 %count) !dbg !68 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %print.int = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 1), !dbg !69
// CHECK:STDOUT:   %D.call = call double @_CD.Main.66be507887ceee78(double %x, i32 %count), !dbg !70
// CHECK:STDOUT:   ret double %D.call, !dbg !71
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr double @_CC.Main.66be507887ceee78(double %x, i32 %count) !dbg !72 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %print.int = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 2), !dbg !73
// CHECK:STDOUT:   %D.call = call double @_CD.Main.66be507887ceee78(double %x, i32 %count), !dbg !74
// CHECK:STDOUT:   ret double %D.call, !dbg !75
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr ptr @_CB.Main.e8193710fd35b608(ptr %x, i32 %count) !dbg !76 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %print.int = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 1), !dbg !77
// CHECK:STDOUT:   %D.call = call ptr @_CD.Main.e8193710fd35b608(ptr %x, i32 %count), !dbg !78
// CHECK:STDOUT:   ret ptr %D.call, !dbg !79
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr ptr @_CC.Main.e8193710fd35b608(ptr %x, i32 %count) !dbg !80 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %print.int = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 2), !dbg !81
// CHECK:STDOUT:   %D.call = call ptr @_CD.Main.e8193710fd35b608(ptr %x, i32 %count), !dbg !82
// CHECK:STDOUT:   ret ptr %D.call, !dbg !83
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr ptr @_CB.Main.04bf2edaaa84aa22(ptr %x, i32 %count) !dbg !84 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %print.int = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 1), !dbg !85
// CHECK:STDOUT:   %D.call = call ptr @_CD.Main.04bf2edaaa84aa22(ptr %x, i32 %count), !dbg !86
// CHECK:STDOUT:   ret ptr %D.call, !dbg !87
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr ptr @_CC.Main.04bf2edaaa84aa22(ptr %x, i32 %count) !dbg !88 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %print.int = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 2), !dbg !89
// CHECK:STDOUT:   %D.call = call ptr @_CD.Main.04bf2edaaa84aa22(ptr %x, i32 %count), !dbg !90
// CHECK:STDOUT:   ret ptr %D.call, !dbg !91
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @printf(ptr, ...)
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr i32 @_CD.Main.b88d1103f417c6d4(i32 %x, i32 %count) !dbg !92 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.sadd = add i32 %count, 1, !dbg !93
// CHECK:STDOUT:   %A.call = call i32 @_CA.Main.b88d1103f417c6d4(i32 %x, i32 %int.sadd), !dbg !94
// CHECK:STDOUT:   ret i32 %A.call, !dbg !95
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr double @_CD.Main.66be507887ceee78(double %x, i32 %count) !dbg !96 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.sadd = add i32 %count, 1, !dbg !97
// CHECK:STDOUT:   %A.call = call double @_CA.Main.66be507887ceee78(double %x, i32 %int.sadd), !dbg !98
// CHECK:STDOUT:   ret double %A.call, !dbg !99
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr ptr @_CD.Main.e8193710fd35b608(ptr %x, i32 %count) !dbg !100 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.sadd = add i32 %count, 1, !dbg !101
// CHECK:STDOUT:   %A.call = call ptr @_CA.Main.e8193710fd35b608(ptr %x, i32 %int.sadd), !dbg !102
// CHECK:STDOUT:   ret ptr %A.call, !dbg !103
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr ptr @_CD.Main.04bf2edaaa84aa22(ptr %x, i32 %count) !dbg !104 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.sadd = add i32 %count, 1, !dbg !105
// CHECK:STDOUT:   %A.call = call ptr @_CA.Main.04bf2edaaa84aa22(ptr %x, i32 %int.sadd), !dbg !106
// CHECK:STDOUT:   ret ptr %A.call, !dbg !107
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @printf, { 7, 6, 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CD.Main.b88d1103f417c6d4, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CD.Main.66be507887ceee78, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CD.Main.e8193710fd35b608, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CD.Main.04bf2edaaa84aa22, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "call_recursive_diamond.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "M", linkageName: "_CM.Main", scope: null, file: !3, line: 45, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 46, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 47, column: 3, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 48, column: 3, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 49, column: 3, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 51, column: 5, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 51, column: 3, scope: !4)
// CHECK:STDOUT: !13 = !DILocation(line: 52, column: 5, scope: !4)
// CHECK:STDOUT: !14 = !DILocation(line: 52, column: 3, scope: !4)
// CHECK:STDOUT: !15 = !DILocation(line: 53, column: 5, scope: !4)
// CHECK:STDOUT: !16 = !DILocation(line: 53, column: 3, scope: !4)
// CHECK:STDOUT: !17 = !DILocation(line: 54, column: 5, scope: !4)
// CHECK:STDOUT: !18 = !DILocation(line: 54, column: 3, scope: !4)
// CHECK:STDOUT: !19 = !DILocation(line: 45, column: 1, scope: !4)
// CHECK:STDOUT: !20 = distinct !DISubprogram(name: "A", linkageName: "_CA.Main.b88d1103f417c6d4", scope: null, file: !3, line: 19, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !21 = !DILocation(line: 20, column: 7, scope: !20)
// CHECK:STDOUT: !22 = !DILocation(line: 20, column: 6, scope: !20)
// CHECK:STDOUT: !23 = !DILocation(line: 21, column: 5, scope: !20)
// CHECK:STDOUT: !24 = !DILocation(line: 23, column: 7, scope: !20)
// CHECK:STDOUT: !25 = !DILocation(line: 23, column: 6, scope: !20)
// CHECK:STDOUT: !26 = !DILocation(line: 24, column: 12, scope: !20)
// CHECK:STDOUT: !27 = !DILocation(line: 24, column: 5, scope: !20)
// CHECK:STDOUT: !28 = !DILocation(line: 26, column: 12, scope: !20)
// CHECK:STDOUT: !29 = !DILocation(line: 26, column: 5, scope: !20)
// CHECK:STDOUT: !30 = distinct !DISubprogram(name: "A", linkageName: "_CA.Main.66be507887ceee78", scope: null, file: !3, line: 19, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !31 = !DILocation(line: 20, column: 7, scope: !30)
// CHECK:STDOUT: !32 = !DILocation(line: 20, column: 6, scope: !30)
// CHECK:STDOUT: !33 = !DILocation(line: 21, column: 5, scope: !30)
// CHECK:STDOUT: !34 = !DILocation(line: 23, column: 7, scope: !30)
// CHECK:STDOUT: !35 = !DILocation(line: 23, column: 6, scope: !30)
// CHECK:STDOUT: !36 = !DILocation(line: 24, column: 12, scope: !30)
// CHECK:STDOUT: !37 = !DILocation(line: 24, column: 5, scope: !30)
// CHECK:STDOUT: !38 = !DILocation(line: 26, column: 12, scope: !30)
// CHECK:STDOUT: !39 = !DILocation(line: 26, column: 5, scope: !30)
// CHECK:STDOUT: !40 = distinct !DISubprogram(name: "A", linkageName: "_CA.Main.e8193710fd35b608", scope: null, file: !3, line: 19, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !41 = !DILocation(line: 20, column: 7, scope: !40)
// CHECK:STDOUT: !42 = !DILocation(line: 20, column: 6, scope: !40)
// CHECK:STDOUT: !43 = !DILocation(line: 21, column: 5, scope: !40)
// CHECK:STDOUT: !44 = !DILocation(line: 23, column: 7, scope: !40)
// CHECK:STDOUT: !45 = !DILocation(line: 23, column: 6, scope: !40)
// CHECK:STDOUT: !46 = !DILocation(line: 24, column: 12, scope: !40)
// CHECK:STDOUT: !47 = !DILocation(line: 24, column: 5, scope: !40)
// CHECK:STDOUT: !48 = !DILocation(line: 26, column: 12, scope: !40)
// CHECK:STDOUT: !49 = !DILocation(line: 26, column: 5, scope: !40)
// CHECK:STDOUT: !50 = distinct !DISubprogram(name: "A", linkageName: "_CA.Main.04bf2edaaa84aa22", scope: null, file: !3, line: 19, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !51 = !DILocation(line: 20, column: 7, scope: !50)
// CHECK:STDOUT: !52 = !DILocation(line: 20, column: 6, scope: !50)
// CHECK:STDOUT: !53 = !DILocation(line: 21, column: 5, scope: !50)
// CHECK:STDOUT: !54 = !DILocation(line: 23, column: 7, scope: !50)
// CHECK:STDOUT: !55 = !DILocation(line: 23, column: 6, scope: !50)
// CHECK:STDOUT: !56 = !DILocation(line: 24, column: 12, scope: !50)
// CHECK:STDOUT: !57 = !DILocation(line: 24, column: 5, scope: !50)
// CHECK:STDOUT: !58 = !DILocation(line: 26, column: 12, scope: !50)
// CHECK:STDOUT: !59 = !DILocation(line: 26, column: 5, scope: !50)
// CHECK:STDOUT: !60 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.b88d1103f417c6d4", scope: null, file: !3, line: 31, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !61 = !DILocation(line: 32, column: 3, scope: !60)
// CHECK:STDOUT: !62 = !DILocation(line: 33, column: 10, scope: !60)
// CHECK:STDOUT: !63 = !DILocation(line: 33, column: 3, scope: !60)
// CHECK:STDOUT: !64 = distinct !DISubprogram(name: "C", linkageName: "_CC.Main.b88d1103f417c6d4", scope: null, file: !3, line: 36, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !65 = !DILocation(line: 37, column: 3, scope: !64)
// CHECK:STDOUT: !66 = !DILocation(line: 38, column: 10, scope: !64)
// CHECK:STDOUT: !67 = !DILocation(line: 38, column: 3, scope: !64)
// CHECK:STDOUT: !68 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.66be507887ceee78", scope: null, file: !3, line: 31, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !69 = !DILocation(line: 32, column: 3, scope: !68)
// CHECK:STDOUT: !70 = !DILocation(line: 33, column: 10, scope: !68)
// CHECK:STDOUT: !71 = !DILocation(line: 33, column: 3, scope: !68)
// CHECK:STDOUT: !72 = distinct !DISubprogram(name: "C", linkageName: "_CC.Main.66be507887ceee78", scope: null, file: !3, line: 36, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !73 = !DILocation(line: 37, column: 3, scope: !72)
// CHECK:STDOUT: !74 = !DILocation(line: 38, column: 10, scope: !72)
// CHECK:STDOUT: !75 = !DILocation(line: 38, column: 3, scope: !72)
// CHECK:STDOUT: !76 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.e8193710fd35b608", scope: null, file: !3, line: 31, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !77 = !DILocation(line: 32, column: 3, scope: !76)
// CHECK:STDOUT: !78 = !DILocation(line: 33, column: 10, scope: !76)
// CHECK:STDOUT: !79 = !DILocation(line: 33, column: 3, scope: !76)
// CHECK:STDOUT: !80 = distinct !DISubprogram(name: "C", linkageName: "_CC.Main.e8193710fd35b608", scope: null, file: !3, line: 36, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !81 = !DILocation(line: 37, column: 3, scope: !80)
// CHECK:STDOUT: !82 = !DILocation(line: 38, column: 10, scope: !80)
// CHECK:STDOUT: !83 = !DILocation(line: 38, column: 3, scope: !80)
// CHECK:STDOUT: !84 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.04bf2edaaa84aa22", scope: null, file: !3, line: 31, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !85 = !DILocation(line: 32, column: 3, scope: !84)
// CHECK:STDOUT: !86 = !DILocation(line: 33, column: 10, scope: !84)
// CHECK:STDOUT: !87 = !DILocation(line: 33, column: 3, scope: !84)
// CHECK:STDOUT: !88 = distinct !DISubprogram(name: "C", linkageName: "_CC.Main.04bf2edaaa84aa22", scope: null, file: !3, line: 36, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !89 = !DILocation(line: 37, column: 3, scope: !88)
// CHECK:STDOUT: !90 = !DILocation(line: 38, column: 10, scope: !88)
// CHECK:STDOUT: !91 = !DILocation(line: 38, column: 3, scope: !88)
// CHECK:STDOUT: !92 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.b88d1103f417c6d4", scope: null, file: !3, line: 41, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !93 = !DILocation(line: 42, column: 15, scope: !92)
// CHECK:STDOUT: !94 = !DILocation(line: 42, column: 10, scope: !92)
// CHECK:STDOUT: !95 = !DILocation(line: 42, column: 3, scope: !92)
// CHECK:STDOUT: !96 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.66be507887ceee78", scope: null, file: !3, line: 41, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !97 = !DILocation(line: 42, column: 15, scope: !96)
// CHECK:STDOUT: !98 = !DILocation(line: 42, column: 10, scope: !96)
// CHECK:STDOUT: !99 = !DILocation(line: 42, column: 3, scope: !96)
// CHECK:STDOUT: !100 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.e8193710fd35b608", scope: null, file: !3, line: 41, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !101 = !DILocation(line: 42, column: 15, scope: !100)
// CHECK:STDOUT: !102 = !DILocation(line: 42, column: 10, scope: !100)
// CHECK:STDOUT: !103 = !DILocation(line: 42, column: 3, scope: !100)
// CHECK:STDOUT: !104 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.04bf2edaaa84aa22", scope: null, file: !3, line: 41, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !105 = !DILocation(line: 42, column: 15, scope: !104)
// CHECK:STDOUT: !106 = !DILocation(line: 42, column: 10, scope: !104)
// CHECK:STDOUT: !107 = !DILocation(line: 42, column: 3, scope: !104)
