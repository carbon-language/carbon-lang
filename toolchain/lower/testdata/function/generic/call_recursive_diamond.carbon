// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/generic/call_recursive_diamond.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/generic/call_recursive_diamond.carbon

import Core library "io";
fn A[T:! Core.Copy](x: T, count: i32) -> T;
fn B[T:! Core.Copy](x: T, count: i32) -> T;
fn C[T:! Core.Copy](x: T, count: i32) -> T;
fn D[T:! Core.Copy](x: T, count: i32) -> T;

// Builds on `call_recursive_mutual.carbon`. The two specifics for each of:
// A, B, C, D with a pointer type could be deduplicated.
fn A[T:! Core.Copy](x: T, count: i32) -> T {
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
fn B[T:! Core.Copy](x: T, count: i32) -> T {
  Core.Print(1);
  return D(x, count);
}

fn C[T:! Core.Copy](x: T, count: i32) -> T {
  Core.Print(2);
  return D(x, count);
}

fn D[T:! Core.Copy](x: T, count: i32) -> T {
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
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %n.var), !dbg !7
// CHECK:STDOUT:   store i32 0, ptr %n.var, align 4, !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %m.var), !dbg !8
// CHECK:STDOUT:   store double 1.000000e+00, ptr %m.var, align 8, !dbg !8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_i32.var), !dbg !9
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_f64.var), !dbg !10
// CHECK:STDOUT:   %.loc53_5 = load i32, ptr %n.var, align 4, !dbg !11
// CHECK:STDOUT:   %A.call.loc53 = call i32 @_CA.Main.5450dc8e8b8e0899(i32 %.loc53_5, i32 0), !dbg !12
// CHECK:STDOUT:   %.loc54_5 = load double, ptr %m.var, align 8, !dbg !13
// CHECK:STDOUT:   %A.call.loc54 = call double @_CA.Main.83aece103b0a8681(double %.loc54_5, i32 0), !dbg !14
// CHECK:STDOUT:   %.loc55_5 = load ptr, ptr %ptr_i32.var, align 8, !dbg !15
// CHECK:STDOUT:   %A.call.loc55 = call ptr @_CA.Main.0a952f8bcc623ce6(ptr %.loc55_5, i32 0), !dbg !16
// CHECK:STDOUT:   %.loc56_5 = load ptr, ptr %ptr_f64.var, align 8, !dbg !17
// CHECK:STDOUT:   %A.call.loc56 = call ptr @_CA.Main.0a952f8bcc623ce6(ptr %.loc56_5, i32 0), !dbg !18
// CHECK:STDOUT:   ret void, !dbg !19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #0
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr i32 @_CA.Main.5450dc8e8b8e0899(i32 %x, i32 %count) !dbg !20 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 4, !dbg !21
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then.loc22, label %if.else.loc22, !dbg !22
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc22:                                    ; preds = %entry
// CHECK:STDOUT:   ret i32 %x, !dbg !23
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc22:                                    ; preds = %entry
// CHECK:STDOUT:   %Int.as.ModWith.impl.Op.call = srem i32 %count, 2, !dbg !24
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.call = icmp eq i32 %Int.as.ModWith.impl.Op.call, 0, !dbg !24
// CHECK:STDOUT:   br i1 %Int.as.EqWith.impl.Equal.call, label %if.then.loc25, label %if.else.loc25, !dbg !25
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc25:                                    ; preds = %if.else.loc22
// CHECK:STDOUT:   %B.call = call i32 @_CB.Main.5450dc8e8b8e0899(i32 %x, i32 %count), !dbg !26
// CHECK:STDOUT:   ret i32 %B.call, !dbg !27
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc25:                                    ; preds = %if.else.loc22
// CHECK:STDOUT:   %C.call = call i32 @_CC.Main.5450dc8e8b8e0899(i32 %x, i32 %count), !dbg !28
// CHECK:STDOUT:   ret i32 %C.call, !dbg !29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr double @_CA.Main.83aece103b0a8681(double %x, i32 %count) !dbg !30 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 4, !dbg !31
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then.loc22, label %if.else.loc22, !dbg !32
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc22:                                    ; preds = %entry
// CHECK:STDOUT:   ret double %x, !dbg !33
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc22:                                    ; preds = %entry
// CHECK:STDOUT:   %Int.as.ModWith.impl.Op.call = srem i32 %count, 2, !dbg !34
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.call = icmp eq i32 %Int.as.ModWith.impl.Op.call, 0, !dbg !34
// CHECK:STDOUT:   br i1 %Int.as.EqWith.impl.Equal.call, label %if.then.loc25, label %if.else.loc25, !dbg !35
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc25:                                    ; preds = %if.else.loc22
// CHECK:STDOUT:   %B.call = call double @_CB.Main.83aece103b0a8681(double %x, i32 %count), !dbg !36
// CHECK:STDOUT:   ret double %B.call, !dbg !37
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc25:                                    ; preds = %if.else.loc22
// CHECK:STDOUT:   %C.call = call double @_CC.Main.83aece103b0a8681(double %x, i32 %count), !dbg !38
// CHECK:STDOUT:   ret double %C.call, !dbg !39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr ptr @_CA.Main.0a952f8bcc623ce6(ptr %x, i32 %count) !dbg !40 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 4, !dbg !41
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then.loc22, label %if.else.loc22, !dbg !42
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc22:                                    ; preds = %entry
// CHECK:STDOUT:   ret ptr %x, !dbg !43
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc22:                                    ; preds = %entry
// CHECK:STDOUT:   %Int.as.ModWith.impl.Op.call = srem i32 %count, 2, !dbg !44
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.call = icmp eq i32 %Int.as.ModWith.impl.Op.call, 0, !dbg !44
// CHECK:STDOUT:   br i1 %Int.as.EqWith.impl.Equal.call, label %if.then.loc25, label %if.else.loc25, !dbg !45
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc25:                                    ; preds = %if.else.loc22
// CHECK:STDOUT:   %B.call = call ptr @_CB.Main.0a952f8bcc623ce6(ptr %x, i32 %count), !dbg !46
// CHECK:STDOUT:   ret ptr %B.call, !dbg !47
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc25:                                    ; preds = %if.else.loc22
// CHECK:STDOUT:   %C.call = call ptr @_CC.Main.0a952f8bcc623ce6(ptr %x, i32 %count), !dbg !48
// CHECK:STDOUT:   ret ptr %C.call, !dbg !49
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr i32 @_CB.Main.5450dc8e8b8e0899(i32 %x, i32 %count) !dbg !50 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 1), !dbg !51
// CHECK:STDOUT:   %D.call = call i32 @_CD.Main.5450dc8e8b8e0899(i32 %x, i32 %count), !dbg !52
// CHECK:STDOUT:   ret i32 %D.call, !dbg !53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr i32 @_CC.Main.5450dc8e8b8e0899(i32 %x, i32 %count) !dbg !54 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 2), !dbg !55
// CHECK:STDOUT:   %D.call = call i32 @_CD.Main.5450dc8e8b8e0899(i32 %x, i32 %count), !dbg !56
// CHECK:STDOUT:   ret i32 %D.call, !dbg !57
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr double @_CB.Main.83aece103b0a8681(double %x, i32 %count) !dbg !58 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 1), !dbg !59
// CHECK:STDOUT:   %D.call = call double @_CD.Main.83aece103b0a8681(double %x, i32 %count), !dbg !60
// CHECK:STDOUT:   ret double %D.call, !dbg !61
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr double @_CC.Main.83aece103b0a8681(double %x, i32 %count) !dbg !62 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 2), !dbg !63
// CHECK:STDOUT:   %D.call = call double @_CD.Main.83aece103b0a8681(double %x, i32 %count), !dbg !64
// CHECK:STDOUT:   ret double %D.call, !dbg !65
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr ptr @_CB.Main.0a952f8bcc623ce6(ptr %x, i32 %count) !dbg !66 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 1), !dbg !67
// CHECK:STDOUT:   %D.call = call ptr @_CD.Main.0a952f8bcc623ce6(ptr %x, i32 %count), !dbg !68
// CHECK:STDOUT:   ret ptr %D.call, !dbg !69
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr ptr @_CC.Main.0a952f8bcc623ce6(ptr %x, i32 %count) !dbg !70 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 2), !dbg !71
// CHECK:STDOUT:   %D.call = call ptr @_CD.Main.0a952f8bcc623ce6(ptr %x, i32 %count), !dbg !72
// CHECK:STDOUT:   ret ptr %D.call, !dbg !73
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @printf(ptr, ...)
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr i32 @_CD.Main.5450dc8e8b8e0899(i32 %x, i32 %count) !dbg !74 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !75
// CHECK:STDOUT:   %A.call = call i32 @_CA.Main.5450dc8e8b8e0899(i32 %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !76
// CHECK:STDOUT:   ret i32 %A.call, !dbg !77
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr double @_CD.Main.83aece103b0a8681(double %x, i32 %count) !dbg !78 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !79
// CHECK:STDOUT:   %A.call = call double @_CA.Main.83aece103b0a8681(double %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !80
// CHECK:STDOUT:   ret double %A.call, !dbg !81
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr ptr @_CD.Main.0a952f8bcc623ce6(ptr %x, i32 %count) !dbg !82 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !83
// CHECK:STDOUT:   %A.call = call ptr @_CA.Main.0a952f8bcc623ce6(ptr %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !84
// CHECK:STDOUT:   ret ptr %A.call, !dbg !85
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CA.Main.0a952f8bcc623ce6, { 1, 2, 0 }
// CHECK:STDOUT: uselistorder ptr @printf, { 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CD.Main.5450dc8e8b8e0899, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CD.Main.83aece103b0a8681, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CD.Main.0a952f8bcc623ce6, { 1, 0 }
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
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "M", linkageName: "_CM.Main", scope: null, file: !3, line: 47, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 48, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 49, column: 3, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 50, column: 3, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 51, column: 3, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 53, column: 5, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 53, column: 3, scope: !4)
// CHECK:STDOUT: !13 = !DILocation(line: 54, column: 5, scope: !4)
// CHECK:STDOUT: !14 = !DILocation(line: 54, column: 3, scope: !4)
// CHECK:STDOUT: !15 = !DILocation(line: 55, column: 5, scope: !4)
// CHECK:STDOUT: !16 = !DILocation(line: 55, column: 3, scope: !4)
// CHECK:STDOUT: !17 = !DILocation(line: 56, column: 5, scope: !4)
// CHECK:STDOUT: !18 = !DILocation(line: 56, column: 3, scope: !4)
// CHECK:STDOUT: !19 = !DILocation(line: 47, column: 1, scope: !4)
// CHECK:STDOUT: !20 = distinct !DISubprogram(name: "A", linkageName: "_CA.Main.5450dc8e8b8e0899", scope: null, file: !3, line: 21, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !21 = !DILocation(line: 22, column: 7, scope: !20)
// CHECK:STDOUT: !22 = !DILocation(line: 22, column: 6, scope: !20)
// CHECK:STDOUT: !23 = !DILocation(line: 23, column: 5, scope: !20)
// CHECK:STDOUT: !24 = !DILocation(line: 25, column: 7, scope: !20)
// CHECK:STDOUT: !25 = !DILocation(line: 25, column: 6, scope: !20)
// CHECK:STDOUT: !26 = !DILocation(line: 26, column: 12, scope: !20)
// CHECK:STDOUT: !27 = !DILocation(line: 26, column: 5, scope: !20)
// CHECK:STDOUT: !28 = !DILocation(line: 28, column: 12, scope: !20)
// CHECK:STDOUT: !29 = !DILocation(line: 28, column: 5, scope: !20)
// CHECK:STDOUT: !30 = distinct !DISubprogram(name: "A", linkageName: "_CA.Main.83aece103b0a8681", scope: null, file: !3, line: 21, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !31 = !DILocation(line: 22, column: 7, scope: !30)
// CHECK:STDOUT: !32 = !DILocation(line: 22, column: 6, scope: !30)
// CHECK:STDOUT: !33 = !DILocation(line: 23, column: 5, scope: !30)
// CHECK:STDOUT: !34 = !DILocation(line: 25, column: 7, scope: !30)
// CHECK:STDOUT: !35 = !DILocation(line: 25, column: 6, scope: !30)
// CHECK:STDOUT: !36 = !DILocation(line: 26, column: 12, scope: !30)
// CHECK:STDOUT: !37 = !DILocation(line: 26, column: 5, scope: !30)
// CHECK:STDOUT: !38 = !DILocation(line: 28, column: 12, scope: !30)
// CHECK:STDOUT: !39 = !DILocation(line: 28, column: 5, scope: !30)
// CHECK:STDOUT: !40 = distinct !DISubprogram(name: "A", linkageName: "_CA.Main.0a952f8bcc623ce6", scope: null, file: !3, line: 21, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !41 = !DILocation(line: 22, column: 7, scope: !40)
// CHECK:STDOUT: !42 = !DILocation(line: 22, column: 6, scope: !40)
// CHECK:STDOUT: !43 = !DILocation(line: 23, column: 5, scope: !40)
// CHECK:STDOUT: !44 = !DILocation(line: 25, column: 7, scope: !40)
// CHECK:STDOUT: !45 = !DILocation(line: 25, column: 6, scope: !40)
// CHECK:STDOUT: !46 = !DILocation(line: 26, column: 12, scope: !40)
// CHECK:STDOUT: !47 = !DILocation(line: 26, column: 5, scope: !40)
// CHECK:STDOUT: !48 = !DILocation(line: 28, column: 12, scope: !40)
// CHECK:STDOUT: !49 = !DILocation(line: 28, column: 5, scope: !40)
// CHECK:STDOUT: !50 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.5450dc8e8b8e0899", scope: null, file: !3, line: 33, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !51 = !DILocation(line: 34, column: 3, scope: !50)
// CHECK:STDOUT: !52 = !DILocation(line: 35, column: 10, scope: !50)
// CHECK:STDOUT: !53 = !DILocation(line: 35, column: 3, scope: !50)
// CHECK:STDOUT: !54 = distinct !DISubprogram(name: "C", linkageName: "_CC.Main.5450dc8e8b8e0899", scope: null, file: !3, line: 38, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !55 = !DILocation(line: 39, column: 3, scope: !54)
// CHECK:STDOUT: !56 = !DILocation(line: 40, column: 10, scope: !54)
// CHECK:STDOUT: !57 = !DILocation(line: 40, column: 3, scope: !54)
// CHECK:STDOUT: !58 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.83aece103b0a8681", scope: null, file: !3, line: 33, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !59 = !DILocation(line: 34, column: 3, scope: !58)
// CHECK:STDOUT: !60 = !DILocation(line: 35, column: 10, scope: !58)
// CHECK:STDOUT: !61 = !DILocation(line: 35, column: 3, scope: !58)
// CHECK:STDOUT: !62 = distinct !DISubprogram(name: "C", linkageName: "_CC.Main.83aece103b0a8681", scope: null, file: !3, line: 38, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !63 = !DILocation(line: 39, column: 3, scope: !62)
// CHECK:STDOUT: !64 = !DILocation(line: 40, column: 10, scope: !62)
// CHECK:STDOUT: !65 = !DILocation(line: 40, column: 3, scope: !62)
// CHECK:STDOUT: !66 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.0a952f8bcc623ce6", scope: null, file: !3, line: 33, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !67 = !DILocation(line: 34, column: 3, scope: !66)
// CHECK:STDOUT: !68 = !DILocation(line: 35, column: 10, scope: !66)
// CHECK:STDOUT: !69 = !DILocation(line: 35, column: 3, scope: !66)
// CHECK:STDOUT: !70 = distinct !DISubprogram(name: "C", linkageName: "_CC.Main.0a952f8bcc623ce6", scope: null, file: !3, line: 38, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !71 = !DILocation(line: 39, column: 3, scope: !70)
// CHECK:STDOUT: !72 = !DILocation(line: 40, column: 10, scope: !70)
// CHECK:STDOUT: !73 = !DILocation(line: 40, column: 3, scope: !70)
// CHECK:STDOUT: !74 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.5450dc8e8b8e0899", scope: null, file: !3, line: 43, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !75 = !DILocation(line: 44, column: 15, scope: !74)
// CHECK:STDOUT: !76 = !DILocation(line: 44, column: 10, scope: !74)
// CHECK:STDOUT: !77 = !DILocation(line: 44, column: 3, scope: !74)
// CHECK:STDOUT: !78 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.83aece103b0a8681", scope: null, file: !3, line: 43, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !79 = !DILocation(line: 44, column: 15, scope: !78)
// CHECK:STDOUT: !80 = !DILocation(line: 44, column: 10, scope: !78)
// CHECK:STDOUT: !81 = !DILocation(line: 44, column: 3, scope: !78)
// CHECK:STDOUT: !82 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.0a952f8bcc623ce6", scope: null, file: !3, line: 43, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !83 = !DILocation(line: 44, column: 15, scope: !82)
// CHECK:STDOUT: !84 = !DILocation(line: 44, column: 10, scope: !82)
// CHECK:STDOUT: !85 = !DILocation(line: 44, column: 3, scope: !82)
