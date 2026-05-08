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
  var ptr_i32 : i32* = &n;
  var ptr_f64 : f64* = &m;

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
// CHECK:STDOUT:   %.loc53_5 = load i32, ptr %n.var, align 4, !dbg !11
// CHECK:STDOUT:   %A.call.loc53 = call i32 @_CA.Main.5662bbc72901b5f6(i32 %.loc53_5, i32 0), !dbg !12
// CHECK:STDOUT:   %.loc54_5 = load double, ptr %m.var, align 8, !dbg !13
// CHECK:STDOUT:   %A.call.loc54 = call double @_CA.Main.e21e625c49b65a7c(double %.loc54_5, i32 0), !dbg !14
// CHECK:STDOUT:   %.loc55_5 = load ptr, ptr %ptr_i32.var, align 8, !dbg !15
// CHECK:STDOUT:   %A.call.loc55 = call ptr @_CA.Main.3ca635d99c484610(ptr %.loc55_5, i32 0), !dbg !16
// CHECK:STDOUT:   %.loc56_5 = load ptr, ptr %ptr_f64.var, align 8, !dbg !17
// CHECK:STDOUT:   %A.call.loc56 = call ptr @_CA.Main.3ca635d99c484610(ptr %.loc56_5, i32 0), !dbg !18
// CHECK:STDOUT:   call void @"_COp.98006a366c2ffd6c:core.Destroy.Core"(ptr %m.var), !dbg !8
// CHECK:STDOUT:   call void @"_COp.7e389eab4a7e5487:core.Destroy.Core"(ptr %n.var), !dbg !7
// CHECK:STDOUT:   ret void, !dbg !19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.98006a366c2ffd6c:core.Destroy.Core"(ptr %self) #0 !dbg !20 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.7e389eab4a7e5487:core.Destroy.Core"(ptr %self) #0 !dbg !27 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CA.Main.5662bbc72901b5f6(i32 %x, i32 %count) #0 !dbg !34 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 4, !dbg !40
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then.loc22, label %if.else.loc22, !dbg !41
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc22:                                    ; preds = %entry
// CHECK:STDOUT:   ret i32 %x, !dbg !42
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc22:                                    ; preds = %entry
// CHECK:STDOUT:   %Int.as.ModWith.impl.Op.call = srem i32 %count, 2, !dbg !43
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.call = icmp eq i32 %Int.as.ModWith.impl.Op.call, 0, !dbg !43
// CHECK:STDOUT:   br i1 %Int.as.EqWith.impl.Equal.call, label %if.then.loc25, label %if.else.loc25, !dbg !44
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc25:                                    ; preds = %if.else.loc22
// CHECK:STDOUT:   %B.call = call i32 @_CB.Main.5662bbc72901b5f6(i32 %x, i32 %count), !dbg !45
// CHECK:STDOUT:   ret i32 %B.call, !dbg !46
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc25:                                    ; preds = %if.else.loc22
// CHECK:STDOUT:   %C.call = call i32 @_CC.Main.5662bbc72901b5f6(i32 %x, i32 %count), !dbg !47
// CHECK:STDOUT:   ret i32 %C.call, !dbg !48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CA.Main.e21e625c49b65a7c(double %x, i32 %count) #0 !dbg !49 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 4, !dbg !55
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then.loc22, label %if.else.loc22, !dbg !56
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc22:                                    ; preds = %entry
// CHECK:STDOUT:   ret double %x, !dbg !57
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc22:                                    ; preds = %entry
// CHECK:STDOUT:   %Int.as.ModWith.impl.Op.call = srem i32 %count, 2, !dbg !58
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.call = icmp eq i32 %Int.as.ModWith.impl.Op.call, 0, !dbg !58
// CHECK:STDOUT:   br i1 %Int.as.EqWith.impl.Equal.call, label %if.then.loc25, label %if.else.loc25, !dbg !59
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc25:                                    ; preds = %if.else.loc22
// CHECK:STDOUT:   %B.call = call double @_CB.Main.e21e625c49b65a7c(double %x, i32 %count), !dbg !60
// CHECK:STDOUT:   ret double %B.call, !dbg !61
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc25:                                    ; preds = %if.else.loc22
// CHECK:STDOUT:   %C.call = call double @_CC.Main.e21e625c49b65a7c(double %x, i32 %count), !dbg !62
// CHECK:STDOUT:   ret double %C.call, !dbg !63
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CA.Main.3ca635d99c484610(ptr %x, i32 %count) #0 !dbg !64 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 4, !dbg !68
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then.loc22, label %if.else.loc22, !dbg !69
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc22:                                    ; preds = %entry
// CHECK:STDOUT:   ret ptr %x, !dbg !70
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc22:                                    ; preds = %entry
// CHECK:STDOUT:   %Int.as.ModWith.impl.Op.call = srem i32 %count, 2, !dbg !71
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.call = icmp eq i32 %Int.as.ModWith.impl.Op.call, 0, !dbg !71
// CHECK:STDOUT:   br i1 %Int.as.EqWith.impl.Equal.call, label %if.then.loc25, label %if.else.loc25, !dbg !72
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc25:                                    ; preds = %if.else.loc22
// CHECK:STDOUT:   %B.call = call ptr @_CB.Main.3ca635d99c484610(ptr %x, i32 %count), !dbg !73
// CHECK:STDOUT:   ret ptr %B.call, !dbg !74
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc25:                                    ; preds = %if.else.loc22
// CHECK:STDOUT:   %C.call = call ptr @_CC.Main.3ca635d99c484610(ptr %x, i32 %count), !dbg !75
// CHECK:STDOUT:   ret ptr %C.call, !dbg !76
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CB.Main.5662bbc72901b5f6(i32 %x, i32 %count) #0 !dbg !77 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 1), !dbg !81
// CHECK:STDOUT:   %D.call = call i32 @_CD.Main.5662bbc72901b5f6(i32 %x, i32 %count), !dbg !82
// CHECK:STDOUT:   ret i32 %D.call, !dbg !83
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CC.Main.5662bbc72901b5f6(i32 %x, i32 %count) #0 !dbg !84 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 2), !dbg !88
// CHECK:STDOUT:   %D.call = call i32 @_CD.Main.5662bbc72901b5f6(i32 %x, i32 %count), !dbg !89
// CHECK:STDOUT:   ret i32 %D.call, !dbg !90
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CB.Main.e21e625c49b65a7c(double %x, i32 %count) #0 !dbg !91 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 1), !dbg !95
// CHECK:STDOUT:   %D.call = call double @_CD.Main.e21e625c49b65a7c(double %x, i32 %count), !dbg !96
// CHECK:STDOUT:   ret double %D.call, !dbg !97
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CC.Main.e21e625c49b65a7c(double %x, i32 %count) #0 !dbg !98 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 2), !dbg !102
// CHECK:STDOUT:   %D.call = call double @_CD.Main.e21e625c49b65a7c(double %x, i32 %count), !dbg !103
// CHECK:STDOUT:   ret double %D.call, !dbg !104
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CB.Main.3ca635d99c484610(ptr %x, i32 %count) #0 !dbg !105 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 1), !dbg !109
// CHECK:STDOUT:   %D.call = call ptr @_CD.Main.3ca635d99c484610(ptr %x, i32 %count), !dbg !110
// CHECK:STDOUT:   ret ptr %D.call, !dbg !111
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CC.Main.3ca635d99c484610(ptr %x, i32 %count) #0 !dbg !112 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 2), !dbg !116
// CHECK:STDOUT:   %D.call = call ptr @_CD.Main.3ca635d99c484610(ptr %x, i32 %count), !dbg !117
// CHECK:STDOUT:   ret ptr %D.call, !dbg !118
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @printf(ptr, ...)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CD.Main.5662bbc72901b5f6(i32 %x, i32 %count) #0 !dbg !119 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !123
// CHECK:STDOUT:   %A.call = call i32 @_CA.Main.5662bbc72901b5f6(i32 %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !124
// CHECK:STDOUT:   ret i32 %A.call, !dbg !125
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CD.Main.e21e625c49b65a7c(double %x, i32 %count) #0 !dbg !126 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !130
// CHECK:STDOUT:   %A.call = call double @_CA.Main.e21e625c49b65a7c(double %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !131
// CHECK:STDOUT:   ret double %A.call, !dbg !132
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CD.Main.3ca635d99c484610(ptr %x, i32 %count) #0 !dbg !133 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !137
// CHECK:STDOUT:   %A.call = call ptr @_CA.Main.3ca635d99c484610(ptr %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !138
// CHECK:STDOUT:   ret ptr %A.call, !dbg !139
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CA.Main.3ca635d99c484610, { 1, 2, 0 }
// CHECK:STDOUT: uselistorder ptr @printf, { 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CD.Main.5662bbc72901b5f6, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CD.Main.e21e625c49b65a7c, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CD.Main.3ca635d99c484610, { 1, 0 }
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
// CHECK:STDOUT: !3 = !DIFile(filename: "call_recursive_diamond.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "M", linkageName: "_CM.Main", scope: null, file: !3, line: 47, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null}
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
// CHECK:STDOUT: !20 = distinct !DISubprogram(name: "Op", linkageName: "_COp.98006a366c2ffd6c:core.Destroy.Core", scope: null, file: !3, line: 49, type: !21, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !24)
// CHECK:STDOUT: !21 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !22 = !{null, !23}
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{!25}
// CHECK:STDOUT: !25 = !DILocalVariable(arg: 1, scope: !20, type: !23)
// CHECK:STDOUT: !26 = !DILocation(line: 49, column: 3, scope: !20)
// CHECK:STDOUT: !27 = distinct !DISubprogram(name: "Op", linkageName: "_COp.7e389eab4a7e5487:core.Destroy.Core", scope: null, file: !3, line: 48, type: !28, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !31)
// CHECK:STDOUT: !28 = !DISubroutineType(types: !29)
// CHECK:STDOUT: !29 = !{null, !30}
// CHECK:STDOUT: !30 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !31 = !{!32}
// CHECK:STDOUT: !32 = !DILocalVariable(arg: 1, scope: !27, type: !30)
// CHECK:STDOUT: !33 = !DILocation(line: 48, column: 3, scope: !27)
// CHECK:STDOUT: !34 = distinct !DISubprogram(name: "A", linkageName: "_CA.Main.5662bbc72901b5f6", scope: null, file: !3, line: 21, type: !35, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !37)
// CHECK:STDOUT: !35 = !DISubroutineType(types: !36)
// CHECK:STDOUT: !36 = !{!30, !30, !30}
// CHECK:STDOUT: !37 = !{!38, !39}
// CHECK:STDOUT: !38 = !DILocalVariable(arg: 1, scope: !34, type: !30)
// CHECK:STDOUT: !39 = !DILocalVariable(arg: 2, scope: !34, type: !30)
// CHECK:STDOUT: !40 = !DILocation(line: 22, column: 7, scope: !34)
// CHECK:STDOUT: !41 = !DILocation(line: 22, column: 6, scope: !34)
// CHECK:STDOUT: !42 = !DILocation(line: 23, column: 5, scope: !34)
// CHECK:STDOUT: !43 = !DILocation(line: 25, column: 7, scope: !34)
// CHECK:STDOUT: !44 = !DILocation(line: 25, column: 6, scope: !34)
// CHECK:STDOUT: !45 = !DILocation(line: 26, column: 12, scope: !34)
// CHECK:STDOUT: !46 = !DILocation(line: 26, column: 5, scope: !34)
// CHECK:STDOUT: !47 = !DILocation(line: 28, column: 12, scope: !34)
// CHECK:STDOUT: !48 = !DILocation(line: 28, column: 5, scope: !34)
// CHECK:STDOUT: !49 = distinct !DISubprogram(name: "A", linkageName: "_CA.Main.e21e625c49b65a7c", scope: null, file: !3, line: 21, type: !50, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !52)
// CHECK:STDOUT: !50 = !DISubroutineType(types: !51)
// CHECK:STDOUT: !51 = !{!23, !23, !30}
// CHECK:STDOUT: !52 = !{!53, !54}
// CHECK:STDOUT: !53 = !DILocalVariable(arg: 1, scope: !49, type: !23)
// CHECK:STDOUT: !54 = !DILocalVariable(arg: 2, scope: !49, type: !30)
// CHECK:STDOUT: !55 = !DILocation(line: 22, column: 7, scope: !49)
// CHECK:STDOUT: !56 = !DILocation(line: 22, column: 6, scope: !49)
// CHECK:STDOUT: !57 = !DILocation(line: 23, column: 5, scope: !49)
// CHECK:STDOUT: !58 = !DILocation(line: 25, column: 7, scope: !49)
// CHECK:STDOUT: !59 = !DILocation(line: 25, column: 6, scope: !49)
// CHECK:STDOUT: !60 = !DILocation(line: 26, column: 12, scope: !49)
// CHECK:STDOUT: !61 = !DILocation(line: 26, column: 5, scope: !49)
// CHECK:STDOUT: !62 = !DILocation(line: 28, column: 12, scope: !49)
// CHECK:STDOUT: !63 = !DILocation(line: 28, column: 5, scope: !49)
// CHECK:STDOUT: !64 = distinct !DISubprogram(name: "A", linkageName: "_CA.Main.3ca635d99c484610", scope: null, file: !3, line: 21, type: !50, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !65)
// CHECK:STDOUT: !65 = !{!66, !67}
// CHECK:STDOUT: !66 = !DILocalVariable(arg: 1, scope: !64, type: !23)
// CHECK:STDOUT: !67 = !DILocalVariable(arg: 2, scope: !64, type: !30)
// CHECK:STDOUT: !68 = !DILocation(line: 22, column: 7, scope: !64)
// CHECK:STDOUT: !69 = !DILocation(line: 22, column: 6, scope: !64)
// CHECK:STDOUT: !70 = !DILocation(line: 23, column: 5, scope: !64)
// CHECK:STDOUT: !71 = !DILocation(line: 25, column: 7, scope: !64)
// CHECK:STDOUT: !72 = !DILocation(line: 25, column: 6, scope: !64)
// CHECK:STDOUT: !73 = !DILocation(line: 26, column: 12, scope: !64)
// CHECK:STDOUT: !74 = !DILocation(line: 26, column: 5, scope: !64)
// CHECK:STDOUT: !75 = !DILocation(line: 28, column: 12, scope: !64)
// CHECK:STDOUT: !76 = !DILocation(line: 28, column: 5, scope: !64)
// CHECK:STDOUT: !77 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.5662bbc72901b5f6", scope: null, file: !3, line: 33, type: !35, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !78)
// CHECK:STDOUT: !78 = !{!79, !80}
// CHECK:STDOUT: !79 = !DILocalVariable(arg: 1, scope: !77, type: !30)
// CHECK:STDOUT: !80 = !DILocalVariable(arg: 2, scope: !77, type: !30)
// CHECK:STDOUT: !81 = !DILocation(line: 34, column: 3, scope: !77)
// CHECK:STDOUT: !82 = !DILocation(line: 35, column: 10, scope: !77)
// CHECK:STDOUT: !83 = !DILocation(line: 35, column: 3, scope: !77)
// CHECK:STDOUT: !84 = distinct !DISubprogram(name: "C", linkageName: "_CC.Main.5662bbc72901b5f6", scope: null, file: !3, line: 38, type: !35, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !85)
// CHECK:STDOUT: !85 = !{!86, !87}
// CHECK:STDOUT: !86 = !DILocalVariable(arg: 1, scope: !84, type: !30)
// CHECK:STDOUT: !87 = !DILocalVariable(arg: 2, scope: !84, type: !30)
// CHECK:STDOUT: !88 = !DILocation(line: 39, column: 3, scope: !84)
// CHECK:STDOUT: !89 = !DILocation(line: 40, column: 10, scope: !84)
// CHECK:STDOUT: !90 = !DILocation(line: 40, column: 3, scope: !84)
// CHECK:STDOUT: !91 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.e21e625c49b65a7c", scope: null, file: !3, line: 33, type: !50, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !92)
// CHECK:STDOUT: !92 = !{!93, !94}
// CHECK:STDOUT: !93 = !DILocalVariable(arg: 1, scope: !91, type: !23)
// CHECK:STDOUT: !94 = !DILocalVariable(arg: 2, scope: !91, type: !30)
// CHECK:STDOUT: !95 = !DILocation(line: 34, column: 3, scope: !91)
// CHECK:STDOUT: !96 = !DILocation(line: 35, column: 10, scope: !91)
// CHECK:STDOUT: !97 = !DILocation(line: 35, column: 3, scope: !91)
// CHECK:STDOUT: !98 = distinct !DISubprogram(name: "C", linkageName: "_CC.Main.e21e625c49b65a7c", scope: null, file: !3, line: 38, type: !50, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !99)
// CHECK:STDOUT: !99 = !{!100, !101}
// CHECK:STDOUT: !100 = !DILocalVariable(arg: 1, scope: !98, type: !23)
// CHECK:STDOUT: !101 = !DILocalVariable(arg: 2, scope: !98, type: !30)
// CHECK:STDOUT: !102 = !DILocation(line: 39, column: 3, scope: !98)
// CHECK:STDOUT: !103 = !DILocation(line: 40, column: 10, scope: !98)
// CHECK:STDOUT: !104 = !DILocation(line: 40, column: 3, scope: !98)
// CHECK:STDOUT: !105 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.3ca635d99c484610", scope: null, file: !3, line: 33, type: !50, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !106)
// CHECK:STDOUT: !106 = !{!107, !108}
// CHECK:STDOUT: !107 = !DILocalVariable(arg: 1, scope: !105, type: !23)
// CHECK:STDOUT: !108 = !DILocalVariable(arg: 2, scope: !105, type: !30)
// CHECK:STDOUT: !109 = !DILocation(line: 34, column: 3, scope: !105)
// CHECK:STDOUT: !110 = !DILocation(line: 35, column: 10, scope: !105)
// CHECK:STDOUT: !111 = !DILocation(line: 35, column: 3, scope: !105)
// CHECK:STDOUT: !112 = distinct !DISubprogram(name: "C", linkageName: "_CC.Main.3ca635d99c484610", scope: null, file: !3, line: 38, type: !50, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !113)
// CHECK:STDOUT: !113 = !{!114, !115}
// CHECK:STDOUT: !114 = !DILocalVariable(arg: 1, scope: !112, type: !23)
// CHECK:STDOUT: !115 = !DILocalVariable(arg: 2, scope: !112, type: !30)
// CHECK:STDOUT: !116 = !DILocation(line: 39, column: 3, scope: !112)
// CHECK:STDOUT: !117 = !DILocation(line: 40, column: 10, scope: !112)
// CHECK:STDOUT: !118 = !DILocation(line: 40, column: 3, scope: !112)
// CHECK:STDOUT: !119 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.5662bbc72901b5f6", scope: null, file: !3, line: 43, type: !35, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !120)
// CHECK:STDOUT: !120 = !{!121, !122}
// CHECK:STDOUT: !121 = !DILocalVariable(arg: 1, scope: !119, type: !30)
// CHECK:STDOUT: !122 = !DILocalVariable(arg: 2, scope: !119, type: !30)
// CHECK:STDOUT: !123 = !DILocation(line: 44, column: 15, scope: !119)
// CHECK:STDOUT: !124 = !DILocation(line: 44, column: 10, scope: !119)
// CHECK:STDOUT: !125 = !DILocation(line: 44, column: 3, scope: !119)
// CHECK:STDOUT: !126 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.e21e625c49b65a7c", scope: null, file: !3, line: 43, type: !50, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !127)
// CHECK:STDOUT: !127 = !{!128, !129}
// CHECK:STDOUT: !128 = !DILocalVariable(arg: 1, scope: !126, type: !23)
// CHECK:STDOUT: !129 = !DILocalVariable(arg: 2, scope: !126, type: !30)
// CHECK:STDOUT: !130 = !DILocation(line: 44, column: 15, scope: !126)
// CHECK:STDOUT: !131 = !DILocation(line: 44, column: 10, scope: !126)
// CHECK:STDOUT: !132 = !DILocation(line: 44, column: 3, scope: !126)
// CHECK:STDOUT: !133 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.3ca635d99c484610", scope: null, file: !3, line: 43, type: !50, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !134)
// CHECK:STDOUT: !134 = !{!135, !136}
// CHECK:STDOUT: !135 = !DILocalVariable(arg: 1, scope: !133, type: !23)
// CHECK:STDOUT: !136 = !DILocalVariable(arg: 2, scope: !133, type: !30)
// CHECK:STDOUT: !137 = !DILocation(line: 44, column: 15, scope: !133)
// CHECK:STDOUT: !138 = !DILocation(line: 44, column: 10, scope: !133)
// CHECK:STDOUT: !139 = !DILocation(line: 44, column: 3, scope: !133)
