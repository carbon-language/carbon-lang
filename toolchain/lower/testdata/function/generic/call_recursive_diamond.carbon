// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --include-carbon-core
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/generic/call_recursive_diamond.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/generic/call_recursive_diamond.carbon

import Core library "io";
fn A[T: Core.Copy](x: T, count: i32) -> T;
fn B[T: Core.Copy](x: T, count: i32) -> T;
fn C[T: Core.Copy](x: T, count: i32) -> T;
fn D[T: Core.Copy](x: T, count: i32) -> T;

// Builds on `call_recursive_mutual.carbon`. The two specifics for each of:
// A, B, C, D with a pointer type could be deduplicated.
fn A[T: Core.Copy](x: T, count: i32) -> T {
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
fn B[T: Core.Copy](x: T, count: i32) -> T {
  Core.Print(1);
  return D(x, count);
}

fn C[T: Core.Copy](x: T, count: i32) -> T {
  Core.Print(2);
  return D(x, count);
}

fn D[T: Core.Copy](x: T, count: i32) -> T {
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

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'call_recursive_diamond.carbon'
// CHECK:STDOUT: source_filename = "call_recursive_diamond.carbon"
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
// CHECK:STDOUT:   %.loc54_5 = load i32, ptr %n.var, align 4, !dbg !22
// CHECK:STDOUT:   %A.call.loc54 = call i32 @_CA.Main.2fad9574e050607e(i32 %.loc54_5, i32 0), !dbg !23
// CHECK:STDOUT:   %.loc55_5 = load double, ptr %m.var, align 8, !dbg !26
// CHECK:STDOUT:   %A.call.loc55 = call double @_CA.Main.21aefb7dfdebb8e4(double %.loc55_5, i32 0), !dbg !27
// CHECK:STDOUT:   %.loc56_5 = load ptr, ptr %ptr_i32.var, align 8, !dbg !31
// CHECK:STDOUT:   %A.call.loc56 = call ptr @_CA.Main.417c9c4a651ab31b(ptr %.loc56_5, i32 0), !dbg !32
// CHECK:STDOUT:   %.loc57_5 = load ptr, ptr %ptr_f64.var, align 8, !dbg !33
// CHECK:STDOUT:   %A.call.loc57 = call ptr @_CA.Main.417c9c4a651ab31b(ptr %.loc57_5, i32 0), !dbg !34
// CHECK:STDOUT:   ret void, !dbg !35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CA.Main.2fad9574e050607e(i32 %x, i32 %count) #0 !dbg !36 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 4, !dbg !43
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then.loc23, label %if.else.loc23, !dbg !44
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc23:                                    ; preds = %entry
// CHECK:STDOUT:   ret i32 %x, !dbg !46
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc23:                                    ; preds = %entry
// CHECK:STDOUT:   %Int.as.ModWith.impl.Op.call = srem i32 %count, 2, !dbg !47
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.call = icmp eq i32 %Int.as.ModWith.impl.Op.call, 0, !dbg !47
// CHECK:STDOUT:   br i1 %Int.as.EqWith.impl.Equal.call, label %if.then.loc26, label %if.else.loc26, !dbg !48
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc26:                                    ; preds = %if.else.loc23
// CHECK:STDOUT:   %B.call = call i32 @_CB.Main.2fad9574e050607e(i32 %x, i32 %count), !dbg !51
// CHECK:STDOUT:   ret i32 %B.call, !dbg !52
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc26:                                    ; preds = %if.else.loc23
// CHECK:STDOUT:   %C.call = call i32 @_CC.Main.2fad9574e050607e(i32 %x, i32 %count), !dbg !55
// CHECK:STDOUT:   ret i32 %C.call, !dbg !56
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CA.Main.21aefb7dfdebb8e4(double %x, i32 %count) #0 !dbg !58 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 4, !dbg !65
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then.loc23, label %if.else.loc23, !dbg !66
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc23:                                    ; preds = %entry
// CHECK:STDOUT:   ret double %x, !dbg !68
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc23:                                    ; preds = %entry
// CHECK:STDOUT:   %Int.as.ModWith.impl.Op.call = srem i32 %count, 2, !dbg !69
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.call = icmp eq i32 %Int.as.ModWith.impl.Op.call, 0, !dbg !69
// CHECK:STDOUT:   br i1 %Int.as.EqWith.impl.Equal.call, label %if.then.loc26, label %if.else.loc26, !dbg !70
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc26:                                    ; preds = %if.else.loc23
// CHECK:STDOUT:   %B.call = call double @_CB.Main.21aefb7dfdebb8e4(double %x, i32 %count), !dbg !73
// CHECK:STDOUT:   ret double %B.call, !dbg !74
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc26:                                    ; preds = %if.else.loc23
// CHECK:STDOUT:   %C.call = call double @_CC.Main.21aefb7dfdebb8e4(double %x, i32 %count), !dbg !77
// CHECK:STDOUT:   ret double %C.call, !dbg !78
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CA.Main.417c9c4a651ab31b(ptr %x, i32 %count) #0 !dbg !80 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 4, !dbg !87
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then.loc23, label %if.else.loc23, !dbg !88
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc23:                                    ; preds = %entry
// CHECK:STDOUT:   ret ptr %x, !dbg !90
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc23:                                    ; preds = %entry
// CHECK:STDOUT:   %Int.as.ModWith.impl.Op.call = srem i32 %count, 2, !dbg !91
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.call = icmp eq i32 %Int.as.ModWith.impl.Op.call, 0, !dbg !91
// CHECK:STDOUT:   br i1 %Int.as.EqWith.impl.Equal.call, label %if.then.loc26, label %if.else.loc26, !dbg !92
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc26:                                    ; preds = %if.else.loc23
// CHECK:STDOUT:   %B.call = call ptr @_CB.Main.417c9c4a651ab31b(ptr %x, i32 %count), !dbg !95
// CHECK:STDOUT:   ret ptr %B.call, !dbg !96
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc26:                                    ; preds = %if.else.loc23
// CHECK:STDOUT:   %C.call = call ptr @_CC.Main.417c9c4a651ab31b(ptr %x, i32 %count), !dbg !99
// CHECK:STDOUT:   ret ptr %C.call, !dbg !100
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CB.Main.2fad9574e050607e(i32 %x, i32 %count) #0 !dbg !124 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 1), !dbg !131
// CHECK:STDOUT:   %D.call = call i32 @_CD.Main.2fad9574e050607e(i32 %x, i32 %count), !dbg !134
// CHECK:STDOUT:   ret i32 %D.call, !dbg !135
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CC.Main.2fad9574e050607e(i32 %x, i32 %count) #0 !dbg !137 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 2), !dbg !144
// CHECK:STDOUT:   %D.call = call i32 @_CD.Main.2fad9574e050607e(i32 %x, i32 %count), !dbg !147
// CHECK:STDOUT:   ret i32 %D.call, !dbg !148
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CB.Main.21aefb7dfdebb8e4(double %x, i32 %count) #0 !dbg !150 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 1), !dbg !157
// CHECK:STDOUT:   %D.call = call double @_CD.Main.21aefb7dfdebb8e4(double %x, i32 %count), !dbg !160
// CHECK:STDOUT:   ret double %D.call, !dbg !161
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CC.Main.21aefb7dfdebb8e4(double %x, i32 %count) #0 !dbg !163 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 2), !dbg !170
// CHECK:STDOUT:   %D.call = call double @_CD.Main.21aefb7dfdebb8e4(double %x, i32 %count), !dbg !173
// CHECK:STDOUT:   ret double %D.call, !dbg !174
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CB.Main.417c9c4a651ab31b(ptr %x, i32 %count) #0 !dbg !176 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 1), !dbg !183
// CHECK:STDOUT:   %D.call = call ptr @_CD.Main.417c9c4a651ab31b(ptr %x, i32 %count), !dbg !186
// CHECK:STDOUT:   ret ptr %D.call, !dbg !187
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CC.Main.417c9c4a651ab31b(ptr %x, i32 %count) #0 !dbg !189 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 2), !dbg !196
// CHECK:STDOUT:   %D.call = call ptr @_CD.Main.417c9c4a651ab31b(ptr %x, i32 %count), !dbg !199
// CHECK:STDOUT:   ret ptr %D.call, !dbg !200
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @printf(ptr, ...)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CD.Main.2fad9574e050607e(i32 %x, i32 %count) #0 !dbg !228 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !236
// CHECK:STDOUT:   %A.call = call i32 @_CA.Main.2fad9574e050607e(i32 %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !237
// CHECK:STDOUT:   ret i32 %A.call, !dbg !238
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CD.Main.21aefb7dfdebb8e4(double %x, i32 %count) #0 !dbg !240 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !248
// CHECK:STDOUT:   %A.call = call double @_CA.Main.21aefb7dfdebb8e4(double %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !249
// CHECK:STDOUT:   ret double %A.call, !dbg !250
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CD.Main.417c9c4a651ab31b(ptr %x, i32 %count) #0 !dbg !252 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !260
// CHECK:STDOUT:   %A.call = call ptr @_CA.Main.417c9c4a651ab31b(ptr %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !261
// CHECK:STDOUT:   ret ptr %A.call, !dbg !262
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CA.Main.417c9c4a651ab31b, { 1, 2, 0 }
// CHECK:STDOUT: uselistorder ptr @printf, { 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CD.Main.2fad9574e050607e, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CD.Main.21aefb7dfdebb8e4, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CD.Main.417c9c4a651ab31b, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "call_recursive_diamond.carbon", directory: "")
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !7 = !{null}
// CHECK:STDOUT: !8 = !DISubroutineType(types: !7)
// CHECK:STDOUT: !9 = distinct !DISubprogram(name: "M", linkageName: "_CM.Main", scope: null, file: !4, line: 48, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !10 = !DILocation(line: 49, column: 3, scope: !9)
// CHECK:STDOUT: !12 = !DILocation(line: 50, column: 3, scope: !9)
// CHECK:STDOUT: !14 = !DILocation(line: 51, column: 3, scope: !9)
// CHECK:STDOUT: !18 = !DILocation(line: 52, column: 3, scope: !9)
// CHECK:STDOUT: !22 = !DILocation(line: 54, column: 5, scope: !9)
// CHECK:STDOUT: !23 = !DILocation(line: 54, column: 3, scope: !9)
// CHECK:STDOUT: !24 = !{!6, !6, !6}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = !DILocation(line: 55, column: 5, scope: !9)
// CHECK:STDOUT: !27 = !DILocation(line: 55, column: 3, scope: !9)
// CHECK:STDOUT: !28 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !29 = !{!28, !28, !6}
// CHECK:STDOUT: !30 = !DISubroutineType(types: !29)
// CHECK:STDOUT: !31 = !DILocation(line: 56, column: 5, scope: !9)
// CHECK:STDOUT: !32 = !DILocation(line: 56, column: 3, scope: !9)
// CHECK:STDOUT: !33 = !DILocation(line: 57, column: 5, scope: !9)
// CHECK:STDOUT: !34 = !DILocation(line: 57, column: 3, scope: !9)
// CHECK:STDOUT: !35 = !DILocation(line: 48, column: 1, scope: !9)
// CHECK:STDOUT: !36 = distinct !DISubprogram(name: "A", linkageName: "_CA.Main.2fad9574e050607e", scope: null, file: !4, line: 22, type: !25, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !57)
// CHECK:STDOUT: !37 = !DILocalVariable(arg: 1, scope: !36, type: !6)
// CHECK:STDOUT: !38 = !DILocalVariable(arg: 2, scope: !36, type: !6)
// CHECK:STDOUT: !43 = !DILocation(line: 23, column: 7, scope: !36)
// CHECK:STDOUT: !44 = !DILocation(line: 23, column: 6, scope: !36)
// CHECK:STDOUT: !46 = !DILocation(line: 24, column: 5, scope: !36)
// CHECK:STDOUT: !47 = !DILocation(line: 26, column: 7, scope: !36)
// CHECK:STDOUT: !48 = !DILocation(line: 26, column: 6, scope: !36)
// CHECK:STDOUT: !51 = !DILocation(line: 27, column: 12, scope: !36)
// CHECK:STDOUT: !52 = !DILocation(line: 27, column: 5, scope: !36)
// CHECK:STDOUT: !55 = !DILocation(line: 29, column: 12, scope: !36)
// CHECK:STDOUT: !56 = !DILocation(line: 29, column: 5, scope: !36)
// CHECK:STDOUT: !57 = !{!37, !38}
// CHECK:STDOUT: !58 = distinct !DISubprogram(name: "A", linkageName: "_CA.Main.21aefb7dfdebb8e4", scope: null, file: !4, line: 22, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !79)
// CHECK:STDOUT: !59 = !DILocalVariable(arg: 1, scope: !58, type: !28)
// CHECK:STDOUT: !60 = !DILocalVariable(arg: 2, scope: !58, type: !6)
// CHECK:STDOUT: !65 = !DILocation(line: 23, column: 7, scope: !58)
// CHECK:STDOUT: !66 = !DILocation(line: 23, column: 6, scope: !58)
// CHECK:STDOUT: !68 = !DILocation(line: 24, column: 5, scope: !58)
// CHECK:STDOUT: !69 = !DILocation(line: 26, column: 7, scope: !58)
// CHECK:STDOUT: !70 = !DILocation(line: 26, column: 6, scope: !58)
// CHECK:STDOUT: !73 = !DILocation(line: 27, column: 12, scope: !58)
// CHECK:STDOUT: !74 = !DILocation(line: 27, column: 5, scope: !58)
// CHECK:STDOUT: !77 = !DILocation(line: 29, column: 12, scope: !58)
// CHECK:STDOUT: !78 = !DILocation(line: 29, column: 5, scope: !58)
// CHECK:STDOUT: !79 = !{!59, !60}
// CHECK:STDOUT: !80 = distinct !DISubprogram(name: "A", linkageName: "_CA.Main.417c9c4a651ab31b", scope: null, file: !4, line: 22, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !101)
// CHECK:STDOUT: !81 = !DILocalVariable(arg: 1, scope: !80, type: !28)
// CHECK:STDOUT: !82 = !DILocalVariable(arg: 2, scope: !80, type: !6)
// CHECK:STDOUT: !87 = !DILocation(line: 23, column: 7, scope: !80)
// CHECK:STDOUT: !88 = !DILocation(line: 23, column: 6, scope: !80)
// CHECK:STDOUT: !90 = !DILocation(line: 24, column: 5, scope: !80)
// CHECK:STDOUT: !91 = !DILocation(line: 26, column: 7, scope: !80)
// CHECK:STDOUT: !92 = !DILocation(line: 26, column: 6, scope: !80)
// CHECK:STDOUT: !95 = !DILocation(line: 27, column: 12, scope: !80)
// CHECK:STDOUT: !96 = !DILocation(line: 27, column: 5, scope: !80)
// CHECK:STDOUT: !99 = !DILocation(line: 29, column: 12, scope: !80)
// CHECK:STDOUT: !100 = !DILocation(line: 29, column: 5, scope: !80)
// CHECK:STDOUT: !101 = !{!81, !82}
// CHECK:STDOUT: !124 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.2fad9574e050607e", scope: null, file: !4, line: 34, type: !25, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !136)
// CHECK:STDOUT: !125 = !DILocalVariable(arg: 1, scope: !124, type: !6)
// CHECK:STDOUT: !126 = !DILocalVariable(arg: 2, scope: !124, type: !6)
// CHECK:STDOUT: !131 = !DILocation(line: 35, column: 3, scope: !124)
// CHECK:STDOUT: !134 = !DILocation(line: 36, column: 10, scope: !124)
// CHECK:STDOUT: !135 = !DILocation(line: 36, column: 3, scope: !124)
// CHECK:STDOUT: !136 = !{!125, !126}
// CHECK:STDOUT: !137 = distinct !DISubprogram(name: "C", linkageName: "_CC.Main.2fad9574e050607e", scope: null, file: !4, line: 39, type: !25, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !149)
// CHECK:STDOUT: !138 = !DILocalVariable(arg: 1, scope: !137, type: !6)
// CHECK:STDOUT: !139 = !DILocalVariable(arg: 2, scope: !137, type: !6)
// CHECK:STDOUT: !144 = !DILocation(line: 40, column: 3, scope: !137)
// CHECK:STDOUT: !147 = !DILocation(line: 41, column: 10, scope: !137)
// CHECK:STDOUT: !148 = !DILocation(line: 41, column: 3, scope: !137)
// CHECK:STDOUT: !149 = !{!138, !139}
// CHECK:STDOUT: !150 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.21aefb7dfdebb8e4", scope: null, file: !4, line: 34, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !162)
// CHECK:STDOUT: !151 = !DILocalVariable(arg: 1, scope: !150, type: !28)
// CHECK:STDOUT: !152 = !DILocalVariable(arg: 2, scope: !150, type: !6)
// CHECK:STDOUT: !157 = !DILocation(line: 35, column: 3, scope: !150)
// CHECK:STDOUT: !160 = !DILocation(line: 36, column: 10, scope: !150)
// CHECK:STDOUT: !161 = !DILocation(line: 36, column: 3, scope: !150)
// CHECK:STDOUT: !162 = !{!151, !152}
// CHECK:STDOUT: !163 = distinct !DISubprogram(name: "C", linkageName: "_CC.Main.21aefb7dfdebb8e4", scope: null, file: !4, line: 39, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !175)
// CHECK:STDOUT: !164 = !DILocalVariable(arg: 1, scope: !163, type: !28)
// CHECK:STDOUT: !165 = !DILocalVariable(arg: 2, scope: !163, type: !6)
// CHECK:STDOUT: !170 = !DILocation(line: 40, column: 3, scope: !163)
// CHECK:STDOUT: !173 = !DILocation(line: 41, column: 10, scope: !163)
// CHECK:STDOUT: !174 = !DILocation(line: 41, column: 3, scope: !163)
// CHECK:STDOUT: !175 = !{!164, !165}
// CHECK:STDOUT: !176 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.417c9c4a651ab31b", scope: null, file: !4, line: 34, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !188)
// CHECK:STDOUT: !177 = !DILocalVariable(arg: 1, scope: !176, type: !28)
// CHECK:STDOUT: !178 = !DILocalVariable(arg: 2, scope: !176, type: !6)
// CHECK:STDOUT: !183 = !DILocation(line: 35, column: 3, scope: !176)
// CHECK:STDOUT: !186 = !DILocation(line: 36, column: 10, scope: !176)
// CHECK:STDOUT: !187 = !DILocation(line: 36, column: 3, scope: !176)
// CHECK:STDOUT: !188 = !{!177, !178}
// CHECK:STDOUT: !189 = distinct !DISubprogram(name: "C", linkageName: "_CC.Main.417c9c4a651ab31b", scope: null, file: !4, line: 39, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !201)
// CHECK:STDOUT: !190 = !DILocalVariable(arg: 1, scope: !189, type: !28)
// CHECK:STDOUT: !191 = !DILocalVariable(arg: 2, scope: !189, type: !6)
// CHECK:STDOUT: !196 = !DILocation(line: 40, column: 3, scope: !189)
// CHECK:STDOUT: !199 = !DILocation(line: 41, column: 10, scope: !189)
// CHECK:STDOUT: !200 = !DILocation(line: 41, column: 3, scope: !189)
// CHECK:STDOUT: !201 = !{!190, !191}
// CHECK:STDOUT: !228 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.2fad9574e050607e", scope: null, file: !4, line: 44, type: !25, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !239)
// CHECK:STDOUT: !229 = !DILocalVariable(arg: 1, scope: !228, type: !6)
// CHECK:STDOUT: !230 = !DILocalVariable(arg: 2, scope: !228, type: !6)
// CHECK:STDOUT: !236 = !DILocation(line: 45, column: 15, scope: !228)
// CHECK:STDOUT: !237 = !DILocation(line: 45, column: 10, scope: !228)
// CHECK:STDOUT: !238 = !DILocation(line: 45, column: 3, scope: !228)
// CHECK:STDOUT: !239 = !{!229, !230}
// CHECK:STDOUT: !240 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.21aefb7dfdebb8e4", scope: null, file: !4, line: 44, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !251)
// CHECK:STDOUT: !241 = !DILocalVariable(arg: 1, scope: !240, type: !28)
// CHECK:STDOUT: !242 = !DILocalVariable(arg: 2, scope: !240, type: !6)
// CHECK:STDOUT: !248 = !DILocation(line: 45, column: 15, scope: !240)
// CHECK:STDOUT: !249 = !DILocation(line: 45, column: 10, scope: !240)
// CHECK:STDOUT: !250 = !DILocation(line: 45, column: 3, scope: !240)
// CHECK:STDOUT: !251 = !{!241, !242}
// CHECK:STDOUT: !252 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.417c9c4a651ab31b", scope: null, file: !4, line: 44, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !263)
// CHECK:STDOUT: !253 = !DILocalVariable(arg: 1, scope: !252, type: !28)
// CHECK:STDOUT: !254 = !DILocalVariable(arg: 2, scope: !252, type: !6)
// CHECK:STDOUT: !260 = !DILocation(line: 45, column: 15, scope: !252)
// CHECK:STDOUT: !261 = !DILocation(line: 45, column: 10, scope: !252)
// CHECK:STDOUT: !262 = !DILocation(line: 45, column: 3, scope: !252)
// CHECK:STDOUT: !263 = !{!253, !254}
// CHECK:STDOUT:
