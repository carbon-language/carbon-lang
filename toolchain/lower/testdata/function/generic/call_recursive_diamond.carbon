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
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.b18855c76072615b(ptr %ptr_f64.var), !dbg !18
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.4f29d15a20894e9b(ptr %ptr_i32.var), !dbg !14
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.2b38c2ca9f7f1a94(ptr %m.var), !dbg !12
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %n.var), !dbg !10
// CHECK:STDOUT:   ret void, !dbg !39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CA.Main.2fad9574e050607e(i32 %x, i32 %count) #0 !dbg !40 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 4, !dbg !47
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then.loc23, label %if.else.loc23, !dbg !48
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc23:                                    ; preds = %entry
// CHECK:STDOUT:   ret i32 %x, !dbg !50
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc23:                                    ; preds = %entry
// CHECK:STDOUT:   %Int.as.ModWith.impl.Op.call = srem i32 %count, 2, !dbg !51
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.call = icmp eq i32 %Int.as.ModWith.impl.Op.call, 0, !dbg !51
// CHECK:STDOUT:   br i1 %Int.as.EqWith.impl.Equal.call, label %if.then.loc26, label %if.else.loc26, !dbg !52
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc26:                                    ; preds = %if.else.loc23
// CHECK:STDOUT:   %B.call = call i32 @_CB.Main.2fad9574e050607e(i32 %x, i32 %count), !dbg !55
// CHECK:STDOUT:   ret i32 %B.call, !dbg !56
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc26:                                    ; preds = %if.else.loc23
// CHECK:STDOUT:   %C.call = call i32 @_CC.Main.2fad9574e050607e(i32 %x, i32 %count), !dbg !59
// CHECK:STDOUT:   ret i32 %C.call, !dbg !60
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CA.Main.21aefb7dfdebb8e4(double %x, i32 %count) #0 !dbg !62 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 4, !dbg !69
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then.loc23, label %if.else.loc23, !dbg !70
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc23:                                    ; preds = %entry
// CHECK:STDOUT:   ret double %x, !dbg !72
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc23:                                    ; preds = %entry
// CHECK:STDOUT:   %Int.as.ModWith.impl.Op.call = srem i32 %count, 2, !dbg !73
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.call = icmp eq i32 %Int.as.ModWith.impl.Op.call, 0, !dbg !73
// CHECK:STDOUT:   br i1 %Int.as.EqWith.impl.Equal.call, label %if.then.loc26, label %if.else.loc26, !dbg !74
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc26:                                    ; preds = %if.else.loc23
// CHECK:STDOUT:   %B.call = call double @_CB.Main.21aefb7dfdebb8e4(double %x, i32 %count), !dbg !77
// CHECK:STDOUT:   ret double %B.call, !dbg !78
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc26:                                    ; preds = %if.else.loc23
// CHECK:STDOUT:   %C.call = call double @_CC.Main.21aefb7dfdebb8e4(double %x, i32 %count), !dbg !81
// CHECK:STDOUT:   ret double %C.call, !dbg !82
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CA.Main.417c9c4a651ab31b(ptr %x, i32 %count) #0 !dbg !84 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 4, !dbg !91
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then.loc23, label %if.else.loc23, !dbg !92
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc23:                                    ; preds = %entry
// CHECK:STDOUT:   ret ptr %x, !dbg !94
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc23:                                    ; preds = %entry
// CHECK:STDOUT:   %Int.as.ModWith.impl.Op.call = srem i32 %count, 2, !dbg !95
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.call = icmp eq i32 %Int.as.ModWith.impl.Op.call, 0, !dbg !95
// CHECK:STDOUT:   br i1 %Int.as.EqWith.impl.Equal.call, label %if.then.loc26, label %if.else.loc26, !dbg !96
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc26:                                    ; preds = %if.else.loc23
// CHECK:STDOUT:   %B.call = call ptr @_CB.Main.417c9c4a651ab31b(ptr %x, i32 %count), !dbg !99
// CHECK:STDOUT:   ret ptr %B.call, !dbg !100
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc26:                                    ; preds = %if.else.loc23
// CHECK:STDOUT:   %C.call = call ptr @_CC.Main.417c9c4a651ab31b(ptr %x, i32 %count), !dbg !103
// CHECK:STDOUT:   ret ptr %C.call, !dbg !104
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.b18855c76072615b(ptr %t) #0 !dbg !129 {
// CHECK:STDOUT:   ret void, !dbg !134
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.4f29d15a20894e9b(ptr %t) #0 !dbg !136 {
// CHECK:STDOUT:   ret void, !dbg !141
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.2b38c2ca9f7f1a94(ptr %t) #0 !dbg !143 {
// CHECK:STDOUT:   ret void, !dbg !148
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.f127f30143696d51(ptr %t) #0 !dbg !150 {
// CHECK:STDOUT:   ret void, !dbg !155
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CB.Main.2fad9574e050607e(i32 %x, i32 %count) #0 !dbg !157 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 1), !dbg !164
// CHECK:STDOUT:   %D.call = call i32 @_CD.Main.2fad9574e050607e(i32 %x, i32 %count), !dbg !167
// CHECK:STDOUT:   ret i32 %D.call, !dbg !168
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CC.Main.2fad9574e050607e(i32 %x, i32 %count) #0 !dbg !170 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 2), !dbg !177
// CHECK:STDOUT:   %D.call = call i32 @_CD.Main.2fad9574e050607e(i32 %x, i32 %count), !dbg !180
// CHECK:STDOUT:   ret i32 %D.call, !dbg !181
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CB.Main.21aefb7dfdebb8e4(double %x, i32 %count) #0 !dbg !183 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 1), !dbg !190
// CHECK:STDOUT:   %D.call = call double @_CD.Main.21aefb7dfdebb8e4(double %x, i32 %count), !dbg !193
// CHECK:STDOUT:   ret double %D.call, !dbg !194
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CC.Main.21aefb7dfdebb8e4(double %x, i32 %count) #0 !dbg !196 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 2), !dbg !203
// CHECK:STDOUT:   %D.call = call double @_CD.Main.21aefb7dfdebb8e4(double %x, i32 %count), !dbg !206
// CHECK:STDOUT:   ret double %D.call, !dbg !207
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CB.Main.417c9c4a651ab31b(ptr %x, i32 %count) #0 !dbg !209 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 1), !dbg !216
// CHECK:STDOUT:   %D.call = call ptr @_CD.Main.417c9c4a651ab31b(ptr %x, i32 %count), !dbg !219
// CHECK:STDOUT:   ret ptr %D.call, !dbg !220
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CC.Main.417c9c4a651ab31b(ptr %x, i32 %count) #0 !dbg !222 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 2), !dbg !229
// CHECK:STDOUT:   %D.call = call ptr @_CD.Main.417c9c4a651ab31b(ptr %x, i32 %count), !dbg !232
// CHECK:STDOUT:   ret ptr %D.call, !dbg !233
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @printf(ptr, ...)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CD.Main.2fad9574e050607e(i32 %x, i32 %count) #0 !dbg !261 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !269
// CHECK:STDOUT:   %A.call = call i32 @_CA.Main.2fad9574e050607e(i32 %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !270
// CHECK:STDOUT:   ret i32 %A.call, !dbg !271
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CD.Main.21aefb7dfdebb8e4(double %x, i32 %count) #0 !dbg !273 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !281
// CHECK:STDOUT:   %A.call = call double @_CA.Main.21aefb7dfdebb8e4(double %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !282
// CHECK:STDOUT:   ret double %A.call, !dbg !283
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CD.Main.417c9c4a651ab31b(ptr %x, i32 %count) #0 !dbg !285 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !293
// CHECK:STDOUT:   %A.call = call ptr @_CA.Main.417c9c4a651ab31b(ptr %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !294
// CHECK:STDOUT:   ret ptr %A.call, !dbg !295
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
// CHECK:STDOUT: !35 = !{null, !28}
// CHECK:STDOUT: !36 = !DISubroutineType(types: !35)
// CHECK:STDOUT: !37 = !{null, !6}
// CHECK:STDOUT: !38 = !DISubroutineType(types: !37)
// CHECK:STDOUT: !39 = !DILocation(line: 48, column: 1, scope: !9)
// CHECK:STDOUT: !40 = distinct !DISubprogram(name: "A", linkageName: "_CA.Main.2fad9574e050607e", scope: null, file: !4, line: 22, type: !25, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !61)
// CHECK:STDOUT: !41 = !DILocalVariable(arg: 1, scope: !40, type: !6)
// CHECK:STDOUT: !42 = !DILocalVariable(arg: 2, scope: !40, type: !6)
// CHECK:STDOUT: !47 = !DILocation(line: 23, column: 7, scope: !40)
// CHECK:STDOUT: !48 = !DILocation(line: 23, column: 6, scope: !40)
// CHECK:STDOUT: !50 = !DILocation(line: 24, column: 5, scope: !40)
// CHECK:STDOUT: !51 = !DILocation(line: 26, column: 7, scope: !40)
// CHECK:STDOUT: !52 = !DILocation(line: 26, column: 6, scope: !40)
// CHECK:STDOUT: !55 = !DILocation(line: 27, column: 12, scope: !40)
// CHECK:STDOUT: !56 = !DILocation(line: 27, column: 5, scope: !40)
// CHECK:STDOUT: !59 = !DILocation(line: 29, column: 12, scope: !40)
// CHECK:STDOUT: !60 = !DILocation(line: 29, column: 5, scope: !40)
// CHECK:STDOUT: !61 = !{!41, !42}
// CHECK:STDOUT: !62 = distinct !DISubprogram(name: "A", linkageName: "_CA.Main.21aefb7dfdebb8e4", scope: null, file: !4, line: 22, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !83)
// CHECK:STDOUT: !63 = !DILocalVariable(arg: 1, scope: !62, type: !28)
// CHECK:STDOUT: !64 = !DILocalVariable(arg: 2, scope: !62, type: !6)
// CHECK:STDOUT: !69 = !DILocation(line: 23, column: 7, scope: !62)
// CHECK:STDOUT: !70 = !DILocation(line: 23, column: 6, scope: !62)
// CHECK:STDOUT: !72 = !DILocation(line: 24, column: 5, scope: !62)
// CHECK:STDOUT: !73 = !DILocation(line: 26, column: 7, scope: !62)
// CHECK:STDOUT: !74 = !DILocation(line: 26, column: 6, scope: !62)
// CHECK:STDOUT: !77 = !DILocation(line: 27, column: 12, scope: !62)
// CHECK:STDOUT: !78 = !DILocation(line: 27, column: 5, scope: !62)
// CHECK:STDOUT: !81 = !DILocation(line: 29, column: 12, scope: !62)
// CHECK:STDOUT: !82 = !DILocation(line: 29, column: 5, scope: !62)
// CHECK:STDOUT: !83 = !{!63, !64}
// CHECK:STDOUT: !84 = distinct !DISubprogram(name: "A", linkageName: "_CA.Main.417c9c4a651ab31b", scope: null, file: !4, line: 22, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !105)
// CHECK:STDOUT: !85 = !DILocalVariable(arg: 1, scope: !84, type: !28)
// CHECK:STDOUT: !86 = !DILocalVariable(arg: 2, scope: !84, type: !6)
// CHECK:STDOUT: !91 = !DILocation(line: 23, column: 7, scope: !84)
// CHECK:STDOUT: !92 = !DILocation(line: 23, column: 6, scope: !84)
// CHECK:STDOUT: !94 = !DILocation(line: 24, column: 5, scope: !84)
// CHECK:STDOUT: !95 = !DILocation(line: 26, column: 7, scope: !84)
// CHECK:STDOUT: !96 = !DILocation(line: 26, column: 6, scope: !84)
// CHECK:STDOUT: !99 = !DILocation(line: 27, column: 12, scope: !84)
// CHECK:STDOUT: !100 = !DILocation(line: 27, column: 5, scope: !84)
// CHECK:STDOUT: !103 = !DILocation(line: 29, column: 12, scope: !84)
// CHECK:STDOUT: !104 = !DILocation(line: 29, column: 5, scope: !84)
// CHECK:STDOUT: !105 = !{!85, !86}
// CHECK:STDOUT: !128 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !129 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.b18855c76072615b", scope: null, file: !128, line: 35, type: !36, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !135)
// CHECK:STDOUT: !130 = !DILocalVariable(arg: 1, scope: !129, type: !28)
// CHECK:STDOUT: !134 = !DILocation(line: 35, column: 1, scope: !129)
// CHECK:STDOUT: !135 = !{!130}
// CHECK:STDOUT: !136 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.4f29d15a20894e9b", scope: null, file: !128, line: 35, type: !36, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !142)
// CHECK:STDOUT: !137 = !DILocalVariable(arg: 1, scope: !136, type: !28)
// CHECK:STDOUT: !141 = !DILocation(line: 35, column: 1, scope: !136)
// CHECK:STDOUT: !142 = !{!137}
// CHECK:STDOUT: !143 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.2b38c2ca9f7f1a94", scope: null, file: !128, line: 35, type: !36, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !149)
// CHECK:STDOUT: !144 = !DILocalVariable(arg: 1, scope: !143, type: !28)
// CHECK:STDOUT: !148 = !DILocation(line: 35, column: 1, scope: !143)
// CHECK:STDOUT: !149 = !{!144}
// CHECK:STDOUT: !150 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.f127f30143696d51", scope: null, file: !128, line: 35, type: !38, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !156)
// CHECK:STDOUT: !151 = !DILocalVariable(arg: 1, scope: !150, type: !6)
// CHECK:STDOUT: !155 = !DILocation(line: 35, column: 1, scope: !150)
// CHECK:STDOUT: !156 = !{!151}
// CHECK:STDOUT: !157 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.2fad9574e050607e", scope: null, file: !4, line: 34, type: !25, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !169)
// CHECK:STDOUT: !158 = !DILocalVariable(arg: 1, scope: !157, type: !6)
// CHECK:STDOUT: !159 = !DILocalVariable(arg: 2, scope: !157, type: !6)
// CHECK:STDOUT: !164 = !DILocation(line: 35, column: 3, scope: !157)
// CHECK:STDOUT: !167 = !DILocation(line: 36, column: 10, scope: !157)
// CHECK:STDOUT: !168 = !DILocation(line: 36, column: 3, scope: !157)
// CHECK:STDOUT: !169 = !{!158, !159}
// CHECK:STDOUT: !170 = distinct !DISubprogram(name: "C", linkageName: "_CC.Main.2fad9574e050607e", scope: null, file: !4, line: 39, type: !25, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !182)
// CHECK:STDOUT: !171 = !DILocalVariable(arg: 1, scope: !170, type: !6)
// CHECK:STDOUT: !172 = !DILocalVariable(arg: 2, scope: !170, type: !6)
// CHECK:STDOUT: !177 = !DILocation(line: 40, column: 3, scope: !170)
// CHECK:STDOUT: !180 = !DILocation(line: 41, column: 10, scope: !170)
// CHECK:STDOUT: !181 = !DILocation(line: 41, column: 3, scope: !170)
// CHECK:STDOUT: !182 = !{!171, !172}
// CHECK:STDOUT: !183 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.21aefb7dfdebb8e4", scope: null, file: !4, line: 34, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !195)
// CHECK:STDOUT: !184 = !DILocalVariable(arg: 1, scope: !183, type: !28)
// CHECK:STDOUT: !185 = !DILocalVariable(arg: 2, scope: !183, type: !6)
// CHECK:STDOUT: !190 = !DILocation(line: 35, column: 3, scope: !183)
// CHECK:STDOUT: !193 = !DILocation(line: 36, column: 10, scope: !183)
// CHECK:STDOUT: !194 = !DILocation(line: 36, column: 3, scope: !183)
// CHECK:STDOUT: !195 = !{!184, !185}
// CHECK:STDOUT: !196 = distinct !DISubprogram(name: "C", linkageName: "_CC.Main.21aefb7dfdebb8e4", scope: null, file: !4, line: 39, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !208)
// CHECK:STDOUT: !197 = !DILocalVariable(arg: 1, scope: !196, type: !28)
// CHECK:STDOUT: !198 = !DILocalVariable(arg: 2, scope: !196, type: !6)
// CHECK:STDOUT: !203 = !DILocation(line: 40, column: 3, scope: !196)
// CHECK:STDOUT: !206 = !DILocation(line: 41, column: 10, scope: !196)
// CHECK:STDOUT: !207 = !DILocation(line: 41, column: 3, scope: !196)
// CHECK:STDOUT: !208 = !{!197, !198}
// CHECK:STDOUT: !209 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.417c9c4a651ab31b", scope: null, file: !4, line: 34, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !221)
// CHECK:STDOUT: !210 = !DILocalVariable(arg: 1, scope: !209, type: !28)
// CHECK:STDOUT: !211 = !DILocalVariable(arg: 2, scope: !209, type: !6)
// CHECK:STDOUT: !216 = !DILocation(line: 35, column: 3, scope: !209)
// CHECK:STDOUT: !219 = !DILocation(line: 36, column: 10, scope: !209)
// CHECK:STDOUT: !220 = !DILocation(line: 36, column: 3, scope: !209)
// CHECK:STDOUT: !221 = !{!210, !211}
// CHECK:STDOUT: !222 = distinct !DISubprogram(name: "C", linkageName: "_CC.Main.417c9c4a651ab31b", scope: null, file: !4, line: 39, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !234)
// CHECK:STDOUT: !223 = !DILocalVariable(arg: 1, scope: !222, type: !28)
// CHECK:STDOUT: !224 = !DILocalVariable(arg: 2, scope: !222, type: !6)
// CHECK:STDOUT: !229 = !DILocation(line: 40, column: 3, scope: !222)
// CHECK:STDOUT: !232 = !DILocation(line: 41, column: 10, scope: !222)
// CHECK:STDOUT: !233 = !DILocation(line: 41, column: 3, scope: !222)
// CHECK:STDOUT: !234 = !{!223, !224}
// CHECK:STDOUT: !261 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.2fad9574e050607e", scope: null, file: !4, line: 44, type: !25, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !272)
// CHECK:STDOUT: !262 = !DILocalVariable(arg: 1, scope: !261, type: !6)
// CHECK:STDOUT: !263 = !DILocalVariable(arg: 2, scope: !261, type: !6)
// CHECK:STDOUT: !269 = !DILocation(line: 45, column: 15, scope: !261)
// CHECK:STDOUT: !270 = !DILocation(line: 45, column: 10, scope: !261)
// CHECK:STDOUT: !271 = !DILocation(line: 45, column: 3, scope: !261)
// CHECK:STDOUT: !272 = !{!262, !263}
// CHECK:STDOUT: !273 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.21aefb7dfdebb8e4", scope: null, file: !4, line: 44, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !284)
// CHECK:STDOUT: !274 = !DILocalVariable(arg: 1, scope: !273, type: !28)
// CHECK:STDOUT: !275 = !DILocalVariable(arg: 2, scope: !273, type: !6)
// CHECK:STDOUT: !281 = !DILocation(line: 45, column: 15, scope: !273)
// CHECK:STDOUT: !282 = !DILocation(line: 45, column: 10, scope: !273)
// CHECK:STDOUT: !283 = !DILocation(line: 45, column: 3, scope: !273)
// CHECK:STDOUT: !284 = !{!274, !275}
// CHECK:STDOUT: !285 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.417c9c4a651ab31b", scope: null, file: !4, line: 44, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !296)
// CHECK:STDOUT: !286 = !DILocalVariable(arg: 1, scope: !285, type: !28)
// CHECK:STDOUT: !287 = !DILocalVariable(arg: 2, scope: !285, type: !6)
// CHECK:STDOUT: !293 = !DILocation(line: 45, column: 15, scope: !285)
// CHECK:STDOUT: !294 = !DILocation(line: 45, column: 10, scope: !285)
// CHECK:STDOUT: !295 = !DILocation(line: 45, column: 3, scope: !285)
// CHECK:STDOUT: !296 = !{!286, !287}
// CHECK:STDOUT:
