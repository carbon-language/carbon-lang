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
// CHECK:STDOUT: define void @_CM.Main() #0 !dbg !14 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %n.var = alloca i32, align 4, !dbg !15
// CHECK:STDOUT:   %m.var = alloca double, align 8, !dbg !17
// CHECK:STDOUT:   %ptr_i32.var = alloca ptr, align 8, !dbg !19
// CHECK:STDOUT:   %ptr_f64.var = alloca ptr, align 8, !dbg !23
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %n.var), !dbg !15
// CHECK:STDOUT:   store i32 0, ptr %n.var, align 4, !dbg !15
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %m.var), !dbg !17
// CHECK:STDOUT:   store double 1.000000e+00, ptr %m.var, align 8, !dbg !17
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_i32.var), !dbg !19
// CHECK:STDOUT:   store ptr %n.var, ptr %ptr_i32.var, align 8, !dbg !19
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_f64.var), !dbg !23
// CHECK:STDOUT:   store ptr %m.var, ptr %ptr_f64.var, align 8, !dbg !23
// CHECK:STDOUT:   %.loc54_5 = load i32, ptr %n.var, align 4, !dbg !27
// CHECK:STDOUT:   %A.call.loc54 = call i32 @_CA.Main.4d6ceefeb7f772f4(i32 %.loc54_5, i32 0), !dbg !28
// CHECK:STDOUT:   %.loc55_5 = load double, ptr %m.var, align 8, !dbg !31
// CHECK:STDOUT:   %A.call.loc55 = call double @_CA.Main.c5e4ba542a56b71f(double %.loc55_5, i32 0), !dbg !32
// CHECK:STDOUT:   %.loc56_5 = load ptr, ptr %ptr_i32.var, align 8, !dbg !35
// CHECK:STDOUT:   %A.call.loc56 = call ptr @_CA.Main.a76b78461416b4a8(ptr %.loc56_5, i32 0), !dbg !36
// CHECK:STDOUT:   %.loc57_5 = load ptr, ptr %ptr_f64.var, align 8, !dbg !37
// CHECK:STDOUT:   %A.call.loc57 = call ptr @_CA.Main.a76b78461416b4a8(ptr %.loc57_5, i32 0), !dbg !38
// CHECK:STDOUT:   call void @"_CSelfDestruct.749b51755634ad17:core.Destroy.Core"(ptr %ptr_f64.var), !dbg !23
// CHECK:STDOUT:   call void @"_CSelfDestruct.d088c7eb746488be:core.Destroy.Core"(ptr %ptr_i32.var), !dbg !19
// CHECK:STDOUT:   call void @"_CSelfDestruct.dc14d4dd21801791:core.Destroy.Core"(ptr %m.var), !dbg !17
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %n.var), !dbg !15
// CHECK:STDOUT:   ret void, !dbg !39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.749b51755634ad17:core.Destroy.Core"(ptr %self) #0 !dbg !40 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !42
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.d088c7eb746488be:core.Destroy.Core"(ptr %self) #0 !dbg !44 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !46
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.7945ecba5c505fc0:core.Destroy.Core"(ptr %self) #0 !dbg !48 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !50
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.dc14d4dd21801791:core.Destroy.Core"(ptr %self) #0 !dbg !52 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !54
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.dc14d4dd21801791:core.Destroy.Core"(ptr %self) #0 !dbg !56 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.dc14d4dd21801791:core.Destroy.Core"(ptr %self), !dbg !58
// CHECK:STDOUT:   ret void, !dbg !58
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.6dae29ece9a708c4:core.Destroy.Core"(ptr %self) #0 !dbg !60 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !62
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !64 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !66
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !68 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self), !dbg !70
// CHECK:STDOUT:   ret void, !dbg !70
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CA.Main.4d6ceefeb7f772f4(i32 %x, i32 %count) #0 !dbg !72 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 4, !dbg !79
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then.loc23, label %if.else.loc23, !dbg !80
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc23:                                    ; preds = %entry
// CHECK:STDOUT:   ret i32 %x, !dbg !82
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc23:                                    ; preds = %entry
// CHECK:STDOUT:   %Int.as.ModWith.impl.Op.call = srem i32 %count, 2, !dbg !83
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.call = icmp eq i32 %Int.as.ModWith.impl.Op.call, 0, !dbg !83
// CHECK:STDOUT:   br i1 %Int.as.EqWith.impl.Equal.call, label %if.then.loc26, label %if.else.loc26, !dbg !84
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc26:                                    ; preds = %if.else.loc23
// CHECK:STDOUT:   %B.call = call i32 @_CB.Main.4d6ceefeb7f772f4(i32 %x, i32 %count), !dbg !87
// CHECK:STDOUT:   ret i32 %B.call, !dbg !88
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc26:                                    ; preds = %if.else.loc23
// CHECK:STDOUT:   %C.call = call i32 @_CC.Main.4d6ceefeb7f772f4(i32 %x, i32 %count), !dbg !91
// CHECK:STDOUT:   ret i32 %C.call, !dbg !92
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CA.Main.c5e4ba542a56b71f(double %x, i32 %count) #0 !dbg !94 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 4, !dbg !101
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then.loc23, label %if.else.loc23, !dbg !102
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc23:                                    ; preds = %entry
// CHECK:STDOUT:   ret double %x, !dbg !104
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc23:                                    ; preds = %entry
// CHECK:STDOUT:   %Int.as.ModWith.impl.Op.call = srem i32 %count, 2, !dbg !105
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.call = icmp eq i32 %Int.as.ModWith.impl.Op.call, 0, !dbg !105
// CHECK:STDOUT:   br i1 %Int.as.EqWith.impl.Equal.call, label %if.then.loc26, label %if.else.loc26, !dbg !106
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc26:                                    ; preds = %if.else.loc23
// CHECK:STDOUT:   %B.call = call double @_CB.Main.c5e4ba542a56b71f(double %x, i32 %count), !dbg !109
// CHECK:STDOUT:   ret double %B.call, !dbg !110
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc26:                                    ; preds = %if.else.loc23
// CHECK:STDOUT:   %C.call = call double @_CC.Main.c5e4ba542a56b71f(double %x, i32 %count), !dbg !113
// CHECK:STDOUT:   ret double %C.call, !dbg !114
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CA.Main.a76b78461416b4a8(ptr %x, i32 %count) #0 !dbg !116 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 4, !dbg !123
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then.loc23, label %if.else.loc23, !dbg !124
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc23:                                    ; preds = %entry
// CHECK:STDOUT:   ret ptr %x, !dbg !126
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc23:                                    ; preds = %entry
// CHECK:STDOUT:   %Int.as.ModWith.impl.Op.call = srem i32 %count, 2, !dbg !127
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.call = icmp eq i32 %Int.as.ModWith.impl.Op.call, 0, !dbg !127
// CHECK:STDOUT:   br i1 %Int.as.EqWith.impl.Equal.call, label %if.then.loc26, label %if.else.loc26, !dbg !128
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc26:                                    ; preds = %if.else.loc23
// CHECK:STDOUT:   %B.call = call ptr @_CB.Main.a76b78461416b4a8(ptr %x, i32 %count), !dbg !131
// CHECK:STDOUT:   ret ptr %B.call, !dbg !132
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc26:                                    ; preds = %if.else.loc23
// CHECK:STDOUT:   %C.call = call ptr @_CC.Main.a76b78461416b4a8(ptr %x, i32 %count), !dbg !135
// CHECK:STDOUT:   ret ptr %C.call, !dbg !136
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CB.Main.4d6ceefeb7f772f4(i32 %x, i32 %count) #0 !dbg !160 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 1), !dbg !167
// CHECK:STDOUT:   %D.call = call i32 @_CD.Main.4d6ceefeb7f772f4(i32 %x, i32 %count), !dbg !170
// CHECK:STDOUT:   ret i32 %D.call, !dbg !171
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CC.Main.4d6ceefeb7f772f4(i32 %x, i32 %count) #0 !dbg !173 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 2), !dbg !180
// CHECK:STDOUT:   %D.call = call i32 @_CD.Main.4d6ceefeb7f772f4(i32 %x, i32 %count), !dbg !183
// CHECK:STDOUT:   ret i32 %D.call, !dbg !184
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CB.Main.c5e4ba542a56b71f(double %x, i32 %count) #0 !dbg !186 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 1), !dbg !193
// CHECK:STDOUT:   %D.call = call double @_CD.Main.c5e4ba542a56b71f(double %x, i32 %count), !dbg !196
// CHECK:STDOUT:   ret double %D.call, !dbg !197
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CC.Main.c5e4ba542a56b71f(double %x, i32 %count) #0 !dbg !199 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 2), !dbg !206
// CHECK:STDOUT:   %D.call = call double @_CD.Main.c5e4ba542a56b71f(double %x, i32 %count), !dbg !209
// CHECK:STDOUT:   ret double %D.call, !dbg !210
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CB.Main.a76b78461416b4a8(ptr %x, i32 %count) #0 !dbg !212 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 1), !dbg !219
// CHECK:STDOUT:   %D.call = call ptr @_CD.Main.a76b78461416b4a8(ptr %x, i32 %count), !dbg !222
// CHECK:STDOUT:   ret ptr %D.call, !dbg !223
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CC.Main.a76b78461416b4a8(ptr %x, i32 %count) #0 !dbg !225 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 2), !dbg !232
// CHECK:STDOUT:   %D.call = call ptr @_CD.Main.a76b78461416b4a8(ptr %x, i32 %count), !dbg !235
// CHECK:STDOUT:   ret ptr %D.call, !dbg !236
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @printf(ptr, ...)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CD.Main.4d6ceefeb7f772f4(i32 %x, i32 %count) #0 !dbg !264 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !272
// CHECK:STDOUT:   %A.call = call i32 @_CA.Main.4d6ceefeb7f772f4(i32 %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !273
// CHECK:STDOUT:   ret i32 %A.call, !dbg !274
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CD.Main.c5e4ba542a56b71f(double %x, i32 %count) #0 !dbg !276 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !284
// CHECK:STDOUT:   %A.call = call double @_CA.Main.c5e4ba542a56b71f(double %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !285
// CHECK:STDOUT:   ret double %A.call, !dbg !286
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CD.Main.a76b78461416b4a8(ptr %x, i32 %count) #0 !dbg !288 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !296
// CHECK:STDOUT:   %A.call = call ptr @_CA.Main.a76b78461416b4a8(ptr %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !297
// CHECK:STDOUT:   ret ptr %A.call, !dbg !298
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CA.Main.a76b78461416b4a8, { 1, 2, 0 }
// CHECK:STDOUT: uselistorder ptr @printf, { 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CD.Main.4d6ceefeb7f772f4, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CD.Main.c5e4ba542a56b71f, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CD.Main.a76b78461416b4a8, { 1, 0 }
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
// CHECK:STDOUT: !9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !10 = !{null, !9}
// CHECK:STDOUT: !11 = !DISubroutineType(types: !10)
// CHECK:STDOUT: !12 = !{null, !6}
// CHECK:STDOUT: !13 = !DISubroutineType(types: !12)
// CHECK:STDOUT: !14 = distinct !DISubprogram(name: "M", linkageName: "_CM.Main", scope: null, file: !4, line: 48, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !15 = !DILocation(line: 49, column: 3, scope: !14)
// CHECK:STDOUT: !17 = !DILocation(line: 50, column: 3, scope: !14)
// CHECK:STDOUT: !19 = !DILocation(line: 51, column: 3, scope: !14)
// CHECK:STDOUT: !23 = !DILocation(line: 52, column: 3, scope: !14)
// CHECK:STDOUT: !27 = !DILocation(line: 54, column: 5, scope: !14)
// CHECK:STDOUT: !28 = !DILocation(line: 54, column: 3, scope: !14)
// CHECK:STDOUT: !29 = !{!6, !6, !6}
// CHECK:STDOUT: !30 = !DISubroutineType(types: !29)
// CHECK:STDOUT: !31 = !DILocation(line: 55, column: 5, scope: !14)
// CHECK:STDOUT: !32 = !DILocation(line: 55, column: 3, scope: !14)
// CHECK:STDOUT: !33 = !{!9, !9, !6}
// CHECK:STDOUT: !34 = !DISubroutineType(types: !33)
// CHECK:STDOUT: !35 = !DILocation(line: 56, column: 5, scope: !14)
// CHECK:STDOUT: !36 = !DILocation(line: 56, column: 3, scope: !14)
// CHECK:STDOUT: !37 = !DILocation(line: 57, column: 5, scope: !14)
// CHECK:STDOUT: !38 = !DILocation(line: 57, column: 3, scope: !14)
// CHECK:STDOUT: !39 = !DILocation(line: 48, column: 1, scope: !14)
// CHECK:STDOUT: !40 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.749b51755634ad17:core.Destroy.Core", scope: null, file: !4, line: 52, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !43)
// CHECK:STDOUT: !41 = !DILocalVariable(arg: 1, scope: !40, type: !9)
// CHECK:STDOUT: !42 = !DILocation(line: 52, column: 3, scope: !40)
// CHECK:STDOUT: !43 = !{!41}
// CHECK:STDOUT: !44 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.d088c7eb746488be:core.Destroy.Core", scope: null, file: !4, line: 51, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !47)
// CHECK:STDOUT: !45 = !DILocalVariable(arg: 1, scope: !44, type: !9)
// CHECK:STDOUT: !46 = !DILocation(line: 51, column: 3, scope: !44)
// CHECK:STDOUT: !47 = !{!45}
// CHECK:STDOUT: !48 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.7945ecba5c505fc0:core.Destroy.Core", scope: null, file: !4, line: 50, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !51)
// CHECK:STDOUT: !49 = !DILocalVariable(arg: 1, scope: !48, type: !9)
// CHECK:STDOUT: !50 = !DILocation(line: 50, column: 3, scope: !48)
// CHECK:STDOUT: !51 = !{!49}
// CHECK:STDOUT: !52 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.dc14d4dd21801791:core.Destroy.Core", scope: null, file: !4, line: 50, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !55)
// CHECK:STDOUT: !53 = !DILocalVariable(arg: 1, scope: !52, type: !9)
// CHECK:STDOUT: !54 = !DILocation(line: 50, column: 3, scope: !52)
// CHECK:STDOUT: !55 = !{!53}
// CHECK:STDOUT: !56 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.dc14d4dd21801791:core.Destroy.Core", scope: null, file: !4, line: 50, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !59)
// CHECK:STDOUT: !57 = !DILocalVariable(arg: 1, scope: !56, type: !9)
// CHECK:STDOUT: !58 = !DILocation(line: 50, column: 3, scope: !56)
// CHECK:STDOUT: !59 = !{!57}
// CHECK:STDOUT: !60 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.6dae29ece9a708c4:core.Destroy.Core", scope: null, file: !4, line: 49, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !63)
// CHECK:STDOUT: !61 = !DILocalVariable(arg: 1, scope: !60, type: !6)
// CHECK:STDOUT: !62 = !DILocation(line: 49, column: 3, scope: !60)
// CHECK:STDOUT: !63 = !{!61}
// CHECK:STDOUT: !64 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core", scope: null, file: !4, line: 49, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !67)
// CHECK:STDOUT: !65 = !DILocalVariable(arg: 1, scope: !64, type: !6)
// CHECK:STDOUT: !66 = !DILocation(line: 49, column: 3, scope: !64)
// CHECK:STDOUT: !67 = !{!65}
// CHECK:STDOUT: !68 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f58524831c37807e:core.Destroy.Core", scope: null, file: !4, line: 49, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !71)
// CHECK:STDOUT: !69 = !DILocalVariable(arg: 1, scope: !68, type: !6)
// CHECK:STDOUT: !70 = !DILocation(line: 49, column: 3, scope: !68)
// CHECK:STDOUT: !71 = !{!69}
// CHECK:STDOUT: !72 = distinct !DISubprogram(name: "A", linkageName: "_CA.Main.4d6ceefeb7f772f4", scope: null, file: !4, line: 22, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !93)
// CHECK:STDOUT: !73 = !DILocalVariable(arg: 1, scope: !72, type: !6)
// CHECK:STDOUT: !74 = !DILocalVariable(arg: 2, scope: !72, type: !6)
// CHECK:STDOUT: !79 = !DILocation(line: 23, column: 7, scope: !72)
// CHECK:STDOUT: !80 = !DILocation(line: 23, column: 6, scope: !72)
// CHECK:STDOUT: !82 = !DILocation(line: 24, column: 5, scope: !72)
// CHECK:STDOUT: !83 = !DILocation(line: 26, column: 7, scope: !72)
// CHECK:STDOUT: !84 = !DILocation(line: 26, column: 6, scope: !72)
// CHECK:STDOUT: !87 = !DILocation(line: 27, column: 12, scope: !72)
// CHECK:STDOUT: !88 = !DILocation(line: 27, column: 5, scope: !72)
// CHECK:STDOUT: !91 = !DILocation(line: 29, column: 12, scope: !72)
// CHECK:STDOUT: !92 = !DILocation(line: 29, column: 5, scope: !72)
// CHECK:STDOUT: !93 = !{!73, !74}
// CHECK:STDOUT: !94 = distinct !DISubprogram(name: "A", linkageName: "_CA.Main.c5e4ba542a56b71f", scope: null, file: !4, line: 22, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !115)
// CHECK:STDOUT: !95 = !DILocalVariable(arg: 1, scope: !94, type: !9)
// CHECK:STDOUT: !96 = !DILocalVariable(arg: 2, scope: !94, type: !6)
// CHECK:STDOUT: !101 = !DILocation(line: 23, column: 7, scope: !94)
// CHECK:STDOUT: !102 = !DILocation(line: 23, column: 6, scope: !94)
// CHECK:STDOUT: !104 = !DILocation(line: 24, column: 5, scope: !94)
// CHECK:STDOUT: !105 = !DILocation(line: 26, column: 7, scope: !94)
// CHECK:STDOUT: !106 = !DILocation(line: 26, column: 6, scope: !94)
// CHECK:STDOUT: !109 = !DILocation(line: 27, column: 12, scope: !94)
// CHECK:STDOUT: !110 = !DILocation(line: 27, column: 5, scope: !94)
// CHECK:STDOUT: !113 = !DILocation(line: 29, column: 12, scope: !94)
// CHECK:STDOUT: !114 = !DILocation(line: 29, column: 5, scope: !94)
// CHECK:STDOUT: !115 = !{!95, !96}
// CHECK:STDOUT: !116 = distinct !DISubprogram(name: "A", linkageName: "_CA.Main.a76b78461416b4a8", scope: null, file: !4, line: 22, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !137)
// CHECK:STDOUT: !117 = !DILocalVariable(arg: 1, scope: !116, type: !9)
// CHECK:STDOUT: !118 = !DILocalVariable(arg: 2, scope: !116, type: !6)
// CHECK:STDOUT: !123 = !DILocation(line: 23, column: 7, scope: !116)
// CHECK:STDOUT: !124 = !DILocation(line: 23, column: 6, scope: !116)
// CHECK:STDOUT: !126 = !DILocation(line: 24, column: 5, scope: !116)
// CHECK:STDOUT: !127 = !DILocation(line: 26, column: 7, scope: !116)
// CHECK:STDOUT: !128 = !DILocation(line: 26, column: 6, scope: !116)
// CHECK:STDOUT: !131 = !DILocation(line: 27, column: 12, scope: !116)
// CHECK:STDOUT: !132 = !DILocation(line: 27, column: 5, scope: !116)
// CHECK:STDOUT: !135 = !DILocation(line: 29, column: 12, scope: !116)
// CHECK:STDOUT: !136 = !DILocation(line: 29, column: 5, scope: !116)
// CHECK:STDOUT: !137 = !{!117, !118}
// CHECK:STDOUT: !160 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.4d6ceefeb7f772f4", scope: null, file: !4, line: 34, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !172)
// CHECK:STDOUT: !161 = !DILocalVariable(arg: 1, scope: !160, type: !6)
// CHECK:STDOUT: !162 = !DILocalVariable(arg: 2, scope: !160, type: !6)
// CHECK:STDOUT: !167 = !DILocation(line: 35, column: 3, scope: !160)
// CHECK:STDOUT: !170 = !DILocation(line: 36, column: 10, scope: !160)
// CHECK:STDOUT: !171 = !DILocation(line: 36, column: 3, scope: !160)
// CHECK:STDOUT: !172 = !{!161, !162}
// CHECK:STDOUT: !173 = distinct !DISubprogram(name: "C", linkageName: "_CC.Main.4d6ceefeb7f772f4", scope: null, file: !4, line: 39, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !185)
// CHECK:STDOUT: !174 = !DILocalVariable(arg: 1, scope: !173, type: !6)
// CHECK:STDOUT: !175 = !DILocalVariable(arg: 2, scope: !173, type: !6)
// CHECK:STDOUT: !180 = !DILocation(line: 40, column: 3, scope: !173)
// CHECK:STDOUT: !183 = !DILocation(line: 41, column: 10, scope: !173)
// CHECK:STDOUT: !184 = !DILocation(line: 41, column: 3, scope: !173)
// CHECK:STDOUT: !185 = !{!174, !175}
// CHECK:STDOUT: !186 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.c5e4ba542a56b71f", scope: null, file: !4, line: 34, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !198)
// CHECK:STDOUT: !187 = !DILocalVariable(arg: 1, scope: !186, type: !9)
// CHECK:STDOUT: !188 = !DILocalVariable(arg: 2, scope: !186, type: !6)
// CHECK:STDOUT: !193 = !DILocation(line: 35, column: 3, scope: !186)
// CHECK:STDOUT: !196 = !DILocation(line: 36, column: 10, scope: !186)
// CHECK:STDOUT: !197 = !DILocation(line: 36, column: 3, scope: !186)
// CHECK:STDOUT: !198 = !{!187, !188}
// CHECK:STDOUT: !199 = distinct !DISubprogram(name: "C", linkageName: "_CC.Main.c5e4ba542a56b71f", scope: null, file: !4, line: 39, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !211)
// CHECK:STDOUT: !200 = !DILocalVariable(arg: 1, scope: !199, type: !9)
// CHECK:STDOUT: !201 = !DILocalVariable(arg: 2, scope: !199, type: !6)
// CHECK:STDOUT: !206 = !DILocation(line: 40, column: 3, scope: !199)
// CHECK:STDOUT: !209 = !DILocation(line: 41, column: 10, scope: !199)
// CHECK:STDOUT: !210 = !DILocation(line: 41, column: 3, scope: !199)
// CHECK:STDOUT: !211 = !{!200, !201}
// CHECK:STDOUT: !212 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.a76b78461416b4a8", scope: null, file: !4, line: 34, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !224)
// CHECK:STDOUT: !213 = !DILocalVariable(arg: 1, scope: !212, type: !9)
// CHECK:STDOUT: !214 = !DILocalVariable(arg: 2, scope: !212, type: !6)
// CHECK:STDOUT: !219 = !DILocation(line: 35, column: 3, scope: !212)
// CHECK:STDOUT: !222 = !DILocation(line: 36, column: 10, scope: !212)
// CHECK:STDOUT: !223 = !DILocation(line: 36, column: 3, scope: !212)
// CHECK:STDOUT: !224 = !{!213, !214}
// CHECK:STDOUT: !225 = distinct !DISubprogram(name: "C", linkageName: "_CC.Main.a76b78461416b4a8", scope: null, file: !4, line: 39, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !237)
// CHECK:STDOUT: !226 = !DILocalVariable(arg: 1, scope: !225, type: !9)
// CHECK:STDOUT: !227 = !DILocalVariable(arg: 2, scope: !225, type: !6)
// CHECK:STDOUT: !232 = !DILocation(line: 40, column: 3, scope: !225)
// CHECK:STDOUT: !235 = !DILocation(line: 41, column: 10, scope: !225)
// CHECK:STDOUT: !236 = !DILocation(line: 41, column: 3, scope: !225)
// CHECK:STDOUT: !237 = !{!226, !227}
// CHECK:STDOUT: !264 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.4d6ceefeb7f772f4", scope: null, file: !4, line: 44, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !275)
// CHECK:STDOUT: !265 = !DILocalVariable(arg: 1, scope: !264, type: !6)
// CHECK:STDOUT: !266 = !DILocalVariable(arg: 2, scope: !264, type: !6)
// CHECK:STDOUT: !272 = !DILocation(line: 45, column: 15, scope: !264)
// CHECK:STDOUT: !273 = !DILocation(line: 45, column: 10, scope: !264)
// CHECK:STDOUT: !274 = !DILocation(line: 45, column: 3, scope: !264)
// CHECK:STDOUT: !275 = !{!265, !266}
// CHECK:STDOUT: !276 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.c5e4ba542a56b71f", scope: null, file: !4, line: 44, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !287)
// CHECK:STDOUT: !277 = !DILocalVariable(arg: 1, scope: !276, type: !9)
// CHECK:STDOUT: !278 = !DILocalVariable(arg: 2, scope: !276, type: !6)
// CHECK:STDOUT: !284 = !DILocation(line: 45, column: 15, scope: !276)
// CHECK:STDOUT: !285 = !DILocation(line: 45, column: 10, scope: !276)
// CHECK:STDOUT: !286 = !DILocation(line: 45, column: 3, scope: !276)
// CHECK:STDOUT: !287 = !{!277, !278}
// CHECK:STDOUT: !288 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.a76b78461416b4a8", scope: null, file: !4, line: 44, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !299)
// CHECK:STDOUT: !289 = !DILocalVariable(arg: 1, scope: !288, type: !9)
// CHECK:STDOUT: !290 = !DILocalVariable(arg: 2, scope: !288, type: !6)
// CHECK:STDOUT: !296 = !DILocation(line: 45, column: 15, scope: !288)
// CHECK:STDOUT: !297 = !DILocation(line: 45, column: 10, scope: !288)
// CHECK:STDOUT: !298 = !DILocation(line: 45, column: 3, scope: !288)
// CHECK:STDOUT: !299 = !{!289, !290}
// CHECK:STDOUT:
