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
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.dc14d4dd21801791:core.Destroy.Core"(ptr %self) #0 !dbg !48 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !50
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !52 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !54
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CA.Main.4d6ceefeb7f772f4(i32 %x, i32 %count) #0 !dbg !56 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 4, !dbg !63
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then.loc23, label %if.else.loc23, !dbg !64
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc23:                                    ; preds = %entry
// CHECK:STDOUT:   ret i32 %x, !dbg !66
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc23:                                    ; preds = %entry
// CHECK:STDOUT:   %Int.as.ModWith.impl.Op.call = srem i32 %count, 2, !dbg !67
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.call = icmp eq i32 %Int.as.ModWith.impl.Op.call, 0, !dbg !67
// CHECK:STDOUT:   br i1 %Int.as.EqWith.impl.Equal.call, label %if.then.loc26, label %if.else.loc26, !dbg !68
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc26:                                    ; preds = %if.else.loc23
// CHECK:STDOUT:   %B.call = call i32 @_CB.Main.4d6ceefeb7f772f4(i32 %x, i32 %count), !dbg !71
// CHECK:STDOUT:   ret i32 %B.call, !dbg !72
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc26:                                    ; preds = %if.else.loc23
// CHECK:STDOUT:   %C.call = call i32 @_CC.Main.4d6ceefeb7f772f4(i32 %x, i32 %count), !dbg !75
// CHECK:STDOUT:   ret i32 %C.call, !dbg !76
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CA.Main.c5e4ba542a56b71f(double %x, i32 %count) #0 !dbg !78 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 4, !dbg !85
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then.loc23, label %if.else.loc23, !dbg !86
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc23:                                    ; preds = %entry
// CHECK:STDOUT:   ret double %x, !dbg !88
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc23:                                    ; preds = %entry
// CHECK:STDOUT:   %Int.as.ModWith.impl.Op.call = srem i32 %count, 2, !dbg !89
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.call = icmp eq i32 %Int.as.ModWith.impl.Op.call, 0, !dbg !89
// CHECK:STDOUT:   br i1 %Int.as.EqWith.impl.Equal.call, label %if.then.loc26, label %if.else.loc26, !dbg !90
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc26:                                    ; preds = %if.else.loc23
// CHECK:STDOUT:   %B.call = call double @_CB.Main.c5e4ba542a56b71f(double %x, i32 %count), !dbg !93
// CHECK:STDOUT:   ret double %B.call, !dbg !94
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc26:                                    ; preds = %if.else.loc23
// CHECK:STDOUT:   %C.call = call double @_CC.Main.c5e4ba542a56b71f(double %x, i32 %count), !dbg !97
// CHECK:STDOUT:   ret double %C.call, !dbg !98
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CA.Main.a76b78461416b4a8(ptr %x, i32 %count) #0 !dbg !100 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 4, !dbg !107
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then.loc23, label %if.else.loc23, !dbg !108
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc23:                                    ; preds = %entry
// CHECK:STDOUT:   ret ptr %x, !dbg !110
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc23:                                    ; preds = %entry
// CHECK:STDOUT:   %Int.as.ModWith.impl.Op.call = srem i32 %count, 2, !dbg !111
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.call = icmp eq i32 %Int.as.ModWith.impl.Op.call, 0, !dbg !111
// CHECK:STDOUT:   br i1 %Int.as.EqWith.impl.Equal.call, label %if.then.loc26, label %if.else.loc26, !dbg !112
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc26:                                    ; preds = %if.else.loc23
// CHECK:STDOUT:   %B.call = call ptr @_CB.Main.a76b78461416b4a8(ptr %x, i32 %count), !dbg !115
// CHECK:STDOUT:   ret ptr %B.call, !dbg !116
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc26:                                    ; preds = %if.else.loc23
// CHECK:STDOUT:   %C.call = call ptr @_CC.Main.a76b78461416b4a8(ptr %x, i32 %count), !dbg !119
// CHECK:STDOUT:   ret ptr %C.call, !dbg !120
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CB.Main.4d6ceefeb7f772f4(i32 %x, i32 %count) #0 !dbg !144 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 1), !dbg !151
// CHECK:STDOUT:   %D.call = call i32 @_CD.Main.4d6ceefeb7f772f4(i32 %x, i32 %count), !dbg !154
// CHECK:STDOUT:   ret i32 %D.call, !dbg !155
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CC.Main.4d6ceefeb7f772f4(i32 %x, i32 %count) #0 !dbg !157 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 2), !dbg !164
// CHECK:STDOUT:   %D.call = call i32 @_CD.Main.4d6ceefeb7f772f4(i32 %x, i32 %count), !dbg !167
// CHECK:STDOUT:   ret i32 %D.call, !dbg !168
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CB.Main.c5e4ba542a56b71f(double %x, i32 %count) #0 !dbg !170 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 1), !dbg !177
// CHECK:STDOUT:   %D.call = call double @_CD.Main.c5e4ba542a56b71f(double %x, i32 %count), !dbg !180
// CHECK:STDOUT:   ret double %D.call, !dbg !181
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CC.Main.c5e4ba542a56b71f(double %x, i32 %count) #0 !dbg !183 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 2), !dbg !190
// CHECK:STDOUT:   %D.call = call double @_CD.Main.c5e4ba542a56b71f(double %x, i32 %count), !dbg !193
// CHECK:STDOUT:   ret double %D.call, !dbg !194
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CB.Main.a76b78461416b4a8(ptr %x, i32 %count) #0 !dbg !196 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 1), !dbg !203
// CHECK:STDOUT:   %D.call = call ptr @_CD.Main.a76b78461416b4a8(ptr %x, i32 %count), !dbg !206
// CHECK:STDOUT:   ret ptr %D.call, !dbg !207
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CC.Main.a76b78461416b4a8(ptr %x, i32 %count) #0 !dbg !209 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 2), !dbg !216
// CHECK:STDOUT:   %D.call = call ptr @_CD.Main.a76b78461416b4a8(ptr %x, i32 %count), !dbg !219
// CHECK:STDOUT:   ret ptr %D.call, !dbg !220
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @printf(ptr, ...)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CD.Main.4d6ceefeb7f772f4(i32 %x, i32 %count) #0 !dbg !248 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !256
// CHECK:STDOUT:   %A.call = call i32 @_CA.Main.4d6ceefeb7f772f4(i32 %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !257
// CHECK:STDOUT:   ret i32 %A.call, !dbg !258
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CD.Main.c5e4ba542a56b71f(double %x, i32 %count) #0 !dbg !260 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !268
// CHECK:STDOUT:   %A.call = call double @_CA.Main.c5e4ba542a56b71f(double %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !269
// CHECK:STDOUT:   ret double %A.call, !dbg !270
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CD.Main.a76b78461416b4a8(ptr %x, i32 %count) #0 !dbg !272 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !280
// CHECK:STDOUT:   %A.call = call ptr @_CA.Main.a76b78461416b4a8(ptr %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !281
// CHECK:STDOUT:   ret ptr %A.call, !dbg !282
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
// CHECK:STDOUT: !48 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.dc14d4dd21801791:core.Destroy.Core", scope: null, file: !4, line: 50, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !51)
// CHECK:STDOUT: !49 = !DILocalVariable(arg: 1, scope: !48, type: !9)
// CHECK:STDOUT: !50 = !DILocation(line: 50, column: 3, scope: !48)
// CHECK:STDOUT: !51 = !{!49}
// CHECK:STDOUT: !52 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f58524831c37807e:core.Destroy.Core", scope: null, file: !4, line: 49, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !55)
// CHECK:STDOUT: !53 = !DILocalVariable(arg: 1, scope: !52, type: !6)
// CHECK:STDOUT: !54 = !DILocation(line: 49, column: 3, scope: !52)
// CHECK:STDOUT: !55 = !{!53}
// CHECK:STDOUT: !56 = distinct !DISubprogram(name: "A", linkageName: "_CA.Main.4d6ceefeb7f772f4", scope: null, file: !4, line: 22, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !77)
// CHECK:STDOUT: !57 = !DILocalVariable(arg: 1, scope: !56, type: !6)
// CHECK:STDOUT: !58 = !DILocalVariable(arg: 2, scope: !56, type: !6)
// CHECK:STDOUT: !63 = !DILocation(line: 23, column: 7, scope: !56)
// CHECK:STDOUT: !64 = !DILocation(line: 23, column: 6, scope: !56)
// CHECK:STDOUT: !66 = !DILocation(line: 24, column: 5, scope: !56)
// CHECK:STDOUT: !67 = !DILocation(line: 26, column: 7, scope: !56)
// CHECK:STDOUT: !68 = !DILocation(line: 26, column: 6, scope: !56)
// CHECK:STDOUT: !71 = !DILocation(line: 27, column: 12, scope: !56)
// CHECK:STDOUT: !72 = !DILocation(line: 27, column: 5, scope: !56)
// CHECK:STDOUT: !75 = !DILocation(line: 29, column: 12, scope: !56)
// CHECK:STDOUT: !76 = !DILocation(line: 29, column: 5, scope: !56)
// CHECK:STDOUT: !77 = !{!57, !58}
// CHECK:STDOUT: !78 = distinct !DISubprogram(name: "A", linkageName: "_CA.Main.c5e4ba542a56b71f", scope: null, file: !4, line: 22, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !99)
// CHECK:STDOUT: !79 = !DILocalVariable(arg: 1, scope: !78, type: !9)
// CHECK:STDOUT: !80 = !DILocalVariable(arg: 2, scope: !78, type: !6)
// CHECK:STDOUT: !85 = !DILocation(line: 23, column: 7, scope: !78)
// CHECK:STDOUT: !86 = !DILocation(line: 23, column: 6, scope: !78)
// CHECK:STDOUT: !88 = !DILocation(line: 24, column: 5, scope: !78)
// CHECK:STDOUT: !89 = !DILocation(line: 26, column: 7, scope: !78)
// CHECK:STDOUT: !90 = !DILocation(line: 26, column: 6, scope: !78)
// CHECK:STDOUT: !93 = !DILocation(line: 27, column: 12, scope: !78)
// CHECK:STDOUT: !94 = !DILocation(line: 27, column: 5, scope: !78)
// CHECK:STDOUT: !97 = !DILocation(line: 29, column: 12, scope: !78)
// CHECK:STDOUT: !98 = !DILocation(line: 29, column: 5, scope: !78)
// CHECK:STDOUT: !99 = !{!79, !80}
// CHECK:STDOUT: !100 = distinct !DISubprogram(name: "A", linkageName: "_CA.Main.a76b78461416b4a8", scope: null, file: !4, line: 22, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !121)
// CHECK:STDOUT: !101 = !DILocalVariable(arg: 1, scope: !100, type: !9)
// CHECK:STDOUT: !102 = !DILocalVariable(arg: 2, scope: !100, type: !6)
// CHECK:STDOUT: !107 = !DILocation(line: 23, column: 7, scope: !100)
// CHECK:STDOUT: !108 = !DILocation(line: 23, column: 6, scope: !100)
// CHECK:STDOUT: !110 = !DILocation(line: 24, column: 5, scope: !100)
// CHECK:STDOUT: !111 = !DILocation(line: 26, column: 7, scope: !100)
// CHECK:STDOUT: !112 = !DILocation(line: 26, column: 6, scope: !100)
// CHECK:STDOUT: !115 = !DILocation(line: 27, column: 12, scope: !100)
// CHECK:STDOUT: !116 = !DILocation(line: 27, column: 5, scope: !100)
// CHECK:STDOUT: !119 = !DILocation(line: 29, column: 12, scope: !100)
// CHECK:STDOUT: !120 = !DILocation(line: 29, column: 5, scope: !100)
// CHECK:STDOUT: !121 = !{!101, !102}
// CHECK:STDOUT: !144 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.4d6ceefeb7f772f4", scope: null, file: !4, line: 34, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !156)
// CHECK:STDOUT: !145 = !DILocalVariable(arg: 1, scope: !144, type: !6)
// CHECK:STDOUT: !146 = !DILocalVariable(arg: 2, scope: !144, type: !6)
// CHECK:STDOUT: !151 = !DILocation(line: 35, column: 3, scope: !144)
// CHECK:STDOUT: !154 = !DILocation(line: 36, column: 10, scope: !144)
// CHECK:STDOUT: !155 = !DILocation(line: 36, column: 3, scope: !144)
// CHECK:STDOUT: !156 = !{!145, !146}
// CHECK:STDOUT: !157 = distinct !DISubprogram(name: "C", linkageName: "_CC.Main.4d6ceefeb7f772f4", scope: null, file: !4, line: 39, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !169)
// CHECK:STDOUT: !158 = !DILocalVariable(arg: 1, scope: !157, type: !6)
// CHECK:STDOUT: !159 = !DILocalVariable(arg: 2, scope: !157, type: !6)
// CHECK:STDOUT: !164 = !DILocation(line: 40, column: 3, scope: !157)
// CHECK:STDOUT: !167 = !DILocation(line: 41, column: 10, scope: !157)
// CHECK:STDOUT: !168 = !DILocation(line: 41, column: 3, scope: !157)
// CHECK:STDOUT: !169 = !{!158, !159}
// CHECK:STDOUT: !170 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.c5e4ba542a56b71f", scope: null, file: !4, line: 34, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !182)
// CHECK:STDOUT: !171 = !DILocalVariable(arg: 1, scope: !170, type: !9)
// CHECK:STDOUT: !172 = !DILocalVariable(arg: 2, scope: !170, type: !6)
// CHECK:STDOUT: !177 = !DILocation(line: 35, column: 3, scope: !170)
// CHECK:STDOUT: !180 = !DILocation(line: 36, column: 10, scope: !170)
// CHECK:STDOUT: !181 = !DILocation(line: 36, column: 3, scope: !170)
// CHECK:STDOUT: !182 = !{!171, !172}
// CHECK:STDOUT: !183 = distinct !DISubprogram(name: "C", linkageName: "_CC.Main.c5e4ba542a56b71f", scope: null, file: !4, line: 39, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !195)
// CHECK:STDOUT: !184 = !DILocalVariable(arg: 1, scope: !183, type: !9)
// CHECK:STDOUT: !185 = !DILocalVariable(arg: 2, scope: !183, type: !6)
// CHECK:STDOUT: !190 = !DILocation(line: 40, column: 3, scope: !183)
// CHECK:STDOUT: !193 = !DILocation(line: 41, column: 10, scope: !183)
// CHECK:STDOUT: !194 = !DILocation(line: 41, column: 3, scope: !183)
// CHECK:STDOUT: !195 = !{!184, !185}
// CHECK:STDOUT: !196 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.a76b78461416b4a8", scope: null, file: !4, line: 34, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !208)
// CHECK:STDOUT: !197 = !DILocalVariable(arg: 1, scope: !196, type: !9)
// CHECK:STDOUT: !198 = !DILocalVariable(arg: 2, scope: !196, type: !6)
// CHECK:STDOUT: !203 = !DILocation(line: 35, column: 3, scope: !196)
// CHECK:STDOUT: !206 = !DILocation(line: 36, column: 10, scope: !196)
// CHECK:STDOUT: !207 = !DILocation(line: 36, column: 3, scope: !196)
// CHECK:STDOUT: !208 = !{!197, !198}
// CHECK:STDOUT: !209 = distinct !DISubprogram(name: "C", linkageName: "_CC.Main.a76b78461416b4a8", scope: null, file: !4, line: 39, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !221)
// CHECK:STDOUT: !210 = !DILocalVariable(arg: 1, scope: !209, type: !9)
// CHECK:STDOUT: !211 = !DILocalVariable(arg: 2, scope: !209, type: !6)
// CHECK:STDOUT: !216 = !DILocation(line: 40, column: 3, scope: !209)
// CHECK:STDOUT: !219 = !DILocation(line: 41, column: 10, scope: !209)
// CHECK:STDOUT: !220 = !DILocation(line: 41, column: 3, scope: !209)
// CHECK:STDOUT: !221 = !{!210, !211}
// CHECK:STDOUT: !248 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.4d6ceefeb7f772f4", scope: null, file: !4, line: 44, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !259)
// CHECK:STDOUT: !249 = !DILocalVariable(arg: 1, scope: !248, type: !6)
// CHECK:STDOUT: !250 = !DILocalVariable(arg: 2, scope: !248, type: !6)
// CHECK:STDOUT: !256 = !DILocation(line: 45, column: 15, scope: !248)
// CHECK:STDOUT: !257 = !DILocation(line: 45, column: 10, scope: !248)
// CHECK:STDOUT: !258 = !DILocation(line: 45, column: 3, scope: !248)
// CHECK:STDOUT: !259 = !{!249, !250}
// CHECK:STDOUT: !260 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.c5e4ba542a56b71f", scope: null, file: !4, line: 44, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !271)
// CHECK:STDOUT: !261 = !DILocalVariable(arg: 1, scope: !260, type: !9)
// CHECK:STDOUT: !262 = !DILocalVariable(arg: 2, scope: !260, type: !6)
// CHECK:STDOUT: !268 = !DILocation(line: 45, column: 15, scope: !260)
// CHECK:STDOUT: !269 = !DILocation(line: 45, column: 10, scope: !260)
// CHECK:STDOUT: !270 = !DILocation(line: 45, column: 3, scope: !260)
// CHECK:STDOUT: !271 = !{!261, !262}
// CHECK:STDOUT: !272 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.a76b78461416b4a8", scope: null, file: !4, line: 44, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !283)
// CHECK:STDOUT: !273 = !DILocalVariable(arg: 1, scope: !272, type: !9)
// CHECK:STDOUT: !274 = !DILocalVariable(arg: 2, scope: !272, type: !6)
// CHECK:STDOUT: !280 = !DILocation(line: 45, column: 15, scope: !272)
// CHECK:STDOUT: !281 = !DILocation(line: 45, column: 10, scope: !272)
// CHECK:STDOUT: !282 = !DILocation(line: 45, column: 3, scope: !272)
// CHECK:STDOUT: !283 = !{!273, !274}
// CHECK:STDOUT:
