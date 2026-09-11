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
// CHECK:STDOUT:   %.loc54_5 = load i32, ptr %n.var, align 4, !dbg !11
// CHECK:STDOUT:   %A.call.loc54 = call i32 @_CA.Main.4d6ceefeb7f772f4(i32 %.loc54_5, i32 0), !dbg !12
// CHECK:STDOUT:   %.loc55_5 = load double, ptr %m.var, align 8, !dbg !13
// CHECK:STDOUT:   %A.call.loc55 = call double @_CA.Main.c5e4ba542a56b71f(double %.loc55_5, i32 0), !dbg !14
// CHECK:STDOUT:   %.loc56_5 = load ptr, ptr %ptr_i32.var, align 8, !dbg !15
// CHECK:STDOUT:   %A.call.loc56 = call ptr @_CA.Main.a76b78461416b4a8(ptr %.loc56_5, i32 0), !dbg !16
// CHECK:STDOUT:   %.loc57_5 = load ptr, ptr %ptr_f64.var, align 8, !dbg !17
// CHECK:STDOUT:   %A.call.loc57 = call ptr @_CA.Main.a76b78461416b4a8(ptr %.loc57_5, i32 0), !dbg !18
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.9f1044688d586372(ptr %ptr_f64.var), !dbg !10
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.f427c19f24d378eb(ptr %ptr_i32.var), !dbg !9
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.23895ff6626cffe1(ptr %m.var), !dbg !8
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48(ptr %n.var), !dbg !7
// CHECK:STDOUT:   ret void, !dbg !19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.a1ae3d31ab69a7f8:core.Destroy.Core"(ptr %self) #0 !dbg !20 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.70db2d92ffddbef7:core.Destroy.Core"(ptr %self) #0 !dbg !27 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CA.Main.4d6ceefeb7f772f4(i32 %x, i32 %count) #0 !dbg !34 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 4, !dbg !40
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then.loc23, label %if.else.loc23, !dbg !41
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc23:                                    ; preds = %entry
// CHECK:STDOUT:   ret i32 %x, !dbg !42
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc23:                                    ; preds = %entry
// CHECK:STDOUT:   %Int.as.ModWith.impl.Op.call = srem i32 %count, 2, !dbg !43
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.call = icmp eq i32 %Int.as.ModWith.impl.Op.call, 0, !dbg !43
// CHECK:STDOUT:   br i1 %Int.as.EqWith.impl.Equal.call, label %if.then.loc26, label %if.else.loc26, !dbg !44
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc26:                                    ; preds = %if.else.loc23
// CHECK:STDOUT:   %B.call = call i32 @_CB.Main.4d6ceefeb7f772f4(i32 %x, i32 %count), !dbg !45
// CHECK:STDOUT:   ret i32 %B.call, !dbg !46
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc26:                                    ; preds = %if.else.loc23
// CHECK:STDOUT:   %C.call = call i32 @_CC.Main.4d6ceefeb7f772f4(i32 %x, i32 %count), !dbg !47
// CHECK:STDOUT:   ret i32 %C.call, !dbg !48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CA.Main.c5e4ba542a56b71f(double %x, i32 %count) #0 !dbg !49 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 4, !dbg !55
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then.loc23, label %if.else.loc23, !dbg !56
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc23:                                    ; preds = %entry
// CHECK:STDOUT:   ret double %x, !dbg !57
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc23:                                    ; preds = %entry
// CHECK:STDOUT:   %Int.as.ModWith.impl.Op.call = srem i32 %count, 2, !dbg !58
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.call = icmp eq i32 %Int.as.ModWith.impl.Op.call, 0, !dbg !58
// CHECK:STDOUT:   br i1 %Int.as.EqWith.impl.Equal.call, label %if.then.loc26, label %if.else.loc26, !dbg !59
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc26:                                    ; preds = %if.else.loc23
// CHECK:STDOUT:   %B.call = call double @_CB.Main.c5e4ba542a56b71f(double %x, i32 %count), !dbg !60
// CHECK:STDOUT:   ret double %B.call, !dbg !61
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc26:                                    ; preds = %if.else.loc23
// CHECK:STDOUT:   %C.call = call double @_CC.Main.c5e4ba542a56b71f(double %x, i32 %count), !dbg !62
// CHECK:STDOUT:   ret double %C.call, !dbg !63
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CA.Main.a76b78461416b4a8(ptr %x, i32 %count) #0 !dbg !64 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 4, !dbg !68
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then.loc23, label %if.else.loc23, !dbg !69
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc23:                                    ; preds = %entry
// CHECK:STDOUT:   ret ptr %x, !dbg !70
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc23:                                    ; preds = %entry
// CHECK:STDOUT:   %Int.as.ModWith.impl.Op.call = srem i32 %count, 2, !dbg !71
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.call = icmp eq i32 %Int.as.ModWith.impl.Op.call, 0, !dbg !71
// CHECK:STDOUT:   br i1 %Int.as.EqWith.impl.Equal.call, label %if.then.loc26, label %if.else.loc26, !dbg !72
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc26:                                    ; preds = %if.else.loc23
// CHECK:STDOUT:   %B.call = call ptr @_CB.Main.a76b78461416b4a8(ptr %x, i32 %count), !dbg !73
// CHECK:STDOUT:   ret ptr %B.call, !dbg !74
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc26:                                    ; preds = %if.else.loc23
// CHECK:STDOUT:   %C.call = call ptr @_CC.Main.a76b78461416b4a8(ptr %x, i32 %count), !dbg !75
// CHECK:STDOUT:   ret ptr %C.call, !dbg !76
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.9f1044688d586372(ptr %self) #0 !dbg !77 {
// CHECK:STDOUT:   call void @_CSubobjectDestroy.Destroy.Core.9f1044688d586372(ptr %self), !dbg !81
// CHECK:STDOUT:   ret void, !dbg !82
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.f427c19f24d378eb(ptr %self) #0 !dbg !83 {
// CHECK:STDOUT:   call void @_CSubobjectDestroy.Destroy.Core.9f1044688d586372(ptr %self), !dbg !86
// CHECK:STDOUT:   ret void, !dbg !87
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.23895ff6626cffe1(ptr %self) #0 !dbg !88 {
// CHECK:STDOUT:   call void @"_COp.a1ae3d31ab69a7f8:core.Destroy.Core"(ptr %self), !dbg !91
// CHECK:STDOUT:   call void @_CSubobjectDestroy.Destroy.Core.9f1044688d586372(ptr %self), !dbg !92
// CHECK:STDOUT:   ret void, !dbg !93
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48(ptr %self) #0 !dbg !94 {
// CHECK:STDOUT:   call void @"_COp.70db2d92ffddbef7:core.Destroy.Core"(ptr %self), !dbg !97
// CHECK:STDOUT:   call void @_CSubobjectDestroy.Destroy.Core.9f1044688d586372(ptr %self), !dbg !98
// CHECK:STDOUT:   ret void, !dbg !99
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CB.Main.4d6ceefeb7f772f4(i32 %x, i32 %count) #0 !dbg !100 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 1), !dbg !104
// CHECK:STDOUT:   %D.call = call i32 @_CD.Main.4d6ceefeb7f772f4(i32 %x, i32 %count), !dbg !105
// CHECK:STDOUT:   ret i32 %D.call, !dbg !106
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CC.Main.4d6ceefeb7f772f4(i32 %x, i32 %count) #0 !dbg !107 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 2), !dbg !111
// CHECK:STDOUT:   %D.call = call i32 @_CD.Main.4d6ceefeb7f772f4(i32 %x, i32 %count), !dbg !112
// CHECK:STDOUT:   ret i32 %D.call, !dbg !113
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CB.Main.c5e4ba542a56b71f(double %x, i32 %count) #0 !dbg !114 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 1), !dbg !118
// CHECK:STDOUT:   %D.call = call double @_CD.Main.c5e4ba542a56b71f(double %x, i32 %count), !dbg !119
// CHECK:STDOUT:   ret double %D.call, !dbg !120
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CC.Main.c5e4ba542a56b71f(double %x, i32 %count) #0 !dbg !121 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 2), !dbg !125
// CHECK:STDOUT:   %D.call = call double @_CD.Main.c5e4ba542a56b71f(double %x, i32 %count), !dbg !126
// CHECK:STDOUT:   ret double %D.call, !dbg !127
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CB.Main.a76b78461416b4a8(ptr %x, i32 %count) #0 !dbg !128 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 1), !dbg !132
// CHECK:STDOUT:   %D.call = call ptr @_CD.Main.a76b78461416b4a8(ptr %x, i32 %count), !dbg !133
// CHECK:STDOUT:   ret ptr %D.call, !dbg !134
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CC.Main.a76b78461416b4a8(ptr %x, i32 %count) #0 !dbg !135 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 2), !dbg !139
// CHECK:STDOUT:   %D.call = call ptr @_CD.Main.a76b78461416b4a8(ptr %x, i32 %count), !dbg !140
// CHECK:STDOUT:   ret ptr %D.call, !dbg !141
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSubobjectDestroy.Destroy.Core.9f1044688d586372(ptr %self) #0 !dbg !142 {
// CHECK:STDOUT:   ret void, !dbg !145
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @printf(ptr, ...)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CD.Main.4d6ceefeb7f772f4(i32 %x, i32 %count) #0 !dbg !146 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !150
// CHECK:STDOUT:   %A.call = call i32 @_CA.Main.4d6ceefeb7f772f4(i32 %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !151
// CHECK:STDOUT:   ret i32 %A.call, !dbg !152
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CD.Main.c5e4ba542a56b71f(double %x, i32 %count) #0 !dbg !153 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !157
// CHECK:STDOUT:   %A.call = call double @_CA.Main.c5e4ba542a56b71f(double %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !158
// CHECK:STDOUT:   ret double %A.call, !dbg !159
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CD.Main.a76b78461416b4a8(ptr %x, i32 %count) #0 !dbg !160 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !164
// CHECK:STDOUT:   %A.call = call ptr @_CA.Main.a76b78461416b4a8(ptr %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !165
// CHECK:STDOUT:   ret ptr %A.call, !dbg !166
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CA.Main.a76b78461416b4a8, { 1, 2, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSubobjectDestroy.Destroy.Core.9f1044688d586372, { 3, 2, 0, 1 }
// CHECK:STDOUT: uselistorder ptr @printf, { 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CD.Main.4d6ceefeb7f772f4, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CD.Main.c5e4ba542a56b71f, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CD.Main.a76b78461416b4a8, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "call_recursive_diamond.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "M", linkageName: "_CM.Main", scope: null, file: !1, line: 48, type: !5, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DILocation(line: 49, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 50, column: 3, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 51, column: 3, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 52, column: 3, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 54, column: 5, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 54, column: 3, scope: !4)
// CHECK:STDOUT: !13 = !DILocation(line: 55, column: 5, scope: !4)
// CHECK:STDOUT: !14 = !DILocation(line: 55, column: 3, scope: !4)
// CHECK:STDOUT: !15 = !DILocation(line: 56, column: 5, scope: !4)
// CHECK:STDOUT: !16 = !DILocation(line: 56, column: 3, scope: !4)
// CHECK:STDOUT: !17 = !DILocation(line: 57, column: 5, scope: !4)
// CHECK:STDOUT: !18 = !DILocation(line: 57, column: 3, scope: !4)
// CHECK:STDOUT: !19 = !DILocation(line: 48, column: 1, scope: !4)
// CHECK:STDOUT: !20 = distinct !DISubprogram(name: "Op", linkageName: "_COp.a1ae3d31ab69a7f8:core.Destroy.Core", scope: null, file: !1, line: 50, type: !21, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !24)
// CHECK:STDOUT: !21 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !22 = !{null, !23}
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{!25}
// CHECK:STDOUT: !25 = !DILocalVariable(arg: 1, scope: !20, type: !23)
// CHECK:STDOUT: !26 = !DILocation(line: 50, column: 3, scope: !20)
// CHECK:STDOUT: !27 = distinct !DISubprogram(name: "Op", linkageName: "_COp.70db2d92ffddbef7:core.Destroy.Core", scope: null, file: !1, line: 49, type: !28, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !31)
// CHECK:STDOUT: !28 = !DISubroutineType(types: !29)
// CHECK:STDOUT: !29 = !{null, !30}
// CHECK:STDOUT: !30 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !31 = !{!32}
// CHECK:STDOUT: !32 = !DILocalVariable(arg: 1, scope: !27, type: !30)
// CHECK:STDOUT: !33 = !DILocation(line: 49, column: 3, scope: !27)
// CHECK:STDOUT: !34 = distinct !DISubprogram(name: "A", linkageName: "_CA.Main.4d6ceefeb7f772f4", scope: null, file: !1, line: 22, type: !35, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !37)
// CHECK:STDOUT: !35 = !DISubroutineType(types: !36)
// CHECK:STDOUT: !36 = !{!30, !30, !30}
// CHECK:STDOUT: !37 = !{!38, !39}
// CHECK:STDOUT: !38 = !DILocalVariable(arg: 1, scope: !34, type: !30)
// CHECK:STDOUT: !39 = !DILocalVariable(arg: 2, scope: !34, type: !30)
// CHECK:STDOUT: !40 = !DILocation(line: 23, column: 7, scope: !34)
// CHECK:STDOUT: !41 = !DILocation(line: 23, column: 6, scope: !34)
// CHECK:STDOUT: !42 = !DILocation(line: 24, column: 5, scope: !34)
// CHECK:STDOUT: !43 = !DILocation(line: 26, column: 7, scope: !34)
// CHECK:STDOUT: !44 = !DILocation(line: 26, column: 6, scope: !34)
// CHECK:STDOUT: !45 = !DILocation(line: 27, column: 12, scope: !34)
// CHECK:STDOUT: !46 = !DILocation(line: 27, column: 5, scope: !34)
// CHECK:STDOUT: !47 = !DILocation(line: 29, column: 12, scope: !34)
// CHECK:STDOUT: !48 = !DILocation(line: 29, column: 5, scope: !34)
// CHECK:STDOUT: !49 = distinct !DISubprogram(name: "A", linkageName: "_CA.Main.c5e4ba542a56b71f", scope: null, file: !1, line: 22, type: !50, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !52)
// CHECK:STDOUT: !50 = !DISubroutineType(types: !51)
// CHECK:STDOUT: !51 = !{!23, !23, !30}
// CHECK:STDOUT: !52 = !{!53, !54}
// CHECK:STDOUT: !53 = !DILocalVariable(arg: 1, scope: !49, type: !23)
// CHECK:STDOUT: !54 = !DILocalVariable(arg: 2, scope: !49, type: !30)
// CHECK:STDOUT: !55 = !DILocation(line: 23, column: 7, scope: !49)
// CHECK:STDOUT: !56 = !DILocation(line: 23, column: 6, scope: !49)
// CHECK:STDOUT: !57 = !DILocation(line: 24, column: 5, scope: !49)
// CHECK:STDOUT: !58 = !DILocation(line: 26, column: 7, scope: !49)
// CHECK:STDOUT: !59 = !DILocation(line: 26, column: 6, scope: !49)
// CHECK:STDOUT: !60 = !DILocation(line: 27, column: 12, scope: !49)
// CHECK:STDOUT: !61 = !DILocation(line: 27, column: 5, scope: !49)
// CHECK:STDOUT: !62 = !DILocation(line: 29, column: 12, scope: !49)
// CHECK:STDOUT: !63 = !DILocation(line: 29, column: 5, scope: !49)
// CHECK:STDOUT: !64 = distinct !DISubprogram(name: "A", linkageName: "_CA.Main.a76b78461416b4a8", scope: null, file: !1, line: 22, type: !50, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !65)
// CHECK:STDOUT: !65 = !{!66, !67}
// CHECK:STDOUT: !66 = !DILocalVariable(arg: 1, scope: !64, type: !23)
// CHECK:STDOUT: !67 = !DILocalVariable(arg: 2, scope: !64, type: !30)
// CHECK:STDOUT: !68 = !DILocation(line: 23, column: 7, scope: !64)
// CHECK:STDOUT: !69 = !DILocation(line: 23, column: 6, scope: !64)
// CHECK:STDOUT: !70 = !DILocation(line: 24, column: 5, scope: !64)
// CHECK:STDOUT: !71 = !DILocation(line: 26, column: 7, scope: !64)
// CHECK:STDOUT: !72 = !DILocation(line: 26, column: 6, scope: !64)
// CHECK:STDOUT: !73 = !DILocation(line: 27, column: 12, scope: !64)
// CHECK:STDOUT: !74 = !DILocation(line: 27, column: 5, scope: !64)
// CHECK:STDOUT: !75 = !DILocation(line: 29, column: 12, scope: !64)
// CHECK:STDOUT: !76 = !DILocation(line: 29, column: 5, scope: !64)
// CHECK:STDOUT: !77 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.9f1044688d586372", scope: null, file: !78, line: 26, type: !21, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !79)
// CHECK:STDOUT: !78 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !79 = !{!80}
// CHECK:STDOUT: !80 = !DILocalVariable(arg: 1, scope: !77, type: !23)
// CHECK:STDOUT: !81 = !DILocation(line: 28, column: 5, scope: !77)
// CHECK:STDOUT: !82 = !DILocation(line: 26, column: 3, scope: !77)
// CHECK:STDOUT: !83 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.f427c19f24d378eb", scope: null, file: !78, line: 26, type: !21, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !84)
// CHECK:STDOUT: !84 = !{!85}
// CHECK:STDOUT: !85 = !DILocalVariable(arg: 1, scope: !83, type: !23)
// CHECK:STDOUT: !86 = !DILocation(line: 28, column: 5, scope: !83)
// CHECK:STDOUT: !87 = !DILocation(line: 26, column: 3, scope: !83)
// CHECK:STDOUT: !88 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.23895ff6626cffe1", scope: null, file: !78, line: 26, type: !21, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !89)
// CHECK:STDOUT: !89 = !{!90}
// CHECK:STDOUT: !90 = !DILocalVariable(arg: 1, scope: !88, type: !23)
// CHECK:STDOUT: !91 = !DILocation(line: 27, column: 5, scope: !88)
// CHECK:STDOUT: !92 = !DILocation(line: 28, column: 5, scope: !88)
// CHECK:STDOUT: !93 = !DILocation(line: 26, column: 3, scope: !88)
// CHECK:STDOUT: !94 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48", scope: null, file: !78, line: 26, type: !28, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !95)
// CHECK:STDOUT: !95 = !{!96}
// CHECK:STDOUT: !96 = !DILocalVariable(arg: 1, scope: !94, type: !30)
// CHECK:STDOUT: !97 = !DILocation(line: 27, column: 5, scope: !94)
// CHECK:STDOUT: !98 = !DILocation(line: 28, column: 5, scope: !94)
// CHECK:STDOUT: !99 = !DILocation(line: 26, column: 3, scope: !94)
// CHECK:STDOUT: !100 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.4d6ceefeb7f772f4", scope: null, file: !1, line: 34, type: !35, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !101)
// CHECK:STDOUT: !101 = !{!102, !103}
// CHECK:STDOUT: !102 = !DILocalVariable(arg: 1, scope: !100, type: !30)
// CHECK:STDOUT: !103 = !DILocalVariable(arg: 2, scope: !100, type: !30)
// CHECK:STDOUT: !104 = !DILocation(line: 35, column: 3, scope: !100)
// CHECK:STDOUT: !105 = !DILocation(line: 36, column: 10, scope: !100)
// CHECK:STDOUT: !106 = !DILocation(line: 36, column: 3, scope: !100)
// CHECK:STDOUT: !107 = distinct !DISubprogram(name: "C", linkageName: "_CC.Main.4d6ceefeb7f772f4", scope: null, file: !1, line: 39, type: !35, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !108)
// CHECK:STDOUT: !108 = !{!109, !110}
// CHECK:STDOUT: !109 = !DILocalVariable(arg: 1, scope: !107, type: !30)
// CHECK:STDOUT: !110 = !DILocalVariable(arg: 2, scope: !107, type: !30)
// CHECK:STDOUT: !111 = !DILocation(line: 40, column: 3, scope: !107)
// CHECK:STDOUT: !112 = !DILocation(line: 41, column: 10, scope: !107)
// CHECK:STDOUT: !113 = !DILocation(line: 41, column: 3, scope: !107)
// CHECK:STDOUT: !114 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.c5e4ba542a56b71f", scope: null, file: !1, line: 34, type: !50, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !115)
// CHECK:STDOUT: !115 = !{!116, !117}
// CHECK:STDOUT: !116 = !DILocalVariable(arg: 1, scope: !114, type: !23)
// CHECK:STDOUT: !117 = !DILocalVariable(arg: 2, scope: !114, type: !30)
// CHECK:STDOUT: !118 = !DILocation(line: 35, column: 3, scope: !114)
// CHECK:STDOUT: !119 = !DILocation(line: 36, column: 10, scope: !114)
// CHECK:STDOUT: !120 = !DILocation(line: 36, column: 3, scope: !114)
// CHECK:STDOUT: !121 = distinct !DISubprogram(name: "C", linkageName: "_CC.Main.c5e4ba542a56b71f", scope: null, file: !1, line: 39, type: !50, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !122)
// CHECK:STDOUT: !122 = !{!123, !124}
// CHECK:STDOUT: !123 = !DILocalVariable(arg: 1, scope: !121, type: !23)
// CHECK:STDOUT: !124 = !DILocalVariable(arg: 2, scope: !121, type: !30)
// CHECK:STDOUT: !125 = !DILocation(line: 40, column: 3, scope: !121)
// CHECK:STDOUT: !126 = !DILocation(line: 41, column: 10, scope: !121)
// CHECK:STDOUT: !127 = !DILocation(line: 41, column: 3, scope: !121)
// CHECK:STDOUT: !128 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.a76b78461416b4a8", scope: null, file: !1, line: 34, type: !50, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !129)
// CHECK:STDOUT: !129 = !{!130, !131}
// CHECK:STDOUT: !130 = !DILocalVariable(arg: 1, scope: !128, type: !23)
// CHECK:STDOUT: !131 = !DILocalVariable(arg: 2, scope: !128, type: !30)
// CHECK:STDOUT: !132 = !DILocation(line: 35, column: 3, scope: !128)
// CHECK:STDOUT: !133 = !DILocation(line: 36, column: 10, scope: !128)
// CHECK:STDOUT: !134 = !DILocation(line: 36, column: 3, scope: !128)
// CHECK:STDOUT: !135 = distinct !DISubprogram(name: "C", linkageName: "_CC.Main.a76b78461416b4a8", scope: null, file: !1, line: 39, type: !50, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !136)
// CHECK:STDOUT: !136 = !{!137, !138}
// CHECK:STDOUT: !137 = !DILocalVariable(arg: 1, scope: !135, type: !23)
// CHECK:STDOUT: !138 = !DILocalVariable(arg: 2, scope: !135, type: !30)
// CHECK:STDOUT: !139 = !DILocation(line: 40, column: 3, scope: !135)
// CHECK:STDOUT: !140 = !DILocation(line: 41, column: 10, scope: !135)
// CHECK:STDOUT: !141 = !DILocation(line: 41, column: 3, scope: !135)
// CHECK:STDOUT: !142 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.Destroy.Core.9f1044688d586372", scope: null, file: !78, line: 19, type: !21, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !143)
// CHECK:STDOUT: !143 = !{!144}
// CHECK:STDOUT: !144 = !DILocalVariable(arg: 1, scope: !142, type: !23)
// CHECK:STDOUT: !145 = !DILocation(line: 19, column: 3, scope: !142)
// CHECK:STDOUT: !146 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.4d6ceefeb7f772f4", scope: null, file: !1, line: 44, type: !35, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !147)
// CHECK:STDOUT: !147 = !{!148, !149}
// CHECK:STDOUT: !148 = !DILocalVariable(arg: 1, scope: !146, type: !30)
// CHECK:STDOUT: !149 = !DILocalVariable(arg: 2, scope: !146, type: !30)
// CHECK:STDOUT: !150 = !DILocation(line: 45, column: 15, scope: !146)
// CHECK:STDOUT: !151 = !DILocation(line: 45, column: 10, scope: !146)
// CHECK:STDOUT: !152 = !DILocation(line: 45, column: 3, scope: !146)
// CHECK:STDOUT: !153 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.c5e4ba542a56b71f", scope: null, file: !1, line: 44, type: !50, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !154)
// CHECK:STDOUT: !154 = !{!155, !156}
// CHECK:STDOUT: !155 = !DILocalVariable(arg: 1, scope: !153, type: !23)
// CHECK:STDOUT: !156 = !DILocalVariable(arg: 2, scope: !153, type: !30)
// CHECK:STDOUT: !157 = !DILocation(line: 45, column: 15, scope: !153)
// CHECK:STDOUT: !158 = !DILocation(line: 45, column: 10, scope: !153)
// CHECK:STDOUT: !159 = !DILocation(line: 45, column: 3, scope: !153)
// CHECK:STDOUT: !160 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.a76b78461416b4a8", scope: null, file: !1, line: 44, type: !50, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !161)
// CHECK:STDOUT: !161 = !{!162, !163}
// CHECK:STDOUT: !162 = !DILocalVariable(arg: 1, scope: !160, type: !23)
// CHECK:STDOUT: !163 = !DILocalVariable(arg: 2, scope: !160, type: !30)
// CHECK:STDOUT: !164 = !DILocation(line: 45, column: 15, scope: !160)
// CHECK:STDOUT: !165 = !DILocation(line: 45, column: 10, scope: !160)
// CHECK:STDOUT: !166 = !DILocation(line: 45, column: 3, scope: !160)
// CHECK:STDOUT:
