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
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.9f1044688d586372(ptr %ptr_f64.var), !dbg !23
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.f427c19f24d378eb(ptr %ptr_i32.var), !dbg !19
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.23895ff6626cffe1(ptr %m.var), !dbg !17
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48(ptr %n.var), !dbg !15
// CHECK:STDOUT:   ret void, !dbg !39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.dc14d4dd21801791:core.Destroy.Core"(ptr %self) #0 !dbg !40 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !42
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !44 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !46
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CA.Main.4d6ceefeb7f772f4(i32 %x, i32 %count) #0 !dbg !48 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 4, !dbg !55
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then.loc23, label %if.else.loc23, !dbg !56
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc23:                                    ; preds = %entry
// CHECK:STDOUT:   ret i32 %x, !dbg !58
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc23:                                    ; preds = %entry
// CHECK:STDOUT:   %Int.as.ModWith.impl.Op.call = srem i32 %count, 2, !dbg !59
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.call = icmp eq i32 %Int.as.ModWith.impl.Op.call, 0, !dbg !59
// CHECK:STDOUT:   br i1 %Int.as.EqWith.impl.Equal.call, label %if.then.loc26, label %if.else.loc26, !dbg !60
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc26:                                    ; preds = %if.else.loc23
// CHECK:STDOUT:   %B.call = call i32 @_CB.Main.4d6ceefeb7f772f4(i32 %x, i32 %count), !dbg !63
// CHECK:STDOUT:   ret i32 %B.call, !dbg !64
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc26:                                    ; preds = %if.else.loc23
// CHECK:STDOUT:   %C.call = call i32 @_CC.Main.4d6ceefeb7f772f4(i32 %x, i32 %count), !dbg !67
// CHECK:STDOUT:   ret i32 %C.call, !dbg !68
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CA.Main.c5e4ba542a56b71f(double %x, i32 %count) #0 !dbg !70 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 4, !dbg !77
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then.loc23, label %if.else.loc23, !dbg !78
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc23:                                    ; preds = %entry
// CHECK:STDOUT:   ret double %x, !dbg !80
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc23:                                    ; preds = %entry
// CHECK:STDOUT:   %Int.as.ModWith.impl.Op.call = srem i32 %count, 2, !dbg !81
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.call = icmp eq i32 %Int.as.ModWith.impl.Op.call, 0, !dbg !81
// CHECK:STDOUT:   br i1 %Int.as.EqWith.impl.Equal.call, label %if.then.loc26, label %if.else.loc26, !dbg !82
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc26:                                    ; preds = %if.else.loc23
// CHECK:STDOUT:   %B.call = call double @_CB.Main.c5e4ba542a56b71f(double %x, i32 %count), !dbg !85
// CHECK:STDOUT:   ret double %B.call, !dbg !86
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc26:                                    ; preds = %if.else.loc23
// CHECK:STDOUT:   %C.call = call double @_CC.Main.c5e4ba542a56b71f(double %x, i32 %count), !dbg !89
// CHECK:STDOUT:   ret double %C.call, !dbg !90
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CA.Main.a76b78461416b4a8(ptr %x, i32 %count) #0 !dbg !92 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 4, !dbg !99
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then.loc23, label %if.else.loc23, !dbg !100
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc23:                                    ; preds = %entry
// CHECK:STDOUT:   ret ptr %x, !dbg !102
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc23:                                    ; preds = %entry
// CHECK:STDOUT:   %Int.as.ModWith.impl.Op.call = srem i32 %count, 2, !dbg !103
// CHECK:STDOUT:   %Int.as.EqWith.impl.Equal.call = icmp eq i32 %Int.as.ModWith.impl.Op.call, 0, !dbg !103
// CHECK:STDOUT:   br i1 %Int.as.EqWith.impl.Equal.call, label %if.then.loc26, label %if.else.loc26, !dbg !104
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc26:                                    ; preds = %if.else.loc23
// CHECK:STDOUT:   %B.call = call ptr @_CB.Main.a76b78461416b4a8(ptr %x, i32 %count), !dbg !107
// CHECK:STDOUT:   ret ptr %B.call, !dbg !108
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc26:                                    ; preds = %if.else.loc23
// CHECK:STDOUT:   %C.call = call ptr @_CC.Main.a76b78461416b4a8(ptr %x, i32 %count), !dbg !111
// CHECK:STDOUT:   ret ptr %C.call, !dbg !112
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.9f1044688d586372(ptr %self) #0 !dbg !137 {
// CHECK:STDOUT:   ret void, !dbg !142
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.f427c19f24d378eb(ptr %self) #0 !dbg !144 {
// CHECK:STDOUT:   ret void, !dbg !149
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.23895ff6626cffe1(ptr %self) #0 !dbg !151 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.dc14d4dd21801791:core.Destroy.Core"(ptr %self), !dbg !155
// CHECK:STDOUT:   ret void, !dbg !156
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48(ptr %self) #0 !dbg !158 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self), !dbg !162
// CHECK:STDOUT:   ret void, !dbg !163
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CB.Main.4d6ceefeb7f772f4(i32 %x, i32 %count) #0 !dbg !165 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 1), !dbg !172
// CHECK:STDOUT:   %D.call = call i32 @_CD.Main.4d6ceefeb7f772f4(i32 %x, i32 %count), !dbg !175
// CHECK:STDOUT:   ret i32 %D.call, !dbg !176
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CC.Main.4d6ceefeb7f772f4(i32 %x, i32 %count) #0 !dbg !178 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 2), !dbg !185
// CHECK:STDOUT:   %D.call = call i32 @_CD.Main.4d6ceefeb7f772f4(i32 %x, i32 %count), !dbg !188
// CHECK:STDOUT:   ret i32 %D.call, !dbg !189
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CB.Main.c5e4ba542a56b71f(double %x, i32 %count) #0 !dbg !191 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 1), !dbg !198
// CHECK:STDOUT:   %D.call = call double @_CD.Main.c5e4ba542a56b71f(double %x, i32 %count), !dbg !201
// CHECK:STDOUT:   ret double %D.call, !dbg !202
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CC.Main.c5e4ba542a56b71f(double %x, i32 %count) #0 !dbg !204 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 2), !dbg !211
// CHECK:STDOUT:   %D.call = call double @_CD.Main.c5e4ba542a56b71f(double %x, i32 %count), !dbg !214
// CHECK:STDOUT:   ret double %D.call, !dbg !215
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CB.Main.a76b78461416b4a8(ptr %x, i32 %count) #0 !dbg !217 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 1), !dbg !224
// CHECK:STDOUT:   %D.call = call ptr @_CD.Main.a76b78461416b4a8(ptr %x, i32 %count), !dbg !227
// CHECK:STDOUT:   ret ptr %D.call, !dbg !228
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CC.Main.a76b78461416b4a8(ptr %x, i32 %count) #0 !dbg !230 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 2), !dbg !237
// CHECK:STDOUT:   %D.call = call ptr @_CD.Main.a76b78461416b4a8(ptr %x, i32 %count), !dbg !240
// CHECK:STDOUT:   ret ptr %D.call, !dbg !241
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @printf(ptr, ...)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CD.Main.4d6ceefeb7f772f4(i32 %x, i32 %count) #0 !dbg !269 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !277
// CHECK:STDOUT:   %A.call = call i32 @_CA.Main.4d6ceefeb7f772f4(i32 %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !278
// CHECK:STDOUT:   ret i32 %A.call, !dbg !279
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CD.Main.c5e4ba542a56b71f(double %x, i32 %count) #0 !dbg !281 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !289
// CHECK:STDOUT:   %A.call = call double @_CA.Main.c5e4ba542a56b71f(double %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !290
// CHECK:STDOUT:   ret double %A.call, !dbg !291
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CD.Main.a76b78461416b4a8(ptr %x, i32 %count) #0 !dbg !293 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !301
// CHECK:STDOUT:   %A.call = call ptr @_CA.Main.a76b78461416b4a8(ptr %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !302
// CHECK:STDOUT:   ret ptr %A.call, !dbg !303
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
// CHECK:STDOUT: !40 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.dc14d4dd21801791:core.Destroy.Core", scope: null, file: !4, line: 50, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !43)
// CHECK:STDOUT: !41 = !DILocalVariable(arg: 1, scope: !40, type: !9)
// CHECK:STDOUT: !42 = !DILocation(line: 50, column: 3, scope: !40)
// CHECK:STDOUT: !43 = !{!41}
// CHECK:STDOUT: !44 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core", scope: null, file: !4, line: 49, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !47)
// CHECK:STDOUT: !45 = !DILocalVariable(arg: 1, scope: !44, type: !6)
// CHECK:STDOUT: !46 = !DILocation(line: 49, column: 3, scope: !44)
// CHECK:STDOUT: !47 = !{!45}
// CHECK:STDOUT: !48 = distinct !DISubprogram(name: "A", linkageName: "_CA.Main.4d6ceefeb7f772f4", scope: null, file: !4, line: 22, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !69)
// CHECK:STDOUT: !49 = !DILocalVariable(arg: 1, scope: !48, type: !6)
// CHECK:STDOUT: !50 = !DILocalVariable(arg: 2, scope: !48, type: !6)
// CHECK:STDOUT: !55 = !DILocation(line: 23, column: 7, scope: !48)
// CHECK:STDOUT: !56 = !DILocation(line: 23, column: 6, scope: !48)
// CHECK:STDOUT: !58 = !DILocation(line: 24, column: 5, scope: !48)
// CHECK:STDOUT: !59 = !DILocation(line: 26, column: 7, scope: !48)
// CHECK:STDOUT: !60 = !DILocation(line: 26, column: 6, scope: !48)
// CHECK:STDOUT: !63 = !DILocation(line: 27, column: 12, scope: !48)
// CHECK:STDOUT: !64 = !DILocation(line: 27, column: 5, scope: !48)
// CHECK:STDOUT: !67 = !DILocation(line: 29, column: 12, scope: !48)
// CHECK:STDOUT: !68 = !DILocation(line: 29, column: 5, scope: !48)
// CHECK:STDOUT: !69 = !{!49, !50}
// CHECK:STDOUT: !70 = distinct !DISubprogram(name: "A", linkageName: "_CA.Main.c5e4ba542a56b71f", scope: null, file: !4, line: 22, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !91)
// CHECK:STDOUT: !71 = !DILocalVariable(arg: 1, scope: !70, type: !9)
// CHECK:STDOUT: !72 = !DILocalVariable(arg: 2, scope: !70, type: !6)
// CHECK:STDOUT: !77 = !DILocation(line: 23, column: 7, scope: !70)
// CHECK:STDOUT: !78 = !DILocation(line: 23, column: 6, scope: !70)
// CHECK:STDOUT: !80 = !DILocation(line: 24, column: 5, scope: !70)
// CHECK:STDOUT: !81 = !DILocation(line: 26, column: 7, scope: !70)
// CHECK:STDOUT: !82 = !DILocation(line: 26, column: 6, scope: !70)
// CHECK:STDOUT: !85 = !DILocation(line: 27, column: 12, scope: !70)
// CHECK:STDOUT: !86 = !DILocation(line: 27, column: 5, scope: !70)
// CHECK:STDOUT: !89 = !DILocation(line: 29, column: 12, scope: !70)
// CHECK:STDOUT: !90 = !DILocation(line: 29, column: 5, scope: !70)
// CHECK:STDOUT: !91 = !{!71, !72}
// CHECK:STDOUT: !92 = distinct !DISubprogram(name: "A", linkageName: "_CA.Main.a76b78461416b4a8", scope: null, file: !4, line: 22, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !113)
// CHECK:STDOUT: !93 = !DILocalVariable(arg: 1, scope: !92, type: !9)
// CHECK:STDOUT: !94 = !DILocalVariable(arg: 2, scope: !92, type: !6)
// CHECK:STDOUT: !99 = !DILocation(line: 23, column: 7, scope: !92)
// CHECK:STDOUT: !100 = !DILocation(line: 23, column: 6, scope: !92)
// CHECK:STDOUT: !102 = !DILocation(line: 24, column: 5, scope: !92)
// CHECK:STDOUT: !103 = !DILocation(line: 26, column: 7, scope: !92)
// CHECK:STDOUT: !104 = !DILocation(line: 26, column: 6, scope: !92)
// CHECK:STDOUT: !107 = !DILocation(line: 27, column: 12, scope: !92)
// CHECK:STDOUT: !108 = !DILocation(line: 27, column: 5, scope: !92)
// CHECK:STDOUT: !111 = !DILocation(line: 29, column: 12, scope: !92)
// CHECK:STDOUT: !112 = !DILocation(line: 29, column: 5, scope: !92)
// CHECK:STDOUT: !113 = !{!93, !94}
// CHECK:STDOUT: !136 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !137 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.9f1044688d586372", scope: null, file: !136, line: 26, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !143)
// CHECK:STDOUT: !138 = !DILocalVariable(arg: 1, scope: !137, type: !9)
// CHECK:STDOUT: !142 = !DILocation(line: 26, column: 3, scope: !137)
// CHECK:STDOUT: !143 = !{!138}
// CHECK:STDOUT: !144 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.f427c19f24d378eb", scope: null, file: !136, line: 26, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !150)
// CHECK:STDOUT: !145 = !DILocalVariable(arg: 1, scope: !144, type: !9)
// CHECK:STDOUT: !149 = !DILocation(line: 26, column: 3, scope: !144)
// CHECK:STDOUT: !150 = !{!145}
// CHECK:STDOUT: !151 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.23895ff6626cffe1", scope: null, file: !136, line: 26, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !157)
// CHECK:STDOUT: !152 = !DILocalVariable(arg: 1, scope: !151, type: !9)
// CHECK:STDOUT: !155 = !DILocation(line: 28, column: 5, scope: !151)
// CHECK:STDOUT: !156 = !DILocation(line: 26, column: 3, scope: !151)
// CHECK:STDOUT: !157 = !{!152}
// CHECK:STDOUT: !158 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48", scope: null, file: !136, line: 26, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !164)
// CHECK:STDOUT: !159 = !DILocalVariable(arg: 1, scope: !158, type: !6)
// CHECK:STDOUT: !162 = !DILocation(line: 28, column: 5, scope: !158)
// CHECK:STDOUT: !163 = !DILocation(line: 26, column: 3, scope: !158)
// CHECK:STDOUT: !164 = !{!159}
// CHECK:STDOUT: !165 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.4d6ceefeb7f772f4", scope: null, file: !4, line: 34, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !177)
// CHECK:STDOUT: !166 = !DILocalVariable(arg: 1, scope: !165, type: !6)
// CHECK:STDOUT: !167 = !DILocalVariable(arg: 2, scope: !165, type: !6)
// CHECK:STDOUT: !172 = !DILocation(line: 35, column: 3, scope: !165)
// CHECK:STDOUT: !175 = !DILocation(line: 36, column: 10, scope: !165)
// CHECK:STDOUT: !176 = !DILocation(line: 36, column: 3, scope: !165)
// CHECK:STDOUT: !177 = !{!166, !167}
// CHECK:STDOUT: !178 = distinct !DISubprogram(name: "C", linkageName: "_CC.Main.4d6ceefeb7f772f4", scope: null, file: !4, line: 39, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !190)
// CHECK:STDOUT: !179 = !DILocalVariable(arg: 1, scope: !178, type: !6)
// CHECK:STDOUT: !180 = !DILocalVariable(arg: 2, scope: !178, type: !6)
// CHECK:STDOUT: !185 = !DILocation(line: 40, column: 3, scope: !178)
// CHECK:STDOUT: !188 = !DILocation(line: 41, column: 10, scope: !178)
// CHECK:STDOUT: !189 = !DILocation(line: 41, column: 3, scope: !178)
// CHECK:STDOUT: !190 = !{!179, !180}
// CHECK:STDOUT: !191 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.c5e4ba542a56b71f", scope: null, file: !4, line: 34, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !203)
// CHECK:STDOUT: !192 = !DILocalVariable(arg: 1, scope: !191, type: !9)
// CHECK:STDOUT: !193 = !DILocalVariable(arg: 2, scope: !191, type: !6)
// CHECK:STDOUT: !198 = !DILocation(line: 35, column: 3, scope: !191)
// CHECK:STDOUT: !201 = !DILocation(line: 36, column: 10, scope: !191)
// CHECK:STDOUT: !202 = !DILocation(line: 36, column: 3, scope: !191)
// CHECK:STDOUT: !203 = !{!192, !193}
// CHECK:STDOUT: !204 = distinct !DISubprogram(name: "C", linkageName: "_CC.Main.c5e4ba542a56b71f", scope: null, file: !4, line: 39, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !216)
// CHECK:STDOUT: !205 = !DILocalVariable(arg: 1, scope: !204, type: !9)
// CHECK:STDOUT: !206 = !DILocalVariable(arg: 2, scope: !204, type: !6)
// CHECK:STDOUT: !211 = !DILocation(line: 40, column: 3, scope: !204)
// CHECK:STDOUT: !214 = !DILocation(line: 41, column: 10, scope: !204)
// CHECK:STDOUT: !215 = !DILocation(line: 41, column: 3, scope: !204)
// CHECK:STDOUT: !216 = !{!205, !206}
// CHECK:STDOUT: !217 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.a76b78461416b4a8", scope: null, file: !4, line: 34, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !229)
// CHECK:STDOUT: !218 = !DILocalVariable(arg: 1, scope: !217, type: !9)
// CHECK:STDOUT: !219 = !DILocalVariable(arg: 2, scope: !217, type: !6)
// CHECK:STDOUT: !224 = !DILocation(line: 35, column: 3, scope: !217)
// CHECK:STDOUT: !227 = !DILocation(line: 36, column: 10, scope: !217)
// CHECK:STDOUT: !228 = !DILocation(line: 36, column: 3, scope: !217)
// CHECK:STDOUT: !229 = !{!218, !219}
// CHECK:STDOUT: !230 = distinct !DISubprogram(name: "C", linkageName: "_CC.Main.a76b78461416b4a8", scope: null, file: !4, line: 39, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !242)
// CHECK:STDOUT: !231 = !DILocalVariable(arg: 1, scope: !230, type: !9)
// CHECK:STDOUT: !232 = !DILocalVariable(arg: 2, scope: !230, type: !6)
// CHECK:STDOUT: !237 = !DILocation(line: 40, column: 3, scope: !230)
// CHECK:STDOUT: !240 = !DILocation(line: 41, column: 10, scope: !230)
// CHECK:STDOUT: !241 = !DILocation(line: 41, column: 3, scope: !230)
// CHECK:STDOUT: !242 = !{!231, !232}
// CHECK:STDOUT: !269 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.4d6ceefeb7f772f4", scope: null, file: !4, line: 44, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !280)
// CHECK:STDOUT: !270 = !DILocalVariable(arg: 1, scope: !269, type: !6)
// CHECK:STDOUT: !271 = !DILocalVariable(arg: 2, scope: !269, type: !6)
// CHECK:STDOUT: !277 = !DILocation(line: 45, column: 15, scope: !269)
// CHECK:STDOUT: !278 = !DILocation(line: 45, column: 10, scope: !269)
// CHECK:STDOUT: !279 = !DILocation(line: 45, column: 3, scope: !269)
// CHECK:STDOUT: !280 = !{!270, !271}
// CHECK:STDOUT: !281 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.c5e4ba542a56b71f", scope: null, file: !4, line: 44, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !292)
// CHECK:STDOUT: !282 = !DILocalVariable(arg: 1, scope: !281, type: !9)
// CHECK:STDOUT: !283 = !DILocalVariable(arg: 2, scope: !281, type: !6)
// CHECK:STDOUT: !289 = !DILocation(line: 45, column: 15, scope: !281)
// CHECK:STDOUT: !290 = !DILocation(line: 45, column: 10, scope: !281)
// CHECK:STDOUT: !291 = !DILocation(line: 45, column: 3, scope: !281)
// CHECK:STDOUT: !292 = !{!282, !283}
// CHECK:STDOUT: !293 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.a76b78461416b4a8", scope: null, file: !4, line: 44, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !304)
// CHECK:STDOUT: !294 = !DILocalVariable(arg: 1, scope: !293, type: !9)
// CHECK:STDOUT: !295 = !DILocalVariable(arg: 2, scope: !293, type: !6)
// CHECK:STDOUT: !301 = !DILocation(line: 45, column: 15, scope: !293)
// CHECK:STDOUT: !302 = !DILocation(line: 45, column: 10, scope: !293)
// CHECK:STDOUT: !303 = !DILocation(line: 45, column: 3, scope: !293)
// CHECK:STDOUT: !304 = !{!294, !295}
// CHECK:STDOUT:
