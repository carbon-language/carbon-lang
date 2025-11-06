// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/generic/call_recursive_mutual.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/generic/call_recursive_mutual.carbon

// Builds on `call_recursive_basic.carbon`.
// The two specifics for mutual recursive functions F and G,
// when T is a pointer, could be deduplicated.
fn G[T:! Core.Copy](x: T, count: i32) -> T;

fn F[T:! Core.Copy](x: T, count: i32) -> T {
  if (count > 3) {
    return x;
  }
  return G(x, count + 1);
}

fn G[T:! Core.Copy](x: T, count: i32) -> T {
  if (count > 4) {
    return x;
  }
  return F(x, count + 1);
}

fn M() {
  var n: i32 = 0;
  var m: f64 = 1.0;
  var ptr_i32 : i32*;
  var ptr_f64 : f64*;

  F(n, 0);
  F(m, 0);
  F(ptr_i32, 0);
  F(ptr_f64, 0);
}

// CHECK:STDOUT: ; ModuleID = 'call_recursive_mutual.carbon'
// CHECK:STDOUT: source_filename = "call_recursive_mutual.carbon"
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
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_f64.var), !dbg !10
// CHECK:STDOUT:   %.loc38_5 = load i32, ptr %n.var, align 4, !dbg !11
// CHECK:STDOUT:   %F.call.loc38 = call i32 @_CF.Main.5450dc8e8b8e0899(i32 %.loc38_5, i32 0), !dbg !12
// CHECK:STDOUT:   %.loc39_5 = load double, ptr %m.var, align 8, !dbg !13
// CHECK:STDOUT:   %F.call.loc39 = call double @_CF.Main.83aece103b0a8681(double %.loc39_5, i32 0), !dbg !14
// CHECK:STDOUT:   %.loc40_5 = load ptr, ptr %ptr_i32.var, align 8, !dbg !15
// CHECK:STDOUT:   %F.call.loc40 = call ptr @_CF.Main.0a952f8bcc623ce6(ptr %.loc40_5, i32 0), !dbg !16
// CHECK:STDOUT:   %.loc41_5 = load ptr, ptr %ptr_f64.var, align 8, !dbg !17
// CHECK:STDOUT:   %F.call.loc41 = call ptr @_CF.Main.0a952f8bcc623ce6(ptr %.loc41_5, i32 0), !dbg !18
// CHECK:STDOUT:   ret void, !dbg !19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.5450dc8e8b8e0899(i32 %x, i32 %count) #0 !dbg !20 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 3, !dbg !21
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !22
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret i32 %x, !dbg !23
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !24
// CHECK:STDOUT:   %G.call = call i32 @_CG.Main.5450dc8e8b8e0899(i32 %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !25
// CHECK:STDOUT:   ret i32 %G.call, !dbg !26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CF.Main.83aece103b0a8681(double %x, i32 %count) #0 !dbg !27 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 3, !dbg !28
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !29
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret double %x, !dbg !30
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !31
// CHECK:STDOUT:   %G.call = call double @_CG.Main.83aece103b0a8681(double %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !32
// CHECK:STDOUT:   ret double %G.call, !dbg !33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CF.Main.0a952f8bcc623ce6(ptr %x, i32 %count) #0 !dbg !34 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 3, !dbg !35
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !36
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret ptr %x, !dbg !37
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !38
// CHECK:STDOUT:   %G.call = call ptr @_CG.Main.0a952f8bcc623ce6(ptr %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !39
// CHECK:STDOUT:   ret ptr %G.call, !dbg !40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CG.Main.5450dc8e8b8e0899(i32 %x, i32 %count) #0 !dbg !41 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 4, !dbg !42
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !43
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret i32 %x, !dbg !44
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !45
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main.5450dc8e8b8e0899(i32 %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !46
// CHECK:STDOUT:   ret i32 %F.call, !dbg !47
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CG.Main.83aece103b0a8681(double %x, i32 %count) #0 !dbg !48 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 4, !dbg !49
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !50
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret double %x, !dbg !51
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !52
// CHECK:STDOUT:   %F.call = call double @_CF.Main.83aece103b0a8681(double %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !53
// CHECK:STDOUT:   ret double %F.call, !dbg !54
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CG.Main.0a952f8bcc623ce6(ptr %x, i32 %count) #0 !dbg !55 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 4, !dbg !56
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !57
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret ptr %x, !dbg !58
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !59
// CHECK:STDOUT:   %F.call = call ptr @_CF.Main.0a952f8bcc623ce6(ptr %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !60
// CHECK:STDOUT:   ret ptr %F.call, !dbg !61
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CF.Main.0a952f8bcc623ce6, { 1, 2, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "call_recursive_mutual.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "M", linkageName: "_CM.Main", scope: null, file: !3, line: 32, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 33, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 34, column: 3, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 35, column: 3, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 36, column: 3, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 38, column: 5, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 38, column: 3, scope: !4)
// CHECK:STDOUT: !13 = !DILocation(line: 39, column: 5, scope: !4)
// CHECK:STDOUT: !14 = !DILocation(line: 39, column: 3, scope: !4)
// CHECK:STDOUT: !15 = !DILocation(line: 40, column: 5, scope: !4)
// CHECK:STDOUT: !16 = !DILocation(line: 40, column: 3, scope: !4)
// CHECK:STDOUT: !17 = !DILocation(line: 41, column: 5, scope: !4)
// CHECK:STDOUT: !18 = !DILocation(line: 41, column: 3, scope: !4)
// CHECK:STDOUT: !19 = !DILocation(line: 32, column: 1, scope: !4)
// CHECK:STDOUT: !20 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.5450dc8e8b8e0899", scope: null, file: !3, line: 18, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !21 = !DILocation(line: 19, column: 7, scope: !20)
// CHECK:STDOUT: !22 = !DILocation(line: 19, column: 6, scope: !20)
// CHECK:STDOUT: !23 = !DILocation(line: 20, column: 5, scope: !20)
// CHECK:STDOUT: !24 = !DILocation(line: 22, column: 15, scope: !20)
// CHECK:STDOUT: !25 = !DILocation(line: 22, column: 10, scope: !20)
// CHECK:STDOUT: !26 = !DILocation(line: 22, column: 3, scope: !20)
// CHECK:STDOUT: !27 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.83aece103b0a8681", scope: null, file: !3, line: 18, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !28 = !DILocation(line: 19, column: 7, scope: !27)
// CHECK:STDOUT: !29 = !DILocation(line: 19, column: 6, scope: !27)
// CHECK:STDOUT: !30 = !DILocation(line: 20, column: 5, scope: !27)
// CHECK:STDOUT: !31 = !DILocation(line: 22, column: 15, scope: !27)
// CHECK:STDOUT: !32 = !DILocation(line: 22, column: 10, scope: !27)
// CHECK:STDOUT: !33 = !DILocation(line: 22, column: 3, scope: !27)
// CHECK:STDOUT: !34 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.0a952f8bcc623ce6", scope: null, file: !3, line: 18, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !35 = !DILocation(line: 19, column: 7, scope: !34)
// CHECK:STDOUT: !36 = !DILocation(line: 19, column: 6, scope: !34)
// CHECK:STDOUT: !37 = !DILocation(line: 20, column: 5, scope: !34)
// CHECK:STDOUT: !38 = !DILocation(line: 22, column: 15, scope: !34)
// CHECK:STDOUT: !39 = !DILocation(line: 22, column: 10, scope: !34)
// CHECK:STDOUT: !40 = !DILocation(line: 22, column: 3, scope: !34)
// CHECK:STDOUT: !41 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.5450dc8e8b8e0899", scope: null, file: !3, line: 25, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !42 = !DILocation(line: 26, column: 7, scope: !41)
// CHECK:STDOUT: !43 = !DILocation(line: 26, column: 6, scope: !41)
// CHECK:STDOUT: !44 = !DILocation(line: 27, column: 5, scope: !41)
// CHECK:STDOUT: !45 = !DILocation(line: 29, column: 15, scope: !41)
// CHECK:STDOUT: !46 = !DILocation(line: 29, column: 10, scope: !41)
// CHECK:STDOUT: !47 = !DILocation(line: 29, column: 3, scope: !41)
// CHECK:STDOUT: !48 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.83aece103b0a8681", scope: null, file: !3, line: 25, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !49 = !DILocation(line: 26, column: 7, scope: !48)
// CHECK:STDOUT: !50 = !DILocation(line: 26, column: 6, scope: !48)
// CHECK:STDOUT: !51 = !DILocation(line: 27, column: 5, scope: !48)
// CHECK:STDOUT: !52 = !DILocation(line: 29, column: 15, scope: !48)
// CHECK:STDOUT: !53 = !DILocation(line: 29, column: 10, scope: !48)
// CHECK:STDOUT: !54 = !DILocation(line: 29, column: 3, scope: !48)
// CHECK:STDOUT: !55 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.0a952f8bcc623ce6", scope: null, file: !3, line: 25, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !56 = !DILocation(line: 26, column: 7, scope: !55)
// CHECK:STDOUT: !57 = !DILocation(line: 26, column: 6, scope: !55)
// CHECK:STDOUT: !58 = !DILocation(line: 27, column: 5, scope: !55)
// CHECK:STDOUT: !59 = !DILocation(line: 29, column: 15, scope: !55)
// CHECK:STDOUT: !60 = !DILocation(line: 29, column: 10, scope: !55)
// CHECK:STDOUT: !61 = !DILocation(line: 29, column: 3, scope: !55)
