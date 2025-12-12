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
// CHECK:STDOUT:   %F.call.loc38 = call i32 @_CF.Main.3440082c84c3c17b(i32 %.loc38_5, i32 0), !dbg !12
// CHECK:STDOUT:   %.loc39_5 = load double, ptr %m.var, align 8, !dbg !13
// CHECK:STDOUT:   %F.call.loc39 = call double @_CF.Main.286b0f7890e5304c(double %.loc39_5, i32 0), !dbg !14
// CHECK:STDOUT:   %.loc40_5 = load ptr, ptr %ptr_i32.var, align 8, !dbg !15
// CHECK:STDOUT:   %F.call.loc40 = call ptr @_CF.Main.4b30cda44e16b9ec(ptr %.loc40_5, i32 0), !dbg !16
// CHECK:STDOUT:   %.loc41_5 = load ptr, ptr %ptr_f64.var, align 8, !dbg !17
// CHECK:STDOUT:   %F.call.loc41 = call ptr @_CF.Main.4b30cda44e16b9ec(ptr %.loc41_5, i32 0), !dbg !18
// CHECK:STDOUT:   ret void, !dbg !19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.3440082c84c3c17b(i32 %x, i32 %count) #0 !dbg !20 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 3, !dbg !27
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !28
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret i32 %x, !dbg !29
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !30
// CHECK:STDOUT:   %G.call = call i32 @_CG.Main.3440082c84c3c17b(i32 %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !31
// CHECK:STDOUT:   ret i32 %G.call, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CF.Main.286b0f7890e5304c(double %x, i32 %count) #0 !dbg !33 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 3, !dbg !40
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !41
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret double %x, !dbg !42
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !43
// CHECK:STDOUT:   %G.call = call double @_CG.Main.286b0f7890e5304c(double %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !44
// CHECK:STDOUT:   ret double %G.call, !dbg !45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CF.Main.4b30cda44e16b9ec(ptr %x, i32 %count) #0 !dbg !46 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 3, !dbg !50
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !51
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret ptr %x, !dbg !52
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !53
// CHECK:STDOUT:   %G.call = call ptr @_CG.Main.4b30cda44e16b9ec(ptr %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !54
// CHECK:STDOUT:   ret ptr %G.call, !dbg !55
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CG.Main.3440082c84c3c17b(i32 %x, i32 %count) #0 !dbg !56 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 4, !dbg !60
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !61
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret i32 %x, !dbg !62
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !63
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main.3440082c84c3c17b(i32 %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !64
// CHECK:STDOUT:   ret i32 %F.call, !dbg !65
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CG.Main.286b0f7890e5304c(double %x, i32 %count) #0 !dbg !66 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 4, !dbg !70
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !71
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret double %x, !dbg !72
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !73
// CHECK:STDOUT:   %F.call = call double @_CF.Main.286b0f7890e5304c(double %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !74
// CHECK:STDOUT:   ret double %F.call, !dbg !75
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CG.Main.4b30cda44e16b9ec(ptr %x, i32 %count) #0 !dbg !76 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 4, !dbg !80
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !81
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret ptr %x, !dbg !82
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !83
// CHECK:STDOUT:   %F.call = call ptr @_CF.Main.4b30cda44e16b9ec(ptr %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !84
// CHECK:STDOUT:   ret ptr %F.call, !dbg !85
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CF.Main.4b30cda44e16b9ec, { 1, 2, 0 }
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
// CHECK:STDOUT: !3 = !DIFile(filename: "call_recursive_mutual.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "M", linkageName: "_CM.Main", scope: null, file: !3, line: 32, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null}
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
// CHECK:STDOUT: !20 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.3440082c84c3c17b", scope: null, file: !3, line: 18, type: !21, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !24)
// CHECK:STDOUT: !21 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !22 = !{!23, !23, !23}
// CHECK:STDOUT: !23 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !24 = !{!25, !26}
// CHECK:STDOUT: !25 = !DILocalVariable(arg: 1, scope: !20, type: !23)
// CHECK:STDOUT: !26 = !DILocalVariable(arg: 2, scope: !20, type: !23)
// CHECK:STDOUT: !27 = !DILocation(line: 19, column: 7, scope: !20)
// CHECK:STDOUT: !28 = !DILocation(line: 19, column: 6, scope: !20)
// CHECK:STDOUT: !29 = !DILocation(line: 20, column: 5, scope: !20)
// CHECK:STDOUT: !30 = !DILocation(line: 22, column: 15, scope: !20)
// CHECK:STDOUT: !31 = !DILocation(line: 22, column: 10, scope: !20)
// CHECK:STDOUT: !32 = !DILocation(line: 22, column: 3, scope: !20)
// CHECK:STDOUT: !33 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.286b0f7890e5304c", scope: null, file: !3, line: 18, type: !34, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !37)
// CHECK:STDOUT: !34 = !DISubroutineType(types: !35)
// CHECK:STDOUT: !35 = !{!36, !36, !23}
// CHECK:STDOUT: !36 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 8)
// CHECK:STDOUT: !37 = !{!38, !39}
// CHECK:STDOUT: !38 = !DILocalVariable(arg: 1, scope: !33, type: !36)
// CHECK:STDOUT: !39 = !DILocalVariable(arg: 2, scope: !33, type: !23)
// CHECK:STDOUT: !40 = !DILocation(line: 19, column: 7, scope: !33)
// CHECK:STDOUT: !41 = !DILocation(line: 19, column: 6, scope: !33)
// CHECK:STDOUT: !42 = !DILocation(line: 20, column: 5, scope: !33)
// CHECK:STDOUT: !43 = !DILocation(line: 22, column: 15, scope: !33)
// CHECK:STDOUT: !44 = !DILocation(line: 22, column: 10, scope: !33)
// CHECK:STDOUT: !45 = !DILocation(line: 22, column: 3, scope: !33)
// CHECK:STDOUT: !46 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.4b30cda44e16b9ec", scope: null, file: !3, line: 18, type: !34, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !47)
// CHECK:STDOUT: !47 = !{!48, !49}
// CHECK:STDOUT: !48 = !DILocalVariable(arg: 1, scope: !46, type: !36)
// CHECK:STDOUT: !49 = !DILocalVariable(arg: 2, scope: !46, type: !23)
// CHECK:STDOUT: !50 = !DILocation(line: 19, column: 7, scope: !46)
// CHECK:STDOUT: !51 = !DILocation(line: 19, column: 6, scope: !46)
// CHECK:STDOUT: !52 = !DILocation(line: 20, column: 5, scope: !46)
// CHECK:STDOUT: !53 = !DILocation(line: 22, column: 15, scope: !46)
// CHECK:STDOUT: !54 = !DILocation(line: 22, column: 10, scope: !46)
// CHECK:STDOUT: !55 = !DILocation(line: 22, column: 3, scope: !46)
// CHECK:STDOUT: !56 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.3440082c84c3c17b", scope: null, file: !3, line: 25, type: !21, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !57)
// CHECK:STDOUT: !57 = !{!58, !59}
// CHECK:STDOUT: !58 = !DILocalVariable(arg: 1, scope: !56, type: !23)
// CHECK:STDOUT: !59 = !DILocalVariable(arg: 2, scope: !56, type: !23)
// CHECK:STDOUT: !60 = !DILocation(line: 26, column: 7, scope: !56)
// CHECK:STDOUT: !61 = !DILocation(line: 26, column: 6, scope: !56)
// CHECK:STDOUT: !62 = !DILocation(line: 27, column: 5, scope: !56)
// CHECK:STDOUT: !63 = !DILocation(line: 29, column: 15, scope: !56)
// CHECK:STDOUT: !64 = !DILocation(line: 29, column: 10, scope: !56)
// CHECK:STDOUT: !65 = !DILocation(line: 29, column: 3, scope: !56)
// CHECK:STDOUT: !66 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.286b0f7890e5304c", scope: null, file: !3, line: 25, type: !34, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !67)
// CHECK:STDOUT: !67 = !{!68, !69}
// CHECK:STDOUT: !68 = !DILocalVariable(arg: 1, scope: !66, type: !36)
// CHECK:STDOUT: !69 = !DILocalVariable(arg: 2, scope: !66, type: !23)
// CHECK:STDOUT: !70 = !DILocation(line: 26, column: 7, scope: !66)
// CHECK:STDOUT: !71 = !DILocation(line: 26, column: 6, scope: !66)
// CHECK:STDOUT: !72 = !DILocation(line: 27, column: 5, scope: !66)
// CHECK:STDOUT: !73 = !DILocation(line: 29, column: 15, scope: !66)
// CHECK:STDOUT: !74 = !DILocation(line: 29, column: 10, scope: !66)
// CHECK:STDOUT: !75 = !DILocation(line: 29, column: 3, scope: !66)
// CHECK:STDOUT: !76 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.4b30cda44e16b9ec", scope: null, file: !3, line: 25, type: !34, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !77)
// CHECK:STDOUT: !77 = !{!78, !79}
// CHECK:STDOUT: !78 = !DILocalVariable(arg: 1, scope: !76, type: !36)
// CHECK:STDOUT: !79 = !DILocalVariable(arg: 2, scope: !76, type: !23)
// CHECK:STDOUT: !80 = !DILocation(line: 26, column: 7, scope: !76)
// CHECK:STDOUT: !81 = !DILocation(line: 26, column: 6, scope: !76)
// CHECK:STDOUT: !82 = !DILocation(line: 27, column: 5, scope: !76)
// CHECK:STDOUT: !83 = !DILocation(line: 29, column: 15, scope: !76)
// CHECK:STDOUT: !84 = !DILocation(line: 29, column: 10, scope: !76)
// CHECK:STDOUT: !85 = !DILocation(line: 29, column: 3, scope: !76)
