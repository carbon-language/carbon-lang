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
  var ptr_i32 : i32* = &n;
  var ptr_f64 : f64* = &m;

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
// CHECK:STDOUT:   store ptr %n.var, ptr %ptr_i32.var, align 8, !dbg !9
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_f64.var), !dbg !10
// CHECK:STDOUT:   store ptr %m.var, ptr %ptr_f64.var, align 8, !dbg !10
// CHECK:STDOUT:   %.loc38_5 = load i32, ptr %n.var, align 4, !dbg !11
// CHECK:STDOUT:   %F.call.loc38 = call i32 @_CF.Main.64ccbb8e5d9a0b8e(i32 %.loc38_5, i32 0), !dbg !12
// CHECK:STDOUT:   %.loc39_5 = load double, ptr %m.var, align 8, !dbg !13
// CHECK:STDOUT:   %F.call.loc39 = call double @_CF.Main.de5b400758c39a65(double %.loc39_5, i32 0), !dbg !14
// CHECK:STDOUT:   %.loc40_5 = load ptr, ptr %ptr_i32.var, align 8, !dbg !15
// CHECK:STDOUT:   %F.call.loc40 = call ptr @_CF.Main.779b9c0f3b54a7f8(ptr %.loc40_5, i32 0), !dbg !16
// CHECK:STDOUT:   %.loc41_5 = load ptr, ptr %ptr_f64.var, align 8, !dbg !17
// CHECK:STDOUT:   %F.call.loc41 = call ptr @_CF.Main.779b9c0f3b54a7f8(ptr %.loc41_5, i32 0), !dbg !18
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
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.64ccbb8e5d9a0b8e(i32 %x, i32 %count) #0 !dbg !34 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 3, !dbg !40
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !41
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret i32 %x, !dbg !42
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !43
// CHECK:STDOUT:   %G.call = call i32 @_CG.Main.64ccbb8e5d9a0b8e(i32 %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !44
// CHECK:STDOUT:   ret i32 %G.call, !dbg !45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CF.Main.de5b400758c39a65(double %x, i32 %count) #0 !dbg !46 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 3, !dbg !52
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !53
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret double %x, !dbg !54
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !55
// CHECK:STDOUT:   %G.call = call double @_CG.Main.de5b400758c39a65(double %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !56
// CHECK:STDOUT:   ret double %G.call, !dbg !57
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CF.Main.779b9c0f3b54a7f8(ptr %x, i32 %count) #0 !dbg !58 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 3, !dbg !62
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !63
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret ptr %x, !dbg !64
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !65
// CHECK:STDOUT:   %G.call = call ptr @_CG.Main.779b9c0f3b54a7f8(ptr %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !66
// CHECK:STDOUT:   ret ptr %G.call, !dbg !67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CG.Main.64ccbb8e5d9a0b8e(i32 %x, i32 %count) #0 !dbg !68 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 4, !dbg !72
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !73
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret i32 %x, !dbg !74
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !75
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main.64ccbb8e5d9a0b8e(i32 %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !76
// CHECK:STDOUT:   ret i32 %F.call, !dbg !77
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CG.Main.de5b400758c39a65(double %x, i32 %count) #0 !dbg !78 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 4, !dbg !82
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !83
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret double %x, !dbg !84
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !85
// CHECK:STDOUT:   %F.call = call double @_CF.Main.de5b400758c39a65(double %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !86
// CHECK:STDOUT:   ret double %F.call, !dbg !87
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CG.Main.779b9c0f3b54a7f8(ptr %x, i32 %count) #0 !dbg !88 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 4, !dbg !92
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !93
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret ptr %x, !dbg !94
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !95
// CHECK:STDOUT:   %F.call = call ptr @_CF.Main.779b9c0f3b54a7f8(ptr %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !96
// CHECK:STDOUT:   ret ptr %F.call, !dbg !97
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CF.Main.779b9c0f3b54a7f8, { 1, 2, 0 }
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
// CHECK:STDOUT: !20 = distinct !DISubprogram(name: "Op", linkageName: "_COp.98006a366c2ffd6c:core.Destroy.Core", scope: null, file: !3, line: 34, type: !21, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !24)
// CHECK:STDOUT: !21 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !22 = !{null, !23}
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{!25}
// CHECK:STDOUT: !25 = !DILocalVariable(arg: 1, scope: !20, type: !23)
// CHECK:STDOUT: !26 = !DILocation(line: 34, column: 3, scope: !20)
// CHECK:STDOUT: !27 = distinct !DISubprogram(name: "Op", linkageName: "_COp.7e389eab4a7e5487:core.Destroy.Core", scope: null, file: !3, line: 33, type: !28, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !31)
// CHECK:STDOUT: !28 = !DISubroutineType(types: !29)
// CHECK:STDOUT: !29 = !{null, !30}
// CHECK:STDOUT: !30 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !31 = !{!32}
// CHECK:STDOUT: !32 = !DILocalVariable(arg: 1, scope: !27, type: !30)
// CHECK:STDOUT: !33 = !DILocation(line: 33, column: 3, scope: !27)
// CHECK:STDOUT: !34 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.64ccbb8e5d9a0b8e", scope: null, file: !3, line: 18, type: !35, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !37)
// CHECK:STDOUT: !35 = !DISubroutineType(types: !36)
// CHECK:STDOUT: !36 = !{!30, !30, !30}
// CHECK:STDOUT: !37 = !{!38, !39}
// CHECK:STDOUT: !38 = !DILocalVariable(arg: 1, scope: !34, type: !30)
// CHECK:STDOUT: !39 = !DILocalVariable(arg: 2, scope: !34, type: !30)
// CHECK:STDOUT: !40 = !DILocation(line: 19, column: 7, scope: !34)
// CHECK:STDOUT: !41 = !DILocation(line: 19, column: 6, scope: !34)
// CHECK:STDOUT: !42 = !DILocation(line: 20, column: 5, scope: !34)
// CHECK:STDOUT: !43 = !DILocation(line: 22, column: 15, scope: !34)
// CHECK:STDOUT: !44 = !DILocation(line: 22, column: 10, scope: !34)
// CHECK:STDOUT: !45 = !DILocation(line: 22, column: 3, scope: !34)
// CHECK:STDOUT: !46 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.de5b400758c39a65", scope: null, file: !3, line: 18, type: !47, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !49)
// CHECK:STDOUT: !47 = !DISubroutineType(types: !48)
// CHECK:STDOUT: !48 = !{!23, !23, !30}
// CHECK:STDOUT: !49 = !{!50, !51}
// CHECK:STDOUT: !50 = !DILocalVariable(arg: 1, scope: !46, type: !23)
// CHECK:STDOUT: !51 = !DILocalVariable(arg: 2, scope: !46, type: !30)
// CHECK:STDOUT: !52 = !DILocation(line: 19, column: 7, scope: !46)
// CHECK:STDOUT: !53 = !DILocation(line: 19, column: 6, scope: !46)
// CHECK:STDOUT: !54 = !DILocation(line: 20, column: 5, scope: !46)
// CHECK:STDOUT: !55 = !DILocation(line: 22, column: 15, scope: !46)
// CHECK:STDOUT: !56 = !DILocation(line: 22, column: 10, scope: !46)
// CHECK:STDOUT: !57 = !DILocation(line: 22, column: 3, scope: !46)
// CHECK:STDOUT: !58 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.779b9c0f3b54a7f8", scope: null, file: !3, line: 18, type: !47, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !59)
// CHECK:STDOUT: !59 = !{!60, !61}
// CHECK:STDOUT: !60 = !DILocalVariable(arg: 1, scope: !58, type: !23)
// CHECK:STDOUT: !61 = !DILocalVariable(arg: 2, scope: !58, type: !30)
// CHECK:STDOUT: !62 = !DILocation(line: 19, column: 7, scope: !58)
// CHECK:STDOUT: !63 = !DILocation(line: 19, column: 6, scope: !58)
// CHECK:STDOUT: !64 = !DILocation(line: 20, column: 5, scope: !58)
// CHECK:STDOUT: !65 = !DILocation(line: 22, column: 15, scope: !58)
// CHECK:STDOUT: !66 = !DILocation(line: 22, column: 10, scope: !58)
// CHECK:STDOUT: !67 = !DILocation(line: 22, column: 3, scope: !58)
// CHECK:STDOUT: !68 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.64ccbb8e5d9a0b8e", scope: null, file: !3, line: 25, type: !35, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !69)
// CHECK:STDOUT: !69 = !{!70, !71}
// CHECK:STDOUT: !70 = !DILocalVariable(arg: 1, scope: !68, type: !30)
// CHECK:STDOUT: !71 = !DILocalVariable(arg: 2, scope: !68, type: !30)
// CHECK:STDOUT: !72 = !DILocation(line: 26, column: 7, scope: !68)
// CHECK:STDOUT: !73 = !DILocation(line: 26, column: 6, scope: !68)
// CHECK:STDOUT: !74 = !DILocation(line: 27, column: 5, scope: !68)
// CHECK:STDOUT: !75 = !DILocation(line: 29, column: 15, scope: !68)
// CHECK:STDOUT: !76 = !DILocation(line: 29, column: 10, scope: !68)
// CHECK:STDOUT: !77 = !DILocation(line: 29, column: 3, scope: !68)
// CHECK:STDOUT: !78 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.de5b400758c39a65", scope: null, file: !3, line: 25, type: !47, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !79)
// CHECK:STDOUT: !79 = !{!80, !81}
// CHECK:STDOUT: !80 = !DILocalVariable(arg: 1, scope: !78, type: !23)
// CHECK:STDOUT: !81 = !DILocalVariable(arg: 2, scope: !78, type: !30)
// CHECK:STDOUT: !82 = !DILocation(line: 26, column: 7, scope: !78)
// CHECK:STDOUT: !83 = !DILocation(line: 26, column: 6, scope: !78)
// CHECK:STDOUT: !84 = !DILocation(line: 27, column: 5, scope: !78)
// CHECK:STDOUT: !85 = !DILocation(line: 29, column: 15, scope: !78)
// CHECK:STDOUT: !86 = !DILocation(line: 29, column: 10, scope: !78)
// CHECK:STDOUT: !87 = !DILocation(line: 29, column: 3, scope: !78)
// CHECK:STDOUT: !88 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.779b9c0f3b54a7f8", scope: null, file: !3, line: 25, type: !47, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !89)
// CHECK:STDOUT: !89 = !{!90, !91}
// CHECK:STDOUT: !90 = !DILocalVariable(arg: 1, scope: !88, type: !23)
// CHECK:STDOUT: !91 = !DILocalVariable(arg: 2, scope: !88, type: !30)
// CHECK:STDOUT: !92 = !DILocation(line: 26, column: 7, scope: !88)
// CHECK:STDOUT: !93 = !DILocation(line: 26, column: 6, scope: !88)
// CHECK:STDOUT: !94 = !DILocation(line: 27, column: 5, scope: !88)
// CHECK:STDOUT: !95 = !DILocation(line: 29, column: 15, scope: !88)
// CHECK:STDOUT: !96 = !DILocation(line: 29, column: 10, scope: !88)
// CHECK:STDOUT: !97 = !DILocation(line: 29, column: 3, scope: !88)
