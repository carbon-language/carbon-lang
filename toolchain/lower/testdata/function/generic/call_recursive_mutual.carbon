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
fn G[T: Core.Copy](x: T, count: i32) -> T;

fn F[T: Core.Copy](x: T, count: i32) -> T {
  if (count > 3) {
    return x;
  }
  return G(x, count + 1);
}

fn G[T: Core.Copy](x: T, count: i32) -> T {
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

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'call_recursive_mutual.carbon'
// CHECK:STDOUT: source_filename = "call_recursive_mutual.carbon"
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
// CHECK:STDOUT:   %.loc38_5 = load i32, ptr %n.var, align 4, !dbg !22
// CHECK:STDOUT:   %F.call.loc38 = call i32 @_CF.Main.2fad9574e050607e(i32 %.loc38_5, i32 0), !dbg !23
// CHECK:STDOUT:   %.loc39_5 = load double, ptr %m.var, align 8, !dbg !26
// CHECK:STDOUT:   %F.call.loc39 = call double @_CF.Main.21aefb7dfdebb8e4(double %.loc39_5, i32 0), !dbg !27
// CHECK:STDOUT:   %.loc40_5 = load ptr, ptr %ptr_i32.var, align 8, !dbg !31
// CHECK:STDOUT:   %F.call.loc40 = call ptr @_CF.Main.417c9c4a651ab31b(ptr %.loc40_5, i32 0), !dbg !32
// CHECK:STDOUT:   %.loc41_5 = load ptr, ptr %ptr_f64.var, align 8, !dbg !33
// CHECK:STDOUT:   %F.call.loc41 = call ptr @_CF.Main.417c9c4a651ab31b(ptr %.loc41_5, i32 0), !dbg !34
// CHECK:STDOUT:   ret void, !dbg !35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.2fad9574e050607e(i32 %x, i32 %count) #0 !dbg !36 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 3, !dbg !43
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !44
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret i32 %x, !dbg !46
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !48
// CHECK:STDOUT:   %G.call = call i32 @_CG.Main.2fad9574e050607e(i32 %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !49
// CHECK:STDOUT:   ret i32 %G.call, !dbg !50
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CF.Main.21aefb7dfdebb8e4(double %x, i32 %count) #0 !dbg !52 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 3, !dbg !59
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !60
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret double %x, !dbg !62
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !64
// CHECK:STDOUT:   %G.call = call double @_CG.Main.21aefb7dfdebb8e4(double %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !65
// CHECK:STDOUT:   ret double %G.call, !dbg !66
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CF.Main.417c9c4a651ab31b(ptr %x, i32 %count) #0 !dbg !68 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 3, !dbg !75
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !76
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret ptr %x, !dbg !78
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !80
// CHECK:STDOUT:   %G.call = call ptr @_CG.Main.417c9c4a651ab31b(ptr %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !81
// CHECK:STDOUT:   ret ptr %G.call, !dbg !82
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CG.Main.2fad9574e050607e(i32 %x, i32 %count) #0 !dbg !100 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 4, !dbg !107
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !108
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret i32 %x, !dbg !110
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !112
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main.2fad9574e050607e(i32 %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !113
// CHECK:STDOUT:   ret i32 %F.call, !dbg !114
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CG.Main.21aefb7dfdebb8e4(double %x, i32 %count) #0 !dbg !116 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 4, !dbg !123
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !124
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret double %x, !dbg !126
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !128
// CHECK:STDOUT:   %F.call = call double @_CF.Main.21aefb7dfdebb8e4(double %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !129
// CHECK:STDOUT:   ret double %F.call, !dbg !130
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CG.Main.417c9c4a651ab31b(ptr %x, i32 %count) #0 !dbg !132 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 4, !dbg !139
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !140
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret ptr %x, !dbg !142
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !144
// CHECK:STDOUT:   %F.call = call ptr @_CF.Main.417c9c4a651ab31b(ptr %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !145
// CHECK:STDOUT:   ret ptr %F.call, !dbg !146
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CF.Main.417c9c4a651ab31b, { 1, 2, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "call_recursive_mutual.carbon", directory: "")
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !7 = !{null}
// CHECK:STDOUT: !8 = !DISubroutineType(types: !7)
// CHECK:STDOUT: !9 = distinct !DISubprogram(name: "M", linkageName: "_CM.Main", scope: null, file: !4, line: 32, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !10 = !DILocation(line: 33, column: 3, scope: !9)
// CHECK:STDOUT: !12 = !DILocation(line: 34, column: 3, scope: !9)
// CHECK:STDOUT: !14 = !DILocation(line: 35, column: 3, scope: !9)
// CHECK:STDOUT: !18 = !DILocation(line: 36, column: 3, scope: !9)
// CHECK:STDOUT: !22 = !DILocation(line: 38, column: 5, scope: !9)
// CHECK:STDOUT: !23 = !DILocation(line: 38, column: 3, scope: !9)
// CHECK:STDOUT: !24 = !{!6, !6, !6}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = !DILocation(line: 39, column: 5, scope: !9)
// CHECK:STDOUT: !27 = !DILocation(line: 39, column: 3, scope: !9)
// CHECK:STDOUT: !28 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !29 = !{!28, !28, !6}
// CHECK:STDOUT: !30 = !DISubroutineType(types: !29)
// CHECK:STDOUT: !31 = !DILocation(line: 40, column: 5, scope: !9)
// CHECK:STDOUT: !32 = !DILocation(line: 40, column: 3, scope: !9)
// CHECK:STDOUT: !33 = !DILocation(line: 41, column: 5, scope: !9)
// CHECK:STDOUT: !34 = !DILocation(line: 41, column: 3, scope: !9)
// CHECK:STDOUT: !35 = !DILocation(line: 32, column: 1, scope: !9)
// CHECK:STDOUT: !36 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.2fad9574e050607e", scope: null, file: !4, line: 18, type: !25, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !51)
// CHECK:STDOUT: !37 = !DILocalVariable(arg: 1, scope: !36, type: !6)
// CHECK:STDOUT: !38 = !DILocalVariable(arg: 2, scope: !36, type: !6)
// CHECK:STDOUT: !43 = !DILocation(line: 19, column: 7, scope: !36)
// CHECK:STDOUT: !44 = !DILocation(line: 19, column: 6, scope: !36)
// CHECK:STDOUT: !46 = !DILocation(line: 20, column: 5, scope: !36)
// CHECK:STDOUT: !48 = !DILocation(line: 22, column: 15, scope: !36)
// CHECK:STDOUT: !49 = !DILocation(line: 22, column: 10, scope: !36)
// CHECK:STDOUT: !50 = !DILocation(line: 22, column: 3, scope: !36)
// CHECK:STDOUT: !51 = !{!37, !38}
// CHECK:STDOUT: !52 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.21aefb7dfdebb8e4", scope: null, file: !4, line: 18, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !67)
// CHECK:STDOUT: !53 = !DILocalVariable(arg: 1, scope: !52, type: !28)
// CHECK:STDOUT: !54 = !DILocalVariable(arg: 2, scope: !52, type: !6)
// CHECK:STDOUT: !59 = !DILocation(line: 19, column: 7, scope: !52)
// CHECK:STDOUT: !60 = !DILocation(line: 19, column: 6, scope: !52)
// CHECK:STDOUT: !62 = !DILocation(line: 20, column: 5, scope: !52)
// CHECK:STDOUT: !64 = !DILocation(line: 22, column: 15, scope: !52)
// CHECK:STDOUT: !65 = !DILocation(line: 22, column: 10, scope: !52)
// CHECK:STDOUT: !66 = !DILocation(line: 22, column: 3, scope: !52)
// CHECK:STDOUT: !67 = !{!53, !54}
// CHECK:STDOUT: !68 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.417c9c4a651ab31b", scope: null, file: !4, line: 18, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !83)
// CHECK:STDOUT: !69 = !DILocalVariable(arg: 1, scope: !68, type: !28)
// CHECK:STDOUT: !70 = !DILocalVariable(arg: 2, scope: !68, type: !6)
// CHECK:STDOUT: !75 = !DILocation(line: 19, column: 7, scope: !68)
// CHECK:STDOUT: !76 = !DILocation(line: 19, column: 6, scope: !68)
// CHECK:STDOUT: !78 = !DILocation(line: 20, column: 5, scope: !68)
// CHECK:STDOUT: !80 = !DILocation(line: 22, column: 15, scope: !68)
// CHECK:STDOUT: !81 = !DILocation(line: 22, column: 10, scope: !68)
// CHECK:STDOUT: !82 = !DILocation(line: 22, column: 3, scope: !68)
// CHECK:STDOUT: !83 = !{!69, !70}
// CHECK:STDOUT: !100 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.2fad9574e050607e", scope: null, file: !4, line: 25, type: !25, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !115)
// CHECK:STDOUT: !101 = !DILocalVariable(arg: 1, scope: !100, type: !6)
// CHECK:STDOUT: !102 = !DILocalVariable(arg: 2, scope: !100, type: !6)
// CHECK:STDOUT: !107 = !DILocation(line: 26, column: 7, scope: !100)
// CHECK:STDOUT: !108 = !DILocation(line: 26, column: 6, scope: !100)
// CHECK:STDOUT: !110 = !DILocation(line: 27, column: 5, scope: !100)
// CHECK:STDOUT: !112 = !DILocation(line: 29, column: 15, scope: !100)
// CHECK:STDOUT: !113 = !DILocation(line: 29, column: 10, scope: !100)
// CHECK:STDOUT: !114 = !DILocation(line: 29, column: 3, scope: !100)
// CHECK:STDOUT: !115 = !{!101, !102}
// CHECK:STDOUT: !116 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.21aefb7dfdebb8e4", scope: null, file: !4, line: 25, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !131)
// CHECK:STDOUT: !117 = !DILocalVariable(arg: 1, scope: !116, type: !28)
// CHECK:STDOUT: !118 = !DILocalVariable(arg: 2, scope: !116, type: !6)
// CHECK:STDOUT: !123 = !DILocation(line: 26, column: 7, scope: !116)
// CHECK:STDOUT: !124 = !DILocation(line: 26, column: 6, scope: !116)
// CHECK:STDOUT: !126 = !DILocation(line: 27, column: 5, scope: !116)
// CHECK:STDOUT: !128 = !DILocation(line: 29, column: 15, scope: !116)
// CHECK:STDOUT: !129 = !DILocation(line: 29, column: 10, scope: !116)
// CHECK:STDOUT: !130 = !DILocation(line: 29, column: 3, scope: !116)
// CHECK:STDOUT: !131 = !{!117, !118}
// CHECK:STDOUT: !132 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.417c9c4a651ab31b", scope: null, file: !4, line: 25, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !147)
// CHECK:STDOUT: !133 = !DILocalVariable(arg: 1, scope: !132, type: !28)
// CHECK:STDOUT: !134 = !DILocalVariable(arg: 2, scope: !132, type: !6)
// CHECK:STDOUT: !139 = !DILocation(line: 26, column: 7, scope: !132)
// CHECK:STDOUT: !140 = !DILocation(line: 26, column: 6, scope: !132)
// CHECK:STDOUT: !142 = !DILocation(line: 27, column: 5, scope: !132)
// CHECK:STDOUT: !144 = !DILocation(line: 29, column: 15, scope: !132)
// CHECK:STDOUT: !145 = !DILocation(line: 29, column: 10, scope: !132)
// CHECK:STDOUT: !146 = !DILocation(line: 29, column: 3, scope: !132)
// CHECK:STDOUT: !147 = !{!133, !134}
// CHECK:STDOUT:
