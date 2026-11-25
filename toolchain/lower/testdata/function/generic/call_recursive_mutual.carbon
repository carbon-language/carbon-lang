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
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.2fad9574e050607e(i32 %x, i32 %count) #0 !dbg !40 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 3, !dbg !47
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !48
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret i32 %x, !dbg !50
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !52
// CHECK:STDOUT:   %G.call = call i32 @_CG.Main.2fad9574e050607e(i32 %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !53
// CHECK:STDOUT:   ret i32 %G.call, !dbg !54
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CF.Main.21aefb7dfdebb8e4(double %x, i32 %count) #0 !dbg !56 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 3, !dbg !63
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !64
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret double %x, !dbg !66
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !68
// CHECK:STDOUT:   %G.call = call double @_CG.Main.21aefb7dfdebb8e4(double %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !69
// CHECK:STDOUT:   ret double %G.call, !dbg !70
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CF.Main.417c9c4a651ab31b(ptr %x, i32 %count) #0 !dbg !72 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 3, !dbg !79
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !80
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret ptr %x, !dbg !82
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !84
// CHECK:STDOUT:   %G.call = call ptr @_CG.Main.417c9c4a651ab31b(ptr %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !85
// CHECK:STDOUT:   ret ptr %G.call, !dbg !86
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.b18855c76072615b(ptr %t) #0 !dbg !105 {
// CHECK:STDOUT:   ret void, !dbg !110
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.4f29d15a20894e9b(ptr %t) #0 !dbg !112 {
// CHECK:STDOUT:   ret void, !dbg !117
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.2b38c2ca9f7f1a94(ptr %t) #0 !dbg !119 {
// CHECK:STDOUT:   ret void, !dbg !124
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.f127f30143696d51(ptr %t) #0 !dbg !126 {
// CHECK:STDOUT:   ret void, !dbg !131
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CG.Main.2fad9574e050607e(i32 %x, i32 %count) #0 !dbg !133 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 4, !dbg !140
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !141
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret i32 %x, !dbg !143
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !145
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main.2fad9574e050607e(i32 %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !146
// CHECK:STDOUT:   ret i32 %F.call, !dbg !147
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CG.Main.21aefb7dfdebb8e4(double %x, i32 %count) #0 !dbg !149 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 4, !dbg !156
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !157
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret double %x, !dbg !159
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !161
// CHECK:STDOUT:   %F.call = call double @_CF.Main.21aefb7dfdebb8e4(double %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !162
// CHECK:STDOUT:   ret double %F.call, !dbg !163
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CG.Main.417c9c4a651ab31b(ptr %x, i32 %count) #0 !dbg !165 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 4, !dbg !172
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !173
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret ptr %x, !dbg !175
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !177
// CHECK:STDOUT:   %F.call = call ptr @_CF.Main.417c9c4a651ab31b(ptr %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !178
// CHECK:STDOUT:   ret ptr %F.call, !dbg !179
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
// CHECK:STDOUT: !35 = !{null, !28}
// CHECK:STDOUT: !36 = !DISubroutineType(types: !35)
// CHECK:STDOUT: !37 = !{null, !6}
// CHECK:STDOUT: !38 = !DISubroutineType(types: !37)
// CHECK:STDOUT: !39 = !DILocation(line: 32, column: 1, scope: !9)
// CHECK:STDOUT: !40 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.2fad9574e050607e", scope: null, file: !4, line: 18, type: !25, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !55)
// CHECK:STDOUT: !41 = !DILocalVariable(arg: 1, scope: !40, type: !6)
// CHECK:STDOUT: !42 = !DILocalVariable(arg: 2, scope: !40, type: !6)
// CHECK:STDOUT: !47 = !DILocation(line: 19, column: 7, scope: !40)
// CHECK:STDOUT: !48 = !DILocation(line: 19, column: 6, scope: !40)
// CHECK:STDOUT: !50 = !DILocation(line: 20, column: 5, scope: !40)
// CHECK:STDOUT: !52 = !DILocation(line: 22, column: 15, scope: !40)
// CHECK:STDOUT: !53 = !DILocation(line: 22, column: 10, scope: !40)
// CHECK:STDOUT: !54 = !DILocation(line: 22, column: 3, scope: !40)
// CHECK:STDOUT: !55 = !{!41, !42}
// CHECK:STDOUT: !56 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.21aefb7dfdebb8e4", scope: null, file: !4, line: 18, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !71)
// CHECK:STDOUT: !57 = !DILocalVariable(arg: 1, scope: !56, type: !28)
// CHECK:STDOUT: !58 = !DILocalVariable(arg: 2, scope: !56, type: !6)
// CHECK:STDOUT: !63 = !DILocation(line: 19, column: 7, scope: !56)
// CHECK:STDOUT: !64 = !DILocation(line: 19, column: 6, scope: !56)
// CHECK:STDOUT: !66 = !DILocation(line: 20, column: 5, scope: !56)
// CHECK:STDOUT: !68 = !DILocation(line: 22, column: 15, scope: !56)
// CHECK:STDOUT: !69 = !DILocation(line: 22, column: 10, scope: !56)
// CHECK:STDOUT: !70 = !DILocation(line: 22, column: 3, scope: !56)
// CHECK:STDOUT: !71 = !{!57, !58}
// CHECK:STDOUT: !72 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.417c9c4a651ab31b", scope: null, file: !4, line: 18, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !87)
// CHECK:STDOUT: !73 = !DILocalVariable(arg: 1, scope: !72, type: !28)
// CHECK:STDOUT: !74 = !DILocalVariable(arg: 2, scope: !72, type: !6)
// CHECK:STDOUT: !79 = !DILocation(line: 19, column: 7, scope: !72)
// CHECK:STDOUT: !80 = !DILocation(line: 19, column: 6, scope: !72)
// CHECK:STDOUT: !82 = !DILocation(line: 20, column: 5, scope: !72)
// CHECK:STDOUT: !84 = !DILocation(line: 22, column: 15, scope: !72)
// CHECK:STDOUT: !85 = !DILocation(line: 22, column: 10, scope: !72)
// CHECK:STDOUT: !86 = !DILocation(line: 22, column: 3, scope: !72)
// CHECK:STDOUT: !87 = !{!73, !74}
// CHECK:STDOUT: !104 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !105 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.b18855c76072615b", scope: null, file: !104, line: 35, type: !36, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !111)
// CHECK:STDOUT: !106 = !DILocalVariable(arg: 1, scope: !105, type: !28)
// CHECK:STDOUT: !110 = !DILocation(line: 35, column: 1, scope: !105)
// CHECK:STDOUT: !111 = !{!106}
// CHECK:STDOUT: !112 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.4f29d15a20894e9b", scope: null, file: !104, line: 35, type: !36, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !118)
// CHECK:STDOUT: !113 = !DILocalVariable(arg: 1, scope: !112, type: !28)
// CHECK:STDOUT: !117 = !DILocation(line: 35, column: 1, scope: !112)
// CHECK:STDOUT: !118 = !{!113}
// CHECK:STDOUT: !119 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.2b38c2ca9f7f1a94", scope: null, file: !104, line: 35, type: !36, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !125)
// CHECK:STDOUT: !120 = !DILocalVariable(arg: 1, scope: !119, type: !28)
// CHECK:STDOUT: !124 = !DILocation(line: 35, column: 1, scope: !119)
// CHECK:STDOUT: !125 = !{!120}
// CHECK:STDOUT: !126 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.f127f30143696d51", scope: null, file: !104, line: 35, type: !38, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !132)
// CHECK:STDOUT: !127 = !DILocalVariable(arg: 1, scope: !126, type: !6)
// CHECK:STDOUT: !131 = !DILocation(line: 35, column: 1, scope: !126)
// CHECK:STDOUT: !132 = !{!127}
// CHECK:STDOUT: !133 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.2fad9574e050607e", scope: null, file: !4, line: 25, type: !25, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !148)
// CHECK:STDOUT: !134 = !DILocalVariable(arg: 1, scope: !133, type: !6)
// CHECK:STDOUT: !135 = !DILocalVariable(arg: 2, scope: !133, type: !6)
// CHECK:STDOUT: !140 = !DILocation(line: 26, column: 7, scope: !133)
// CHECK:STDOUT: !141 = !DILocation(line: 26, column: 6, scope: !133)
// CHECK:STDOUT: !143 = !DILocation(line: 27, column: 5, scope: !133)
// CHECK:STDOUT: !145 = !DILocation(line: 29, column: 15, scope: !133)
// CHECK:STDOUT: !146 = !DILocation(line: 29, column: 10, scope: !133)
// CHECK:STDOUT: !147 = !DILocation(line: 29, column: 3, scope: !133)
// CHECK:STDOUT: !148 = !{!134, !135}
// CHECK:STDOUT: !149 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.21aefb7dfdebb8e4", scope: null, file: !4, line: 25, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !164)
// CHECK:STDOUT: !150 = !DILocalVariable(arg: 1, scope: !149, type: !28)
// CHECK:STDOUT: !151 = !DILocalVariable(arg: 2, scope: !149, type: !6)
// CHECK:STDOUT: !156 = !DILocation(line: 26, column: 7, scope: !149)
// CHECK:STDOUT: !157 = !DILocation(line: 26, column: 6, scope: !149)
// CHECK:STDOUT: !159 = !DILocation(line: 27, column: 5, scope: !149)
// CHECK:STDOUT: !161 = !DILocation(line: 29, column: 15, scope: !149)
// CHECK:STDOUT: !162 = !DILocation(line: 29, column: 10, scope: !149)
// CHECK:STDOUT: !163 = !DILocation(line: 29, column: 3, scope: !149)
// CHECK:STDOUT: !164 = !{!150, !151}
// CHECK:STDOUT: !165 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.417c9c4a651ab31b", scope: null, file: !4, line: 25, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !180)
// CHECK:STDOUT: !166 = !DILocalVariable(arg: 1, scope: !165, type: !28)
// CHECK:STDOUT: !167 = !DILocalVariable(arg: 2, scope: !165, type: !6)
// CHECK:STDOUT: !172 = !DILocation(line: 26, column: 7, scope: !165)
// CHECK:STDOUT: !173 = !DILocation(line: 26, column: 6, scope: !165)
// CHECK:STDOUT: !175 = !DILocation(line: 27, column: 5, scope: !165)
// CHECK:STDOUT: !177 = !DILocation(line: 29, column: 15, scope: !165)
// CHECK:STDOUT: !178 = !DILocation(line: 29, column: 10, scope: !165)
// CHECK:STDOUT: !179 = !DILocation(line: 29, column: 3, scope: !165)
// CHECK:STDOUT: !180 = !{!166, !167}
// CHECK:STDOUT:
