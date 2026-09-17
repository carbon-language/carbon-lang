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
// CHECK:STDOUT:   %.loc38_5 = load i32, ptr %n.var, align 4, !dbg !27
// CHECK:STDOUT:   %F.call.loc38 = call i32 @_CF.Main.4d6ceefeb7f772f4(i32 %.loc38_5, i32 0), !dbg !28
// CHECK:STDOUT:   %.loc39_5 = load double, ptr %m.var, align 8, !dbg !31
// CHECK:STDOUT:   %F.call.loc39 = call double @_CF.Main.c5e4ba542a56b71f(double %.loc39_5, i32 0), !dbg !32
// CHECK:STDOUT:   %.loc40_5 = load ptr, ptr %ptr_i32.var, align 8, !dbg !35
// CHECK:STDOUT:   %F.call.loc40 = call ptr @_CF.Main.a76b78461416b4a8(ptr %.loc40_5, i32 0), !dbg !36
// CHECK:STDOUT:   %.loc41_5 = load ptr, ptr %ptr_f64.var, align 8, !dbg !37
// CHECK:STDOUT:   %F.call.loc41 = call ptr @_CF.Main.a76b78461416b4a8(ptr %.loc41_5, i32 0), !dbg !38
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
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.4d6ceefeb7f772f4(i32 %x, i32 %count) #0 !dbg !48 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 3, !dbg !55
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !56
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret i32 %x, !dbg !58
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !60
// CHECK:STDOUT:   %G.call = call i32 @_CG.Main.4d6ceefeb7f772f4(i32 %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !61
// CHECK:STDOUT:   ret i32 %G.call, !dbg !62
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CF.Main.c5e4ba542a56b71f(double %x, i32 %count) #0 !dbg !64 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 3, !dbg !71
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !72
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret double %x, !dbg !74
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !76
// CHECK:STDOUT:   %G.call = call double @_CG.Main.c5e4ba542a56b71f(double %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !77
// CHECK:STDOUT:   ret double %G.call, !dbg !78
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CF.Main.a76b78461416b4a8(ptr %x, i32 %count) #0 !dbg !80 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 3, !dbg !87
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !88
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret ptr %x, !dbg !90
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !92
// CHECK:STDOUT:   %G.call = call ptr @_CG.Main.a76b78461416b4a8(ptr %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !93
// CHECK:STDOUT:   ret ptr %G.call, !dbg !94
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.9f1044688d586372(ptr %self) #0 !dbg !113 {
// CHECK:STDOUT:   ret void, !dbg !118
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.f427c19f24d378eb(ptr %self) #0 !dbg !120 {
// CHECK:STDOUT:   ret void, !dbg !125
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.23895ff6626cffe1(ptr %self) #0 !dbg !127 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.dc14d4dd21801791:core.Destroy.Core"(ptr %self), !dbg !131
// CHECK:STDOUT:   ret void, !dbg !132
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48(ptr %self) #0 !dbg !134 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self), !dbg !138
// CHECK:STDOUT:   ret void, !dbg !139
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CG.Main.4d6ceefeb7f772f4(i32 %x, i32 %count) #0 !dbg !141 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 4, !dbg !148
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !149
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret i32 %x, !dbg !151
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !153
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main.4d6ceefeb7f772f4(i32 %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !154
// CHECK:STDOUT:   ret i32 %F.call, !dbg !155
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CG.Main.c5e4ba542a56b71f(double %x, i32 %count) #0 !dbg !157 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 4, !dbg !164
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !165
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret double %x, !dbg !167
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !169
// CHECK:STDOUT:   %F.call = call double @_CF.Main.c5e4ba542a56b71f(double %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !170
// CHECK:STDOUT:   ret double %F.call, !dbg !171
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CG.Main.a76b78461416b4a8(ptr %x, i32 %count) #0 !dbg !173 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 4, !dbg !180
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !181
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret ptr %x, !dbg !183
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !185
// CHECK:STDOUT:   %F.call = call ptr @_CF.Main.a76b78461416b4a8(ptr %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !186
// CHECK:STDOUT:   ret ptr %F.call, !dbg !187
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CF.Main.a76b78461416b4a8, { 1, 2, 0 }
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
// CHECK:STDOUT: !9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !10 = !{null, !9}
// CHECK:STDOUT: !11 = !DISubroutineType(types: !10)
// CHECK:STDOUT: !12 = !{null, !6}
// CHECK:STDOUT: !13 = !DISubroutineType(types: !12)
// CHECK:STDOUT: !14 = distinct !DISubprogram(name: "M", linkageName: "_CM.Main", scope: null, file: !4, line: 32, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !15 = !DILocation(line: 33, column: 3, scope: !14)
// CHECK:STDOUT: !17 = !DILocation(line: 34, column: 3, scope: !14)
// CHECK:STDOUT: !19 = !DILocation(line: 35, column: 3, scope: !14)
// CHECK:STDOUT: !23 = !DILocation(line: 36, column: 3, scope: !14)
// CHECK:STDOUT: !27 = !DILocation(line: 38, column: 5, scope: !14)
// CHECK:STDOUT: !28 = !DILocation(line: 38, column: 3, scope: !14)
// CHECK:STDOUT: !29 = !{!6, !6, !6}
// CHECK:STDOUT: !30 = !DISubroutineType(types: !29)
// CHECK:STDOUT: !31 = !DILocation(line: 39, column: 5, scope: !14)
// CHECK:STDOUT: !32 = !DILocation(line: 39, column: 3, scope: !14)
// CHECK:STDOUT: !33 = !{!9, !9, !6}
// CHECK:STDOUT: !34 = !DISubroutineType(types: !33)
// CHECK:STDOUT: !35 = !DILocation(line: 40, column: 5, scope: !14)
// CHECK:STDOUT: !36 = !DILocation(line: 40, column: 3, scope: !14)
// CHECK:STDOUT: !37 = !DILocation(line: 41, column: 5, scope: !14)
// CHECK:STDOUT: !38 = !DILocation(line: 41, column: 3, scope: !14)
// CHECK:STDOUT: !39 = !DILocation(line: 32, column: 1, scope: !14)
// CHECK:STDOUT: !40 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.dc14d4dd21801791:core.Destroy.Core", scope: null, file: !4, line: 34, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !43)
// CHECK:STDOUT: !41 = !DILocalVariable(arg: 1, scope: !40, type: !9)
// CHECK:STDOUT: !42 = !DILocation(line: 34, column: 3, scope: !40)
// CHECK:STDOUT: !43 = !{!41}
// CHECK:STDOUT: !44 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core", scope: null, file: !4, line: 33, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !47)
// CHECK:STDOUT: !45 = !DILocalVariable(arg: 1, scope: !44, type: !6)
// CHECK:STDOUT: !46 = !DILocation(line: 33, column: 3, scope: !44)
// CHECK:STDOUT: !47 = !{!45}
// CHECK:STDOUT: !48 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.4d6ceefeb7f772f4", scope: null, file: !4, line: 18, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !63)
// CHECK:STDOUT: !49 = !DILocalVariable(arg: 1, scope: !48, type: !6)
// CHECK:STDOUT: !50 = !DILocalVariable(arg: 2, scope: !48, type: !6)
// CHECK:STDOUT: !55 = !DILocation(line: 19, column: 7, scope: !48)
// CHECK:STDOUT: !56 = !DILocation(line: 19, column: 6, scope: !48)
// CHECK:STDOUT: !58 = !DILocation(line: 20, column: 5, scope: !48)
// CHECK:STDOUT: !60 = !DILocation(line: 22, column: 15, scope: !48)
// CHECK:STDOUT: !61 = !DILocation(line: 22, column: 10, scope: !48)
// CHECK:STDOUT: !62 = !DILocation(line: 22, column: 3, scope: !48)
// CHECK:STDOUT: !63 = !{!49, !50}
// CHECK:STDOUT: !64 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.c5e4ba542a56b71f", scope: null, file: !4, line: 18, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !79)
// CHECK:STDOUT: !65 = !DILocalVariable(arg: 1, scope: !64, type: !9)
// CHECK:STDOUT: !66 = !DILocalVariable(arg: 2, scope: !64, type: !6)
// CHECK:STDOUT: !71 = !DILocation(line: 19, column: 7, scope: !64)
// CHECK:STDOUT: !72 = !DILocation(line: 19, column: 6, scope: !64)
// CHECK:STDOUT: !74 = !DILocation(line: 20, column: 5, scope: !64)
// CHECK:STDOUT: !76 = !DILocation(line: 22, column: 15, scope: !64)
// CHECK:STDOUT: !77 = !DILocation(line: 22, column: 10, scope: !64)
// CHECK:STDOUT: !78 = !DILocation(line: 22, column: 3, scope: !64)
// CHECK:STDOUT: !79 = !{!65, !66}
// CHECK:STDOUT: !80 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.a76b78461416b4a8", scope: null, file: !4, line: 18, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !95)
// CHECK:STDOUT: !81 = !DILocalVariable(arg: 1, scope: !80, type: !9)
// CHECK:STDOUT: !82 = !DILocalVariable(arg: 2, scope: !80, type: !6)
// CHECK:STDOUT: !87 = !DILocation(line: 19, column: 7, scope: !80)
// CHECK:STDOUT: !88 = !DILocation(line: 19, column: 6, scope: !80)
// CHECK:STDOUT: !90 = !DILocation(line: 20, column: 5, scope: !80)
// CHECK:STDOUT: !92 = !DILocation(line: 22, column: 15, scope: !80)
// CHECK:STDOUT: !93 = !DILocation(line: 22, column: 10, scope: !80)
// CHECK:STDOUT: !94 = !DILocation(line: 22, column: 3, scope: !80)
// CHECK:STDOUT: !95 = !{!81, !82}
// CHECK:STDOUT: !112 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !113 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.9f1044688d586372", scope: null, file: !112, line: 26, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !119)
// CHECK:STDOUT: !114 = !DILocalVariable(arg: 1, scope: !113, type: !9)
// CHECK:STDOUT: !118 = !DILocation(line: 26, column: 3, scope: !113)
// CHECK:STDOUT: !119 = !{!114}
// CHECK:STDOUT: !120 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.f427c19f24d378eb", scope: null, file: !112, line: 26, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !126)
// CHECK:STDOUT: !121 = !DILocalVariable(arg: 1, scope: !120, type: !9)
// CHECK:STDOUT: !125 = !DILocation(line: 26, column: 3, scope: !120)
// CHECK:STDOUT: !126 = !{!121}
// CHECK:STDOUT: !127 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.23895ff6626cffe1", scope: null, file: !112, line: 26, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !133)
// CHECK:STDOUT: !128 = !DILocalVariable(arg: 1, scope: !127, type: !9)
// CHECK:STDOUT: !131 = !DILocation(line: 28, column: 5, scope: !127)
// CHECK:STDOUT: !132 = !DILocation(line: 26, column: 3, scope: !127)
// CHECK:STDOUT: !133 = !{!128}
// CHECK:STDOUT: !134 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48", scope: null, file: !112, line: 26, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !140)
// CHECK:STDOUT: !135 = !DILocalVariable(arg: 1, scope: !134, type: !6)
// CHECK:STDOUT: !138 = !DILocation(line: 28, column: 5, scope: !134)
// CHECK:STDOUT: !139 = !DILocation(line: 26, column: 3, scope: !134)
// CHECK:STDOUT: !140 = !{!135}
// CHECK:STDOUT: !141 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.4d6ceefeb7f772f4", scope: null, file: !4, line: 25, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !156)
// CHECK:STDOUT: !142 = !DILocalVariable(arg: 1, scope: !141, type: !6)
// CHECK:STDOUT: !143 = !DILocalVariable(arg: 2, scope: !141, type: !6)
// CHECK:STDOUT: !148 = !DILocation(line: 26, column: 7, scope: !141)
// CHECK:STDOUT: !149 = !DILocation(line: 26, column: 6, scope: !141)
// CHECK:STDOUT: !151 = !DILocation(line: 27, column: 5, scope: !141)
// CHECK:STDOUT: !153 = !DILocation(line: 29, column: 15, scope: !141)
// CHECK:STDOUT: !154 = !DILocation(line: 29, column: 10, scope: !141)
// CHECK:STDOUT: !155 = !DILocation(line: 29, column: 3, scope: !141)
// CHECK:STDOUT: !156 = !{!142, !143}
// CHECK:STDOUT: !157 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.c5e4ba542a56b71f", scope: null, file: !4, line: 25, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !172)
// CHECK:STDOUT: !158 = !DILocalVariable(arg: 1, scope: !157, type: !9)
// CHECK:STDOUT: !159 = !DILocalVariable(arg: 2, scope: !157, type: !6)
// CHECK:STDOUT: !164 = !DILocation(line: 26, column: 7, scope: !157)
// CHECK:STDOUT: !165 = !DILocation(line: 26, column: 6, scope: !157)
// CHECK:STDOUT: !167 = !DILocation(line: 27, column: 5, scope: !157)
// CHECK:STDOUT: !169 = !DILocation(line: 29, column: 15, scope: !157)
// CHECK:STDOUT: !170 = !DILocation(line: 29, column: 10, scope: !157)
// CHECK:STDOUT: !171 = !DILocation(line: 29, column: 3, scope: !157)
// CHECK:STDOUT: !172 = !{!158, !159}
// CHECK:STDOUT: !173 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.a76b78461416b4a8", scope: null, file: !4, line: 25, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !188)
// CHECK:STDOUT: !174 = !DILocalVariable(arg: 1, scope: !173, type: !9)
// CHECK:STDOUT: !175 = !DILocalVariable(arg: 2, scope: !173, type: !6)
// CHECK:STDOUT: !180 = !DILocation(line: 26, column: 7, scope: !173)
// CHECK:STDOUT: !181 = !DILocation(line: 26, column: 6, scope: !173)
// CHECK:STDOUT: !183 = !DILocation(line: 27, column: 5, scope: !173)
// CHECK:STDOUT: !185 = !DILocation(line: 29, column: 15, scope: !173)
// CHECK:STDOUT: !186 = !DILocation(line: 29, column: 10, scope: !173)
// CHECK:STDOUT: !187 = !DILocation(line: 29, column: 3, scope: !173)
// CHECK:STDOUT: !188 = !{!174, !175}
// CHECK:STDOUT:
