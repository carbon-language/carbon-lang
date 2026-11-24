// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/generic/call_recursive_basic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/generic/call_recursive_basic.carbon

class C {
  impl as Core.Copy {
    fn Op(unused self) -> Self { return {}; }
  }
}

// The two specifics for recursive function F, when T is a pointer, could be
// deduplicated.
fn F[T: Core.Copy](x: T, count: i32) -> T {
  if (count > 0) {
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

  var c: C = {};
  F(c, 0);
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'call_recursive_basic.carbon'
// CHECK:STDOUT: source_filename = "call_recursive_basic.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @C.val.loc15_43 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_COp.C.Main:Copy.Core"(ptr sret({}) %return, ptr %self) #0 !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %return, ptr align 1 @C.val.loc15_43, i64 0, i1 false), !dbg !17
// CHECK:STDOUT:   ret void, !dbg !17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CM.Main() #0 !dbg !19 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %n.var = alloca i32, align 4, !dbg !20
// CHECK:STDOUT:   %m.var = alloca double, align 8, !dbg !22
// CHECK:STDOUT:   %ptr_i32.var = alloca ptr, align 8, !dbg !24
// CHECK:STDOUT:   %ptr_f64.var = alloca ptr, align 8, !dbg !28
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !44
// CHECK:STDOUT:   %.loc40_9.3.temp = alloca {}, align 1, !dbg !47
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %n.var), !dbg !20
// CHECK:STDOUT:   store i32 0, ptr %n.var, align 4, !dbg !20
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %m.var), !dbg !22
// CHECK:STDOUT:   store double 1.000000e+00, ptr %m.var, align 8, !dbg !22
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_i32.var), !dbg !24
// CHECK:STDOUT:   store ptr %n.var, ptr %ptr_i32.var, align 8, !dbg !24
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_f64.var), !dbg !28
// CHECK:STDOUT:   store ptr %m.var, ptr %ptr_f64.var, align 8, !dbg !28
// CHECK:STDOUT:   %.loc34_5 = load i32, ptr %n.var, align 4, !dbg !32
// CHECK:STDOUT:   %F.call.loc34 = call i32 @_CF.Main.2fad9574e050607e(i32 %.loc34_5, i32 0), !dbg !33
// CHECK:STDOUT:   %.loc35_5 = load double, ptr %m.var, align 8, !dbg !36
// CHECK:STDOUT:   %F.call.loc35 = call double @_CF.Main.21aefb7dfdebb8e4(double %.loc35_5, i32 0), !dbg !37
// CHECK:STDOUT:   %.loc36_5 = load ptr, ptr %ptr_i32.var, align 8, !dbg !40
// CHECK:STDOUT:   %F.call.loc36 = call ptr @_CF.Main.417c9c4a651ab31b(ptr %.loc36_5, i32 0), !dbg !41
// CHECK:STDOUT:   %.loc37_5 = load ptr, ptr %ptr_f64.var, align 8, !dbg !42
// CHECK:STDOUT:   %F.call.loc37 = call ptr @_CF.Main.417c9c4a651ab31b(ptr %.loc37_5, i32 0), !dbg !43
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !44
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %c.var, ptr align 1 @C.val.loc15_43, i64 0, i1 false), !dbg !44
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc40_9.3.temp), !dbg !47
// CHECK:STDOUT:   call void @_CF.Main.42513763e01cba2c(ptr %.loc40_9.3.temp, ptr %c.var, i32 0), !dbg !47
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.356af3b6aef18564(ptr %.loc40_9.3.temp), !dbg !47
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.356af3b6aef18564(ptr %c.var), !dbg !44
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.b18855c76072615b(ptr %ptr_f64.var), !dbg !28
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.4f29d15a20894e9b(ptr %ptr_i32.var), !dbg !24
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.2b38c2ca9f7f1a94(ptr %m.var), !dbg !22
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %n.var), !dbg !20
// CHECK:STDOUT:   ret void, !dbg !52
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.2fad9574e050607e(i32 %x, i32 %count) #0 !dbg !53 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 0, !dbg !60
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !61
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret i32 %x, !dbg !63
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !65
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main.2fad9574e050607e(i32 %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !66
// CHECK:STDOUT:   ret i32 %F.call, !dbg !67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CF.Main.21aefb7dfdebb8e4(double %x, i32 %count) #0 !dbg !69 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 0, !dbg !76
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !77
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret double %x, !dbg !79
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !81
// CHECK:STDOUT:   %F.call = call double @_CF.Main.21aefb7dfdebb8e4(double %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !82
// CHECK:STDOUT:   ret double %F.call, !dbg !83
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CF.Main.417c9c4a651ab31b(ptr %x, i32 %count) #0 !dbg !85 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 0, !dbg !92
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !93
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret ptr %x, !dbg !95
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !97
// CHECK:STDOUT:   %F.call = call ptr @_CF.Main.417c9c4a651ab31b(ptr %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !98
// CHECK:STDOUT:   ret ptr %F.call, !dbg !99
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.42513763e01cba2c(ptr sret({}) %return, ptr %x, i32 %count) #0 !dbg !117 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 0, !dbg !124
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !125
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   call void @"_COp.C.Main:Copy.Core"(ptr %return, ptr %x), !dbg !126
// CHECK:STDOUT:   ret void, !dbg !127
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !129
// CHECK:STDOUT:   call void @_CF.Main.42513763e01cba2c(ptr %return, ptr %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !130
// CHECK:STDOUT:   ret void, !dbg !131
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.356af3b6aef18564(ptr %t) #0 !dbg !134 {
// CHECK:STDOUT:   ret void, !dbg !139
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.b18855c76072615b(ptr %t) #0 !dbg !141 {
// CHECK:STDOUT:   ret void, !dbg !146
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.4f29d15a20894e9b(ptr %t) #0 !dbg !148 {
// CHECK:STDOUT:   ret void, !dbg !153
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.2b38c2ca9f7f1a94(ptr %t) #0 !dbg !155 {
// CHECK:STDOUT:   ret void, !dbg !160
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.f127f30143696d51(ptr %t) #0 !dbg !162 {
// CHECK:STDOUT:   ret void, !dbg !167
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CF.Main.417c9c4a651ab31b, { 1, 2, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.356af3b6aef18564, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "call_recursive_basic.carbon", directory: "")
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{!7, !7}
// CHECK:STDOUT: !9 = !DISubroutineType(types: !8)
// CHECK:STDOUT: !10 = !{null}
// CHECK:STDOUT: !11 = !DISubroutineType(types: !10)
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "Op", linkageName: "_COp.C.Main:Copy.Core", scope: null, file: !4, line: 15, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !18)
// CHECK:STDOUT: !13 = !DILocalVariable(arg: 1, scope: !12, type: !7)
// CHECK:STDOUT: !17 = !DILocation(line: 15, column: 34, scope: !12)
// CHECK:STDOUT: !18 = !{!13}
// CHECK:STDOUT: !19 = distinct !DISubprogram(name: "M", linkageName: "_CM.Main", scope: null, file: !4, line: 28, type: !11, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !20 = !DILocation(line: 29, column: 3, scope: !19)
// CHECK:STDOUT: !22 = !DILocation(line: 30, column: 3, scope: !19)
// CHECK:STDOUT: !24 = !DILocation(line: 31, column: 3, scope: !19)
// CHECK:STDOUT: !28 = !DILocation(line: 32, column: 3, scope: !19)
// CHECK:STDOUT: !32 = !DILocation(line: 34, column: 5, scope: !19)
// CHECK:STDOUT: !33 = !DILocation(line: 34, column: 3, scope: !19)
// CHECK:STDOUT: !34 = !{!6, !6, !6}
// CHECK:STDOUT: !35 = !DISubroutineType(types: !34)
// CHECK:STDOUT: !36 = !DILocation(line: 35, column: 5, scope: !19)
// CHECK:STDOUT: !37 = !DILocation(line: 35, column: 3, scope: !19)
// CHECK:STDOUT: !38 = !{!7, !7, !6}
// CHECK:STDOUT: !39 = !DISubroutineType(types: !38)
// CHECK:STDOUT: !40 = !DILocation(line: 36, column: 5, scope: !19)
// CHECK:STDOUT: !41 = !DILocation(line: 36, column: 3, scope: !19)
// CHECK:STDOUT: !42 = !DILocation(line: 37, column: 5, scope: !19)
// CHECK:STDOUT: !43 = !DILocation(line: 37, column: 3, scope: !19)
// CHECK:STDOUT: !44 = !DILocation(line: 39, column: 3, scope: !19)
// CHECK:STDOUT: !47 = !DILocation(line: 40, column: 3, scope: !19)
// CHECK:STDOUT: !48 = !{null, !7}
// CHECK:STDOUT: !49 = !DISubroutineType(types: !48)
// CHECK:STDOUT: !50 = !{null, !6}
// CHECK:STDOUT: !51 = !DISubroutineType(types: !50)
// CHECK:STDOUT: !52 = !DILocation(line: 28, column: 1, scope: !19)
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.2fad9574e050607e", scope: null, file: !4, line: 21, type: !35, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !68)
// CHECK:STDOUT: !54 = !DILocalVariable(arg: 1, scope: !53, type: !6)
// CHECK:STDOUT: !55 = !DILocalVariable(arg: 2, scope: !53, type: !6)
// CHECK:STDOUT: !60 = !DILocation(line: 22, column: 7, scope: !53)
// CHECK:STDOUT: !61 = !DILocation(line: 22, column: 6, scope: !53)
// CHECK:STDOUT: !63 = !DILocation(line: 23, column: 5, scope: !53)
// CHECK:STDOUT: !65 = !DILocation(line: 25, column: 15, scope: !53)
// CHECK:STDOUT: !66 = !DILocation(line: 25, column: 10, scope: !53)
// CHECK:STDOUT: !67 = !DILocation(line: 25, column: 3, scope: !53)
// CHECK:STDOUT: !68 = !{!54, !55}
// CHECK:STDOUT: !69 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.21aefb7dfdebb8e4", scope: null, file: !4, line: 21, type: !39, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !84)
// CHECK:STDOUT: !70 = !DILocalVariable(arg: 1, scope: !69, type: !7)
// CHECK:STDOUT: !71 = !DILocalVariable(arg: 2, scope: !69, type: !6)
// CHECK:STDOUT: !76 = !DILocation(line: 22, column: 7, scope: !69)
// CHECK:STDOUT: !77 = !DILocation(line: 22, column: 6, scope: !69)
// CHECK:STDOUT: !79 = !DILocation(line: 23, column: 5, scope: !69)
// CHECK:STDOUT: !81 = !DILocation(line: 25, column: 15, scope: !69)
// CHECK:STDOUT: !82 = !DILocation(line: 25, column: 10, scope: !69)
// CHECK:STDOUT: !83 = !DILocation(line: 25, column: 3, scope: !69)
// CHECK:STDOUT: !84 = !{!70, !71}
// CHECK:STDOUT: !85 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.417c9c4a651ab31b", scope: null, file: !4, line: 21, type: !39, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !100)
// CHECK:STDOUT: !86 = !DILocalVariable(arg: 1, scope: !85, type: !7)
// CHECK:STDOUT: !87 = !DILocalVariable(arg: 2, scope: !85, type: !6)
// CHECK:STDOUT: !92 = !DILocation(line: 22, column: 7, scope: !85)
// CHECK:STDOUT: !93 = !DILocation(line: 22, column: 6, scope: !85)
// CHECK:STDOUT: !95 = !DILocation(line: 23, column: 5, scope: !85)
// CHECK:STDOUT: !97 = !DILocation(line: 25, column: 15, scope: !85)
// CHECK:STDOUT: !98 = !DILocation(line: 25, column: 10, scope: !85)
// CHECK:STDOUT: !99 = !DILocation(line: 25, column: 3, scope: !85)
// CHECK:STDOUT: !100 = !{!86, !87}
// CHECK:STDOUT: !117 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.42513763e01cba2c", scope: null, file: !4, line: 21, type: !39, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !132)
// CHECK:STDOUT: !118 = !DILocalVariable(arg: 1, scope: !117, type: !7)
// CHECK:STDOUT: !119 = !DILocalVariable(arg: 2, scope: !117, type: !6)
// CHECK:STDOUT: !124 = !DILocation(line: 22, column: 7, scope: !117)
// CHECK:STDOUT: !125 = !DILocation(line: 22, column: 6, scope: !117)
// CHECK:STDOUT: !126 = !DILocation(line: 23, column: 12, scope: !117)
// CHECK:STDOUT: !127 = !DILocation(line: 23, column: 5, scope: !117)
// CHECK:STDOUT: !129 = !DILocation(line: 25, column: 15, scope: !117)
// CHECK:STDOUT: !130 = !DILocation(line: 25, column: 10, scope: !117)
// CHECK:STDOUT: !131 = !DILocation(line: 25, column: 3, scope: !117)
// CHECK:STDOUT: !132 = !{!118, !119}
// CHECK:STDOUT: !133 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !134 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.356af3b6aef18564", scope: null, file: !133, line: 35, type: !49, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !140)
// CHECK:STDOUT: !135 = !DILocalVariable(arg: 1, scope: !134, type: !7)
// CHECK:STDOUT: !139 = !DILocation(line: 35, column: 1, scope: !134)
// CHECK:STDOUT: !140 = !{!135}
// CHECK:STDOUT: !141 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.b18855c76072615b", scope: null, file: !133, line: 35, type: !49, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !147)
// CHECK:STDOUT: !142 = !DILocalVariable(arg: 1, scope: !141, type: !7)
// CHECK:STDOUT: !146 = !DILocation(line: 35, column: 1, scope: !141)
// CHECK:STDOUT: !147 = !{!142}
// CHECK:STDOUT: !148 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.4f29d15a20894e9b", scope: null, file: !133, line: 35, type: !49, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !154)
// CHECK:STDOUT: !149 = !DILocalVariable(arg: 1, scope: !148, type: !7)
// CHECK:STDOUT: !153 = !DILocation(line: 35, column: 1, scope: !148)
// CHECK:STDOUT: !154 = !{!149}
// CHECK:STDOUT: !155 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.2b38c2ca9f7f1a94", scope: null, file: !133, line: 35, type: !49, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !161)
// CHECK:STDOUT: !156 = !DILocalVariable(arg: 1, scope: !155, type: !7)
// CHECK:STDOUT: !160 = !DILocation(line: 35, column: 1, scope: !155)
// CHECK:STDOUT: !161 = !{!156}
// CHECK:STDOUT: !162 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.f127f30143696d51", scope: null, file: !133, line: 35, type: !51, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !168)
// CHECK:STDOUT: !163 = !DILocalVariable(arg: 1, scope: !162, type: !6)
// CHECK:STDOUT: !167 = !DILocation(line: 35, column: 1, scope: !162)
// CHECK:STDOUT: !168 = !{!163}
// CHECK:STDOUT:
