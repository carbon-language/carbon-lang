// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/generic/call_recursive_reorder.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/generic/call_recursive_reorder.carbon

// Builds on `call_recursive_basic.carbon`.

fn F[T: type, U: type](x: T, y: U, count: i32) -> i32 {
  if (count > 2) {
    return count;
  }
  return F(y, x, count + 1);
}

fn M() {
  var n: i32 = 0;
  var m: f64 = 1.0;
  var b: bool = false;
  var ptr_i32 : i32* = &n;
  var ptr_f64 : f64* = &m;
  var ptr_bool : bool* = &b;

  F(ptr_i32, ptr_bool, 0);
  F(ptr_f64, ptr_bool, 0);
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'call_recursive_reorder.carbon'
// CHECK:STDOUT: source_filename = "call_recursive_reorder.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CM.Main() #0 !dbg !9 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %n.var = alloca i32, align 4, !dbg !10
// CHECK:STDOUT:   %m.var = alloca double, align 8, !dbg !12
// CHECK:STDOUT:   %b.var = alloca i1, align 1, !dbg !14
// CHECK:STDOUT:   %ptr_i32.var = alloca ptr, align 8, !dbg !16
// CHECK:STDOUT:   %ptr_f64.var = alloca ptr, align 8, !dbg !20
// CHECK:STDOUT:   %ptr_bool.var = alloca ptr, align 8, !dbg !24
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %n.var), !dbg !10
// CHECK:STDOUT:   store i32 0, ptr %n.var, align 4, !dbg !10
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %m.var), !dbg !12
// CHECK:STDOUT:   store double 1.000000e+00, ptr %m.var, align 8, !dbg !12
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b.var), !dbg !14
// CHECK:STDOUT:   store i8 0, ptr %b.var, align 1, !dbg !14
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_i32.var), !dbg !16
// CHECK:STDOUT:   store ptr %n.var, ptr %ptr_i32.var, align 8, !dbg !16
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_f64.var), !dbg !20
// CHECK:STDOUT:   store ptr %m.var, ptr %ptr_f64.var, align 8, !dbg !20
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_bool.var), !dbg !24
// CHECK:STDOUT:   store ptr %b.var, ptr %ptr_bool.var, align 8, !dbg !24
// CHECK:STDOUT:   %.loc30_5 = load ptr, ptr %ptr_i32.var, align 8, !dbg !28
// CHECK:STDOUT:   %.loc30_14 = load ptr, ptr %ptr_bool.var, align 8, !dbg !29
// CHECK:STDOUT:   %F.call.loc30 = call i32 @_CF.Main.97e933e3a3e841e9(ptr %.loc30_5, ptr %.loc30_14, i32 0), !dbg !30
// CHECK:STDOUT:   %.loc31_5 = load ptr, ptr %ptr_f64.var, align 8, !dbg !34
// CHECK:STDOUT:   %.loc31_14 = load ptr, ptr %ptr_bool.var, align 8, !dbg !35
// CHECK:STDOUT:   %F.call.loc31 = call i32 @_CF.Main.97e933e3a3e841e9(ptr %.loc31_5, ptr %.loc31_14, i32 0), !dbg !36
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.393ac80c0c8a44df(ptr %ptr_bool.var), !dbg !24
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.b18855c76072615b(ptr %ptr_f64.var), !dbg !20
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.4f29d15a20894e9b(ptr %ptr_i32.var), !dbg !16
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.29e6aa423646b61a(ptr %b.var), !dbg !14
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.2b38c2ca9f7f1a94(ptr %m.var), !dbg !12
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %n.var), !dbg !10
// CHECK:STDOUT:   ret void, !dbg !41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.97e933e3a3e841e9(ptr %x, ptr %y, i32 %count) #0 !dbg !42 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 2, !dbg !51
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !52
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret i32 %count, !dbg !54
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !57
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main.97e933e3a3e841e9(ptr %y, ptr %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !58
// CHECK:STDOUT:   ret i32 %F.call, !dbg !59
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.393ac80c0c8a44df(ptr %t) #0 !dbg !81 {
// CHECK:STDOUT:   ret void, !dbg !86
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.b18855c76072615b(ptr %t) #0 !dbg !88 {
// CHECK:STDOUT:   ret void, !dbg !93
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.4f29d15a20894e9b(ptr %t) #0 !dbg !95 {
// CHECK:STDOUT:   ret void, !dbg !100
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.29e6aa423646b61a(ptr %t) #0 !dbg !102 {
// CHECK:STDOUT:   ret void, !dbg !107
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.2b38c2ca9f7f1a94(ptr %t) #0 !dbg !109 {
// CHECK:STDOUT:   ret void, !dbg !114
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.f127f30143696d51(ptr %t) #0 !dbg !116 {
// CHECK:STDOUT:   ret void, !dbg !121
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CF.Main.97e933e3a3e841e9, { 0, 2, 1 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "call_recursive_reorder.carbon", directory: "")
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !7 = !{null}
// CHECK:STDOUT: !8 = !DISubroutineType(types: !7)
// CHECK:STDOUT: !9 = distinct !DISubprogram(name: "M", linkageName: "_CM.Main", scope: null, file: !4, line: 22, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !10 = !DILocation(line: 23, column: 3, scope: !9)
// CHECK:STDOUT: !12 = !DILocation(line: 24, column: 3, scope: !9)
// CHECK:STDOUT: !14 = !DILocation(line: 25, column: 3, scope: !9)
// CHECK:STDOUT: !16 = !DILocation(line: 26, column: 3, scope: !9)
// CHECK:STDOUT: !20 = !DILocation(line: 27, column: 3, scope: !9)
// CHECK:STDOUT: !24 = !DILocation(line: 28, column: 3, scope: !9)
// CHECK:STDOUT: !28 = !DILocation(line: 30, column: 5, scope: !9)
// CHECK:STDOUT: !29 = !DILocation(line: 30, column: 14, scope: !9)
// CHECK:STDOUT: !30 = !DILocation(line: 30, column: 3, scope: !9)
// CHECK:STDOUT: !31 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !32 = !{!6, !31, !31, !6}
// CHECK:STDOUT: !33 = !DISubroutineType(types: !32)
// CHECK:STDOUT: !34 = !DILocation(line: 31, column: 5, scope: !9)
// CHECK:STDOUT: !35 = !DILocation(line: 31, column: 14, scope: !9)
// CHECK:STDOUT: !36 = !DILocation(line: 31, column: 3, scope: !9)
// CHECK:STDOUT: !37 = !{null, !31}
// CHECK:STDOUT: !38 = !DISubroutineType(types: !37)
// CHECK:STDOUT: !39 = !{null, !6}
// CHECK:STDOUT: !40 = !DISubroutineType(types: !39)
// CHECK:STDOUT: !41 = !DILocation(line: 22, column: 1, scope: !9)
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.97e933e3a3e841e9", scope: null, file: !4, line: 15, type: !33, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !60)
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 1, scope: !42, type: !31)
// CHECK:STDOUT: !44 = !DILocalVariable(arg: 2, scope: !42, type: !31)
// CHECK:STDOUT: !45 = !DILocalVariable(arg: 3, scope: !42, type: !6)
// CHECK:STDOUT: !51 = !DILocation(line: 16, column: 7, scope: !42)
// CHECK:STDOUT: !52 = !DILocation(line: 16, column: 6, scope: !42)
// CHECK:STDOUT: !54 = !DILocation(line: 17, column: 5, scope: !42)
// CHECK:STDOUT: !57 = !DILocation(line: 19, column: 18, scope: !42)
// CHECK:STDOUT: !58 = !DILocation(line: 19, column: 10, scope: !42)
// CHECK:STDOUT: !59 = !DILocation(line: 19, column: 3, scope: !42)
// CHECK:STDOUT: !60 = !{!43, !44, !45}
// CHECK:STDOUT: !80 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !81 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.393ac80c0c8a44df", scope: null, file: !80, line: 35, type: !38, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !87)
// CHECK:STDOUT: !82 = !DILocalVariable(arg: 1, scope: !81, type: !31)
// CHECK:STDOUT: !86 = !DILocation(line: 35, column: 1, scope: !81)
// CHECK:STDOUT: !87 = !{!82}
// CHECK:STDOUT: !88 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.b18855c76072615b", scope: null, file: !80, line: 35, type: !38, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !94)
// CHECK:STDOUT: !89 = !DILocalVariable(arg: 1, scope: !88, type: !31)
// CHECK:STDOUT: !93 = !DILocation(line: 35, column: 1, scope: !88)
// CHECK:STDOUT: !94 = !{!89}
// CHECK:STDOUT: !95 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.4f29d15a20894e9b", scope: null, file: !80, line: 35, type: !38, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !101)
// CHECK:STDOUT: !96 = !DILocalVariable(arg: 1, scope: !95, type: !31)
// CHECK:STDOUT: !100 = !DILocation(line: 35, column: 1, scope: !95)
// CHECK:STDOUT: !101 = !{!96}
// CHECK:STDOUT: !102 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.29e6aa423646b61a", scope: null, file: !80, line: 35, type: !38, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !108)
// CHECK:STDOUT: !103 = !DILocalVariable(arg: 1, scope: !102, type: !31)
// CHECK:STDOUT: !107 = !DILocation(line: 35, column: 1, scope: !102)
// CHECK:STDOUT: !108 = !{!103}
// CHECK:STDOUT: !109 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.2b38c2ca9f7f1a94", scope: null, file: !80, line: 35, type: !38, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !115)
// CHECK:STDOUT: !110 = !DILocalVariable(arg: 1, scope: !109, type: !31)
// CHECK:STDOUT: !114 = !DILocation(line: 35, column: 1, scope: !109)
// CHECK:STDOUT: !115 = !{!110}
// CHECK:STDOUT: !116 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.f127f30143696d51", scope: null, file: !80, line: 35, type: !40, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !122)
// CHECK:STDOUT: !117 = !DILocalVariable(arg: 1, scope: !116, type: !6)
// CHECK:STDOUT: !121 = !DILocation(line: 35, column: 1, scope: !116)
// CHECK:STDOUT: !122 = !{!117}
// CHECK:STDOUT:
