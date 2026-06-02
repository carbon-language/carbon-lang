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

fn F[T:! type, U:! type](x: T, y: U, count: i32) -> i32 {
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

// CHECK:STDOUT: ; ModuleID = 'call_recursive_reorder.carbon'
// CHECK:STDOUT: source_filename = "call_recursive_reorder.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CM.Main() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %n.var = alloca i32, align 4, !dbg !7
// CHECK:STDOUT:   %m.var = alloca double, align 8, !dbg !8
// CHECK:STDOUT:   %b.var = alloca i1, align 1, !dbg !9
// CHECK:STDOUT:   %ptr_i32.var = alloca ptr, align 8, !dbg !10
// CHECK:STDOUT:   %ptr_f64.var = alloca ptr, align 8, !dbg !11
// CHECK:STDOUT:   %ptr_bool.var = alloca ptr, align 8, !dbg !12
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %n.var), !dbg !7
// CHECK:STDOUT:   store i32 0, ptr %n.var, align 4, !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %m.var), !dbg !8
// CHECK:STDOUT:   store double 1.000000e+00, ptr %m.var, align 8, !dbg !8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b.var), !dbg !9
// CHECK:STDOUT:   store i8 0, ptr %b.var, align 1, !dbg !9
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_i32.var), !dbg !10
// CHECK:STDOUT:   store ptr %n.var, ptr %ptr_i32.var, align 8, !dbg !10
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_f64.var), !dbg !11
// CHECK:STDOUT:   store ptr %m.var, ptr %ptr_f64.var, align 8, !dbg !11
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_bool.var), !dbg !12
// CHECK:STDOUT:   store ptr %b.var, ptr %ptr_bool.var, align 8, !dbg !12
// CHECK:STDOUT:   %.loc30_5 = load ptr, ptr %ptr_i32.var, align 8, !dbg !13
// CHECK:STDOUT:   %.loc30_14 = load ptr, ptr %ptr_bool.var, align 8, !dbg !14
// CHECK:STDOUT:   %F.call.loc30 = call i32 @_CF.Main.a6405d95a65b9c74(ptr %.loc30_5, ptr %.loc30_14, i32 0), !dbg !15
// CHECK:STDOUT:   %.loc31_5 = load ptr, ptr %ptr_f64.var, align 8, !dbg !16
// CHECK:STDOUT:   %.loc31_14 = load ptr, ptr %ptr_bool.var, align 8, !dbg !17
// CHECK:STDOUT:   %F.call.loc31 = call i32 @_CF.Main.a6405d95a65b9c74(ptr %.loc31_5, ptr %.loc31_14, i32 0), !dbg !18
// CHECK:STDOUT:   call void @"_COp.b46d7363edf385cc:core.Destroy.Core"(ptr %m.var), !dbg !8
// CHECK:STDOUT:   call void @"_COp.94e3ca86614aba5d:core.Destroy.Core"(ptr %n.var), !dbg !7
// CHECK:STDOUT:   ret void, !dbg !19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.b46d7363edf385cc:core.Destroy.Core"(ptr %self) #0 !dbg !20 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.94e3ca86614aba5d:core.Destroy.Core"(ptr %self) #0 !dbg !27 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.a6405d95a65b9c74(ptr %x, ptr %y, i32 %count) #0 !dbg !34 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 2, !dbg !41
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !42
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret i32 %count, !dbg !43
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !44
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main.a6405d95a65b9c74(ptr %y, ptr %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !45
// CHECK:STDOUT:   ret i32 %F.call, !dbg !46
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CF.Main.a6405d95a65b9c74, { 0, 2, 1 }
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
// CHECK:STDOUT: !3 = !DIFile(filename: "call_recursive_reorder.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "M", linkageName: "_CM.Main", scope: null, file: !3, line: 22, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DILocation(line: 23, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 24, column: 3, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 25, column: 3, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 26, column: 3, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 27, column: 3, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 28, column: 3, scope: !4)
// CHECK:STDOUT: !13 = !DILocation(line: 30, column: 5, scope: !4)
// CHECK:STDOUT: !14 = !DILocation(line: 30, column: 14, scope: !4)
// CHECK:STDOUT: !15 = !DILocation(line: 30, column: 3, scope: !4)
// CHECK:STDOUT: !16 = !DILocation(line: 31, column: 5, scope: !4)
// CHECK:STDOUT: !17 = !DILocation(line: 31, column: 14, scope: !4)
// CHECK:STDOUT: !18 = !DILocation(line: 31, column: 3, scope: !4)
// CHECK:STDOUT: !19 = !DILocation(line: 22, column: 1, scope: !4)
// CHECK:STDOUT: !20 = distinct !DISubprogram(name: "Op", linkageName: "_COp.b46d7363edf385cc:core.Destroy.Core", scope: null, file: !3, line: 24, type: !21, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !24)
// CHECK:STDOUT: !21 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !22 = !{null, !23}
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{!25}
// CHECK:STDOUT: !25 = !DILocalVariable(arg: 1, scope: !20, type: !23)
// CHECK:STDOUT: !26 = !DILocation(line: 24, column: 3, scope: !20)
// CHECK:STDOUT: !27 = distinct !DISubprogram(name: "Op", linkageName: "_COp.94e3ca86614aba5d:core.Destroy.Core", scope: null, file: !3, line: 23, type: !28, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !31)
// CHECK:STDOUT: !28 = !DISubroutineType(types: !29)
// CHECK:STDOUT: !29 = !{null, !30}
// CHECK:STDOUT: !30 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !31 = !{!32}
// CHECK:STDOUT: !32 = !DILocalVariable(arg: 1, scope: !27, type: !30)
// CHECK:STDOUT: !33 = !DILocation(line: 23, column: 3, scope: !27)
// CHECK:STDOUT: !34 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.a6405d95a65b9c74", scope: null, file: !3, line: 15, type: !35, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !37)
// CHECK:STDOUT: !35 = !DISubroutineType(types: !36)
// CHECK:STDOUT: !36 = !{!30, !23, !23, !30}
// CHECK:STDOUT: !37 = !{!38, !39, !40}
// CHECK:STDOUT: !38 = !DILocalVariable(arg: 1, scope: !34, type: !23)
// CHECK:STDOUT: !39 = !DILocalVariable(arg: 2, scope: !34, type: !23)
// CHECK:STDOUT: !40 = !DILocalVariable(arg: 3, scope: !34, type: !30)
// CHECK:STDOUT: !41 = !DILocation(line: 16, column: 7, scope: !34)
// CHECK:STDOUT: !42 = !DILocation(line: 16, column: 6, scope: !34)
// CHECK:STDOUT: !43 = !DILocation(line: 17, column: 5, scope: !34)
// CHECK:STDOUT: !44 = !DILocation(line: 19, column: 18, scope: !34)
// CHECK:STDOUT: !45 = !DILocation(line: 19, column: 10, scope: !34)
// CHECK:STDOUT: !46 = !DILocation(line: 19, column: 3, scope: !34)
