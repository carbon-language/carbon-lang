// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/generic/call.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/generic/call.carbon

fn F[T:! type](_: T) {
}

class C {}
class D {}

fn G() {
  var c: C = {};
  var d: D = {};
  var n: i32 = 0;
  var m: f64 = 0.0;

  F(c);
  F(d);
  F(n);
  F(m);
  F(i32);
}

// CHECK:STDOUT: ; ModuleID = 'call.carbon'
// CHECK:STDOUT: source_filename = "call.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: %type = type {}
// CHECK:STDOUT:
// CHECK:STDOUT: @C.val.loc20_3 = internal constant {} zeroinitializer
// CHECK:STDOUT: @D.val.loc21_3 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca {}, align 8, !dbg !7
// CHECK:STDOUT:   %d.var = alloca {}, align 8, !dbg !8
// CHECK:STDOUT:   %n.var = alloca i32, align 4, !dbg !9
// CHECK:STDOUT:   %m.var = alloca double, align 8, !dbg !10
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !7
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %c.var, ptr align 1 @C.val.loc20_3, i64 0, i1 false), !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %d.var), !dbg !8
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %d.var, ptr align 1 @D.val.loc21_3, i64 0, i1 false), !dbg !8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %n.var), !dbg !9
// CHECK:STDOUT:   store i32 0, ptr %n.var, align 4, !dbg !9
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %m.var), !dbg !10
// CHECK:STDOUT:   store double 0.000000e+00, ptr %m.var, align 8, !dbg !10
// CHECK:STDOUT:   call void @_CF.Main.0c4ab795cec6cb27(ptr %c.var), !dbg !11
// CHECK:STDOUT:   call void @_CF.Main.0c4ab795cec6cb27(ptr %d.var), !dbg !12
// CHECK:STDOUT:   %.loc27 = load i32, ptr %n.var, align 4, !dbg !13
// CHECK:STDOUT:   call void @_CF.Main.b88d1103f417c6d4(i32 %.loc27), !dbg !14
// CHECK:STDOUT:   %.loc28 = load double, ptr %m.var, align 8, !dbg !15
// CHECK:STDOUT:   call void @_CF.Main.d4b5665541d5d7a8(double %.loc28), !dbg !16
// CHECK:STDOUT:   call void @_CF.Main.5754c7a55c7cbe4a(%type zeroinitializer), !dbg !17
// CHECK:STDOUT:   call void @"_COp.98006a366c2ffd6c:core.Destroy.Core"(ptr %m.var), !dbg !10
// CHECK:STDOUT:   call void @"_COp.7e389eab4a7e5487:core.Destroy.Core"(ptr %n.var), !dbg !9
// CHECK:STDOUT:   call void @"_COp.bad4d5b3ac96d9d9:core.Destroy.Core"(ptr %d.var), !dbg !8
// CHECK:STDOUT:   call void @"_COp.71e25083d63c4d6c:core.Destroy.Core"(ptr %c.var), !dbg !7
// CHECK:STDOUT:   ret void, !dbg !18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.98006a366c2ffd6c:core.Destroy.Core"(ptr %self) #0 !dbg !19 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.7e389eab4a7e5487:core.Destroy.Core"(ptr %self) #0 !dbg !26 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.bad4d5b3ac96d9d9:core.Destroy.Core"(ptr %self) #0 !dbg !33 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.71e25083d63c4d6c:core.Destroy.Core"(ptr %self) #0 !dbg !37 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.0c4ab795cec6cb27(ptr %_) #0 !dbg !41 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.b88d1103f417c6d4(i32 %_) #0 !dbg !45 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.d4b5665541d5d7a8(double %_) #0 !dbg !49 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !52
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.5754c7a55c7cbe4a(%type %_) #0 !dbg !53 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !56
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CF.Main.0c4ab795cec6cb27, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "call.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !3, line: 19, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DILocation(line: 20, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 21, column: 3, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 22, column: 3, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 23, column: 3, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 25, column: 3, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 26, column: 3, scope: !4)
// CHECK:STDOUT: !13 = !DILocation(line: 27, column: 5, scope: !4)
// CHECK:STDOUT: !14 = !DILocation(line: 27, column: 3, scope: !4)
// CHECK:STDOUT: !15 = !DILocation(line: 28, column: 5, scope: !4)
// CHECK:STDOUT: !16 = !DILocation(line: 28, column: 3, scope: !4)
// CHECK:STDOUT: !17 = !DILocation(line: 29, column: 3, scope: !4)
// CHECK:STDOUT: !18 = !DILocation(line: 19, column: 1, scope: !4)
// CHECK:STDOUT: !19 = distinct !DISubprogram(name: "Op", linkageName: "_COp.98006a366c2ffd6c:core.Destroy.Core", scope: null, file: !3, line: 23, type: !20, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !23)
// CHECK:STDOUT: !20 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !21 = !{null, !22}
// CHECK:STDOUT: !22 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !23 = !{!24}
// CHECK:STDOUT: !24 = !DILocalVariable(arg: 1, scope: !19, type: !22)
// CHECK:STDOUT: !25 = !DILocation(line: 23, column: 3, scope: !19)
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "Op", linkageName: "_COp.7e389eab4a7e5487:core.Destroy.Core", scope: null, file: !3, line: 22, type: !27, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !30)
// CHECK:STDOUT: !27 = !DISubroutineType(types: !28)
// CHECK:STDOUT: !28 = !{null, !29}
// CHECK:STDOUT: !29 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !30 = !{!31}
// CHECK:STDOUT: !31 = !DILocalVariable(arg: 1, scope: !26, type: !29)
// CHECK:STDOUT: !32 = !DILocation(line: 22, column: 3, scope: !26)
// CHECK:STDOUT: !33 = distinct !DISubprogram(name: "Op", linkageName: "_COp.bad4d5b3ac96d9d9:core.Destroy.Core", scope: null, file: !3, line: 21, type: !20, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !34)
// CHECK:STDOUT: !34 = !{!35}
// CHECK:STDOUT: !35 = !DILocalVariable(arg: 1, scope: !33, type: !22)
// CHECK:STDOUT: !36 = !DILocation(line: 21, column: 3, scope: !33)
// CHECK:STDOUT: !37 = distinct !DISubprogram(name: "Op", linkageName: "_COp.71e25083d63c4d6c:core.Destroy.Core", scope: null, file: !3, line: 20, type: !20, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !38)
// CHECK:STDOUT: !38 = !{!39}
// CHECK:STDOUT: !39 = !DILocalVariable(arg: 1, scope: !37, type: !22)
// CHECK:STDOUT: !40 = !DILocation(line: 20, column: 3, scope: !37)
// CHECK:STDOUT: !41 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.0c4ab795cec6cb27", scope: null, file: !3, line: 13, type: !20, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !42)
// CHECK:STDOUT: !42 = !{!43}
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 1, scope: !41, type: !22)
// CHECK:STDOUT: !44 = !DILocation(line: 13, column: 1, scope: !41)
// CHECK:STDOUT: !45 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.b88d1103f417c6d4", scope: null, file: !3, line: 13, type: !27, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !46)
// CHECK:STDOUT: !46 = !{!47}
// CHECK:STDOUT: !47 = !DILocalVariable(arg: 1, scope: !45, type: !29)
// CHECK:STDOUT: !48 = !DILocation(line: 13, column: 1, scope: !45)
// CHECK:STDOUT: !49 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.d4b5665541d5d7a8", scope: null, file: !3, line: 13, type: !20, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !50)
// CHECK:STDOUT: !50 = !{!51}
// CHECK:STDOUT: !51 = !DILocalVariable(arg: 1, scope: !49, type: !22)
// CHECK:STDOUT: !52 = !DILocation(line: 13, column: 1, scope: !49)
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.5754c7a55c7cbe4a", scope: null, file: !3, line: 13, type: !20, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !54)
// CHECK:STDOUT: !54 = !{!55}
// CHECK:STDOUT: !55 = !DILocalVariable(arg: 1, scope: !53, type: !22)
// CHECK:STDOUT: !56 = !DILocation(line: 13, column: 1, scope: !53)
