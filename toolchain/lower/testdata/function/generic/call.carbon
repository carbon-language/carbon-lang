// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/generic/call.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/generic/call.carbon

fn F[T:! type](x: T) {
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
// CHECK:STDOUT: @C.val.loc18_3.2 = internal constant {} zeroinitializer
// CHECK:STDOUT: @D.val.loc19_3.2 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CG.Main() !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca {}, align 8, !dbg !7
// CHECK:STDOUT:   %d.var = alloca {}, align 8, !dbg !7
// CHECK:STDOUT:   %n.var = alloca i32, align 4, !dbg !7
// CHECK:STDOUT:   %m.var = alloca double, align 8, !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 0, ptr %c.var), !dbg !7
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %c.var, ptr align 1 @C.val.loc18_3.2, i64 0, i1 false), !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 0, ptr %d.var), !dbg !7
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %d.var, ptr align 1 @D.val.loc19_3.2, i64 0, i1 false), !dbg !8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 4, ptr %n.var), !dbg !7
// CHECK:STDOUT:   store i32 0, ptr %n.var, align 4, !dbg !9
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 8, ptr %m.var), !dbg !7
// CHECK:STDOUT:   store double 0.000000e+00, ptr %m.var, align 8, !dbg !10
// CHECK:STDOUT:   call void @_CF.Main.15b1f98bd9cc0c5b(ptr %c.var), !dbg !11
// CHECK:STDOUT:   call void @_CF.Main.2cc450fc05045897(ptr %d.var), !dbg !12
// CHECK:STDOUT:   %.loc25 = load i32, ptr %n.var, align 4, !dbg !13
// CHECK:STDOUT:   call void @_CF.Main.b88d1103f417c6d4(i32 %.loc25), !dbg !14
// CHECK:STDOUT:   %.loc26 = load double, ptr %m.var, align 8, !dbg !15
// CHECK:STDOUT:   call void @_CF.Main.66be507887ceee78(double %.loc26), !dbg !16
// CHECK:STDOUT:   call void @_CF.Main.5754c7a55c7cbe4a(%type zeroinitializer), !dbg !17
// CHECK:STDOUT:   ret void, !dbg !18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(i64 immarg, ptr captures(none)) #0
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CF.Main.15b1f98bd9cc0c5b(ptr %x) !dbg !19 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CF.Main.2cc450fc05045897(ptr %x) !dbg !21 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CF.Main.b88d1103f417c6d4(i32 %x) !dbg !23 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !24
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CF.Main.66be507887ceee78(double %x) !dbg !25 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CF.Main.5754c7a55c7cbe4a(%type %x) !dbg !27 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "call.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !3, line: 17, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 18, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 19, column: 3, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 20, column: 3, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 21, column: 3, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 23, column: 3, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 24, column: 3, scope: !4)
// CHECK:STDOUT: !13 = !DILocation(line: 25, column: 5, scope: !4)
// CHECK:STDOUT: !14 = !DILocation(line: 25, column: 3, scope: !4)
// CHECK:STDOUT: !15 = !DILocation(line: 26, column: 5, scope: !4)
// CHECK:STDOUT: !16 = !DILocation(line: 26, column: 3, scope: !4)
// CHECK:STDOUT: !17 = !DILocation(line: 27, column: 3, scope: !4)
// CHECK:STDOUT: !18 = !DILocation(line: 17, column: 1, scope: !4)
// CHECK:STDOUT: !19 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.15b1f98bd9cc0c5b", scope: null, file: !3, line: 11, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !20 = !DILocation(line: 11, column: 1, scope: !19)
// CHECK:STDOUT: !21 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.2cc450fc05045897", scope: null, file: !3, line: 11, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !22 = !DILocation(line: 11, column: 1, scope: !21)
// CHECK:STDOUT: !23 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.b88d1103f417c6d4", scope: null, file: !3, line: 11, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !24 = !DILocation(line: 11, column: 1, scope: !23)
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.66be507887ceee78", scope: null, file: !3, line: 11, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !26 = !DILocation(line: 11, column: 1, scope: !25)
// CHECK:STDOUT: !27 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.5754c7a55c7cbe4a", scope: null, file: !3, line: 11, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !28 = !DILocation(line: 11, column: 1, scope: !27)
