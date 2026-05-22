// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/generic/call_method.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/generic/call_method.carbon

class C {
  fn F[unused self: Self, T:! Core.Copy](x: T) -> T {
    return x;
  }
}

fn CallF() -> i32 {
  var c: C = {};
  var n: i32 = 0;
  return c.F(n);
}

// CHECK:STDOUT: ; ModuleID = 'call_method.carbon'
// CHECK:STDOUT: source_filename = "call_method.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @C.val.loc20_3 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CCallF.Main() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca {}, align 8, !dbg !8
// CHECK:STDOUT:   %n.var = alloca i32, align 4, !dbg !9
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !8
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %c.var, ptr align 1 @C.val.loc20_3, i64 0, i1 false), !dbg !8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %n.var), !dbg !9
// CHECK:STDOUT:   store i32 0, ptr %n.var, align 4, !dbg !9
// CHECK:STDOUT:   %.loc22_14 = load i32, ptr %n.var, align 4, !dbg !10
// CHECK:STDOUT:   %C.F.call = call i32 @_CF.C.Main.980bd46fd9fb5b99(ptr %c.var, i32 %.loc22_14), !dbg !11
// CHECK:STDOUT:   call void @"_COp.94e3ca86614aba5d:core.Destroy.Core"(ptr %n.var), !dbg !9
// CHECK:STDOUT:   call void @"_COp.b8b1f433ad216ca6:core.Destroy.Core"(ptr %c.var), !dbg !8
// CHECK:STDOUT:   ret i32 %C.F.call, !dbg !12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.94e3ca86614aba5d:core.Destroy.Core"(ptr %self) #0 !dbg !13 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.b8b1f433ad216ca6:core.Destroy.Core"(ptr %self) #0 !dbg !19 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CF.C.Main.980bd46fd9fb5b99(ptr %self, i32 %x) #0 !dbg !26 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %x, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
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
// CHECK:STDOUT: !3 = !DIFile(filename: "call_method.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "CallF", linkageName: "_CCallF.Main", scope: null, file: !3, line: 19, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7}
// CHECK:STDOUT: !7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !8 = !DILocation(line: 20, column: 3, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 21, column: 3, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 22, column: 14, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 22, column: 10, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 22, column: 3, scope: !4)
// CHECK:STDOUT: !13 = distinct !DISubprogram(name: "Op", linkageName: "_COp.94e3ca86614aba5d:core.Destroy.Core", scope: null, file: !3, line: 21, type: !14, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !16)
// CHECK:STDOUT: !14 = !DISubroutineType(types: !15)
// CHECK:STDOUT: !15 = !{null, !7}
// CHECK:STDOUT: !16 = !{!17}
// CHECK:STDOUT: !17 = !DILocalVariable(arg: 1, scope: !13, type: !7)
// CHECK:STDOUT: !18 = !DILocation(line: 21, column: 3, scope: !13)
// CHECK:STDOUT: !19 = distinct !DISubprogram(name: "Op", linkageName: "_COp.b8b1f433ad216ca6:core.Destroy.Core", scope: null, file: !3, line: 20, type: !20, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !23)
// CHECK:STDOUT: !20 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !21 = !{null, !22}
// CHECK:STDOUT: !22 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !23 = !{!24}
// CHECK:STDOUT: !24 = !DILocalVariable(arg: 1, scope: !19, type: !22)
// CHECK:STDOUT: !25 = !DILocation(line: 20, column: 3, scope: !19)
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "F", linkageName: "_CF.C.Main.980bd46fd9fb5b99", scope: null, file: !3, line: 14, type: !27, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !29)
// CHECK:STDOUT: !27 = !DISubroutineType(types: !28)
// CHECK:STDOUT: !28 = !{!7, !22, !7}
// CHECK:STDOUT: !29 = !{!30, !31}
// CHECK:STDOUT: !30 = !DILocalVariable(arg: 1, scope: !26, type: !22)
// CHECK:STDOUT: !31 = !DILocalVariable(arg: 2, scope: !26, type: !7)
// CHECK:STDOUT: !32 = !DILocation(line: 15, column: 5, scope: !26)
