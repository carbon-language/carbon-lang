// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/definition/var_param.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/definition/var_param.carbon

class X {}

fn OneVar_i32(var _: i32) {}
fn OneVar_X(var _: X) {}

fn TwoVars(var _: i32, var _: X) {}

fn VarThenLet(var _: i32, _: X) {}
fn LetThenVar(_: i32, var _: X) {}

fn Call() {
  OneVar_i32(1);
  OneVar_X({});
  TwoVars(1, {});
  VarThenLet(1, {});
  LetThenVar(1, {});
}

// CHECK:STDOUT: ; ModuleID = 'var_param.carbon'
// CHECK:STDOUT: source_filename = "var_param.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @X.val = internal constant {} zeroinitializer
// CHECK:STDOUT: @X.val.loc16 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_COneVar_i32.Main(ptr %_) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_COneVar_X.Main(ptr %_) #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTwoVars.Main(ptr %_, ptr %_1) #0 !dbg !18 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !24
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CVarThenLet.Main(ptr %_, ptr %_1) #0 !dbg !25 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CLetThenVar.Main(i32 %_, ptr %_1) #0 !dbg !30 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCall.Main() #0 !dbg !35 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var.loc15 = alloca i32, align 4, !dbg !38
// CHECK:STDOUT:   %_.var.loc16 = alloca {}, align 1, !dbg !39
// CHECK:STDOUT:   %_.var.loc18_12 = alloca i32, align 4, !dbg !40
// CHECK:STDOUT:   %_.var.loc18_24 = alloca {}, align 1, !dbg !41
// CHECK:STDOUT:   %_.var.loc20 = alloca i32, align 4, !dbg !42
// CHECK:STDOUT:   %.loc27_18.2.temp = alloca {}, align 1, !dbg !43
// CHECK:STDOUT:   %_.var.loc21 = alloca {}, align 1, !dbg !44
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc15), !dbg !38
// CHECK:STDOUT:   store i32 1, ptr %_.var.loc15, align 4, !dbg !38
// CHECK:STDOUT:   call void @_COneVar_i32.Main(ptr %_.var.loc15), !dbg !45
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc16), !dbg !39
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %_.var.loc16, ptr align 1 @X.val.loc16, i64 0, i1 false), !dbg !39
// CHECK:STDOUT:   call void @_COneVar_X.Main(ptr %_.var.loc16), !dbg !46
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc18_12), !dbg !40
// CHECK:STDOUT:   store i32 1, ptr %_.var.loc18_12, align 4, !dbg !40
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc18_24), !dbg !41
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %_.var.loc18_24, ptr align 1 @X.val.loc16, i64 0, i1 false), !dbg !41
// CHECK:STDOUT:   call void @_CTwoVars.Main(ptr %_.var.loc18_12, ptr %_.var.loc18_24), !dbg !47
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc20), !dbg !42
// CHECK:STDOUT:   store i32 1, ptr %_.var.loc20, align 4, !dbg !42
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc27_18.2.temp), !dbg !43
// CHECK:STDOUT:   call void @_CVarThenLet.Main(ptr %_.var.loc20, ptr @X.val.loc16), !dbg !48
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc21), !dbg !44
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %_.var.loc21, ptr align 1 @X.val.loc16, i64 0, i1 false), !dbg !44
// CHECK:STDOUT:   call void @_CLetThenVar.Main(i32 1, ptr %_.var.loc21), !dbg !49
// CHECK:STDOUT:   call void @"_COp.429fc69ea0bf165d:core.Destroy.Core"(ptr %_.var.loc21), !dbg !44
// CHECK:STDOUT:   call void @"_COp.429fc69ea0bf165d:core.Destroy.Core"(ptr @X.val), !dbg !43
// CHECK:STDOUT:   call void @"_COp.5e27612b9dd31a14:core.Destroy.Core"(ptr %_.var.loc20), !dbg !42
// CHECK:STDOUT:   call void @"_COp.429fc69ea0bf165d:core.Destroy.Core"(ptr %_.var.loc18_24), !dbg !41
// CHECK:STDOUT:   call void @"_COp.5e27612b9dd31a14:core.Destroy.Core"(ptr %_.var.loc18_12), !dbg !40
// CHECK:STDOUT:   call void @"_COp.429fc69ea0bf165d:core.Destroy.Core"(ptr %_.var.loc16), !dbg !39
// CHECK:STDOUT:   call void @"_COp.5e27612b9dd31a14:core.Destroy.Core"(ptr %_.var.loc15), !dbg !38
// CHECK:STDOUT:   ret void, !dbg !50
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.429fc69ea0bf165d:core.Destroy.Core"(ptr %self) #0 !dbg !51 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !54
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.5e27612b9dd31a14:core.Destroy.Core"(ptr %self) #0 !dbg !55 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !58
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @"_COp.429fc69ea0bf165d:core.Destroy.Core", { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @"_COp.5e27612b9dd31a14:core.Destroy.Core", { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 6, 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 2, 1, 0 }
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
// CHECK:STDOUT: !3 = !DIFile(filename: "var_param.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "OneVar_i32", linkageName: "_COneVar_i32.Main", scope: null, file: !3, line: 15, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !8)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null, !7}
// CHECK:STDOUT: !7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !8 = !{!9}
// CHECK:STDOUT: !9 = !DILocalVariable(arg: 1, scope: !4, type: !7)
// CHECK:STDOUT: !10 = !DILocation(line: 15, column: 1, scope: !4)
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "OneVar_X", linkageName: "_COneVar_X.Main", scope: null, file: !3, line: 16, type: !12, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !15)
// CHECK:STDOUT: !12 = !DISubroutineType(types: !13)
// CHECK:STDOUT: !13 = !{null, !14}
// CHECK:STDOUT: !14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !15 = !{!16}
// CHECK:STDOUT: !16 = !DILocalVariable(arg: 1, scope: !11, type: !14)
// CHECK:STDOUT: !17 = !DILocation(line: 16, column: 1, scope: !11)
// CHECK:STDOUT: !18 = distinct !DISubprogram(name: "TwoVars", linkageName: "_CTwoVars.Main", scope: null, file: !3, line: 18, type: !19, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !21)
// CHECK:STDOUT: !19 = !DISubroutineType(types: !20)
// CHECK:STDOUT: !20 = !{null, !7, !14}
// CHECK:STDOUT: !21 = !{!22, !23}
// CHECK:STDOUT: !22 = !DILocalVariable(arg: 1, scope: !18, type: !7)
// CHECK:STDOUT: !23 = !DILocalVariable(arg: 2, scope: !18, type: !14)
// CHECK:STDOUT: !24 = !DILocation(line: 18, column: 1, scope: !18)
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "VarThenLet", linkageName: "_CVarThenLet.Main", scope: null, file: !3, line: 20, type: !19, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !26)
// CHECK:STDOUT: !26 = !{!27, !28}
// CHECK:STDOUT: !27 = !DILocalVariable(arg: 1, scope: !25, type: !7)
// CHECK:STDOUT: !28 = !DILocalVariable(arg: 2, scope: !25, type: !14)
// CHECK:STDOUT: !29 = !DILocation(line: 20, column: 1, scope: !25)
// CHECK:STDOUT: !30 = distinct !DISubprogram(name: "LetThenVar", linkageName: "_CLetThenVar.Main", scope: null, file: !3, line: 21, type: !19, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !31)
// CHECK:STDOUT: !31 = !{!32, !33}
// CHECK:STDOUT: !32 = !DILocalVariable(arg: 1, scope: !30, type: !7)
// CHECK:STDOUT: !33 = !DILocalVariable(arg: 2, scope: !30, type: !14)
// CHECK:STDOUT: !34 = !DILocation(line: 21, column: 1, scope: !30)
// CHECK:STDOUT: !35 = distinct !DISubprogram(name: "Call", linkageName: "_CCall.Main", scope: null, file: !3, line: 23, type: !36, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !36 = !DISubroutineType(types: !37)
// CHECK:STDOUT: !37 = !{null}
// CHECK:STDOUT: !38 = !DILocation(line: 15, column: 15, scope: !35)
// CHECK:STDOUT: !39 = !DILocation(line: 16, column: 13, scope: !35)
// CHECK:STDOUT: !40 = !DILocation(line: 18, column: 12, scope: !35)
// CHECK:STDOUT: !41 = !DILocation(line: 18, column: 24, scope: !35)
// CHECK:STDOUT: !42 = !DILocation(line: 20, column: 15, scope: !35)
// CHECK:STDOUT: !43 = !DILocation(line: 27, column: 17, scope: !35)
// CHECK:STDOUT: !44 = !DILocation(line: 21, column: 23, scope: !35)
// CHECK:STDOUT: !45 = !DILocation(line: 24, column: 3, scope: !35)
// CHECK:STDOUT: !46 = !DILocation(line: 25, column: 3, scope: !35)
// CHECK:STDOUT: !47 = !DILocation(line: 26, column: 3, scope: !35)
// CHECK:STDOUT: !48 = !DILocation(line: 27, column: 3, scope: !35)
// CHECK:STDOUT: !49 = !DILocation(line: 28, column: 3, scope: !35)
// CHECK:STDOUT: !50 = !DILocation(line: 23, column: 1, scope: !35)
// CHECK:STDOUT: !51 = distinct !DISubprogram(name: "Op", linkageName: "_COp.429fc69ea0bf165d:core.Destroy.Core", scope: null, file: !3, line: 21, type: !12, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !52)
// CHECK:STDOUT: !52 = !{!53}
// CHECK:STDOUT: !53 = !DILocalVariable(arg: 1, scope: !51, type: !14)
// CHECK:STDOUT: !54 = !DILocation(line: 21, column: 23, scope: !51)
// CHECK:STDOUT: !55 = distinct !DISubprogram(name: "Op", linkageName: "_COp.5e27612b9dd31a14:core.Destroy.Core", scope: null, file: !3, line: 20, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !56)
// CHECK:STDOUT: !56 = !{!57}
// CHECK:STDOUT: !57 = !DILocalVariable(arg: 1, scope: !55, type: !7)
// CHECK:STDOUT: !58 = !DILocation(line: 20, column: 15, scope: !55)
