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
  // TODO: This passes a global as the `X` parameter, then destroys a different
  // `X` global. We shouldn't be calling Destroy on a global.
  VarThenLet(1, {});
  LetThenVar(1, {});
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'var_param.carbon'
// CHECK:STDOUT: source_filename = "var_param.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @X.val = internal constant {} zeroinitializer
// CHECK:STDOUT: @X.val.loc16 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_COneVar_i32.Main(ptr %_) #0 !dbg !30 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_COneVar_X.Main(ptr %_) #0 !dbg !36 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTwoVars.Main(ptr %_, ptr %_1) #0 !dbg !42 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !49
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CVarThenLet.Main(ptr %_, ptr %_1) #0 !dbg !51 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !57
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CLetThenVar.Main(i32 %_, ptr %_1) #0 !dbg !59 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !65
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCall.Main() #0 !dbg !67 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var.loc15 = alloca i32, align 4, !dbg !68
// CHECK:STDOUT:   %_.var.loc16 = alloca {}, align 1, !dbg !70
// CHECK:STDOUT:   %_.var.loc18_12 = alloca i32, align 4, !dbg !72
// CHECK:STDOUT:   %_.var.loc18_24 = alloca {}, align 1, !dbg !73
// CHECK:STDOUT:   %_.var.loc20 = alloca i32, align 4, !dbg !75
// CHECK:STDOUT:   %.loc29_18.2.temp = alloca {}, align 1, !dbg !76
// CHECK:STDOUT:   %_.var.loc21 = alloca {}, align 1, !dbg !78
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc15), !dbg !68
// CHECK:STDOUT:   store i32 1, ptr %_.var.loc15, align 4, !dbg !68
// CHECK:STDOUT:   call void @_COneVar_i32.Main(ptr %_.var.loc15), !dbg !69
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %_.var.loc15), !dbg !68
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc16), !dbg !70
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %_.var.loc16, ptr align 1 @X.val.loc16, i64 0, i1 false), !dbg !70
// CHECK:STDOUT:   call void @_COneVar_X.Main(ptr %_.var.loc16), !dbg !71
// CHECK:STDOUT:   call void @"_CSelfDestruct.8b0bf151036f045a:core.Destroy.Core"(ptr %_.var.loc16), !dbg !70
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc18_12), !dbg !72
// CHECK:STDOUT:   store i32 1, ptr %_.var.loc18_12, align 4, !dbg !72
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc18_24), !dbg !73
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %_.var.loc18_24, ptr align 1 @X.val.loc16, i64 0, i1 false), !dbg !73
// CHECK:STDOUT:   call void @_CTwoVars.Main(ptr %_.var.loc18_12, ptr %_.var.loc18_24), !dbg !74
// CHECK:STDOUT:   call void @"_CSelfDestruct.8b0bf151036f045a:core.Destroy.Core"(ptr %_.var.loc18_24), !dbg !73
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %_.var.loc18_12), !dbg !72
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc20), !dbg !75
// CHECK:STDOUT:   store i32 1, ptr %_.var.loc20, align 4, !dbg !75
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc29_18.2.temp), !dbg !76
// CHECK:STDOUT:   call void @_CVarThenLet.Main(ptr %_.var.loc20, ptr @X.val.loc16), !dbg !77
// CHECK:STDOUT:   call void @"_CSelfDestruct.8b0bf151036f045a:core.Destroy.Core"(ptr @X.val), !dbg !76
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %_.var.loc20), !dbg !75
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc21), !dbg !78
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %_.var.loc21, ptr align 1 @X.val.loc16, i64 0, i1 false), !dbg !78
// CHECK:STDOUT:   call void @_CLetThenVar.Main(i32 1, ptr %_.var.loc21), !dbg !79
// CHECK:STDOUT:   call void @"_CSelfDestruct.8b0bf151036f045a:core.Destroy.Core"(ptr %_.var.loc21), !dbg !78
// CHECK:STDOUT:   ret void, !dbg !80
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.6dae29ece9a708c4:core.Destroy.Core"(ptr %self) #0 !dbg !81 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !83
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !85 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !87
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !89 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self), !dbg !91
// CHECK:STDOUT:   ret void, !dbg !91
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.cb8adced19392b99:core.Destroy.Core"(ptr %self) #0 !dbg !93 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !95
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.8b0bf151036f045a:core.Destroy.Core"(ptr %self) #0 !dbg !97 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !99
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.8b0bf151036f045a:core.Destroy.Core"(ptr %self) #0 !dbg !101 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.8b0bf151036f045a:core.Destroy.Core"(ptr %self), !dbg !103
// CHECK:STDOUT:   ret void, !dbg !103
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core", { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @"_CSelfDestruct.8b0bf151036f045a:core.Destroy.Core", { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 6, 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !18 = !DIFile(filename: "var_param.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !{null, !20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{null, !23}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = !{null, !20, !23}
// CHECK:STDOUT: !27 = !DISubroutineType(types: !26)
// CHECK:STDOUT: !28 = !{null}
// CHECK:STDOUT: !29 = !DISubroutineType(types: !28)
// CHECK:STDOUT: !30 = distinct !DISubprogram(name: "OneVar_i32", linkageName: "_COneVar_i32.Main", scope: null, file: !18, line: 15, type: !22, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !35)
// CHECK:STDOUT: !31 = !DILocalVariable(arg: 1, scope: !30, type: !20)
// CHECK:STDOUT: !34 = !DILocation(line: 15, column: 1, scope: !30)
// CHECK:STDOUT: !35 = !{!31}
// CHECK:STDOUT: !36 = distinct !DISubprogram(name: "OneVar_X", linkageName: "_COneVar_X.Main", scope: null, file: !18, line: 16, type: !25, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !41)
// CHECK:STDOUT: !37 = !DILocalVariable(arg: 1, scope: !36, type: !23)
// CHECK:STDOUT: !40 = !DILocation(line: 16, column: 1, scope: !36)
// CHECK:STDOUT: !41 = !{!37}
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "TwoVars", linkageName: "_CTwoVars.Main", scope: null, file: !18, line: 18, type: !27, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !50)
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 1, scope: !42, type: !20)
// CHECK:STDOUT: !44 = !DILocalVariable(arg: 2, scope: !42, type: !23)
// CHECK:STDOUT: !49 = !DILocation(line: 18, column: 1, scope: !42)
// CHECK:STDOUT: !50 = !{!43, !44}
// CHECK:STDOUT: !51 = distinct !DISubprogram(name: "VarThenLet", linkageName: "_CVarThenLet.Main", scope: null, file: !18, line: 20, type: !27, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !58)
// CHECK:STDOUT: !52 = !DILocalVariable(arg: 1, scope: !51, type: !20)
// CHECK:STDOUT: !53 = !DILocalVariable(arg: 2, scope: !51, type: !23)
// CHECK:STDOUT: !57 = !DILocation(line: 20, column: 1, scope: !51)
// CHECK:STDOUT: !58 = !{!52, !53}
// CHECK:STDOUT: !59 = distinct !DISubprogram(name: "LetThenVar", linkageName: "_CLetThenVar.Main", scope: null, file: !18, line: 21, type: !27, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !66)
// CHECK:STDOUT: !60 = !DILocalVariable(arg: 1, scope: !59, type: !20)
// CHECK:STDOUT: !61 = !DILocalVariable(arg: 2, scope: !59, type: !23)
// CHECK:STDOUT: !65 = !DILocation(line: 21, column: 1, scope: !59)
// CHECK:STDOUT: !66 = !{!60, !61}
// CHECK:STDOUT: !67 = distinct !DISubprogram(name: "Call", linkageName: "_CCall.Main", scope: null, file: !18, line: 23, type: !29, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !68 = !DILocation(line: 15, column: 15, scope: !67)
// CHECK:STDOUT: !69 = !DILocation(line: 24, column: 3, scope: !67)
// CHECK:STDOUT: !70 = !DILocation(line: 16, column: 13, scope: !67)
// CHECK:STDOUT: !71 = !DILocation(line: 25, column: 3, scope: !67)
// CHECK:STDOUT: !72 = !DILocation(line: 18, column: 12, scope: !67)
// CHECK:STDOUT: !73 = !DILocation(line: 18, column: 24, scope: !67)
// CHECK:STDOUT: !74 = !DILocation(line: 26, column: 3, scope: !67)
// CHECK:STDOUT: !75 = !DILocation(line: 20, column: 15, scope: !67)
// CHECK:STDOUT: !76 = !DILocation(line: 29, column: 17, scope: !67)
// CHECK:STDOUT: !77 = !DILocation(line: 29, column: 3, scope: !67)
// CHECK:STDOUT: !78 = !DILocation(line: 21, column: 23, scope: !67)
// CHECK:STDOUT: !79 = !DILocation(line: 30, column: 3, scope: !67)
// CHECK:STDOUT: !80 = !DILocation(line: 23, column: 1, scope: !67)
// CHECK:STDOUT: !81 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.6dae29ece9a708c4:core.Destroy.Core", scope: null, file: !18, line: 15, type: !22, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !84)
// CHECK:STDOUT: !82 = !DILocalVariable(arg: 1, scope: !81, type: !20)
// CHECK:STDOUT: !83 = !DILocation(line: 15, column: 15, scope: !81)
// CHECK:STDOUT: !84 = !{!82}
// CHECK:STDOUT: !85 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core", scope: null, file: !18, line: 15, type: !22, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !88)
// CHECK:STDOUT: !86 = !DILocalVariable(arg: 1, scope: !85, type: !20)
// CHECK:STDOUT: !87 = !DILocation(line: 15, column: 15, scope: !85)
// CHECK:STDOUT: !88 = !{!86}
// CHECK:STDOUT: !89 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f58524831c37807e:core.Destroy.Core", scope: null, file: !18, line: 15, type: !22, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !92)
// CHECK:STDOUT: !90 = !DILocalVariable(arg: 1, scope: !89, type: !20)
// CHECK:STDOUT: !91 = !DILocation(line: 15, column: 15, scope: !89)
// CHECK:STDOUT: !92 = !{!90}
// CHECK:STDOUT: !93 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.cb8adced19392b99:core.Destroy.Core", scope: null, file: !18, line: 16, type: !25, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !96)
// CHECK:STDOUT: !94 = !DILocalVariable(arg: 1, scope: !93, type: !23)
// CHECK:STDOUT: !95 = !DILocation(line: 16, column: 13, scope: !93)
// CHECK:STDOUT: !96 = !{!94}
// CHECK:STDOUT: !97 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.8b0bf151036f045a:core.Destroy.Core", scope: null, file: !18, line: 16, type: !25, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !100)
// CHECK:STDOUT: !98 = !DILocalVariable(arg: 1, scope: !97, type: !23)
// CHECK:STDOUT: !99 = !DILocation(line: 16, column: 13, scope: !97)
// CHECK:STDOUT: !100 = !{!98}
// CHECK:STDOUT: !101 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.8b0bf151036f045a:core.Destroy.Core", scope: null, file: !18, line: 16, type: !25, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !104)
// CHECK:STDOUT: !102 = !DILocalVariable(arg: 1, scope: !101, type: !23)
// CHECK:STDOUT: !103 = !DILocation(line: 16, column: 13, scope: !101)
// CHECK:STDOUT: !104 = !{!102}
// CHECK:STDOUT:
