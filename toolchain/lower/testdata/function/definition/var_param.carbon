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
// CHECK:STDOUT: define void @_COneVar_i32.Main(ptr %_) #0 !dbg !32 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_COneVar_X.Main(ptr %_) #0 !dbg !38 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !42
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTwoVars.Main(ptr %_, ptr %_1) #0 !dbg !44 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !51
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CVarThenLet.Main(ptr %_, ptr %_1) #0 !dbg !53 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !59
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CLetThenVar.Main(i32 %_, ptr %_1) #0 !dbg !61 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCall.Main() #0 !dbg !69 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var.loc15 = alloca i32, align 4, !dbg !70
// CHECK:STDOUT:   %_.var.loc16 = alloca {}, align 1, !dbg !72
// CHECK:STDOUT:   %_.var.loc18_12 = alloca i32, align 4, !dbg !74
// CHECK:STDOUT:   %_.var.loc18_24 = alloca {}, align 1, !dbg !75
// CHECK:STDOUT:   %_.var.loc20 = alloca i32, align 4, !dbg !77
// CHECK:STDOUT:   %.loc29_18.2.temp = alloca {}, align 1, !dbg !78
// CHECK:STDOUT:   %_.var.loc21 = alloca {}, align 1, !dbg !80
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc15), !dbg !70
// CHECK:STDOUT:   store i32 1, ptr %_.var.loc15, align 4, !dbg !70
// CHECK:STDOUT:   call void @_COneVar_i32.Main(ptr %_.var.loc15), !dbg !71
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc16), !dbg !72
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %_.var.loc16, ptr align 1 @X.val.loc16, i64 0, i1 false), !dbg !72
// CHECK:STDOUT:   call void @_COneVar_X.Main(ptr %_.var.loc16), !dbg !73
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc18_12), !dbg !74
// CHECK:STDOUT:   store i32 1, ptr %_.var.loc18_12, align 4, !dbg !74
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc18_24), !dbg !75
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %_.var.loc18_24, ptr align 1 @X.val.loc16, i64 0, i1 false), !dbg !75
// CHECK:STDOUT:   call void @_CTwoVars.Main(ptr %_.var.loc18_12, ptr %_.var.loc18_24), !dbg !76
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc20), !dbg !77
// CHECK:STDOUT:   store i32 1, ptr %_.var.loc20, align 4, !dbg !77
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc29_18.2.temp), !dbg !78
// CHECK:STDOUT:   call void @_CVarThenLet.Main(ptr %_.var.loc20, ptr @X.val.loc16), !dbg !79
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc21), !dbg !80
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %_.var.loc21, ptr align 1 @X.val.loc16, i64 0, i1 false), !dbg !80
// CHECK:STDOUT:   call void @_CLetThenVar.Main(i32 1, ptr %_.var.loc21), !dbg !81
// CHECK:STDOUT:   ret void, !dbg !82
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 6, 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!21}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIFile(filename: "var_param.carbon", directory: "")
// CHECK:STDOUT: !21 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !20, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !22 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !23 = !{null, !22}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !25 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !26 = !{null, !25}
// CHECK:STDOUT: !27 = !DISubroutineType(types: !26)
// CHECK:STDOUT: !28 = !{null, !22, !25}
// CHECK:STDOUT: !29 = !DISubroutineType(types: !28)
// CHECK:STDOUT: !30 = !{null}
// CHECK:STDOUT: !31 = !DISubroutineType(types: !30)
// CHECK:STDOUT: !32 = distinct !DISubprogram(name: "OneVar_i32", linkageName: "_COneVar_i32.Main", scope: null, file: !20, line: 15, type: !24, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !37)
// CHECK:STDOUT: !33 = !DILocalVariable(arg: 1, scope: !32, type: !22)
// CHECK:STDOUT: !36 = !DILocation(line: 15, column: 1, scope: !32)
// CHECK:STDOUT: !37 = !{!33}
// CHECK:STDOUT: !38 = distinct !DISubprogram(name: "OneVar_X", linkageName: "_COneVar_X.Main", scope: null, file: !20, line: 16, type: !27, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !43)
// CHECK:STDOUT: !39 = !DILocalVariable(arg: 1, scope: !38, type: !25)
// CHECK:STDOUT: !42 = !DILocation(line: 16, column: 1, scope: !38)
// CHECK:STDOUT: !43 = !{!39}
// CHECK:STDOUT: !44 = distinct !DISubprogram(name: "TwoVars", linkageName: "_CTwoVars.Main", scope: null, file: !20, line: 18, type: !29, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !52)
// CHECK:STDOUT: !45 = !DILocalVariable(arg: 1, scope: !44, type: !22)
// CHECK:STDOUT: !46 = !DILocalVariable(arg: 2, scope: !44, type: !25)
// CHECK:STDOUT: !51 = !DILocation(line: 18, column: 1, scope: !44)
// CHECK:STDOUT: !52 = !{!45, !46}
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "VarThenLet", linkageName: "_CVarThenLet.Main", scope: null, file: !20, line: 20, type: !29, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !60)
// CHECK:STDOUT: !54 = !DILocalVariable(arg: 1, scope: !53, type: !22)
// CHECK:STDOUT: !55 = !DILocalVariable(arg: 2, scope: !53, type: !25)
// CHECK:STDOUT: !59 = !DILocation(line: 20, column: 1, scope: !53)
// CHECK:STDOUT: !60 = !{!54, !55}
// CHECK:STDOUT: !61 = distinct !DISubprogram(name: "LetThenVar", linkageName: "_CLetThenVar.Main", scope: null, file: !20, line: 21, type: !29, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !68)
// CHECK:STDOUT: !62 = !DILocalVariable(arg: 1, scope: !61, type: !22)
// CHECK:STDOUT: !63 = !DILocalVariable(arg: 2, scope: !61, type: !25)
// CHECK:STDOUT: !67 = !DILocation(line: 21, column: 1, scope: !61)
// CHECK:STDOUT: !68 = !{!62, !63}
// CHECK:STDOUT: !69 = distinct !DISubprogram(name: "Call", linkageName: "_CCall.Main", scope: null, file: !20, line: 23, type: !31, spFlags: DISPFlagDefinition, unit: !21)
// CHECK:STDOUT: !70 = !DILocation(line: 15, column: 15, scope: !69)
// CHECK:STDOUT: !71 = !DILocation(line: 24, column: 3, scope: !69)
// CHECK:STDOUT: !72 = !DILocation(line: 16, column: 13, scope: !69)
// CHECK:STDOUT: !73 = !DILocation(line: 25, column: 3, scope: !69)
// CHECK:STDOUT: !74 = !DILocation(line: 18, column: 12, scope: !69)
// CHECK:STDOUT: !75 = !DILocation(line: 18, column: 24, scope: !69)
// CHECK:STDOUT: !76 = !DILocation(line: 26, column: 3, scope: !69)
// CHECK:STDOUT: !77 = !DILocation(line: 20, column: 15, scope: !69)
// CHECK:STDOUT: !78 = !DILocation(line: 29, column: 17, scope: !69)
// CHECK:STDOUT: !79 = !DILocation(line: 29, column: 3, scope: !69)
// CHECK:STDOUT: !80 = !DILocation(line: 21, column: 23, scope: !69)
// CHECK:STDOUT: !81 = !DILocation(line: 30, column: 3, scope: !69)
// CHECK:STDOUT: !82 = !DILocation(line: 23, column: 1, scope: !69)
// CHECK:STDOUT:
