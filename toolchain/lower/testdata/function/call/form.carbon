// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/call/form.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/call/form.carbon

// --- ref_form_param.carbon
library "[[@TEST_NAME]]";

fn F(x:? form(ref i32));

fn G() {
  var y: i32 = 0;
  F(ref y);
}

// --- var_form_param.carbon
library "[[@TEST_NAME]]";

fn F(x:? form(var i32));

fn G() {
  F(0);
}

// --- val_form_param.carbon
library "[[@TEST_NAME]]";

fn F(x:? form(val i32));

fn G() {
  F(0);
}

// --- ref_return_form.carbon
library "[[@TEST_NAME]]";

fn F() ->? form(ref i32);

fn G() {
  let unused ref x: i32 = F();
}

// --- var_return_form.carbon
library "[[@TEST_NAME]]";

fn F() ->? form(var i32);

fn G() {
  let unused var x: i32 = F();
}

// --- val_return_form.carbon
library "[[@TEST_NAME]]";

fn F() ->? form(val i32);

fn G() {
  let unused x: i32 = F();
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'ref_form_param.carbon'
// CHECK:STDOUT: source_filename = "ref_form_param.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CF.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !25 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %y.var = alloca i32, align 4, !dbg !26
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %y.var), !dbg !26
// CHECK:STDOUT:   store i32 0, ptr %y.var, align 4, !dbg !26
// CHECK:STDOUT:   call void @_CF.Main(ptr %y.var), !dbg !30
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %y.var), !dbg !26
// CHECK:STDOUT:   ret void, !dbg !31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.6dae29ece9a708c4:core.Destroy.Core"(ptr %self) #0 !dbg !32 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !36 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !40 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self), !dbg !42
// CHECK:STDOUT:   ret void, !dbg !42
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !18 = !DIFile(filename: "ref_form_param.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !{null, !20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = !{null}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !18, line: 5, type: !24, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !26 = !DILocation(line: 6, column: 3, scope: !25)
// CHECK:STDOUT: !30 = !DILocation(line: 7, column: 3, scope: !25)
// CHECK:STDOUT: !31 = !DILocation(line: 5, column: 1, scope: !25)
// CHECK:STDOUT: !32 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.6dae29ece9a708c4:core.Destroy.Core", scope: null, file: !18, line: 6, type: !22, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !35)
// CHECK:STDOUT: !33 = !DILocalVariable(arg: 1, scope: !32, type: !20)
// CHECK:STDOUT: !34 = !DILocation(line: 6, column: 3, scope: !32)
// CHECK:STDOUT: !35 = !{!33}
// CHECK:STDOUT: !36 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core", scope: null, file: !18, line: 6, type: !22, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !39)
// CHECK:STDOUT: !37 = !DILocalVariable(arg: 1, scope: !36, type: !20)
// CHECK:STDOUT: !38 = !DILocation(line: 6, column: 3, scope: !36)
// CHECK:STDOUT: !39 = !{!37}
// CHECK:STDOUT: !40 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f58524831c37807e:core.Destroy.Core", scope: null, file: !18, line: 6, type: !22, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !43)
// CHECK:STDOUT: !41 = !DILocalVariable(arg: 1, scope: !40, type: !20)
// CHECK:STDOUT: !42 = !DILocation(line: 6, column: 3, scope: !40)
// CHECK:STDOUT: !43 = !{!41}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'var_form_param.carbon'
// CHECK:STDOUT: source_filename = "var_form_param.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CF.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !46 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x.var = alloca i32, align 4, !dbg !47
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %x.var), !dbg !47
// CHECK:STDOUT:   store i32 0, ptr %x.var, align 4, !dbg !47
// CHECK:STDOUT:   call void @_CF.Main(ptr %x.var), !dbg !48
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %x.var), !dbg !47
// CHECK:STDOUT:   ret void, !dbg !49
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.6dae29ece9a708c4:core.Destroy.Core"(ptr %self) #0 !dbg !50 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !52
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !54 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !56
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !58 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self), !dbg !60
// CHECK:STDOUT:   ret void, !dbg !60
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!45}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !{null, !20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = !{null}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !44 = !DIFile(filename: "var_form_param.carbon", directory: "")
// CHECK:STDOUT: !45 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !44, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !46 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !44, line: 5, type: !24, spFlags: DISPFlagDefinition, unit: !45)
// CHECK:STDOUT: !47 = !DILocation(line: 3, column: 6, scope: !46)
// CHECK:STDOUT: !48 = !DILocation(line: 6, column: 3, scope: !46)
// CHECK:STDOUT: !49 = !DILocation(line: 5, column: 1, scope: !46)
// CHECK:STDOUT: !50 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.6dae29ece9a708c4:core.Destroy.Core", scope: null, file: !44, line: 3, type: !22, spFlags: DISPFlagDefinition, unit: !45, retainedNodes: !53)
// CHECK:STDOUT: !51 = !DILocalVariable(arg: 1, scope: !50, type: !20)
// CHECK:STDOUT: !52 = !DILocation(line: 3, column: 6, scope: !50)
// CHECK:STDOUT: !53 = !{!51}
// CHECK:STDOUT: !54 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core", scope: null, file: !44, line: 3, type: !22, spFlags: DISPFlagDefinition, unit: !45, retainedNodes: !57)
// CHECK:STDOUT: !55 = !DILocalVariable(arg: 1, scope: !54, type: !20)
// CHECK:STDOUT: !56 = !DILocation(line: 3, column: 6, scope: !54)
// CHECK:STDOUT: !57 = !{!55}
// CHECK:STDOUT: !58 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f58524831c37807e:core.Destroy.Core", scope: null, file: !44, line: 3, type: !22, spFlags: DISPFlagDefinition, unit: !45, retainedNodes: !61)
// CHECK:STDOUT: !59 = !DILocalVariable(arg: 1, scope: !58, type: !20)
// CHECK:STDOUT: !60 = !DILocation(line: 3, column: 6, scope: !58)
// CHECK:STDOUT: !61 = !{!59}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'val_form_param.carbon'
// CHECK:STDOUT: source_filename = "val_form_param.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CF.Main(i32)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !64 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main(i32 0), !dbg !65
// CHECK:STDOUT:   ret void, !dbg !66
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!63}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !23 = !{null}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !62 = !DIFile(filename: "val_form_param.carbon", directory: "")
// CHECK:STDOUT: !63 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !62, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !64 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !62, line: 5, type: !24, spFlags: DISPFlagDefinition, unit: !63)
// CHECK:STDOUT: !65 = !DILocation(line: 6, column: 3, scope: !64)
// CHECK:STDOUT: !66 = !DILocation(line: 5, column: 1, scope: !64)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'ref_return_form.carbon'
// CHECK:STDOUT: source_filename = "ref_return_form.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare ptr @_CF.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !72 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %F.call = call ptr @_CF.Main(), !dbg !73
// CHECK:STDOUT:   ret void, !dbg !76
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!68}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !23 = !{null}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !67 = !DIFile(filename: "ref_return_form.carbon", directory: "")
// CHECK:STDOUT: !68 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !67, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !72 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !67, line: 5, type: !24, spFlags: DISPFlagDefinition, unit: !68)
// CHECK:STDOUT: !73 = !DILocation(line: 6, column: 27, scope: !72)
// CHECK:STDOUT: !76 = !DILocation(line: 5, column: 1, scope: !72)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'var_return_form.carbon'
// CHECK:STDOUT: source_filename = "var_return_form.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @_CF.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !81 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x.var = alloca i32, align 4, !dbg !82
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %x.var), !dbg !82
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main(), !dbg !83
// CHECK:STDOUT:   store i32 %F.call, ptr %x.var, align 4, !dbg !82
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %x.var), !dbg !82
// CHECK:STDOUT:   ret void, !dbg !86
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.6dae29ece9a708c4:core.Destroy.Core"(ptr %self) #0 !dbg !87 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !89
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !91 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !93
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !95 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self), !dbg !97
// CHECK:STDOUT:   ret void, !dbg !97
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!78}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !{null, !20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = !{null}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !77 = !DIFile(filename: "var_return_form.carbon", directory: "")
// CHECK:STDOUT: !78 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !77, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !81 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !77, line: 5, type: !24, spFlags: DISPFlagDefinition, unit: !78)
// CHECK:STDOUT: !82 = !DILocation(line: 6, column: 14, scope: !81)
// CHECK:STDOUT: !83 = !DILocation(line: 6, column: 27, scope: !81)
// CHECK:STDOUT: !86 = !DILocation(line: 5, column: 1, scope: !81)
// CHECK:STDOUT: !87 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.6dae29ece9a708c4:core.Destroy.Core", scope: null, file: !77, line: 6, type: !22, spFlags: DISPFlagDefinition, unit: !78, retainedNodes: !90)
// CHECK:STDOUT: !88 = !DILocalVariable(arg: 1, scope: !87, type: !20)
// CHECK:STDOUT: !89 = !DILocation(line: 6, column: 14, scope: !87)
// CHECK:STDOUT: !90 = !{!88}
// CHECK:STDOUT: !91 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core", scope: null, file: !77, line: 6, type: !22, spFlags: DISPFlagDefinition, unit: !78, retainedNodes: !94)
// CHECK:STDOUT: !92 = !DILocalVariable(arg: 1, scope: !91, type: !20)
// CHECK:STDOUT: !93 = !DILocation(line: 6, column: 14, scope: !91)
// CHECK:STDOUT: !94 = !{!92}
// CHECK:STDOUT: !95 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f58524831c37807e:core.Destroy.Core", scope: null, file: !77, line: 6, type: !22, spFlags: DISPFlagDefinition, unit: !78, retainedNodes: !98)
// CHECK:STDOUT: !96 = !DILocalVariable(arg: 1, scope: !95, type: !20)
// CHECK:STDOUT: !97 = !DILocation(line: 6, column: 14, scope: !95)
// CHECK:STDOUT: !98 = !{!96}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'val_return_form.carbon'
// CHECK:STDOUT: source_filename = "val_return_form.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @_CF.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !101 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main(), !dbg !102
// CHECK:STDOUT:   ret void, !dbg !105
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!100}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !23 = !{null}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !99 = !DIFile(filename: "val_return_form.carbon", directory: "")
// CHECK:STDOUT: !100 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !99, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !101 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !99, line: 5, type: !24, spFlags: DISPFlagDefinition, unit: !100)
// CHECK:STDOUT: !102 = !DILocation(line: 6, column: 23, scope: !101)
// CHECK:STDOUT: !105 = !DILocation(line: 5, column: 1, scope: !101)
// CHECK:STDOUT:
