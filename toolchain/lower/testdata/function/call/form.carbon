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
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48(ptr %y.var), !dbg !26
// CHECK:STDOUT:   ret void, !dbg !31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !32 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48(ptr %self) #0 !dbg !36 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self), !dbg !40
// CHECK:STDOUT:   ret void, !dbg !41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !14 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
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
// CHECK:STDOUT: !32 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core", scope: null, file: !18, line: 6, type: !22, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !35)
// CHECK:STDOUT: !33 = !DILocalVariable(arg: 1, scope: !32, type: !20)
// CHECK:STDOUT: !34 = !DILocation(line: 6, column: 3, scope: !32)
// CHECK:STDOUT: !35 = !{!33}
// CHECK:STDOUT: !36 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48", scope: null, file: !14, line: 24, type: !22, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !42)
// CHECK:STDOUT: !37 = !DILocalVariable(arg: 1, scope: !36, type: !20)
// CHECK:STDOUT: !40 = !DILocation(line: 26, column: 5, scope: !36)
// CHECK:STDOUT: !41 = !DILocation(line: 24, column: 3, scope: !36)
// CHECK:STDOUT: !42 = !{!37}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'var_form_param.carbon'
// CHECK:STDOUT: source_filename = "var_form_param.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CF.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !45 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x.var = alloca i32, align 4, !dbg !46
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %x.var), !dbg !46
// CHECK:STDOUT:   store i32 0, ptr %x.var, align 4, !dbg !46
// CHECK:STDOUT:   call void @_CF.Main(ptr %x.var), !dbg !47
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48(ptr %x.var), !dbg !46
// CHECK:STDOUT:   ret void, !dbg !48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !49 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !51
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48(ptr %self) #0 !dbg !53 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self), !dbg !57
// CHECK:STDOUT:   ret void, !dbg !58
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!44}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !14 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !{null, !20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = !{null}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !43 = !DIFile(filename: "var_form_param.carbon", directory: "")
// CHECK:STDOUT: !44 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !43, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !45 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !43, line: 5, type: !24, spFlags: DISPFlagDefinition, unit: !44)
// CHECK:STDOUT: !46 = !DILocation(line: 3, column: 6, scope: !45)
// CHECK:STDOUT: !47 = !DILocation(line: 6, column: 3, scope: !45)
// CHECK:STDOUT: !48 = !DILocation(line: 5, column: 1, scope: !45)
// CHECK:STDOUT: !49 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core", scope: null, file: !43, line: 3, type: !22, spFlags: DISPFlagDefinition, unit: !44, retainedNodes: !52)
// CHECK:STDOUT: !50 = !DILocalVariable(arg: 1, scope: !49, type: !20)
// CHECK:STDOUT: !51 = !DILocation(line: 3, column: 6, scope: !49)
// CHECK:STDOUT: !52 = !{!50}
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48", scope: null, file: !14, line: 24, type: !22, spFlags: DISPFlagDefinition, unit: !44, retainedNodes: !59)
// CHECK:STDOUT: !54 = !DILocalVariable(arg: 1, scope: !53, type: !20)
// CHECK:STDOUT: !57 = !DILocation(line: 26, column: 5, scope: !53)
// CHECK:STDOUT: !58 = !DILocation(line: 24, column: 3, scope: !53)
// CHECK:STDOUT: !59 = !{!54}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'val_form_param.carbon'
// CHECK:STDOUT: source_filename = "val_form_param.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CF.Main(i32)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !62 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main(i32 0), !dbg !63
// CHECK:STDOUT:   ret void, !dbg !64
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!61}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !23 = !{null}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !60 = !DIFile(filename: "val_form_param.carbon", directory: "")
// CHECK:STDOUT: !61 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !60, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !62 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !60, line: 5, type: !24, spFlags: DISPFlagDefinition, unit: !61)
// CHECK:STDOUT: !63 = !DILocation(line: 6, column: 3, scope: !62)
// CHECK:STDOUT: !64 = !DILocation(line: 5, column: 1, scope: !62)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'ref_return_form.carbon'
// CHECK:STDOUT: source_filename = "ref_return_form.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare ptr @_CF.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !70 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %F.call = call ptr @_CF.Main(), !dbg !71
// CHECK:STDOUT:   ret void, !dbg !74
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!66}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !23 = !{null}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !65 = !DIFile(filename: "ref_return_form.carbon", directory: "")
// CHECK:STDOUT: !66 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !65, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !70 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !65, line: 5, type: !24, spFlags: DISPFlagDefinition, unit: !66)
// CHECK:STDOUT: !71 = !DILocation(line: 6, column: 27, scope: !70)
// CHECK:STDOUT: !74 = !DILocation(line: 5, column: 1, scope: !70)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'var_return_form.carbon'
// CHECK:STDOUT: source_filename = "var_return_form.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @_CF.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !79 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x.var = alloca i32, align 4, !dbg !80
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %x.var), !dbg !80
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main(), !dbg !81
// CHECK:STDOUT:   store i32 %F.call, ptr %x.var, align 4, !dbg !80
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48(ptr %x.var), !dbg !80
// CHECK:STDOUT:   ret void, !dbg !84
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !85 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !87
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48(ptr %self) #0 !dbg !89 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self), !dbg !93
// CHECK:STDOUT:   ret void, !dbg !94
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!76}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !14 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !{null, !20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = !{null}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !75 = !DIFile(filename: "var_return_form.carbon", directory: "")
// CHECK:STDOUT: !76 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !75, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !79 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !75, line: 5, type: !24, spFlags: DISPFlagDefinition, unit: !76)
// CHECK:STDOUT: !80 = !DILocation(line: 6, column: 14, scope: !79)
// CHECK:STDOUT: !81 = !DILocation(line: 6, column: 27, scope: !79)
// CHECK:STDOUT: !84 = !DILocation(line: 5, column: 1, scope: !79)
// CHECK:STDOUT: !85 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core", scope: null, file: !75, line: 6, type: !22, spFlags: DISPFlagDefinition, unit: !76, retainedNodes: !88)
// CHECK:STDOUT: !86 = !DILocalVariable(arg: 1, scope: !85, type: !20)
// CHECK:STDOUT: !87 = !DILocation(line: 6, column: 14, scope: !85)
// CHECK:STDOUT: !88 = !{!86}
// CHECK:STDOUT: !89 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48", scope: null, file: !14, line: 24, type: !22, spFlags: DISPFlagDefinition, unit: !76, retainedNodes: !95)
// CHECK:STDOUT: !90 = !DILocalVariable(arg: 1, scope: !89, type: !20)
// CHECK:STDOUT: !93 = !DILocation(line: 26, column: 5, scope: !89)
// CHECK:STDOUT: !94 = !DILocation(line: 24, column: 3, scope: !89)
// CHECK:STDOUT: !95 = !{!90}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'val_return_form.carbon'
// CHECK:STDOUT: source_filename = "val_return_form.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @_CF.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !98 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main(), !dbg !99
// CHECK:STDOUT:   ret void, !dbg !102
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!97}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !23 = !{null}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !96 = !DIFile(filename: "val_return_form.carbon", directory: "")
// CHECK:STDOUT: !97 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !96, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !98 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !96, line: 5, type: !24, spFlags: DISPFlagDefinition, unit: !97)
// CHECK:STDOUT: !99 = !DILocation(line: 6, column: 23, scope: !98)
// CHECK:STDOUT: !102 = !DILocation(line: 5, column: 1, scope: !98)
// CHECK:STDOUT:
