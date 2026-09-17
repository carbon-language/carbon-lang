// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/definition/destroy.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/definition/destroy.carbon

// --- let_param.carbon

library "[[@TEST_NAME]]";

class C {}

fn F(_: C) {
}

fn G();

fn CallF() {
  F({});
  G();
}

// --- var_param.carbon

library "[[@TEST_NAME]]";

class C {}

fn F(var _: C) {
}

fn G();

fn CallF() {
  F({});
  G();
}

// --- return_slot.carbon

library "[[@TEST_NAME]]";

class C {}

fn F() -> C {
  return {};
}

fn G();

fn Forward() -> C {
  return F();
}

fn InitVar() {
  var _: C = F();
  G();
}

fn InitLet() {
  let _: C = F();
  G();
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'let_param.carbon'
// CHECK:STDOUT: source_filename = "let_param.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @C.val = internal constant {} zeroinitializer
// CHECK:STDOUT: @C.val.loc12_6.6 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main(ptr %_) #0 !dbg !21 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !24
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CG.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCallF.Main() #0 !dbg !26 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc12_6.2.temp = alloca {}, align 1, !dbg !27
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc12_6.2.temp), !dbg !27
// CHECK:STDOUT:   call void @_CF.Main(ptr @C.val.loc12_6.6), !dbg !28
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.2eb69dd0c58be63c(ptr @C.val), !dbg !27
// CHECK:STDOUT:   call void @_CG.Main(), !dbg !29
// CHECK:STDOUT:   ret void, !dbg !30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.08d9e38b928ba575:core.Destroy.Core"(ptr %self) #0 !dbg !31 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.2eb69dd0c58be63c(ptr %self) #0 !dbg !35 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.08d9e38b928ba575:core.Destroy.Core"(ptr %self), !dbg !39
// CHECK:STDOUT:   ret void, !dbg !40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!15}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !14 = !DIFile(filename: "let_param.carbon", directory: "")
// CHECK:STDOUT: !15 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !14, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !16 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !17 = !{null, !16}
// CHECK:STDOUT: !18 = !DISubroutineType(types: !17)
// CHECK:STDOUT: !19 = !{null}
// CHECK:STDOUT: !20 = !DISubroutineType(types: !19)
// CHECK:STDOUT: !21 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !14, line: 6, type: !18, spFlags: DISPFlagDefinition, unit: !15, retainedNodes: !25)
// CHECK:STDOUT: !22 = !DILocalVariable(arg: 1, scope: !21, type: !16)
// CHECK:STDOUT: !24 = !DILocation(line: 6, column: 1, scope: !21)
// CHECK:STDOUT: !25 = !{!22}
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "CallF", linkageName: "_CCallF.Main", scope: null, file: !14, line: 11, type: !20, spFlags: DISPFlagDefinition, unit: !15)
// CHECK:STDOUT: !27 = !DILocation(line: 12, column: 5, scope: !26)
// CHECK:STDOUT: !28 = !DILocation(line: 12, column: 3, scope: !26)
// CHECK:STDOUT: !29 = !DILocation(line: 13, column: 3, scope: !26)
// CHECK:STDOUT: !30 = !DILocation(line: 11, column: 1, scope: !26)
// CHECK:STDOUT: !31 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.08d9e38b928ba575:core.Destroy.Core", scope: null, file: !14, line: 12, type: !18, spFlags: DISPFlagDefinition, unit: !15, retainedNodes: !34)
// CHECK:STDOUT: !32 = !DILocalVariable(arg: 1, scope: !31, type: !16)
// CHECK:STDOUT: !33 = !DILocation(line: 12, column: 5, scope: !31)
// CHECK:STDOUT: !34 = !{!32}
// CHECK:STDOUT: !35 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.2eb69dd0c58be63c", scope: null, file: !10, line: 24, type: !18, spFlags: DISPFlagDefinition, unit: !15, retainedNodes: !41)
// CHECK:STDOUT: !36 = !DILocalVariable(arg: 1, scope: !35, type: !16)
// CHECK:STDOUT: !39 = !DILocation(line: 26, column: 5, scope: !35)
// CHECK:STDOUT: !40 = !DILocation(line: 24, column: 3, scope: !35)
// CHECK:STDOUT: !41 = !{!36}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'var_param.carbon'
// CHECK:STDOUT: source_filename = "var_param.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @C.val.loc6_6.1 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main(ptr %_) #0 !dbg !44 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CG.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCallF.Main() #0 !dbg !50 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca {}, align 1, !dbg !51
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !51
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %_.var, ptr align 1 @C.val.loc6_6.1, i64 0, i1 false), !dbg !51
// CHECK:STDOUT:   call void @_CF.Main(ptr %_.var), !dbg !52
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.2eb69dd0c58be63c(ptr %_.var), !dbg !51
// CHECK:STDOUT:   call void @_CG.Main(), !dbg !53
// CHECK:STDOUT:   ret void, !dbg !54
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.08d9e38b928ba575:core.Destroy.Core"(ptr %self) #0 !dbg !55 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !57
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.2eb69dd0c58be63c(ptr %self) #0 !dbg !59 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.08d9e38b928ba575:core.Destroy.Core"(ptr %self), !dbg !63
// CHECK:STDOUT:   ret void, !dbg !64
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!43}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !16 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !17 = !{null, !16}
// CHECK:STDOUT: !18 = !DISubroutineType(types: !17)
// CHECK:STDOUT: !19 = !{null}
// CHECK:STDOUT: !20 = !DISubroutineType(types: !19)
// CHECK:STDOUT: !42 = !DIFile(filename: "var_param.carbon", directory: "")
// CHECK:STDOUT: !43 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !42, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !44 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !42, line: 6, type: !18, spFlags: DISPFlagDefinition, unit: !43, retainedNodes: !49)
// CHECK:STDOUT: !45 = !DILocalVariable(arg: 1, scope: !44, type: !16)
// CHECK:STDOUT: !48 = !DILocation(line: 6, column: 1, scope: !44)
// CHECK:STDOUT: !49 = !{!45}
// CHECK:STDOUT: !50 = distinct !DISubprogram(name: "CallF", linkageName: "_CCallF.Main", scope: null, file: !42, line: 11, type: !20, spFlags: DISPFlagDefinition, unit: !43)
// CHECK:STDOUT: !51 = !DILocation(line: 6, column: 6, scope: !50)
// CHECK:STDOUT: !52 = !DILocation(line: 12, column: 3, scope: !50)
// CHECK:STDOUT: !53 = !DILocation(line: 13, column: 3, scope: !50)
// CHECK:STDOUT: !54 = !DILocation(line: 11, column: 1, scope: !50)
// CHECK:STDOUT: !55 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.08d9e38b928ba575:core.Destroy.Core", scope: null, file: !42, line: 6, type: !18, spFlags: DISPFlagDefinition, unit: !43, retainedNodes: !58)
// CHECK:STDOUT: !56 = !DILocalVariable(arg: 1, scope: !55, type: !16)
// CHECK:STDOUT: !57 = !DILocation(line: 6, column: 6, scope: !55)
// CHECK:STDOUT: !58 = !{!56}
// CHECK:STDOUT: !59 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.2eb69dd0c58be63c", scope: null, file: !10, line: 24, type: !18, spFlags: DISPFlagDefinition, unit: !43, retainedNodes: !65)
// CHECK:STDOUT: !60 = !DILocalVariable(arg: 1, scope: !59, type: !16)
// CHECK:STDOUT: !63 = !DILocation(line: 26, column: 5, scope: !59)
// CHECK:STDOUT: !64 = !DILocation(line: 24, column: 3, scope: !59)
// CHECK:STDOUT: !65 = !{!60}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'return_slot.carbon'
// CHECK:STDOUT: source_filename = "return_slot.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @C.val.loc7_12 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main(ptr sret({}) %return) #0 !dbg !70 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %return, ptr align 1 @C.val.loc7_12, i64 0, i1 false), !dbg !73
// CHECK:STDOUT:   ret void, !dbg !73
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CG.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CForward.Main(ptr sret({}) %return) #0 !dbg !74 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main(ptr %return), !dbg !77
// CHECK:STDOUT:   ret void, !dbg !78
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CInitVar.Main() #0 !dbg !79 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca {}, align 1, !dbg !80
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !80
// CHECK:STDOUT:   call void @_CF.Main(ptr %_.var), !dbg !81
// CHECK:STDOUT:   call void @_CG.Main(), !dbg !83
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.2eb69dd0c58be63c(ptr %_.var), !dbg !80
// CHECK:STDOUT:   ret void, !dbg !84
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.08d9e38b928ba575:core.Destroy.Core"(ptr %self) #0 !dbg !85 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !87
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CInitLet.Main() #0 !dbg !89 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc22_16.1.temp = alloca {}, align 1, !dbg !90
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc22_16.1.temp), !dbg !90
// CHECK:STDOUT:   call void @_CF.Main(ptr %.loc22_16.1.temp), !dbg !90
// CHECK:STDOUT:   call void @_CG.Main(), !dbg !93
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.2eb69dd0c58be63c(ptr %.loc22_16.1.temp), !dbg !90
// CHECK:STDOUT:   ret void, !dbg !94
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.2eb69dd0c58be63c(ptr %self) #0 !dbg !95 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.08d9e38b928ba575:core.Destroy.Core"(ptr %self), !dbg !99
// CHECK:STDOUT:   ret void, !dbg !100
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Destroy.Core.2eb69dd0c58be63c, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!67}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !16 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !17 = !{null, !16}
// CHECK:STDOUT: !18 = !DISubroutineType(types: !17)
// CHECK:STDOUT: !19 = !{null}
// CHECK:STDOUT: !20 = !DISubroutineType(types: !19)
// CHECK:STDOUT: !66 = !DIFile(filename: "return_slot.carbon", directory: "")
// CHECK:STDOUT: !67 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !66, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !68 = !{!16}
// CHECK:STDOUT: !69 = !DISubroutineType(types: !68)
// CHECK:STDOUT: !70 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !66, line: 6, type: !69, spFlags: DISPFlagDefinition, unit: !67)
// CHECK:STDOUT: !73 = !DILocation(line: 7, column: 3, scope: !70)
// CHECK:STDOUT: !74 = distinct !DISubprogram(name: "Forward", linkageName: "_CForward.Main", scope: null, file: !66, line: 12, type: !69, spFlags: DISPFlagDefinition, unit: !67)
// CHECK:STDOUT: !77 = !DILocation(line: 13, column: 10, scope: !74)
// CHECK:STDOUT: !78 = !DILocation(line: 13, column: 3, scope: !74)
// CHECK:STDOUT: !79 = distinct !DISubprogram(name: "InitVar", linkageName: "_CInitVar.Main", scope: null, file: !66, line: 16, type: !20, spFlags: DISPFlagDefinition, unit: !67)
// CHECK:STDOUT: !80 = !DILocation(line: 17, column: 3, scope: !79)
// CHECK:STDOUT: !81 = !DILocation(line: 17, column: 14, scope: !79)
// CHECK:STDOUT: !83 = !DILocation(line: 18, column: 3, scope: !79)
// CHECK:STDOUT: !84 = !DILocation(line: 16, column: 1, scope: !79)
// CHECK:STDOUT: !85 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.08d9e38b928ba575:core.Destroy.Core", scope: null, file: !66, line: 17, type: !18, spFlags: DISPFlagDefinition, unit: !67, retainedNodes: !88)
// CHECK:STDOUT: !86 = !DILocalVariable(arg: 1, scope: !85, type: !16)
// CHECK:STDOUT: !87 = !DILocation(line: 17, column: 3, scope: !85)
// CHECK:STDOUT: !88 = !{!86}
// CHECK:STDOUT: !89 = distinct !DISubprogram(name: "InitLet", linkageName: "_CInitLet.Main", scope: null, file: !66, line: 21, type: !20, spFlags: DISPFlagDefinition, unit: !67)
// CHECK:STDOUT: !90 = !DILocation(line: 22, column: 14, scope: !89)
// CHECK:STDOUT: !93 = !DILocation(line: 23, column: 3, scope: !89)
// CHECK:STDOUT: !94 = !DILocation(line: 21, column: 1, scope: !89)
// CHECK:STDOUT: !95 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.2eb69dd0c58be63c", scope: null, file: !10, line: 24, type: !18, spFlags: DISPFlagDefinition, unit: !67, retainedNodes: !101)
// CHECK:STDOUT: !96 = !DILocalVariable(arg: 1, scope: !95, type: !16)
// CHECK:STDOUT: !99 = !DILocation(line: 26, column: 5, scope: !95)
// CHECK:STDOUT: !100 = !DILocation(line: 24, column: 3, scope: !95)
// CHECK:STDOUT: !101 = !{!96}
// CHECK:STDOUT:
