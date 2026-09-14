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
// CHECK:STDOUT:   call void @"_COp.cbd14a388b4a1c25:core.Destroy.Core"(ptr @C.val), !dbg !27
// CHECK:STDOUT:   call void @_CG.Main(), !dbg !29
// CHECK:STDOUT:   ret void, !dbg !30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.cbd14a388b4a1c25:core.Destroy.Core"(ptr %self) #0 !dbg !31 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!15}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
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
// CHECK:STDOUT: !31 = distinct !DISubprogram(name: "Op", linkageName: "_COp.cbd14a388b4a1c25:core.Destroy.Core", scope: null, file: !14, line: 12, type: !18, spFlags: DISPFlagDefinition, unit: !15, retainedNodes: !34)
// CHECK:STDOUT: !32 = !DILocalVariable(arg: 1, scope: !31, type: !16)
// CHECK:STDOUT: !33 = !DILocation(line: 12, column: 5, scope: !31)
// CHECK:STDOUT: !34 = !{!32}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'var_param.carbon'
// CHECK:STDOUT: source_filename = "var_param.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @C.val.loc6 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main(ptr %_) #0 !dbg !37 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CG.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCallF.Main() #0 !dbg !43 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca {}, align 1, !dbg !44
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !44
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %_.var, ptr align 1 @C.val.loc6, i64 0, i1 false), !dbg !44
// CHECK:STDOUT:   call void @_CF.Main(ptr %_.var), !dbg !45
// CHECK:STDOUT:   call void @"_COp.cbd14a388b4a1c25:core.Destroy.Core"(ptr %_.var), !dbg !44
// CHECK:STDOUT:   call void @_CG.Main(), !dbg !46
// CHECK:STDOUT:   ret void, !dbg !47
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.cbd14a388b4a1c25:core.Destroy.Core"(ptr %self) #0 !dbg !48 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !50
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!36}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !16 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !17 = !{null, !16}
// CHECK:STDOUT: !18 = !DISubroutineType(types: !17)
// CHECK:STDOUT: !19 = !{null}
// CHECK:STDOUT: !20 = !DISubroutineType(types: !19)
// CHECK:STDOUT: !35 = !DIFile(filename: "var_param.carbon", directory: "")
// CHECK:STDOUT: !36 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !35, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !37 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !35, line: 6, type: !18, spFlags: DISPFlagDefinition, unit: !36, retainedNodes: !42)
// CHECK:STDOUT: !38 = !DILocalVariable(arg: 1, scope: !37, type: !16)
// CHECK:STDOUT: !41 = !DILocation(line: 6, column: 1, scope: !37)
// CHECK:STDOUT: !42 = !{!38}
// CHECK:STDOUT: !43 = distinct !DISubprogram(name: "CallF", linkageName: "_CCallF.Main", scope: null, file: !35, line: 11, type: !20, spFlags: DISPFlagDefinition, unit: !36)
// CHECK:STDOUT: !44 = !DILocation(line: 6, column: 6, scope: !43)
// CHECK:STDOUT: !45 = !DILocation(line: 12, column: 3, scope: !43)
// CHECK:STDOUT: !46 = !DILocation(line: 13, column: 3, scope: !43)
// CHECK:STDOUT: !47 = !DILocation(line: 11, column: 1, scope: !43)
// CHECK:STDOUT: !48 = distinct !DISubprogram(name: "Op", linkageName: "_COp.cbd14a388b4a1c25:core.Destroy.Core", scope: null, file: !35, line: 6, type: !18, spFlags: DISPFlagDefinition, unit: !36, retainedNodes: !51)
// CHECK:STDOUT: !49 = !DILocalVariable(arg: 1, scope: !48, type: !16)
// CHECK:STDOUT: !50 = !DILocation(line: 6, column: 6, scope: !48)
// CHECK:STDOUT: !51 = !{!49}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'return_slot.carbon'
// CHECK:STDOUT: source_filename = "return_slot.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @C.val.loc7_12 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main(ptr sret({}) %return) #0 !dbg !56 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %return, ptr align 1 @C.val.loc7_12, i64 0, i1 false), !dbg !59
// CHECK:STDOUT:   ret void, !dbg !59
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CG.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CForward.Main(ptr sret({}) %return) #0 !dbg !60 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main(ptr %return), !dbg !63
// CHECK:STDOUT:   ret void, !dbg !64
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CInitVar.Main() #0 !dbg !65 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca {}, align 1, !dbg !66
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !66
// CHECK:STDOUT:   call void @_CF.Main(ptr %_.var), !dbg !67
// CHECK:STDOUT:   call void @_CG.Main(), !dbg !69
// CHECK:STDOUT:   call void @"_COp.cbd14a388b4a1c25:core.Destroy.Core"(ptr %_.var), !dbg !66
// CHECK:STDOUT:   ret void, !dbg !70
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.cbd14a388b4a1c25:core.Destroy.Core"(ptr %self) #0 !dbg !71 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !73
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CInitLet.Main() #0 !dbg !75 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc22_16.1.temp = alloca {}, align 1, !dbg !76
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc22_16.1.temp), !dbg !76
// CHECK:STDOUT:   call void @_CF.Main(ptr %.loc22_16.1.temp), !dbg !76
// CHECK:STDOUT:   call void @_CG.Main(), !dbg !79
// CHECK:STDOUT:   call void @"_COp.cbd14a388b4a1c25:core.Destroy.Core"(ptr %.loc22_16.1.temp), !dbg !76
// CHECK:STDOUT:   ret void, !dbg !80
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!53}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !16 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !17 = !{null, !16}
// CHECK:STDOUT: !18 = !DISubroutineType(types: !17)
// CHECK:STDOUT: !19 = !{null}
// CHECK:STDOUT: !20 = !DISubroutineType(types: !19)
// CHECK:STDOUT: !52 = !DIFile(filename: "return_slot.carbon", directory: "")
// CHECK:STDOUT: !53 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !52, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !54 = !{!16}
// CHECK:STDOUT: !55 = !DISubroutineType(types: !54)
// CHECK:STDOUT: !56 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !52, line: 6, type: !55, spFlags: DISPFlagDefinition, unit: !53)
// CHECK:STDOUT: !59 = !DILocation(line: 7, column: 3, scope: !56)
// CHECK:STDOUT: !60 = distinct !DISubprogram(name: "Forward", linkageName: "_CForward.Main", scope: null, file: !52, line: 12, type: !55, spFlags: DISPFlagDefinition, unit: !53)
// CHECK:STDOUT: !63 = !DILocation(line: 13, column: 10, scope: !60)
// CHECK:STDOUT: !64 = !DILocation(line: 13, column: 3, scope: !60)
// CHECK:STDOUT: !65 = distinct !DISubprogram(name: "InitVar", linkageName: "_CInitVar.Main", scope: null, file: !52, line: 16, type: !20, spFlags: DISPFlagDefinition, unit: !53)
// CHECK:STDOUT: !66 = !DILocation(line: 17, column: 3, scope: !65)
// CHECK:STDOUT: !67 = !DILocation(line: 17, column: 14, scope: !65)
// CHECK:STDOUT: !69 = !DILocation(line: 18, column: 3, scope: !65)
// CHECK:STDOUT: !70 = !DILocation(line: 16, column: 1, scope: !65)
// CHECK:STDOUT: !71 = distinct !DISubprogram(name: "Op", linkageName: "_COp.cbd14a388b4a1c25:core.Destroy.Core", scope: null, file: !52, line: 17, type: !18, spFlags: DISPFlagDefinition, unit: !53, retainedNodes: !74)
// CHECK:STDOUT: !72 = !DILocalVariable(arg: 1, scope: !71, type: !16)
// CHECK:STDOUT: !73 = !DILocation(line: 17, column: 3, scope: !71)
// CHECK:STDOUT: !74 = !{!72}
// CHECK:STDOUT: !75 = distinct !DISubprogram(name: "InitLet", linkageName: "_CInitLet.Main", scope: null, file: !52, line: 21, type: !20, spFlags: DISPFlagDefinition, unit: !53)
// CHECK:STDOUT: !76 = !DILocation(line: 22, column: 14, scope: !75)
// CHECK:STDOUT: !79 = !DILocation(line: 23, column: 3, scope: !75)
// CHECK:STDOUT: !80 = !DILocation(line: 21, column: 1, scope: !75)
// CHECK:STDOUT:
