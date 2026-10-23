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
// CHECK:STDOUT: define void @_CF.Main(ptr %_) #0 !dbg !23 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CG.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCallF.Main() #0 !dbg !28 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc12_6.2.temp = alloca {}, align 1, !dbg !29
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc12_6.2.temp), !dbg !29
// CHECK:STDOUT:   call void @_CF.Main(ptr @C.val.loc12_6.6), !dbg !30
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.356af3b6aef18564(ptr @C.val), !dbg !29
// CHECK:STDOUT:   call void @_CG.Main(), !dbg !31
// CHECK:STDOUT:   ret void, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.356af3b6aef18564(ptr %t) #0 !dbg !33 {
// CHECK:STDOUT:   ret void, !dbg !38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!17}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !16 = !DIFile(filename: "let_param.carbon", directory: "")
// CHECK:STDOUT: !17 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !16, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !18 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !19 = !{null, !18}
// CHECK:STDOUT: !20 = !DISubroutineType(types: !19)
// CHECK:STDOUT: !21 = !{null}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !16, line: 6, type: !20, spFlags: DISPFlagDefinition, unit: !17, retainedNodes: !27)
// CHECK:STDOUT: !24 = !DILocalVariable(arg: 1, scope: !23, type: !18)
// CHECK:STDOUT: !26 = !DILocation(line: 6, column: 1, scope: !23)
// CHECK:STDOUT: !27 = !{!24}
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "CallF", linkageName: "_CCallF.Main", scope: null, file: !16, line: 11, type: !22, spFlags: DISPFlagDefinition, unit: !17)
// CHECK:STDOUT: !29 = !DILocation(line: 12, column: 5, scope: !28)
// CHECK:STDOUT: !30 = !DILocation(line: 12, column: 3, scope: !28)
// CHECK:STDOUT: !31 = !DILocation(line: 13, column: 3, scope: !28)
// CHECK:STDOUT: !32 = !DILocation(line: 11, column: 1, scope: !28)
// CHECK:STDOUT: !33 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.356af3b6aef18564", scope: null, file: !6, line: 32, type: !20, spFlags: DISPFlagDefinition, unit: !17, retainedNodes: !39)
// CHECK:STDOUT: !34 = !DILocalVariable(arg: 1, scope: !33, type: !18)
// CHECK:STDOUT: !38 = !DILocation(line: 32, column: 1, scope: !33)
// CHECK:STDOUT: !39 = !{!34}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'var_param.carbon'
// CHECK:STDOUT: source_filename = "var_param.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @C.val.loc6_6.1 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main(ptr %_) #0 !dbg !42 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !46
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CG.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCallF.Main() #0 !dbg !48 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca {}, align 1, !dbg !49
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !49
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %_.var, ptr align 1 @C.val.loc6_6.1, i64 0, i1 false), !dbg !49
// CHECK:STDOUT:   call void @_CF.Main(ptr %_.var), !dbg !50
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.356af3b6aef18564(ptr %_.var), !dbg !49
// CHECK:STDOUT:   call void @_CG.Main(), !dbg !51
// CHECK:STDOUT:   ret void, !dbg !52
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.356af3b6aef18564(ptr %t) #0 !dbg !53 {
// CHECK:STDOUT:   ret void, !dbg !58
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!41}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !18 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !19 = !{null, !18}
// CHECK:STDOUT: !20 = !DISubroutineType(types: !19)
// CHECK:STDOUT: !21 = !{null}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !40 = !DIFile(filename: "var_param.carbon", directory: "")
// CHECK:STDOUT: !41 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !40, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !40, line: 6, type: !20, spFlags: DISPFlagDefinition, unit: !41, retainedNodes: !47)
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 1, scope: !42, type: !18)
// CHECK:STDOUT: !46 = !DILocation(line: 6, column: 1, scope: !42)
// CHECK:STDOUT: !47 = !{!43}
// CHECK:STDOUT: !48 = distinct !DISubprogram(name: "CallF", linkageName: "_CCallF.Main", scope: null, file: !40, line: 11, type: !22, spFlags: DISPFlagDefinition, unit: !41)
// CHECK:STDOUT: !49 = !DILocation(line: 6, column: 6, scope: !48)
// CHECK:STDOUT: !50 = !DILocation(line: 12, column: 3, scope: !48)
// CHECK:STDOUT: !51 = !DILocation(line: 13, column: 3, scope: !48)
// CHECK:STDOUT: !52 = !DILocation(line: 11, column: 1, scope: !48)
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.356af3b6aef18564", scope: null, file: !6, line: 32, type: !20, spFlags: DISPFlagDefinition, unit: !41, retainedNodes: !59)
// CHECK:STDOUT: !54 = !DILocalVariable(arg: 1, scope: !53, type: !18)
// CHECK:STDOUT: !58 = !DILocation(line: 32, column: 1, scope: !53)
// CHECK:STDOUT: !59 = !{!54}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'return_slot.carbon'
// CHECK:STDOUT: source_filename = "return_slot.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @C.val.loc7_12 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main(ptr sret({}) %return) #0 !dbg !64 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %return, ptr align 1 @C.val.loc7_12, i64 0, i1 false), !dbg !67
// CHECK:STDOUT:   ret void, !dbg !67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CG.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CForward.Main(ptr sret({}) %return) #0 !dbg !68 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main(ptr %return), !dbg !71
// CHECK:STDOUT:   ret void, !dbg !72
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CInitVar.Main() #0 !dbg !73 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca {}, align 1, !dbg !74
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !74
// CHECK:STDOUT:   call void @_CF.Main(ptr %_.var), !dbg !75
// CHECK:STDOUT:   call void @_CG.Main(), !dbg !77
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.356af3b6aef18564(ptr %_.var), !dbg !74
// CHECK:STDOUT:   ret void, !dbg !78
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CInitLet.Main() #0 !dbg !79 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc22_16.1.temp = alloca {}, align 1, !dbg !80
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc22_16.1.temp), !dbg !80
// CHECK:STDOUT:   call void @_CF.Main(ptr %.loc22_16.1.temp), !dbg !80
// CHECK:STDOUT:   call void @_CG.Main(), !dbg !83
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.356af3b6aef18564(ptr %.loc22_16.1.temp), !dbg !80
// CHECK:STDOUT:   ret void, !dbg !84
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.356af3b6aef18564(ptr %t) #0 !dbg !85 {
// CHECK:STDOUT:   ret void, !dbg !90
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.356af3b6aef18564, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!61}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !18 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !19 = !{null, !18}
// CHECK:STDOUT: !20 = !DISubroutineType(types: !19)
// CHECK:STDOUT: !21 = !{null}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !60 = !DIFile(filename: "return_slot.carbon", directory: "")
// CHECK:STDOUT: !61 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !60, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !62 = !{!18}
// CHECK:STDOUT: !63 = !DISubroutineType(types: !62)
// CHECK:STDOUT: !64 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !60, line: 6, type: !63, spFlags: DISPFlagDefinition, unit: !61)
// CHECK:STDOUT: !67 = !DILocation(line: 7, column: 3, scope: !64)
// CHECK:STDOUT: !68 = distinct !DISubprogram(name: "Forward", linkageName: "_CForward.Main", scope: null, file: !60, line: 12, type: !63, spFlags: DISPFlagDefinition, unit: !61)
// CHECK:STDOUT: !71 = !DILocation(line: 13, column: 10, scope: !68)
// CHECK:STDOUT: !72 = !DILocation(line: 13, column: 3, scope: !68)
// CHECK:STDOUT: !73 = distinct !DISubprogram(name: "InitVar", linkageName: "_CInitVar.Main", scope: null, file: !60, line: 16, type: !22, spFlags: DISPFlagDefinition, unit: !61)
// CHECK:STDOUT: !74 = !DILocation(line: 17, column: 3, scope: !73)
// CHECK:STDOUT: !75 = !DILocation(line: 17, column: 14, scope: !73)
// CHECK:STDOUT: !77 = !DILocation(line: 18, column: 3, scope: !73)
// CHECK:STDOUT: !78 = !DILocation(line: 16, column: 1, scope: !73)
// CHECK:STDOUT: !79 = distinct !DISubprogram(name: "InitLet", linkageName: "_CInitLet.Main", scope: null, file: !60, line: 21, type: !22, spFlags: DISPFlagDefinition, unit: !61)
// CHECK:STDOUT: !80 = !DILocation(line: 22, column: 14, scope: !79)
// CHECK:STDOUT: !83 = !DILocation(line: 23, column: 3, scope: !79)
// CHECK:STDOUT: !84 = !DILocation(line: 21, column: 1, scope: !79)
// CHECK:STDOUT: !85 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.356af3b6aef18564", scope: null, file: !6, line: 32, type: !20, spFlags: DISPFlagDefinition, unit: !61, retainedNodes: !91)
// CHECK:STDOUT: !86 = !DILocalVariable(arg: 1, scope: !85, type: !18)
// CHECK:STDOUT: !90 = !DILocation(line: 32, column: 1, scope: !85)
// CHECK:STDOUT: !91 = !{!86}
// CHECK:STDOUT:
