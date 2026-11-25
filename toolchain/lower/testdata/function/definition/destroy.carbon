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
// CHECK:STDOUT:   call void @_CG.Main(), !dbg !29
// CHECK:STDOUT:   ret void, !dbg !30
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
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'var_param.carbon'
// CHECK:STDOUT: source_filename = "var_param.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @C.val.loc6 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main(ptr %_) #0 !dbg !33 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CG.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCallF.Main() #0 !dbg !39 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca {}, align 1, !dbg !40
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !40
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %_.var, ptr align 1 @C.val.loc6, i64 0, i1 false), !dbg !40
// CHECK:STDOUT:   call void @_CF.Main(ptr %_.var), !dbg !41
// CHECK:STDOUT:   call void @_CG.Main(), !dbg !42
// CHECK:STDOUT:   ret void, !dbg !43
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!32}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !16 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !17 = !{null, !16}
// CHECK:STDOUT: !18 = !DISubroutineType(types: !17)
// CHECK:STDOUT: !19 = !{null}
// CHECK:STDOUT: !20 = !DISubroutineType(types: !19)
// CHECK:STDOUT: !31 = !DIFile(filename: "var_param.carbon", directory: "")
// CHECK:STDOUT: !32 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !31, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !33 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !31, line: 6, type: !18, spFlags: DISPFlagDefinition, unit: !32, retainedNodes: !38)
// CHECK:STDOUT: !34 = !DILocalVariable(arg: 1, scope: !33, type: !16)
// CHECK:STDOUT: !37 = !DILocation(line: 6, column: 1, scope: !33)
// CHECK:STDOUT: !38 = !{!34}
// CHECK:STDOUT: !39 = distinct !DISubprogram(name: "CallF", linkageName: "_CCallF.Main", scope: null, file: !31, line: 11, type: !20, spFlags: DISPFlagDefinition, unit: !32)
// CHECK:STDOUT: !40 = !DILocation(line: 6, column: 6, scope: !39)
// CHECK:STDOUT: !41 = !DILocation(line: 12, column: 3, scope: !39)
// CHECK:STDOUT: !42 = !DILocation(line: 13, column: 3, scope: !39)
// CHECK:STDOUT: !43 = !DILocation(line: 11, column: 1, scope: !39)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'return_slot.carbon'
// CHECK:STDOUT: source_filename = "return_slot.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @C.val.loc7_12 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main(ptr sret({}) %return) #0 !dbg !48 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %return, ptr align 1 @C.val.loc7_12, i64 0, i1 false), !dbg !51
// CHECK:STDOUT:   ret void, !dbg !51
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CG.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CForward.Main(ptr sret({}) %return) #0 !dbg !52 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main(ptr %return), !dbg !55
// CHECK:STDOUT:   ret void, !dbg !56
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CInitVar.Main() #0 !dbg !57 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca {}, align 1, !dbg !58
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !58
// CHECK:STDOUT:   call void @_CF.Main(ptr %_.var), !dbg !59
// CHECK:STDOUT:   call void @_CG.Main(), !dbg !61
// CHECK:STDOUT:   ret void, !dbg !62
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CInitLet.Main() #0 !dbg !63 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc22_16.1.temp = alloca {}, align 1, !dbg !64
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc22_16.1.temp), !dbg !64
// CHECK:STDOUT:   call void @_CF.Main(ptr %.loc22_16.1.temp), !dbg !64
// CHECK:STDOUT:   call void @_CG.Main(), !dbg !67
// CHECK:STDOUT:   ret void, !dbg !68
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!45}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !16 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !19 = !{null}
// CHECK:STDOUT: !20 = !DISubroutineType(types: !19)
// CHECK:STDOUT: !44 = !DIFile(filename: "return_slot.carbon", directory: "")
// CHECK:STDOUT: !45 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !44, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !46 = !{!16}
// CHECK:STDOUT: !47 = !DISubroutineType(types: !46)
// CHECK:STDOUT: !48 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !44, line: 6, type: !47, spFlags: DISPFlagDefinition, unit: !45)
// CHECK:STDOUT: !51 = !DILocation(line: 7, column: 3, scope: !48)
// CHECK:STDOUT: !52 = distinct !DISubprogram(name: "Forward", linkageName: "_CForward.Main", scope: null, file: !44, line: 12, type: !47, spFlags: DISPFlagDefinition, unit: !45)
// CHECK:STDOUT: !55 = !DILocation(line: 13, column: 10, scope: !52)
// CHECK:STDOUT: !56 = !DILocation(line: 13, column: 3, scope: !52)
// CHECK:STDOUT: !57 = distinct !DISubprogram(name: "InitVar", linkageName: "_CInitVar.Main", scope: null, file: !44, line: 16, type: !20, spFlags: DISPFlagDefinition, unit: !45)
// CHECK:STDOUT: !58 = !DILocation(line: 17, column: 3, scope: !57)
// CHECK:STDOUT: !59 = !DILocation(line: 17, column: 14, scope: !57)
// CHECK:STDOUT: !61 = !DILocation(line: 18, column: 3, scope: !57)
// CHECK:STDOUT: !62 = !DILocation(line: 16, column: 1, scope: !57)
// CHECK:STDOUT: !63 = distinct !DISubprogram(name: "InitLet", linkageName: "_CInitLet.Main", scope: null, file: !44, line: 21, type: !20, spFlags: DISPFlagDefinition, unit: !45)
// CHECK:STDOUT: !64 = !DILocation(line: 22, column: 14, scope: !63)
// CHECK:STDOUT: !67 = !DILocation(line: 23, column: 3, scope: !63)
// CHECK:STDOUT: !68 = !DILocation(line: 21, column: 1, scope: !63)
// CHECK:STDOUT:
