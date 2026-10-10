// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/builtins/maybe_unformed.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/builtins/maybe_unformed.carbon

fn MaybeUnformedBuiltin(t: type) -> type = "maybe_unformed.make_type";

fn TakeI32(_: MaybeUnformedBuiltin(i32)) {}
fn MakeI32() -> MaybeUnformedBuiltin(i32);

fn TakeBool(_: MaybeUnformedBuiltin(bool)) {}
fn MakeBool() -> MaybeUnformedBuiltin(bool);

fn PassI32() {
  // This is passed by value, as that preserves all the bits.
  TakeI32(MakeI32());
}

fn PassBool() {
  // This is passed by pointer to preserve all the bits.
  TakeBool(MakeBool());
}

fn ConvertBoolToUnformedBool(var b: bool) {
  TakeBool(b);
}

// Also test that an adapter wrapped around the builtin is passed appropriately,
// using the same calling convention as the adapted type.

class MaybeUnformedAdapter(T: type) {
  adapt MaybeUnformedBuiltin(T);
}

fn TakeAdapterI32(_: MaybeUnformedAdapter(i32)) {}
fn MakeAdapterI32() -> MaybeUnformedAdapter(i32);

fn PassAdapterI32() {
  // TODO: The `value_of_initializer` optimization doesn't get used here, so we
  // unnecessarily round-trip this value through memory.
  TakeAdapterI32(MakeAdapterI32());
}

fn TakeAdapterBool(_: MaybeUnformedAdapter(bool)) {}
fn MakeAdapterBool() -> MaybeUnformedAdapter(bool);

fn PassAdapterBool() {
  TakeAdapterBool(MakeAdapterBool());
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'maybe_unformed.carbon'
// CHECK:STDOUT: source_filename = "maybe_unformed.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTakeI32.Main(i32 %_) #0 !dbg !48 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !51
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @_CMakeI32.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTakeBool.Main(ptr %_) #0 !dbg !53 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !56
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CMakeBool.Main(ptr sret(i1))
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassI32.Main() #0 !dbg !58 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %MakeI32.call = call i32 @_CMakeI32.Main(), !dbg !59
// CHECK:STDOUT:   call void @_CTakeI32.Main(i32 %MakeI32.call), !dbg !60
// CHECK:STDOUT:   ret void, !dbg !61
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassBool.Main() #0 !dbg !62 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc28_21.1.temp = alloca i1, align 1, !dbg !63
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc28_21.1.temp), !dbg !63
// CHECK:STDOUT:   call void @_CMakeBool.Main(ptr %.loc28_21.1.temp), !dbg !63
// CHECK:STDOUT:   call void @_CTakeBool.Main(ptr %.loc28_21.1.temp), !dbg !64
// CHECK:STDOUT:   call void @"_CSelfDestruct.2ec8e1132c6ccd4f:core.Destroy.Core"(ptr %.loc28_21.1.temp), !dbg !63
// CHECK:STDOUT:   ret void, !dbg !65
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.2ec8e1132c6ccd4f:core.Destroy.Core"(ptr %self) #0 !dbg !66 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !68
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvertBoolToUnformedBool.Main(ptr %b) #0 !dbg !70 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CTakeBool.Main(ptr %b), !dbg !75
// CHECK:STDOUT:   ret void, !dbg !76
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTakeAdapterI32.Main(i32 %_) #0 !dbg !78 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !81
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @_CMakeAdapterI32.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassAdapterI32.Main() #0 !dbg !83 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc48_33.1.temp = alloca i32, align 4, !dbg !84
// CHECK:STDOUT:   %MakeAdapterI32.call = call i32 @_CMakeAdapterI32.Main(), !dbg !84
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc48_33.1.temp), !dbg !84
// CHECK:STDOUT:   store i32 %MakeAdapterI32.call, ptr %.loc48_33.1.temp, align 4, !dbg !84
// CHECK:STDOUT:   %.loc48_33.3 = load i32, ptr %.loc48_33.1.temp, align 4, !dbg !84
// CHECK:STDOUT:   call void @_CTakeAdapterI32.Main(i32 %.loc48_33.3), !dbg !85
// CHECK:STDOUT:   call void @"_CSelfDestruct.93f97fc21cfafcdb:core.Destroy.Core"(ptr %.loc48_33.1.temp), !dbg !84
// CHECK:STDOUT:   ret void, !dbg !86
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.d19d32d3b3a297a1:core.Destroy.Core"(ptr %self) #0 !dbg !87 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !89
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.1fff740ae901e27a:core.Destroy.Core"(ptr %self) #0 !dbg !91 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !93
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.1fff740ae901e27a:core.Destroy.Core"(ptr %self) #0 !dbg !95 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.1fff740ae901e27a:core.Destroy.Core"(ptr %self), !dbg !97
// CHECK:STDOUT:   ret void, !dbg !97
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.93f97fc21cfafcdb:core.Destroy.Core"(ptr %self) #0 !dbg !99 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !101
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTakeAdapterBool.Main(ptr %_) #0 !dbg !103 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !106
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CMakeAdapterBool.Main(ptr sret(i1))
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassAdapterBool.Main() #0 !dbg !108 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc55_35.1.temp = alloca i1, align 1, !dbg !109
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc55_35.1.temp), !dbg !109
// CHECK:STDOUT:   call void @_CMakeAdapterBool.Main(ptr %.loc55_35.1.temp), !dbg !109
// CHECK:STDOUT:   call void @_CTakeAdapterBool.Main(ptr %.loc55_35.1.temp), !dbg !110
// CHECK:STDOUT:   call void @"_CSelfDestruct.2d8e1c00e1772e85:core.Destroy.Core"(ptr %.loc55_35.1.temp), !dbg !109
// CHECK:STDOUT:   ret void, !dbg !111
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.2d8e1c00e1772e85:core.Destroy.Core"(ptr %self) #0 !dbg !112 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !114
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!35}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !34 = !DIFile(filename: "maybe_unformed.carbon", directory: "")
// CHECK:STDOUT: !35 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !34, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !36 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !37 = !{null, !36}
// CHECK:STDOUT: !38 = !DISubroutineType(types: !37)
// CHECK:STDOUT: !39 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !42 = !{null, !39}
// CHECK:STDOUT: !43 = !DISubroutineType(types: !42)
// CHECK:STDOUT: !44 = !{null}
// CHECK:STDOUT: !45 = !DISubroutineType(types: !44)
// CHECK:STDOUT: !48 = distinct !DISubprogram(name: "TakeI32", linkageName: "_CTakeI32.Main", scope: null, file: !34, line: 15, type: !38, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !52)
// CHECK:STDOUT: !49 = !DILocalVariable(arg: 1, scope: !48, type: !36)
// CHECK:STDOUT: !51 = !DILocation(line: 15, column: 1, scope: !48)
// CHECK:STDOUT: !52 = !{!49}
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "TakeBool", linkageName: "_CTakeBool.Main", scope: null, file: !34, line: 18, type: !43, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !57)
// CHECK:STDOUT: !54 = !DILocalVariable(arg: 1, scope: !53, type: !39)
// CHECK:STDOUT: !56 = !DILocation(line: 18, column: 1, scope: !53)
// CHECK:STDOUT: !57 = !{!54}
// CHECK:STDOUT: !58 = distinct !DISubprogram(name: "PassI32", linkageName: "_CPassI32.Main", scope: null, file: !34, line: 21, type: !45, spFlags: DISPFlagDefinition, unit: !35)
// CHECK:STDOUT: !59 = !DILocation(line: 23, column: 11, scope: !58)
// CHECK:STDOUT: !60 = !DILocation(line: 23, column: 3, scope: !58)
// CHECK:STDOUT: !61 = !DILocation(line: 21, column: 1, scope: !58)
// CHECK:STDOUT: !62 = distinct !DISubprogram(name: "PassBool", linkageName: "_CPassBool.Main", scope: null, file: !34, line: 26, type: !45, spFlags: DISPFlagDefinition, unit: !35)
// CHECK:STDOUT: !63 = !DILocation(line: 28, column: 12, scope: !62)
// CHECK:STDOUT: !64 = !DILocation(line: 28, column: 3, scope: !62)
// CHECK:STDOUT: !65 = !DILocation(line: 26, column: 1, scope: !62)
// CHECK:STDOUT: !66 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.2ec8e1132c6ccd4f:core.Destroy.Core", scope: null, file: !34, line: 28, type: !43, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !69)
// CHECK:STDOUT: !67 = !DILocalVariable(arg: 1, scope: !66, type: !39)
// CHECK:STDOUT: !68 = !DILocation(line: 28, column: 12, scope: !66)
// CHECK:STDOUT: !69 = !{!67}
// CHECK:STDOUT: !70 = distinct !DISubprogram(name: "ConvertBoolToUnformedBool", linkageName: "_CConvertBoolToUnformedBool.Main", scope: null, file: !34, line: 31, type: !43, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !77)
// CHECK:STDOUT: !71 = !DILocalVariable(arg: 1, scope: !70, type: !39)
// CHECK:STDOUT: !75 = !DILocation(line: 32, column: 3, scope: !70)
// CHECK:STDOUT: !76 = !DILocation(line: 31, column: 1, scope: !70)
// CHECK:STDOUT: !77 = !{!71}
// CHECK:STDOUT: !78 = distinct !DISubprogram(name: "TakeAdapterI32", linkageName: "_CTakeAdapterI32.Main", scope: null, file: !34, line: 42, type: !38, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !82)
// CHECK:STDOUT: !79 = !DILocalVariable(arg: 1, scope: !78, type: !36)
// CHECK:STDOUT: !81 = !DILocation(line: 42, column: 1, scope: !78)
// CHECK:STDOUT: !82 = !{!79}
// CHECK:STDOUT: !83 = distinct !DISubprogram(name: "PassAdapterI32", linkageName: "_CPassAdapterI32.Main", scope: null, file: !34, line: 45, type: !45, spFlags: DISPFlagDefinition, unit: !35)
// CHECK:STDOUT: !84 = !DILocation(line: 48, column: 18, scope: !83)
// CHECK:STDOUT: !85 = !DILocation(line: 48, column: 3, scope: !83)
// CHECK:STDOUT: !86 = !DILocation(line: 45, column: 1, scope: !83)
// CHECK:STDOUT: !87 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.d19d32d3b3a297a1:core.Destroy.Core", scope: null, file: !34, line: 48, type: !38, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !90)
// CHECK:STDOUT: !88 = !DILocalVariable(arg: 1, scope: !87, type: !36)
// CHECK:STDOUT: !89 = !DILocation(line: 48, column: 18, scope: !87)
// CHECK:STDOUT: !90 = !{!88}
// CHECK:STDOUT: !91 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.1fff740ae901e27a:core.Destroy.Core", scope: null, file: !34, line: 48, type: !43, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !94)
// CHECK:STDOUT: !92 = !DILocalVariable(arg: 1, scope: !91, type: !39)
// CHECK:STDOUT: !93 = !DILocation(line: 48, column: 18, scope: !91)
// CHECK:STDOUT: !94 = !{!92}
// CHECK:STDOUT: !95 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.1fff740ae901e27a:core.Destroy.Core", scope: null, file: !34, line: 48, type: !43, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !98)
// CHECK:STDOUT: !96 = !DILocalVariable(arg: 1, scope: !95, type: !39)
// CHECK:STDOUT: !97 = !DILocation(line: 48, column: 18, scope: !95)
// CHECK:STDOUT: !98 = !{!96}
// CHECK:STDOUT: !99 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.93f97fc21cfafcdb:core.Destroy.Core", scope: null, file: !34, line: 48, type: !38, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !102)
// CHECK:STDOUT: !100 = !DILocalVariable(arg: 1, scope: !99, type: !36)
// CHECK:STDOUT: !101 = !DILocation(line: 48, column: 18, scope: !99)
// CHECK:STDOUT: !102 = !{!100}
// CHECK:STDOUT: !103 = distinct !DISubprogram(name: "TakeAdapterBool", linkageName: "_CTakeAdapterBool.Main", scope: null, file: !34, line: 51, type: !43, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !107)
// CHECK:STDOUT: !104 = !DILocalVariable(arg: 1, scope: !103, type: !39)
// CHECK:STDOUT: !106 = !DILocation(line: 51, column: 1, scope: !103)
// CHECK:STDOUT: !107 = !{!104}
// CHECK:STDOUT: !108 = distinct !DISubprogram(name: "PassAdapterBool", linkageName: "_CPassAdapterBool.Main", scope: null, file: !34, line: 54, type: !45, spFlags: DISPFlagDefinition, unit: !35)
// CHECK:STDOUT: !109 = !DILocation(line: 55, column: 19, scope: !108)
// CHECK:STDOUT: !110 = !DILocation(line: 55, column: 3, scope: !108)
// CHECK:STDOUT: !111 = !DILocation(line: 54, column: 1, scope: !108)
// CHECK:STDOUT: !112 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.2d8e1c00e1772e85:core.Destroy.Core", scope: null, file: !34, line: 55, type: !43, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !115)
// CHECK:STDOUT: !113 = !DILocalVariable(arg: 1, scope: !112, type: !39)
// CHECK:STDOUT: !114 = !DILocation(line: 55, column: 19, scope: !112)
// CHECK:STDOUT: !115 = !{!113}
// CHECK:STDOUT:
