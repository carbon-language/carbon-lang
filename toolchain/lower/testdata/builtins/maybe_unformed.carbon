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
// CHECK:STDOUT: define void @_CTakeI32.Main(i32 %_) #0 !dbg !46 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !49
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @_CMakeI32.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTakeBool.Main(ptr %_) #0 !dbg !51 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !54
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CMakeBool.Main(ptr sret(i1))
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassI32.Main() #0 !dbg !56 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %MakeI32.call = call i32 @_CMakeI32.Main(), !dbg !57
// CHECK:STDOUT:   call void @_CTakeI32.Main(i32 %MakeI32.call), !dbg !58
// CHECK:STDOUT:   ret void, !dbg !59
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassBool.Main() #0 !dbg !60 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc28_21.1.temp = alloca i1, align 1, !dbg !61
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc28_21.1.temp), !dbg !61
// CHECK:STDOUT:   call void @_CMakeBool.Main(ptr %.loc28_21.1.temp), !dbg !61
// CHECK:STDOUT:   call void @_CTakeBool.Main(ptr %.loc28_21.1.temp), !dbg !62
// CHECK:STDOUT:   call void @"_COp.c4089b4f1697e4a6:core.Destroy.Core"(ptr %.loc28_21.1.temp), !dbg !61
// CHECK:STDOUT:   ret void, !dbg !63
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.c4089b4f1697e4a6:core.Destroy.Core"(ptr %self) #0 !dbg !64 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !66
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvertBoolToUnformedBool.Main(ptr %b) #0 !dbg !68 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CTakeBool.Main(ptr %b), !dbg !73
// CHECK:STDOUT:   ret void, !dbg !74
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTakeAdapterI32.Main(i32 %_) #0 !dbg !76 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !79
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @_CMakeAdapterI32.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassAdapterI32.Main() #0 !dbg !81 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc48_33.1.temp = alloca i32, align 4, !dbg !82
// CHECK:STDOUT:   %MakeAdapterI32.call = call i32 @_CMakeAdapterI32.Main(), !dbg !82
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc48_33.1.temp), !dbg !82
// CHECK:STDOUT:   store i32 %MakeAdapterI32.call, ptr %.loc48_33.1.temp, align 4, !dbg !82
// CHECK:STDOUT:   %.loc48_33.3 = load i32, ptr %.loc48_33.1.temp, align 4, !dbg !82
// CHECK:STDOUT:   call void @_CTakeAdapterI32.Main(i32 %.loc48_33.3), !dbg !83
// CHECK:STDOUT:   call void @"_COp.9d67511b0143461b:core.Destroy.Core"(ptr %.loc48_33.1.temp), !dbg !82
// CHECK:STDOUT:   ret void, !dbg !84
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.70db2d92ffddbef7:core.Destroy.Core"(ptr %self) #0 !dbg !85 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !87
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.18fc07ffd08329ec:core.Destroy.Core"(ptr %self) #0 !dbg !89 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !91
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.9d67511b0143461b:core.Destroy.Core"(ptr %self) #0 !dbg !93 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !95
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTakeAdapterBool.Main(ptr %_) #0 !dbg !97 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !100
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CMakeAdapterBool.Main(ptr sret(i1))
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassAdapterBool.Main() #0 !dbg !102 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc55_35.1.temp = alloca i1, align 1, !dbg !103
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc55_35.1.temp), !dbg !103
// CHECK:STDOUT:   call void @_CMakeAdapterBool.Main(ptr %.loc55_35.1.temp), !dbg !103
// CHECK:STDOUT:   call void @_CTakeAdapterBool.Main(ptr %.loc55_35.1.temp), !dbg !104
// CHECK:STDOUT:   call void @"_COp.7fe843c0b66498d6:core.Destroy.Core"(ptr %.loc55_35.1.temp), !dbg !103
// CHECK:STDOUT:   ret void, !dbg !105
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.7fe843c0b66498d6:core.Destroy.Core"(ptr %self) #0 !dbg !106 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !108
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!33}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !32 = !DIFile(filename: "maybe_unformed.carbon", directory: "")
// CHECK:STDOUT: !33 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !32, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !34 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !35 = !{null, !34}
// CHECK:STDOUT: !36 = !DISubroutineType(types: !35)
// CHECK:STDOUT: !37 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !40 = !{null, !37}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !42 = !{null}
// CHECK:STDOUT: !43 = !DISubroutineType(types: !42)
// CHECK:STDOUT: !46 = distinct !DISubprogram(name: "TakeI32", linkageName: "_CTakeI32.Main", scope: null, file: !32, line: 15, type: !36, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !50)
// CHECK:STDOUT: !47 = !DILocalVariable(arg: 1, scope: !46, type: !34)
// CHECK:STDOUT: !49 = !DILocation(line: 15, column: 1, scope: !46)
// CHECK:STDOUT: !50 = !{!47}
// CHECK:STDOUT: !51 = distinct !DISubprogram(name: "TakeBool", linkageName: "_CTakeBool.Main", scope: null, file: !32, line: 18, type: !41, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !55)
// CHECK:STDOUT: !52 = !DILocalVariable(arg: 1, scope: !51, type: !37)
// CHECK:STDOUT: !54 = !DILocation(line: 18, column: 1, scope: !51)
// CHECK:STDOUT: !55 = !{!52}
// CHECK:STDOUT: !56 = distinct !DISubprogram(name: "PassI32", linkageName: "_CPassI32.Main", scope: null, file: !32, line: 21, type: !43, spFlags: DISPFlagDefinition, unit: !33)
// CHECK:STDOUT: !57 = !DILocation(line: 23, column: 11, scope: !56)
// CHECK:STDOUT: !58 = !DILocation(line: 23, column: 3, scope: !56)
// CHECK:STDOUT: !59 = !DILocation(line: 21, column: 1, scope: !56)
// CHECK:STDOUT: !60 = distinct !DISubprogram(name: "PassBool", linkageName: "_CPassBool.Main", scope: null, file: !32, line: 26, type: !43, spFlags: DISPFlagDefinition, unit: !33)
// CHECK:STDOUT: !61 = !DILocation(line: 28, column: 12, scope: !60)
// CHECK:STDOUT: !62 = !DILocation(line: 28, column: 3, scope: !60)
// CHECK:STDOUT: !63 = !DILocation(line: 26, column: 1, scope: !60)
// CHECK:STDOUT: !64 = distinct !DISubprogram(name: "Op", linkageName: "_COp.c4089b4f1697e4a6:core.Destroy.Core", scope: null, file: !32, line: 28, type: !41, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !67)
// CHECK:STDOUT: !65 = !DILocalVariable(arg: 1, scope: !64, type: !37)
// CHECK:STDOUT: !66 = !DILocation(line: 28, column: 12, scope: !64)
// CHECK:STDOUT: !67 = !{!65}
// CHECK:STDOUT: !68 = distinct !DISubprogram(name: "ConvertBoolToUnformedBool", linkageName: "_CConvertBoolToUnformedBool.Main", scope: null, file: !32, line: 31, type: !41, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !75)
// CHECK:STDOUT: !69 = !DILocalVariable(arg: 1, scope: !68, type: !37)
// CHECK:STDOUT: !73 = !DILocation(line: 32, column: 3, scope: !68)
// CHECK:STDOUT: !74 = !DILocation(line: 31, column: 1, scope: !68)
// CHECK:STDOUT: !75 = !{!69}
// CHECK:STDOUT: !76 = distinct !DISubprogram(name: "TakeAdapterI32", linkageName: "_CTakeAdapterI32.Main", scope: null, file: !32, line: 42, type: !36, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !80)
// CHECK:STDOUT: !77 = !DILocalVariable(arg: 1, scope: !76, type: !34)
// CHECK:STDOUT: !79 = !DILocation(line: 42, column: 1, scope: !76)
// CHECK:STDOUT: !80 = !{!77}
// CHECK:STDOUT: !81 = distinct !DISubprogram(name: "PassAdapterI32", linkageName: "_CPassAdapterI32.Main", scope: null, file: !32, line: 45, type: !43, spFlags: DISPFlagDefinition, unit: !33)
// CHECK:STDOUT: !82 = !DILocation(line: 48, column: 18, scope: !81)
// CHECK:STDOUT: !83 = !DILocation(line: 48, column: 3, scope: !81)
// CHECK:STDOUT: !84 = !DILocation(line: 45, column: 1, scope: !81)
// CHECK:STDOUT: !85 = distinct !DISubprogram(name: "Op", linkageName: "_COp.70db2d92ffddbef7:core.Destroy.Core", scope: null, file: !32, line: 48, type: !36, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !88)
// CHECK:STDOUT: !86 = !DILocalVariable(arg: 1, scope: !85, type: !34)
// CHECK:STDOUT: !87 = !DILocation(line: 48, column: 18, scope: !85)
// CHECK:STDOUT: !88 = !{!86}
// CHECK:STDOUT: !89 = distinct !DISubprogram(name: "Op", linkageName: "_COp.18fc07ffd08329ec:core.Destroy.Core", scope: null, file: !32, line: 48, type: !41, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !92)
// CHECK:STDOUT: !90 = !DILocalVariable(arg: 1, scope: !89, type: !37)
// CHECK:STDOUT: !91 = !DILocation(line: 48, column: 18, scope: !89)
// CHECK:STDOUT: !92 = !{!90}
// CHECK:STDOUT: !93 = distinct !DISubprogram(name: "Op", linkageName: "_COp.9d67511b0143461b:core.Destroy.Core", scope: null, file: !32, line: 48, type: !36, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !96)
// CHECK:STDOUT: !94 = !DILocalVariable(arg: 1, scope: !93, type: !34)
// CHECK:STDOUT: !95 = !DILocation(line: 48, column: 18, scope: !93)
// CHECK:STDOUT: !96 = !{!94}
// CHECK:STDOUT: !97 = distinct !DISubprogram(name: "TakeAdapterBool", linkageName: "_CTakeAdapterBool.Main", scope: null, file: !32, line: 51, type: !41, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !101)
// CHECK:STDOUT: !98 = !DILocalVariable(arg: 1, scope: !97, type: !37)
// CHECK:STDOUT: !100 = !DILocation(line: 51, column: 1, scope: !97)
// CHECK:STDOUT: !101 = !{!98}
// CHECK:STDOUT: !102 = distinct !DISubprogram(name: "PassAdapterBool", linkageName: "_CPassAdapterBool.Main", scope: null, file: !32, line: 54, type: !43, spFlags: DISPFlagDefinition, unit: !33)
// CHECK:STDOUT: !103 = !DILocation(line: 55, column: 19, scope: !102)
// CHECK:STDOUT: !104 = !DILocation(line: 55, column: 3, scope: !102)
// CHECK:STDOUT: !105 = !DILocation(line: 54, column: 1, scope: !102)
// CHECK:STDOUT: !106 = distinct !DISubprogram(name: "Op", linkageName: "_COp.7fe843c0b66498d6:core.Destroy.Core", scope: null, file: !32, line: 55, type: !41, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !109)
// CHECK:STDOUT: !107 = !DILocalVariable(arg: 1, scope: !106, type: !37)
// CHECK:STDOUT: !108 = !DILocation(line: 55, column: 19, scope: !106)
// CHECK:STDOUT: !109 = !{!107}
// CHECK:STDOUT:
