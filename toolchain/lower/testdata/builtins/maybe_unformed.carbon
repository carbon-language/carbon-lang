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

class MaybeUnformedAdapter(T:! type) {
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

// CHECK:STDOUT: ; ModuleID = 'maybe_unformed.carbon'
// CHECK:STDOUT: source_filename = "maybe_unformed.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTakeI32.Main(i32 %_) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @_CMakeI32.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTakeBool.Main(ptr %_) #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CMakeBool.Main(ptr sret(i1))
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassI32.Main() #0 !dbg !18 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %MakeI32.call = call i32 @_CMakeI32.Main(), !dbg !21
// CHECK:STDOUT:   call void @_CTakeI32.Main(i32 %MakeI32.call), !dbg !22
// CHECK:STDOUT:   ret void, !dbg !23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassBool.Main() #0 !dbg !24 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc28_21.1.temp = alloca i1, align 1, !dbg !25
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc28_21.1.temp), !dbg !25
// CHECK:STDOUT:   call void @_CMakeBool.Main(ptr %.loc28_21.1.temp), !dbg !25
// CHECK:STDOUT:   call void @_CTakeBool.Main(ptr %.loc28_21.1.temp), !dbg !26
// CHECK:STDOUT:   call void @"_COp.ac1fbb61b7f8ef04:core.Destroy.Core"(ptr %.loc28_21.1.temp), !dbg !25
// CHECK:STDOUT:   ret void, !dbg !27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.ac1fbb61b7f8ef04:core.Destroy.Core"(ptr %self) #0 !dbg !28 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvertBoolToUnformedBool.Main(ptr %b) #0 !dbg !32 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CTakeBool.Main(ptr %b), !dbg !35
// CHECK:STDOUT:   ret void, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTakeAdapterI32.Main(i32 %_) #0 !dbg !37 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @_CMakeAdapterI32.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassAdapterI32.Main() #0 !dbg !41 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc48_33.1.temp = alloca i32, align 4, !dbg !42
// CHECK:STDOUT:   %MakeAdapterI32.call = call i32 @_CMakeAdapterI32.Main(), !dbg !42
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc48_33.1.temp), !dbg !42
// CHECK:STDOUT:   store i32 %MakeAdapterI32.call, ptr %.loc48_33.1.temp, align 4, !dbg !42
// CHECK:STDOUT:   %.loc48_33.3 = load i32, ptr %.loc48_33.1.temp, align 4, !dbg !42
// CHECK:STDOUT:   call void @_CTakeAdapterI32.Main(i32 %.loc48_33.3), !dbg !43
// CHECK:STDOUT:   call void @"_COp.e91169768c6f6a50:core.Destroy.Core"(ptr %.loc48_33.1.temp), !dbg !42
// CHECK:STDOUT:   ret void, !dbg !44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.1fa36cd3e044ac37:core.Destroy.Core"(ptr %self) #0 !dbg !45 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.e91169768c6f6a50:core.Destroy.Core"(ptr %self) #0 !dbg !49 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !52
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTakeAdapterBool.Main(ptr %_) #0 !dbg !53 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !56
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CMakeAdapterBool.Main(ptr sret(i1))
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassAdapterBool.Main() #0 !dbg !57 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc55_35.1.temp = alloca i1, align 1, !dbg !58
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc55_35.1.temp), !dbg !58
// CHECK:STDOUT:   call void @_CMakeAdapterBool.Main(ptr %.loc55_35.1.temp), !dbg !58
// CHECK:STDOUT:   call void @_CTakeAdapterBool.Main(ptr %.loc55_35.1.temp), !dbg !59
// CHECK:STDOUT:   call void @"_COp.968f5dbb300aa798:core.Destroy.Core"(ptr %.loc55_35.1.temp), !dbg !58
// CHECK:STDOUT:   ret void, !dbg !60
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.968f5dbb300aa798:core.Destroy.Core"(ptr %self) #0 !dbg !61 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !64
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
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "maybe_unformed.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "TakeI32", linkageName: "_CTakeI32.Main", scope: null, file: !3, line: 15, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !8)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null, !7}
// CHECK:STDOUT: !7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !8 = !{!9}
// CHECK:STDOUT: !9 = !DILocalVariable(arg: 1, scope: !4, type: !7)
// CHECK:STDOUT: !10 = !DILocation(line: 15, column: 1, scope: !4)
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "TakeBool", linkageName: "_CTakeBool.Main", scope: null, file: !3, line: 18, type: !12, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !15)
// CHECK:STDOUT: !12 = !DISubroutineType(types: !13)
// CHECK:STDOUT: !13 = !{null, !14}
// CHECK:STDOUT: !14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !15 = !{!16}
// CHECK:STDOUT: !16 = !DILocalVariable(arg: 1, scope: !11, type: !14)
// CHECK:STDOUT: !17 = !DILocation(line: 18, column: 1, scope: !11)
// CHECK:STDOUT: !18 = distinct !DISubprogram(name: "PassI32", linkageName: "_CPassI32.Main", scope: null, file: !3, line: 21, type: !19, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !19 = !DISubroutineType(types: !20)
// CHECK:STDOUT: !20 = !{null}
// CHECK:STDOUT: !21 = !DILocation(line: 23, column: 11, scope: !18)
// CHECK:STDOUT: !22 = !DILocation(line: 23, column: 3, scope: !18)
// CHECK:STDOUT: !23 = !DILocation(line: 21, column: 1, scope: !18)
// CHECK:STDOUT: !24 = distinct !DISubprogram(name: "PassBool", linkageName: "_CPassBool.Main", scope: null, file: !3, line: 26, type: !19, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !25 = !DILocation(line: 28, column: 12, scope: !24)
// CHECK:STDOUT: !26 = !DILocation(line: 28, column: 3, scope: !24)
// CHECK:STDOUT: !27 = !DILocation(line: 26, column: 1, scope: !24)
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "Op", linkageName: "_COp.ac1fbb61b7f8ef04:core.Destroy.Core", scope: null, file: !3, line: 28, type: !12, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !29)
// CHECK:STDOUT: !29 = !{!30}
// CHECK:STDOUT: !30 = !DILocalVariable(arg: 1, scope: !28, type: !14)
// CHECK:STDOUT: !31 = !DILocation(line: 28, column: 12, scope: !28)
// CHECK:STDOUT: !32 = distinct !DISubprogram(name: "ConvertBoolToUnformedBool", linkageName: "_CConvertBoolToUnformedBool.Main", scope: null, file: !3, line: 31, type: !12, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !33)
// CHECK:STDOUT: !33 = !{!34}
// CHECK:STDOUT: !34 = !DILocalVariable(arg: 1, scope: !32, type: !14)
// CHECK:STDOUT: !35 = !DILocation(line: 32, column: 3, scope: !32)
// CHECK:STDOUT: !36 = !DILocation(line: 31, column: 1, scope: !32)
// CHECK:STDOUT: !37 = distinct !DISubprogram(name: "TakeAdapterI32", linkageName: "_CTakeAdapterI32.Main", scope: null, file: !3, line: 42, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !38)
// CHECK:STDOUT: !38 = !{!39}
// CHECK:STDOUT: !39 = !DILocalVariable(arg: 1, scope: !37, type: !7)
// CHECK:STDOUT: !40 = !DILocation(line: 42, column: 1, scope: !37)
// CHECK:STDOUT: !41 = distinct !DISubprogram(name: "PassAdapterI32", linkageName: "_CPassAdapterI32.Main", scope: null, file: !3, line: 45, type: !19, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !42 = !DILocation(line: 48, column: 18, scope: !41)
// CHECK:STDOUT: !43 = !DILocation(line: 48, column: 3, scope: !41)
// CHECK:STDOUT: !44 = !DILocation(line: 45, column: 1, scope: !41)
// CHECK:STDOUT: !45 = distinct !DISubprogram(name: "Op", linkageName: "_COp.1fa36cd3e044ac37:core.Destroy.Core", scope: null, file: !3, line: 48, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !46)
// CHECK:STDOUT: !46 = !{!47}
// CHECK:STDOUT: !47 = !DILocalVariable(arg: 1, scope: !45, type: !7)
// CHECK:STDOUT: !48 = !DILocation(line: 48, column: 18, scope: !45)
// CHECK:STDOUT: !49 = distinct !DISubprogram(name: "Op", linkageName: "_COp.e91169768c6f6a50:core.Destroy.Core", scope: null, file: !3, line: 48, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !50)
// CHECK:STDOUT: !50 = !{!51}
// CHECK:STDOUT: !51 = !DILocalVariable(arg: 1, scope: !49, type: !7)
// CHECK:STDOUT: !52 = !DILocation(line: 48, column: 18, scope: !49)
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "TakeAdapterBool", linkageName: "_CTakeAdapterBool.Main", scope: null, file: !3, line: 51, type: !12, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !54)
// CHECK:STDOUT: !54 = !{!55}
// CHECK:STDOUT: !55 = !DILocalVariable(arg: 1, scope: !53, type: !14)
// CHECK:STDOUT: !56 = !DILocation(line: 51, column: 1, scope: !53)
// CHECK:STDOUT: !57 = distinct !DISubprogram(name: "PassAdapterBool", linkageName: "_CPassAdapterBool.Main", scope: null, file: !3, line: 54, type: !19, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !58 = !DILocation(line: 55, column: 19, scope: !57)
// CHECK:STDOUT: !59 = !DILocation(line: 55, column: 3, scope: !57)
// CHECK:STDOUT: !60 = !DILocation(line: 54, column: 1, scope: !57)
// CHECK:STDOUT: !61 = distinct !DISubprogram(name: "Op", linkageName: "_COp.968f5dbb300aa798:core.Destroy.Core", scope: null, file: !3, line: 55, type: !12, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !62)
// CHECK:STDOUT: !62 = !{!63}
// CHECK:STDOUT: !63 = !DILocalVariable(arg: 1, scope: !61, type: !14)
// CHECK:STDOUT: !64 = !DILocation(line: 55, column: 19, scope: !61)
