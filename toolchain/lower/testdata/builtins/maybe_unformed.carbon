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

fn TakeI32(x: MaybeUnformedBuiltin(i32)) {}
fn MakeI32() -> MaybeUnformedBuiltin(i32);

fn TakeBool(x: MaybeUnformedBuiltin(bool)) {}
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

fn TakeAdapterI32(x: MaybeUnformedAdapter(i32)) {}
fn MakeAdapterI32() -> MaybeUnformedAdapter(i32);

fn PassAdapterI32() {
  // TODO: The `value_of_initializer` optimization doesn't get used here, so we
  // unnecessarily round-trip this value through memory.
  TakeAdapterI32(MakeAdapterI32());
}

fn TakeAdapterBool(x: MaybeUnformedAdapter(bool)) {}
fn MakeAdapterBool() -> MaybeUnformedAdapter(bool);

fn PassAdapterBool() {
  TakeAdapterBool(MakeAdapterBool());
}

// CHECK:STDOUT: ; ModuleID = 'maybe_unformed.carbon'
// CHECK:STDOUT: source_filename = "maybe_unformed.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTakeI32.Main(i32 %x) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @_CMakeI32.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTakeBool.Main(ptr %x) #0 !dbg !8 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CMakeBool.Main(ptr sret(i1))
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassI32.Main() #0 !dbg !10 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %MakeI32.call = call i32 @_CMakeI32.Main(), !dbg !11
// CHECK:STDOUT:   call void @_CTakeI32.Main(i32 %MakeI32.call), !dbg !12
// CHECK:STDOUT:   ret void, !dbg !13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassBool.Main() #0 !dbg !14 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc28_21.1.temp = alloca i1, align 1, !dbg !15
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc28_21.1.temp), !dbg !15
// CHECK:STDOUT:   call void @_CMakeBool.Main(ptr %.loc28_21.1.temp), !dbg !15
// CHECK:STDOUT:   call void @_CTakeBool.Main(ptr %.loc28_21.1.temp), !dbg !16
// CHECK:STDOUT:   ret void, !dbg !17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvertBoolToUnformedBool.Main(ptr %b) #0 !dbg !18 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CTakeBool.Main(ptr %b), !dbg !19
// CHECK:STDOUT:   ret void, !dbg !20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTakeAdapterI32.Main(i32 %x) #0 !dbg !21 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @_CMakeAdapterI32.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassAdapterI32.Main() #0 !dbg !23 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc48_33.1.temp = alloca i32, align 4, !dbg !24
// CHECK:STDOUT:   %MakeAdapterI32.call = call i32 @_CMakeAdapterI32.Main(), !dbg !24
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc48_33.1.temp), !dbg !24
// CHECK:STDOUT:   store i32 %MakeAdapterI32.call, ptr %.loc48_33.1.temp, align 4, !dbg !24
// CHECK:STDOUT:   %.loc48_33.3 = load i32, ptr %.loc48_33.1.temp, align 4, !dbg !24
// CHECK:STDOUT:   call void @_CTakeAdapterI32.Main(i32 %.loc48_33.3), !dbg !25
// CHECK:STDOUT:   ret void, !dbg !26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTakeAdapterBool.Main(ptr %x) #0 !dbg !27 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CMakeAdapterBool.Main(ptr sret(i1))
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassAdapterBool.Main() #0 !dbg !29 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc55_35.1.temp = alloca i1, align 1, !dbg !30
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc55_35.1.temp), !dbg !30
// CHECK:STDOUT:   call void @_CMakeAdapterBool.Main(ptr %.loc55_35.1.temp), !dbg !30
// CHECK:STDOUT:   call void @_CTakeAdapterBool.Main(ptr %.loc55_35.1.temp), !dbg !31
// CHECK:STDOUT:   ret void, !dbg !32
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
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "maybe_unformed.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "TakeI32", linkageName: "_CTakeI32.Main", scope: null, file: !3, line: 15, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 15, column: 1, scope: !4)
// CHECK:STDOUT: !8 = distinct !DISubprogram(name: "TakeBool", linkageName: "_CTakeBool.Main", scope: null, file: !3, line: 18, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !9 = !DILocation(line: 18, column: 1, scope: !8)
// CHECK:STDOUT: !10 = distinct !DISubprogram(name: "PassI32", linkageName: "_CPassI32.Main", scope: null, file: !3, line: 21, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !11 = !DILocation(line: 23, column: 11, scope: !10)
// CHECK:STDOUT: !12 = !DILocation(line: 23, column: 3, scope: !10)
// CHECK:STDOUT: !13 = !DILocation(line: 21, column: 1, scope: !10)
// CHECK:STDOUT: !14 = distinct !DISubprogram(name: "PassBool", linkageName: "_CPassBool.Main", scope: null, file: !3, line: 26, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !15 = !DILocation(line: 28, column: 12, scope: !14)
// CHECK:STDOUT: !16 = !DILocation(line: 28, column: 3, scope: !14)
// CHECK:STDOUT: !17 = !DILocation(line: 26, column: 1, scope: !14)
// CHECK:STDOUT: !18 = distinct !DISubprogram(name: "ConvertBoolToUnformedBool", linkageName: "_CConvertBoolToUnformedBool.Main", scope: null, file: !3, line: 31, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !19 = !DILocation(line: 32, column: 3, scope: !18)
// CHECK:STDOUT: !20 = !DILocation(line: 31, column: 1, scope: !18)
// CHECK:STDOUT: !21 = distinct !DISubprogram(name: "TakeAdapterI32", linkageName: "_CTakeAdapterI32.Main", scope: null, file: !3, line: 42, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !22 = !DILocation(line: 42, column: 1, scope: !21)
// CHECK:STDOUT: !23 = distinct !DISubprogram(name: "PassAdapterI32", linkageName: "_CPassAdapterI32.Main", scope: null, file: !3, line: 45, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !24 = !DILocation(line: 48, column: 18, scope: !23)
// CHECK:STDOUT: !25 = !DILocation(line: 48, column: 3, scope: !23)
// CHECK:STDOUT: !26 = !DILocation(line: 45, column: 1, scope: !23)
// CHECK:STDOUT: !27 = distinct !DISubprogram(name: "TakeAdapterBool", linkageName: "_CTakeAdapterBool.Main", scope: null, file: !3, line: 51, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !28 = !DILocation(line: 51, column: 1, scope: !27)
// CHECK:STDOUT: !29 = distinct !DISubprogram(name: "PassAdapterBool", linkageName: "_CPassAdapterBool.Main", scope: null, file: !3, line: 54, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !30 = !DILocation(line: 55, column: 19, scope: !29)
// CHECK:STDOUT: !31 = !DILocation(line: 55, column: 3, scope: !29)
// CHECK:STDOUT: !32 = !DILocation(line: 54, column: 1, scope: !29)
