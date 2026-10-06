// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/var/global.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/var/global.carbon

// TODO: We should initialize all of these variables statically rather than on startup.

// --- simple.carbon

library "[[@TEST_NAME]]";

var a: i32 = 1;
fn GetA() -> i32* { return &a; }

// --- var_in_tuple.carbon

library "[[@TEST_NAME]]";

let (var b: i32, var c: i32) = (2, 3);

fn GetB() -> i32* { return &b; }
fn GetC() -> i32* { return &c; }

// --- tuple_in_var.carbon

library "[[@TEST_NAME]]";

var (d: i32, e: i32) = (4, 5);

fn GetD() -> i32* { return &d; }
fn GetE() -> i32* { return &e; }

// --- unnamed.carbon

library "[[@TEST_NAME]]";

var (_: i32, x: i32) = (1, 2);
var (_: i32, _: i32) = (3, 4);

// --- not_main_package.carbon

package Other;

var a: i32 = 1;

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'simple.carbon'
// CHECK:STDOUT: source_filename = "simple.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @_Ca.Main = global i32 0
// CHECK:STDOUT: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 0, ptr @_C__global_init.Main, ptr null }]
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CGetA.Main() #0 !dbg !24 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret ptr @_Ca.Main, !dbg !27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define internal void @_C__global_init.Main() #0 !dbg !30 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   store i32 1, ptr @_Ca.Main, align 4, !dbg !31
// CHECK:STDOUT:   ret void, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !18 = !DIFile(filename: "simple.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{!21}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !24 = distinct !DISubprogram(name: "GetA", linkageName: "_CGetA.Main", scope: null, file: !18, line: 5, type: !23, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !27 = !DILocation(line: 5, column: 21, scope: !24)
// CHECK:STDOUT: !28 = !{null}
// CHECK:STDOUT: !29 = !DISubroutineType(types: !28)
// CHECK:STDOUT: !30 = distinct !DISubprogram(name: "__global_init", linkageName: "_C__global_init.Main", scope: null, file: !18, type: !29, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !31 = !DILocation(line: 4, column: 1, scope: !30)
// CHECK:STDOUT: !32 = !DILocation(line: 0, scope: !30)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'var_in_tuple.carbon'
// CHECK:STDOUT: source_filename = "var_in_tuple.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @_Cb.Main = global i32 0
// CHECK:STDOUT: @_Cc.Main = global i32 0
// CHECK:STDOUT: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 0, ptr @_C__global_init.Main, ptr null }]
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CGetB.Main() #0 !dbg !35 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret ptr @_Cb.Main, !dbg !38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CGetC.Main() #0 !dbg !39 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret ptr @_Cc.Main, !dbg !42
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define internal void @_C__global_init.Main() #0 !dbg !43 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   store i32 2, ptr @_Cb.Main, align 4, !dbg !44
// CHECK:STDOUT:   store i32 3, ptr @_Cc.Main, align 4, !dbg !45
// CHECK:STDOUT:   ret void, !dbg !46
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!34}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{!21}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !28 = !{null}
// CHECK:STDOUT: !29 = !DISubroutineType(types: !28)
// CHECK:STDOUT: !33 = !DIFile(filename: "var_in_tuple.carbon", directory: "")
// CHECK:STDOUT: !34 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !33, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !35 = distinct !DISubprogram(name: "GetB", linkageName: "_CGetB.Main", scope: null, file: !33, line: 6, type: !23, spFlags: DISPFlagDefinition, unit: !34)
// CHECK:STDOUT: !38 = !DILocation(line: 6, column: 21, scope: !35)
// CHECK:STDOUT: !39 = distinct !DISubprogram(name: "GetC", linkageName: "_CGetC.Main", scope: null, file: !33, line: 7, type: !23, spFlags: DISPFlagDefinition, unit: !34)
// CHECK:STDOUT: !42 = !DILocation(line: 7, column: 21, scope: !39)
// CHECK:STDOUT: !43 = distinct !DISubprogram(name: "__global_init", linkageName: "_C__global_init.Main", scope: null, file: !33, type: !29, spFlags: DISPFlagDefinition, unit: !34)
// CHECK:STDOUT: !44 = !DILocation(line: 4, column: 6, scope: !43)
// CHECK:STDOUT: !45 = !DILocation(line: 4, column: 18, scope: !43)
// CHECK:STDOUT: !46 = !DILocation(line: 0, scope: !43)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'tuple_in_var.carbon'
// CHECK:STDOUT: source_filename = "tuple_in_var.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @_Cd.Main = global { i32, i32 } zeroinitializer
// CHECK:STDOUT: @tuple.8a1.loc4_1 = internal constant { i32, i32 } { i32 4, i32 5 }
// CHECK:STDOUT: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 0, ptr @_C__global_init.Main, ptr null }]
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CGetD.Main() #0 !dbg !49 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret ptr @_Cd.Main, !dbg !52
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CGetE.Main() #0 !dbg !53 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret ptr getelementptr inbounds nuw ({ i32, i32 }, ptr @_Cd.Main, i32 0, i32 1), !dbg !56
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define internal void @_C__global_init.Main() #0 !dbg !57 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 @_Cd.Main, ptr align 4 @tuple.8a1.loc4_1, i64 8, i1 false), !dbg !58
// CHECK:STDOUT:   ret void, !dbg !59
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_Cd.Main, { 0, 2, 1 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!48}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{!21}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !28 = !{null}
// CHECK:STDOUT: !29 = !DISubroutineType(types: !28)
// CHECK:STDOUT: !47 = !DIFile(filename: "tuple_in_var.carbon", directory: "")
// CHECK:STDOUT: !48 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !47, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !49 = distinct !DISubprogram(name: "GetD", linkageName: "_CGetD.Main", scope: null, file: !47, line: 6, type: !23, spFlags: DISPFlagDefinition, unit: !48)
// CHECK:STDOUT: !52 = !DILocation(line: 6, column: 21, scope: !49)
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "GetE", linkageName: "_CGetE.Main", scope: null, file: !47, line: 7, type: !23, spFlags: DISPFlagDefinition, unit: !48)
// CHECK:STDOUT: !56 = !DILocation(line: 7, column: 21, scope: !53)
// CHECK:STDOUT: !57 = distinct !DISubprogram(name: "__global_init", linkageName: "_C__global_init.Main", scope: null, file: !47, type: !29, spFlags: DISPFlagDefinition, unit: !48)
// CHECK:STDOUT: !58 = !DILocation(line: 4, column: 1, scope: !57)
// CHECK:STDOUT: !59 = !DILocation(line: 0, scope: !57)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'unnamed.carbon'
// CHECK:STDOUT: source_filename = "unnamed.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @_Cx.Main = global { i32, i32 } zeroinitializer
// CHECK:STDOUT: @var.anon.var_patt.loc5 = internal global { i32, i32 } zeroinitializer
// CHECK:STDOUT: @tuple.102.loc4_1 = internal constant { i32, i32 } { i32 1, i32 2 }
// CHECK:STDOUT: @tuple.e16.loc5_1 = internal constant { i32, i32 } { i32 3, i32 4 }
// CHECK:STDOUT: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 0, ptr @_C__global_init.Main, ptr null }]
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define internal void @_C__global_init.Main() #0 !dbg !62 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 @_Cx.Main, ptr align 4 @tuple.102.loc4_1, i64 8, i1 false), !dbg !63
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 @var.anon.var_patt.loc5, ptr align 4 @tuple.e16.loc5_1, i64 8, i1 false), !dbg !65
// CHECK:STDOUT:   ret void, !dbg !66
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!61}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !28 = !{null}
// CHECK:STDOUT: !29 = !DISubroutineType(types: !28)
// CHECK:STDOUT: !60 = !DIFile(filename: "unnamed.carbon", directory: "")
// CHECK:STDOUT: !61 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !60, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !62 = distinct !DISubprogram(name: "__global_init", linkageName: "_C__global_init.Main", scope: null, file: !60, type: !29, spFlags: DISPFlagDefinition, unit: !61)
// CHECK:STDOUT: !63 = !DILocation(line: 4, column: 1, scope: !62)
// CHECK:STDOUT: !65 = !DILocation(line: 5, column: 1, scope: !62)
// CHECK:STDOUT: !66 = !DILocation(line: 0, scope: !62)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'not_main_package.carbon'
// CHECK:STDOUT: source_filename = "not_main_package.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @_Ca.Other = global i32 0
// CHECK:STDOUT: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 0, ptr @_C__global_init.Other, ptr null }]
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define internal void @_C__global_init.Other() #0 !dbg !69 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   store i32 1, ptr @_Ca.Other, align 4, !dbg !70
// CHECK:STDOUT:   ret void, !dbg !71
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!68}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !28 = !{null}
// CHECK:STDOUT: !29 = !DISubroutineType(types: !28)
// CHECK:STDOUT: !67 = !DIFile(filename: "not_main_package.carbon", directory: "")
// CHECK:STDOUT: !68 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !67, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !69 = distinct !DISubprogram(name: "__global_init", linkageName: "_C__global_init.Other", scope: null, file: !67, type: !29, spFlags: DISPFlagDefinition, unit: !68)
// CHECK:STDOUT: !70 = !DILocation(line: 4, column: 1, scope: !69)
// CHECK:STDOUT: !71 = !DILocation(line: 0, scope: !69)
// CHECK:STDOUT:
