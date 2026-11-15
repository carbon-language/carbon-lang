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
// CHECK:STDOUT: define ptr @_CGetA.Main() #0 !dbg !26 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret ptr @_Ca.Main, !dbg !29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define internal void @_C__global_init.Main() #0 !dbg !32 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   store i32 1, ptr @_Ca.Main, align 4, !dbg !33
// CHECK:STDOUT:   ret void, !dbg !34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!21}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIFile(filename: "simple.carbon", directory: "")
// CHECK:STDOUT: !21 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !20, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{!23}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "GetA", linkageName: "_CGetA.Main", scope: null, file: !20, line: 5, type: !25, spFlags: DISPFlagDefinition, unit: !21)
// CHECK:STDOUT: !29 = !DILocation(line: 5, column: 21, scope: !26)
// CHECK:STDOUT: !30 = !{null}
// CHECK:STDOUT: !31 = !DISubroutineType(types: !30)
// CHECK:STDOUT: !32 = distinct !DISubprogram(name: "__global_init", linkageName: "_C__global_init.Main", scope: null, file: !20, type: !31, spFlags: DISPFlagDefinition, unit: !21)
// CHECK:STDOUT: !33 = !DILocation(line: 4, column: 1, scope: !32)
// CHECK:STDOUT: !34 = !DILocation(line: 0, scope: !32)
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
// CHECK:STDOUT: define ptr @_CGetB.Main() #0 !dbg !37 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret ptr @_Cb.Main, !dbg !40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CGetC.Main() #0 !dbg !41 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret ptr @_Cc.Main, !dbg !44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define internal void @_C__global_init.Main() #0 !dbg !45 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   store i32 2, ptr @_Cb.Main, align 4, !dbg !46
// CHECK:STDOUT:   store i32 3, ptr @_Cc.Main, align 4, !dbg !47
// CHECK:STDOUT:   ret void, !dbg !48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!36}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{!23}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !30 = !{null}
// CHECK:STDOUT: !31 = !DISubroutineType(types: !30)
// CHECK:STDOUT: !35 = !DIFile(filename: "var_in_tuple.carbon", directory: "")
// CHECK:STDOUT: !36 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !35, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !37 = distinct !DISubprogram(name: "GetB", linkageName: "_CGetB.Main", scope: null, file: !35, line: 6, type: !25, spFlags: DISPFlagDefinition, unit: !36)
// CHECK:STDOUT: !40 = !DILocation(line: 6, column: 21, scope: !37)
// CHECK:STDOUT: !41 = distinct !DISubprogram(name: "GetC", linkageName: "_CGetC.Main", scope: null, file: !35, line: 7, type: !25, spFlags: DISPFlagDefinition, unit: !36)
// CHECK:STDOUT: !44 = !DILocation(line: 7, column: 21, scope: !41)
// CHECK:STDOUT: !45 = distinct !DISubprogram(name: "__global_init", linkageName: "_C__global_init.Main", scope: null, file: !35, type: !31, spFlags: DISPFlagDefinition, unit: !36)
// CHECK:STDOUT: !46 = !DILocation(line: 4, column: 6, scope: !45)
// CHECK:STDOUT: !47 = !DILocation(line: 4, column: 18, scope: !45)
// CHECK:STDOUT: !48 = !DILocation(line: 0, scope: !45)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'tuple_in_var.carbon'
// CHECK:STDOUT: source_filename = "tuple_in_var.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @_Cd.Main = global { i32, i32 } zeroinitializer
// CHECK:STDOUT: @tuple.a84.loc4_1 = internal constant { i32, i32 } { i32 4, i32 5 }
// CHECK:STDOUT: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 0, ptr @_C__global_init.Main, ptr null }]
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CGetD.Main() #0 !dbg !51 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret ptr @_Cd.Main, !dbg !54
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CGetE.Main() #0 !dbg !55 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret ptr getelementptr inbounds nuw (i8, ptr @_Cd.Main, i64 4), !dbg !58
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define internal void @_C__global_init.Main() #0 !dbg !59 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 @_Cd.Main, ptr align 4 @tuple.a84.loc4_1, i64 8, i1 false), !dbg !60
// CHECK:STDOUT:   ret void, !dbg !61
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!50}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{!23}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !30 = !{null}
// CHECK:STDOUT: !31 = !DISubroutineType(types: !30)
// CHECK:STDOUT: !49 = !DIFile(filename: "tuple_in_var.carbon", directory: "")
// CHECK:STDOUT: !50 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !49, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !51 = distinct !DISubprogram(name: "GetD", linkageName: "_CGetD.Main", scope: null, file: !49, line: 6, type: !25, spFlags: DISPFlagDefinition, unit: !50)
// CHECK:STDOUT: !54 = !DILocation(line: 6, column: 21, scope: !51)
// CHECK:STDOUT: !55 = distinct !DISubprogram(name: "GetE", linkageName: "_CGetE.Main", scope: null, file: !49, line: 7, type: !25, spFlags: DISPFlagDefinition, unit: !50)
// CHECK:STDOUT: !58 = !DILocation(line: 7, column: 21, scope: !55)
// CHECK:STDOUT: !59 = distinct !DISubprogram(name: "__global_init", linkageName: "_C__global_init.Main", scope: null, file: !49, type: !31, spFlags: DISPFlagDefinition, unit: !50)
// CHECK:STDOUT: !60 = !DILocation(line: 4, column: 1, scope: !59)
// CHECK:STDOUT: !61 = !DILocation(line: 0, scope: !59)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'unnamed.carbon'
// CHECK:STDOUT: source_filename = "unnamed.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @_Cx.Main = global { i32, i32 } zeroinitializer
// CHECK:STDOUT: @var.anon.var_patt.loc5 = internal global { i32, i32 } zeroinitializer
// CHECK:STDOUT: @tuple.380.loc4_1 = internal constant { i32, i32 } { i32 1, i32 2 }
// CHECK:STDOUT: @tuple.859.loc5_1 = internal constant { i32, i32 } { i32 3, i32 4 }
// CHECK:STDOUT: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 0, ptr @_C__global_init.Main, ptr null }]
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define internal void @_C__global_init.Main() #0 !dbg !64 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 @_Cx.Main, ptr align 4 @tuple.380.loc4_1, i64 8, i1 false), !dbg !65
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 @var.anon.var_patt.loc5, ptr align 4 @tuple.859.loc5_1, i64 8, i1 false), !dbg !67
// CHECK:STDOUT:   ret void, !dbg !70
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!63}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !30 = !{null}
// CHECK:STDOUT: !31 = !DISubroutineType(types: !30)
// CHECK:STDOUT: !62 = !DIFile(filename: "unnamed.carbon", directory: "")
// CHECK:STDOUT: !63 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !62, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !64 = distinct !DISubprogram(name: "__global_init", linkageName: "_C__global_init.Main", scope: null, file: !62, type: !31, spFlags: DISPFlagDefinition, unit: !63)
// CHECK:STDOUT: !65 = !DILocation(line: 4, column: 1, scope: !64)
// CHECK:STDOUT: !67 = !DILocation(line: 5, column: 1, scope: !64)
// CHECK:STDOUT: !70 = !DILocation(line: 0, scope: !64)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'not_main_package.carbon'
// CHECK:STDOUT: source_filename = "not_main_package.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @_Ca.Other = global i32 0
// CHECK:STDOUT: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 0, ptr @_C__global_init.Other, ptr null }]
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define internal void @_C__global_init.Other() #0 !dbg !73 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   store i32 1, ptr @_Ca.Other, align 4, !dbg !74
// CHECK:STDOUT:   ret void, !dbg !75
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!72}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !30 = !{null}
// CHECK:STDOUT: !31 = !DISubroutineType(types: !30)
// CHECK:STDOUT: !71 = !DIFile(filename: "not_main_package.carbon", directory: "")
// CHECK:STDOUT: !72 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !71, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !73 = distinct !DISubprogram(name: "__global_init", linkageName: "_C__global_init.Other", scope: null, file: !71, type: !31, spFlags: DISPFlagDefinition, unit: !72)
// CHECK:STDOUT: !74 = !DILocation(line: 4, column: 1, scope: !73)
// CHECK:STDOUT: !75 = !DILocation(line: 0, scope: !73)
// CHECK:STDOUT:
