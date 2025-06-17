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

// CHECK:STDOUT: ; ModuleID = 'simple.carbon'
// CHECK:STDOUT: source_filename = "simple.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @_Ca.Main = global i32 0
// CHECK:STDOUT: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 0, ptr @_C__global_init.Main, ptr null }]
// CHECK:STDOUT:
// CHECK:STDOUT: define ptr @_CGetA.Main() !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret ptr @_Ca.Main, !dbg !7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_C__global_init.Main() !dbg !8 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   store i32 1, ptr @_Ca.Main, align 4, !dbg !9
// CHECK:STDOUT:   ret void, !dbg !10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "simple.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "GetA", linkageName: "_CGetA.Main", scope: null, file: !3, line: 5, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 5, column: 21, scope: !4)
// CHECK:STDOUT: !8 = distinct !DISubprogram(name: "__global_init", linkageName: "_C__global_init.Main", scope: null, file: !3, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !9 = !DILocation(line: 4, column: 1, scope: !8)
// CHECK:STDOUT: !10 = !DILocation(line: 0, scope: !8)
// CHECK:STDOUT: ; ModuleID = 'var_in_tuple.carbon'
// CHECK:STDOUT: source_filename = "var_in_tuple.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @_Cb.Main = global i32 0
// CHECK:STDOUT: @_Cc.Main = global i32 0
// CHECK:STDOUT: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 0, ptr @_C__global_init.Main, ptr null }]
// CHECK:STDOUT:
// CHECK:STDOUT: define ptr @_CGetB.Main() !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret ptr @_Cb.Main, !dbg !7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define ptr @_CGetC.Main() !dbg !8 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret ptr @_Cc.Main, !dbg !9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_C__global_init.Main() !dbg !10 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   store i32 2, ptr @_Cb.Main, align 4, !dbg !11
// CHECK:STDOUT:   store i32 3, ptr @_Cc.Main, align 4, !dbg !12
// CHECK:STDOUT:   ret void, !dbg !13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "var_in_tuple.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "GetB", linkageName: "_CGetB.Main", scope: null, file: !3, line: 6, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 6, column: 21, scope: !4)
// CHECK:STDOUT: !8 = distinct !DISubprogram(name: "GetC", linkageName: "_CGetC.Main", scope: null, file: !3, line: 7, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !9 = !DILocation(line: 7, column: 21, scope: !8)
// CHECK:STDOUT: !10 = distinct !DISubprogram(name: "__global_init", linkageName: "_C__global_init.Main", scope: null, file: !3, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !11 = !DILocation(line: 4, column: 6, scope: !10)
// CHECK:STDOUT: !12 = !DILocation(line: 4, column: 18, scope: !10)
// CHECK:STDOUT: !13 = !DILocation(line: 0, scope: !10)
// CHECK:STDOUT: ; ModuleID = 'tuple_in_var.carbon'
// CHECK:STDOUT: source_filename = "tuple_in_var.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @_Cd.Main = global { i32, i32 } zeroinitializer
// CHECK:STDOUT: @tuple.loc4_1 = internal constant { i32, i32 } { i32 4, i32 5 }
// CHECK:STDOUT: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 0, ptr @_C__global_init.Main, ptr null }]
// CHECK:STDOUT:
// CHECK:STDOUT: define ptr @_CGetD.Main() !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret ptr @_Cd.Main, !dbg !7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define ptr @_CGetE.Main() !dbg !8 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret ptr getelementptr inbounds nuw ({ i32, i32 }, ptr @_Cd.Main, i32 0, i32 1), !dbg !9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_C__global_init.Main() !dbg !10 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 @_Cd.Main, ptr align 4 @tuple.loc4_1, i64 8, i1 false), !dbg !11
// CHECK:STDOUT:   ret void, !dbg !12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #0
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_Cd.Main, { 0, 2, 1 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "tuple_in_var.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "GetD", linkageName: "_CGetD.Main", scope: null, file: !3, line: 6, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 6, column: 21, scope: !4)
// CHECK:STDOUT: !8 = distinct !DISubprogram(name: "GetE", linkageName: "_CGetE.Main", scope: null, file: !3, line: 7, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !9 = !DILocation(line: 7, column: 21, scope: !8)
// CHECK:STDOUT: !10 = distinct !DISubprogram(name: "__global_init", linkageName: "_C__global_init.Main", scope: null, file: !3, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !11 = !DILocation(line: 4, column: 1, scope: !10)
// CHECK:STDOUT: !12 = !DILocation(line: 0, scope: !10)
// CHECK:STDOUT: ; ModuleID = 'unnamed.carbon'
// CHECK:STDOUT: source_filename = "unnamed.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @_Cx.Main = global { i32, i32 } zeroinitializer
// CHECK:STDOUT: @var.anon.var_patt.loc5 = internal global { i32, i32 } zeroinitializer
// CHECK:STDOUT: @tuple.21c.loc4_1 = internal constant { i32, i32 } { i32 1, i32 2 }
// CHECK:STDOUT: @tuple.ffd.loc5_1 = internal constant { i32, i32 } { i32 3, i32 4 }
// CHECK:STDOUT: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 0, ptr @_C__global_init.Main, ptr null }]
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_C__global_init.Main() !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 @_Cx.Main, ptr align 4 @tuple.21c.loc4_1, i64 8, i1 false), !dbg !7
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 @var.anon.var_patt.loc5, ptr align 4 @tuple.ffd.loc5_1, i64 8, i1 false), !dbg !8
// CHECK:STDOUT:   ret void, !dbg !9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #0
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "unnamed.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "__global_init", linkageName: "_C__global_init.Main", scope: null, file: !3, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 4, column: 1, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 5, column: 1, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 0, scope: !4)
