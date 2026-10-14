// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/generic/call_different_associated_const.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/generic/call_different_associated_const.carbon

// Cannot coalesce the lowering for G specifics, as they have associated contants of different types.
// Check different functions are still emitted when trying to deduplicate emitted definitons.

interface I {
  let T: type;
  // TODO: An interface with multiple associated entities is not yet
  // implemented in mangling.
}

fn G(generic U: I) {
  var _: U.T*;
}

class C {}

impl C as I where .T = bool {
}

class D {}
impl D as I where .T = i32 {
}

fn H() {
  G(C);
  G(D);
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'call_different_associated_const.carbon'
// CHECK:STDOUT: source_filename = "call_different_associated_const.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CH.Main() #0 !dbg !38 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CG.Main.0676fa98bb7dbd66(), !dbg !39
// CHECK:STDOUT:   call void @_CG.Main.3e979ea3a37f4bf1(), !dbg !40
// CHECK:STDOUT:   ret void, !dbg !41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CG.Main.0676fa98bb7dbd66() #0 !dbg !42 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca ptr, align 8, !dbg !43
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !43
// CHECK:STDOUT:   store ptr poison, ptr %_.var, align 8, !dbg !43
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.4055ca6f11c20235(ptr %_.var), !dbg !43
// CHECK:STDOUT:   ret void, !dbg !48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CG.Main.3e979ea3a37f4bf1() #0 !dbg !49 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca ptr, align 8, !dbg !50
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !50
// CHECK:STDOUT:   store ptr poison, ptr %_.var, align 8, !dbg !50
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.39f93332b5bec0ba(ptr %_.var), !dbg !50
// CHECK:STDOUT:   ret void, !dbg !52
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.4055ca6f11c20235(ptr %t) #0 !dbg !53 {
// CHECK:STDOUT:   ret void, !dbg !58
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.39f93332b5bec0ba(ptr %t) #0 !dbg !60 {
// CHECK:STDOUT:   ret void, !dbg !65
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!35}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !28 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !34 = !DIFile(filename: "call_different_associated_const.carbon", directory: "")
// CHECK:STDOUT: !35 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !34, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !36 = !{null}
// CHECK:STDOUT: !37 = !DISubroutineType(types: !36)
// CHECK:STDOUT: !38 = distinct !DISubprogram(name: "H", linkageName: "_CH.Main", scope: null, file: !34, line: 35, type: !37, spFlags: DISPFlagDefinition, unit: !35)
// CHECK:STDOUT: !39 = !DILocation(line: 36, column: 3, scope: !38)
// CHECK:STDOUT: !40 = !DILocation(line: 37, column: 3, scope: !38)
// CHECK:STDOUT: !41 = !DILocation(line: 35, column: 1, scope: !38)
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.0676fa98bb7dbd66", scope: null, file: !34, line: 22, type: !37, spFlags: DISPFlagDefinition, unit: !35)
// CHECK:STDOUT: !43 = !DILocation(line: 23, column: 3, scope: !42)
// CHECK:STDOUT: !45 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !46 = !{null, !45}
// CHECK:STDOUT: !47 = !DISubroutineType(types: !46)
// CHECK:STDOUT: !48 = !DILocation(line: 22, column: 1, scope: !42)
// CHECK:STDOUT: !49 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.3e979ea3a37f4bf1", scope: null, file: !34, line: 22, type: !37, spFlags: DISPFlagDefinition, unit: !35)
// CHECK:STDOUT: !50 = !DILocation(line: 23, column: 3, scope: !49)
// CHECK:STDOUT: !52 = !DILocation(line: 22, column: 1, scope: !49)
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.4055ca6f11c20235", scope: null, file: !28, line: 30, type: !47, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !59)
// CHECK:STDOUT: !54 = !DILocalVariable(arg: 1, scope: !53, type: !45)
// CHECK:STDOUT: !58 = !DILocation(line: 30, column: 1, scope: !53)
// CHECK:STDOUT: !59 = !{!54}
// CHECK:STDOUT: !60 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.39f93332b5bec0ba", scope: null, file: !28, line: 30, type: !47, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !66)
// CHECK:STDOUT: !61 = !DILocalVariable(arg: 1, scope: !60, type: !45)
// CHECK:STDOUT: !65 = !DILocation(line: 30, column: 1, scope: !60)
// CHECK:STDOUT: !66 = !{!61}
// CHECK:STDOUT:
