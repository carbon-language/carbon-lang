// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/struct/one_entry.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/struct/one_entry.carbon

fn Run() -> i32 {
  var x: {.a: i32} = {.a = 4};
  var _: {.a: i32} = x;
  return 0;
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'one_entry.carbon'
// CHECK:STDOUT: source_filename = "one_entry.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @main() #0 !dbg !26 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x.var = alloca { i32 }, align 4, !dbg !29
// CHECK:STDOUT:   %_.var = alloca { i32 }, align 4, !dbg !31
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %x.var), !dbg !29
// CHECK:STDOUT:   store { i32 } { i32 4 }, ptr %x.var, align 4, !dbg !29
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !31
// CHECK:STDOUT:   %.loc15_22.1.a = getelementptr inbounds nuw { i32 }, ptr %x.var, i32 0, i32 0, !dbg !32
// CHECK:STDOUT:   %.loc15_22.2 = load i32, ptr %.loc15_22.1.a, align 4, !dbg !32
// CHECK:STDOUT:   %.loc15_22.3.struct.init = insertvalue { i32 } poison, i32 %.loc15_22.2, 0, !dbg !32
// CHECK:STDOUT:   store { i32 } %.loc15_22.3.struct.init, ptr %_.var, align 4, !dbg !31
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.53f7422607519096(ptr %_.var), !dbg !31
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.53f7422607519096(ptr %x.var), !dbg !29
// CHECK:STDOUT:   ret i32 0, !dbg !34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.a9280e8bea6fc5f9:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !35 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc15_3.1.a = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !37
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.3b18dc1d3d647932(ptr %.loc15_3.1.a), !dbg !37
// CHECK:STDOUT:   ret void, !dbg !37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.53f7422607519096(ptr %t) #0 !dbg !41 {
// CHECK:STDOUT:   call void @"_COp.a9280e8bea6fc5f9:core.SubobjectDestroy.Core"(ptr %t), !dbg !45
// CHECK:STDOUT:   ret void, !dbg !46
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.3b18dc1d3d647932(ptr %t) #0 !dbg !48 {
// CHECK:STDOUT:   ret void, !dbg !53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.53f7422607519096, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !18 = !DIFile(filename: "one_entry.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !{!20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{null, !23}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !18, line: 13, type: !22, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !29 = !DILocation(line: 14, column: 3, scope: !26)
// CHECK:STDOUT: !31 = !DILocation(line: 15, column: 3, scope: !26)
// CHECK:STDOUT: !32 = !DILocation(line: 15, column: 22, scope: !26)
// CHECK:STDOUT: !34 = !DILocation(line: 16, column: 3, scope: !26)
// CHECK:STDOUT: !35 = distinct !DISubprogram(name: "Op", linkageName: "_COp.a9280e8bea6fc5f9:core.SubobjectDestroy.Core", scope: null, file: !18, line: 15, type: !25, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !40)
// CHECK:STDOUT: !36 = !DILocalVariable(arg: 1, scope: !35, type: !23)
// CHECK:STDOUT: !37 = !DILocation(line: 15, column: 3, scope: !35)
// CHECK:STDOUT: !38 = !{null, !20}
// CHECK:STDOUT: !39 = !DISubroutineType(types: !38)
// CHECK:STDOUT: !40 = !{!36}
// CHECK:STDOUT: !41 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.53f7422607519096", scope: null, file: !10, line: 26, type: !25, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !47)
// CHECK:STDOUT: !42 = !DILocalVariable(arg: 1, scope: !41, type: !23)
// CHECK:STDOUT: !45 = !DILocation(line: 28, column: 5, scope: !41)
// CHECK:STDOUT: !46 = !DILocation(line: 26, column: 1, scope: !41)
// CHECK:STDOUT: !47 = !{!42}
// CHECK:STDOUT: !48 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.3b18dc1d3d647932", scope: null, file: !10, line: 26, type: !39, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !54)
// CHECK:STDOUT: !49 = !DILocalVariable(arg: 1, scope: !48, type: !20)
// CHECK:STDOUT: !53 = !DILocation(line: 26, column: 1, scope: !48)
// CHECK:STDOUT: !54 = !{!49}
// CHECK:STDOUT:
