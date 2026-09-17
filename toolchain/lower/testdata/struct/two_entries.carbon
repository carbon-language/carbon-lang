// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/struct/two_entries.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/struct/two_entries.carbon

fn Run() -> i32 {
  var x: {.a: i32, .b: i32} = {.a = 1, .b = 2};
  var _: {.a: i32, .b: i32} = x;
  return 0;
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'two_entries.carbon'
// CHECK:STDOUT: source_filename = "two_entries.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @struct.631.loc14_3.1 = internal constant { i32, i32 } { i32 1, i32 2 }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @main() #0 !dbg !28 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x.var = alloca { i32, i32 }, align 4, !dbg !31
// CHECK:STDOUT:   %_.var = alloca { i32, i32 }, align 4, !dbg !34
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %x.var), !dbg !31
// CHECK:STDOUT:   %.loc14_46.3.a = getelementptr inbounds nuw { i32, i32 }, ptr %x.var, i32 0, i32 0, !dbg !32
// CHECK:STDOUT:   %.loc14_46.6.b = getelementptr inbounds nuw { i32, i32 }, ptr %x.var, i32 0, i32 1, !dbg !32
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %x.var, ptr align 4 @struct.631.loc14_3.1, i64 8, i1 false), !dbg !31
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !34
// CHECK:STDOUT:   %.loc15_31.1.a = getelementptr inbounds nuw { i32, i32 }, ptr %x.var, i32 0, i32 0, !dbg !35
// CHECK:STDOUT:   %.loc15_31.2 = load i32, ptr %.loc15_31.1.a, align 4, !dbg !35
// CHECK:STDOUT:   %.loc15_31.3.a = getelementptr inbounds nuw { i32, i32 }, ptr %_.var, i32 0, i32 0, !dbg !35
// CHECK:STDOUT:   store i32 %.loc15_31.2, ptr %.loc15_31.3.a, align 4, !dbg !35
// CHECK:STDOUT:   %.loc15_31.5.b = getelementptr inbounds nuw { i32, i32 }, ptr %x.var, i32 0, i32 1, !dbg !35
// CHECK:STDOUT:   %.loc15_31.6 = load i32, ptr %.loc15_31.5.b, align 4, !dbg !35
// CHECK:STDOUT:   %.loc15_31.7.b = getelementptr inbounds nuw { i32, i32 }, ptr %_.var, i32 0, i32 1, !dbg !35
// CHECK:STDOUT:   store i32 %.loc15_31.6, ptr %.loc15_31.7.b, align 4, !dbg !35
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.34557f80e8ccb486(ptr %_.var), !dbg !34
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.34557f80e8ccb486(ptr %x.var), !dbg !31
// CHECK:STDOUT:   ret i32 0, !dbg !37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !38 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.b4c762864cd9a02b:core.Destroy.Core"(ptr %self) #0 !dbg !42 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.34557f80e8ccb486(ptr %self) #0 !dbg !46 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.b4c762864cd9a02b:core.Destroy.Core"(ptr %self), !dbg !50
// CHECK:STDOUT:   ret void, !dbg !51
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Destroy.Core.34557f80e8ccb486, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !14 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !18 = !DIFile(filename: "two_entries.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !{!20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = !{null, !20}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !25 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !26 = !{null, !25}
// CHECK:STDOUT: !27 = !DISubroutineType(types: !26)
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !18, line: 13, type: !22, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !31 = !DILocation(line: 14, column: 3, scope: !28)
// CHECK:STDOUT: !32 = !DILocation(line: 14, column: 31, scope: !28)
// CHECK:STDOUT: !34 = !DILocation(line: 15, column: 3, scope: !28)
// CHECK:STDOUT: !35 = !DILocation(line: 15, column: 31, scope: !28)
// CHECK:STDOUT: !37 = !DILocation(line: 16, column: 3, scope: !28)
// CHECK:STDOUT: !38 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core", scope: null, file: !18, line: 15, type: !24, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !41)
// CHECK:STDOUT: !39 = !DILocalVariable(arg: 1, scope: !38, type: !20)
// CHECK:STDOUT: !40 = !DILocation(line: 15, column: 3, scope: !38)
// CHECK:STDOUT: !41 = !{!39}
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.b4c762864cd9a02b:core.Destroy.Core", scope: null, file: !18, line: 15, type: !27, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !45)
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 1, scope: !42, type: !25)
// CHECK:STDOUT: !44 = !DILocation(line: 15, column: 3, scope: !42)
// CHECK:STDOUT: !45 = !{!43}
// CHECK:STDOUT: !46 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.34557f80e8ccb486", scope: null, file: !14, line: 24, type: !27, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !52)
// CHECK:STDOUT: !47 = !DILocalVariable(arg: 1, scope: !46, type: !25)
// CHECK:STDOUT: !50 = !DILocation(line: 26, column: 5, scope: !46)
// CHECK:STDOUT: !51 = !DILocation(line: 24, column: 3, scope: !46)
// CHECK:STDOUT: !52 = !{!47}
// CHECK:STDOUT:
