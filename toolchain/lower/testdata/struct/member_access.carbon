// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/struct/member_access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/struct/member_access.carbon

fn Run() -> i32 {
  var x: {.a: f64, .b: i32} = {.a = 0.0, .b = 1};
  var y: i32 = x.b;
  var _: i32 = y;
  return 0;
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'member_access.carbon'
// CHECK:STDOUT: source_filename = "member_access.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @struct.77f.loc14_3.1 = internal constant <{ double, i32 }> <{ double 0.000000e+00, i32 1 }>
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @main() #0 !dbg !42 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x.var = alloca <{ double, i32 }>, align 8, !dbg !45
// CHECK:STDOUT:   %y.var = alloca i32, align 4, !dbg !48
// CHECK:STDOUT:   %_.var = alloca i32, align 4, !dbg !51
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %x.var), !dbg !45
// CHECK:STDOUT:   %.loc14_48.3.a = getelementptr inbounds nuw <{ double, i32 }>, ptr %x.var, i32 0, i32 0, !dbg !46
// CHECK:STDOUT:   %.loc14_48.6.b = getelementptr inbounds nuw <{ double, i32 }>, ptr %x.var, i32 0, i32 1, !dbg !46
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %x.var, ptr align 8 @struct.77f.loc14_3.1, i64 12, i1 false), !dbg !45
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %y.var), !dbg !48
// CHECK:STDOUT:   %.loc15_17.1.b = getelementptr inbounds nuw <{ double, i32 }>, ptr %x.var, i32 0, i32 1, !dbg !49
// CHECK:STDOUT:   %.loc15_17.2 = load i32, ptr %.loc15_17.1.b, align 4, !dbg !49
// CHECK:STDOUT:   store i32 %.loc15_17.2, ptr %y.var, align 4, !dbg !48
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !51
// CHECK:STDOUT:   %.loc16_16 = load i32, ptr %y.var, align 4, !dbg !52
// CHECK:STDOUT:   store i32 %.loc16_16, ptr %_.var, align 4, !dbg !51
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48(ptr %_.var), !dbg !51
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48(ptr %y.var), !dbg !48
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.e232413dc245a25b(ptr %x.var), !dbg !45
// CHECK:STDOUT:   ret i32 0, !dbg !54
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !55 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !57
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.dc14d4dd21801791:core.Destroy.Core"(ptr %self) #0 !dbg !59 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !61
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.09c9bb256d6f25f6:core.Destroy.Core"(ptr %self) #0 !dbg !63 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !65
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48(ptr %self) #0 !dbg !67 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self), !dbg !71
// CHECK:STDOUT:   ret void, !dbg !72
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.e232413dc245a25b(ptr %self) #0 !dbg !74 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.09c9bb256d6f25f6:core.Destroy.Core"(ptr %self), !dbg !78
// CHECK:STDOUT:   ret void, !dbg !79
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!33}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !28 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !32 = !DIFile(filename: "member_access.carbon", directory: "")
// CHECK:STDOUT: !33 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !32, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !34 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !35 = !{!34}
// CHECK:STDOUT: !36 = !DISubroutineType(types: !35)
// CHECK:STDOUT: !37 = !{null, !34}
// CHECK:STDOUT: !38 = !DISubroutineType(types: !37)
// CHECK:STDOUT: !39 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !40 = !{null, !39}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !32, line: 13, type: !36, spFlags: DISPFlagDefinition, unit: !33)
// CHECK:STDOUT: !45 = !DILocation(line: 14, column: 3, scope: !42)
// CHECK:STDOUT: !46 = !DILocation(line: 14, column: 31, scope: !42)
// CHECK:STDOUT: !48 = !DILocation(line: 15, column: 3, scope: !42)
// CHECK:STDOUT: !49 = !DILocation(line: 15, column: 16, scope: !42)
// CHECK:STDOUT: !51 = !DILocation(line: 16, column: 3, scope: !42)
// CHECK:STDOUT: !52 = !DILocation(line: 16, column: 16, scope: !42)
// CHECK:STDOUT: !54 = !DILocation(line: 17, column: 3, scope: !42)
// CHECK:STDOUT: !55 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core", scope: null, file: !32, line: 16, type: !38, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !58)
// CHECK:STDOUT: !56 = !DILocalVariable(arg: 1, scope: !55, type: !34)
// CHECK:STDOUT: !57 = !DILocation(line: 16, column: 3, scope: !55)
// CHECK:STDOUT: !58 = !{!56}
// CHECK:STDOUT: !59 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.dc14d4dd21801791:core.Destroy.Core", scope: null, file: !32, line: 14, type: !41, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !62)
// CHECK:STDOUT: !60 = !DILocalVariable(arg: 1, scope: !59, type: !39)
// CHECK:STDOUT: !61 = !DILocation(line: 14, column: 3, scope: !59)
// CHECK:STDOUT: !62 = !{!60}
// CHECK:STDOUT: !63 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.09c9bb256d6f25f6:core.Destroy.Core", scope: null, file: !32, line: 14, type: !41, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !66)
// CHECK:STDOUT: !64 = !DILocalVariable(arg: 1, scope: !63, type: !39)
// CHECK:STDOUT: !65 = !DILocation(line: 14, column: 3, scope: !63)
// CHECK:STDOUT: !66 = !{!64}
// CHECK:STDOUT: !67 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48", scope: null, file: !28, line: 24, type: !38, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !73)
// CHECK:STDOUT: !68 = !DILocalVariable(arg: 1, scope: !67, type: !34)
// CHECK:STDOUT: !71 = !DILocation(line: 26, column: 5, scope: !67)
// CHECK:STDOUT: !72 = !DILocation(line: 24, column: 3, scope: !67)
// CHECK:STDOUT: !73 = !{!68}
// CHECK:STDOUT: !74 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.e232413dc245a25b", scope: null, file: !28, line: 24, type: !41, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !80)
// CHECK:STDOUT: !75 = !DILocalVariable(arg: 1, scope: !74, type: !39)
// CHECK:STDOUT: !78 = !DILocation(line: 26, column: 5, scope: !74)
// CHECK:STDOUT: !79 = !DILocation(line: 24, column: 3, scope: !74)
// CHECK:STDOUT: !80 = !{!75}
// CHECK:STDOUT:
