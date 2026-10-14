// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/primitives/numeric_literals.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/primitives/numeric_literals.carbon

fn F() {
  // 8 and 9 trigger special behavior in APInt when mishandling signed versus
  // unsigned, so we pay extra attention to those.
  var _: array(i32, 4) = (
    8,
    9,
    0x8,
    0b1000,
  );
  var _: array(f64, 6) = (
    0.9,
    8.0,
    80.0,
    1.0e7,
    1.0e8,
    1.0e-8,
  );
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'numeric_literals.carbon'
// CHECK:STDOUT: source_filename = "numeric_literals.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @array.d08.loc16_3.1 = internal constant [4 x i32] [i32 8, i32 9, i32 8, i32 8]
// CHECK:STDOUT: @array.1d9.loc22_3.1 = internal constant [6 x double] [double 9.000000e-01, double 8.000000e+00, double 8.000000e+01, double 1.000000e+07, double 1.000000e+08, double 1.000000e-08]
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !42 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var.loc16 = alloca [4 x i32], align 4, !dbg !43
// CHECK:STDOUT:   %_.var.loc22 = alloca [6 x double], align 8, !dbg !46
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc16), !dbg !43
// CHECK:STDOUT:   %.loc21_3.3.array.index = getelementptr inbounds [4 x i32], ptr %_.var.loc16, i32 0, i64 0, !dbg !44
// CHECK:STDOUT:   %.loc21_3.6.array.index = getelementptr inbounds [4 x i32], ptr %_.var.loc16, i32 0, i64 1, !dbg !44
// CHECK:STDOUT:   %.loc21_3.9.array.index = getelementptr inbounds [4 x i32], ptr %_.var.loc16, i32 0, i64 2, !dbg !44
// CHECK:STDOUT:   %.loc21_3.12.array.index = getelementptr inbounds [4 x i32], ptr %_.var.loc16, i32 0, i64 3, !dbg !44
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %_.var.loc16, ptr align 4 @array.d08.loc16_3.1, i64 16, i1 false), !dbg !43
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc22), !dbg !46
// CHECK:STDOUT:   %.loc29_3.3.array.index = getelementptr inbounds [6 x double], ptr %_.var.loc22, i32 0, i64 0, !dbg !47
// CHECK:STDOUT:   %.loc29_3.6.array.index = getelementptr inbounds [6 x double], ptr %_.var.loc22, i32 0, i64 1, !dbg !47
// CHECK:STDOUT:   %.loc29_3.9.array.index = getelementptr inbounds [6 x double], ptr %_.var.loc22, i32 0, i64 2, !dbg !47
// CHECK:STDOUT:   %.loc29_3.12.array.index = getelementptr inbounds [6 x double], ptr %_.var.loc22, i32 0, i64 3, !dbg !47
// CHECK:STDOUT:   %.loc29_3.15.array.index = getelementptr inbounds [6 x double], ptr %_.var.loc22, i32 0, i64 4, !dbg !47
// CHECK:STDOUT:   %.loc29_3.18.array.index = getelementptr inbounds [6 x double], ptr %_.var.loc22, i32 0, i64 5, !dbg !47
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %_.var.loc22, ptr align 8 @array.1d9.loc22_3.1, i64 48, i1 false), !dbg !46
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.8abe4517847c7600(ptr %_.var.loc22), !dbg !46
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.38c74a7591f90b8d(ptr %_.var.loc16), !dbg !43
// CHECK:STDOUT:   ret void, !dbg !49
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.7b7aad17e53156ec:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !50 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc22_3.1.array.index = getelementptr inbounds [6 x double], ptr %self, i32 0, i64 5, !dbg !52
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.236ebccb4c02fafd(ptr %.loc22_3.1.array.index), !dbg !52
// CHECK:STDOUT:   %.loc22_3.4.array.index = getelementptr inbounds [6 x double], ptr %self, i32 0, i64 4, !dbg !52
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.236ebccb4c02fafd(ptr %.loc22_3.4.array.index), !dbg !52
// CHECK:STDOUT:   %.loc22_3.7.array.index = getelementptr inbounds [6 x double], ptr %self, i32 0, i64 3, !dbg !52
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.236ebccb4c02fafd(ptr %.loc22_3.7.array.index), !dbg !52
// CHECK:STDOUT:   %.loc22_3.10.array.index = getelementptr inbounds [6 x double], ptr %self, i32 0, i64 2, !dbg !52
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.236ebccb4c02fafd(ptr %.loc22_3.10.array.index), !dbg !52
// CHECK:STDOUT:   %.loc22_3.13.array.index = getelementptr inbounds [6 x double], ptr %self, i32 0, i64 1, !dbg !52
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.236ebccb4c02fafd(ptr %.loc22_3.13.array.index), !dbg !52
// CHECK:STDOUT:   %.loc22_3.16.array.index = getelementptr inbounds [6 x double], ptr %self, i32 0, i64 0, !dbg !52
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.236ebccb4c02fafd(ptr %.loc22_3.16.array.index), !dbg !52
// CHECK:STDOUT:   ret void, !dbg !52
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.88ad3a558ba7d35d:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !54 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc16_3.1.array.index = getelementptr inbounds [4 x i32], ptr %self, i32 0, i64 3, !dbg !56
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.7f8dce37f94a83bf(ptr %.loc16_3.1.array.index), !dbg !56
// CHECK:STDOUT:   %.loc16_3.4.array.index = getelementptr inbounds [4 x i32], ptr %self, i32 0, i64 2, !dbg !56
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.7f8dce37f94a83bf(ptr %.loc16_3.4.array.index), !dbg !56
// CHECK:STDOUT:   %.loc16_3.7.array.index = getelementptr inbounds [4 x i32], ptr %self, i32 0, i64 1, !dbg !56
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.7f8dce37f94a83bf(ptr %.loc16_3.7.array.index), !dbg !56
// CHECK:STDOUT:   %.loc16_3.10.array.index = getelementptr inbounds [4 x i32], ptr %self, i32 0, i64 0, !dbg !56
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.7f8dce37f94a83bf(ptr %.loc16_3.10.array.index), !dbg !56
// CHECK:STDOUT:   ret void, !dbg !56
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.8abe4517847c7600(ptr %t) #0 !dbg !60 {
// CHECK:STDOUT:   call void @"_COp.7b7aad17e53156ec:core.SubobjectDestroy.Core"(ptr %t), !dbg !64
// CHECK:STDOUT:   ret void, !dbg !65
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.38c74a7591f90b8d(ptr %t) #0 !dbg !67 {
// CHECK:STDOUT:   call void @"_COp.88ad3a558ba7d35d:core.SubobjectDestroy.Core"(ptr %t), !dbg !71
// CHECK:STDOUT:   ret void, !dbg !72
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.236ebccb4c02fafd(ptr %t) #0 !dbg !74 {
// CHECK:STDOUT:   ret void, !dbg !79
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.7f8dce37f94a83bf(ptr %t) #0 !dbg !81 {
// CHECK:STDOUT:   ret void, !dbg !86
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.236ebccb4c02fafd, { 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.7f8dce37f94a83bf, { 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!35}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !26 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !34 = !DIFile(filename: "numeric_literals.carbon", directory: "")
// CHECK:STDOUT: !35 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !34, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !36 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !37 = !{null}
// CHECK:STDOUT: !38 = !DISubroutineType(types: !37)
// CHECK:STDOUT: !39 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !40 = !{null, !39}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !34, line: 13, type: !38, spFlags: DISPFlagDefinition, unit: !35)
// CHECK:STDOUT: !43 = !DILocation(line: 16, column: 3, scope: !42)
// CHECK:STDOUT: !44 = !DILocation(line: 16, column: 26, scope: !42)
// CHECK:STDOUT: !46 = !DILocation(line: 22, column: 3, scope: !42)
// CHECK:STDOUT: !47 = !DILocation(line: 22, column: 26, scope: !42)
// CHECK:STDOUT: !49 = !DILocation(line: 13, column: 1, scope: !42)
// CHECK:STDOUT: !50 = distinct !DISubprogram(name: "Op", linkageName: "_COp.7b7aad17e53156ec:core.SubobjectDestroy.Core", scope: null, file: !34, line: 22, type: !41, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !53)
// CHECK:STDOUT: !51 = !DILocalVariable(arg: 1, scope: !50, type: !39)
// CHECK:STDOUT: !52 = !DILocation(line: 22, column: 3, scope: !50)
// CHECK:STDOUT: !53 = !{!51}
// CHECK:STDOUT: !54 = distinct !DISubprogram(name: "Op", linkageName: "_COp.88ad3a558ba7d35d:core.SubobjectDestroy.Core", scope: null, file: !34, line: 16, type: !41, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !59)
// CHECK:STDOUT: !55 = !DILocalVariable(arg: 1, scope: !54, type: !39)
// CHECK:STDOUT: !56 = !DILocation(line: 16, column: 3, scope: !54)
// CHECK:STDOUT: !57 = !{null, !36}
// CHECK:STDOUT: !58 = !DISubroutineType(types: !57)
// CHECK:STDOUT: !59 = !{!55}
// CHECK:STDOUT: !60 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.8abe4517847c7600", scope: null, file: !26, line: 30, type: !41, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !66)
// CHECK:STDOUT: !61 = !DILocalVariable(arg: 1, scope: !60, type: !39)
// CHECK:STDOUT: !64 = !DILocation(line: 32, column: 5, scope: !60)
// CHECK:STDOUT: !65 = !DILocation(line: 30, column: 1, scope: !60)
// CHECK:STDOUT: !66 = !{!61}
// CHECK:STDOUT: !67 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.38c74a7591f90b8d", scope: null, file: !26, line: 30, type: !41, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !73)
// CHECK:STDOUT: !68 = !DILocalVariable(arg: 1, scope: !67, type: !39)
// CHECK:STDOUT: !71 = !DILocation(line: 32, column: 5, scope: !67)
// CHECK:STDOUT: !72 = !DILocation(line: 30, column: 1, scope: !67)
// CHECK:STDOUT: !73 = !{!68}
// CHECK:STDOUT: !74 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.236ebccb4c02fafd", scope: null, file: !26, line: 30, type: !41, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !80)
// CHECK:STDOUT: !75 = !DILocalVariable(arg: 1, scope: !74, type: !39)
// CHECK:STDOUT: !79 = !DILocation(line: 30, column: 1, scope: !74)
// CHECK:STDOUT: !80 = !{!75}
// CHECK:STDOUT: !81 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.7f8dce37f94a83bf", scope: null, file: !26, line: 30, type: !58, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !87)
// CHECK:STDOUT: !82 = !DILocalVariable(arg: 1, scope: !81, type: !36)
// CHECK:STDOUT: !86 = !DILocation(line: 30, column: 1, scope: !81)
// CHECK:STDOUT: !87 = !{!82}
// CHECK:STDOUT:
