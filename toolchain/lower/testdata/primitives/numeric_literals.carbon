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
// CHECK:STDOUT: @array.e93.loc16_3.1 = internal constant [4 x i32] [i32 8, i32 9, i32 8, i32 8]
// CHECK:STDOUT: @array.33a.loc22_3.1 = internal constant [6 x double] [double 9.000000e-01, double 8.000000e+00, double 8.000000e+01, double 1.000000e+07, double 1.000000e+08, double 1.000000e-08]
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !39 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var.loc16 = alloca [4 x i32], align 4, !dbg !40
// CHECK:STDOUT:   %_.var.loc22 = alloca [6 x double], align 8, !dbg !43
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc16), !dbg !40
// CHECK:STDOUT:   %.loc21_3.3.array.index = getelementptr inbounds [4 x i32], ptr %_.var.loc16, i32 0, i64 0, !dbg !41
// CHECK:STDOUT:   %.loc21_3.6.array.index = getelementptr inbounds [4 x i32], ptr %_.var.loc16, i32 0, i64 1, !dbg !41
// CHECK:STDOUT:   %.loc21_3.9.array.index = getelementptr inbounds [4 x i32], ptr %_.var.loc16, i32 0, i64 2, !dbg !41
// CHECK:STDOUT:   %.loc21_3.12.array.index = getelementptr inbounds [4 x i32], ptr %_.var.loc16, i32 0, i64 3, !dbg !41
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %_.var.loc16, ptr align 4 @array.e93.loc16_3.1, i64 16, i1 false), !dbg !40
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc22), !dbg !43
// CHECK:STDOUT:   %.loc29_3.3.array.index = getelementptr inbounds [6 x double], ptr %_.var.loc22, i32 0, i64 0, !dbg !44
// CHECK:STDOUT:   %.loc29_3.6.array.index = getelementptr inbounds [6 x double], ptr %_.var.loc22, i32 0, i64 1, !dbg !44
// CHECK:STDOUT:   %.loc29_3.9.array.index = getelementptr inbounds [6 x double], ptr %_.var.loc22, i32 0, i64 2, !dbg !44
// CHECK:STDOUT:   %.loc29_3.12.array.index = getelementptr inbounds [6 x double], ptr %_.var.loc22, i32 0, i64 3, !dbg !44
// CHECK:STDOUT:   %.loc29_3.15.array.index = getelementptr inbounds [6 x double], ptr %_.var.loc22, i32 0, i64 4, !dbg !44
// CHECK:STDOUT:   %.loc29_3.18.array.index = getelementptr inbounds [6 x double], ptr %_.var.loc22, i32 0, i64 5, !dbg !44
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %_.var.loc22, ptr align 8 @array.33a.loc22_3.1, i64 48, i1 false), !dbg !43
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.b5f30f9c486f7750(ptr %_.var.loc22), !dbg !43
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.8a58b7b15e68238e(ptr %_.var.loc16), !dbg !40
// CHECK:STDOUT:   ret void, !dbg !49
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.b5f30f9c486f7750(ptr %t) #0 !dbg !50 {
// CHECK:STDOUT:   ret void, !dbg !55
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.8a58b7b15e68238e(ptr %t) #0 !dbg !57 {
// CHECK:STDOUT:   ret void, !dbg !62
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 1, 0 }
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
// CHECK:STDOUT: !37 = !{null}
// CHECK:STDOUT: !38 = !DISubroutineType(types: !37)
// CHECK:STDOUT: !39 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !34, line: 13, type: !38, spFlags: DISPFlagDefinition, unit: !35)
// CHECK:STDOUT: !40 = !DILocation(line: 16, column: 3, scope: !39)
// CHECK:STDOUT: !41 = !DILocation(line: 16, column: 26, scope: !39)
// CHECK:STDOUT: !43 = !DILocation(line: 22, column: 3, scope: !39)
// CHECK:STDOUT: !44 = !DILocation(line: 22, column: 26, scope: !39)
// CHECK:STDOUT: !46 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !47 = !{null, !46}
// CHECK:STDOUT: !48 = !DISubroutineType(types: !47)
// CHECK:STDOUT: !49 = !DILocation(line: 13, column: 1, scope: !39)
// CHECK:STDOUT: !50 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.b5f30f9c486f7750", scope: null, file: !26, line: 32, type: !48, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !56)
// CHECK:STDOUT: !51 = !DILocalVariable(arg: 1, scope: !50, type: !46)
// CHECK:STDOUT: !55 = !DILocation(line: 32, column: 1, scope: !50)
// CHECK:STDOUT: !56 = !{!51}
// CHECK:STDOUT: !57 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.8a58b7b15e68238e", scope: null, file: !26, line: 32, type: !48, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !63)
// CHECK:STDOUT: !58 = !DILocalVariable(arg: 1, scope: !57, type: !46)
// CHECK:STDOUT: !62 = !DILocation(line: 32, column: 1, scope: !57)
// CHECK:STDOUT: !63 = !{!58}
// CHECK:STDOUT:
