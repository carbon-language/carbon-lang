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
// CHECK:STDOUT: @array.f6b.loc16_3.1 = internal constant [4 x i32] [i32 8, i32 9, i32 8, i32 8]
// CHECK:STDOUT: @array.69b.loc22_3.1 = internal constant [6 x double] [double 9.000000e-01, double 8.000000e+00, double 8.000000e+01, double 1.000000e+07, double 1.000000e+08, double 1.000000e-08]
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
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %_.var.loc16, ptr align 4 @array.f6b.loc16_3.1, i64 16, i1 false), !dbg !43
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc22), !dbg !46
// CHECK:STDOUT:   %.loc29_3.3.array.index = getelementptr inbounds [6 x double], ptr %_.var.loc22, i32 0, i64 0, !dbg !47
// CHECK:STDOUT:   %.loc29_3.6.array.index = getelementptr inbounds [6 x double], ptr %_.var.loc22, i32 0, i64 1, !dbg !47
// CHECK:STDOUT:   %.loc29_3.9.array.index = getelementptr inbounds [6 x double], ptr %_.var.loc22, i32 0, i64 2, !dbg !47
// CHECK:STDOUT:   %.loc29_3.12.array.index = getelementptr inbounds [6 x double], ptr %_.var.loc22, i32 0, i64 3, !dbg !47
// CHECK:STDOUT:   %.loc29_3.15.array.index = getelementptr inbounds [6 x double], ptr %_.var.loc22, i32 0, i64 4, !dbg !47
// CHECK:STDOUT:   %.loc29_3.18.array.index = getelementptr inbounds [6 x double], ptr %_.var.loc22, i32 0, i64 5, !dbg !47
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %_.var.loc22, ptr align 8 @array.69b.loc22_3.1, i64 48, i1 false), !dbg !46
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.63943f795a3f356c(ptr %_.var.loc22), !dbg !46
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.a81e8431cbd7c92a(ptr %_.var.loc16), !dbg !43
// CHECK:STDOUT:   ret void, !dbg !49
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.dc14d4dd21801791:core.Destroy.Core"(ptr %self) #0 !dbg !50 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !52
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.2891f36f5aa8bde4:core.Destroy.Core"(ptr %self) #0 !dbg !54 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !56
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !58 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !60
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.88cc3346b7991af5:core.Destroy.Core"(ptr %self) #0 !dbg !62 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !64
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.63943f795a3f356c(ptr %self) #0 !dbg !66 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.2891f36f5aa8bde4:core.Destroy.Core"(ptr %self), !dbg !70
// CHECK:STDOUT:   ret void, !dbg !71
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.a81e8431cbd7c92a(ptr %self) #0 !dbg !73 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.88cc3346b7991af5:core.Destroy.Core"(ptr %self), !dbg !77
// CHECK:STDOUT:   ret void, !dbg !78
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 1, 0 }
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
// CHECK:STDOUT: !32 = !DIFile(filename: "numeric_literals.carbon", directory: "")
// CHECK:STDOUT: !33 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !32, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !34 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !35 = !{null}
// CHECK:STDOUT: !36 = !DISubroutineType(types: !35)
// CHECK:STDOUT: !37 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !38 = !{null, !37}
// CHECK:STDOUT: !39 = !DISubroutineType(types: !38)
// CHECK:STDOUT: !40 = !{null, !34}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !32, line: 13, type: !36, spFlags: DISPFlagDefinition, unit: !33)
// CHECK:STDOUT: !43 = !DILocation(line: 16, column: 3, scope: !42)
// CHECK:STDOUT: !44 = !DILocation(line: 16, column: 26, scope: !42)
// CHECK:STDOUT: !46 = !DILocation(line: 22, column: 3, scope: !42)
// CHECK:STDOUT: !47 = !DILocation(line: 22, column: 26, scope: !42)
// CHECK:STDOUT: !49 = !DILocation(line: 13, column: 1, scope: !42)
// CHECK:STDOUT: !50 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.dc14d4dd21801791:core.Destroy.Core", scope: null, file: !32, line: 22, type: !39, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !53)
// CHECK:STDOUT: !51 = !DILocalVariable(arg: 1, scope: !50, type: !37)
// CHECK:STDOUT: !52 = !DILocation(line: 22, column: 3, scope: !50)
// CHECK:STDOUT: !53 = !{!51}
// CHECK:STDOUT: !54 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.2891f36f5aa8bde4:core.Destroy.Core", scope: null, file: !32, line: 22, type: !39, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !57)
// CHECK:STDOUT: !55 = !DILocalVariable(arg: 1, scope: !54, type: !37)
// CHECK:STDOUT: !56 = !DILocation(line: 22, column: 3, scope: !54)
// CHECK:STDOUT: !57 = !{!55}
// CHECK:STDOUT: !58 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core", scope: null, file: !32, line: 16, type: !41, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !61)
// CHECK:STDOUT: !59 = !DILocalVariable(arg: 1, scope: !58, type: !34)
// CHECK:STDOUT: !60 = !DILocation(line: 16, column: 3, scope: !58)
// CHECK:STDOUT: !61 = !{!59}
// CHECK:STDOUT: !62 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.88cc3346b7991af5:core.Destroy.Core", scope: null, file: !32, line: 16, type: !39, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !65)
// CHECK:STDOUT: !63 = !DILocalVariable(arg: 1, scope: !62, type: !37)
// CHECK:STDOUT: !64 = !DILocation(line: 16, column: 3, scope: !62)
// CHECK:STDOUT: !65 = !{!63}
// CHECK:STDOUT: !66 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.63943f795a3f356c", scope: null, file: !28, line: 24, type: !39, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !72)
// CHECK:STDOUT: !67 = !DILocalVariable(arg: 1, scope: !66, type: !37)
// CHECK:STDOUT: !70 = !DILocation(line: 26, column: 5, scope: !66)
// CHECK:STDOUT: !71 = !DILocation(line: 24, column: 3, scope: !66)
// CHECK:STDOUT: !72 = !{!67}
// CHECK:STDOUT: !73 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.a81e8431cbd7c92a", scope: null, file: !28, line: 24, type: !39, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !79)
// CHECK:STDOUT: !74 = !DILocalVariable(arg: 1, scope: !73, type: !37)
// CHECK:STDOUT: !77 = !DILocation(line: 26, column: 5, scope: !73)
// CHECK:STDOUT: !78 = !DILocation(line: 24, column: 3, scope: !73)
// CHECK:STDOUT: !79 = !{!74}
// CHECK:STDOUT:
