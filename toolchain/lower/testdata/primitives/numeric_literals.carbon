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
// CHECK:STDOUT: @array.311.loc16_3 = internal constant [4 x i32] [i32 8, i32 9, i32 8, i32 8]
// CHECK:STDOUT: @array.4d5.loc22_3 = internal constant [6 x double] [double 9.000000e-01, double 8.000000e+00, double 8.000000e+01, double 1.000000e+07, double 1.000000e+08, double 1.000000e-08]
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !44 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var.loc16 = alloca [4 x i32], align 4, !dbg !45
// CHECK:STDOUT:   %_.var.loc22 = alloca [6 x double], align 8, !dbg !48
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc16), !dbg !45
// CHECK:STDOUT:   %.loc21_3.3.array.index = getelementptr inbounds [4 x i32], ptr %_.var.loc16, i32 0, i64 0, !dbg !46
// CHECK:STDOUT:   %.loc21_3.6.array.index = getelementptr inbounds [4 x i32], ptr %_.var.loc16, i32 0, i64 1, !dbg !46
// CHECK:STDOUT:   %.loc21_3.9.array.index = getelementptr inbounds [4 x i32], ptr %_.var.loc16, i32 0, i64 2, !dbg !46
// CHECK:STDOUT:   %.loc21_3.12.array.index = getelementptr inbounds [4 x i32], ptr %_.var.loc16, i32 0, i64 3, !dbg !46
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %_.var.loc16, ptr align 4 @array.311.loc16_3, i64 16, i1 false), !dbg !45
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc22), !dbg !48
// CHECK:STDOUT:   %.loc29_3.3.array.index = getelementptr inbounds [6 x double], ptr %_.var.loc22, i32 0, i64 0, !dbg !49
// CHECK:STDOUT:   %.loc29_3.6.array.index = getelementptr inbounds [6 x double], ptr %_.var.loc22, i32 0, i64 1, !dbg !49
// CHECK:STDOUT:   %.loc29_3.9.array.index = getelementptr inbounds [6 x double], ptr %_.var.loc22, i32 0, i64 2, !dbg !49
// CHECK:STDOUT:   %.loc29_3.12.array.index = getelementptr inbounds [6 x double], ptr %_.var.loc22, i32 0, i64 3, !dbg !49
// CHECK:STDOUT:   %.loc29_3.15.array.index = getelementptr inbounds [6 x double], ptr %_.var.loc22, i32 0, i64 4, !dbg !49
// CHECK:STDOUT:   %.loc29_3.18.array.index = getelementptr inbounds [6 x double], ptr %_.var.loc22, i32 0, i64 5, !dbg !49
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %_.var.loc22, ptr align 8 @array.4d5.loc22_3, i64 48, i1 false), !dbg !48
// CHECK:STDOUT:   call void @"_CSelfDestruct.adf7ea314117bbfa:core.Destroy.Core"(ptr %_.var.loc22), !dbg !48
// CHECK:STDOUT:   call void @"_CSelfDestruct.d15a62aea0967f8e:core.Destroy.Core"(ptr %_.var.loc16), !dbg !45
// CHECK:STDOUT:   ret void, !dbg !51
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.dc14d4dd21801791:core.Destroy.Core"(ptr %self) #0 !dbg !52 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !54
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.adf7ea314117bbfa:core.Destroy.Core"(ptr %self) #0 !dbg !56 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc22_3.1.array.index = getelementptr inbounds [6 x double], ptr %self, i32 0, i64 5, !dbg !58
// CHECK:STDOUT:   call void @"_CSelfDestruct.dc14d4dd21801791:core.Destroy.Core"(ptr %.loc22_3.1.array.index), !dbg !58
// CHECK:STDOUT:   %.loc22_3.2.array.index = getelementptr inbounds [6 x double], ptr %self, i32 0, i64 4, !dbg !58
// CHECK:STDOUT:   call void @"_CSelfDestruct.dc14d4dd21801791:core.Destroy.Core"(ptr %.loc22_3.2.array.index), !dbg !58
// CHECK:STDOUT:   %.loc22_3.3.array.index = getelementptr inbounds [6 x double], ptr %self, i32 0, i64 3, !dbg !58
// CHECK:STDOUT:   call void @"_CSelfDestruct.dc14d4dd21801791:core.Destroy.Core"(ptr %.loc22_3.3.array.index), !dbg !58
// CHECK:STDOUT:   %.loc22_3.4.array.index = getelementptr inbounds [6 x double], ptr %self, i32 0, i64 2, !dbg !58
// CHECK:STDOUT:   call void @"_CSelfDestruct.dc14d4dd21801791:core.Destroy.Core"(ptr %.loc22_3.4.array.index), !dbg !58
// CHECK:STDOUT:   %.loc22_3.5.array.index = getelementptr inbounds [6 x double], ptr %self, i32 0, i64 1, !dbg !58
// CHECK:STDOUT:   call void @"_CSelfDestruct.dc14d4dd21801791:core.Destroy.Core"(ptr %.loc22_3.5.array.index), !dbg !58
// CHECK:STDOUT:   %.loc22_3.6.array.index = getelementptr inbounds [6 x double], ptr %self, i32 0, i64 0, !dbg !58
// CHECK:STDOUT:   call void @"_CSelfDestruct.dc14d4dd21801791:core.Destroy.Core"(ptr %.loc22_3.6.array.index), !dbg !58
// CHECK:STDOUT:   ret void, !dbg !58
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.adf7ea314117bbfa:core.Destroy.Core"(ptr %self) #0 !dbg !60 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.adf7ea314117bbfa:core.Destroy.Core"(ptr %self), !dbg !62
// CHECK:STDOUT:   ret void, !dbg !62
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !64 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !66
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.d15a62aea0967f8e:core.Destroy.Core"(ptr %self) #0 !dbg !68 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc16_3.1.array.index = getelementptr inbounds [4 x i32], ptr %self, i32 0, i64 3, !dbg !70
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %.loc16_3.1.array.index), !dbg !70
// CHECK:STDOUT:   %.loc16_3.2.array.index = getelementptr inbounds [4 x i32], ptr %self, i32 0, i64 2, !dbg !70
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %.loc16_3.2.array.index), !dbg !70
// CHECK:STDOUT:   %.loc16_3.3.array.index = getelementptr inbounds [4 x i32], ptr %self, i32 0, i64 1, !dbg !70
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %.loc16_3.3.array.index), !dbg !70
// CHECK:STDOUT:   %.loc16_3.4.array.index = getelementptr inbounds [4 x i32], ptr %self, i32 0, i64 0, !dbg !70
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %.loc16_3.4.array.index), !dbg !70
// CHECK:STDOUT:   ret void, !dbg !70
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.d15a62aea0967f8e:core.Destroy.Core"(ptr %self) #0 !dbg !72 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.d15a62aea0967f8e:core.Destroy.Core"(ptr %self), !dbg !74
// CHECK:STDOUT:   ret void, !dbg !74
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
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
// CHECK:STDOUT: !34 = !DIFile(filename: "numeric_literals.carbon", directory: "")
// CHECK:STDOUT: !35 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !34, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !36 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !37 = !{null}
// CHECK:STDOUT: !38 = !DISubroutineType(types: !37)
// CHECK:STDOUT: !39 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !40 = !{null, !39}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !42 = !{null, !36}
// CHECK:STDOUT: !43 = !DISubroutineType(types: !42)
// CHECK:STDOUT: !44 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !34, line: 13, type: !38, spFlags: DISPFlagDefinition, unit: !35)
// CHECK:STDOUT: !45 = !DILocation(line: 16, column: 3, scope: !44)
// CHECK:STDOUT: !46 = !DILocation(line: 16, column: 26, scope: !44)
// CHECK:STDOUT: !48 = !DILocation(line: 22, column: 3, scope: !44)
// CHECK:STDOUT: !49 = !DILocation(line: 22, column: 26, scope: !44)
// CHECK:STDOUT: !51 = !DILocation(line: 13, column: 1, scope: !44)
// CHECK:STDOUT: !52 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.dc14d4dd21801791:core.Destroy.Core", scope: null, file: !34, line: 22, type: !41, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !55)
// CHECK:STDOUT: !53 = !DILocalVariable(arg: 1, scope: !52, type: !39)
// CHECK:STDOUT: !54 = !DILocation(line: 22, column: 3, scope: !52)
// CHECK:STDOUT: !55 = !{!53}
// CHECK:STDOUT: !56 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.adf7ea314117bbfa:core.Destroy.Core", scope: null, file: !34, line: 22, type: !41, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !59)
// CHECK:STDOUT: !57 = !DILocalVariable(arg: 1, scope: !56, type: !39)
// CHECK:STDOUT: !58 = !DILocation(line: 22, column: 3, scope: !56)
// CHECK:STDOUT: !59 = !{!57}
// CHECK:STDOUT: !60 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.adf7ea314117bbfa:core.Destroy.Core", scope: null, file: !34, line: 22, type: !41, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !63)
// CHECK:STDOUT: !61 = !DILocalVariable(arg: 1, scope: !60, type: !39)
// CHECK:STDOUT: !62 = !DILocation(line: 22, column: 3, scope: !60)
// CHECK:STDOUT: !63 = !{!61}
// CHECK:STDOUT: !64 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f58524831c37807e:core.Destroy.Core", scope: null, file: !34, line: 16, type: !43, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !67)
// CHECK:STDOUT: !65 = !DILocalVariable(arg: 1, scope: !64, type: !36)
// CHECK:STDOUT: !66 = !DILocation(line: 16, column: 3, scope: !64)
// CHECK:STDOUT: !67 = !{!65}
// CHECK:STDOUT: !68 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.d15a62aea0967f8e:core.Destroy.Core", scope: null, file: !34, line: 16, type: !41, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !71)
// CHECK:STDOUT: !69 = !DILocalVariable(arg: 1, scope: !68, type: !39)
// CHECK:STDOUT: !70 = !DILocation(line: 16, column: 3, scope: !68)
// CHECK:STDOUT: !71 = !{!69}
// CHECK:STDOUT: !72 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.d15a62aea0967f8e:core.Destroy.Core", scope: null, file: !34, line: 16, type: !41, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !75)
// CHECK:STDOUT: !73 = !DILocalVariable(arg: 1, scope: !72, type: !39)
// CHECK:STDOUT: !74 = !DILocation(line: 16, column: 3, scope: !72)
// CHECK:STDOUT: !75 = !{!73}
// CHECK:STDOUT:
