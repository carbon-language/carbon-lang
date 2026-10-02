// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/array.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/array/assign_return_value.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/array/assign_return_value.carbon

fn F() -> (i32, i32) { return (12, 24); }

fn Run() {
  var _: array(i32, 2) = F();
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'assign_return_value.carbon'
// CHECK:STDOUT: source_filename = "assign_return_value.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @tuple.1cd.loc13_39 = internal constant { i32, i32 } { i32 12, i32 24 }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main(ptr sret({ i32, i32 }) %return) #0 !dbg !32 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 0, !dbg !35
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 1, !dbg !35
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %return, ptr align 4 @tuple.1cd.loc13_39, i64 8, i1 false), !dbg !36
// CHECK:STDOUT:   ret void, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @main() #0 !dbg !37 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca [2 x i32], align 4, !dbg !38
// CHECK:STDOUT:   %.loc16_28.1.temp = alloca { i32, i32 }, align 4, !dbg !39
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !38
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc16_28.1.temp), !dbg !39
// CHECK:STDOUT:   call void @_CF.Main(ptr %.loc16_28.1.temp), !dbg !39
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %.loc16_28.1.temp, i32 0, i32 0, !dbg !39
// CHECK:STDOUT:   %.loc16_28.3 = load i32, ptr %tuple.elem0.tuple.elem, align 4, !dbg !39
// CHECK:STDOUT:   %.loc16_28.4.array.index = getelementptr inbounds [2 x i32], ptr %_.var, i32 0, i64 0, !dbg !39
// CHECK:STDOUT:   store i32 %.loc16_28.3, ptr %.loc16_28.4.array.index, align 4, !dbg !39
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %.loc16_28.1.temp, i32 0, i32 1, !dbg !39
// CHECK:STDOUT:   %.loc16_28.6 = load i32, ptr %tuple.elem1.tuple.elem, align 4, !dbg !39
// CHECK:STDOUT:   %.loc16_28.7.array.index = getelementptr inbounds [2 x i32], ptr %_.var, i32 0, i64 1, !dbg !39
// CHECK:STDOUT:   store i32 %.loc16_28.6, ptr %.loc16_28.7.array.index, align 4, !dbg !39
// CHECK:STDOUT:   call void @"_CSelfDestruct.e554695bf635d9c1:core.Destroy.Core"(ptr %.loc16_28.1.temp), !dbg !39
// CHECK:STDOUT:   call void @"_CSelfDestruct.e690471ed5f74dcb:core.Destroy.Core"(ptr %_.var), !dbg !38
// CHECK:STDOUT:   ret i32 0, !dbg !41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !42 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.e554695bf635d9c1:core.Destroy.Core"(ptr %self) #0 !dbg !46 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 1, !dbg !48
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %tuple.elem1.tuple.elem), !dbg !48
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 0, !dbg !48
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %tuple.elem0.tuple.elem), !dbg !48
// CHECK:STDOUT:   ret void, !dbg !48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.e554695bf635d9c1:core.Destroy.Core"(ptr %self) #0 !dbg !50 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.e554695bf635d9c1:core.Destroy.Core"(ptr %self), !dbg !52
// CHECK:STDOUT:   ret void, !dbg !52
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.e690471ed5f74dcb:core.Destroy.Core"(ptr %self) #0 !dbg !54 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc16_3.1.array.index = getelementptr inbounds [2 x i32], ptr %self, i32 0, i64 1, !dbg !56
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %.loc16_3.1.array.index), !dbg !56
// CHECK:STDOUT:   %.loc16_3.2.array.index = getelementptr inbounds [2 x i32], ptr %self, i32 0, i64 0, !dbg !56
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %.loc16_3.2.array.index), !dbg !56
// CHECK:STDOUT:   ret void, !dbg !56
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.e690471ed5f74dcb:core.Destroy.Core"(ptr %self) #0 !dbg !58 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.e690471ed5f74dcb:core.Destroy.Core"(ptr %self), !dbg !60
// CHECK:STDOUT:   ret void, !dbg !60
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!21}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIFile(filename: "assign_return_value.carbon", directory: "")
// CHECK:STDOUT: !21 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !20, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !22 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{!23}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = !{!22}
// CHECK:STDOUT: !27 = !DISubroutineType(types: !26)
// CHECK:STDOUT: !28 = !{null, !22}
// CHECK:STDOUT: !29 = !DISubroutineType(types: !28)
// CHECK:STDOUT: !30 = !{null, !23}
// CHECK:STDOUT: !31 = !DISubroutineType(types: !30)
// CHECK:STDOUT: !32 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !20, line: 13, type: !25, spFlags: DISPFlagDefinition, unit: !21)
// CHECK:STDOUT: !35 = !DILocation(line: 13, column: 31, scope: !32)
// CHECK:STDOUT: !36 = !DILocation(line: 13, column: 24, scope: !32)
// CHECK:STDOUT: !37 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !20, line: 15, type: !27, spFlags: DISPFlagDefinition, unit: !21)
// CHECK:STDOUT: !38 = !DILocation(line: 16, column: 3, scope: !37)
// CHECK:STDOUT: !39 = !DILocation(line: 16, column: 26, scope: !37)
// CHECK:STDOUT: !41 = !DILocation(line: 15, column: 1, scope: !37)
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f58524831c37807e:core.Destroy.Core", scope: null, file: !20, line: 16, type: !29, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !45)
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 1, scope: !42, type: !22)
// CHECK:STDOUT: !44 = !DILocation(line: 16, column: 26, scope: !42)
// CHECK:STDOUT: !45 = !{!43}
// CHECK:STDOUT: !46 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.e554695bf635d9c1:core.Destroy.Core", scope: null, file: !20, line: 16, type: !31, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !49)
// CHECK:STDOUT: !47 = !DILocalVariable(arg: 1, scope: !46, type: !23)
// CHECK:STDOUT: !48 = !DILocation(line: 16, column: 26, scope: !46)
// CHECK:STDOUT: !49 = !{!47}
// CHECK:STDOUT: !50 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.e554695bf635d9c1:core.Destroy.Core", scope: null, file: !20, line: 16, type: !31, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !53)
// CHECK:STDOUT: !51 = !DILocalVariable(arg: 1, scope: !50, type: !23)
// CHECK:STDOUT: !52 = !DILocation(line: 16, column: 26, scope: !50)
// CHECK:STDOUT: !53 = !{!51}
// CHECK:STDOUT: !54 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.e690471ed5f74dcb:core.Destroy.Core", scope: null, file: !20, line: 16, type: !31, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !57)
// CHECK:STDOUT: !55 = !DILocalVariable(arg: 1, scope: !54, type: !23)
// CHECK:STDOUT: !56 = !DILocation(line: 16, column: 3, scope: !54)
// CHECK:STDOUT: !57 = !{!55}
// CHECK:STDOUT: !58 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.e690471ed5f74dcb:core.Destroy.Core", scope: null, file: !20, line: 16, type: !31, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !61)
// CHECK:STDOUT: !59 = !DILocalVariable(arg: 1, scope: !58, type: !23)
// CHECK:STDOUT: !60 = !DILocation(line: 16, column: 3, scope: !58)
// CHECK:STDOUT: !61 = !{!59}
// CHECK:STDOUT:
