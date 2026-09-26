// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/class/basic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/class/basic.carbon

class C {
  var a: i32;
  var b: C*;
}

fn F(c: C) -> C;

fn Run() {
  var c: C;
  var _: C = F(c);
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'basic.carbon'
// CHECK:STDOUT: source_filename = "basic.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CF.Main(ptr sret({ i32, ptr }), ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @main() #0 !dbg !30 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca { i32, ptr }, align 8, !dbg !31
// CHECK:STDOUT:   %_.var = alloca { i32, ptr }, align 8, !dbg !33
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !31
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !33
// CHECK:STDOUT:   call void @_CF.Main(ptr %_.var, ptr %c.var), !dbg !35
// CHECK:STDOUT:   call void @"_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core"(ptr %_.var), !dbg !33
// CHECK:STDOUT:   call void @"_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core"(ptr %c.var), !dbg !31
// CHECK:STDOUT:   ret i32 0, !dbg !37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.6dae29ece9a708c4:core.Destroy.Core"(ptr %self) #0 !dbg !38 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !42 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !46 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self), !dbg !48
// CHECK:STDOUT:   ret void, !dbg !48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.6b633cf020343e69:core.Destroy.Core"(ptr %self) #0 !dbg !50 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !52
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.09861f44a9b84abf:core.Destroy.Core"(ptr %self) #0 !dbg !54 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc22_3.1.a = getelementptr inbounds nuw { i32, ptr }, ptr %self, i32 0, i32 0, !dbg !56
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %.loc22_3.1.a), !dbg !56
// CHECK:STDOUT:   %.loc22_3.2.b = getelementptr inbounds nuw { i32, ptr }, ptr %self, i32 0, i32 1, !dbg !56
// CHECK:STDOUT:   call void @"_CSelfDestruct.6b633cf020343e69:core.Destroy.Core"(ptr %.loc22_3.2.b), !dbg !56
// CHECK:STDOUT:   ret void, !dbg !56
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.09861f44a9b84abf:core.Destroy.Core"(ptr %self) #0 !dbg !58 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.09861f44a9b84abf:core.Destroy.Core"(ptr %self), !dbg !60
// CHECK:STDOUT:   ret void, !dbg !60
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.08d9e38b928ba575:core.Destroy.Core"(ptr %self) #0 !dbg !62 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc22_3.1.a = getelementptr inbounds nuw { i32, ptr }, ptr %self, i32 0, i32 0, !dbg !64
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %.loc22_3.1.a), !dbg !64
// CHECK:STDOUT:   %.loc22_3.2.b = getelementptr inbounds nuw { i32, ptr }, ptr %self, i32 0, i32 1, !dbg !64
// CHECK:STDOUT:   call void @"_CSelfDestruct.6b633cf020343e69:core.Destroy.Core"(ptr %.loc22_3.2.b), !dbg !64
// CHECK:STDOUT:   ret void, !dbg !64
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core"(ptr %self) #0 !dbg !66 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.08d9e38b928ba575:core.Destroy.Core"(ptr %self), !dbg !68
// CHECK:STDOUT:   ret void, !dbg !68
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @"_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !18 = !DIFile(filename: "basic.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{!20}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = !{null, !20}
// CHECK:STDOUT: !27 = !DISubroutineType(types: !26)
// CHECK:STDOUT: !28 = !{null, !21}
// CHECK:STDOUT: !29 = !DISubroutineType(types: !28)
// CHECK:STDOUT: !30 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !18, line: 20, type: !25, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !31 = !DILocation(line: 21, column: 3, scope: !30)
// CHECK:STDOUT: !33 = !DILocation(line: 22, column: 3, scope: !30)
// CHECK:STDOUT: !35 = !DILocation(line: 22, column: 14, scope: !30)
// CHECK:STDOUT: !37 = !DILocation(line: 20, column: 1, scope: !30)
// CHECK:STDOUT: !38 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.6dae29ece9a708c4:core.Destroy.Core", scope: null, file: !18, line: 22, type: !27, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !41)
// CHECK:STDOUT: !39 = !DILocalVariable(arg: 1, scope: !38, type: !20)
// CHECK:STDOUT: !40 = !DILocation(line: 22, column: 3, scope: !38)
// CHECK:STDOUT: !41 = !{!39}
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core", scope: null, file: !18, line: 22, type: !27, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !45)
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 1, scope: !42, type: !20)
// CHECK:STDOUT: !44 = !DILocation(line: 22, column: 3, scope: !42)
// CHECK:STDOUT: !45 = !{!43}
// CHECK:STDOUT: !46 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f58524831c37807e:core.Destroy.Core", scope: null, file: !18, line: 22, type: !27, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !49)
// CHECK:STDOUT: !47 = !DILocalVariable(arg: 1, scope: !46, type: !20)
// CHECK:STDOUT: !48 = !DILocation(line: 22, column: 3, scope: !46)
// CHECK:STDOUT: !49 = !{!47}
// CHECK:STDOUT: !50 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.6b633cf020343e69:core.Destroy.Core", scope: null, file: !18, line: 22, type: !29, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !53)
// CHECK:STDOUT: !51 = !DILocalVariable(arg: 1, scope: !50, type: !21)
// CHECK:STDOUT: !52 = !DILocation(line: 22, column: 3, scope: !50)
// CHECK:STDOUT: !53 = !{!51}
// CHECK:STDOUT: !54 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.09861f44a9b84abf:core.Destroy.Core", scope: null, file: !18, line: 22, type: !29, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !57)
// CHECK:STDOUT: !55 = !DILocalVariable(arg: 1, scope: !54, type: !21)
// CHECK:STDOUT: !56 = !DILocation(line: 22, column: 3, scope: !54)
// CHECK:STDOUT: !57 = !{!55}
// CHECK:STDOUT: !58 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.09861f44a9b84abf:core.Destroy.Core", scope: null, file: !18, line: 22, type: !29, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !61)
// CHECK:STDOUT: !59 = !DILocalVariable(arg: 1, scope: !58, type: !21)
// CHECK:STDOUT: !60 = !DILocation(line: 22, column: 3, scope: !58)
// CHECK:STDOUT: !61 = !{!59}
// CHECK:STDOUT: !62 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.08d9e38b928ba575:core.Destroy.Core", scope: null, file: !18, line: 22, type: !29, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !65)
// CHECK:STDOUT: !63 = !DILocalVariable(arg: 1, scope: !62, type: !21)
// CHECK:STDOUT: !64 = !DILocation(line: 22, column: 3, scope: !62)
// CHECK:STDOUT: !65 = !{!63}
// CHECK:STDOUT: !66 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core", scope: null, file: !18, line: 22, type: !29, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !69)
// CHECK:STDOUT: !67 = !DILocalVariable(arg: 1, scope: !66, type: !21)
// CHECK:STDOUT: !68 = !DILocation(line: 22, column: 3, scope: !66)
// CHECK:STDOUT: !69 = !{!67}
// CHECK:STDOUT:
