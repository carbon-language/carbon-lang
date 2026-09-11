// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/var/uninitialized.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/var/uninitialized.carbon

fn Run() {
  var unused a: i32;
  var unused b: f64;
  var unused c: ();
  var unused d: {};
  var unused e: bool;
  var unused f: i32*;
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'uninitialized.carbon'
// CHECK:STDOUT: source_filename = "uninitialized.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @main() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca i32, align 4, !dbg !8
// CHECK:STDOUT:   %b.var = alloca double, align 8, !dbg !9
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !10
// CHECK:STDOUT:   %d.var = alloca {}, align 1, !dbg !11
// CHECK:STDOUT:   %e.var = alloca i1, align 1, !dbg !12
// CHECK:STDOUT:   %f.var = alloca ptr, align 8, !dbg !13
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !8
// CHECK:STDOUT:   store i32 poison, ptr %a.var, align 4, !dbg !8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b.var), !dbg !9
// CHECK:STDOUT:   store double poison, ptr %b.var, align 8, !dbg !9
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !10
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %d.var), !dbg !11
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %e.var), !dbg !12
// CHECK:STDOUT:   store i8 poison, ptr %e.var, align 1, !dbg !12
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %f.var), !dbg !13
// CHECK:STDOUT:   store ptr poison, ptr %f.var, align 8, !dbg !13
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.f427c19f24d378eb(ptr %f.var), !dbg !13
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.ad5376756201bea4(ptr %e.var), !dbg !12
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.17de40ae0fb4b1fd(ptr %d.var), !dbg !11
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.c7a99711ca3557ac(ptr %c.var), !dbg !10
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.23895ff6626cffe1(ptr %b.var), !dbg !9
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48(ptr %a.var), !dbg !8
// CHECK:STDOUT:   ret i32 0, !dbg !14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.a1ae3d31ab69a7f8:core.Destroy.Core"(ptr %self) #0 !dbg !15 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.70db2d92ffddbef7:core.Destroy.Core"(ptr %self) #0 !dbg !22 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.f427c19f24d378eb(ptr %self) #0 !dbg !28 {
// CHECK:STDOUT:   call void @_CSubobjectDestroy.Destroy.Core.f427c19f24d378eb(ptr %self), !dbg !32
// CHECK:STDOUT:   ret void, !dbg !33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.ad5376756201bea4(ptr %self) #0 !dbg !34 {
// CHECK:STDOUT:   call void @_CSubobjectDestroy.Destroy.Core.f427c19f24d378eb(ptr %self), !dbg !37
// CHECK:STDOUT:   ret void, !dbg !38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.17de40ae0fb4b1fd(ptr %self) #0 !dbg !39 {
// CHECK:STDOUT:   call void @_CSubobjectDestroy.Destroy.Core.f427c19f24d378eb(ptr %self), !dbg !42
// CHECK:STDOUT:   ret void, !dbg !43
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.c7a99711ca3557ac(ptr %self) #0 !dbg !44 {
// CHECK:STDOUT:   call void @_CSubobjectDestroy.Destroy.Core.f427c19f24d378eb(ptr %self), !dbg !47
// CHECK:STDOUT:   ret void, !dbg !48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.23895ff6626cffe1(ptr %self) #0 !dbg !49 {
// CHECK:STDOUT:   call void @"_COp.a1ae3d31ab69a7f8:core.Destroy.Core"(ptr %self), !dbg !52
// CHECK:STDOUT:   call void @_CSubobjectDestroy.Destroy.Core.f427c19f24d378eb(ptr %self), !dbg !53
// CHECK:STDOUT:   ret void, !dbg !54
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48(ptr %self) #0 !dbg !55 {
// CHECK:STDOUT:   call void @"_COp.70db2d92ffddbef7:core.Destroy.Core"(ptr %self), !dbg !58
// CHECK:STDOUT:   call void @_CSubobjectDestroy.Destroy.Core.f427c19f24d378eb(ptr %self), !dbg !59
// CHECK:STDOUT:   ret void, !dbg !60
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSubobjectDestroy.Destroy.Core.f427c19f24d378eb(ptr %self) #0 !dbg !61 {
// CHECK:STDOUT:   ret void, !dbg !64
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSubobjectDestroy.Destroy.Core.f427c19f24d378eb, { 5, 4, 0, 1, 2, 3 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "uninitialized.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !1, line: 13, type: !5, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7}
// CHECK:STDOUT: !7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !8 = !DILocation(line: 14, column: 3, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 15, column: 3, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 16, column: 3, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 17, column: 3, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 18, column: 3, scope: !4)
// CHECK:STDOUT: !13 = !DILocation(line: 19, column: 3, scope: !4)
// CHECK:STDOUT: !14 = !DILocation(line: 13, column: 1, scope: !4)
// CHECK:STDOUT: !15 = distinct !DISubprogram(name: "Op", linkageName: "_COp.a1ae3d31ab69a7f8:core.Destroy.Core", scope: null, file: !1, line: 15, type: !16, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !19)
// CHECK:STDOUT: !16 = !DISubroutineType(types: !17)
// CHECK:STDOUT: !17 = !{null, !18}
// CHECK:STDOUT: !18 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !19 = !{!20}
// CHECK:STDOUT: !20 = !DILocalVariable(arg: 1, scope: !15, type: !18)
// CHECK:STDOUT: !21 = !DILocation(line: 15, column: 3, scope: !15)
// CHECK:STDOUT: !22 = distinct !DISubprogram(name: "Op", linkageName: "_COp.70db2d92ffddbef7:core.Destroy.Core", scope: null, file: !1, line: 14, type: !23, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !25)
// CHECK:STDOUT: !23 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !24 = !{null, !7}
// CHECK:STDOUT: !25 = !{!26}
// CHECK:STDOUT: !26 = !DILocalVariable(arg: 1, scope: !22, type: !7)
// CHECK:STDOUT: !27 = !DILocation(line: 14, column: 3, scope: !22)
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.f427c19f24d378eb", scope: null, file: !29, line: 26, type: !16, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !30)
// CHECK:STDOUT: !29 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !30 = !{!31}
// CHECK:STDOUT: !31 = !DILocalVariable(arg: 1, scope: !28, type: !18)
// CHECK:STDOUT: !32 = !DILocation(line: 28, column: 5, scope: !28)
// CHECK:STDOUT: !33 = !DILocation(line: 26, column: 3, scope: !28)
// CHECK:STDOUT: !34 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.ad5376756201bea4", scope: null, file: !29, line: 26, type: !16, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !35)
// CHECK:STDOUT: !35 = !{!36}
// CHECK:STDOUT: !36 = !DILocalVariable(arg: 1, scope: !34, type: !18)
// CHECK:STDOUT: !37 = !DILocation(line: 28, column: 5, scope: !34)
// CHECK:STDOUT: !38 = !DILocation(line: 26, column: 3, scope: !34)
// CHECK:STDOUT: !39 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.17de40ae0fb4b1fd", scope: null, file: !29, line: 26, type: !16, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !40)
// CHECK:STDOUT: !40 = !{!41}
// CHECK:STDOUT: !41 = !DILocalVariable(arg: 1, scope: !39, type: !18)
// CHECK:STDOUT: !42 = !DILocation(line: 28, column: 5, scope: !39)
// CHECK:STDOUT: !43 = !DILocation(line: 26, column: 3, scope: !39)
// CHECK:STDOUT: !44 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.c7a99711ca3557ac", scope: null, file: !29, line: 26, type: !16, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !45)
// CHECK:STDOUT: !45 = !{!46}
// CHECK:STDOUT: !46 = !DILocalVariable(arg: 1, scope: !44, type: !18)
// CHECK:STDOUT: !47 = !DILocation(line: 28, column: 5, scope: !44)
// CHECK:STDOUT: !48 = !DILocation(line: 26, column: 3, scope: !44)
// CHECK:STDOUT: !49 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.23895ff6626cffe1", scope: null, file: !29, line: 26, type: !16, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !50)
// CHECK:STDOUT: !50 = !{!51}
// CHECK:STDOUT: !51 = !DILocalVariable(arg: 1, scope: !49, type: !18)
// CHECK:STDOUT: !52 = !DILocation(line: 27, column: 5, scope: !49)
// CHECK:STDOUT: !53 = !DILocation(line: 28, column: 5, scope: !49)
// CHECK:STDOUT: !54 = !DILocation(line: 26, column: 3, scope: !49)
// CHECK:STDOUT: !55 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48", scope: null, file: !29, line: 26, type: !23, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !56)
// CHECK:STDOUT: !56 = !{!57}
// CHECK:STDOUT: !57 = !DILocalVariable(arg: 1, scope: !55, type: !7)
// CHECK:STDOUT: !58 = !DILocation(line: 27, column: 5, scope: !55)
// CHECK:STDOUT: !59 = !DILocation(line: 28, column: 5, scope: !55)
// CHECK:STDOUT: !60 = !DILocation(line: 26, column: 3, scope: !55)
// CHECK:STDOUT: !61 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.Destroy.Core.f427c19f24d378eb", scope: null, file: !29, line: 19, type: !16, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !62)
// CHECK:STDOUT: !62 = !{!63}
// CHECK:STDOUT: !63 = !DILocalVariable(arg: 1, scope: !61, type: !18)
// CHECK:STDOUT: !64 = !DILocation(line: 19, column: 3, scope: !61)
// CHECK:STDOUT:
