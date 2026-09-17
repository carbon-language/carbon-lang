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
// CHECK:STDOUT: define i32 @main() #0 !dbg !14 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca i32, align 4, !dbg !15
// CHECK:STDOUT:   %b.var = alloca double, align 8, !dbg !17
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !19
// CHECK:STDOUT:   %d.var = alloca {}, align 1, !dbg !21
// CHECK:STDOUT:   %e.var = alloca i1, align 1, !dbg !23
// CHECK:STDOUT:   %f.var = alloca ptr, align 8, !dbg !25
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !15
// CHECK:STDOUT:   store i32 poison, ptr %a.var, align 4, !dbg !15
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b.var), !dbg !17
// CHECK:STDOUT:   store double poison, ptr %b.var, align 8, !dbg !17
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !19
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %d.var), !dbg !21
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %e.var), !dbg !23
// CHECK:STDOUT:   store i8 poison, ptr %e.var, align 1, !dbg !23
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %f.var), !dbg !25
// CHECK:STDOUT:   store ptr poison, ptr %f.var, align 8, !dbg !25
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.f427c19f24d378eb(ptr %f.var), !dbg !25
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.ad5376756201bea4(ptr %e.var), !dbg !23
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.17de40ae0fb4b1fd(ptr %d.var), !dbg !21
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.c7a99711ca3557ac(ptr %c.var), !dbg !19
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.23895ff6626cffe1(ptr %b.var), !dbg !17
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48(ptr %a.var), !dbg !15
// CHECK:STDOUT:   ret i32 0, !dbg !27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.dc14d4dd21801791:core.Destroy.Core"(ptr %self) #0 !dbg !28 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !32 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.f427c19f24d378eb(ptr %self) #0 !dbg !37 {
// CHECK:STDOUT:   ret void, !dbg !42
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.ad5376756201bea4(ptr %self) #0 !dbg !44 {
// CHECK:STDOUT:   ret void, !dbg !49
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.17de40ae0fb4b1fd(ptr %self) #0 !dbg !51 {
// CHECK:STDOUT:   ret void, !dbg !56
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.c7a99711ca3557ac(ptr %self) #0 !dbg !58 {
// CHECK:STDOUT:   ret void, !dbg !63
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.23895ff6626cffe1(ptr %self) #0 !dbg !65 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.dc14d4dd21801791:core.Destroy.Core"(ptr %self), !dbg !69
// CHECK:STDOUT:   ret void, !dbg !70
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48(ptr %self) #0 !dbg !72 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self), !dbg !76
// CHECK:STDOUT:   ret void, !dbg !77
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "uninitialized.carbon", directory: "")
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !7 = !{!6}
// CHECK:STDOUT: !8 = !DISubroutineType(types: !7)
// CHECK:STDOUT: !9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !10 = !{null, !9}
// CHECK:STDOUT: !11 = !DISubroutineType(types: !10)
// CHECK:STDOUT: !12 = !{null, !6}
// CHECK:STDOUT: !13 = !DISubroutineType(types: !12)
// CHECK:STDOUT: !14 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !4, line: 13, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !15 = !DILocation(line: 14, column: 3, scope: !14)
// CHECK:STDOUT: !17 = !DILocation(line: 15, column: 3, scope: !14)
// CHECK:STDOUT: !19 = !DILocation(line: 16, column: 3, scope: !14)
// CHECK:STDOUT: !21 = !DILocation(line: 17, column: 3, scope: !14)
// CHECK:STDOUT: !23 = !DILocation(line: 18, column: 3, scope: !14)
// CHECK:STDOUT: !25 = !DILocation(line: 19, column: 3, scope: !14)
// CHECK:STDOUT: !27 = !DILocation(line: 13, column: 1, scope: !14)
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.dc14d4dd21801791:core.Destroy.Core", scope: null, file: !4, line: 15, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !31)
// CHECK:STDOUT: !29 = !DILocalVariable(arg: 1, scope: !28, type: !9)
// CHECK:STDOUT: !30 = !DILocation(line: 15, column: 3, scope: !28)
// CHECK:STDOUT: !31 = !{!29}
// CHECK:STDOUT: !32 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core", scope: null, file: !4, line: 14, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !35)
// CHECK:STDOUT: !33 = !DILocalVariable(arg: 1, scope: !32, type: !6)
// CHECK:STDOUT: !34 = !DILocation(line: 14, column: 3, scope: !32)
// CHECK:STDOUT: !35 = !{!33}
// CHECK:STDOUT: !36 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !37 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.f427c19f24d378eb", scope: null, file: !36, line: 26, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !43)
// CHECK:STDOUT: !38 = !DILocalVariable(arg: 1, scope: !37, type: !9)
// CHECK:STDOUT: !42 = !DILocation(line: 26, column: 3, scope: !37)
// CHECK:STDOUT: !43 = !{!38}
// CHECK:STDOUT: !44 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.ad5376756201bea4", scope: null, file: !36, line: 26, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !50)
// CHECK:STDOUT: !45 = !DILocalVariable(arg: 1, scope: !44, type: !9)
// CHECK:STDOUT: !49 = !DILocation(line: 26, column: 3, scope: !44)
// CHECK:STDOUT: !50 = !{!45}
// CHECK:STDOUT: !51 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.17de40ae0fb4b1fd", scope: null, file: !36, line: 26, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !57)
// CHECK:STDOUT: !52 = !DILocalVariable(arg: 1, scope: !51, type: !9)
// CHECK:STDOUT: !56 = !DILocation(line: 26, column: 3, scope: !51)
// CHECK:STDOUT: !57 = !{!52}
// CHECK:STDOUT: !58 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.c7a99711ca3557ac", scope: null, file: !36, line: 26, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !64)
// CHECK:STDOUT: !59 = !DILocalVariable(arg: 1, scope: !58, type: !9)
// CHECK:STDOUT: !63 = !DILocation(line: 26, column: 3, scope: !58)
// CHECK:STDOUT: !64 = !{!59}
// CHECK:STDOUT: !65 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.23895ff6626cffe1", scope: null, file: !36, line: 26, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !71)
// CHECK:STDOUT: !66 = !DILocalVariable(arg: 1, scope: !65, type: !9)
// CHECK:STDOUT: !69 = !DILocation(line: 28, column: 5, scope: !65)
// CHECK:STDOUT: !70 = !DILocation(line: 26, column: 3, scope: !65)
// CHECK:STDOUT: !71 = !{!66}
// CHECK:STDOUT: !72 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48", scope: null, file: !36, line: 26, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !78)
// CHECK:STDOUT: !73 = !DILocalVariable(arg: 1, scope: !72, type: !6)
// CHECK:STDOUT: !76 = !DILocation(line: 28, column: 5, scope: !72)
// CHECK:STDOUT: !77 = !DILocation(line: 26, column: 3, scope: !72)
// CHECK:STDOUT: !78 = !{!73}
// CHECK:STDOUT:
