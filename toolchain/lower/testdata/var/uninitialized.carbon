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
// CHECK:STDOUT: define i32 @main() #0 !dbg !9 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca i32, align 4, !dbg !10
// CHECK:STDOUT:   %b.var = alloca double, align 8, !dbg !12
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !14
// CHECK:STDOUT:   %d.var = alloca {}, align 1, !dbg !16
// CHECK:STDOUT:   %e.var = alloca i1, align 1, !dbg !18
// CHECK:STDOUT:   %f.var = alloca ptr, align 8, !dbg !20
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !10
// CHECK:STDOUT:   store i32 poison, ptr %a.var, align 4, !dbg !10
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b.var), !dbg !12
// CHECK:STDOUT:   store double poison, ptr %b.var, align 8, !dbg !12
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !14
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %d.var), !dbg !16
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %e.var), !dbg !18
// CHECK:STDOUT:   store i8 poison, ptr %e.var, align 1, !dbg !18
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %f.var), !dbg !20
// CHECK:STDOUT:   store ptr poison, ptr %f.var, align 8, !dbg !20
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.4f29d15a20894e9b(ptr %f.var), !dbg !20
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.29e6aa423646b61a(ptr %e.var), !dbg !18
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.61b5bb2079973306(ptr %d.var), !dbg !16
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.9045abb8230cbcdf(ptr %c.var), !dbg !14
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.2b38c2ca9f7f1a94(ptr %b.var), !dbg !12
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %a.var), !dbg !10
// CHECK:STDOUT:   ret i32 0, !dbg !27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.4f29d15a20894e9b(ptr %t) #0 !dbg !29 {
// CHECK:STDOUT:   ret void, !dbg !34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.29e6aa423646b61a(ptr %t) #0 !dbg !36 {
// CHECK:STDOUT:   ret void, !dbg !41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.61b5bb2079973306(ptr %t) #0 !dbg !43 {
// CHECK:STDOUT:   ret void, !dbg !48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.9045abb8230cbcdf(ptr %t) #0 !dbg !50 {
// CHECK:STDOUT:   ret void, !dbg !55
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.2b38c2ca9f7f1a94(ptr %t) #0 !dbg !57 {
// CHECK:STDOUT:   ret void, !dbg !62
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.f127f30143696d51(ptr %t) #0 !dbg !64 {
// CHECK:STDOUT:   ret void, !dbg !69
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
// CHECK:STDOUT: !9 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !4, line: 13, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !10 = !DILocation(line: 14, column: 3, scope: !9)
// CHECK:STDOUT: !12 = !DILocation(line: 15, column: 3, scope: !9)
// CHECK:STDOUT: !14 = !DILocation(line: 16, column: 3, scope: !9)
// CHECK:STDOUT: !16 = !DILocation(line: 17, column: 3, scope: !9)
// CHECK:STDOUT: !18 = !DILocation(line: 18, column: 3, scope: !9)
// CHECK:STDOUT: !20 = !DILocation(line: 19, column: 3, scope: !9)
// CHECK:STDOUT: !22 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !23 = !{null, !22}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !25 = !{null, !6}
// CHECK:STDOUT: !26 = !DISubroutineType(types: !25)
// CHECK:STDOUT: !27 = !DILocation(line: 13, column: 1, scope: !9)
// CHECK:STDOUT: !28 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !29 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.4f29d15a20894e9b", scope: null, file: !28, line: 35, type: !24, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !35)
// CHECK:STDOUT: !30 = !DILocalVariable(arg: 1, scope: !29, type: !22)
// CHECK:STDOUT: !34 = !DILocation(line: 35, column: 1, scope: !29)
// CHECK:STDOUT: !35 = !{!30}
// CHECK:STDOUT: !36 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.29e6aa423646b61a", scope: null, file: !28, line: 35, type: !24, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !42)
// CHECK:STDOUT: !37 = !DILocalVariable(arg: 1, scope: !36, type: !22)
// CHECK:STDOUT: !41 = !DILocation(line: 35, column: 1, scope: !36)
// CHECK:STDOUT: !42 = !{!37}
// CHECK:STDOUT: !43 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.61b5bb2079973306", scope: null, file: !28, line: 35, type: !24, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !49)
// CHECK:STDOUT: !44 = !DILocalVariable(arg: 1, scope: !43, type: !22)
// CHECK:STDOUT: !48 = !DILocation(line: 35, column: 1, scope: !43)
// CHECK:STDOUT: !49 = !{!44}
// CHECK:STDOUT: !50 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.9045abb8230cbcdf", scope: null, file: !28, line: 35, type: !24, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !56)
// CHECK:STDOUT: !51 = !DILocalVariable(arg: 1, scope: !50, type: !22)
// CHECK:STDOUT: !55 = !DILocation(line: 35, column: 1, scope: !50)
// CHECK:STDOUT: !56 = !{!51}
// CHECK:STDOUT: !57 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.2b38c2ca9f7f1a94", scope: null, file: !28, line: 35, type: !24, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !63)
// CHECK:STDOUT: !58 = !DILocalVariable(arg: 1, scope: !57, type: !22)
// CHECK:STDOUT: !62 = !DILocation(line: 35, column: 1, scope: !57)
// CHECK:STDOUT: !63 = !{!58}
// CHECK:STDOUT: !64 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.f127f30143696d51", scope: null, file: !28, line: 35, type: !26, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !70)
// CHECK:STDOUT: !65 = !DILocalVariable(arg: 1, scope: !64, type: !6)
// CHECK:STDOUT: !69 = !DILocation(line: 35, column: 1, scope: !64)
// CHECK:STDOUT: !70 = !{!65}
// CHECK:STDOUT:
