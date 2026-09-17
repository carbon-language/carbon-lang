// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/generic/import_core_witness.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/generic/import_core_witness.carbon

// --- lib.carbon

library "[[@TEST_NAME]]";

fn F(generic x: i32) {
  // This generates a witness for the destructor of `y` which is reused by the
  // instantiation for the call in `main.carbon`.
  var unused y: i32 = x;
}

// --- main.carbon

import library "lib";

fn Run() {
  F(0);
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'lib.carbon'
// CHECK:STDOUT: source_filename = "lib.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !9 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "lib.carbon", directory: "")
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !7 = !{null, !6}
// CHECK:STDOUT: !8 = !DISubroutineType(types: !7)
// CHECK:STDOUT: !9 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core", scope: null, file: !4, line: 7, type: !8, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !12)
// CHECK:STDOUT: !10 = !DILocalVariable(arg: 1, scope: !9, type: !6)
// CHECK:STDOUT: !11 = !DILocation(line: 7, column: 3, scope: !9)
// CHECK:STDOUT: !12 = !{!10}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'main.carbon'
// CHECK:STDOUT: source_filename = "main.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @main() #0 !dbg !17 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main.79e272ab3b2c3188(), !dbg !18
// CHECK:STDOUT:   ret i32 0, !dbg !21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.79e272ab3b2c3188() #0 !dbg !22 {
// CHECK:STDOUT:   %1 = alloca i32, align 4, !dbg !23
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %1), !dbg !23
// CHECK:STDOUT:   store i32 0, ptr %1, align 4, !dbg !23
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48(ptr %1), !dbg !23
// CHECK:STDOUT:   ret void, !dbg !25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48(ptr %self) #0 !dbg !27 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self), !dbg !31
// CHECK:STDOUT:   ret void, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!14}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "lib.carbon", directory: "")
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !7 = !{null, !6}
// CHECK:STDOUT: !8 = !DISubroutineType(types: !7)
// CHECK:STDOUT: !13 = !DIFile(filename: "main.carbon", directory: "")
// CHECK:STDOUT: !14 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !13, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !15 = !{!6}
// CHECK:STDOUT: !16 = !DISubroutineType(types: !15)
// CHECK:STDOUT: !17 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !13, line: 4, type: !16, spFlags: DISPFlagDefinition, unit: !14)
// CHECK:STDOUT: !18 = !DILocation(line: 5, column: 3, scope: !17)
// CHECK:STDOUT: !19 = !{null}
// CHECK:STDOUT: !20 = !DISubroutineType(types: !19)
// CHECK:STDOUT: !21 = !DILocation(line: 4, column: 1, scope: !17)
// CHECK:STDOUT: !22 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.79e272ab3b2c3188", scope: null, file: !4, line: 4, type: !20, spFlags: DISPFlagDefinition, unit: !14)
// CHECK:STDOUT: !23 = !DILocation(line: 7, column: 3, scope: !22)
// CHECK:STDOUT: !25 = !DILocation(line: 4, column: 1, scope: !22)
// CHECK:STDOUT: !26 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !27 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48", scope: null, file: !26, line: 26, type: !8, spFlags: DISPFlagDefinition, unit: !14, retainedNodes: !33)
// CHECK:STDOUT: !28 = !DILocalVariable(arg: 1, scope: !27, type: !6)
// CHECK:STDOUT: !31 = !DILocation(line: 28, column: 5, scope: !27)
// CHECK:STDOUT: !32 = !DILocation(line: 26, column: 3, scope: !27)
// CHECK:STDOUT: !33 = !{!28}
// CHECK:STDOUT:
