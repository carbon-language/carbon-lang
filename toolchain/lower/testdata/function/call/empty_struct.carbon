// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/call/empty_struct.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/call/empty_struct.carbon

fn Echo(_: {}) -> {} {
  return {};
}

fn Main() {
  var _: {} = Echo({});
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'empty_struct.carbon'
// CHECK:STDOUT: source_filename = "empty_struct.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CEcho.Main() #0 !dbg !20 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !24
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMain.Main() #0 !dbg !25 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca {}, align 1, !dbg !26
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !26
// CHECK:STDOUT:   call void @_CEcho.Main(), !dbg !27
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.456f12c8ffeeaec5(ptr %_.var), !dbg !26
// CHECK:STDOUT:   ret void, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.456f12c8ffeeaec5(ptr %t) #0 !dbg !33 {
// CHECK:STDOUT:   ret void, !dbg !38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!17}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !16 = !DIFile(filename: "empty_struct.carbon", directory: "")
// CHECK:STDOUT: !17 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !16, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !18 = !{null}
// CHECK:STDOUT: !19 = !DISubroutineType(types: !18)
// CHECK:STDOUT: !20 = distinct !DISubprogram(name: "Echo", linkageName: "_CEcho.Main", scope: null, file: !16, line: 13, type: !19, spFlags: DISPFlagDefinition, unit: !17)
// CHECK:STDOUT: !24 = !DILocation(line: 14, column: 3, scope: !20)
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "Main", linkageName: "_CMain.Main", scope: null, file: !16, line: 17, type: !19, spFlags: DISPFlagDefinition, unit: !17)
// CHECK:STDOUT: !26 = !DILocation(line: 18, column: 3, scope: !25)
// CHECK:STDOUT: !27 = !DILocation(line: 18, column: 15, scope: !25)
// CHECK:STDOUT: !29 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !30 = !{null, !29}
// CHECK:STDOUT: !31 = !DISubroutineType(types: !30)
// CHECK:STDOUT: !32 = !DILocation(line: 17, column: 1, scope: !25)
// CHECK:STDOUT: !33 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.456f12c8ffeeaec5", scope: null, file: !6, line: 30, type: !31, spFlags: DISPFlagDefinition, unit: !17, retainedNodes: !39)
// CHECK:STDOUT: !34 = !DILocalVariable(arg: 1, scope: !33, type: !29)
// CHECK:STDOUT: !38 = !DILocation(line: 30, column: 1, scope: !33)
// CHECK:STDOUT: !39 = !{!34}
// CHECK:STDOUT:
