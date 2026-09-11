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
// CHECK:STDOUT: define void @_CEcho.Main() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMain.Main() #0 !dbg !8 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca {}, align 1, !dbg !9
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !9
// CHECK:STDOUT:   call void @_CEcho.Main(), !dbg !10
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.17de40ae0fb4b1fd(ptr %_.var), !dbg !9
// CHECK:STDOUT:   ret void, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.17de40ae0fb4b1fd(ptr %self) #0 !dbg !12 {
// CHECK:STDOUT:   ret void, !dbg !19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "empty_struct.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Echo", linkageName: "_CEcho.Main", scope: null, file: !1, line: 13, type: !5, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DILocation(line: 14, column: 3, scope: !4)
// CHECK:STDOUT: !8 = distinct !DISubprogram(name: "Main", linkageName: "_CMain.Main", scope: null, file: !1, line: 17, type: !5, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !9 = !DILocation(line: 18, column: 3, scope: !8)
// CHECK:STDOUT: !10 = !DILocation(line: 18, column: 15, scope: !8)
// CHECK:STDOUT: !11 = !DILocation(line: 17, column: 1, scope: !8)
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.17de40ae0fb4b1fd", scope: null, file: !13, line: 24, type: !14, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !17)
// CHECK:STDOUT: !13 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !14 = !DISubroutineType(types: !15)
// CHECK:STDOUT: !15 = !{null, !16}
// CHECK:STDOUT: !16 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !17 = !{!18}
// CHECK:STDOUT: !18 = !DILocalVariable(arg: 1, scope: !12, type: !16)
// CHECK:STDOUT: !19 = !DILocation(line: 24, column: 3, scope: !12)
// CHECK:STDOUT:
