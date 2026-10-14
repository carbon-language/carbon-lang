// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/call/i32.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/call/i32.carbon

fn Echo(a: i32) -> i32 {
  return a;
}

fn Main() {
  var _: i32 = Echo(1);
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'i32.carbon'
// CHECK:STDOUT: source_filename = "i32.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CEcho.Main(i32 %a) #0 !dbg !25 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %a, !dbg !31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMain.Main() #0 !dbg !33 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca i32, align 4, !dbg !34
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !34
// CHECK:STDOUT:   %Echo.call = call i32 @_CEcho.Main(i32 1), !dbg !35
// CHECK:STDOUT:   store i32 %Echo.call, ptr %_.var, align 4, !dbg !34
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.7f8dce37f94a83bf(ptr %_.var), !dbg !34
// CHECK:STDOUT:   ret void, !dbg !39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.7f8dce37f94a83bf(ptr %t) #0 !dbg !40 {
// CHECK:STDOUT:   ret void, !dbg !45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !18 = !DIFile(filename: "i32.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !{!20, !20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = !{null}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "Echo", linkageName: "_CEcho.Main", scope: null, file: !18, line: 13, type: !22, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !32)
// CHECK:STDOUT: !26 = !DILocalVariable(arg: 1, scope: !25, type: !20)
// CHECK:STDOUT: !31 = !DILocation(line: 14, column: 3, scope: !25)
// CHECK:STDOUT: !32 = !{!26}
// CHECK:STDOUT: !33 = distinct !DISubprogram(name: "Main", linkageName: "_CMain.Main", scope: null, file: !18, line: 17, type: !24, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !34 = !DILocation(line: 18, column: 3, scope: !33)
// CHECK:STDOUT: !35 = !DILocation(line: 18, column: 16, scope: !33)
// CHECK:STDOUT: !37 = !{null, !20}
// CHECK:STDOUT: !38 = !DISubroutineType(types: !37)
// CHECK:STDOUT: !39 = !DILocation(line: 17, column: 1, scope: !33)
// CHECK:STDOUT: !40 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.7f8dce37f94a83bf", scope: null, file: !10, line: 30, type: !38, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !46)
// CHECK:STDOUT: !41 = !DILocalVariable(arg: 1, scope: !40, type: !20)
// CHECK:STDOUT: !45 = !DILocation(line: 30, column: 1, scope: !40)
// CHECK:STDOUT: !46 = !{!41}
// CHECK:STDOUT:
