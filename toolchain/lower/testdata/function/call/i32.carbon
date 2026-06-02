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

// CHECK:STDOUT: ; ModuleID = 'i32.carbon'
// CHECK:STDOUT: source_filename = "i32.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CEcho.Main(i32 %a) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %a, !dbg !10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMain.Main() #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca i32, align 4, !dbg !14
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !14
// CHECK:STDOUT:   %Echo.call = call i32 @_CEcho.Main(i32 1), !dbg !15
// CHECK:STDOUT:   store i32 %Echo.call, ptr %_.var, align 4, !dbg !14
// CHECK:STDOUT:   call void @"_COp.1fa36cd3e044ac37:core.Destroy.Core"(ptr %_.var), !dbg !14
// CHECK:STDOUT:   ret void, !dbg !16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.1fa36cd3e044ac37:core.Destroy.Core"(ptr %self) #0 !dbg !17 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "i32.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Echo", linkageName: "_CEcho.Main", scope: null, file: !3, line: 13, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !8)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7, !7}
// CHECK:STDOUT: !7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !8 = !{!9}
// CHECK:STDOUT: !9 = !DILocalVariable(arg: 1, scope: !4, type: !7)
// CHECK:STDOUT: !10 = !DILocation(line: 14, column: 3, scope: !4)
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "Main", linkageName: "_CMain.Main", scope: null, file: !3, line: 17, type: !12, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !12 = !DISubroutineType(types: !13)
// CHECK:STDOUT: !13 = !{null}
// CHECK:STDOUT: !14 = !DILocation(line: 18, column: 3, scope: !11)
// CHECK:STDOUT: !15 = !DILocation(line: 18, column: 16, scope: !11)
// CHECK:STDOUT: !16 = !DILocation(line: 17, column: 1, scope: !11)
// CHECK:STDOUT: !17 = distinct !DISubprogram(name: "Op", linkageName: "_COp.1fa36cd3e044ac37:core.Destroy.Core", scope: null, file: !3, line: 18, type: !18, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !20)
// CHECK:STDOUT: !18 = !DISubroutineType(types: !19)
// CHECK:STDOUT: !19 = !{null, !7}
// CHECK:STDOUT: !20 = !{!21}
// CHECK:STDOUT: !21 = !DILocalVariable(arg: 1, scope: !17, type: !7)
// CHECK:STDOUT: !22 = !DILocation(line: 18, column: 3, scope: !17)
