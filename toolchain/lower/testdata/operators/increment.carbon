// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/operators/increment.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/operators/increment.carbon

// --- increment.carbon

library "[[@TEST_NAME]]";

fn IncrSigned() {
  var from: i32 = 0;
  ++from;
}

// CHECK:STDOUT: ; ModuleID = 'increment.carbon'
// CHECK:STDOUT: source_filename = "increment.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CIncrSigned.Main() !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %from.var = alloca i32, align 4, !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 4, ptr %from.var), !dbg !7
// CHECK:STDOUT:   store i32 0, ptr %from.var, align 4, !dbg !7
// CHECK:STDOUT:   call void @"_COp.Int.Core:Inc.Core.be1e879c1ad406d8"(ptr %from.var), !dbg !8
// CHECK:STDOUT:   %.loc5_3.2 = load i32, ptr %from.var, align 4, !dbg !7
// CHECK:STDOUT:   ret void, !dbg !9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(i64 immarg, ptr captures(none)) #0
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr void @"_COp.Int.Core:Inc.Core.be1e879c1ad406d8"(ptr %self) !dbg !10 {
// CHECK:STDOUT:   call void @"_COp:thunk.Int.Core:AddAssignWith.Core.25a9a5e901f5b032"(ptr %self, i32 1), !dbg !12
// CHECK:STDOUT:   ret void, !dbg !13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr void @"_COp:thunk.Int.Core:AddAssignWith.Core.25a9a5e901f5b032"(ptr %self, i32 %other) !dbg !14 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.8b3d5d6a6c17be04:ImplicitAs.Core.b88d1103f417c6d4"(i32 %other), !dbg !15
// CHECK:STDOUT:   %2 = load i32, ptr %self, align 4, !dbg !16
// CHECK:STDOUT:   %3 = add i32 %2, %1, !dbg !16
// CHECK:STDOUT:   store i32 %3, ptr %self, align 4, !dbg !16
// CHECK:STDOUT:   ret void, !dbg !16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.8b3d5d6a6c17be04:ImplicitAs.Core.b88d1103f417c6d4"(i32 %self) !dbg !17 {
// CHECK:STDOUT:   ret i32 %self, !dbg !19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "increment.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "IncrSigned", linkageName: "_CIncrSigned.Main", scope: null, file: !3, line: 4, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 5, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 6, column: 3, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 4, column: 1, scope: !4)
// CHECK:STDOUT: !10 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Int.Core:Inc.Core.be1e879c1ad406d8", scope: null, file: !11, line: 331, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !11 = !DIFile(filename: "{{.*}}/prelude/types/int.carbon", directory: "")
// CHECK:STDOUT: !12 = !DILocation(line: 333, column: 5, scope: !10)
// CHECK:STDOUT: !13 = !DILocation(line: 331, column: 3, scope: !10)
// CHECK:STDOUT: !14 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk.Int.Core:AddAssignWith.Core.25a9a5e901f5b032", scope: null, file: !11, line: 267, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !15 = !DILocation(line: 4294967295, scope: !14)
// CHECK:STDOUT: !16 = !DILocation(line: 267, column: 3, scope: !14)
// CHECK:STDOUT: !17 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.8b3d5d6a6c17be04:ImplicitAs.Core.b88d1103f417c6d4", scope: null, file: !18, line: 18, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !18 = !DIFile(filename: "{{.*}}/prelude/operators/as.carbon", directory: "")
// CHECK:STDOUT: !19 = !DILocation(line: 18, column: 38, scope: !17)
