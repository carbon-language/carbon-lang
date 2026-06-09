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
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CIncrSigned.Main() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %from.var = alloca i32, align 4, !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %from.var), !dbg !7
// CHECK:STDOUT:   store i32 0, ptr %from.var, align 4, !dbg !7
// CHECK:STDOUT:   call void @"_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8"(ptr %from.var), !dbg !8
// CHECK:STDOUT:   call void @"_COp.5e27612b9dd31a14:core.Destroy.Core"(ptr %from.var), !dbg !7
// CHECK:STDOUT:   ret void, !dbg !9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.5e27612b9dd31a14:core.Destroy.Core"(ptr %self) #0 !dbg !10 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8"(ptr %self) #0 !dbg !17 {
// CHECK:STDOUT:   call void @"_COp:thunk:AddAssignWith.6f9e8c1abf8bb3ea.Core:Int.9452f4c51951679b.Core:AddAssignWith.bcc2a64c00f3554f.Core.44914b6ff6901661"(ptr %self, i32 1), !dbg !21
// CHECK:STDOUT:   ret void, !dbg !22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp:thunk:AddAssignWith.6f9e8c1abf8bb3ea.Core:Int.9452f4c51951679b.Core:AddAssignWith.bcc2a64c00f3554f.Core.44914b6ff6901661"(ptr %self, i32 %other) #2 !dbg !23 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.40372ba0ea03ef47"(i32 %other), !dbg !29
// CHECK:STDOUT:   %2 = load i32, ptr %self, align 4, !dbg !30
// CHECK:STDOUT:   %3 = add i32 %2, %1, !dbg !30
// CHECK:STDOUT:   store i32 %3, ptr %self, align 4, !dbg !30
// CHECK:STDOUT:   ret void, !dbg !30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.40372ba0ea03ef47"(i32 %self) #0 !dbg !31 {
// CHECK:STDOUT:   %1 = call i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self), !dbg !36
// CHECK:STDOUT:   ret i32 %1, !dbg !37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self) #0 !dbg !38 {
// CHECK:STDOUT:   ret i32 %self, !dbg !41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #2 = { alwaysinline nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "increment.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "IncrSigned", linkageName: "_CIncrSigned.Main", scope: null, file: !3, line: 4, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DILocation(line: 5, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 6, column: 3, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 4, column: 1, scope: !4)
// CHECK:STDOUT: !10 = distinct !DISubprogram(name: "Op", linkageName: "_COp.5e27612b9dd31a14:core.Destroy.Core", scope: null, file: !3, line: 5, type: !11, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !14)
// CHECK:STDOUT: !11 = !DISubroutineType(types: !12)
// CHECK:STDOUT: !12 = !{null, !13}
// CHECK:STDOUT: !13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !14 = !{!15}
// CHECK:STDOUT: !15 = !DILocalVariable(arg: 1, scope: !10, type: !13)
// CHECK:STDOUT: !16 = !DILocation(line: 5, column: 3, scope: !10)
// CHECK:STDOUT: !17 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8", scope: null, file: !18, line: 370, type: !11, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !19)
// CHECK:STDOUT: !18 = !DIFile(filename: "{{.*}}/prelude/types/int.carbon", directory: "")
// CHECK:STDOUT: !19 = !{!20}
// CHECK:STDOUT: !20 = !DILocalVariable(arg: 1, scope: !17, type: !13)
// CHECK:STDOUT: !21 = !DILocation(line: 372, column: 5, scope: !17)
// CHECK:STDOUT: !22 = !DILocation(line: 370, column: 3, scope: !17)
// CHECK:STDOUT: !23 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:AddAssignWith.6f9e8c1abf8bb3ea.Core:Int.9452f4c51951679b.Core:AddAssignWith.bcc2a64c00f3554f.Core.44914b6ff6901661", scope: null, file: !18, line: 306, type: !24, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !26)
// CHECK:STDOUT: !24 = !DISubroutineType(types: !25)
// CHECK:STDOUT: !25 = !{null, !13, !13}
// CHECK:STDOUT: !26 = !{!27, !28}
// CHECK:STDOUT: !27 = !DILocalVariable(arg: 1, scope: !23, type: !13)
// CHECK:STDOUT: !28 = !DILocalVariable(arg: 2, scope: !23, type: !13)
// CHECK:STDOUT: !29 = !DILocation(line: 4294967295, scope: !23)
// CHECK:STDOUT: !30 = !DILocation(line: 306, column: 3, scope: !23)
// CHECK:STDOUT: !31 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.40372ba0ea03ef47", scope: null, file: !18, line: 63, type: !32, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !34)
// CHECK:STDOUT: !32 = !DISubroutineType(types: !33)
// CHECK:STDOUT: !33 = !{!13, !13}
// CHECK:STDOUT: !34 = !{!35}
// CHECK:STDOUT: !35 = !DILocalVariable(arg: 1, scope: !31, type: !13)
// CHECK:STDOUT: !36 = !DILocation(line: 65, column: 17, scope: !31)
// CHECK:STDOUT: !37 = !DILocation(line: 65, column: 5, scope: !31)
// CHECK:STDOUT: !38 = distinct !DISubprogram(name: "AsInt", linkageName: "_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8", scope: null, file: !18, line: 58, type: !32, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !39)
// CHECK:STDOUT: !39 = !{!40}
// CHECK:STDOUT: !40 = !DILocalVariable(arg: 1, scope: !38, type: !13)
// CHECK:STDOUT: !41 = !DILocation(line: 58, column: 28, scope: !38)
