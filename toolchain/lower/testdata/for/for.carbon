// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/for/for.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/for/for.carbon

import Core library "range";

fn F();
fn G();
fn H();

fn For() {
  F();
  for (_: i32 in Core.Range(100)) {
    G();
  }
  H();
}

// CHECK:STDOUT: ; ModuleID = 'for.carbon'
// CHECK:STDOUT: source_filename = "for.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CF.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CG.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CH.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CFor.Main() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc21_32.1.temp = alloca { i32, i32 }, align 8, !dbg !7
// CHECK:STDOUT:   %var = alloca i32, align 4, !dbg !8
// CHECK:STDOUT:   %.loc21_33.1.temp = alloca <{ i32, i1 }>, align 8, !dbg !8
// CHECK:STDOUT:   call void @_CF.Main(), !dbg !9
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc21_32.1.temp), !dbg !7
// CHECK:STDOUT:   call void @_CRange.Core(ptr %.loc21_32.1.temp, i32 100), !dbg !7
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.NewCursor.call = call i32 @"_CNewCursor.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8"(ptr %.loc21_32.1.temp), !dbg !8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %var), !dbg !8
// CHECK:STDOUT:   store i32 %IntRange.as.Iterate.impl.NewCursor.call, ptr %var, align 4, !dbg !8
// CHECK:STDOUT:   br label %for.next, !dbg !8
// CHECK:STDOUT:
// CHECK:STDOUT: for.next:                                         ; preds = %for.body, %entry
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc21_33.1.temp), !dbg !8
// CHECK:STDOUT:   call void @"_CNext.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8"(ptr %.loc21_33.1.temp, ptr %.loc21_32.1.temp, ptr %var), !dbg !8
// CHECK:STDOUT:   %Optional.HasValue.call = call i1 @_CHasValue.Optional.Core.3b28e338799fefcf(ptr %.loc21_33.1.temp), !dbg !8
// CHECK:STDOUT:   br i1 %Optional.HasValue.call, label %for.body, label %for.done, !dbg !8
// CHECK:STDOUT:
// CHECK:STDOUT: for.body:                                         ; preds = %for.next
// CHECK:STDOUT:   %Optional.Get.call = call i32 @_CGet.Optional.Core.3b28e338799fefcf(ptr %.loc21_33.1.temp), !dbg !8
// CHECK:STDOUT:   call void @_CG.Main(), !dbg !10
// CHECK:STDOUT:   br label %for.next, !dbg !11
// CHECK:STDOUT:
// CHECK:STDOUT: for.done:                                         ; preds = %for.next
// CHECK:STDOUT:   call void @_CH.Main(), !dbg !12
// CHECK:STDOUT:   call void @"_COp.236a1799d8d1092c:core.Destroy.Core"(ptr %.loc21_33.1.temp), !dbg !8
// CHECK:STDOUT:   call void @"_COp.94e3ca86614aba5d:core.Destroy.Core"(ptr %var), !dbg !8
// CHECK:STDOUT:   call void @"_COp.d99cd304dde232b0:core.Destroy.Core"(ptr %.loc21_32.1.temp), !dbg !7
// CHECK:STDOUT:   ret void, !dbg !13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CRange.Core(ptr sret({ i32, i32 }), i32)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.94e3ca86614aba5d:core.Destroy.Core"(ptr %self) #0 !dbg !14 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.58cdb9d721e0f2e8:core.Destroy.Core"(ptr %self) #0 !dbg !22 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.236a1799d8d1092c:core.Destroy.Core"(ptr %self) #0 !dbg !30 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.d99cd304dde232b0:core.Destroy.Core"(ptr %self) #0 !dbg !34 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CNewCursor.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8"(ptr %self) #0 !dbg !38 {
// CHECK:STDOUT:   %start = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 0, !dbg !43
// CHECK:STDOUT:   %1 = load i32, ptr %start, align 4, !dbg !43
// CHECK:STDOUT:   ret i32 %1, !dbg !44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNext.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8"(ptr sret(<{ i32, i1 }>) %return, ptr %self, ptr %cursor) #0 !dbg !45 {
// CHECK:STDOUT:   %1 = alloca i32, align 4, !dbg !51
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %1), !dbg !51
// CHECK:STDOUT:   %2 = load i32, ptr %cursor, align 4, !dbg !52
// CHECK:STDOUT:   store i32 %2, ptr %1, align 4, !dbg !51
// CHECK:STDOUT:   %end = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 1, !dbg !53
// CHECK:STDOUT:   %3 = load i32, ptr %end, align 4, !dbg !53
// CHECK:STDOUT:   %4 = load i32, ptr %1, align 4, !dbg !54
// CHECK:STDOUT:   %5 = icmp slt i32 %4, %3, !dbg !54
// CHECK:STDOUT:   br i1 %5, label %6, label %8, !dbg !55
// CHECK:STDOUT:
// CHECK:STDOUT: 6:                                                ; preds = %0
// CHECK:STDOUT:   call void @"_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8"(ptr %cursor), !dbg !56
// CHECK:STDOUT:   %7 = load i32, ptr %1, align 4, !dbg !57
// CHECK:STDOUT:   call void @_CSome.Optional.Core.3b28e338799fefcf(ptr %return, i32 %7), !dbg !58
// CHECK:STDOUT:   call void @"_COp.94e3ca86614aba5d:core.Destroy.Core"(ptr %1), !dbg !51
// CHECK:STDOUT:   ret void, !dbg !59
// CHECK:STDOUT:
// CHECK:STDOUT: 8:                                                ; preds = %0
// CHECK:STDOUT:   call void @_CNone.Optional.Core.3b28e338799fefcf(ptr %return), !dbg !60
// CHECK:STDOUT:   call void @"_COp.94e3ca86614aba5d:core.Destroy.Core"(ptr %1), !dbg !51
// CHECK:STDOUT:   ret void, !dbg !61
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CHasValue.Optional.Core.3b28e338799fefcf(ptr %self) #0 !dbg !62 {
// CHECK:STDOUT:   %1 = call i1 @"_CHas.ca6ae00af8d27ba2:OptionalStorage.Core.df89832508ab6447"(ptr %self), !dbg !67
// CHECK:STDOUT:   ret i1 %1, !dbg !68
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CGet.Optional.Core.3b28e338799fefcf(ptr %self) #0 !dbg !69 {
// CHECK:STDOUT:   %1 = call i32 @"_CGet.ca6ae00af8d27ba2:OptionalStorage.Core.df89832508ab6447"(ptr %self), !dbg !72
// CHECK:STDOUT:   ret i32 %1, !dbg !73
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CInclusiveRange.Core(ptr sret({ i32, i32 }), i32, i32)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8"(ptr %self) #0 !dbg !74 {
// CHECK:STDOUT:   call void @"_COp:thunk:AddAssignWith.ad7ccd31b6bdf5c9.Core:Int.9452f4c51951679b.Core:AddAssignWith.78e744410a40fd74.Core.9c83a169ad974fa6"(ptr %self, i32 1), !dbg !78
// CHECK:STDOUT:   ret void, !dbg !79
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSome.Optional.Core.3b28e338799fefcf(ptr sret(<{ i32, i1 }>) %return, i32 %value) #0 !dbg !80 {
// CHECK:STDOUT:   call void @"_CSome.ca6ae00af8d27ba2:OptionalStorage.Core.df89832508ab6447"(ptr %return, i32 %value), !dbg !85
// CHECK:STDOUT:   ret void, !dbg !86
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CNone.Optional.Core.3b28e338799fefcf(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !87 {
// CHECK:STDOUT:   call void @"_CNone.ca6ae00af8d27ba2:OptionalStorage.Core.df89832508ab6447"(ptr %return), !dbg !90
// CHECK:STDOUT:   ret void, !dbg !91
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @"_CHas.ca6ae00af8d27ba2:OptionalStorage.Core.df89832508ab6447"(ptr %value) #0 !dbg !92 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %value, i32 0, i32 1, !dbg !95
// CHECK:STDOUT:   %1 = load i8, ptr %has_value, align 1, !dbg !95
// CHECK:STDOUT:   %2 = trunc i8 %1 to i1, !dbg !95
// CHECK:STDOUT:   ret i1 %2, !dbg !96
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CGet.ca6ae00af8d27ba2:OptionalStorage.Core.df89832508ab6447"(ptr %value) #0 !dbg !97 {
// CHECK:STDOUT:   %value1 = getelementptr inbounds nuw <{ i32, i1 }>, ptr %value, i32 0, i32 0, !dbg !100
// CHECK:STDOUT:   %1 = load i32, ptr %value1, align 4, !dbg !100
// CHECK:STDOUT:   ret i32 %1, !dbg !101
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp:thunk:AddAssignWith.ad7ccd31b6bdf5c9.Core:Int.9452f4c51951679b.Core:AddAssignWith.78e744410a40fd74.Core.9c83a169ad974fa6"(ptr %self, i32 %other) #2 !dbg !102 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.b912251d28f69b6d:ImplicitAs.a629ad9d3aac3895.Core.245bb80fde7c8401"(i32 %other), !dbg !108
// CHECK:STDOUT:   %2 = load i32, ptr %self, align 4, !dbg !109
// CHECK:STDOUT:   %3 = add i32 %2, %1, !dbg !109
// CHECK:STDOUT:   store i32 %3, ptr %self, align 4, !dbg !109
// CHECK:STDOUT:   ret void, !dbg !109
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.ca6ae00af8d27ba2:OptionalStorage.Core.df89832508ab6447"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !110 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 0, !dbg !113
// CHECK:STDOUT:   store i32 %self, ptr %value, align 4, !dbg !113
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !114
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !114
// CHECK:STDOUT:   ret void, !dbg !115
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNone.ca6ae00af8d27ba2:OptionalStorage.Core.df89832508ab6447"(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !116 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !117
// CHECK:STDOUT:   store i8 0, ptr %has_value, align 1, !dbg !117
// CHECK:STDOUT:   ret void, !dbg !118
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.b912251d28f69b6d:ImplicitAs.a629ad9d3aac3895.Core.245bb80fde7c8401"(i32 %self) #0 !dbg !119 {
// CHECK:STDOUT:   %1 = call i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self), !dbg !124
// CHECK:STDOUT:   ret i32 %1, !dbg !125
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self) #0 !dbg !126 {
// CHECK:STDOUT:   ret i32 %self, !dbg !129
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 3, 2, 1 }
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
// CHECK:STDOUT: !3 = !DIFile(filename: "for.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "For", linkageName: "_CFor.Main", scope: null, file: !3, line: 19, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DILocation(line: 21, column: 18, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 21, column: 7, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 20, column: 3, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 22, column: 5, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 21, column: 3, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 24, column: 3, scope: !4)
// CHECK:STDOUT: !13 = !DILocation(line: 19, column: 1, scope: !4)
// CHECK:STDOUT: !14 = distinct !DISubprogram(name: "Op", linkageName: "_COp.94e3ca86614aba5d:core.Destroy.Core", scope: null, file: !15, line: 24, type: !16, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !19)
// CHECK:STDOUT: !15 = !DIFile(filename: "{{.*}}/range.carbon", directory: "")
// CHECK:STDOUT: !16 = !DISubroutineType(types: !17)
// CHECK:STDOUT: !17 = !{null, !18}
// CHECK:STDOUT: !18 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !19 = !{!20}
// CHECK:STDOUT: !20 = !DILocalVariable(arg: 1, scope: !14, type: !18)
// CHECK:STDOUT: !21 = !DILocation(line: 24, column: 65, scope: !14)
// CHECK:STDOUT: !22 = distinct !DISubprogram(name: "Op", linkageName: "_COp.58cdb9d721e0f2e8:core.Destroy.Core", scope: null, file: !23, line: 124, type: !24, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !27)
// CHECK:STDOUT: !23 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !24 = !DISubroutineType(types: !25)
// CHECK:STDOUT: !25 = !{null, !26}
// CHECK:STDOUT: !26 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !27 = !{!28}
// CHECK:STDOUT: !28 = !DILocalVariable(arg: 1, scope: !22, type: !26)
// CHECK:STDOUT: !29 = !DILocation(line: 124, column: 19, scope: !22)
// CHECK:STDOUT: !30 = distinct !DISubprogram(name: "Op", linkageName: "_COp.236a1799d8d1092c:core.Destroy.Core", scope: null, file: !3, line: 21, type: !24, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !31)
// CHECK:STDOUT: !31 = !{!32}
// CHECK:STDOUT: !32 = !DILocalVariable(arg: 1, scope: !30, type: !26)
// CHECK:STDOUT: !33 = !DILocation(line: 21, column: 7, scope: !30)
// CHECK:STDOUT: !34 = distinct !DISubprogram(name: "Op", linkageName: "_COp.d99cd304dde232b0:core.Destroy.Core", scope: null, file: !3, line: 21, type: !24, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !35)
// CHECK:STDOUT: !35 = !{!36}
// CHECK:STDOUT: !36 = !DILocalVariable(arg: 1, scope: !34, type: !26)
// CHECK:STDOUT: !37 = !DILocation(line: 21, column: 18, scope: !34)
// CHECK:STDOUT: !38 = distinct !DISubprogram(name: "NewCursor", linkageName: "_CNewCursor.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8", scope: null, file: !15, line: 25, type: !39, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !41)
// CHECK:STDOUT: !39 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !40 = !{!18, !26}
// CHECK:STDOUT: !41 = !{!42}
// CHECK:STDOUT: !42 = !DILocalVariable(arg: 1, scope: !38, type: !26)
// CHECK:STDOUT: !43 = !DILocation(line: 25, column: 51, scope: !38)
// CHECK:STDOUT: !44 = !DILocation(line: 25, column: 44, scope: !38)
// CHECK:STDOUT: !45 = distinct !DISubprogram(name: "Next", linkageName: "_CNext.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8", scope: null, file: !15, line: 26, type: !46, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !48)
// CHECK:STDOUT: !46 = !DISubroutineType(types: !47)
// CHECK:STDOUT: !47 = !{!26, !26, !26}
// CHECK:STDOUT: !48 = !{!49, !50}
// CHECK:STDOUT: !49 = !DILocalVariable(arg: 1, scope: !45, type: !26)
// CHECK:STDOUT: !50 = !DILocalVariable(arg: 2, scope: !45, type: !26)
// CHECK:STDOUT: !51 = !DILocation(line: 27, column: 7, scope: !45)
// CHECK:STDOUT: !52 = !DILocation(line: 27, column: 27, scope: !45)
// CHECK:STDOUT: !53 = !DILocation(line: 28, column: 19, scope: !45)
// CHECK:STDOUT: !54 = !DILocation(line: 28, column: 11, scope: !45)
// CHECK:STDOUT: !55 = !DILocation(line: 28, column: 10, scope: !45)
// CHECK:STDOUT: !56 = !DILocation(line: 29, column: 9, scope: !45)
// CHECK:STDOUT: !57 = !DILocation(line: 30, column: 38, scope: !45)
// CHECK:STDOUT: !58 = !DILocation(line: 30, column: 16, scope: !45)
// CHECK:STDOUT: !59 = !DILocation(line: 30, column: 9, scope: !45)
// CHECK:STDOUT: !60 = !DILocation(line: 32, column: 16, scope: !45)
// CHECK:STDOUT: !61 = !DILocation(line: 32, column: 9, scope: !45)
// CHECK:STDOUT: !62 = distinct !DISubprogram(name: "HasValue", linkageName: "_CHasValue.Optional.Core.3b28e338799fefcf", scope: null, file: !23, line: 36, type: !63, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !65)
// CHECK:STDOUT: !63 = !DISubroutineType(types: !64)
// CHECK:STDOUT: !64 = !{!26, !26}
// CHECK:STDOUT: !65 = !{!66}
// CHECK:STDOUT: !66 = !DILocalVariable(arg: 1, scope: !62, type: !26)
// CHECK:STDOUT: !67 = !DILocation(line: 37, column: 12, scope: !62)
// CHECK:STDOUT: !68 = !DILocation(line: 37, column: 5, scope: !62)
// CHECK:STDOUT: !69 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.Optional.Core.3b28e338799fefcf", scope: null, file: !23, line: 39, type: !39, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !70)
// CHECK:STDOUT: !70 = !{!71}
// CHECK:STDOUT: !71 = !DILocalVariable(arg: 1, scope: !69, type: !26)
// CHECK:STDOUT: !72 = !DILocation(line: 40, column: 12, scope: !69)
// CHECK:STDOUT: !73 = !DILocation(line: 40, column: 5, scope: !69)
// CHECK:STDOUT: !74 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8", scope: null, file: !75, line: 363, type: !16, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !76)
// CHECK:STDOUT: !75 = !DIFile(filename: "{{.*}}/prelude/types/int.carbon", directory: "")
// CHECK:STDOUT: !76 = !{!77}
// CHECK:STDOUT: !77 = !DILocalVariable(arg: 1, scope: !74, type: !18)
// CHECK:STDOUT: !78 = !DILocation(line: 365, column: 5, scope: !74)
// CHECK:STDOUT: !79 = !DILocation(line: 363, column: 3, scope: !74)
// CHECK:STDOUT: !80 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.3b28e338799fefcf", scope: null, file: !23, line: 33, type: !81, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !83)
// CHECK:STDOUT: !81 = !DISubroutineType(types: !82)
// CHECK:STDOUT: !82 = !{!26, !18}
// CHECK:STDOUT: !83 = !{!84}
// CHECK:STDOUT: !84 = !DILocalVariable(arg: 1, scope: !80, type: !18)
// CHECK:STDOUT: !85 = !DILocation(line: 34, column: 12, scope: !80)
// CHECK:STDOUT: !86 = !DILocation(line: 34, column: 5, scope: !80)
// CHECK:STDOUT: !87 = distinct !DISubprogram(name: "None", linkageName: "_CNone.Optional.Core.3b28e338799fefcf", scope: null, file: !23, line: 30, type: !88, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !88 = !DISubroutineType(types: !89)
// CHECK:STDOUT: !89 = !{!26}
// CHECK:STDOUT: !90 = !DILocation(line: 31, column: 12, scope: !87)
// CHECK:STDOUT: !91 = !DILocation(line: 31, column: 5, scope: !87)
// CHECK:STDOUT: !92 = distinct !DISubprogram(name: "Has", linkageName: "_CHas.ca6ae00af8d27ba2:OptionalStorage.Core.df89832508ab6447", scope: null, file: !23, line: 141, type: !63, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !93)
// CHECK:STDOUT: !93 = !{!94}
// CHECK:STDOUT: !94 = !DILocalVariable(arg: 1, scope: !92, type: !26)
// CHECK:STDOUT: !95 = !DILocation(line: 142, column: 12, scope: !92)
// CHECK:STDOUT: !96 = !DILocation(line: 142, column: 5, scope: !92)
// CHECK:STDOUT: !97 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.ca6ae00af8d27ba2:OptionalStorage.Core.df89832508ab6447", scope: null, file: !23, line: 144, type: !39, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !98)
// CHECK:STDOUT: !98 = !{!99}
// CHECK:STDOUT: !99 = !DILocalVariable(arg: 1, scope: !97, type: !26)
// CHECK:STDOUT: !100 = !DILocation(line: 145, column: 12, scope: !97)
// CHECK:STDOUT: !101 = !DILocation(line: 145, column: 5, scope: !97)
// CHECK:STDOUT: !102 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:AddAssignWith.ad7ccd31b6bdf5c9.Core:Int.9452f4c51951679b.Core:AddAssignWith.78e744410a40fd74.Core.9c83a169ad974fa6", scope: null, file: !75, line: 299, type: !103, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !105)
// CHECK:STDOUT: !103 = !DISubroutineType(types: !104)
// CHECK:STDOUT: !104 = !{null, !18, !18}
// CHECK:STDOUT: !105 = !{!106, !107}
// CHECK:STDOUT: !106 = !DILocalVariable(arg: 1, scope: !102, type: !18)
// CHECK:STDOUT: !107 = !DILocalVariable(arg: 2, scope: !102, type: !18)
// CHECK:STDOUT: !108 = !DILocation(line: 4294967295, scope: !102)
// CHECK:STDOUT: !109 = !DILocation(line: 299, column: 3, scope: !102)
// CHECK:STDOUT: !110 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.ca6ae00af8d27ba2:OptionalStorage.Core.df89832508ab6447", scope: null, file: !23, line: 131, type: !81, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !111)
// CHECK:STDOUT: !111 = !{!112}
// CHECK:STDOUT: !112 = !DILocalVariable(arg: 1, scope: !110, type: !18)
// CHECK:STDOUT: !113 = !DILocation(line: 137, column: 5, scope: !110)
// CHECK:STDOUT: !114 = !DILocation(line: 138, column: 5, scope: !110)
// CHECK:STDOUT: !115 = !DILocation(line: 139, column: 5, scope: !110)
// CHECK:STDOUT: !116 = distinct !DISubprogram(name: "None", linkageName: "_CNone.ca6ae00af8d27ba2:OptionalStorage.Core.df89832508ab6447", scope: null, file: !23, line: 125, type: !88, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !117 = !DILocation(line: 128, column: 5, scope: !116)
// CHECK:STDOUT: !118 = !DILocation(line: 129, column: 5, scope: !116)
// CHECK:STDOUT: !119 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.b912251d28f69b6d:ImplicitAs.a629ad9d3aac3895.Core.245bb80fde7c8401", scope: null, file: !75, line: 62, type: !120, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !122)
// CHECK:STDOUT: !120 = !DISubroutineType(types: !121)
// CHECK:STDOUT: !121 = !{!18, !18}
// CHECK:STDOUT: !122 = !{!123}
// CHECK:STDOUT: !123 = !DILocalVariable(arg: 1, scope: !119, type: !18)
// CHECK:STDOUT: !124 = !DILocation(line: 64, column: 17, scope: !119)
// CHECK:STDOUT: !125 = !DILocation(line: 64, column: 5, scope: !119)
// CHECK:STDOUT: !126 = distinct !DISubprogram(name: "AsInt", linkageName: "_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8", scope: null, file: !75, line: 57, type: !120, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !127)
// CHECK:STDOUT: !127 = !{!128}
// CHECK:STDOUT: !128 = !DILocalVariable(arg: 1, scope: !126, type: !18)
// CHECK:STDOUT: !129 = !DILocation(line: 57, column: 36, scope: !126)
