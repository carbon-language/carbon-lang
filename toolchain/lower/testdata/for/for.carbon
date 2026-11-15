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

// CHECK:STDOUT: ; ---
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
// CHECK:STDOUT: define void @_CFor.Main() #0 !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc19_32.1.temp = alloca { i32, i32 }, align 4, !dbg !15
// CHECK:STDOUT:   %var = alloca i32, align 4, !dbg !16
// CHECK:STDOUT:   %.loc19_33.1.temp = alloca <{ i32, i1 }>, align 4, !dbg !16
// CHECK:STDOUT:   call void @_CF.Main(), !dbg !13
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc19_32.1.temp), !dbg !15
// CHECK:STDOUT:   call void @_CRange.Core(ptr %.loc19_32.1.temp, i32 100), !dbg !15
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.NewCursor.call = call i32 @"_CNewCursor.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8"(ptr %.loc19_32.1.temp), !dbg !16
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %var), !dbg !16
// CHECK:STDOUT:   store i32 %IntRange.as.Iterate.impl.NewCursor.call, ptr %var, align 4, !dbg !16
// CHECK:STDOUT:   br label %for.next, !dbg !16
// CHECK:STDOUT:
// CHECK:STDOUT: for.next:                                         ; preds = %for.body, %entry
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc19_33.1.temp), !dbg !16
// CHECK:STDOUT:   call void @"_CNext.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8"(ptr %.loc19_33.1.temp, ptr %.loc19_32.1.temp, ptr %var), !dbg !16
// CHECK:STDOUT:   %Optional.HasValue.call = call i1 @_CHasValue.Optional.Core.29b2425ca9ad75a4(ptr %.loc19_33.1.temp), !dbg !16
// CHECK:STDOUT:   br i1 %Optional.HasValue.call, label %for.body, label %for.done, !dbg !16
// CHECK:STDOUT:
// CHECK:STDOUT: for.body:                                         ; preds = %for.next
// CHECK:STDOUT:   %Optional.Get.call = call i32 @_CGet.Optional.Core.29b2425ca9ad75a4(ptr %.loc19_33.1.temp), !dbg !16
// CHECK:STDOUT:   call void @_CG.Main(), !dbg !23
// CHECK:STDOUT:   br label %for.next, !dbg !24
// CHECK:STDOUT:
// CHECK:STDOUT: for.done:                                         ; preds = %for.next
// CHECK:STDOUT:   call void @_CH.Main(), !dbg !25
// CHECK:STDOUT:   ret void, !dbg !26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CRange.Core(ptr sret({ i32, i32 }), i32)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CNewCursor.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8"(ptr %self) #0 !dbg !30 {
// CHECK:STDOUT:   %start = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 0, !dbg !35
// CHECK:STDOUT:   %1 = load i32, ptr %start, align 4, !dbg !35
// CHECK:STDOUT:   ret i32 %1, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNext.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8"(ptr sret(<{ i32, i1 }>) %return, ptr %self, ptr %cursor) #0 !dbg !38 {
// CHECK:STDOUT:   %1 = alloca i32, align 4, !dbg !45
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %1), !dbg !45
// CHECK:STDOUT:   %2 = load i32, ptr %cursor, align 4, !dbg !46
// CHECK:STDOUT:   store i32 %2, ptr %1, align 4, !dbg !45
// CHECK:STDOUT:   %end = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 1, !dbg !49
// CHECK:STDOUT:   %3 = load i32, ptr %end, align 4, !dbg !49
// CHECK:STDOUT:   %4 = load i32, ptr %1, align 4, !dbg !48
// CHECK:STDOUT:   %5 = icmp slt i32 %4, %3, !dbg !48
// CHECK:STDOUT:   br i1 %5, label %6, label %8, !dbg !50
// CHECK:STDOUT:
// CHECK:STDOUT: 6:                                                ; preds = %0
// CHECK:STDOUT:   call void @"_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8"(ptr %cursor), !dbg !52
// CHECK:STDOUT:   %7 = load i32, ptr %1, align 4, !dbg !55
// CHECK:STDOUT:   call void @_CSome.Optional.Core.29b2425ca9ad75a4(ptr %return, i32 %7), !dbg !56
// CHECK:STDOUT:   ret void, !dbg !57
// CHECK:STDOUT:
// CHECK:STDOUT: 8:                                                ; preds = %0
// CHECK:STDOUT:   call void @_CNone.Optional.Core.29b2425ca9ad75a4(ptr %return), !dbg !58
// CHECK:STDOUT:   ret void, !dbg !61
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CHasValue.Optional.Core.29b2425ca9ad75a4(ptr %self) #0 !dbg !64 {
// CHECK:STDOUT:   %1 = call i1 @"_CHas.c1cc3e41344c17ec:OptionalStorage.Core.b21e3f34d194eaf5"(ptr %self), !dbg !70
// CHECK:STDOUT:   ret i1 %1, !dbg !71
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CGet.Optional.Core.29b2425ca9ad75a4(ptr %self) #0 !dbg !73 {
// CHECK:STDOUT:   %1 = call i32 @"_CGet.c1cc3e41344c17ec:OptionalStorage.Core.b21e3f34d194eaf5"(ptr %self), !dbg !79
// CHECK:STDOUT:   ret i32 %1, !dbg !80
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CInclusiveRange.Core(ptr sret({ i32, i32 }), i32, i32)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8"(ptr %self) #0 !dbg !83 {
// CHECK:STDOUT:   call void @"_COp:thunk:AddAssignWith.b59e374b0649c9ad.Core:Int.9452f4c51951679b.Core:AddAssignWith.ab03190a9e911cb4.Core.306167be98c24f26"(ptr %self, i32 1), !dbg !86
// CHECK:STDOUT:   ret void, !dbg !89
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSome.Optional.Core.29b2425ca9ad75a4(ptr sret(<{ i32, i1 }>) %return, i32 %value) #0 !dbg !91 {
// CHECK:STDOUT:   call void @"_CSome.c1cc3e41344c17ec:OptionalStorage.Core.b21e3f34d194eaf5"(ptr %return, i32 %value), !dbg !96
// CHECK:STDOUT:   ret void, !dbg !97
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CNone.Optional.Core.29b2425ca9ad75a4(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !99 {
// CHECK:STDOUT:   call void @"_CNone.c1cc3e41344c17ec:OptionalStorage.Core.b21e3f34d194eaf5"(ptr %return), !dbg !102
// CHECK:STDOUT:   ret void, !dbg !103
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @"_CHas.c1cc3e41344c17ec:OptionalStorage.Core.b21e3f34d194eaf5"(ptr %value) #0 !dbg !104 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %value, i32 0, i32 1, !dbg !109
// CHECK:STDOUT:   %1 = load i8, ptr %has_value, align 1, !dbg !109
// CHECK:STDOUT:   %2 = trunc i8 %1 to i1, !dbg !109
// CHECK:STDOUT:   ret i1 %2, !dbg !110
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CGet.c1cc3e41344c17ec:OptionalStorage.Core.b21e3f34d194eaf5"(ptr %value) #0 !dbg !112 {
// CHECK:STDOUT:   %value1 = getelementptr inbounds nuw <{ i32, i1 }>, ptr %value, i32 0, i32 0, !dbg !117
// CHECK:STDOUT:   %1 = load i32, ptr %value1, align 4, !dbg !117
// CHECK:STDOUT:   ret i32 %1, !dbg !118
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp:thunk:AddAssignWith.b59e374b0649c9ad.Core:Int.9452f4c51951679b.Core:AddAssignWith.ab03190a9e911cb4.Core.306167be98c24f26"(ptr %self, i32 %other) #2 !dbg !120 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.df8c28cb3df17b7d:ImplicitAs.c95da57eee1d2d0f.Core.25ef994321ea0701"(i32 %other), !dbg !124
// CHECK:STDOUT:   %2 = load i32, ptr %self, align 4, !dbg !124
// CHECK:STDOUT:   %3 = add i32 %2, %1, !dbg !124
// CHECK:STDOUT:   store i32 %3, ptr %self, align 4, !dbg !124
// CHECK:STDOUT:   ret void, !dbg !124
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.c1cc3e41344c17ec:OptionalStorage.Core.b21e3f34d194eaf5"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !128 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 0, !dbg !137
// CHECK:STDOUT:   store i32 %self, ptr %value, align 4, !dbg !137
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !139
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !139
// CHECK:STDOUT:   ret void, !dbg !140
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNone.c1cc3e41344c17ec:OptionalStorage.Core.b21e3f34d194eaf5"(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !142 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !149
// CHECK:STDOUT:   store i8 0, ptr %has_value, align 1, !dbg !149
// CHECK:STDOUT:   ret void, !dbg !150
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.df8c28cb3df17b7d:ImplicitAs.c95da57eee1d2d0f.Core.25ef994321ea0701"(i32 %self) #0 !dbg !151 {
// CHECK:STDOUT:   %1 = call i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self), !dbg !156
// CHECK:STDOUT:   ret i32 %1, !dbg !158
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self) #0 !dbg !160 {
// CHECK:STDOUT:   ret i32 %self, !dbg !166
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 3, 2, 1 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #2 = { alwaysinline nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "for.carbon", directory: "")
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !7 = !{null}
// CHECK:STDOUT: !8 = !DISubroutineType(types: !7)
// CHECK:STDOUT: !9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !10 = !{!9, !6}
// CHECK:STDOUT: !11 = !DISubroutineType(types: !10)
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "For", linkageName: "_CFor.Main", scope: null, file: !4, line: 17, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !13 = !DILocation(line: 18, column: 3, scope: !12)
// CHECK:STDOUT: !15 = !DILocation(line: 19, column: 18, scope: !12)
// CHECK:STDOUT: !16 = !DILocation(line: 19, column: 7, scope: !12)
// CHECK:STDOUT: !17 = !{!6, !9}
// CHECK:STDOUT: !18 = !DISubroutineType(types: !17)
// CHECK:STDOUT: !19 = !{!9, !9, !6}
// CHECK:STDOUT: !20 = !DISubroutineType(types: !19)
// CHECK:STDOUT: !21 = !{!9, !9}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = !DILocation(line: 20, column: 5, scope: !12)
// CHECK:STDOUT: !24 = !DILocation(line: 19, column: 3, scope: !12)
// CHECK:STDOUT: !25 = !DILocation(line: 22, column: 3, scope: !12)
// CHECK:STDOUT: !26 = !DILocation(line: 17, column: 1, scope: !12)
// CHECK:STDOUT: !29 = !DIFile(filename: "{{.*}}/prelude/range.carbon", directory: "")
// CHECK:STDOUT: !30 = distinct !DISubprogram(name: "NewCursor", linkageName: "_CNewCursor.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8", scope: null, file: !29, line: 19, type: !18, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !37)
// CHECK:STDOUT: !31 = !DILocalVariable(arg: 1, scope: !30, type: !9)
// CHECK:STDOUT: !35 = !DILocation(line: 19, column: 43, scope: !30)
// CHECK:STDOUT: !36 = !DILocation(line: 19, column: 36, scope: !30)
// CHECK:STDOUT: !37 = !{!31}
// CHECK:STDOUT: !38 = distinct !DISubprogram(name: "Next", linkageName: "_CNext.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8", scope: null, file: !29, line: 20, type: !20, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !62)
// CHECK:STDOUT: !39 = !DILocalVariable(arg: 1, scope: !38, type: !9)
// CHECK:STDOUT: !40 = !DILocalVariable(arg: 2, scope: !38, type: !6)
// CHECK:STDOUT: !45 = !DILocation(line: 21, column: 7, scope: !38)
// CHECK:STDOUT: !46 = !DILocation(line: 21, column: 27, scope: !38)
// CHECK:STDOUT: !48 = !DILocation(line: 22, column: 11, scope: !38)
// CHECK:STDOUT: !49 = !DILocation(line: 22, column: 19, scope: !38)
// CHECK:STDOUT: !50 = !DILocation(line: 22, column: 10, scope: !38)
// CHECK:STDOUT: !52 = !DILocation(line: 23, column: 9, scope: !38)
// CHECK:STDOUT: !53 = !{null, !6}
// CHECK:STDOUT: !54 = !DISubroutineType(types: !53)
// CHECK:STDOUT: !55 = !DILocation(line: 24, column: 38, scope: !38)
// CHECK:STDOUT: !56 = !DILocation(line: 24, column: 16, scope: !38)
// CHECK:STDOUT: !57 = !DILocation(line: 24, column: 9, scope: !38)
// CHECK:STDOUT: !58 = !DILocation(line: 26, column: 16, scope: !38)
// CHECK:STDOUT: !59 = !{!9}
// CHECK:STDOUT: !60 = !DISubroutineType(types: !59)
// CHECK:STDOUT: !61 = !DILocation(line: 26, column: 9, scope: !38)
// CHECK:STDOUT: !62 = !{!39, !40}
// CHECK:STDOUT: !63 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !64 = distinct !DISubprogram(name: "HasValue", linkageName: "_CHasValue.Optional.Core.29b2425ca9ad75a4", scope: null, file: !63, line: 36, type: !22, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !72)
// CHECK:STDOUT: !65 = !DILocalVariable(arg: 1, scope: !64, type: !9)
// CHECK:STDOUT: !70 = !DILocation(line: 37, column: 12, scope: !64)
// CHECK:STDOUT: !71 = !DILocation(line: 37, column: 5, scope: !64)
// CHECK:STDOUT: !72 = !{!65}
// CHECK:STDOUT: !73 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.Optional.Core.29b2425ca9ad75a4", scope: null, file: !63, line: 39, type: !18, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !81)
// CHECK:STDOUT: !74 = !DILocalVariable(arg: 1, scope: !73, type: !9)
// CHECK:STDOUT: !79 = !DILocation(line: 40, column: 12, scope: !73)
// CHECK:STDOUT: !80 = !DILocation(line: 40, column: 5, scope: !73)
// CHECK:STDOUT: !81 = !{!74}
// CHECK:STDOUT: !82 = !DIFile(filename: "{{.*}}/prelude/types/int.carbon", directory: "")
// CHECK:STDOUT: !83 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8", scope: null, file: !82, line: 396, type: !54, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !90)
// CHECK:STDOUT: !84 = !DILocalVariable(arg: 1, scope: !83, type: !6)
// CHECK:STDOUT: !86 = !DILocation(line: 398, column: 5, scope: !83)
// CHECK:STDOUT: !87 = !{null, !6, !6}
// CHECK:STDOUT: !88 = !DISubroutineType(types: !87)
// CHECK:STDOUT: !89 = !DILocation(line: 396, column: 3, scope: !83)
// CHECK:STDOUT: !90 = !{!84}
// CHECK:STDOUT: !91 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.29b2425ca9ad75a4", scope: null, file: !63, line: 33, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !98)
// CHECK:STDOUT: !92 = !DILocalVariable(arg: 1, scope: !91, type: !6)
// CHECK:STDOUT: !96 = !DILocation(line: 34, column: 12, scope: !91)
// CHECK:STDOUT: !97 = !DILocation(line: 34, column: 5, scope: !91)
// CHECK:STDOUT: !98 = !{!92}
// CHECK:STDOUT: !99 = distinct !DISubprogram(name: "None", linkageName: "_CNone.Optional.Core.29b2425ca9ad75a4", scope: null, file: !63, line: 30, type: !60, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !102 = !DILocation(line: 31, column: 12, scope: !99)
// CHECK:STDOUT: !103 = !DILocation(line: 31, column: 5, scope: !99)
// CHECK:STDOUT: !104 = distinct !DISubprogram(name: "Has", linkageName: "_CHas.c1cc3e41344c17ec:OptionalStorage.Core.b21e3f34d194eaf5", scope: null, file: !63, line: 141, type: !22, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !111)
// CHECK:STDOUT: !105 = !DILocalVariable(arg: 1, scope: !104, type: !9)
// CHECK:STDOUT: !109 = !DILocation(line: 142, column: 12, scope: !104)
// CHECK:STDOUT: !110 = !DILocation(line: 142, column: 5, scope: !104)
// CHECK:STDOUT: !111 = !{!105}
// CHECK:STDOUT: !112 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.c1cc3e41344c17ec:OptionalStorage.Core.b21e3f34d194eaf5", scope: null, file: !63, line: 144, type: !18, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !119)
// CHECK:STDOUT: !113 = !DILocalVariable(arg: 1, scope: !112, type: !9)
// CHECK:STDOUT: !117 = !DILocation(line: 145, column: 12, scope: !112)
// CHECK:STDOUT: !118 = !DILocation(line: 145, column: 5, scope: !112)
// CHECK:STDOUT: !119 = !{!113}
// CHECK:STDOUT: !120 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:AddAssignWith.b59e374b0649c9ad.Core:Int.9452f4c51951679b.Core:AddAssignWith.ab03190a9e911cb4.Core.306167be98c24f26", scope: null, file: !82, line: 331, type: !88, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !127)
// CHECK:STDOUT: !121 = !DILocalVariable(arg: 1, scope: !120, type: !6)
// CHECK:STDOUT: !122 = !DILocalVariable(arg: 2, scope: !120, type: !6)
// CHECK:STDOUT: !124 = !DILocation(line: 331, column: 3, scope: !120)
// CHECK:STDOUT: !125 = !{!6, !6}
// CHECK:STDOUT: !126 = !DISubroutineType(types: !125)
// CHECK:STDOUT: !127 = !{!121, !122}
// CHECK:STDOUT: !128 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.c1cc3e41344c17ec:OptionalStorage.Core.b21e3f34d194eaf5", scope: null, file: !63, line: 131, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !141)
// CHECK:STDOUT: !129 = !DILocalVariable(arg: 1, scope: !128, type: !6)
// CHECK:STDOUT: !137 = !DILocation(line: 137, column: 5, scope: !128)
// CHECK:STDOUT: !139 = !DILocation(line: 138, column: 5, scope: !128)
// CHECK:STDOUT: !140 = !DILocation(line: 139, column: 5, scope: !128)
// CHECK:STDOUT: !141 = !{!129}
// CHECK:STDOUT: !142 = distinct !DISubprogram(name: "None", linkageName: "_CNone.c1cc3e41344c17ec:OptionalStorage.Core.b21e3f34d194eaf5", scope: null, file: !63, line: 125, type: !60, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !149 = !DILocation(line: 128, column: 5, scope: !142)
// CHECK:STDOUT: !150 = !DILocation(line: 129, column: 5, scope: !142)
// CHECK:STDOUT: !151 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.df8c28cb3df17b7d:ImplicitAs.c95da57eee1d2d0f.Core.25ef994321ea0701", scope: null, file: !82, line: 64, type: !126, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !159)
// CHECK:STDOUT: !152 = !DILocalVariable(arg: 1, scope: !151, type: !6)
// CHECK:STDOUT: !156 = !DILocation(line: 66, column: 17, scope: !151)
// CHECK:STDOUT: !158 = !DILocation(line: 66, column: 5, scope: !151)
// CHECK:STDOUT: !159 = !{!152}
// CHECK:STDOUT: !160 = distinct !DISubprogram(name: "AsInt", linkageName: "_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8", scope: null, file: !82, line: 59, type: !126, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !167)
// CHECK:STDOUT: !161 = !DILocalVariable(arg: 1, scope: !160, type: !6)
// CHECK:STDOUT: !166 = !DILocation(line: 59, column: 28, scope: !160)
// CHECK:STDOUT: !167 = !{!161}
// CHECK:STDOUT:
