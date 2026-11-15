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
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.c6daab0befde28a8(ptr %.loc19_33.1.temp), !dbg !16
// CHECK:STDOUT:   br label %for.next, !dbg !26
// CHECK:STDOUT:
// CHECK:STDOUT: for.done:                                         ; preds = %for.next
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.c6daab0befde28a8(ptr %.loc19_33.1.temp), !dbg !16
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.9a03059f269926ed(ptr %var), !dbg !16
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.ab8fcbb9cd97e8e7(ptr %.loc19_32.1.temp), !dbg !15
// CHECK:STDOUT:   call void @_CH.Main(), !dbg !29
// CHECK:STDOUT:   ret void, !dbg !30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CRange.Core(ptr sret({ i32, i32 }), i32)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CNewCursor.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8"(ptr %self) #0 !dbg !34 {
// CHECK:STDOUT:   %start = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 0, !dbg !39
// CHECK:STDOUT:   %1 = load i32, ptr %start, align 4, !dbg !39
// CHECK:STDOUT:   ret i32 %1, !dbg !40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNext.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8"(ptr sret(<{ i32, i1 }>) %return, ptr %self, ptr %cursor) #0 !dbg !42 {
// CHECK:STDOUT:   %1 = alloca i32, align 4, !dbg !49
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %1), !dbg !49
// CHECK:STDOUT:   %2 = load i32, ptr %cursor, align 4, !dbg !50
// CHECK:STDOUT:   store i32 %2, ptr %1, align 4, !dbg !49
// CHECK:STDOUT:   %end = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 1, !dbg !53
// CHECK:STDOUT:   %3 = load i32, ptr %end, align 4, !dbg !53
// CHECK:STDOUT:   %4 = load i32, ptr %1, align 4, !dbg !52
// CHECK:STDOUT:   %5 = icmp slt i32 %4, %3, !dbg !52
// CHECK:STDOUT:   br i1 %5, label %6, label %8, !dbg !54
// CHECK:STDOUT:
// CHECK:STDOUT: 6:                                                ; preds = %0
// CHECK:STDOUT:   call void @"_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8"(ptr %cursor), !dbg !56
// CHECK:STDOUT:   %7 = load i32, ptr %1, align 4, !dbg !57
// CHECK:STDOUT:   call void @_CSome.Optional.Core.29b2425ca9ad75a4(ptr %return, i32 %7), !dbg !58
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.9a03059f269926ed(ptr %1), !dbg !49
// CHECK:STDOUT:   ret void, !dbg !59
// CHECK:STDOUT:
// CHECK:STDOUT: 8:                                                ; preds = %0
// CHECK:STDOUT:   call void @_CNone.Optional.Core.29b2425ca9ad75a4(ptr %return), !dbg !60
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.9a03059f269926ed(ptr %1), !dbg !49
// CHECK:STDOUT:   ret void, !dbg !63
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CHasValue.Optional.Core.29b2425ca9ad75a4(ptr %self) #0 !dbg !66 {
// CHECK:STDOUT:   %1 = call i1 @"_CHas.c1cc3e41344c17ec:OptionalStorage.Core.b21e3f34d194eaf5"(ptr %self), !dbg !72
// CHECK:STDOUT:   ret i1 %1, !dbg !73
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CGet.Optional.Core.29b2425ca9ad75a4(ptr %self) #0 !dbg !75 {
// CHECK:STDOUT:   %1 = call i32 @"_CGet.c1cc3e41344c17ec:OptionalStorage.Core.b21e3f34d194eaf5"(ptr %self), !dbg !81
// CHECK:STDOUT:   ret i32 %1, !dbg !82
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.c6daab0befde28a8(ptr %t) #0 !dbg !85 {
// CHECK:STDOUT:   ret void, !dbg !90
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.9a03059f269926ed(ptr %t) #0 !dbg !92 {
// CHECK:STDOUT:   ret void, !dbg !97
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.ab8fcbb9cd97e8e7(ptr %t) #0 !dbg !99 {
// CHECK:STDOUT:   ret void, !dbg !104
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CInclusiveRange.Core(ptr sret({ i32, i32 }), i32, i32)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8"(ptr %self) #0 !dbg !107 {
// CHECK:STDOUT:   call void @"_COp:thunk:AddAssignWith.b59e374b0649c9ad.Core:Int.9452f4c51951679b.Core:AddAssignWith.ab03190a9e911cb4.Core.306167be98c24f26"(ptr %self, i32 1), !dbg !110
// CHECK:STDOUT:   ret void, !dbg !113
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSome.Optional.Core.29b2425ca9ad75a4(ptr sret(<{ i32, i1 }>) %return, i32 %value) #0 !dbg !115 {
// CHECK:STDOUT:   call void @"_CSome.c1cc3e41344c17ec:OptionalStorage.Core.b21e3f34d194eaf5"(ptr %return, i32 %value), !dbg !120
// CHECK:STDOUT:   ret void, !dbg !121
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CNone.Optional.Core.29b2425ca9ad75a4(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !123 {
// CHECK:STDOUT:   call void @"_CNone.c1cc3e41344c17ec:OptionalStorage.Core.b21e3f34d194eaf5"(ptr %return), !dbg !126
// CHECK:STDOUT:   ret void, !dbg !127
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @"_CHas.c1cc3e41344c17ec:OptionalStorage.Core.b21e3f34d194eaf5"(ptr %value) #0 !dbg !128 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %value, i32 0, i32 1, !dbg !133
// CHECK:STDOUT:   %1 = load i8, ptr %has_value, align 1, !dbg !133
// CHECK:STDOUT:   %2 = trunc i8 %1 to i1, !dbg !133
// CHECK:STDOUT:   ret i1 %2, !dbg !134
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CGet.c1cc3e41344c17ec:OptionalStorage.Core.b21e3f34d194eaf5"(ptr %value) #0 !dbg !136 {
// CHECK:STDOUT:   %value1 = getelementptr inbounds nuw <{ i32, i1 }>, ptr %value, i32 0, i32 0, !dbg !141
// CHECK:STDOUT:   %1 = load i32, ptr %value1, align 4, !dbg !141
// CHECK:STDOUT:   ret i32 %1, !dbg !142
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp:thunk:AddAssignWith.b59e374b0649c9ad.Core:Int.9452f4c51951679b.Core:AddAssignWith.ab03190a9e911cb4.Core.306167be98c24f26"(ptr %self, i32 %other) #2 !dbg !144 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.df8c28cb3df17b7d:ImplicitAs.c95da57eee1d2d0f.Core.25ef994321ea0701"(i32 %other), !dbg !148
// CHECK:STDOUT:   %2 = load i32, ptr %self, align 4, !dbg !148
// CHECK:STDOUT:   %3 = add i32 %2, %1, !dbg !148
// CHECK:STDOUT:   store i32 %3, ptr %self, align 4, !dbg !148
// CHECK:STDOUT:   ret void, !dbg !148
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.c1cc3e41344c17ec:OptionalStorage.Core.b21e3f34d194eaf5"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !152 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 0, !dbg !161
// CHECK:STDOUT:   store i32 %self, ptr %value, align 4, !dbg !161
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !163
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !163
// CHECK:STDOUT:   ret void, !dbg !164
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNone.c1cc3e41344c17ec:OptionalStorage.Core.b21e3f34d194eaf5"(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !166 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !173
// CHECK:STDOUT:   store i8 0, ptr %has_value, align 1, !dbg !173
// CHECK:STDOUT:   ret void, !dbg !174
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.df8c28cb3df17b7d:ImplicitAs.c95da57eee1d2d0f.Core.25ef994321ea0701"(i32 %self) #0 !dbg !175 {
// CHECK:STDOUT:   %1 = call i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self), !dbg !180
// CHECK:STDOUT:   ret i32 %1, !dbg !182
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self) #0 !dbg !184 {
// CHECK:STDOUT:   ret i32 %self, !dbg !190
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 3, 2, 1 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.c6daab0befde28a8, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.9a03059f269926ed, { 2, 1, 0 }
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
// CHECK:STDOUT: !24 = !{null, !9}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = !DILocation(line: 19, column: 3, scope: !12)
// CHECK:STDOUT: !27 = !{null, !6}
// CHECK:STDOUT: !28 = !DISubroutineType(types: !27)
// CHECK:STDOUT: !29 = !DILocation(line: 22, column: 3, scope: !12)
// CHECK:STDOUT: !30 = !DILocation(line: 17, column: 1, scope: !12)
// CHECK:STDOUT: !33 = !DIFile(filename: "{{.*}}/prelude/range.carbon", directory: "")
// CHECK:STDOUT: !34 = distinct !DISubprogram(name: "NewCursor", linkageName: "_CNewCursor.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8", scope: null, file: !33, line: 19, type: !18, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !41)
// CHECK:STDOUT: !35 = !DILocalVariable(arg: 1, scope: !34, type: !9)
// CHECK:STDOUT: !39 = !DILocation(line: 19, column: 43, scope: !34)
// CHECK:STDOUT: !40 = !DILocation(line: 19, column: 36, scope: !34)
// CHECK:STDOUT: !41 = !{!35}
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "Next", linkageName: "_CNext.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8", scope: null, file: !33, line: 20, type: !20, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !64)
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 1, scope: !42, type: !9)
// CHECK:STDOUT: !44 = !DILocalVariable(arg: 2, scope: !42, type: !6)
// CHECK:STDOUT: !49 = !DILocation(line: 21, column: 7, scope: !42)
// CHECK:STDOUT: !50 = !DILocation(line: 21, column: 27, scope: !42)
// CHECK:STDOUT: !52 = !DILocation(line: 22, column: 11, scope: !42)
// CHECK:STDOUT: !53 = !DILocation(line: 22, column: 19, scope: !42)
// CHECK:STDOUT: !54 = !DILocation(line: 22, column: 10, scope: !42)
// CHECK:STDOUT: !56 = !DILocation(line: 23, column: 9, scope: !42)
// CHECK:STDOUT: !57 = !DILocation(line: 24, column: 38, scope: !42)
// CHECK:STDOUT: !58 = !DILocation(line: 24, column: 16, scope: !42)
// CHECK:STDOUT: !59 = !DILocation(line: 24, column: 9, scope: !42)
// CHECK:STDOUT: !60 = !DILocation(line: 26, column: 16, scope: !42)
// CHECK:STDOUT: !61 = !{!9}
// CHECK:STDOUT: !62 = !DISubroutineType(types: !61)
// CHECK:STDOUT: !63 = !DILocation(line: 26, column: 9, scope: !42)
// CHECK:STDOUT: !64 = !{!43, !44}
// CHECK:STDOUT: !65 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !66 = distinct !DISubprogram(name: "HasValue", linkageName: "_CHasValue.Optional.Core.29b2425ca9ad75a4", scope: null, file: !65, line: 36, type: !22, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !74)
// CHECK:STDOUT: !67 = !DILocalVariable(arg: 1, scope: !66, type: !9)
// CHECK:STDOUT: !72 = !DILocation(line: 37, column: 12, scope: !66)
// CHECK:STDOUT: !73 = !DILocation(line: 37, column: 5, scope: !66)
// CHECK:STDOUT: !74 = !{!67}
// CHECK:STDOUT: !75 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.Optional.Core.29b2425ca9ad75a4", scope: null, file: !65, line: 39, type: !18, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !83)
// CHECK:STDOUT: !76 = !DILocalVariable(arg: 1, scope: !75, type: !9)
// CHECK:STDOUT: !81 = !DILocation(line: 40, column: 12, scope: !75)
// CHECK:STDOUT: !82 = !DILocation(line: 40, column: 5, scope: !75)
// CHECK:STDOUT: !83 = !{!76}
// CHECK:STDOUT: !84 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !85 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.c6daab0befde28a8", scope: null, file: !84, line: 35, type: !25, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !91)
// CHECK:STDOUT: !86 = !DILocalVariable(arg: 1, scope: !85, type: !9)
// CHECK:STDOUT: !90 = !DILocation(line: 35, column: 1, scope: !85)
// CHECK:STDOUT: !91 = !{!86}
// CHECK:STDOUT: !92 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.9a03059f269926ed", scope: null, file: !84, line: 35, type: !28, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !98)
// CHECK:STDOUT: !93 = !DILocalVariable(arg: 1, scope: !92, type: !6)
// CHECK:STDOUT: !97 = !DILocation(line: 35, column: 1, scope: !92)
// CHECK:STDOUT: !98 = !{!93}
// CHECK:STDOUT: !99 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.ab8fcbb9cd97e8e7", scope: null, file: !84, line: 35, type: !25, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !105)
// CHECK:STDOUT: !100 = !DILocalVariable(arg: 1, scope: !99, type: !9)
// CHECK:STDOUT: !104 = !DILocation(line: 35, column: 1, scope: !99)
// CHECK:STDOUT: !105 = !{!100}
// CHECK:STDOUT: !106 = !DIFile(filename: "{{.*}}/prelude/types/int.carbon", directory: "")
// CHECK:STDOUT: !107 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8", scope: null, file: !106, line: 396, type: !28, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !114)
// CHECK:STDOUT: !108 = !DILocalVariable(arg: 1, scope: !107, type: !6)
// CHECK:STDOUT: !110 = !DILocation(line: 398, column: 5, scope: !107)
// CHECK:STDOUT: !111 = !{null, !6, !6}
// CHECK:STDOUT: !112 = !DISubroutineType(types: !111)
// CHECK:STDOUT: !113 = !DILocation(line: 396, column: 3, scope: !107)
// CHECK:STDOUT: !114 = !{!108}
// CHECK:STDOUT: !115 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.29b2425ca9ad75a4", scope: null, file: !65, line: 33, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !122)
// CHECK:STDOUT: !116 = !DILocalVariable(arg: 1, scope: !115, type: !6)
// CHECK:STDOUT: !120 = !DILocation(line: 34, column: 12, scope: !115)
// CHECK:STDOUT: !121 = !DILocation(line: 34, column: 5, scope: !115)
// CHECK:STDOUT: !122 = !{!116}
// CHECK:STDOUT: !123 = distinct !DISubprogram(name: "None", linkageName: "_CNone.Optional.Core.29b2425ca9ad75a4", scope: null, file: !65, line: 30, type: !62, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !126 = !DILocation(line: 31, column: 12, scope: !123)
// CHECK:STDOUT: !127 = !DILocation(line: 31, column: 5, scope: !123)
// CHECK:STDOUT: !128 = distinct !DISubprogram(name: "Has", linkageName: "_CHas.c1cc3e41344c17ec:OptionalStorage.Core.b21e3f34d194eaf5", scope: null, file: !65, line: 141, type: !22, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !135)
// CHECK:STDOUT: !129 = !DILocalVariable(arg: 1, scope: !128, type: !9)
// CHECK:STDOUT: !133 = !DILocation(line: 142, column: 12, scope: !128)
// CHECK:STDOUT: !134 = !DILocation(line: 142, column: 5, scope: !128)
// CHECK:STDOUT: !135 = !{!129}
// CHECK:STDOUT: !136 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.c1cc3e41344c17ec:OptionalStorage.Core.b21e3f34d194eaf5", scope: null, file: !65, line: 144, type: !18, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !143)
// CHECK:STDOUT: !137 = !DILocalVariable(arg: 1, scope: !136, type: !9)
// CHECK:STDOUT: !141 = !DILocation(line: 145, column: 12, scope: !136)
// CHECK:STDOUT: !142 = !DILocation(line: 145, column: 5, scope: !136)
// CHECK:STDOUT: !143 = !{!137}
// CHECK:STDOUT: !144 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:AddAssignWith.b59e374b0649c9ad.Core:Int.9452f4c51951679b.Core:AddAssignWith.ab03190a9e911cb4.Core.306167be98c24f26", scope: null, file: !106, line: 331, type: !112, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !151)
// CHECK:STDOUT: !145 = !DILocalVariable(arg: 1, scope: !144, type: !6)
// CHECK:STDOUT: !146 = !DILocalVariable(arg: 2, scope: !144, type: !6)
// CHECK:STDOUT: !148 = !DILocation(line: 331, column: 3, scope: !144)
// CHECK:STDOUT: !149 = !{!6, !6}
// CHECK:STDOUT: !150 = !DISubroutineType(types: !149)
// CHECK:STDOUT: !151 = !{!145, !146}
// CHECK:STDOUT: !152 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.c1cc3e41344c17ec:OptionalStorage.Core.b21e3f34d194eaf5", scope: null, file: !65, line: 131, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !165)
// CHECK:STDOUT: !153 = !DILocalVariable(arg: 1, scope: !152, type: !6)
// CHECK:STDOUT: !161 = !DILocation(line: 137, column: 5, scope: !152)
// CHECK:STDOUT: !163 = !DILocation(line: 138, column: 5, scope: !152)
// CHECK:STDOUT: !164 = !DILocation(line: 139, column: 5, scope: !152)
// CHECK:STDOUT: !165 = !{!153}
// CHECK:STDOUT: !166 = distinct !DISubprogram(name: "None", linkageName: "_CNone.c1cc3e41344c17ec:OptionalStorage.Core.b21e3f34d194eaf5", scope: null, file: !65, line: 125, type: !62, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !173 = !DILocation(line: 128, column: 5, scope: !166)
// CHECK:STDOUT: !174 = !DILocation(line: 129, column: 5, scope: !166)
// CHECK:STDOUT: !175 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.df8c28cb3df17b7d:ImplicitAs.c95da57eee1d2d0f.Core.25ef994321ea0701", scope: null, file: !106, line: 64, type: !150, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !183)
// CHECK:STDOUT: !176 = !DILocalVariable(arg: 1, scope: !175, type: !6)
// CHECK:STDOUT: !180 = !DILocation(line: 66, column: 17, scope: !175)
// CHECK:STDOUT: !182 = !DILocation(line: 66, column: 5, scope: !175)
// CHECK:STDOUT: !183 = !{!176}
// CHECK:STDOUT: !184 = distinct !DISubprogram(name: "AsInt", linkageName: "_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8", scope: null, file: !106, line: 59, type: !150, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !191)
// CHECK:STDOUT: !185 = !DILocalVariable(arg: 1, scope: !184, type: !6)
// CHECK:STDOUT: !190 = !DILocation(line: 59, column: 28, scope: !184)
// CHECK:STDOUT: !191 = !{!185}
// CHECK:STDOUT:
