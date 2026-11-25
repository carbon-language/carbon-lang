// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/array/iterate.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/array/iterate.carbon

fn G(n: i32);

fn F() {
  let a: array(i32, 6) = (1, 2, 3, 4, 5, 6);
  for (n: i32 in a) {
    G(n);
  }
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'iterate.carbon'
// CHECK:STDOUT: source_filename = "iterate.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @array = internal constant [6 x i32] [i32 1, i32 2, i32 3, i32 4, i32 5, i32 6]
// CHECK:STDOUT: @array.loc16_43.24 = internal constant [6 x i32] [i32 1, i32 2, i32 3, i32 4, i32 5, i32 6]
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CG.Main(i32)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc16_43.3.temp = alloca [6 x i32], align 4, !dbg !12
// CHECK:STDOUT:   %var = alloca i32, align 4, !dbg !17
// CHECK:STDOUT:   %.loc17_19.5.temp = alloca <{ i32, i1 }>, align 4, !dbg !17
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc16_43.3.temp), !dbg !12
// CHECK:STDOUT:   %.loc16_43.4.array.index = getelementptr inbounds i8, ptr %.loc16_43.3.temp, i64 0, !dbg !12
// CHECK:STDOUT:   %.loc16_43.7.array.index = getelementptr inbounds i8, ptr %.loc16_43.3.temp, i64 4, !dbg !12
// CHECK:STDOUT:   %.loc16_43.10.array.index = getelementptr inbounds i8, ptr %.loc16_43.3.temp, i64 8, !dbg !12
// CHECK:STDOUT:   %.loc16_43.13.array.index = getelementptr inbounds i8, ptr %.loc16_43.3.temp, i64 12, !dbg !12
// CHECK:STDOUT:   %.loc16_43.16.array.index = getelementptr inbounds i8, ptr %.loc16_43.3.temp, i64 16, !dbg !12
// CHECK:STDOUT:   %.loc16_43.19.array.index = getelementptr inbounds i8, ptr %.loc16_43.3.temp, i64 20, !dbg !12
// CHECK:STDOUT:   %Array.as.Iterate.impl.NewCursor.call = call i32 @"_CNewCursor.Array.ef119564f44a13e5.Core:Iterate.Core.ea5855de090934bd"(ptr @array.loc16_43.24), !dbg !17
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %var), !dbg !17
// CHECK:STDOUT:   store i32 %Array.as.Iterate.impl.NewCursor.call, ptr %var, align 4, !dbg !17
// CHECK:STDOUT:   br label %for.next, !dbg !17
// CHECK:STDOUT:
// CHECK:STDOUT: for.next:                                         ; preds = %for.body, %entry
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc17_19.5.temp), !dbg !17
// CHECK:STDOUT:   call void @"_CNext.Array.ef119564f44a13e5.Core:Iterate.Core.ea5855de090934bd"(ptr %.loc17_19.5.temp, ptr @array.loc16_43.24, ptr %var), !dbg !17
// CHECK:STDOUT:   %Optional.HasValue.call = call i1 @_CHasValue.Optional.Core.29b2425ca9ad75a4(ptr %.loc17_19.5.temp), !dbg !17
// CHECK:STDOUT:   br i1 %Optional.HasValue.call, label %for.body, label %for.done, !dbg !17
// CHECK:STDOUT:
// CHECK:STDOUT: for.body:                                         ; preds = %for.next
// CHECK:STDOUT:   %Optional.Get.call = call i32 @_CGet.Optional.Core.29b2425ca9ad75a4(ptr %.loc17_19.5.temp), !dbg !17
// CHECK:STDOUT:   call void @_CG.Main(i32 %Optional.Get.call), !dbg !26
// CHECK:STDOUT:   br label %for.next, !dbg !27
// CHECK:STDOUT:
// CHECK:STDOUT: for.done:                                         ; preds = %for.next
// CHECK:STDOUT:   ret void, !dbg !28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CNewCursor.Array.ef119564f44a13e5.Core:Iterate.Core.ea5855de090934bd"(ptr %self) #0 !dbg !30 {
// CHECK:STDOUT:   ret i32 0, !dbg !35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNext.Array.ef119564f44a13e5.Core:Iterate.Core.ea5855de090934bd"(ptr sret(<{ i32, i1 }>) %return, ptr %self, ptr %cursor) #0 !dbg !37 {
// CHECK:STDOUT:   %1 = load i32, ptr %cursor, align 4, !dbg !44
// CHECK:STDOUT:   %2 = icmp slt i32 %1, 6, !dbg !44
// CHECK:STDOUT:   br i1 %2, label %3, label %8, !dbg !45
// CHECK:STDOUT:
// CHECK:STDOUT: 3:                                                ; preds = %0
// CHECK:STDOUT:   call void @"_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8"(ptr %cursor), !dbg !47
// CHECK:STDOUT:   %4 = load i32, ptr %cursor, align 4, !dbg !49
// CHECK:STDOUT:   %5 = sub i32 %4, 1, !dbg !49
// CHECK:STDOUT:   %6 = sext i32 %5 to i64, !dbg !48
// CHECK:STDOUT:   %array.index = getelementptr inbounds [4 x i8], ptr %self, i64 %6, !dbg !48
// CHECK:STDOUT:   %7 = load i32, ptr %array.index, align 4, !dbg !48
// CHECK:STDOUT:   call void @_CSome.Optional.Core.29b2425ca9ad75a4(ptr %return, i32 %7), !dbg !50
// CHECK:STDOUT:   ret void, !dbg !53
// CHECK:STDOUT:
// CHECK:STDOUT: 8:                                                ; preds = %0
// CHECK:STDOUT:   call void @_CNone.Optional.Core.29b2425ca9ad75a4(ptr %return), !dbg !54
// CHECK:STDOUT:   ret void, !dbg !57
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CHasValue.Optional.Core.29b2425ca9ad75a4(ptr %self) #0 !dbg !60 {
// CHECK:STDOUT:   %1 = call i1 @"_CHas.c1cc3e41344c17ec:OptionalStorage.Core.b21e3f34d194eaf5"(ptr %self), !dbg !66
// CHECK:STDOUT:   ret i1 %1, !dbg !67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CGet.Optional.Core.29b2425ca9ad75a4(ptr %self) #0 !dbg !69 {
// CHECK:STDOUT:   %1 = call i32 @"_CGet.c1cc3e41344c17ec:OptionalStorage.Core.b21e3f34d194eaf5"(ptr %self), !dbg !75
// CHECK:STDOUT:   ret i32 %1, !dbg !76
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8"(ptr %self) #0 !dbg !79 {
// CHECK:STDOUT:   call void @"_COp:thunk:AddAssignWith.b59e374b0649c9ad.Core:Int.9452f4c51951679b.Core:AddAssignWith.ab03190a9e911cb4.Core.306167be98c24f26"(ptr %self, i32 1), !dbg !82
// CHECK:STDOUT:   ret void, !dbg !85
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSome.Optional.Core.29b2425ca9ad75a4(ptr sret(<{ i32, i1 }>) %return, i32 %value) #0 !dbg !87 {
// CHECK:STDOUT:   call void @"_CSome.c1cc3e41344c17ec:OptionalStorage.Core.b21e3f34d194eaf5"(ptr %return, i32 %value), !dbg !92
// CHECK:STDOUT:   ret void, !dbg !93
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CNone.Optional.Core.29b2425ca9ad75a4(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !95 {
// CHECK:STDOUT:   call void @"_CNone.c1cc3e41344c17ec:OptionalStorage.Core.b21e3f34d194eaf5"(ptr %return), !dbg !98
// CHECK:STDOUT:   ret void, !dbg !99
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @"_CHas.c1cc3e41344c17ec:OptionalStorage.Core.b21e3f34d194eaf5"(ptr %value) #0 !dbg !100 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw i8, ptr %value, i64 4, !dbg !105
// CHECK:STDOUT:   %1 = load i8, ptr %has_value, align 1, !dbg !105
// CHECK:STDOUT:   %2 = trunc i8 %1 to i1, !dbg !105
// CHECK:STDOUT:   ret i1 %2, !dbg !106
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CGet.c1cc3e41344c17ec:OptionalStorage.Core.b21e3f34d194eaf5"(ptr %value) #0 !dbg !108 {
// CHECK:STDOUT:   %value1 = getelementptr inbounds nuw i8, ptr %value, i64 0, !dbg !113
// CHECK:STDOUT:   %1 = load i32, ptr %value1, align 4, !dbg !113
// CHECK:STDOUT:   ret i32 %1, !dbg !114
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp:thunk:AddAssignWith.b59e374b0649c9ad.Core:Int.9452f4c51951679b.Core:AddAssignWith.ab03190a9e911cb4.Core.306167be98c24f26"(ptr %self, i32 %other) #2 !dbg !116 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.df8c28cb3df17b7d:ImplicitAs.c95da57eee1d2d0f.Core.25ef994321ea0701"(i32 %other), !dbg !120
// CHECK:STDOUT:   %2 = load i32, ptr %self, align 4, !dbg !120
// CHECK:STDOUT:   %3 = add i32 %2, %1, !dbg !120
// CHECK:STDOUT:   store i32 %3, ptr %self, align 4, !dbg !120
// CHECK:STDOUT:   ret void, !dbg !120
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.c1cc3e41344c17ec:OptionalStorage.Core.b21e3f34d194eaf5"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !124 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw i8, ptr %return, i64 0, !dbg !133
// CHECK:STDOUT:   store i32 %self, ptr %value, align 4, !dbg !133
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw i8, ptr %return, i64 4, !dbg !135
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !135
// CHECK:STDOUT:   ret void, !dbg !136
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNone.c1cc3e41344c17ec:OptionalStorage.Core.b21e3f34d194eaf5"(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !138 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw i8, ptr %return, i64 4, !dbg !145
// CHECK:STDOUT:   store i8 0, ptr %has_value, align 1, !dbg !145
// CHECK:STDOUT:   ret void, !dbg !146
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.df8c28cb3df17b7d:ImplicitAs.c95da57eee1d2d0f.Core.25ef994321ea0701"(i32 %self) #0 !dbg !147 {
// CHECK:STDOUT:   %1 = call i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self), !dbg !152
// CHECK:STDOUT:   ret i32 %1, !dbg !154
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self) #0 !dbg !156 {
// CHECK:STDOUT:   ret i32 %self, !dbg !162
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 2, 1, 0 }
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
// CHECK:STDOUT: !4 = !DIFile(filename: "iterate.carbon", directory: "")
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !7 = !{null, !6}
// CHECK:STDOUT: !8 = !DISubroutineType(types: !7)
// CHECK:STDOUT: !9 = !{null}
// CHECK:STDOUT: !10 = !DISubroutineType(types: !9)
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !4, line: 15, type: !10, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !12 = !DILocation(line: 16, column: 26, scope: !11)
// CHECK:STDOUT: !17 = !DILocation(line: 17, column: 7, scope: !11)
// CHECK:STDOUT: !18 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !19 = !{!6, !18}
// CHECK:STDOUT: !20 = !DISubroutineType(types: !19)
// CHECK:STDOUT: !21 = !{!18, !18, !6}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = !{!18, !18}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !26 = !DILocation(line: 18, column: 5, scope: !11)
// CHECK:STDOUT: !27 = !DILocation(line: 17, column: 3, scope: !11)
// CHECK:STDOUT: !28 = !DILocation(line: 15, column: 1, scope: !11)
// CHECK:STDOUT: !29 = !DIFile(filename: "{{.*}}/prelude/iterate.carbon", directory: "")
// CHECK:STDOUT: !30 = distinct !DISubprogram(name: "NewCursor", linkageName: "_CNewCursor.Array.ef119564f44a13e5.Core:Iterate.Core.ea5855de090934bd", scope: null, file: !29, line: 23, type: !20, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !36)
// CHECK:STDOUT: !31 = !DILocalVariable(arg: 1, scope: !30, type: !18)
// CHECK:STDOUT: !35 = !DILocation(line: 23, column: 38, scope: !30)
// CHECK:STDOUT: !36 = !{!31}
// CHECK:STDOUT: !37 = distinct !DISubprogram(name: "Next", linkageName: "_CNext.Array.ef119564f44a13e5.Core:Iterate.Core.ea5855de090934bd", scope: null, file: !29, line: 24, type: !22, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !58)
// CHECK:STDOUT: !38 = !DILocalVariable(arg: 1, scope: !37, type: !18)
// CHECK:STDOUT: !39 = !DILocalVariable(arg: 2, scope: !37, type: !6)
// CHECK:STDOUT: !44 = !DILocation(line: 25, column: 9, scope: !37)
// CHECK:STDOUT: !45 = !DILocation(line: 25, column: 8, scope: !37)
// CHECK:STDOUT: !47 = !DILocation(line: 26, column: 7, scope: !37)
// CHECK:STDOUT: !48 = !DILocation(line: 27, column: 31, scope: !37)
// CHECK:STDOUT: !49 = !DILocation(line: 27, column: 36, scope: !37)
// CHECK:STDOUT: !50 = !DILocation(line: 27, column: 14, scope: !37)
// CHECK:STDOUT: !51 = !{!18, !6}
// CHECK:STDOUT: !52 = !DISubroutineType(types: !51)
// CHECK:STDOUT: !53 = !DILocation(line: 27, column: 7, scope: !37)
// CHECK:STDOUT: !54 = !DILocation(line: 29, column: 14, scope: !37)
// CHECK:STDOUT: !55 = !{!18}
// CHECK:STDOUT: !56 = !DISubroutineType(types: !55)
// CHECK:STDOUT: !57 = !DILocation(line: 29, column: 7, scope: !37)
// CHECK:STDOUT: !58 = !{!38, !39}
// CHECK:STDOUT: !59 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !60 = distinct !DISubprogram(name: "HasValue", linkageName: "_CHasValue.Optional.Core.29b2425ca9ad75a4", scope: null, file: !59, line: 36, type: !24, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !68)
// CHECK:STDOUT: !61 = !DILocalVariable(arg: 1, scope: !60, type: !18)
// CHECK:STDOUT: !66 = !DILocation(line: 37, column: 12, scope: !60)
// CHECK:STDOUT: !67 = !DILocation(line: 37, column: 5, scope: !60)
// CHECK:STDOUT: !68 = !{!61}
// CHECK:STDOUT: !69 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.Optional.Core.29b2425ca9ad75a4", scope: null, file: !59, line: 39, type: !20, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !77)
// CHECK:STDOUT: !70 = !DILocalVariable(arg: 1, scope: !69, type: !18)
// CHECK:STDOUT: !75 = !DILocation(line: 40, column: 12, scope: !69)
// CHECK:STDOUT: !76 = !DILocation(line: 40, column: 5, scope: !69)
// CHECK:STDOUT: !77 = !{!70}
// CHECK:STDOUT: !78 = !DIFile(filename: "{{.*}}/prelude/types/int.carbon", directory: "")
// CHECK:STDOUT: !79 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8", scope: null, file: !78, line: 396, type: !8, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !86)
// CHECK:STDOUT: !80 = !DILocalVariable(arg: 1, scope: !79, type: !6)
// CHECK:STDOUT: !82 = !DILocation(line: 398, column: 5, scope: !79)
// CHECK:STDOUT: !83 = !{null, !6, !6}
// CHECK:STDOUT: !84 = !DISubroutineType(types: !83)
// CHECK:STDOUT: !85 = !DILocation(line: 396, column: 3, scope: !79)
// CHECK:STDOUT: !86 = !{!80}
// CHECK:STDOUT: !87 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.29b2425ca9ad75a4", scope: null, file: !59, line: 33, type: !52, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !94)
// CHECK:STDOUT: !88 = !DILocalVariable(arg: 1, scope: !87, type: !6)
// CHECK:STDOUT: !92 = !DILocation(line: 34, column: 12, scope: !87)
// CHECK:STDOUT: !93 = !DILocation(line: 34, column: 5, scope: !87)
// CHECK:STDOUT: !94 = !{!88}
// CHECK:STDOUT: !95 = distinct !DISubprogram(name: "None", linkageName: "_CNone.Optional.Core.29b2425ca9ad75a4", scope: null, file: !59, line: 30, type: !56, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !98 = !DILocation(line: 31, column: 12, scope: !95)
// CHECK:STDOUT: !99 = !DILocation(line: 31, column: 5, scope: !95)
// CHECK:STDOUT: !100 = distinct !DISubprogram(name: "Has", linkageName: "_CHas.c1cc3e41344c17ec:OptionalStorage.Core.b21e3f34d194eaf5", scope: null, file: !59, line: 141, type: !24, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !107)
// CHECK:STDOUT: !101 = !DILocalVariable(arg: 1, scope: !100, type: !18)
// CHECK:STDOUT: !105 = !DILocation(line: 142, column: 12, scope: !100)
// CHECK:STDOUT: !106 = !DILocation(line: 142, column: 5, scope: !100)
// CHECK:STDOUT: !107 = !{!101}
// CHECK:STDOUT: !108 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.c1cc3e41344c17ec:OptionalStorage.Core.b21e3f34d194eaf5", scope: null, file: !59, line: 144, type: !20, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !115)
// CHECK:STDOUT: !109 = !DILocalVariable(arg: 1, scope: !108, type: !18)
// CHECK:STDOUT: !113 = !DILocation(line: 145, column: 12, scope: !108)
// CHECK:STDOUT: !114 = !DILocation(line: 145, column: 5, scope: !108)
// CHECK:STDOUT: !115 = !{!109}
// CHECK:STDOUT: !116 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:AddAssignWith.b59e374b0649c9ad.Core:Int.9452f4c51951679b.Core:AddAssignWith.ab03190a9e911cb4.Core.306167be98c24f26", scope: null, file: !78, line: 331, type: !84, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !123)
// CHECK:STDOUT: !117 = !DILocalVariable(arg: 1, scope: !116, type: !6)
// CHECK:STDOUT: !118 = !DILocalVariable(arg: 2, scope: !116, type: !6)
// CHECK:STDOUT: !120 = !DILocation(line: 331, column: 3, scope: !116)
// CHECK:STDOUT: !121 = !{!6, !6}
// CHECK:STDOUT: !122 = !DISubroutineType(types: !121)
// CHECK:STDOUT: !123 = !{!117, !118}
// CHECK:STDOUT: !124 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.c1cc3e41344c17ec:OptionalStorage.Core.b21e3f34d194eaf5", scope: null, file: !59, line: 131, type: !52, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !137)
// CHECK:STDOUT: !125 = !DILocalVariable(arg: 1, scope: !124, type: !6)
// CHECK:STDOUT: !133 = !DILocation(line: 137, column: 5, scope: !124)
// CHECK:STDOUT: !135 = !DILocation(line: 138, column: 5, scope: !124)
// CHECK:STDOUT: !136 = !DILocation(line: 139, column: 5, scope: !124)
// CHECK:STDOUT: !137 = !{!125}
// CHECK:STDOUT: !138 = distinct !DISubprogram(name: "None", linkageName: "_CNone.c1cc3e41344c17ec:OptionalStorage.Core.b21e3f34d194eaf5", scope: null, file: !59, line: 125, type: !56, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !145 = !DILocation(line: 128, column: 5, scope: !138)
// CHECK:STDOUT: !146 = !DILocation(line: 129, column: 5, scope: !138)
// CHECK:STDOUT: !147 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.df8c28cb3df17b7d:ImplicitAs.c95da57eee1d2d0f.Core.25ef994321ea0701", scope: null, file: !78, line: 64, type: !122, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !155)
// CHECK:STDOUT: !148 = !DILocalVariable(arg: 1, scope: !147, type: !6)
// CHECK:STDOUT: !152 = !DILocation(line: 66, column: 17, scope: !147)
// CHECK:STDOUT: !154 = !DILocation(line: 66, column: 5, scope: !147)
// CHECK:STDOUT: !155 = !{!148}
// CHECK:STDOUT: !156 = distinct !DISubprogram(name: "AsInt", linkageName: "_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8", scope: null, file: !78, line: 59, type: !122, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !163)
// CHECK:STDOUT: !157 = !DILocalVariable(arg: 1, scope: !156, type: !6)
// CHECK:STDOUT: !162 = !DILocation(line: 59, column: 28, scope: !156)
// CHECK:STDOUT: !163 = !{!157}
// CHECK:STDOUT:
