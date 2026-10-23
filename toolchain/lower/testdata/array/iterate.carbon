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
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !14 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc16_43.3.temp = alloca [6 x i32], align 4, !dbg !15
// CHECK:STDOUT:   %var = alloca i32, align 4, !dbg !20
// CHECK:STDOUT:   %.loc17_19.5.temp = alloca <{ i32, i1 }>, align 4, !dbg !20
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc16_43.3.temp), !dbg !15
// CHECK:STDOUT:   %.loc16_43.4.array.index = getelementptr inbounds [6 x i32], ptr %.loc16_43.3.temp, i32 0, i64 0, !dbg !15
// CHECK:STDOUT:   %.loc16_43.7.array.index = getelementptr inbounds [6 x i32], ptr %.loc16_43.3.temp, i32 0, i64 1, !dbg !15
// CHECK:STDOUT:   %.loc16_43.10.array.index = getelementptr inbounds [6 x i32], ptr %.loc16_43.3.temp, i32 0, i64 2, !dbg !15
// CHECK:STDOUT:   %.loc16_43.13.array.index = getelementptr inbounds [6 x i32], ptr %.loc16_43.3.temp, i32 0, i64 3, !dbg !15
// CHECK:STDOUT:   %.loc16_43.16.array.index = getelementptr inbounds [6 x i32], ptr %.loc16_43.3.temp, i32 0, i64 4, !dbg !15
// CHECK:STDOUT:   %.loc16_43.19.array.index = getelementptr inbounds [6 x i32], ptr %.loc16_43.3.temp, i32 0, i64 5, !dbg !15
// CHECK:STDOUT:   %Array.as.Iterate.impl.NewCursor.call = call i32 @"_CNewCursor.Array.d9c03cafdb3c033f.Core:Iterate.Core.9e8a519348dcc21a"(ptr @array.loc16_43.24), !dbg !20
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %var), !dbg !20
// CHECK:STDOUT:   store i32 %Array.as.Iterate.impl.NewCursor.call, ptr %var, align 4, !dbg !20
// CHECK:STDOUT:   br label %for.next, !dbg !20
// CHECK:STDOUT:
// CHECK:STDOUT: for.next:                                         ; preds = %for.body, %entry
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc17_19.5.temp), !dbg !20
// CHECK:STDOUT:   call void @"_CNext.Array.d9c03cafdb3c033f.Core:Iterate.Core.9e8a519348dcc21a"(ptr %.loc17_19.5.temp, ptr @array.loc16_43.24, ptr %var), !dbg !20
// CHECK:STDOUT:   %Optional.HasValue.call = call i1 @_CHasValue.Optional.Core.f7d25475af49e62f(ptr %.loc17_19.5.temp), !dbg !20
// CHECK:STDOUT:   br i1 %Optional.HasValue.call, label %for.body, label %for.done, !dbg !20
// CHECK:STDOUT:
// CHECK:STDOUT: for.body:                                         ; preds = %for.next
// CHECK:STDOUT:   %Optional.Get.call = call i32 @_CGet.Optional.Core.f7d25475af49e62f(ptr %.loc17_19.5.temp), !dbg !20
// CHECK:STDOUT:   call void @_CG.Main(i32 %Optional.Get.call), !dbg !28
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.1b119d8ccf65a76f(ptr %.loc17_19.5.temp), !dbg !20
// CHECK:STDOUT:   br label %for.next, !dbg !29
// CHECK:STDOUT:
// CHECK:STDOUT: for.done:                                         ; preds = %for.next
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.1b119d8ccf65a76f(ptr %.loc17_19.5.temp), !dbg !20
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.b9c4c4d5428719ba(ptr %var), !dbg !20
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f24fb663cda48956(ptr @array), !dbg !15
// CHECK:STDOUT:   ret void, !dbg !30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.b3126fe3b97d1b85:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !31 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.b29714547c418f6c:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !35 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc17_19.1.has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %self, i32 0, i32 1, !dbg !37
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.94cb175d66c1dda9(ptr %.loc17_19.1.has_value), !dbg !37
// CHECK:STDOUT:   %.loc17_19.4.value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %self, i32 0, i32 0, !dbg !37
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.56a8014ecefcdde9(ptr %.loc17_19.4.value), !dbg !37
// CHECK:STDOUT:   ret void, !dbg !37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.07c81106aaf54f73:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !39 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc17_19.1.has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %self, i32 0, i32 1, !dbg !41
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.94cb175d66c1dda9(ptr %.loc17_19.1.has_value), !dbg !41
// CHECK:STDOUT:   %.loc17_19.4.value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %self, i32 0, i32 0, !dbg !41
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.56a8014ecefcdde9(ptr %.loc17_19.4.value), !dbg !41
// CHECK:STDOUT:   ret void, !dbg !41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.85d5d6325bc2f620:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !43 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc16_43.1.array.index = getelementptr inbounds [6 x i32], ptr %self, i32 0, i64 5, !dbg !45
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.b9c4c4d5428719ba(ptr %.loc16_43.1.array.index), !dbg !45
// CHECK:STDOUT:   %.loc16_43.4.array.index = getelementptr inbounds [6 x i32], ptr %self, i32 0, i64 4, !dbg !45
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.b9c4c4d5428719ba(ptr %.loc16_43.4.array.index), !dbg !45
// CHECK:STDOUT:   %.loc16_43.7.array.index = getelementptr inbounds [6 x i32], ptr %self, i32 0, i64 3, !dbg !45
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.b9c4c4d5428719ba(ptr %.loc16_43.7.array.index), !dbg !45
// CHECK:STDOUT:   %.loc16_43.10.array.index = getelementptr inbounds [6 x i32], ptr %self, i32 0, i64 2, !dbg !45
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.b9c4c4d5428719ba(ptr %.loc16_43.10.array.index), !dbg !45
// CHECK:STDOUT:   %.loc16_43.13.array.index = getelementptr inbounds [6 x i32], ptr %self, i32 0, i64 1, !dbg !45
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.b9c4c4d5428719ba(ptr %.loc16_43.13.array.index), !dbg !45
// CHECK:STDOUT:   %.loc16_43.16.array.index = getelementptr inbounds [6 x i32], ptr %self, i32 0, i64 0, !dbg !45
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.b9c4c4d5428719ba(ptr %.loc16_43.16.array.index), !dbg !45
// CHECK:STDOUT:   ret void, !dbg !45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CNewCursor.Array.d9c03cafdb3c033f.Core:Iterate.Core.9e8a519348dcc21a"(ptr %self) #0 !dbg !48 {
// CHECK:STDOUT:   ret i32 0, !dbg !53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNext.Array.d9c03cafdb3c033f.Core:Iterate.Core.9e8a519348dcc21a"(ptr sret(<{ i32, i1 }>) %return, ptr %self, ptr %cursor) #0 !dbg !55 {
// CHECK:STDOUT:   %1 = load i32, ptr %cursor, align 4, !dbg !62
// CHECK:STDOUT:   %2 = icmp slt i32 %1, 6, !dbg !62
// CHECK:STDOUT:   br i1 %2, label %3, label %7, !dbg !63
// CHECK:STDOUT:
// CHECK:STDOUT: 3:                                                ; preds = %0
// CHECK:STDOUT:   call void @"_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8"(ptr %cursor), !dbg !65
// CHECK:STDOUT:   %4 = load i32, ptr %cursor, align 4, !dbg !67
// CHECK:STDOUT:   %5 = sub i32 %4, 1, !dbg !67
// CHECK:STDOUT:   %array.index = getelementptr inbounds [6 x i32], ptr %self, i32 0, i32 %5, !dbg !66
// CHECK:STDOUT:   %6 = load i32, ptr %array.index, align 4, !dbg !66
// CHECK:STDOUT:   call void @_CSome.Optional.Core.f7d25475af49e62f(ptr %return, i32 %6), !dbg !68
// CHECK:STDOUT:   ret void, !dbg !71
// CHECK:STDOUT:
// CHECK:STDOUT: 7:                                                ; preds = %0
// CHECK:STDOUT:   call void @_CNone.Optional.Core.f7d25475af49e62f(ptr %return), !dbg !72
// CHECK:STDOUT:   ret void, !dbg !75
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CHasValue.Optional.Core.f7d25475af49e62f(ptr %self) #0 !dbg !78 {
// CHECK:STDOUT:   %1 = call i1 @"_CHas.21149ca0dc753fdf:OptionalStorage.Core.f39dc4a0608179aa"(ptr %self), !dbg !84
// CHECK:STDOUT:   ret i1 %1, !dbg !85
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CGet.Optional.Core.f7d25475af49e62f(ptr %self) #0 !dbg !87 {
// CHECK:STDOUT:   %1 = call i32 @"_CGet.21149ca0dc753fdf:OptionalStorage.Core.f39dc4a0608179aa"(ptr %self), !dbg !93
// CHECK:STDOUT:   ret i32 %1, !dbg !94
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.1b119d8ccf65a76f(ptr %t) #0 !dbg !97 {
// CHECK:STDOUT:   ret void, !dbg !102
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.b9c4c4d5428719ba(ptr %t) #0 !dbg !104 {
// CHECK:STDOUT:   ret void, !dbg !109
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.f24fb663cda48956(ptr %t) #0 !dbg !111 {
// CHECK:STDOUT:   call void @"_COp.85d5d6325bc2f620:core.SubobjectDestroy.Core"(ptr %t), !dbg !115
// CHECK:STDOUT:   ret void, !dbg !116
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.94cb175d66c1dda9(ptr %t) #0 !dbg !118 {
// CHECK:STDOUT:   ret void, !dbg !123
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.56a8014ecefcdde9(ptr %t) #0 !dbg !125 {
// CHECK:STDOUT:   ret void, !dbg !130
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8"(ptr %self) #0 !dbg !133 {
// CHECK:STDOUT:   call void @"_COp:thunk:AddAssignWith.b59e374b0649c9ad.Core:Int.9452f4c51951679b.Core:AddAssignWith.ab03190a9e911cb4.Core.306167be98c24f26"(ptr %self, i32 1), !dbg !136
// CHECK:STDOUT:   ret void, !dbg !139
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSome.Optional.Core.f7d25475af49e62f(ptr sret(<{ i32, i1 }>) %return, i32 %value) #0 !dbg !141 {
// CHECK:STDOUT:   call void @"_CSome.21149ca0dc753fdf:OptionalStorage.Core.f39dc4a0608179aa"(ptr %return, i32 %value), !dbg !146
// CHECK:STDOUT:   ret void, !dbg !147
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CNone.Optional.Core.f7d25475af49e62f(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !149 {
// CHECK:STDOUT:   call void @"_CNone.21149ca0dc753fdf:OptionalStorage.Core.f39dc4a0608179aa"(ptr %return), !dbg !152
// CHECK:STDOUT:   ret void, !dbg !153
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @"_CHas.21149ca0dc753fdf:OptionalStorage.Core.f39dc4a0608179aa"(ptr %value) #0 !dbg !154 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %value, i32 0, i32 1, !dbg !159
// CHECK:STDOUT:   %1 = load i8, ptr %has_value, align 1, !dbg !159
// CHECK:STDOUT:   %2 = trunc i8 %1 to i1, !dbg !159
// CHECK:STDOUT:   ret i1 %2, !dbg !160
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CGet.21149ca0dc753fdf:OptionalStorage.Core.f39dc4a0608179aa"(ptr %value) #0 !dbg !162 {
// CHECK:STDOUT:   %value1 = getelementptr inbounds nuw <{ i32, i1 }>, ptr %value, i32 0, i32 0, !dbg !167
// CHECK:STDOUT:   %1 = load i32, ptr %value1, align 4, !dbg !167
// CHECK:STDOUT:   ret i32 %1, !dbg !168
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp:thunk:AddAssignWith.b59e374b0649c9ad.Core:Int.9452f4c51951679b.Core:AddAssignWith.ab03190a9e911cb4.Core.306167be98c24f26"(ptr %self, i32 %other) #2 !dbg !170 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.df8c28cb3df17b7d:ImplicitAs.c95da57eee1d2d0f.Core.25ef994321ea0701"(i32 %other), !dbg !174
// CHECK:STDOUT:   %2 = load i32, ptr %self, align 4, !dbg !174
// CHECK:STDOUT:   %3 = add i32 %2, %1, !dbg !174
// CHECK:STDOUT:   store i32 %3, ptr %self, align 4, !dbg !174
// CHECK:STDOUT:   ret void, !dbg !174
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.21149ca0dc753fdf:OptionalStorage.Core.f39dc4a0608179aa"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !178 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 0, !dbg !187
// CHECK:STDOUT:   store i32 %self, ptr %value, align 4, !dbg !187
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !189
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !189
// CHECK:STDOUT:   ret void, !dbg !190
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNone.21149ca0dc753fdf:OptionalStorage.Core.f39dc4a0608179aa"(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !192 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !199
// CHECK:STDOUT:   store i8 0, ptr %has_value, align 1, !dbg !199
// CHECK:STDOUT:   ret void, !dbg !200
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.df8c28cb3df17b7d:ImplicitAs.c95da57eee1d2d0f.Core.25ef994321ea0701"(i32 %self) #0 !dbg !201 {
// CHECK:STDOUT:   %1 = call i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self), !dbg !206
// CHECK:STDOUT:   ret i32 %1, !dbg !208
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self) #0 !dbg !210 {
// CHECK:STDOUT:   ret i32 %self, !dbg !216
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.1b119d8ccf65a76f, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.b9c4c4d5428719ba, { 6, 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.94cb175d66c1dda9, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.56a8014ecefcdde9, { 1, 0 }
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
// CHECK:STDOUT: !11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !12 = !{null, !11}
// CHECK:STDOUT: !13 = !DISubroutineType(types: !12)
// CHECK:STDOUT: !14 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !4, line: 15, type: !10, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !15 = !DILocation(line: 16, column: 26, scope: !14)
// CHECK:STDOUT: !20 = !DILocation(line: 17, column: 7, scope: !14)
// CHECK:STDOUT: !21 = !{!6, !11}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = !{!11, !11, !6}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !25 = !{!11, !11}
// CHECK:STDOUT: !26 = !DISubroutineType(types: !25)
// CHECK:STDOUT: !28 = !DILocation(line: 18, column: 5, scope: !14)
// CHECK:STDOUT: !29 = !DILocation(line: 17, column: 3, scope: !14)
// CHECK:STDOUT: !30 = !DILocation(line: 15, column: 1, scope: !14)
// CHECK:STDOUT: !31 = distinct !DISubprogram(name: "Op", linkageName: "_COp.b3126fe3b97d1b85:core.SubobjectDestroy.Core", scope: null, file: !4, line: 17, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !34)
// CHECK:STDOUT: !32 = !DILocalVariable(arg: 1, scope: !31, type: !11)
// CHECK:STDOUT: !33 = !DILocation(line: 17, column: 7, scope: !31)
// CHECK:STDOUT: !34 = !{!32}
// CHECK:STDOUT: !35 = distinct !DISubprogram(name: "Op", linkageName: "_COp.b29714547c418f6c:core.SubobjectDestroy.Core", scope: null, file: !4, line: 17, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !38)
// CHECK:STDOUT: !36 = !DILocalVariable(arg: 1, scope: !35, type: !11)
// CHECK:STDOUT: !37 = !DILocation(line: 17, column: 7, scope: !35)
// CHECK:STDOUT: !38 = !{!36}
// CHECK:STDOUT: !39 = distinct !DISubprogram(name: "Op", linkageName: "_COp.07c81106aaf54f73:core.SubobjectDestroy.Core", scope: null, file: !4, line: 17, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !42)
// CHECK:STDOUT: !40 = !DILocalVariable(arg: 1, scope: !39, type: !11)
// CHECK:STDOUT: !41 = !DILocation(line: 17, column: 7, scope: !39)
// CHECK:STDOUT: !42 = !{!40}
// CHECK:STDOUT: !43 = distinct !DISubprogram(name: "Op", linkageName: "_COp.85d5d6325bc2f620:core.SubobjectDestroy.Core", scope: null, file: !4, line: 16, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !46)
// CHECK:STDOUT: !44 = !DILocalVariable(arg: 1, scope: !43, type: !11)
// CHECK:STDOUT: !45 = !DILocation(line: 16, column: 26, scope: !43)
// CHECK:STDOUT: !46 = !{!44}
// CHECK:STDOUT: !47 = !DIFile(filename: "{{.*}}/prelude/iterate.carbon", directory: "")
// CHECK:STDOUT: !48 = distinct !DISubprogram(name: "NewCursor", linkageName: "_CNewCursor.Array.d9c03cafdb3c033f.Core:Iterate.Core.9e8a519348dcc21a", scope: null, file: !47, line: 23, type: !22, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !54)
// CHECK:STDOUT: !49 = !DILocalVariable(arg: 1, scope: !48, type: !11)
// CHECK:STDOUT: !53 = !DILocation(line: 23, column: 38, scope: !48)
// CHECK:STDOUT: !54 = !{!49}
// CHECK:STDOUT: !55 = distinct !DISubprogram(name: "Next", linkageName: "_CNext.Array.d9c03cafdb3c033f.Core:Iterate.Core.9e8a519348dcc21a", scope: null, file: !47, line: 24, type: !24, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !76)
// CHECK:STDOUT: !56 = !DILocalVariable(arg: 1, scope: !55, type: !11)
// CHECK:STDOUT: !57 = !DILocalVariable(arg: 2, scope: !55, type: !6)
// CHECK:STDOUT: !62 = !DILocation(line: 25, column: 9, scope: !55)
// CHECK:STDOUT: !63 = !DILocation(line: 25, column: 8, scope: !55)
// CHECK:STDOUT: !65 = !DILocation(line: 26, column: 7, scope: !55)
// CHECK:STDOUT: !66 = !DILocation(line: 27, column: 31, scope: !55)
// CHECK:STDOUT: !67 = !DILocation(line: 27, column: 36, scope: !55)
// CHECK:STDOUT: !68 = !DILocation(line: 27, column: 14, scope: !55)
// CHECK:STDOUT: !69 = !{!11, !6}
// CHECK:STDOUT: !70 = !DISubroutineType(types: !69)
// CHECK:STDOUT: !71 = !DILocation(line: 27, column: 7, scope: !55)
// CHECK:STDOUT: !72 = !DILocation(line: 29, column: 14, scope: !55)
// CHECK:STDOUT: !73 = !{!11}
// CHECK:STDOUT: !74 = !DISubroutineType(types: !73)
// CHECK:STDOUT: !75 = !DILocation(line: 29, column: 7, scope: !55)
// CHECK:STDOUT: !76 = !{!56, !57}
// CHECK:STDOUT: !77 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !78 = distinct !DISubprogram(name: "HasValue", linkageName: "_CHasValue.Optional.Core.f7d25475af49e62f", scope: null, file: !77, line: 36, type: !26, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !86)
// CHECK:STDOUT: !79 = !DILocalVariable(arg: 1, scope: !78, type: !11)
// CHECK:STDOUT: !84 = !DILocation(line: 37, column: 12, scope: !78)
// CHECK:STDOUT: !85 = !DILocation(line: 37, column: 5, scope: !78)
// CHECK:STDOUT: !86 = !{!79}
// CHECK:STDOUT: !87 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.Optional.Core.f7d25475af49e62f", scope: null, file: !77, line: 39, type: !22, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !95)
// CHECK:STDOUT: !88 = !DILocalVariable(arg: 1, scope: !87, type: !11)
// CHECK:STDOUT: !93 = !DILocation(line: 40, column: 12, scope: !87)
// CHECK:STDOUT: !94 = !DILocation(line: 40, column: 5, scope: !87)
// CHECK:STDOUT: !95 = !{!88}
// CHECK:STDOUT: !96 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !97 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.1b119d8ccf65a76f", scope: null, file: !96, line: 34, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !103)
// CHECK:STDOUT: !98 = !DILocalVariable(arg: 1, scope: !97, type: !11)
// CHECK:STDOUT: !102 = !DILocation(line: 34, column: 1, scope: !97)
// CHECK:STDOUT: !103 = !{!98}
// CHECK:STDOUT: !104 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.b9c4c4d5428719ba", scope: null, file: !96, line: 34, type: !8, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !110)
// CHECK:STDOUT: !105 = !DILocalVariable(arg: 1, scope: !104, type: !6)
// CHECK:STDOUT: !109 = !DILocation(line: 34, column: 1, scope: !104)
// CHECK:STDOUT: !110 = !{!105}
// CHECK:STDOUT: !111 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.f24fb663cda48956", scope: null, file: !96, line: 34, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !117)
// CHECK:STDOUT: !112 = !DILocalVariable(arg: 1, scope: !111, type: !11)
// CHECK:STDOUT: !115 = !DILocation(line: 36, column: 5, scope: !111)
// CHECK:STDOUT: !116 = !DILocation(line: 34, column: 1, scope: !111)
// CHECK:STDOUT: !117 = !{!112}
// CHECK:STDOUT: !118 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.94cb175d66c1dda9", scope: null, file: !96, line: 34, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !124)
// CHECK:STDOUT: !119 = !DILocalVariable(arg: 1, scope: !118, type: !11)
// CHECK:STDOUT: !123 = !DILocation(line: 34, column: 1, scope: !118)
// CHECK:STDOUT: !124 = !{!119}
// CHECK:STDOUT: !125 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.56a8014ecefcdde9", scope: null, file: !96, line: 34, type: !8, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !131)
// CHECK:STDOUT: !126 = !DILocalVariable(arg: 1, scope: !125, type: !6)
// CHECK:STDOUT: !130 = !DILocation(line: 34, column: 1, scope: !125)
// CHECK:STDOUT: !131 = !{!126}
// CHECK:STDOUT: !132 = !DIFile(filename: "{{.*}}/prelude/types/int.carbon", directory: "")
// CHECK:STDOUT: !133 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8", scope: null, file: !132, line: 396, type: !8, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !140)
// CHECK:STDOUT: !134 = !DILocalVariable(arg: 1, scope: !133, type: !6)
// CHECK:STDOUT: !136 = !DILocation(line: 398, column: 5, scope: !133)
// CHECK:STDOUT: !137 = !{null, !6, !6}
// CHECK:STDOUT: !138 = !DISubroutineType(types: !137)
// CHECK:STDOUT: !139 = !DILocation(line: 396, column: 3, scope: !133)
// CHECK:STDOUT: !140 = !{!134}
// CHECK:STDOUT: !141 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.f7d25475af49e62f", scope: null, file: !77, line: 33, type: !70, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !148)
// CHECK:STDOUT: !142 = !DILocalVariable(arg: 1, scope: !141, type: !6)
// CHECK:STDOUT: !146 = !DILocation(line: 34, column: 12, scope: !141)
// CHECK:STDOUT: !147 = !DILocation(line: 34, column: 5, scope: !141)
// CHECK:STDOUT: !148 = !{!142}
// CHECK:STDOUT: !149 = distinct !DISubprogram(name: "None", linkageName: "_CNone.Optional.Core.f7d25475af49e62f", scope: null, file: !77, line: 30, type: !74, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !152 = !DILocation(line: 31, column: 12, scope: !149)
// CHECK:STDOUT: !153 = !DILocation(line: 31, column: 5, scope: !149)
// CHECK:STDOUT: !154 = distinct !DISubprogram(name: "Has", linkageName: "_CHas.21149ca0dc753fdf:OptionalStorage.Core.f39dc4a0608179aa", scope: null, file: !77, line: 141, type: !26, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !161)
// CHECK:STDOUT: !155 = !DILocalVariable(arg: 1, scope: !154, type: !11)
// CHECK:STDOUT: !159 = !DILocation(line: 142, column: 12, scope: !154)
// CHECK:STDOUT: !160 = !DILocation(line: 142, column: 5, scope: !154)
// CHECK:STDOUT: !161 = !{!155}
// CHECK:STDOUT: !162 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.21149ca0dc753fdf:OptionalStorage.Core.f39dc4a0608179aa", scope: null, file: !77, line: 144, type: !22, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !169)
// CHECK:STDOUT: !163 = !DILocalVariable(arg: 1, scope: !162, type: !11)
// CHECK:STDOUT: !167 = !DILocation(line: 145, column: 12, scope: !162)
// CHECK:STDOUT: !168 = !DILocation(line: 145, column: 5, scope: !162)
// CHECK:STDOUT: !169 = !{!163}
// CHECK:STDOUT: !170 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:AddAssignWith.b59e374b0649c9ad.Core:Int.9452f4c51951679b.Core:AddAssignWith.ab03190a9e911cb4.Core.306167be98c24f26", scope: null, file: !132, line: 331, type: !138, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !177)
// CHECK:STDOUT: !171 = !DILocalVariable(arg: 1, scope: !170, type: !6)
// CHECK:STDOUT: !172 = !DILocalVariable(arg: 2, scope: !170, type: !6)
// CHECK:STDOUT: !174 = !DILocation(line: 331, column: 3, scope: !170)
// CHECK:STDOUT: !175 = !{!6, !6}
// CHECK:STDOUT: !176 = !DISubroutineType(types: !175)
// CHECK:STDOUT: !177 = !{!171, !172}
// CHECK:STDOUT: !178 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.21149ca0dc753fdf:OptionalStorage.Core.f39dc4a0608179aa", scope: null, file: !77, line: 131, type: !70, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !191)
// CHECK:STDOUT: !179 = !DILocalVariable(arg: 1, scope: !178, type: !6)
// CHECK:STDOUT: !187 = !DILocation(line: 137, column: 5, scope: !178)
// CHECK:STDOUT: !189 = !DILocation(line: 138, column: 5, scope: !178)
// CHECK:STDOUT: !190 = !DILocation(line: 139, column: 5, scope: !178)
// CHECK:STDOUT: !191 = !{!179}
// CHECK:STDOUT: !192 = distinct !DISubprogram(name: "None", linkageName: "_CNone.21149ca0dc753fdf:OptionalStorage.Core.f39dc4a0608179aa", scope: null, file: !77, line: 125, type: !74, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !199 = !DILocation(line: 128, column: 5, scope: !192)
// CHECK:STDOUT: !200 = !DILocation(line: 129, column: 5, scope: !192)
// CHECK:STDOUT: !201 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.df8c28cb3df17b7d:ImplicitAs.c95da57eee1d2d0f.Core.25ef994321ea0701", scope: null, file: !132, line: 64, type: !176, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !209)
// CHECK:STDOUT: !202 = !DILocalVariable(arg: 1, scope: !201, type: !6)
// CHECK:STDOUT: !206 = !DILocation(line: 66, column: 17, scope: !201)
// CHECK:STDOUT: !208 = !DILocation(line: 66, column: 5, scope: !201)
// CHECK:STDOUT: !209 = !{!202}
// CHECK:STDOUT: !210 = distinct !DISubprogram(name: "AsInt", linkageName: "_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8", scope: null, file: !132, line: 59, type: !176, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !217)
// CHECK:STDOUT: !211 = !DILocalVariable(arg: 1, scope: !210, type: !6)
// CHECK:STDOUT: !216 = !DILocation(line: 59, column: 28, scope: !210)
// CHECK:STDOUT: !217 = !{!211}
// CHECK:STDOUT:
