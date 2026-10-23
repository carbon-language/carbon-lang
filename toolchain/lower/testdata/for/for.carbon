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
// CHECK:STDOUT: define void @_CFor.Main() #0 !dbg !14 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc19_32.1.temp = alloca { i32, i32 }, align 4, !dbg !17
// CHECK:STDOUT:   %var = alloca i32, align 4, !dbg !18
// CHECK:STDOUT:   %.loc19_33.1.temp = alloca <{ i32, i1 }>, align 4, !dbg !18
// CHECK:STDOUT:   call void @_CF.Main(), !dbg !15
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc19_32.1.temp), !dbg !17
// CHECK:STDOUT:   call void @_CRange.Core(ptr %.loc19_32.1.temp, i32 100), !dbg !17
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.NewCursor.call = call i32 @"_CNewCursor.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8"(ptr %.loc19_32.1.temp), !dbg !18
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %var), !dbg !18
// CHECK:STDOUT:   store i32 %IntRange.as.Iterate.impl.NewCursor.call, ptr %var, align 4, !dbg !18
// CHECK:STDOUT:   br label %for.next, !dbg !18
// CHECK:STDOUT:
// CHECK:STDOUT: for.next:                                         ; preds = %for.body, %entry
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc19_33.1.temp), !dbg !18
// CHECK:STDOUT:   call void @"_CNext.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8"(ptr %.loc19_33.1.temp, ptr %.loc19_32.1.temp, ptr %var), !dbg !18
// CHECK:STDOUT:   %Optional.HasValue.call = call i1 @_CHasValue.Optional.Core.f7d25475af49e62f(ptr %.loc19_33.1.temp), !dbg !18
// CHECK:STDOUT:   br i1 %Optional.HasValue.call, label %for.body, label %for.done, !dbg !18
// CHECK:STDOUT:
// CHECK:STDOUT: for.body:                                         ; preds = %for.next
// CHECK:STDOUT:   %Optional.Get.call = call i32 @_CGet.Optional.Core.f7d25475af49e62f(ptr %.loc19_33.1.temp), !dbg !18
// CHECK:STDOUT:   call void @_CG.Main(), !dbg !25
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.1b119d8ccf65a76f(ptr %.loc19_33.1.temp), !dbg !18
// CHECK:STDOUT:   br label %for.next, !dbg !26
// CHECK:STDOUT:
// CHECK:STDOUT: for.done:                                         ; preds = %for.next
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.1b119d8ccf65a76f(ptr %.loc19_33.1.temp), !dbg !18
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.b9c4c4d5428719ba(ptr %var), !dbg !18
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.bc2663835398caae(ptr %.loc19_32.1.temp), !dbg !17
// CHECK:STDOUT:   call void @_CH.Main(), !dbg !29
// CHECK:STDOUT:   ret void, !dbg !30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CRange.Core(ptr sret({ i32, i32 }), i32)
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
// CHECK:STDOUT:   %.loc19_33.1.has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %self, i32 0, i32 1, !dbg !37
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.94cb175d66c1dda9(ptr %.loc19_33.1.has_value), !dbg !37
// CHECK:STDOUT:   %.loc19_33.4.value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %self, i32 0, i32 0, !dbg !37
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.56a8014ecefcdde9(ptr %.loc19_33.4.value), !dbg !37
// CHECK:STDOUT:   ret void, !dbg !37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.07c81106aaf54f73:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !39 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc19_33.1.has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %self, i32 0, i32 1, !dbg !41
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.94cb175d66c1dda9(ptr %.loc19_33.1.has_value), !dbg !41
// CHECK:STDOUT:   %.loc19_33.4.value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %self, i32 0, i32 0, !dbg !41
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.56a8014ecefcdde9(ptr %.loc19_33.4.value), !dbg !41
// CHECK:STDOUT:   ret void, !dbg !41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.7e86931818f7778c:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !43 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc19_32.1.end = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 1, !dbg !45
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.b9c4c4d5428719ba(ptr %.loc19_32.1.end), !dbg !45
// CHECK:STDOUT:   %.loc19_32.4.start = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 0, !dbg !45
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.b9c4c4d5428719ba(ptr %.loc19_32.4.start), !dbg !45
// CHECK:STDOUT:   ret void, !dbg !45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.30087fca60df1631:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !47 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc19_32.1.end = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 1, !dbg !49
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.b9c4c4d5428719ba(ptr %.loc19_32.1.end), !dbg !49
// CHECK:STDOUT:   %.loc19_32.4.start = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 0, !dbg !49
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.b9c4c4d5428719ba(ptr %.loc19_32.4.start), !dbg !49
// CHECK:STDOUT:   ret void, !dbg !49
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CNewCursor.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8"(ptr %self) #0 !dbg !54 {
// CHECK:STDOUT:   %start = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 0, !dbg !59
// CHECK:STDOUT:   %1 = load i32, ptr %start, align 4, !dbg !59
// CHECK:STDOUT:   ret i32 %1, !dbg !60
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNext.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8"(ptr sret(<{ i32, i1 }>) %return, ptr %self, ptr %cursor) #0 !dbg !62 {
// CHECK:STDOUT:   %1 = alloca i32, align 4, !dbg !69
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %1), !dbg !69
// CHECK:STDOUT:   %2 = load i32, ptr %cursor, align 4, !dbg !70
// CHECK:STDOUT:   store i32 %2, ptr %1, align 4, !dbg !69
// CHECK:STDOUT:   %end = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 1, !dbg !73
// CHECK:STDOUT:   %3 = load i32, ptr %end, align 4, !dbg !73
// CHECK:STDOUT:   %4 = load i32, ptr %1, align 4, !dbg !72
// CHECK:STDOUT:   %5 = icmp slt i32 %4, %3, !dbg !72
// CHECK:STDOUT:   br i1 %5, label %6, label %8, !dbg !74
// CHECK:STDOUT:
// CHECK:STDOUT: 6:                                                ; preds = %0
// CHECK:STDOUT:   call void @"_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8"(ptr %cursor), !dbg !76
// CHECK:STDOUT:   %7 = load i32, ptr %1, align 4, !dbg !77
// CHECK:STDOUT:   call void @_CSome.Optional.Core.f7d25475af49e62f(ptr %return, i32 %7), !dbg !78
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.b9c4c4d5428719ba(ptr %1), !dbg !69
// CHECK:STDOUT:   ret void, !dbg !79
// CHECK:STDOUT:
// CHECK:STDOUT: 8:                                                ; preds = %0
// CHECK:STDOUT:   call void @_CNone.Optional.Core.f7d25475af49e62f(ptr %return), !dbg !80
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.b9c4c4d5428719ba(ptr %1), !dbg !69
// CHECK:STDOUT:   ret void, !dbg !83
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CHasValue.Optional.Core.f7d25475af49e62f(ptr %self) #0 !dbg !86 {
// CHECK:STDOUT:   %1 = call i1 @"_CHas.21149ca0dc753fdf:OptionalStorage.Core.f39dc4a0608179aa"(ptr %self), !dbg !92
// CHECK:STDOUT:   ret i1 %1, !dbg !93
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CGet.Optional.Core.f7d25475af49e62f(ptr %self) #0 !dbg !95 {
// CHECK:STDOUT:   %1 = call i32 @"_CGet.21149ca0dc753fdf:OptionalStorage.Core.f39dc4a0608179aa"(ptr %self), !dbg !101
// CHECK:STDOUT:   ret i32 %1, !dbg !102
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.1b119d8ccf65a76f(ptr %t) #0 !dbg !105 {
// CHECK:STDOUT:   ret void, !dbg !110
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.b9c4c4d5428719ba(ptr %t) #0 !dbg !112 {
// CHECK:STDOUT:   ret void, !dbg !117
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.bc2663835398caae(ptr %t) #0 !dbg !119 {
// CHECK:STDOUT:   call void @"_COp.30087fca60df1631:core.SubobjectDestroy.Core"(ptr %t), !dbg !123
// CHECK:STDOUT:   ret void, !dbg !124
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.94cb175d66c1dda9(ptr %t) #0 !dbg !126 {
// CHECK:STDOUT:   ret void, !dbg !131
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.56a8014ecefcdde9(ptr %t) #0 !dbg !133 {
// CHECK:STDOUT:   ret void, !dbg !138
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CInclusiveRange.Core(ptr sret({ i32, i32 }), i32, i32)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8"(ptr %self) #0 !dbg !141 {
// CHECK:STDOUT:   call void @"_COp:thunk:AddAssignWith.b59e374b0649c9ad.Core:Int.9452f4c51951679b.Core:AddAssignWith.ab03190a9e911cb4.Core.306167be98c24f26"(ptr %self, i32 1), !dbg !144
// CHECK:STDOUT:   ret void, !dbg !147
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSome.Optional.Core.f7d25475af49e62f(ptr sret(<{ i32, i1 }>) %return, i32 %value) #0 !dbg !149 {
// CHECK:STDOUT:   call void @"_CSome.21149ca0dc753fdf:OptionalStorage.Core.f39dc4a0608179aa"(ptr %return, i32 %value), !dbg !154
// CHECK:STDOUT:   ret void, !dbg !155
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CNone.Optional.Core.f7d25475af49e62f(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !157 {
// CHECK:STDOUT:   call void @"_CNone.21149ca0dc753fdf:OptionalStorage.Core.f39dc4a0608179aa"(ptr %return), !dbg !160
// CHECK:STDOUT:   ret void, !dbg !161
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @"_CHas.21149ca0dc753fdf:OptionalStorage.Core.f39dc4a0608179aa"(ptr %value) #0 !dbg !162 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %value, i32 0, i32 1, !dbg !167
// CHECK:STDOUT:   %1 = load i8, ptr %has_value, align 1, !dbg !167
// CHECK:STDOUT:   %2 = trunc i8 %1 to i1, !dbg !167
// CHECK:STDOUT:   ret i1 %2, !dbg !168
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CGet.21149ca0dc753fdf:OptionalStorage.Core.f39dc4a0608179aa"(ptr %value) #0 !dbg !170 {
// CHECK:STDOUT:   %value1 = getelementptr inbounds nuw <{ i32, i1 }>, ptr %value, i32 0, i32 0, !dbg !175
// CHECK:STDOUT:   %1 = load i32, ptr %value1, align 4, !dbg !175
// CHECK:STDOUT:   ret i32 %1, !dbg !176
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp:thunk:AddAssignWith.b59e374b0649c9ad.Core:Int.9452f4c51951679b.Core:AddAssignWith.ab03190a9e911cb4.Core.306167be98c24f26"(ptr %self, i32 %other) #2 !dbg !178 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.df8c28cb3df17b7d:ImplicitAs.c95da57eee1d2d0f.Core.25ef994321ea0701"(i32 %other), !dbg !182
// CHECK:STDOUT:   %2 = load i32, ptr %self, align 4, !dbg !182
// CHECK:STDOUT:   %3 = add i32 %2, %1, !dbg !182
// CHECK:STDOUT:   store i32 %3, ptr %self, align 4, !dbg !182
// CHECK:STDOUT:   ret void, !dbg !182
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.21149ca0dc753fdf:OptionalStorage.Core.f39dc4a0608179aa"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !186 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 0, !dbg !195
// CHECK:STDOUT:   store i32 %self, ptr %value, align 4, !dbg !195
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !197
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !197
// CHECK:STDOUT:   ret void, !dbg !198
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNone.21149ca0dc753fdf:OptionalStorage.Core.f39dc4a0608179aa"(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !200 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !207
// CHECK:STDOUT:   store i8 0, ptr %has_value, align 1, !dbg !207
// CHECK:STDOUT:   ret void, !dbg !208
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.df8c28cb3df17b7d:ImplicitAs.c95da57eee1d2d0f.Core.25ef994321ea0701"(i32 %self) #0 !dbg !209 {
// CHECK:STDOUT:   %1 = call i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self), !dbg !214
// CHECK:STDOUT:   ret i32 %1, !dbg !216
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self) #0 !dbg !218 {
// CHECK:STDOUT:   ret i32 %self, !dbg !224
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 3, 2, 1 }
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
// CHECK:STDOUT: !4 = !DIFile(filename: "for.carbon", directory: "")
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !7 = !{null}
// CHECK:STDOUT: !8 = !DISubroutineType(types: !7)
// CHECK:STDOUT: !9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !10 = !{!9, !6}
// CHECK:STDOUT: !11 = !DISubroutineType(types: !10)
// CHECK:STDOUT: !12 = !{null, !9}
// CHECK:STDOUT: !13 = !DISubroutineType(types: !12)
// CHECK:STDOUT: !14 = distinct !DISubprogram(name: "For", linkageName: "_CFor.Main", scope: null, file: !4, line: 17, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !15 = !DILocation(line: 18, column: 3, scope: !14)
// CHECK:STDOUT: !17 = !DILocation(line: 19, column: 18, scope: !14)
// CHECK:STDOUT: !18 = !DILocation(line: 19, column: 7, scope: !14)
// CHECK:STDOUT: !19 = !{!6, !9}
// CHECK:STDOUT: !20 = !DISubroutineType(types: !19)
// CHECK:STDOUT: !21 = !{!9, !9, !6}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = !{!9, !9}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !25 = !DILocation(line: 20, column: 5, scope: !14)
// CHECK:STDOUT: !26 = !DILocation(line: 19, column: 3, scope: !14)
// CHECK:STDOUT: !27 = !{null, !6}
// CHECK:STDOUT: !28 = !DISubroutineType(types: !27)
// CHECK:STDOUT: !29 = !DILocation(line: 22, column: 3, scope: !14)
// CHECK:STDOUT: !30 = !DILocation(line: 17, column: 1, scope: !14)
// CHECK:STDOUT: !31 = distinct !DISubprogram(name: "Op", linkageName: "_COp.b3126fe3b97d1b85:core.SubobjectDestroy.Core", scope: null, file: !4, line: 19, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !34)
// CHECK:STDOUT: !32 = !DILocalVariable(arg: 1, scope: !31, type: !9)
// CHECK:STDOUT: !33 = !DILocation(line: 19, column: 7, scope: !31)
// CHECK:STDOUT: !34 = !{!32}
// CHECK:STDOUT: !35 = distinct !DISubprogram(name: "Op", linkageName: "_COp.b29714547c418f6c:core.SubobjectDestroy.Core", scope: null, file: !4, line: 19, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !38)
// CHECK:STDOUT: !36 = !DILocalVariable(arg: 1, scope: !35, type: !9)
// CHECK:STDOUT: !37 = !DILocation(line: 19, column: 7, scope: !35)
// CHECK:STDOUT: !38 = !{!36}
// CHECK:STDOUT: !39 = distinct !DISubprogram(name: "Op", linkageName: "_COp.07c81106aaf54f73:core.SubobjectDestroy.Core", scope: null, file: !4, line: 19, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !42)
// CHECK:STDOUT: !40 = !DILocalVariable(arg: 1, scope: !39, type: !9)
// CHECK:STDOUT: !41 = !DILocation(line: 19, column: 7, scope: !39)
// CHECK:STDOUT: !42 = !{!40}
// CHECK:STDOUT: !43 = distinct !DISubprogram(name: "Op", linkageName: "_COp.7e86931818f7778c:core.SubobjectDestroy.Core", scope: null, file: !4, line: 19, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !46)
// CHECK:STDOUT: !44 = !DILocalVariable(arg: 1, scope: !43, type: !9)
// CHECK:STDOUT: !45 = !DILocation(line: 19, column: 18, scope: !43)
// CHECK:STDOUT: !46 = !{!44}
// CHECK:STDOUT: !47 = distinct !DISubprogram(name: "Op", linkageName: "_COp.30087fca60df1631:core.SubobjectDestroy.Core", scope: null, file: !4, line: 19, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !50)
// CHECK:STDOUT: !48 = !DILocalVariable(arg: 1, scope: !47, type: !9)
// CHECK:STDOUT: !49 = !DILocation(line: 19, column: 18, scope: !47)
// CHECK:STDOUT: !50 = !{!48}
// CHECK:STDOUT: !53 = !DIFile(filename: "{{.*}}/prelude/range.carbon", directory: "")
// CHECK:STDOUT: !54 = distinct !DISubprogram(name: "NewCursor", linkageName: "_CNewCursor.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8", scope: null, file: !53, line: 19, type: !20, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !61)
// CHECK:STDOUT: !55 = !DILocalVariable(arg: 1, scope: !54, type: !9)
// CHECK:STDOUT: !59 = !DILocation(line: 19, column: 43, scope: !54)
// CHECK:STDOUT: !60 = !DILocation(line: 19, column: 36, scope: !54)
// CHECK:STDOUT: !61 = !{!55}
// CHECK:STDOUT: !62 = distinct !DISubprogram(name: "Next", linkageName: "_CNext.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8", scope: null, file: !53, line: 20, type: !22, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !84)
// CHECK:STDOUT: !63 = !DILocalVariable(arg: 1, scope: !62, type: !9)
// CHECK:STDOUT: !64 = !DILocalVariable(arg: 2, scope: !62, type: !6)
// CHECK:STDOUT: !69 = !DILocation(line: 21, column: 7, scope: !62)
// CHECK:STDOUT: !70 = !DILocation(line: 21, column: 27, scope: !62)
// CHECK:STDOUT: !72 = !DILocation(line: 22, column: 11, scope: !62)
// CHECK:STDOUT: !73 = !DILocation(line: 22, column: 19, scope: !62)
// CHECK:STDOUT: !74 = !DILocation(line: 22, column: 10, scope: !62)
// CHECK:STDOUT: !76 = !DILocation(line: 23, column: 9, scope: !62)
// CHECK:STDOUT: !77 = !DILocation(line: 24, column: 38, scope: !62)
// CHECK:STDOUT: !78 = !DILocation(line: 24, column: 16, scope: !62)
// CHECK:STDOUT: !79 = !DILocation(line: 24, column: 9, scope: !62)
// CHECK:STDOUT: !80 = !DILocation(line: 26, column: 16, scope: !62)
// CHECK:STDOUT: !81 = !{!9}
// CHECK:STDOUT: !82 = !DISubroutineType(types: !81)
// CHECK:STDOUT: !83 = !DILocation(line: 26, column: 9, scope: !62)
// CHECK:STDOUT: !84 = !{!63, !64}
// CHECK:STDOUT: !85 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !86 = distinct !DISubprogram(name: "HasValue", linkageName: "_CHasValue.Optional.Core.f7d25475af49e62f", scope: null, file: !85, line: 36, type: !24, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !94)
// CHECK:STDOUT: !87 = !DILocalVariable(arg: 1, scope: !86, type: !9)
// CHECK:STDOUT: !92 = !DILocation(line: 37, column: 12, scope: !86)
// CHECK:STDOUT: !93 = !DILocation(line: 37, column: 5, scope: !86)
// CHECK:STDOUT: !94 = !{!87}
// CHECK:STDOUT: !95 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.Optional.Core.f7d25475af49e62f", scope: null, file: !85, line: 39, type: !20, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !103)
// CHECK:STDOUT: !96 = !DILocalVariable(arg: 1, scope: !95, type: !9)
// CHECK:STDOUT: !101 = !DILocation(line: 40, column: 12, scope: !95)
// CHECK:STDOUT: !102 = !DILocation(line: 40, column: 5, scope: !95)
// CHECK:STDOUT: !103 = !{!96}
// CHECK:STDOUT: !104 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !105 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.1b119d8ccf65a76f", scope: null, file: !104, line: 34, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !111)
// CHECK:STDOUT: !106 = !DILocalVariable(arg: 1, scope: !105, type: !9)
// CHECK:STDOUT: !110 = !DILocation(line: 34, column: 1, scope: !105)
// CHECK:STDOUT: !111 = !{!106}
// CHECK:STDOUT: !112 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.b9c4c4d5428719ba", scope: null, file: !104, line: 34, type: !28, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !118)
// CHECK:STDOUT: !113 = !DILocalVariable(arg: 1, scope: !112, type: !6)
// CHECK:STDOUT: !117 = !DILocation(line: 34, column: 1, scope: !112)
// CHECK:STDOUT: !118 = !{!113}
// CHECK:STDOUT: !119 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.bc2663835398caae", scope: null, file: !104, line: 34, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !125)
// CHECK:STDOUT: !120 = !DILocalVariable(arg: 1, scope: !119, type: !9)
// CHECK:STDOUT: !123 = !DILocation(line: 36, column: 5, scope: !119)
// CHECK:STDOUT: !124 = !DILocation(line: 34, column: 1, scope: !119)
// CHECK:STDOUT: !125 = !{!120}
// CHECK:STDOUT: !126 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.94cb175d66c1dda9", scope: null, file: !104, line: 34, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !132)
// CHECK:STDOUT: !127 = !DILocalVariable(arg: 1, scope: !126, type: !9)
// CHECK:STDOUT: !131 = !DILocation(line: 34, column: 1, scope: !126)
// CHECK:STDOUT: !132 = !{!127}
// CHECK:STDOUT: !133 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.56a8014ecefcdde9", scope: null, file: !104, line: 34, type: !28, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !139)
// CHECK:STDOUT: !134 = !DILocalVariable(arg: 1, scope: !133, type: !6)
// CHECK:STDOUT: !138 = !DILocation(line: 34, column: 1, scope: !133)
// CHECK:STDOUT: !139 = !{!134}
// CHECK:STDOUT: !140 = !DIFile(filename: "{{.*}}/prelude/types/int.carbon", directory: "")
// CHECK:STDOUT: !141 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8", scope: null, file: !140, line: 396, type: !28, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !148)
// CHECK:STDOUT: !142 = !DILocalVariable(arg: 1, scope: !141, type: !6)
// CHECK:STDOUT: !144 = !DILocation(line: 398, column: 5, scope: !141)
// CHECK:STDOUT: !145 = !{null, !6, !6}
// CHECK:STDOUT: !146 = !DISubroutineType(types: !145)
// CHECK:STDOUT: !147 = !DILocation(line: 396, column: 3, scope: !141)
// CHECK:STDOUT: !148 = !{!142}
// CHECK:STDOUT: !149 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.f7d25475af49e62f", scope: null, file: !85, line: 33, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !156)
// CHECK:STDOUT: !150 = !DILocalVariable(arg: 1, scope: !149, type: !6)
// CHECK:STDOUT: !154 = !DILocation(line: 34, column: 12, scope: !149)
// CHECK:STDOUT: !155 = !DILocation(line: 34, column: 5, scope: !149)
// CHECK:STDOUT: !156 = !{!150}
// CHECK:STDOUT: !157 = distinct !DISubprogram(name: "None", linkageName: "_CNone.Optional.Core.f7d25475af49e62f", scope: null, file: !85, line: 30, type: !82, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !160 = !DILocation(line: 31, column: 12, scope: !157)
// CHECK:STDOUT: !161 = !DILocation(line: 31, column: 5, scope: !157)
// CHECK:STDOUT: !162 = distinct !DISubprogram(name: "Has", linkageName: "_CHas.21149ca0dc753fdf:OptionalStorage.Core.f39dc4a0608179aa", scope: null, file: !85, line: 141, type: !24, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !169)
// CHECK:STDOUT: !163 = !DILocalVariable(arg: 1, scope: !162, type: !9)
// CHECK:STDOUT: !167 = !DILocation(line: 142, column: 12, scope: !162)
// CHECK:STDOUT: !168 = !DILocation(line: 142, column: 5, scope: !162)
// CHECK:STDOUT: !169 = !{!163}
// CHECK:STDOUT: !170 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.21149ca0dc753fdf:OptionalStorage.Core.f39dc4a0608179aa", scope: null, file: !85, line: 144, type: !20, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !177)
// CHECK:STDOUT: !171 = !DILocalVariable(arg: 1, scope: !170, type: !9)
// CHECK:STDOUT: !175 = !DILocation(line: 145, column: 12, scope: !170)
// CHECK:STDOUT: !176 = !DILocation(line: 145, column: 5, scope: !170)
// CHECK:STDOUT: !177 = !{!171}
// CHECK:STDOUT: !178 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:AddAssignWith.b59e374b0649c9ad.Core:Int.9452f4c51951679b.Core:AddAssignWith.ab03190a9e911cb4.Core.306167be98c24f26", scope: null, file: !140, line: 331, type: !146, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !185)
// CHECK:STDOUT: !179 = !DILocalVariable(arg: 1, scope: !178, type: !6)
// CHECK:STDOUT: !180 = !DILocalVariable(arg: 2, scope: !178, type: !6)
// CHECK:STDOUT: !182 = !DILocation(line: 331, column: 3, scope: !178)
// CHECK:STDOUT: !183 = !{!6, !6}
// CHECK:STDOUT: !184 = !DISubroutineType(types: !183)
// CHECK:STDOUT: !185 = !{!179, !180}
// CHECK:STDOUT: !186 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.21149ca0dc753fdf:OptionalStorage.Core.f39dc4a0608179aa", scope: null, file: !85, line: 131, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !199)
// CHECK:STDOUT: !187 = !DILocalVariable(arg: 1, scope: !186, type: !6)
// CHECK:STDOUT: !195 = !DILocation(line: 137, column: 5, scope: !186)
// CHECK:STDOUT: !197 = !DILocation(line: 138, column: 5, scope: !186)
// CHECK:STDOUT: !198 = !DILocation(line: 139, column: 5, scope: !186)
// CHECK:STDOUT: !199 = !{!187}
// CHECK:STDOUT: !200 = distinct !DISubprogram(name: "None", linkageName: "_CNone.21149ca0dc753fdf:OptionalStorage.Core.f39dc4a0608179aa", scope: null, file: !85, line: 125, type: !82, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !207 = !DILocation(line: 128, column: 5, scope: !200)
// CHECK:STDOUT: !208 = !DILocation(line: 129, column: 5, scope: !200)
// CHECK:STDOUT: !209 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.df8c28cb3df17b7d:ImplicitAs.c95da57eee1d2d0f.Core.25ef994321ea0701", scope: null, file: !140, line: 64, type: !184, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !217)
// CHECK:STDOUT: !210 = !DILocalVariable(arg: 1, scope: !209, type: !6)
// CHECK:STDOUT: !214 = !DILocation(line: 66, column: 17, scope: !209)
// CHECK:STDOUT: !216 = !DILocation(line: 66, column: 5, scope: !209)
// CHECK:STDOUT: !217 = !{!210}
// CHECK:STDOUT: !218 = distinct !DISubprogram(name: "AsInt", linkageName: "_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8", scope: null, file: !140, line: 59, type: !184, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !225)
// CHECK:STDOUT: !219 = !DILocalVariable(arg: 1, scope: !218, type: !6)
// CHECK:STDOUT: !224 = !DILocation(line: 59, column: 28, scope: !218)
// CHECK:STDOUT: !225 = !{!219}
// CHECK:STDOUT:
