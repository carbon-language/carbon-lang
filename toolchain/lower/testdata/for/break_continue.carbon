// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/for/break_continue.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/for/break_continue.carbon

fn F() -> bool;
fn G() -> bool;
fn H();

fn For() {
  for (_: i32 in Core.Range(100)) {
    if (F()) { break; }
    if (G()) { continue; }
    H();
  }
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'break_continue.carbon'
// CHECK:STDOUT: source_filename = "break_continue.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare i1 @_CF.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: declare i1 @_CG.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CH.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CFor.Main() #0 !dbg !16 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc18_32.1.temp = alloca { i32, i32 }, align 4, !dbg !18
// CHECK:STDOUT:   %var = alloca i32, align 4, !dbg !19
// CHECK:STDOUT:   %.loc18_33.1.temp = alloca <{ i32, i1 }>, align 4, !dbg !19
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc18_32.1.temp), !dbg !18
// CHECK:STDOUT:   call void @_CRange.Core(ptr %.loc18_32.1.temp, i32 100), !dbg !18
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.NewCursor.call = call i32 @"_CNewCursor.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8"(ptr %.loc18_32.1.temp), !dbg !19
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %var), !dbg !19
// CHECK:STDOUT:   store i32 %IntRange.as.Iterate.impl.NewCursor.call, ptr %var, align 4, !dbg !19
// CHECK:STDOUT:   br label %for.next, !dbg !19
// CHECK:STDOUT:
// CHECK:STDOUT: for.next:                                         ; preds = %if.else.loc20, %if.then.loc20, %entry
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc18_33.1.temp), !dbg !19
// CHECK:STDOUT:   call void @"_CNext.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8"(ptr %.loc18_33.1.temp, ptr %.loc18_32.1.temp, ptr %var), !dbg !19
// CHECK:STDOUT:   %Optional.HasValue.call = call i1 @_CHasValue.Optional.Core.f7d25475af49e62f(ptr %.loc18_33.1.temp), !dbg !19
// CHECK:STDOUT:   br i1 %Optional.HasValue.call, label %for.body, label %for.done, !dbg !19
// CHECK:STDOUT:
// CHECK:STDOUT: for.body:                                         ; preds = %for.next
// CHECK:STDOUT:   %Optional.Get.call = call i32 @_CGet.Optional.Core.f7d25475af49e62f(ptr %.loc18_33.1.temp), !dbg !19
// CHECK:STDOUT:   %F.call = call i1 @_CF.Main(), !dbg !26
// CHECK:STDOUT:   br i1 %F.call, label %if.then.loc19, label %if.else.loc19, !dbg !27
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc19:                                    ; preds = %for.body
// CHECK:STDOUT:   br label %for.done, !dbg !28
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc19:                                    ; preds = %for.body
// CHECK:STDOUT:   %G.call = call i1 @_CG.Main(), !dbg !29
// CHECK:STDOUT:   br i1 %G.call, label %if.then.loc20, label %if.else.loc20, !dbg !30
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc20:                                    ; preds = %if.else.loc19
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.1b119d8ccf65a76f(ptr %.loc18_33.1.temp), !dbg !19
// CHECK:STDOUT:   br label %for.next, !dbg !31
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc20:                                    ; preds = %if.else.loc19
// CHECK:STDOUT:   call void @_CH.Main(), !dbg !32
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.1b119d8ccf65a76f(ptr %.loc18_33.1.temp), !dbg !19
// CHECK:STDOUT:   br label %for.next, !dbg !33
// CHECK:STDOUT:
// CHECK:STDOUT: for.done:                                         ; preds = %if.then.loc19, %for.next
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.1b119d8ccf65a76f(ptr %.loc18_33.1.temp), !dbg !19
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.b9c4c4d5428719ba(ptr %var), !dbg !19
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.bc2663835398caae(ptr %.loc18_32.1.temp), !dbg !18
// CHECK:STDOUT:   ret void, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CRange.Core(ptr sret({ i32, i32 }), i32)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.b3126fe3b97d1b85:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !37 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.b29714547c418f6c:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !41 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc18_33.1.has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %self, i32 0, i32 1, !dbg !43
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.94cb175d66c1dda9(ptr %.loc18_33.1.has_value), !dbg !43
// CHECK:STDOUT:   %.loc18_33.4.value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %self, i32 0, i32 0, !dbg !43
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.56a8014ecefcdde9(ptr %.loc18_33.4.value), !dbg !43
// CHECK:STDOUT:   ret void, !dbg !43
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.07c81106aaf54f73:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !45 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc18_33.1.has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %self, i32 0, i32 1, !dbg !47
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.94cb175d66c1dda9(ptr %.loc18_33.1.has_value), !dbg !47
// CHECK:STDOUT:   %.loc18_33.4.value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %self, i32 0, i32 0, !dbg !47
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.56a8014ecefcdde9(ptr %.loc18_33.4.value), !dbg !47
// CHECK:STDOUT:   ret void, !dbg !47
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.7e86931818f7778c:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !49 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc18_32.1.end = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 1, !dbg !51
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.b9c4c4d5428719ba(ptr %.loc18_32.1.end), !dbg !51
// CHECK:STDOUT:   %.loc18_32.4.start = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 0, !dbg !51
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.b9c4c4d5428719ba(ptr %.loc18_32.4.start), !dbg !51
// CHECK:STDOUT:   ret void, !dbg !51
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.30087fca60df1631:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !53 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc18_32.1.end = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 1, !dbg !55
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.b9c4c4d5428719ba(ptr %.loc18_32.1.end), !dbg !55
// CHECK:STDOUT:   %.loc18_32.4.start = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 0, !dbg !55
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.b9c4c4d5428719ba(ptr %.loc18_32.4.start), !dbg !55
// CHECK:STDOUT:   ret void, !dbg !55
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CNewCursor.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8"(ptr %self) #0 !dbg !60 {
// CHECK:STDOUT:   %start = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 0, !dbg !65
// CHECK:STDOUT:   %1 = load i32, ptr %start, align 4, !dbg !65
// CHECK:STDOUT:   ret i32 %1, !dbg !66
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNext.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8"(ptr sret(<{ i32, i1 }>) %return, ptr %self, ptr %cursor) #0 !dbg !68 {
// CHECK:STDOUT:   %1 = alloca i32, align 4, !dbg !75
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %1), !dbg !75
// CHECK:STDOUT:   %2 = load i32, ptr %cursor, align 4, !dbg !76
// CHECK:STDOUT:   store i32 %2, ptr %1, align 4, !dbg !75
// CHECK:STDOUT:   %end = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 1, !dbg !79
// CHECK:STDOUT:   %3 = load i32, ptr %end, align 4, !dbg !79
// CHECK:STDOUT:   %4 = load i32, ptr %1, align 4, !dbg !78
// CHECK:STDOUT:   %5 = icmp slt i32 %4, %3, !dbg !78
// CHECK:STDOUT:   br i1 %5, label %6, label %8, !dbg !80
// CHECK:STDOUT:
// CHECK:STDOUT: 6:                                                ; preds = %0
// CHECK:STDOUT:   call void @"_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8"(ptr %cursor), !dbg !82
// CHECK:STDOUT:   %7 = load i32, ptr %1, align 4, !dbg !83
// CHECK:STDOUT:   call void @_CSome.Optional.Core.f7d25475af49e62f(ptr %return, i32 %7), !dbg !84
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.b9c4c4d5428719ba(ptr %1), !dbg !75
// CHECK:STDOUT:   ret void, !dbg !85
// CHECK:STDOUT:
// CHECK:STDOUT: 8:                                                ; preds = %0
// CHECK:STDOUT:   call void @_CNone.Optional.Core.f7d25475af49e62f(ptr %return), !dbg !86
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.b9c4c4d5428719ba(ptr %1), !dbg !75
// CHECK:STDOUT:   ret void, !dbg !87
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CHasValue.Optional.Core.f7d25475af49e62f(ptr %self) #0 !dbg !90 {
// CHECK:STDOUT:   %1 = call i1 @"_CHas.21149ca0dc753fdf:OptionalStorage.Core.f39dc4a0608179aa"(ptr %self), !dbg !96
// CHECK:STDOUT:   ret i1 %1, !dbg !97
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CGet.Optional.Core.f7d25475af49e62f(ptr %self) #0 !dbg !99 {
// CHECK:STDOUT:   %1 = call i32 @"_CGet.21149ca0dc753fdf:OptionalStorage.Core.f39dc4a0608179aa"(ptr %self), !dbg !105
// CHECK:STDOUT:   ret i32 %1, !dbg !106
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.1b119d8ccf65a76f(ptr %t) #0 !dbg !109 {
// CHECK:STDOUT:   ret void, !dbg !114
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.b9c4c4d5428719ba(ptr %t) #0 !dbg !116 {
// CHECK:STDOUT:   ret void, !dbg !121
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.bc2663835398caae(ptr %t) #0 !dbg !123 {
// CHECK:STDOUT:   call void @"_COp.30087fca60df1631:core.SubobjectDestroy.Core"(ptr %t), !dbg !127
// CHECK:STDOUT:   ret void, !dbg !128
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.94cb175d66c1dda9(ptr %t) #0 !dbg !130 {
// CHECK:STDOUT:   ret void, !dbg !135
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.56a8014ecefcdde9(ptr %t) #0 !dbg !137 {
// CHECK:STDOUT:   ret void, !dbg !142
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CInclusiveRange.Core(ptr sret({ i32, i32 }), i32, i32)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8"(ptr %self) #0 !dbg !145 {
// CHECK:STDOUT:   call void @"_COp:thunk:AddAssignWith.b59e374b0649c9ad.Core:Int.9452f4c51951679b.Core:AddAssignWith.ab03190a9e911cb4.Core.306167be98c24f26"(ptr %self, i32 1), !dbg !148
// CHECK:STDOUT:   ret void, !dbg !151
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSome.Optional.Core.f7d25475af49e62f(ptr sret(<{ i32, i1 }>) %return, i32 %value) #0 !dbg !153 {
// CHECK:STDOUT:   call void @"_CSome.21149ca0dc753fdf:OptionalStorage.Core.f39dc4a0608179aa"(ptr %return, i32 %value), !dbg !158
// CHECK:STDOUT:   ret void, !dbg !159
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CNone.Optional.Core.f7d25475af49e62f(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !161 {
// CHECK:STDOUT:   call void @"_CNone.21149ca0dc753fdf:OptionalStorage.Core.f39dc4a0608179aa"(ptr %return), !dbg !164
// CHECK:STDOUT:   ret void, !dbg !165
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @"_CHas.21149ca0dc753fdf:OptionalStorage.Core.f39dc4a0608179aa"(ptr %value) #0 !dbg !166 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %value, i32 0, i32 1, !dbg !171
// CHECK:STDOUT:   %1 = load i8, ptr %has_value, align 1, !dbg !171
// CHECK:STDOUT:   %2 = trunc i8 %1 to i1, !dbg !171
// CHECK:STDOUT:   ret i1 %2, !dbg !172
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CGet.21149ca0dc753fdf:OptionalStorage.Core.f39dc4a0608179aa"(ptr %value) #0 !dbg !174 {
// CHECK:STDOUT:   %value1 = getelementptr inbounds nuw <{ i32, i1 }>, ptr %value, i32 0, i32 0, !dbg !179
// CHECK:STDOUT:   %1 = load i32, ptr %value1, align 4, !dbg !179
// CHECK:STDOUT:   ret i32 %1, !dbg !180
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp:thunk:AddAssignWith.b59e374b0649c9ad.Core:Int.9452f4c51951679b.Core:AddAssignWith.ab03190a9e911cb4.Core.306167be98c24f26"(ptr %self, i32 %other) #2 !dbg !182 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.df8c28cb3df17b7d:ImplicitAs.c95da57eee1d2d0f.Core.25ef994321ea0701"(i32 %other), !dbg !185
// CHECK:STDOUT:   %2 = load i32, ptr %self, align 4, !dbg !186
// CHECK:STDOUT:   %3 = add i32 %2, %1, !dbg !186
// CHECK:STDOUT:   store i32 %3, ptr %self, align 4, !dbg !186
// CHECK:STDOUT:   ret void, !dbg !186
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.21149ca0dc753fdf:OptionalStorage.Core.f39dc4a0608179aa"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !190 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 0, !dbg !199
// CHECK:STDOUT:   store i32 %self, ptr %value, align 4, !dbg !199
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !201
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !201
// CHECK:STDOUT:   ret void, !dbg !202
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNone.21149ca0dc753fdf:OptionalStorage.Core.f39dc4a0608179aa"(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !204 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !211
// CHECK:STDOUT:   store i8 0, ptr %has_value, align 1, !dbg !211
// CHECK:STDOUT:   ret void, !dbg !212
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.df8c28cb3df17b7d:ImplicitAs.c95da57eee1d2d0f.Core.25ef994321ea0701"(i32 %self) #0 !dbg !213 {
// CHECK:STDOUT:   %1 = call i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self), !dbg !218
// CHECK:STDOUT:   ret i32 %1, !dbg !220
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self) #0 !dbg !222 {
// CHECK:STDOUT:   ret i32 %self, !dbg !228
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 3, 2, 1 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.1b119d8ccf65a76f, { 2, 1, 0 }
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
// CHECK:STDOUT: !4 = !DIFile(filename: "break_continue.carbon", directory: "")
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{!7}
// CHECK:STDOUT: !9 = !DISubroutineType(types: !8)
// CHECK:STDOUT: !10 = !{null}
// CHECK:STDOUT: !11 = !DISubroutineType(types: !10)
// CHECK:STDOUT: !12 = !{!7, !6}
// CHECK:STDOUT: !13 = !DISubroutineType(types: !12)
// CHECK:STDOUT: !14 = !{null, !7}
// CHECK:STDOUT: !15 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !16 = distinct !DISubprogram(name: "For", linkageName: "_CFor.Main", scope: null, file: !4, line: 17, type: !11, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !18 = !DILocation(line: 18, column: 18, scope: !16)
// CHECK:STDOUT: !19 = !DILocation(line: 18, column: 7, scope: !16)
// CHECK:STDOUT: !20 = !{!6, !7}
// CHECK:STDOUT: !21 = !DISubroutineType(types: !20)
// CHECK:STDOUT: !22 = !{!7, !7, !6}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !24 = !{!7, !7}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = !DILocation(line: 19, column: 9, scope: !16)
// CHECK:STDOUT: !27 = !DILocation(line: 19, column: 8, scope: !16)
// CHECK:STDOUT: !28 = !DILocation(line: 19, column: 16, scope: !16)
// CHECK:STDOUT: !29 = !DILocation(line: 20, column: 9, scope: !16)
// CHECK:STDOUT: !30 = !DILocation(line: 20, column: 8, scope: !16)
// CHECK:STDOUT: !31 = !DILocation(line: 20, column: 16, scope: !16)
// CHECK:STDOUT: !32 = !DILocation(line: 21, column: 5, scope: !16)
// CHECK:STDOUT: !33 = !DILocation(line: 18, column: 3, scope: !16)
// CHECK:STDOUT: !34 = !{null, !6}
// CHECK:STDOUT: !35 = !DISubroutineType(types: !34)
// CHECK:STDOUT: !36 = !DILocation(line: 17, column: 1, scope: !16)
// CHECK:STDOUT: !37 = distinct !DISubprogram(name: "Op", linkageName: "_COp.b3126fe3b97d1b85:core.SubobjectDestroy.Core", scope: null, file: !4, line: 18, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !40)
// CHECK:STDOUT: !38 = !DILocalVariable(arg: 1, scope: !37, type: !7)
// CHECK:STDOUT: !39 = !DILocation(line: 18, column: 7, scope: !37)
// CHECK:STDOUT: !40 = !{!38}
// CHECK:STDOUT: !41 = distinct !DISubprogram(name: "Op", linkageName: "_COp.b29714547c418f6c:core.SubobjectDestroy.Core", scope: null, file: !4, line: 18, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !44)
// CHECK:STDOUT: !42 = !DILocalVariable(arg: 1, scope: !41, type: !7)
// CHECK:STDOUT: !43 = !DILocation(line: 18, column: 7, scope: !41)
// CHECK:STDOUT: !44 = !{!42}
// CHECK:STDOUT: !45 = distinct !DISubprogram(name: "Op", linkageName: "_COp.07c81106aaf54f73:core.SubobjectDestroy.Core", scope: null, file: !4, line: 18, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !48)
// CHECK:STDOUT: !46 = !DILocalVariable(arg: 1, scope: !45, type: !7)
// CHECK:STDOUT: !47 = !DILocation(line: 18, column: 7, scope: !45)
// CHECK:STDOUT: !48 = !{!46}
// CHECK:STDOUT: !49 = distinct !DISubprogram(name: "Op", linkageName: "_COp.7e86931818f7778c:core.SubobjectDestroy.Core", scope: null, file: !4, line: 18, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !52)
// CHECK:STDOUT: !50 = !DILocalVariable(arg: 1, scope: !49, type: !7)
// CHECK:STDOUT: !51 = !DILocation(line: 18, column: 18, scope: !49)
// CHECK:STDOUT: !52 = !{!50}
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "Op", linkageName: "_COp.30087fca60df1631:core.SubobjectDestroy.Core", scope: null, file: !4, line: 18, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !56)
// CHECK:STDOUT: !54 = !DILocalVariable(arg: 1, scope: !53, type: !7)
// CHECK:STDOUT: !55 = !DILocation(line: 18, column: 18, scope: !53)
// CHECK:STDOUT: !56 = !{!54}
// CHECK:STDOUT: !59 = !DIFile(filename: "{{.*}}/prelude/range.carbon", directory: "")
// CHECK:STDOUT: !60 = distinct !DISubprogram(name: "NewCursor", linkageName: "_CNewCursor.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8", scope: null, file: !59, line: 19, type: !21, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !67)
// CHECK:STDOUT: !61 = !DILocalVariable(arg: 1, scope: !60, type: !7)
// CHECK:STDOUT: !65 = !DILocation(line: 19, column: 43, scope: !60)
// CHECK:STDOUT: !66 = !DILocation(line: 19, column: 36, scope: !60)
// CHECK:STDOUT: !67 = !{!61}
// CHECK:STDOUT: !68 = distinct !DISubprogram(name: "Next", linkageName: "_CNext.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8", scope: null, file: !59, line: 20, type: !23, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !88)
// CHECK:STDOUT: !69 = !DILocalVariable(arg: 1, scope: !68, type: !7)
// CHECK:STDOUT: !70 = !DILocalVariable(arg: 2, scope: !68, type: !6)
// CHECK:STDOUT: !75 = !DILocation(line: 21, column: 7, scope: !68)
// CHECK:STDOUT: !76 = !DILocation(line: 21, column: 27, scope: !68)
// CHECK:STDOUT: !78 = !DILocation(line: 22, column: 11, scope: !68)
// CHECK:STDOUT: !79 = !DILocation(line: 22, column: 19, scope: !68)
// CHECK:STDOUT: !80 = !DILocation(line: 22, column: 10, scope: !68)
// CHECK:STDOUT: !82 = !DILocation(line: 23, column: 9, scope: !68)
// CHECK:STDOUT: !83 = !DILocation(line: 24, column: 38, scope: !68)
// CHECK:STDOUT: !84 = !DILocation(line: 24, column: 16, scope: !68)
// CHECK:STDOUT: !85 = !DILocation(line: 24, column: 9, scope: !68)
// CHECK:STDOUT: !86 = !DILocation(line: 26, column: 16, scope: !68)
// CHECK:STDOUT: !87 = !DILocation(line: 26, column: 9, scope: !68)
// CHECK:STDOUT: !88 = !{!69, !70}
// CHECK:STDOUT: !89 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !90 = distinct !DISubprogram(name: "HasValue", linkageName: "_CHasValue.Optional.Core.f7d25475af49e62f", scope: null, file: !89, line: 36, type: !25, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !98)
// CHECK:STDOUT: !91 = !DILocalVariable(arg: 1, scope: !90, type: !7)
// CHECK:STDOUT: !96 = !DILocation(line: 37, column: 12, scope: !90)
// CHECK:STDOUT: !97 = !DILocation(line: 37, column: 5, scope: !90)
// CHECK:STDOUT: !98 = !{!91}
// CHECK:STDOUT: !99 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.Optional.Core.f7d25475af49e62f", scope: null, file: !89, line: 39, type: !21, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !107)
// CHECK:STDOUT: !100 = !DILocalVariable(arg: 1, scope: !99, type: !7)
// CHECK:STDOUT: !105 = !DILocation(line: 40, column: 12, scope: !99)
// CHECK:STDOUT: !106 = !DILocation(line: 40, column: 5, scope: !99)
// CHECK:STDOUT: !107 = !{!100}
// CHECK:STDOUT: !108 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !109 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.1b119d8ccf65a76f", scope: null, file: !108, line: 34, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !115)
// CHECK:STDOUT: !110 = !DILocalVariable(arg: 1, scope: !109, type: !7)
// CHECK:STDOUT: !114 = !DILocation(line: 34, column: 1, scope: !109)
// CHECK:STDOUT: !115 = !{!110}
// CHECK:STDOUT: !116 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.b9c4c4d5428719ba", scope: null, file: !108, line: 34, type: !35, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !122)
// CHECK:STDOUT: !117 = !DILocalVariable(arg: 1, scope: !116, type: !6)
// CHECK:STDOUT: !121 = !DILocation(line: 34, column: 1, scope: !116)
// CHECK:STDOUT: !122 = !{!117}
// CHECK:STDOUT: !123 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.bc2663835398caae", scope: null, file: !108, line: 34, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !129)
// CHECK:STDOUT: !124 = !DILocalVariable(arg: 1, scope: !123, type: !7)
// CHECK:STDOUT: !127 = !DILocation(line: 36, column: 5, scope: !123)
// CHECK:STDOUT: !128 = !DILocation(line: 34, column: 1, scope: !123)
// CHECK:STDOUT: !129 = !{!124}
// CHECK:STDOUT: !130 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.94cb175d66c1dda9", scope: null, file: !108, line: 34, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !136)
// CHECK:STDOUT: !131 = !DILocalVariable(arg: 1, scope: !130, type: !7)
// CHECK:STDOUT: !135 = !DILocation(line: 34, column: 1, scope: !130)
// CHECK:STDOUT: !136 = !{!131}
// CHECK:STDOUT: !137 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.56a8014ecefcdde9", scope: null, file: !108, line: 34, type: !35, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !143)
// CHECK:STDOUT: !138 = !DILocalVariable(arg: 1, scope: !137, type: !6)
// CHECK:STDOUT: !142 = !DILocation(line: 34, column: 1, scope: !137)
// CHECK:STDOUT: !143 = !{!138}
// CHECK:STDOUT: !144 = !DIFile(filename: "{{.*}}/prelude/types/int.carbon", directory: "")
// CHECK:STDOUT: !145 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8", scope: null, file: !144, line: 396, type: !35, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !152)
// CHECK:STDOUT: !146 = !DILocalVariable(arg: 1, scope: !145, type: !6)
// CHECK:STDOUT: !148 = !DILocation(line: 398, column: 5, scope: !145)
// CHECK:STDOUT: !149 = !{null, !6, !6}
// CHECK:STDOUT: !150 = !DISubroutineType(types: !149)
// CHECK:STDOUT: !151 = !DILocation(line: 396, column: 3, scope: !145)
// CHECK:STDOUT: !152 = !{!146}
// CHECK:STDOUT: !153 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.f7d25475af49e62f", scope: null, file: !89, line: 33, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !160)
// CHECK:STDOUT: !154 = !DILocalVariable(arg: 1, scope: !153, type: !6)
// CHECK:STDOUT: !158 = !DILocation(line: 34, column: 12, scope: !153)
// CHECK:STDOUT: !159 = !DILocation(line: 34, column: 5, scope: !153)
// CHECK:STDOUT: !160 = !{!154}
// CHECK:STDOUT: !161 = distinct !DISubprogram(name: "None", linkageName: "_CNone.Optional.Core.f7d25475af49e62f", scope: null, file: !89, line: 30, type: !9, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !164 = !DILocation(line: 31, column: 12, scope: !161)
// CHECK:STDOUT: !165 = !DILocation(line: 31, column: 5, scope: !161)
// CHECK:STDOUT: !166 = distinct !DISubprogram(name: "Has", linkageName: "_CHas.21149ca0dc753fdf:OptionalStorage.Core.f39dc4a0608179aa", scope: null, file: !89, line: 141, type: !25, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !173)
// CHECK:STDOUT: !167 = !DILocalVariable(arg: 1, scope: !166, type: !7)
// CHECK:STDOUT: !171 = !DILocation(line: 142, column: 12, scope: !166)
// CHECK:STDOUT: !172 = !DILocation(line: 142, column: 5, scope: !166)
// CHECK:STDOUT: !173 = !{!167}
// CHECK:STDOUT: !174 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.21149ca0dc753fdf:OptionalStorage.Core.f39dc4a0608179aa", scope: null, file: !89, line: 144, type: !21, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !181)
// CHECK:STDOUT: !175 = !DILocalVariable(arg: 1, scope: !174, type: !7)
// CHECK:STDOUT: !179 = !DILocation(line: 145, column: 12, scope: !174)
// CHECK:STDOUT: !180 = !DILocation(line: 145, column: 5, scope: !174)
// CHECK:STDOUT: !181 = !{!175}
// CHECK:STDOUT: !182 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:AddAssignWith.b59e374b0649c9ad.Core:Int.9452f4c51951679b.Core:AddAssignWith.ab03190a9e911cb4.Core.306167be98c24f26", scope: null, file: !144, line: 331, type: !150, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !189)
// CHECK:STDOUT: !183 = !DILocalVariable(arg: 1, scope: !182, type: !6)
// CHECK:STDOUT: !184 = !DILocalVariable(arg: 2, scope: !182, type: !6)
// CHECK:STDOUT: !185 = !DILocation(line: 0, scope: !182)
// CHECK:STDOUT: !186 = !DILocation(line: 331, column: 3, scope: !182)
// CHECK:STDOUT: !187 = !{!6, !6}
// CHECK:STDOUT: !188 = !DISubroutineType(types: !187)
// CHECK:STDOUT: !189 = !{!183, !184}
// CHECK:STDOUT: !190 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.21149ca0dc753fdf:OptionalStorage.Core.f39dc4a0608179aa", scope: null, file: !89, line: 131, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !203)
// CHECK:STDOUT: !191 = !DILocalVariable(arg: 1, scope: !190, type: !6)
// CHECK:STDOUT: !199 = !DILocation(line: 137, column: 5, scope: !190)
// CHECK:STDOUT: !201 = !DILocation(line: 138, column: 5, scope: !190)
// CHECK:STDOUT: !202 = !DILocation(line: 139, column: 5, scope: !190)
// CHECK:STDOUT: !203 = !{!191}
// CHECK:STDOUT: !204 = distinct !DISubprogram(name: "None", linkageName: "_CNone.21149ca0dc753fdf:OptionalStorage.Core.f39dc4a0608179aa", scope: null, file: !89, line: 125, type: !9, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !211 = !DILocation(line: 128, column: 5, scope: !204)
// CHECK:STDOUT: !212 = !DILocation(line: 129, column: 5, scope: !204)
// CHECK:STDOUT: !213 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.df8c28cb3df17b7d:ImplicitAs.c95da57eee1d2d0f.Core.25ef994321ea0701", scope: null, file: !144, line: 64, type: !188, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !221)
// CHECK:STDOUT: !214 = !DILocalVariable(arg: 1, scope: !213, type: !6)
// CHECK:STDOUT: !218 = !DILocation(line: 66, column: 17, scope: !213)
// CHECK:STDOUT: !220 = !DILocation(line: 66, column: 5, scope: !213)
// CHECK:STDOUT: !221 = !{!214}
// CHECK:STDOUT: !222 = distinct !DISubprogram(name: "AsInt", linkageName: "_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8", scope: null, file: !144, line: 59, type: !188, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !229)
// CHECK:STDOUT: !223 = !DILocalVariable(arg: 1, scope: !222, type: !6)
// CHECK:STDOUT: !228 = !DILocation(line: 59, column: 28, scope: !222)
// CHECK:STDOUT: !229 = !{!223}
// CHECK:STDOUT:
