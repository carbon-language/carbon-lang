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
// CHECK:STDOUT: define void @_CFor.Main() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc18_32.1.temp = alloca { i32, i32 }, align 4, !dbg !7
// CHECK:STDOUT:   %var = alloca i32, align 4, !dbg !8
// CHECK:STDOUT:   %.loc18_33.1.temp = alloca <{ i32, i1 }>, align 4, !dbg !8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc18_32.1.temp), !dbg !7
// CHECK:STDOUT:   call void @_CRange.Core(ptr %.loc18_32.1.temp, i32 100), !dbg !7
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.NewCursor.call = call i32 @"_CNewCursor.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8"(ptr %.loc18_32.1.temp), !dbg !8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %var), !dbg !8
// CHECK:STDOUT:   store i32 %IntRange.as.Iterate.impl.NewCursor.call, ptr %var, align 4, !dbg !8
// CHECK:STDOUT:   br label %for.next, !dbg !8
// CHECK:STDOUT:
// CHECK:STDOUT: for.next:                                         ; preds = %if.else.loc20, %if.then.loc20, %entry
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc18_33.1.temp), !dbg !8
// CHECK:STDOUT:   call void @"_CNext.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8"(ptr %.loc18_33.1.temp, ptr %.loc18_32.1.temp, ptr %var), !dbg !8
// CHECK:STDOUT:   %Optional.HasValue.call = call i1 @_CHasValue.Optional.Core.ba7a4025e0666ad5(ptr %.loc18_33.1.temp), !dbg !8
// CHECK:STDOUT:   br i1 %Optional.HasValue.call, label %for.body, label %for.done, !dbg !8
// CHECK:STDOUT:
// CHECK:STDOUT: for.body:                                         ; preds = %for.next
// CHECK:STDOUT:   %Optional.Get.call = call i32 @_CGet.Optional.Core.ba7a4025e0666ad5(ptr %.loc18_33.1.temp), !dbg !8
// CHECK:STDOUT:   %F.call = call i1 @_CF.Main(), !dbg !9
// CHECK:STDOUT:   br i1 %F.call, label %if.then.loc19, label %if.else.loc19, !dbg !10
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc19:                                    ; preds = %for.body
// CHECK:STDOUT:   br label %for.done, !dbg !11
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc19:                                    ; preds = %for.body
// CHECK:STDOUT:   %G.call = call i1 @_CG.Main(), !dbg !12
// CHECK:STDOUT:   br i1 %G.call, label %if.then.loc20, label %if.else.loc20, !dbg !13
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc20:                                    ; preds = %if.else.loc19
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.e76e36aefa0c6287(ptr %.loc18_33.1.temp), !dbg !8
// CHECK:STDOUT:   br label %for.next, !dbg !14
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc20:                                    ; preds = %if.else.loc19
// CHECK:STDOUT:   call void @_CH.Main(), !dbg !15
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.e76e36aefa0c6287(ptr %.loc18_33.1.temp), !dbg !8
// CHECK:STDOUT:   br label %for.next, !dbg !16
// CHECK:STDOUT:
// CHECK:STDOUT: for.done:                                         ; preds = %if.then.loc19, %for.next
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.e76e36aefa0c6287(ptr %.loc18_33.1.temp), !dbg !8
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48(ptr %var), !dbg !8
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.5a42872629477478(ptr %.loc18_32.1.temp), !dbg !7
// CHECK:STDOUT:   ret void, !dbg !17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CRange.Core(ptr sret({ i32, i32 }), i32)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_CSubobjectDestroy.6f4dee545ed23f91:core.Destroy.Core"(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.70db2d92ffddbef7:core.Destroy.Core"(ptr %self) #0 !dbg !18 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.18fc07ffd08329ec:core.Destroy.Core"(ptr %self) #0 !dbg !26 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.94d33dc7db8d438e:core.Destroy.Core"(ptr %self) #0 !dbg !33 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.ff0d3d801bd6a6ee:core.Destroy.Core"(ptr %self) #0 !dbg !37 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.ba5b124063947f4f:core.Destroy.Core"(ptr %self) #0 !dbg !41 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.243175292981777b:core.Destroy.Core"(ptr %self) #0 !dbg !45 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.593df19eb3c40d5e:core.Destroy.Core"(ptr %self) #0 !dbg !49 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !52
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.58d03c615b8abca5:core.Destroy.Core"(ptr %self) #0 !dbg !53 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !56
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CNewCursor.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8"(ptr %self) #0 !dbg !57 {
// CHECK:STDOUT:   %start = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 0, !dbg !62
// CHECK:STDOUT:   %1 = load i32, ptr %start, align 4, !dbg !62
// CHECK:STDOUT:   ret i32 %1, !dbg !63
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNext.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8"(ptr sret(<{ i32, i1 }>) %return, ptr %self, ptr %cursor) #0 !dbg !64 {
// CHECK:STDOUT:   %1 = alloca i32, align 4, !dbg !70
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %1), !dbg !70
// CHECK:STDOUT:   %2 = load i32, ptr %cursor, align 4, !dbg !71
// CHECK:STDOUT:   store i32 %2, ptr %1, align 4, !dbg !70
// CHECK:STDOUT:   %end = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 1, !dbg !72
// CHECK:STDOUT:   %3 = load i32, ptr %end, align 4, !dbg !72
// CHECK:STDOUT:   %4 = load i32, ptr %1, align 4, !dbg !73
// CHECK:STDOUT:   %5 = icmp slt i32 %4, %3, !dbg !73
// CHECK:STDOUT:   br i1 %5, label %6, label %8, !dbg !74
// CHECK:STDOUT:
// CHECK:STDOUT: 6:                                                ; preds = %0
// CHECK:STDOUT:   call void @"_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8"(ptr %cursor), !dbg !75
// CHECK:STDOUT:   %7 = load i32, ptr %1, align 4, !dbg !76
// CHECK:STDOUT:   call void @_CSome.Optional.Core.ba7a4025e0666ad5(ptr %return, i32 %7), !dbg !77
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48(ptr %1), !dbg !70
// CHECK:STDOUT:   ret void, !dbg !78
// CHECK:STDOUT:
// CHECK:STDOUT: 8:                                                ; preds = %0
// CHECK:STDOUT:   call void @_CNone.Optional.Core.ba7a4025e0666ad5(ptr %return), !dbg !79
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48(ptr %1), !dbg !70
// CHECK:STDOUT:   ret void, !dbg !80
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CHasValue.Optional.Core.ba7a4025e0666ad5(ptr %self) #0 !dbg !81 {
// CHECK:STDOUT:   %1 = call i1 @"_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.46b5dffdde151aec"(ptr %self), !dbg !87
// CHECK:STDOUT:   ret i1 %1, !dbg !88
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CGet.Optional.Core.ba7a4025e0666ad5(ptr %self) #0 !dbg !89 {
// CHECK:STDOUT:   %1 = call i32 @"_CGet.5d6f3f3a4b44a8f1:OptionalStorage.Core.46b5dffdde151aec"(ptr %self), !dbg !92
// CHECK:STDOUT:   ret i32 %1, !dbg !93
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.e76e36aefa0c6287(ptr %self) #0 !dbg !94 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.243175292981777b:core.Destroy.Core"(ptr %self), !dbg !98
// CHECK:STDOUT:   ret void, !dbg !99
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48(ptr %self) #0 !dbg !100 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.70db2d92ffddbef7:core.Destroy.Core"(ptr %self), !dbg !103
// CHECK:STDOUT:   ret void, !dbg !104
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.5a42872629477478(ptr %self) #0 !dbg !105 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.58d03c615b8abca5:core.Destroy.Core"(ptr %self), !dbg !108
// CHECK:STDOUT:   ret void, !dbg !109
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CInclusiveRange.Core(ptr sret({ i32, i32 }), i32, i32)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8"(ptr %self) #0 !dbg !110 {
// CHECK:STDOUT:   call void @"_COp:thunk:AddAssignWith.39e2e78a3c2e036d.Core:Int.9452f4c51951679b.Core:AddAssignWith.bcc2a64c00f3554f.Core.83481f1872d67895"(ptr %self, i32 1), !dbg !114
// CHECK:STDOUT:   ret void, !dbg !115
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSome.Optional.Core.ba7a4025e0666ad5(ptr sret(<{ i32, i1 }>) %return, i32 %value) #0 !dbg !116 {
// CHECK:STDOUT:   call void @"_CSome.5d6f3f3a4b44a8f1:OptionalStorage.Core.46b5dffdde151aec"(ptr %return, i32 %value), !dbg !121
// CHECK:STDOUT:   ret void, !dbg !122
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CNone.Optional.Core.ba7a4025e0666ad5(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !123 {
// CHECK:STDOUT:   call void @"_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.46b5dffdde151aec"(ptr %return), !dbg !126
// CHECK:STDOUT:   ret void, !dbg !127
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @"_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.46b5dffdde151aec"(ptr %value) #0 !dbg !128 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %value, i32 0, i32 1, !dbg !131
// CHECK:STDOUT:   %1 = load i8, ptr %has_value, align 1, !dbg !131
// CHECK:STDOUT:   %2 = trunc i8 %1 to i1, !dbg !131
// CHECK:STDOUT:   ret i1 %2, !dbg !132
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CGet.5d6f3f3a4b44a8f1:OptionalStorage.Core.46b5dffdde151aec"(ptr %value) #0 !dbg !133 {
// CHECK:STDOUT:   %value1 = getelementptr inbounds nuw <{ i32, i1 }>, ptr %value, i32 0, i32 0, !dbg !136
// CHECK:STDOUT:   %1 = load i32, ptr %value1, align 4, !dbg !136
// CHECK:STDOUT:   ret i32 %1, !dbg !137
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp:thunk:AddAssignWith.39e2e78a3c2e036d.Core:Int.9452f4c51951679b.Core:AddAssignWith.bcc2a64c00f3554f.Core.83481f1872d67895"(ptr %self, i32 %other) #2 !dbg !138 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.bb469f8924cce044"(i32 %other), !dbg !144
// CHECK:STDOUT:   %2 = load i32, ptr %self, align 4, !dbg !145
// CHECK:STDOUT:   %3 = add i32 %2, %1, !dbg !145
// CHECK:STDOUT:   store i32 %3, ptr %self, align 4, !dbg !145
// CHECK:STDOUT:   ret void, !dbg !145
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.5d6f3f3a4b44a8f1:OptionalStorage.Core.46b5dffdde151aec"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !146 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 0, !dbg !149
// CHECK:STDOUT:   store i32 %self, ptr %value, align 4, !dbg !149
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !150
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !150
// CHECK:STDOUT:   ret void, !dbg !151
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.46b5dffdde151aec"(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !152 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !153
// CHECK:STDOUT:   store i8 0, ptr %has_value, align 1, !dbg !153
// CHECK:STDOUT:   ret void, !dbg !154
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.bb469f8924cce044"(i32 %self) #0 !dbg !155 {
// CHECK:STDOUT:   %1 = call i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self), !dbg !160
// CHECK:STDOUT:   ret i32 %1, !dbg !161
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self) #0 !dbg !162 {
// CHECK:STDOUT:   ret i32 %self, !dbg !165
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 3, 2, 1 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Destroy.Core.e76e36aefa0c6287, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48, { 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #2 = { alwaysinline nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "break_continue.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "For", linkageName: "_CFor.Main", scope: null, file: !1, line: 17, type: !5, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DILocation(line: 18, column: 18, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 18, column: 7, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 19, column: 9, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 19, column: 8, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 19, column: 16, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 20, column: 9, scope: !4)
// CHECK:STDOUT: !13 = !DILocation(line: 20, column: 8, scope: !4)
// CHECK:STDOUT: !14 = !DILocation(line: 20, column: 16, scope: !4)
// CHECK:STDOUT: !15 = !DILocation(line: 21, column: 5, scope: !4)
// CHECK:STDOUT: !16 = !DILocation(line: 18, column: 3, scope: !4)
// CHECK:STDOUT: !17 = !DILocation(line: 17, column: 1, scope: !4)
// CHECK:STDOUT: !18 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.70db2d92ffddbef7:core.Destroy.Core", scope: null, file: !19, line: 18, type: !20, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !23)
// CHECK:STDOUT: !19 = !DIFile(filename: "{{.*}}/prelude/range.carbon", directory: "")
// CHECK:STDOUT: !20 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !21 = !{null, !22}
// CHECK:STDOUT: !22 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !23 = !{!24}
// CHECK:STDOUT: !24 = !DILocalVariable(arg: 1, scope: !18, type: !22)
// CHECK:STDOUT: !25 = !DILocation(line: 18, column: 39, scope: !18)
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.18fc07ffd08329ec:core.Destroy.Core", scope: null, file: !1, line: 18, type: !27, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !30)
// CHECK:STDOUT: !27 = !DISubroutineType(types: !28)
// CHECK:STDOUT: !28 = !{null, !29}
// CHECK:STDOUT: !29 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !30 = !{!31}
// CHECK:STDOUT: !31 = !DILocalVariable(arg: 1, scope: !26, type: !29)
// CHECK:STDOUT: !32 = !DILocation(line: 18, column: 7, scope: !26)
// CHECK:STDOUT: !33 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.94d33dc7db8d438e:core.Destroy.Core", scope: null, file: !1, line: 18, type: !20, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !34)
// CHECK:STDOUT: !34 = !{!35}
// CHECK:STDOUT: !35 = !DILocalVariable(arg: 1, scope: !33, type: !22)
// CHECK:STDOUT: !36 = !DILocation(line: 18, column: 7, scope: !33)
// CHECK:STDOUT: !37 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.ff0d3d801bd6a6ee:core.Destroy.Core", scope: null, file: !1, line: 18, type: !27, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !38)
// CHECK:STDOUT: !38 = !{!39}
// CHECK:STDOUT: !39 = !DILocalVariable(arg: 1, scope: !37, type: !29)
// CHECK:STDOUT: !40 = !DILocation(line: 18, column: 7, scope: !37)
// CHECK:STDOUT: !41 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.ba5b124063947f4f:core.Destroy.Core", scope: null, file: !1, line: 18, type: !27, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !42)
// CHECK:STDOUT: !42 = !{!43}
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 1, scope: !41, type: !29)
// CHECK:STDOUT: !44 = !DILocation(line: 18, column: 7, scope: !41)
// CHECK:STDOUT: !45 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.243175292981777b:core.Destroy.Core", scope: null, file: !1, line: 18, type: !27, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !46)
// CHECK:STDOUT: !46 = !{!47}
// CHECK:STDOUT: !47 = !DILocalVariable(arg: 1, scope: !45, type: !29)
// CHECK:STDOUT: !48 = !DILocation(line: 18, column: 7, scope: !45)
// CHECK:STDOUT: !49 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.593df19eb3c40d5e:core.Destroy.Core", scope: null, file: !1, line: 18, type: !27, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !50)
// CHECK:STDOUT: !50 = !{!51}
// CHECK:STDOUT: !51 = !DILocalVariable(arg: 1, scope: !49, type: !29)
// CHECK:STDOUT: !52 = !DILocation(line: 18, column: 18, scope: !49)
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.58d03c615b8abca5:core.Destroy.Core", scope: null, file: !1, line: 18, type: !27, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !54)
// CHECK:STDOUT: !54 = !{!55}
// CHECK:STDOUT: !55 = !DILocalVariable(arg: 1, scope: !53, type: !29)
// CHECK:STDOUT: !56 = !DILocation(line: 18, column: 18, scope: !53)
// CHECK:STDOUT: !57 = distinct !DISubprogram(name: "NewCursor", linkageName: "_CNewCursor.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8", scope: null, file: !19, line: 19, type: !58, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !60)
// CHECK:STDOUT: !58 = !DISubroutineType(types: !59)
// CHECK:STDOUT: !59 = !{!22, !29}
// CHECK:STDOUT: !60 = !{!61}
// CHECK:STDOUT: !61 = !DILocalVariable(arg: 1, scope: !57, type: !29)
// CHECK:STDOUT: !62 = !DILocation(line: 19, column: 43, scope: !57)
// CHECK:STDOUT: !63 = !DILocation(line: 19, column: 36, scope: !57)
// CHECK:STDOUT: !64 = distinct !DISubprogram(name: "Next", linkageName: "_CNext.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8", scope: null, file: !19, line: 20, type: !65, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !67)
// CHECK:STDOUT: !65 = !DISubroutineType(types: !66)
// CHECK:STDOUT: !66 = !{!29, !29, !29}
// CHECK:STDOUT: !67 = !{!68, !69}
// CHECK:STDOUT: !68 = !DILocalVariable(arg: 1, scope: !64, type: !29)
// CHECK:STDOUT: !69 = !DILocalVariable(arg: 2, scope: !64, type: !29)
// CHECK:STDOUT: !70 = !DILocation(line: 21, column: 7, scope: !64)
// CHECK:STDOUT: !71 = !DILocation(line: 21, column: 27, scope: !64)
// CHECK:STDOUT: !72 = !DILocation(line: 22, column: 19, scope: !64)
// CHECK:STDOUT: !73 = !DILocation(line: 22, column: 11, scope: !64)
// CHECK:STDOUT: !74 = !DILocation(line: 22, column: 10, scope: !64)
// CHECK:STDOUT: !75 = !DILocation(line: 23, column: 9, scope: !64)
// CHECK:STDOUT: !76 = !DILocation(line: 24, column: 38, scope: !64)
// CHECK:STDOUT: !77 = !DILocation(line: 24, column: 16, scope: !64)
// CHECK:STDOUT: !78 = !DILocation(line: 24, column: 9, scope: !64)
// CHECK:STDOUT: !79 = !DILocation(line: 26, column: 16, scope: !64)
// CHECK:STDOUT: !80 = !DILocation(line: 26, column: 9, scope: !64)
// CHECK:STDOUT: !81 = distinct !DISubprogram(name: "HasValue", linkageName: "_CHasValue.Optional.Core.ba7a4025e0666ad5", scope: null, file: !82, line: 36, type: !83, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !85)
// CHECK:STDOUT: !82 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !83 = !DISubroutineType(types: !84)
// CHECK:STDOUT: !84 = !{!29, !29}
// CHECK:STDOUT: !85 = !{!86}
// CHECK:STDOUT: !86 = !DILocalVariable(arg: 1, scope: !81, type: !29)
// CHECK:STDOUT: !87 = !DILocation(line: 37, column: 12, scope: !81)
// CHECK:STDOUT: !88 = !DILocation(line: 37, column: 5, scope: !81)
// CHECK:STDOUT: !89 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.Optional.Core.ba7a4025e0666ad5", scope: null, file: !82, line: 39, type: !58, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !90)
// CHECK:STDOUT: !90 = !{!91}
// CHECK:STDOUT: !91 = !DILocalVariable(arg: 1, scope: !89, type: !29)
// CHECK:STDOUT: !92 = !DILocation(line: 40, column: 12, scope: !89)
// CHECK:STDOUT: !93 = !DILocation(line: 40, column: 5, scope: !89)
// CHECK:STDOUT: !94 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.e76e36aefa0c6287", scope: null, file: !95, line: 26, type: !27, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !96)
// CHECK:STDOUT: !95 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !96 = !{!97}
// CHECK:STDOUT: !97 = !DILocalVariable(arg: 1, scope: !94, type: !29)
// CHECK:STDOUT: !98 = !DILocation(line: 28, column: 5, scope: !94)
// CHECK:STDOUT: !99 = !DILocation(line: 26, column: 3, scope: !94)
// CHECK:STDOUT: !100 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48", scope: null, file: !95, line: 26, type: !20, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !101)
// CHECK:STDOUT: !101 = !{!102}
// CHECK:STDOUT: !102 = !DILocalVariable(arg: 1, scope: !100, type: !22)
// CHECK:STDOUT: !103 = !DILocation(line: 28, column: 5, scope: !100)
// CHECK:STDOUT: !104 = !DILocation(line: 26, column: 3, scope: !100)
// CHECK:STDOUT: !105 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.5a42872629477478", scope: null, file: !95, line: 26, type: !27, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !106)
// CHECK:STDOUT: !106 = !{!107}
// CHECK:STDOUT: !107 = !DILocalVariable(arg: 1, scope: !105, type: !29)
// CHECK:STDOUT: !108 = !DILocation(line: 28, column: 5, scope: !105)
// CHECK:STDOUT: !109 = !DILocation(line: 26, column: 3, scope: !105)
// CHECK:STDOUT: !110 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8", scope: null, file: !111, line: 396, type: !20, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !112)
// CHECK:STDOUT: !111 = !DIFile(filename: "{{.*}}/prelude/types/int.carbon", directory: "")
// CHECK:STDOUT: !112 = !{!113}
// CHECK:STDOUT: !113 = !DILocalVariable(arg: 1, scope: !110, type: !22)
// CHECK:STDOUT: !114 = !DILocation(line: 398, column: 5, scope: !110)
// CHECK:STDOUT: !115 = !DILocation(line: 396, column: 3, scope: !110)
// CHECK:STDOUT: !116 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.ba7a4025e0666ad5", scope: null, file: !82, line: 33, type: !117, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !119)
// CHECK:STDOUT: !117 = !DISubroutineType(types: !118)
// CHECK:STDOUT: !118 = !{!29, !22}
// CHECK:STDOUT: !119 = !{!120}
// CHECK:STDOUT: !120 = !DILocalVariable(arg: 1, scope: !116, type: !22)
// CHECK:STDOUT: !121 = !DILocation(line: 34, column: 12, scope: !116)
// CHECK:STDOUT: !122 = !DILocation(line: 34, column: 5, scope: !116)
// CHECK:STDOUT: !123 = distinct !DISubprogram(name: "None", linkageName: "_CNone.Optional.Core.ba7a4025e0666ad5", scope: null, file: !82, line: 30, type: !124, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !124 = !DISubroutineType(types: !125)
// CHECK:STDOUT: !125 = !{!29}
// CHECK:STDOUT: !126 = !DILocation(line: 31, column: 12, scope: !123)
// CHECK:STDOUT: !127 = !DILocation(line: 31, column: 5, scope: !123)
// CHECK:STDOUT: !128 = distinct !DISubprogram(name: "Has", linkageName: "_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.46b5dffdde151aec", scope: null, file: !82, line: 141, type: !83, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !129)
// CHECK:STDOUT: !129 = !{!130}
// CHECK:STDOUT: !130 = !DILocalVariable(arg: 1, scope: !128, type: !29)
// CHECK:STDOUT: !131 = !DILocation(line: 142, column: 12, scope: !128)
// CHECK:STDOUT: !132 = !DILocation(line: 142, column: 5, scope: !128)
// CHECK:STDOUT: !133 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.5d6f3f3a4b44a8f1:OptionalStorage.Core.46b5dffdde151aec", scope: null, file: !82, line: 144, type: !58, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !134)
// CHECK:STDOUT: !134 = !{!135}
// CHECK:STDOUT: !135 = !DILocalVariable(arg: 1, scope: !133, type: !29)
// CHECK:STDOUT: !136 = !DILocation(line: 145, column: 12, scope: !133)
// CHECK:STDOUT: !137 = !DILocation(line: 145, column: 5, scope: !133)
// CHECK:STDOUT: !138 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:AddAssignWith.39e2e78a3c2e036d.Core:Int.9452f4c51951679b.Core:AddAssignWith.bcc2a64c00f3554f.Core.83481f1872d67895", scope: null, file: !111, line: 331, type: !139, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !141)
// CHECK:STDOUT: !139 = !DISubroutineType(types: !140)
// CHECK:STDOUT: !140 = !{null, !22, !22}
// CHECK:STDOUT: !141 = !{!142, !143}
// CHECK:STDOUT: !142 = !DILocalVariable(arg: 1, scope: !138, type: !22)
// CHECK:STDOUT: !143 = !DILocalVariable(arg: 2, scope: !138, type: !22)
// CHECK:STDOUT: !144 = !DILocation(line: 0, scope: !138)
// CHECK:STDOUT: !145 = !DILocation(line: 331, column: 3, scope: !138)
// CHECK:STDOUT: !146 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.5d6f3f3a4b44a8f1:OptionalStorage.Core.46b5dffdde151aec", scope: null, file: !82, line: 131, type: !117, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !147)
// CHECK:STDOUT: !147 = !{!148}
// CHECK:STDOUT: !148 = !DILocalVariable(arg: 1, scope: !146, type: !22)
// CHECK:STDOUT: !149 = !DILocation(line: 137, column: 5, scope: !146)
// CHECK:STDOUT: !150 = !DILocation(line: 138, column: 5, scope: !146)
// CHECK:STDOUT: !151 = !DILocation(line: 139, column: 5, scope: !146)
// CHECK:STDOUT: !152 = distinct !DISubprogram(name: "None", linkageName: "_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.46b5dffdde151aec", scope: null, file: !82, line: 125, type: !124, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !153 = !DILocation(line: 128, column: 5, scope: !152)
// CHECK:STDOUT: !154 = !DILocation(line: 129, column: 5, scope: !152)
// CHECK:STDOUT: !155 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.bb469f8924cce044", scope: null, file: !111, line: 64, type: !156, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !158)
// CHECK:STDOUT: !156 = !DISubroutineType(types: !157)
// CHECK:STDOUT: !157 = !{!22, !22}
// CHECK:STDOUT: !158 = !{!159}
// CHECK:STDOUT: !159 = !DILocalVariable(arg: 1, scope: !155, type: !22)
// CHECK:STDOUT: !160 = !DILocation(line: 66, column: 17, scope: !155)
// CHECK:STDOUT: !161 = !DILocation(line: 66, column: 5, scope: !155)
// CHECK:STDOUT: !162 = distinct !DISubprogram(name: "AsInt", linkageName: "_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8", scope: null, file: !111, line: 59, type: !156, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !163)
// CHECK:STDOUT: !163 = !{!164}
// CHECK:STDOUT: !164 = !DILocalVariable(arg: 1, scope: !162, type: !22)
// CHECK:STDOUT: !165 = !DILocation(line: 59, column: 28, scope: !162)
// CHECK:STDOUT:
