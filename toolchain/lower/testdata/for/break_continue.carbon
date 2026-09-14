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
// CHECK:STDOUT: define void @_CFor.Main() #0 !dbg !18 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc18_32.1.temp = alloca { i32, i32 }, align 4, !dbg !20
// CHECK:STDOUT:   %var = alloca i32, align 4, !dbg !21
// CHECK:STDOUT:   %.loc18_33.1.temp = alloca <{ i32, i1 }>, align 4, !dbg !21
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc18_32.1.temp), !dbg !20
// CHECK:STDOUT:   call void @_CRange.Core(ptr %.loc18_32.1.temp, i32 100), !dbg !20
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.NewCursor.call = call i32 @"_CNewCursor.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8"(ptr %.loc18_32.1.temp), !dbg !21
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %var), !dbg !21
// CHECK:STDOUT:   store i32 %IntRange.as.Iterate.impl.NewCursor.call, ptr %var, align 4, !dbg !21
// CHECK:STDOUT:   br label %for.next, !dbg !21
// CHECK:STDOUT:
// CHECK:STDOUT: for.next:                                         ; preds = %if.else.loc20, %if.then.loc20, %entry
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc18_33.1.temp), !dbg !21
// CHECK:STDOUT:   call void @"_CNext.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8"(ptr %.loc18_33.1.temp, ptr %.loc18_32.1.temp, ptr %var), !dbg !21
// CHECK:STDOUT:   %Optional.HasValue.call = call i1 @_CHasValue.Optional.Core.a5214d4ceffb522b(ptr %.loc18_33.1.temp), !dbg !21
// CHECK:STDOUT:   br i1 %Optional.HasValue.call, label %for.body, label %for.done, !dbg !21
// CHECK:STDOUT:
// CHECK:STDOUT: for.body:                                         ; preds = %for.next
// CHECK:STDOUT:   %Optional.Get.call = call i32 @_CGet.Optional.Core.a5214d4ceffb522b(ptr %.loc18_33.1.temp), !dbg !21
// CHECK:STDOUT:   %F.call = call i1 @_CF.Main(), !dbg !28
// CHECK:STDOUT:   br i1 %F.call, label %if.then.loc19, label %if.else.loc19, !dbg !29
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc19:                                    ; preds = %for.body
// CHECK:STDOUT:   br label %for.done, !dbg !30
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc19:                                    ; preds = %for.body
// CHECK:STDOUT:   %G.call = call i1 @_CG.Main(), !dbg !31
// CHECK:STDOUT:   br i1 %G.call, label %if.then.loc20, label %if.else.loc20, !dbg !32
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc20:                                    ; preds = %if.else.loc19
// CHECK:STDOUT:   call void @"_COp.79101dbcae945290:core.Destroy.Core"(ptr %.loc18_33.1.temp), !dbg !21
// CHECK:STDOUT:   br label %for.next, !dbg !33
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc20:                                    ; preds = %if.else.loc19
// CHECK:STDOUT:   call void @_CH.Main(), !dbg !34
// CHECK:STDOUT:   call void @"_COp.79101dbcae945290:core.Destroy.Core"(ptr %.loc18_33.1.temp), !dbg !21
// CHECK:STDOUT:   br label %for.next, !dbg !35
// CHECK:STDOUT:
// CHECK:STDOUT: for.done:                                         ; preds = %if.then.loc19, %for.next
// CHECK:STDOUT:   call void @"_COp.79101dbcae945290:core.Destroy.Core"(ptr %.loc18_33.1.temp), !dbg !21
// CHECK:STDOUT:   call void @"_COp.70db2d92ffddbef7:core.Destroy.Core"(ptr %var), !dbg !21
// CHECK:STDOUT:   call void @"_COp.58d03c615b8abca5:core.Destroy.Core"(ptr %.loc18_32.1.temp), !dbg !20
// CHECK:STDOUT:   ret void, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CRange.Core(ptr sret({ i32, i32 }), i32)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.6f4dee545ed23f91:core.Destroy.Core"(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.70db2d92ffddbef7:core.Destroy.Core"(ptr %self) #0 !dbg !38 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.18fc07ffd08329ec:core.Destroy.Core"(ptr %self) #0 !dbg !42 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.6a45900cb03df98d:core.Destroy.Core"(ptr %self) #0 !dbg !46 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.4622d8af498f8150:core.Destroy.Core"(ptr %self) #0 !dbg !50 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !52
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.5432973a0cb6f903:core.Destroy.Core"(ptr %self) #0 !dbg !54 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !56
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.79101dbcae945290:core.Destroy.Core"(ptr %self) #0 !dbg !58 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !60
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.593df19eb3c40d5e:core.Destroy.Core"(ptr %self) #0 !dbg !62 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !64
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.58d03c615b8abca5:core.Destroy.Core"(ptr %self) #0 !dbg !66 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !68
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CNewCursor.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8"(ptr %self) #0 !dbg !72 {
// CHECK:STDOUT:   %start = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 0, !dbg !77
// CHECK:STDOUT:   %1 = load i32, ptr %start, align 4, !dbg !77
// CHECK:STDOUT:   ret i32 %1, !dbg !78
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNext.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8"(ptr sret(<{ i32, i1 }>) %return, ptr %self, ptr %cursor) #0 !dbg !80 {
// CHECK:STDOUT:   %1 = alloca i32, align 4, !dbg !87
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %1), !dbg !87
// CHECK:STDOUT:   %2 = load i32, ptr %cursor, align 4, !dbg !89
// CHECK:STDOUT:   store i32 %2, ptr %1, align 4, !dbg !87
// CHECK:STDOUT:   %end = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 1, !dbg !92
// CHECK:STDOUT:   %3 = load i32, ptr %end, align 4, !dbg !92
// CHECK:STDOUT:   %4 = load i32, ptr %1, align 4, !dbg !91
// CHECK:STDOUT:   %5 = icmp slt i32 %4, %3, !dbg !91
// CHECK:STDOUT:   br i1 %5, label %6, label %8, !dbg !93
// CHECK:STDOUT:
// CHECK:STDOUT: 6:                                                ; preds = %0
// CHECK:STDOUT:   call void @"_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8"(ptr %cursor), !dbg !96
// CHECK:STDOUT:   %7 = load i32, ptr %1, align 4, !dbg !97
// CHECK:STDOUT:   call void @_CSome.Optional.Core.a5214d4ceffb522b(ptr %return, i32 %7), !dbg !98
// CHECK:STDOUT:   call void @"_COp.70db2d92ffddbef7:core.Destroy.Core"(ptr %1), !dbg !87
// CHECK:STDOUT:   ret void, !dbg !99
// CHECK:STDOUT:
// CHECK:STDOUT: 8:                                                ; preds = %0
// CHECK:STDOUT:   call void @_CNone.Optional.Core.a5214d4ceffb522b(ptr %return), !dbg !100
// CHECK:STDOUT:   call void @"_COp.70db2d92ffddbef7:core.Destroy.Core"(ptr %1), !dbg !87
// CHECK:STDOUT:   ret void, !dbg !101
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CHasValue.Optional.Core.a5214d4ceffb522b(ptr %self) #0 !dbg !104 {
// CHECK:STDOUT:   %1 = call i1 @"_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.553a1f59d1dd8381"(ptr %self), !dbg !110
// CHECK:STDOUT:   ret i1 %1, !dbg !111
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CGet.Optional.Core.a5214d4ceffb522b(ptr %self) #0 !dbg !113 {
// CHECK:STDOUT:   %1 = call i32 @"_CGet.5d6f3f3a4b44a8f1:OptionalStorage.Core.553a1f59d1dd8381"(ptr %self), !dbg !119
// CHECK:STDOUT:   ret i32 %1, !dbg !120
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CInclusiveRange.Core(ptr sret({ i32, i32 }), i32, i32)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8"(ptr %self) #0 !dbg !123 {
// CHECK:STDOUT:   call void @"_COp:thunk:AddAssignWith.39e2e78a3c2e036d.Core:Int.9452f4c51951679b.Core:AddAssignWith.bcc2a64c00f3554f.Core.83481f1872d67895"(ptr %self, i32 1), !dbg !126
// CHECK:STDOUT:   ret void, !dbg !129
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSome.Optional.Core.a5214d4ceffb522b(ptr sret(<{ i32, i1 }>) %return, i32 %value) #0 !dbg !131 {
// CHECK:STDOUT:   call void @"_CSome.5d6f3f3a4b44a8f1:OptionalStorage.Core.553a1f59d1dd8381"(ptr %return, i32 %value), !dbg !136
// CHECK:STDOUT:   ret void, !dbg !137
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CNone.Optional.Core.a5214d4ceffb522b(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !139 {
// CHECK:STDOUT:   call void @"_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.553a1f59d1dd8381"(ptr %return), !dbg !142
// CHECK:STDOUT:   ret void, !dbg !143
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @"_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.553a1f59d1dd8381"(ptr %value) #0 !dbg !144 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %value, i32 0, i32 1, !dbg !149
// CHECK:STDOUT:   %1 = load i8, ptr %has_value, align 1, !dbg !149
// CHECK:STDOUT:   %2 = trunc i8 %1 to i1, !dbg !149
// CHECK:STDOUT:   ret i1 %2, !dbg !150
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CGet.5d6f3f3a4b44a8f1:OptionalStorage.Core.553a1f59d1dd8381"(ptr %value) #0 !dbg !152 {
// CHECK:STDOUT:   %value1 = getelementptr inbounds nuw <{ i32, i1 }>, ptr %value, i32 0, i32 0, !dbg !157
// CHECK:STDOUT:   %1 = load i32, ptr %value1, align 4, !dbg !157
// CHECK:STDOUT:   ret i32 %1, !dbg !158
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp:thunk:AddAssignWith.39e2e78a3c2e036d.Core:Int.9452f4c51951679b.Core:AddAssignWith.bcc2a64c00f3554f.Core.83481f1872d67895"(ptr %self, i32 %other) #2 !dbg !160 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.bb469f8924cce044"(i32 %other), !dbg !163
// CHECK:STDOUT:   %2 = load i32, ptr %self, align 4, !dbg !164
// CHECK:STDOUT:   %3 = add i32 %2, %1, !dbg !164
// CHECK:STDOUT:   store i32 %3, ptr %self, align 4, !dbg !164
// CHECK:STDOUT:   ret void, !dbg !164
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.5d6f3f3a4b44a8f1:OptionalStorage.Core.553a1f59d1dd8381"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !168 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 0, !dbg !177
// CHECK:STDOUT:   store i32 %self, ptr %value, align 4, !dbg !177
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !179
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !179
// CHECK:STDOUT:   ret void, !dbg !180
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.553a1f59d1dd8381"(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !182 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !189
// CHECK:STDOUT:   store i8 0, ptr %has_value, align 1, !dbg !189
// CHECK:STDOUT:   ret void, !dbg !190
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.bb469f8924cce044"(i32 %self) #0 !dbg !191 {
// CHECK:STDOUT:   %1 = call i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self), !dbg !196
// CHECK:STDOUT:   ret i32 %1, !dbg !198
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self) #0 !dbg !200 {
// CHECK:STDOUT:   ret i32 %self, !dbg !206
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @"_COp.79101dbcae945290:core.Destroy.Core", { 2, 1, 0 }
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
// CHECK:STDOUT: !14 = !{null, !6}
// CHECK:STDOUT: !15 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !16 = !{null, !7}
// CHECK:STDOUT: !17 = !DISubroutineType(types: !16)
// CHECK:STDOUT: !18 = distinct !DISubprogram(name: "For", linkageName: "_CFor.Main", scope: null, file: !4, line: 17, type: !11, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !20 = !DILocation(line: 18, column: 18, scope: !18)
// CHECK:STDOUT: !21 = !DILocation(line: 18, column: 7, scope: !18)
// CHECK:STDOUT: !22 = !{!6, !7}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !24 = !{!7, !7, !7}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = !{!7, !7}
// CHECK:STDOUT: !27 = !DISubroutineType(types: !26)
// CHECK:STDOUT: !28 = !DILocation(line: 19, column: 9, scope: !18)
// CHECK:STDOUT: !29 = !DILocation(line: 19, column: 8, scope: !18)
// CHECK:STDOUT: !30 = !DILocation(line: 19, column: 16, scope: !18)
// CHECK:STDOUT: !31 = !DILocation(line: 20, column: 9, scope: !18)
// CHECK:STDOUT: !32 = !DILocation(line: 20, column: 8, scope: !18)
// CHECK:STDOUT: !33 = !DILocation(line: 20, column: 16, scope: !18)
// CHECK:STDOUT: !34 = !DILocation(line: 21, column: 5, scope: !18)
// CHECK:STDOUT: !35 = !DILocation(line: 18, column: 3, scope: !18)
// CHECK:STDOUT: !36 = !DILocation(line: 17, column: 1, scope: !18)
// CHECK:STDOUT: !37 = !DIFile(filename: "{{.*}}/prelude/range.carbon", directory: "")
// CHECK:STDOUT: !38 = distinct !DISubprogram(name: "Op", linkageName: "_COp.70db2d92ffddbef7:core.Destroy.Core", scope: null, file: !37, line: 18, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !41)
// CHECK:STDOUT: !39 = !DILocalVariable(arg: 1, scope: !38, type: !6)
// CHECK:STDOUT: !40 = !DILocation(line: 18, column: 39, scope: !38)
// CHECK:STDOUT: !41 = !{!39}
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "Op", linkageName: "_COp.18fc07ffd08329ec:core.Destroy.Core", scope: null, file: !4, line: 18, type: !17, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !45)
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 1, scope: !42, type: !7)
// CHECK:STDOUT: !44 = !DILocation(line: 18, column: 7, scope: !42)
// CHECK:STDOUT: !45 = !{!43}
// CHECK:STDOUT: !46 = distinct !DISubprogram(name: "Op", linkageName: "_COp.6a45900cb03df98d:core.Destroy.Core", scope: null, file: !4, line: 18, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !49)
// CHECK:STDOUT: !47 = !DILocalVariable(arg: 1, scope: !46, type: !6)
// CHECK:STDOUT: !48 = !DILocation(line: 18, column: 7, scope: !46)
// CHECK:STDOUT: !49 = !{!47}
// CHECK:STDOUT: !50 = distinct !DISubprogram(name: "Op", linkageName: "_COp.4622d8af498f8150:core.Destroy.Core", scope: null, file: !4, line: 18, type: !17, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !53)
// CHECK:STDOUT: !51 = !DILocalVariable(arg: 1, scope: !50, type: !7)
// CHECK:STDOUT: !52 = !DILocation(line: 18, column: 7, scope: !50)
// CHECK:STDOUT: !53 = !{!51}
// CHECK:STDOUT: !54 = distinct !DISubprogram(name: "Op", linkageName: "_COp.5432973a0cb6f903:core.Destroy.Core", scope: null, file: !4, line: 18, type: !17, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !57)
// CHECK:STDOUT: !55 = !DILocalVariable(arg: 1, scope: !54, type: !7)
// CHECK:STDOUT: !56 = !DILocation(line: 18, column: 7, scope: !54)
// CHECK:STDOUT: !57 = !{!55}
// CHECK:STDOUT: !58 = distinct !DISubprogram(name: "Op", linkageName: "_COp.79101dbcae945290:core.Destroy.Core", scope: null, file: !4, line: 18, type: !17, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !61)
// CHECK:STDOUT: !59 = !DILocalVariable(arg: 1, scope: !58, type: !7)
// CHECK:STDOUT: !60 = !DILocation(line: 18, column: 7, scope: !58)
// CHECK:STDOUT: !61 = !{!59}
// CHECK:STDOUT: !62 = distinct !DISubprogram(name: "Op", linkageName: "_COp.593df19eb3c40d5e:core.Destroy.Core", scope: null, file: !4, line: 18, type: !17, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !65)
// CHECK:STDOUT: !63 = !DILocalVariable(arg: 1, scope: !62, type: !7)
// CHECK:STDOUT: !64 = !DILocation(line: 18, column: 18, scope: !62)
// CHECK:STDOUT: !65 = !{!63}
// CHECK:STDOUT: !66 = distinct !DISubprogram(name: "Op", linkageName: "_COp.58d03c615b8abca5:core.Destroy.Core", scope: null, file: !4, line: 18, type: !17, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !69)
// CHECK:STDOUT: !67 = !DILocalVariable(arg: 1, scope: !66, type: !7)
// CHECK:STDOUT: !68 = !DILocation(line: 18, column: 18, scope: !66)
// CHECK:STDOUT: !69 = !{!67}
// CHECK:STDOUT: !72 = distinct !DISubprogram(name: "NewCursor", linkageName: "_CNewCursor.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8", scope: null, file: !37, line: 19, type: !23, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !79)
// CHECK:STDOUT: !73 = !DILocalVariable(arg: 1, scope: !72, type: !7)
// CHECK:STDOUT: !77 = !DILocation(line: 19, column: 43, scope: !72)
// CHECK:STDOUT: !78 = !DILocation(line: 19, column: 36, scope: !72)
// CHECK:STDOUT: !79 = !{!73}
// CHECK:STDOUT: !80 = distinct !DISubprogram(name: "Next", linkageName: "_CNext.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8", scope: null, file: !37, line: 20, type: !25, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !102)
// CHECK:STDOUT: !81 = !DILocalVariable(arg: 1, scope: !80, type: !7)
// CHECK:STDOUT: !82 = !DILocalVariable(arg: 2, scope: !80, type: !7)
// CHECK:STDOUT: !87 = !DILocation(line: 21, column: 7, scope: !80)
// CHECK:STDOUT: !89 = !DILocation(line: 21, column: 27, scope: !80)
// CHECK:STDOUT: !91 = !DILocation(line: 22, column: 11, scope: !80)
// CHECK:STDOUT: !92 = !DILocation(line: 22, column: 19, scope: !80)
// CHECK:STDOUT: !93 = !DILocation(line: 22, column: 10, scope: !80)
// CHECK:STDOUT: !96 = !DILocation(line: 23, column: 9, scope: !80)
// CHECK:STDOUT: !97 = !DILocation(line: 24, column: 38, scope: !80)
// CHECK:STDOUT: !98 = !DILocation(line: 24, column: 16, scope: !80)
// CHECK:STDOUT: !99 = !DILocation(line: 24, column: 9, scope: !80)
// CHECK:STDOUT: !100 = !DILocation(line: 26, column: 16, scope: !80)
// CHECK:STDOUT: !101 = !DILocation(line: 26, column: 9, scope: !80)
// CHECK:STDOUT: !102 = !{!81, !82}
// CHECK:STDOUT: !103 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !104 = distinct !DISubprogram(name: "HasValue", linkageName: "_CHasValue.Optional.Core.a5214d4ceffb522b", scope: null, file: !103, line: 36, type: !27, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !112)
// CHECK:STDOUT: !105 = !DILocalVariable(arg: 1, scope: !104, type: !7)
// CHECK:STDOUT: !110 = !DILocation(line: 37, column: 12, scope: !104)
// CHECK:STDOUT: !111 = !DILocation(line: 37, column: 5, scope: !104)
// CHECK:STDOUT: !112 = !{!105}
// CHECK:STDOUT: !113 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.Optional.Core.a5214d4ceffb522b", scope: null, file: !103, line: 39, type: !23, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !121)
// CHECK:STDOUT: !114 = !DILocalVariable(arg: 1, scope: !113, type: !7)
// CHECK:STDOUT: !119 = !DILocation(line: 40, column: 12, scope: !113)
// CHECK:STDOUT: !120 = !DILocation(line: 40, column: 5, scope: !113)
// CHECK:STDOUT: !121 = !{!114}
// CHECK:STDOUT: !122 = !DIFile(filename: "{{.*}}/prelude/types/int.carbon", directory: "")
// CHECK:STDOUT: !123 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8", scope: null, file: !122, line: 396, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !130)
// CHECK:STDOUT: !124 = !DILocalVariable(arg: 1, scope: !123, type: !6)
// CHECK:STDOUT: !126 = !DILocation(line: 398, column: 5, scope: !123)
// CHECK:STDOUT: !127 = !{null, !6, !6}
// CHECK:STDOUT: !128 = !DISubroutineType(types: !127)
// CHECK:STDOUT: !129 = !DILocation(line: 396, column: 3, scope: !123)
// CHECK:STDOUT: !130 = !{!124}
// CHECK:STDOUT: !131 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.a5214d4ceffb522b", scope: null, file: !103, line: 33, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !138)
// CHECK:STDOUT: !132 = !DILocalVariable(arg: 1, scope: !131, type: !6)
// CHECK:STDOUT: !136 = !DILocation(line: 34, column: 12, scope: !131)
// CHECK:STDOUT: !137 = !DILocation(line: 34, column: 5, scope: !131)
// CHECK:STDOUT: !138 = !{!132}
// CHECK:STDOUT: !139 = distinct !DISubprogram(name: "None", linkageName: "_CNone.Optional.Core.a5214d4ceffb522b", scope: null, file: !103, line: 30, type: !9, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !142 = !DILocation(line: 31, column: 12, scope: !139)
// CHECK:STDOUT: !143 = !DILocation(line: 31, column: 5, scope: !139)
// CHECK:STDOUT: !144 = distinct !DISubprogram(name: "Has", linkageName: "_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.553a1f59d1dd8381", scope: null, file: !103, line: 141, type: !27, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !151)
// CHECK:STDOUT: !145 = !DILocalVariable(arg: 1, scope: !144, type: !7)
// CHECK:STDOUT: !149 = !DILocation(line: 142, column: 12, scope: !144)
// CHECK:STDOUT: !150 = !DILocation(line: 142, column: 5, scope: !144)
// CHECK:STDOUT: !151 = !{!145}
// CHECK:STDOUT: !152 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.5d6f3f3a4b44a8f1:OptionalStorage.Core.553a1f59d1dd8381", scope: null, file: !103, line: 144, type: !23, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !159)
// CHECK:STDOUT: !153 = !DILocalVariable(arg: 1, scope: !152, type: !7)
// CHECK:STDOUT: !157 = !DILocation(line: 145, column: 12, scope: !152)
// CHECK:STDOUT: !158 = !DILocation(line: 145, column: 5, scope: !152)
// CHECK:STDOUT: !159 = !{!153}
// CHECK:STDOUT: !160 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:AddAssignWith.39e2e78a3c2e036d.Core:Int.9452f4c51951679b.Core:AddAssignWith.bcc2a64c00f3554f.Core.83481f1872d67895", scope: null, file: !122, line: 331, type: !128, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !167)
// CHECK:STDOUT: !161 = !DILocalVariable(arg: 1, scope: !160, type: !6)
// CHECK:STDOUT: !162 = !DILocalVariable(arg: 2, scope: !160, type: !6)
// CHECK:STDOUT: !163 = !DILocation(line: 0, scope: !160)
// CHECK:STDOUT: !164 = !DILocation(line: 331, column: 3, scope: !160)
// CHECK:STDOUT: !165 = !{!6, !6}
// CHECK:STDOUT: !166 = !DISubroutineType(types: !165)
// CHECK:STDOUT: !167 = !{!161, !162}
// CHECK:STDOUT: !168 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.5d6f3f3a4b44a8f1:OptionalStorage.Core.553a1f59d1dd8381", scope: null, file: !103, line: 131, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !181)
// CHECK:STDOUT: !169 = !DILocalVariable(arg: 1, scope: !168, type: !6)
// CHECK:STDOUT: !177 = !DILocation(line: 137, column: 5, scope: !168)
// CHECK:STDOUT: !179 = !DILocation(line: 138, column: 5, scope: !168)
// CHECK:STDOUT: !180 = !DILocation(line: 139, column: 5, scope: !168)
// CHECK:STDOUT: !181 = !{!169}
// CHECK:STDOUT: !182 = distinct !DISubprogram(name: "None", linkageName: "_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.553a1f59d1dd8381", scope: null, file: !103, line: 125, type: !9, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !189 = !DILocation(line: 128, column: 5, scope: !182)
// CHECK:STDOUT: !190 = !DILocation(line: 129, column: 5, scope: !182)
// CHECK:STDOUT: !191 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.bb469f8924cce044", scope: null, file: !122, line: 64, type: !166, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !199)
// CHECK:STDOUT: !192 = !DILocalVariable(arg: 1, scope: !191, type: !6)
// CHECK:STDOUT: !196 = !DILocation(line: 66, column: 17, scope: !191)
// CHECK:STDOUT: !198 = !DILocation(line: 66, column: 5, scope: !191)
// CHECK:STDOUT: !199 = !{!192}
// CHECK:STDOUT: !200 = distinct !DISubprogram(name: "AsInt", linkageName: "_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8", scope: null, file: !122, line: 59, type: !166, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !207)
// CHECK:STDOUT: !201 = !DILocalVariable(arg: 1, scope: !200, type: !6)
// CHECK:STDOUT: !206 = !DILocation(line: 59, column: 28, scope: !200)
// CHECK:STDOUT: !207 = !{!201}
// CHECK:STDOUT:
