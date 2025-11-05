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

import Core library "range";

fn F() -> bool;
fn G() -> bool;
fn H();

fn For() {
  for (n: i32 in Core.Range(100)) {
    if (F()) { break; }
    if (G()) { continue; }
    H();
  }
}

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
// CHECK:STDOUT:   %.loc20_32.1.temp = alloca { i32, i32 }, align 8, !dbg !7
// CHECK:STDOUT:   %var = alloca i32, align 4, !dbg !8
// CHECK:STDOUT:   %.loc20_33.1.temp = alloca { i32, i1 }, align 8, !dbg !8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc20_32.1.temp), !dbg !7
// CHECK:STDOUT:   call void @_CRange.Core(ptr %.loc20_32.1.temp, i32 100), !dbg !7
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.NewCursor.call = call i32 @"_CNewCursor.IntRange.Core:Iterate.Core.be1e879c1ad406d8"(ptr %.loc20_32.1.temp), !dbg !8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %var), !dbg !8
// CHECK:STDOUT:   store i32 %IntRange.as.Iterate.impl.NewCursor.call, ptr %var, align 4, !dbg !8
// CHECK:STDOUT:   br label %for.next, !dbg !8
// CHECK:STDOUT:
// CHECK:STDOUT: for.next:                                         ; preds = %if.else.loc22, %if.then.loc22, %entry
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc20_33.1.temp), !dbg !8
// CHECK:STDOUT:   call void @"_CNext.IntRange.Core:Iterate.Core.be1e879c1ad406d8"(ptr %.loc20_33.1.temp, ptr %.loc20_32.1.temp, ptr %var), !dbg !8
// CHECK:STDOUT:   %Optional.HasValue.call = call i1 @_CHasValue.Optional.Core.c1b882a73b8b9531(ptr %.loc20_33.1.temp), !dbg !8
// CHECK:STDOUT:   br i1 %Optional.HasValue.call, label %for.body, label %for.done, !dbg !8
// CHECK:STDOUT:
// CHECK:STDOUT: for.body:                                         ; preds = %for.next
// CHECK:STDOUT:   %Optional.Get.call = call i32 @_CGet.Optional.Core.c1b882a73b8b9531(ptr %.loc20_33.1.temp), !dbg !8
// CHECK:STDOUT:   %F.call = call i1 @_CF.Main(), !dbg !9
// CHECK:STDOUT:   br i1 %F.call, label %if.then.loc21, label %if.else.loc21, !dbg !10
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc21:                                    ; preds = %for.body
// CHECK:STDOUT:   br label %for.done, !dbg !11
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc21:                                    ; preds = %for.body
// CHECK:STDOUT:   %G.call = call i1 @_CG.Main(), !dbg !12
// CHECK:STDOUT:   br i1 %G.call, label %if.then.loc22, label %if.else.loc22, !dbg !13
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc22:                                    ; preds = %if.else.loc21
// CHECK:STDOUT:   br label %for.next, !dbg !14
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc22:                                    ; preds = %if.else.loc21
// CHECK:STDOUT:   call void @_CH.Main(), !dbg !15
// CHECK:STDOUT:   br label %for.next, !dbg !16
// CHECK:STDOUT:
// CHECK:STDOUT: for.done:                                         ; preds = %if.then.loc21, %for.next
// CHECK:STDOUT:   ret void, !dbg !17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CRange.Core(ptr sret({ i32, i32 }), i32)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CNewCursor.IntRange.Core:Iterate.Core.be1e879c1ad406d8"(ptr %self) #0 !dbg !18 {
// CHECK:STDOUT:   %start = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 0, !dbg !20
// CHECK:STDOUT:   %1 = load i32, ptr %start, align 4, !dbg !20
// CHECK:STDOUT:   ret i32 %1, !dbg !21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNext.IntRange.Core:Iterate.Core.be1e879c1ad406d8"(ptr sret({ i32, i1 }) %return, ptr %self, ptr %cursor) #0 !dbg !22 {
// CHECK:STDOUT:   %1 = alloca i32, align 4, !dbg !23
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %1), !dbg !23
// CHECK:STDOUT:   %2 = load i32, ptr %cursor, align 4, !dbg !24
// CHECK:STDOUT:   store i32 %2, ptr %1, align 4, !dbg !23
// CHECK:STDOUT:   %end = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 1, !dbg !25
// CHECK:STDOUT:   %3 = load i32, ptr %end, align 4, !dbg !25
// CHECK:STDOUT:   %4 = load i32, ptr %1, align 4, !dbg !26
// CHECK:STDOUT:   %5 = icmp slt i32 %4, %3, !dbg !26
// CHECK:STDOUT:   br i1 %5, label %6, label %8, !dbg !27
// CHECK:STDOUT:
// CHECK:STDOUT: 6:                                                ; preds = %0
// CHECK:STDOUT:   call void @"_COp.Int.Core:Inc.Core.be1e879c1ad406d8"(ptr %cursor), !dbg !28
// CHECK:STDOUT:   %7 = load i32, ptr %1, align 4, !dbg !29
// CHECK:STDOUT:   call void @_CSome.Optional.Core.c1b882a73b8b9531(ptr %return, i32 %7), !dbg !30
// CHECK:STDOUT:   ret void, !dbg !31
// CHECK:STDOUT:
// CHECK:STDOUT: 8:                                                ; preds = %0
// CHECK:STDOUT:   call void @_CNone.Optional.Core.c1b882a73b8b9531(ptr %return), !dbg !32
// CHECK:STDOUT:   ret void, !dbg !33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CHasValue.Optional.Core.c1b882a73b8b9531(ptr %self) #0 !dbg !34 {
// CHECK:STDOUT:   %1 = call i1 @"_CHas.225258f1a45e9386:OptionalStorage.Core.5450dc8e8b8e0899"(ptr %self), !dbg !36
// CHECK:STDOUT:   ret i1 %1, !dbg !37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CGet.Optional.Core.c1b882a73b8b9531(ptr %self) #0 !dbg !38 {
// CHECK:STDOUT:   %1 = call i32 @"_CGet.225258f1a45e9386:OptionalStorage.Core.5450dc8e8b8e0899"(ptr %self), !dbg !39
// CHECK:STDOUT:   ret i32 %1, !dbg !40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CInclusiveRange.Core(ptr sret({ i32, i32 }), i32, i32)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.Int.Core:Inc.Core.be1e879c1ad406d8"(ptr %self) #0 !dbg !41 {
// CHECK:STDOUT:   call void @"_COp:thunk.Int.Core:AddAssignWith.Core.5dfb78ae56583d8e"(ptr %self, i32 1), !dbg !43
// CHECK:STDOUT:   ret void, !dbg !44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSome.Optional.Core.c1b882a73b8b9531(ptr sret({ i32, i1 }) %return, i32 %value) #0 !dbg !45 {
// CHECK:STDOUT:   call void @"_CSome.225258f1a45e9386:OptionalStorage.Core.5450dc8e8b8e0899"(ptr %return, i32 %value), !dbg !46
// CHECK:STDOUT:   ret void, !dbg !47
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CNone.Optional.Core.c1b882a73b8b9531(ptr sret({ i32, i1 }) %return) #0 !dbg !48 {
// CHECK:STDOUT:   call void @"_CNone.225258f1a45e9386:OptionalStorage.Core.5450dc8e8b8e0899"(ptr %return), !dbg !49
// CHECK:STDOUT:   ret void, !dbg !50
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @"_CHas.225258f1a45e9386:OptionalStorage.Core.5450dc8e8b8e0899"(ptr %value) #0 !dbg !51 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw { i32, i1 }, ptr %value, i32 0, i32 1, !dbg !52
// CHECK:STDOUT:   %1 = load i8, ptr %has_value, align 1, !dbg !52
// CHECK:STDOUT:   %2 = trunc i8 %1 to i1, !dbg !52
// CHECK:STDOUT:   ret i1 %2, !dbg !53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CGet.225258f1a45e9386:OptionalStorage.Core.5450dc8e8b8e0899"(ptr %value) #0 !dbg !54 {
// CHECK:STDOUT:   %value1 = getelementptr inbounds nuw { i32, i1 }, ptr %value, i32 0, i32 0, !dbg !55
// CHECK:STDOUT:   %1 = load i32, ptr %value1, align 4, !dbg !55
// CHECK:STDOUT:   ret i32 %1, !dbg !56
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp:thunk.Int.Core:AddAssignWith.Core.5dfb78ae56583d8e"(ptr %self, i32 %other) #2 !dbg !57 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.225258f1a45e9386:ImplicitAs.Core.5450dc8e8b8e0899"(i32 %other), !dbg !58
// CHECK:STDOUT:   %2 = load i32, ptr %self, align 4, !dbg !59
// CHECK:STDOUT:   %3 = add i32 %2, %1, !dbg !59
// CHECK:STDOUT:   store i32 %3, ptr %self, align 4, !dbg !59
// CHECK:STDOUT:   ret void, !dbg !59
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.225258f1a45e9386:OptionalStorage.Core.5450dc8e8b8e0899"(ptr sret({ i32, i1 }) %return, i32 %self) #0 !dbg !60 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw { i32, i1 }, ptr %return, i32 0, i32 0, !dbg !61
// CHECK:STDOUT:   store i32 %self, ptr %value, align 4, !dbg !61
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw { i32, i1 }, ptr %return, i32 0, i32 1, !dbg !62
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !62
// CHECK:STDOUT:   ret void, !dbg !63
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNone.225258f1a45e9386:OptionalStorage.Core.5450dc8e8b8e0899"(ptr sret({ i32, i1 }) %return) #0 !dbg !64 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw { i32, i1 }, ptr %return, i32 0, i32 1, !dbg !65
// CHECK:STDOUT:   store i8 0, ptr %has_value, align 1, !dbg !65
// CHECK:STDOUT:   ret void, !dbg !66
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.225258f1a45e9386:ImplicitAs.Core.5450dc8e8b8e0899"(i32 %self) #0 !dbg !67 {
// CHECK:STDOUT:   ret i32 %self, !dbg !69
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
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "break_continue.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "For", linkageName: "_CFor.Main", scope: null, file: !3, line: 19, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 20, column: 18, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 20, column: 7, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 21, column: 9, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 21, column: 8, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 21, column: 16, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 22, column: 9, scope: !4)
// CHECK:STDOUT: !13 = !DILocation(line: 22, column: 8, scope: !4)
// CHECK:STDOUT: !14 = !DILocation(line: 22, column: 16, scope: !4)
// CHECK:STDOUT: !15 = !DILocation(line: 23, column: 5, scope: !4)
// CHECK:STDOUT: !16 = !DILocation(line: 20, column: 3, scope: !4)
// CHECK:STDOUT: !17 = !DILocation(line: 19, column: 1, scope: !4)
// CHECK:STDOUT: !18 = distinct !DISubprogram(name: "NewCursor", linkageName: "_CNewCursor.IntRange.Core:Iterate.Core.be1e879c1ad406d8", scope: null, file: !19, line: 25, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !19 = !DIFile(filename: "{{.*}}/range.carbon", directory: "")
// CHECK:STDOUT: !20 = !DILocation(line: 25, column: 51, scope: !18)
// CHECK:STDOUT: !21 = !DILocation(line: 25, column: 44, scope: !18)
// CHECK:STDOUT: !22 = distinct !DISubprogram(name: "Next", linkageName: "_CNext.IntRange.Core:Iterate.Core.be1e879c1ad406d8", scope: null, file: !19, line: 26, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !23 = !DILocation(line: 27, column: 7, scope: !22)
// CHECK:STDOUT: !24 = !DILocation(line: 27, column: 27, scope: !22)
// CHECK:STDOUT: !25 = !DILocation(line: 28, column: 19, scope: !22)
// CHECK:STDOUT: !26 = !DILocation(line: 28, column: 11, scope: !22)
// CHECK:STDOUT: !27 = !DILocation(line: 28, column: 10, scope: !22)
// CHECK:STDOUT: !28 = !DILocation(line: 29, column: 9, scope: !22)
// CHECK:STDOUT: !29 = !DILocation(line: 30, column: 38, scope: !22)
// CHECK:STDOUT: !30 = !DILocation(line: 30, column: 16, scope: !22)
// CHECK:STDOUT: !31 = !DILocation(line: 30, column: 9, scope: !22)
// CHECK:STDOUT: !32 = !DILocation(line: 32, column: 16, scope: !22)
// CHECK:STDOUT: !33 = !DILocation(line: 32, column: 9, scope: !22)
// CHECK:STDOUT: !34 = distinct !DISubprogram(name: "HasValue", linkageName: "_CHasValue.Optional.Core.c1b882a73b8b9531", scope: null, file: !35, line: 32, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !35 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !36 = !DILocation(line: 33, column: 12, scope: !34)
// CHECK:STDOUT: !37 = !DILocation(line: 33, column: 5, scope: !34)
// CHECK:STDOUT: !38 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.Optional.Core.c1b882a73b8b9531", scope: null, file: !35, line: 35, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !39 = !DILocation(line: 36, column: 12, scope: !38)
// CHECK:STDOUT: !40 = !DILocation(line: 36, column: 5, scope: !38)
// CHECK:STDOUT: !41 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Int.Core:Inc.Core.be1e879c1ad406d8", scope: null, file: !42, line: 339, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !42 = !DIFile(filename: "{{.*}}/prelude/types/int.carbon", directory: "")
// CHECK:STDOUT: !43 = !DILocation(line: 341, column: 5, scope: !41)
// CHECK:STDOUT: !44 = !DILocation(line: 339, column: 3, scope: !41)
// CHECK:STDOUT: !45 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.c1b882a73b8b9531", scope: null, file: !35, line: 29, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !46 = !DILocation(line: 30, column: 12, scope: !45)
// CHECK:STDOUT: !47 = !DILocation(line: 30, column: 5, scope: !45)
// CHECK:STDOUT: !48 = distinct !DISubprogram(name: "None", linkageName: "_CNone.Optional.Core.c1b882a73b8b9531", scope: null, file: !35, line: 26, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !49 = !DILocation(line: 27, column: 12, scope: !48)
// CHECK:STDOUT: !50 = !DILocation(line: 27, column: 5, scope: !48)
// CHECK:STDOUT: !51 = distinct !DISubprogram(name: "Has", linkageName: "_CHas.225258f1a45e9386:OptionalStorage.Core.5450dc8e8b8e0899", scope: null, file: !35, line: 112, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !52 = !DILocation(line: 113, column: 12, scope: !51)
// CHECK:STDOUT: !53 = !DILocation(line: 113, column: 5, scope: !51)
// CHECK:STDOUT: !54 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.225258f1a45e9386:OptionalStorage.Core.5450dc8e8b8e0899", scope: null, file: !35, line: 115, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !55 = !DILocation(line: 116, column: 12, scope: !54)
// CHECK:STDOUT: !56 = !DILocation(line: 116, column: 5, scope: !54)
// CHECK:STDOUT: !57 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk.Int.Core:AddAssignWith.Core.5dfb78ae56583d8e", scope: null, file: !42, line: 275, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !58 = !DILocation(line: 4294967295, scope: !57)
// CHECK:STDOUT: !59 = !DILocation(line: 275, column: 3, scope: !57)
// CHECK:STDOUT: !60 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.225258f1a45e9386:OptionalStorage.Core.5450dc8e8b8e0899", scope: null, file: !35, line: 104, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !61 = !DILocation(line: 108, column: 5, scope: !60)
// CHECK:STDOUT: !62 = !DILocation(line: 109, column: 5, scope: !60)
// CHECK:STDOUT: !63 = !DILocation(line: 110, column: 5, scope: !60)
// CHECK:STDOUT: !64 = distinct !DISubprogram(name: "None", linkageName: "_CNone.225258f1a45e9386:OptionalStorage.Core.5450dc8e8b8e0899", scope: null, file: !35, line: 99, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !65 = !DILocation(line: 101, column: 5, scope: !64)
// CHECK:STDOUT: !66 = !DILocation(line: 102, column: 5, scope: !64)
// CHECK:STDOUT: !67 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.225258f1a45e9386:ImplicitAs.Core.5450dc8e8b8e0899", scope: null, file: !68, line: 24, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !68 = !DIFile(filename: "{{.*}}/prelude/operators/as.carbon", directory: "")
// CHECK:STDOUT: !69 = !DILocation(line: 24, column: 38, scope: !67)
