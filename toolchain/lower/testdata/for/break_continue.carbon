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
  for (_: i32 in Core.Range(100)) {
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
// CHECK:STDOUT:   %.loc20_32.1.temp = alloca { i32, i32 }, align 4, !dbg !7
// CHECK:STDOUT:   %var = alloca i32, align 4, !dbg !8
// CHECK:STDOUT:   %.loc20_33.1.temp = alloca <{ i32, i1 }>, align 4, !dbg !8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc20_32.1.temp), !dbg !7
// CHECK:STDOUT:   call void @_CRange.Core(ptr %.loc20_32.1.temp, i32 100), !dbg !7
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.NewCursor.call = call i32 @"_CNewCursor.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8"(ptr %.loc20_32.1.temp), !dbg !8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %var), !dbg !8
// CHECK:STDOUT:   store i32 %IntRange.as.Iterate.impl.NewCursor.call, ptr %var, align 4, !dbg !8
// CHECK:STDOUT:   br label %for.next, !dbg !8
// CHECK:STDOUT:
// CHECK:STDOUT: for.next:                                         ; preds = %if.else.loc22, %if.then.loc22, %entry
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc20_33.1.temp), !dbg !8
// CHECK:STDOUT:   call void @"_CNext.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8"(ptr %.loc20_33.1.temp, ptr %.loc20_32.1.temp, ptr %var), !dbg !8
// CHECK:STDOUT:   %Optional.HasValue.call = call i1 @_CHasValue.Optional.Core.95ac245efc335d3f(ptr %.loc20_33.1.temp), !dbg !8
// CHECK:STDOUT:   br i1 %Optional.HasValue.call, label %for.body, label %for.done, !dbg !8
// CHECK:STDOUT:
// CHECK:STDOUT: for.body:                                         ; preds = %for.next
// CHECK:STDOUT:   %Optional.Get.call = call i32 @_CGet.Optional.Core.95ac245efc335d3f(ptr %.loc20_33.1.temp), !dbg !8
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
// CHECK:STDOUT:   call void @"_COp.ab3effcb00e348a7:core.Destroy.Core"(ptr %.loc20_33.1.temp), !dbg !8
// CHECK:STDOUT:   call void @"_COp.5e27612b9dd31a14:core.Destroy.Core"(ptr %var), !dbg !8
// CHECK:STDOUT:   call void @"_COp.0d5da2659ce22e5f:core.Destroy.Core"(ptr %.loc20_32.1.temp), !dbg !7
// CHECK:STDOUT:   ret void, !dbg !17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CRange.Core(ptr sret({ i32, i32 }), i32)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.5e27612b9dd31a14:core.Destroy.Core"(ptr %self) #0 !dbg !18 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.94ba1c30d421702e:core.Destroy.Core"(ptr %self) #0 !dbg !26 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.0fc9634857315db2:core.Destroy.Core"(ptr %self) #0 !dbg !33 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.69700c1fc1b2744c:core.Destroy.Core"(ptr %self) #0 !dbg !37 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.278d9a31079e7ff2:core.Destroy.Core"(ptr %self) #0 !dbg !41 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.ab3effcb00e348a7:core.Destroy.Core"(ptr %self) #0 !dbg !45 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.db43fa38232c28dd:core.Destroy.Core"(ptr %self) #0 !dbg !49 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !52
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.0d5da2659ce22e5f:core.Destroy.Core"(ptr %self) #0 !dbg !53 {
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
// CHECK:STDOUT:   call void @_CSome.Optional.Core.95ac245efc335d3f(ptr %return, i32 %7), !dbg !77
// CHECK:STDOUT:   call void @"_COp.5e27612b9dd31a14:core.Destroy.Core"(ptr %1), !dbg !70
// CHECK:STDOUT:   ret void, !dbg !78
// CHECK:STDOUT:
// CHECK:STDOUT: 8:                                                ; preds = %0
// CHECK:STDOUT:   call void @_CNone.Optional.Core.95ac245efc335d3f(ptr %return), !dbg !79
// CHECK:STDOUT:   call void @"_COp.5e27612b9dd31a14:core.Destroy.Core"(ptr %1), !dbg !70
// CHECK:STDOUT:   ret void, !dbg !80
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CHasValue.Optional.Core.95ac245efc335d3f(ptr %self) #0 !dbg !81 {
// CHECK:STDOUT:   %1 = call i1 @"_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.39417379e302314a"(ptr %self), !dbg !87
// CHECK:STDOUT:   ret i1 %1, !dbg !88
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CGet.Optional.Core.95ac245efc335d3f(ptr %self) #0 !dbg !89 {
// CHECK:STDOUT:   %1 = call i32 @"_CGet.5d6f3f3a4b44a8f1:OptionalStorage.Core.39417379e302314a"(ptr %self), !dbg !92
// CHECK:STDOUT:   ret i32 %1, !dbg !93
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CInclusiveRange.Core(ptr sret({ i32, i32 }), i32, i32)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8"(ptr %self) #0 !dbg !94 {
// CHECK:STDOUT:   call void @"_COp:thunk:AddAssignWith.d395a13f81aa0e29.Core:Int.9452f4c51951679b.Core:AddAssignWith.bcc2a64c00f3554f.Core.44914b6ff6901661"(ptr %self, i32 1), !dbg !98
// CHECK:STDOUT:   ret void, !dbg !99
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSome.Optional.Core.95ac245efc335d3f(ptr sret(<{ i32, i1 }>) %return, i32 %value) #0 !dbg !100 {
// CHECK:STDOUT:   call void @"_CSome.5d6f3f3a4b44a8f1:OptionalStorage.Core.39417379e302314a"(ptr %return, i32 %value), !dbg !105
// CHECK:STDOUT:   ret void, !dbg !106
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CNone.Optional.Core.95ac245efc335d3f(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !107 {
// CHECK:STDOUT:   call void @"_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.39417379e302314a"(ptr %return), !dbg !110
// CHECK:STDOUT:   ret void, !dbg !111
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @"_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.39417379e302314a"(ptr %value) #0 !dbg !112 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %value, i32 0, i32 1, !dbg !115
// CHECK:STDOUT:   %1 = load i8, ptr %has_value, align 1, !dbg !115
// CHECK:STDOUT:   %2 = trunc i8 %1 to i1, !dbg !115
// CHECK:STDOUT:   ret i1 %2, !dbg !116
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CGet.5d6f3f3a4b44a8f1:OptionalStorage.Core.39417379e302314a"(ptr %value) #0 !dbg !117 {
// CHECK:STDOUT:   %value1 = getelementptr inbounds nuw <{ i32, i1 }>, ptr %value, i32 0, i32 0, !dbg !120
// CHECK:STDOUT:   %1 = load i32, ptr %value1, align 4, !dbg !120
// CHECK:STDOUT:   ret i32 %1, !dbg !121
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp:thunk:AddAssignWith.d395a13f81aa0e29.Core:Int.9452f4c51951679b.Core:AddAssignWith.bcc2a64c00f3554f.Core.44914b6ff6901661"(ptr %self, i32 %other) #2 !dbg !122 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.40372ba0ea03ef47"(i32 %other), !dbg !128
// CHECK:STDOUT:   %2 = load i32, ptr %self, align 4, !dbg !129
// CHECK:STDOUT:   %3 = add i32 %2, %1, !dbg !129
// CHECK:STDOUT:   store i32 %3, ptr %self, align 4, !dbg !129
// CHECK:STDOUT:   ret void, !dbg !129
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.5d6f3f3a4b44a8f1:OptionalStorage.Core.39417379e302314a"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !130 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 0, !dbg !133
// CHECK:STDOUT:   store i32 %self, ptr %value, align 4, !dbg !133
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !134
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !134
// CHECK:STDOUT:   ret void, !dbg !135
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.39417379e302314a"(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !136 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !137
// CHECK:STDOUT:   store i8 0, ptr %has_value, align 1, !dbg !137
// CHECK:STDOUT:   ret void, !dbg !138
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.40372ba0ea03ef47"(i32 %self) #0 !dbg !139 {
// CHECK:STDOUT:   %1 = call i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self), !dbg !144
// CHECK:STDOUT:   ret i32 %1, !dbg !145
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self) #0 !dbg !146 {
// CHECK:STDOUT:   ret i32 %self, !dbg !149
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
// CHECK:STDOUT: !3 = !DIFile(filename: "break_continue.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "For", linkageName: "_CFor.Main", scope: null, file: !3, line: 19, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null}
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
// CHECK:STDOUT: !18 = distinct !DISubprogram(name: "Op", linkageName: "_COp.5e27612b9dd31a14:core.Destroy.Core", scope: null, file: !19, line: 24, type: !20, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !23)
// CHECK:STDOUT: !19 = !DIFile(filename: "{{.*}}/range.carbon", directory: "")
// CHECK:STDOUT: !20 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !21 = !{null, !22}
// CHECK:STDOUT: !22 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !23 = !{!24}
// CHECK:STDOUT: !24 = !DILocalVariable(arg: 1, scope: !18, type: !22)
// CHECK:STDOUT: !25 = !DILocation(line: 24, column: 39, scope: !18)
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "Op", linkageName: "_COp.94ba1c30d421702e:core.Destroy.Core", scope: null, file: !3, line: 20, type: !27, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !30)
// CHECK:STDOUT: !27 = !DISubroutineType(types: !28)
// CHECK:STDOUT: !28 = !{null, !29}
// CHECK:STDOUT: !29 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !30 = !{!31}
// CHECK:STDOUT: !31 = !DILocalVariable(arg: 1, scope: !26, type: !29)
// CHECK:STDOUT: !32 = !DILocation(line: 20, column: 7, scope: !26)
// CHECK:STDOUT: !33 = distinct !DISubprogram(name: "Op", linkageName: "_COp.0fc9634857315db2:core.Destroy.Core", scope: null, file: !3, line: 20, type: !20, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !34)
// CHECK:STDOUT: !34 = !{!35}
// CHECK:STDOUT: !35 = !DILocalVariable(arg: 1, scope: !33, type: !22)
// CHECK:STDOUT: !36 = !DILocation(line: 20, column: 7, scope: !33)
// CHECK:STDOUT: !37 = distinct !DISubprogram(name: "Op", linkageName: "_COp.69700c1fc1b2744c:core.Destroy.Core", scope: null, file: !3, line: 20, type: !27, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !38)
// CHECK:STDOUT: !38 = !{!39}
// CHECK:STDOUT: !39 = !DILocalVariable(arg: 1, scope: !37, type: !29)
// CHECK:STDOUT: !40 = !DILocation(line: 20, column: 7, scope: !37)
// CHECK:STDOUT: !41 = distinct !DISubprogram(name: "Op", linkageName: "_COp.278d9a31079e7ff2:core.Destroy.Core", scope: null, file: !3, line: 20, type: !27, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !42)
// CHECK:STDOUT: !42 = !{!43}
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 1, scope: !41, type: !29)
// CHECK:STDOUT: !44 = !DILocation(line: 20, column: 7, scope: !41)
// CHECK:STDOUT: !45 = distinct !DISubprogram(name: "Op", linkageName: "_COp.ab3effcb00e348a7:core.Destroy.Core", scope: null, file: !3, line: 20, type: !27, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !46)
// CHECK:STDOUT: !46 = !{!47}
// CHECK:STDOUT: !47 = !DILocalVariable(arg: 1, scope: !45, type: !29)
// CHECK:STDOUT: !48 = !DILocation(line: 20, column: 7, scope: !45)
// CHECK:STDOUT: !49 = distinct !DISubprogram(name: "Op", linkageName: "_COp.db43fa38232c28dd:core.Destroy.Core", scope: null, file: !3, line: 20, type: !27, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !50)
// CHECK:STDOUT: !50 = !{!51}
// CHECK:STDOUT: !51 = !DILocalVariable(arg: 1, scope: !49, type: !29)
// CHECK:STDOUT: !52 = !DILocation(line: 20, column: 18, scope: !49)
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "Op", linkageName: "_COp.0d5da2659ce22e5f:core.Destroy.Core", scope: null, file: !3, line: 20, type: !27, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !54)
// CHECK:STDOUT: !54 = !{!55}
// CHECK:STDOUT: !55 = !DILocalVariable(arg: 1, scope: !53, type: !29)
// CHECK:STDOUT: !56 = !DILocation(line: 20, column: 18, scope: !53)
// CHECK:STDOUT: !57 = distinct !DISubprogram(name: "NewCursor", linkageName: "_CNewCursor.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8", scope: null, file: !19, line: 25, type: !58, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !60)
// CHECK:STDOUT: !58 = !DISubroutineType(types: !59)
// CHECK:STDOUT: !59 = !{!22, !29}
// CHECK:STDOUT: !60 = !{!61}
// CHECK:STDOUT: !61 = !DILocalVariable(arg: 1, scope: !57, type: !29)
// CHECK:STDOUT: !62 = !DILocation(line: 25, column: 43, scope: !57)
// CHECK:STDOUT: !63 = !DILocation(line: 25, column: 36, scope: !57)
// CHECK:STDOUT: !64 = distinct !DISubprogram(name: "Next", linkageName: "_CNext.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8", scope: null, file: !19, line: 26, type: !65, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !67)
// CHECK:STDOUT: !65 = !DISubroutineType(types: !66)
// CHECK:STDOUT: !66 = !{!29, !29, !29}
// CHECK:STDOUT: !67 = !{!68, !69}
// CHECK:STDOUT: !68 = !DILocalVariable(arg: 1, scope: !64, type: !29)
// CHECK:STDOUT: !69 = !DILocalVariable(arg: 2, scope: !64, type: !29)
// CHECK:STDOUT: !70 = !DILocation(line: 27, column: 7, scope: !64)
// CHECK:STDOUT: !71 = !DILocation(line: 27, column: 27, scope: !64)
// CHECK:STDOUT: !72 = !DILocation(line: 28, column: 19, scope: !64)
// CHECK:STDOUT: !73 = !DILocation(line: 28, column: 11, scope: !64)
// CHECK:STDOUT: !74 = !DILocation(line: 28, column: 10, scope: !64)
// CHECK:STDOUT: !75 = !DILocation(line: 29, column: 9, scope: !64)
// CHECK:STDOUT: !76 = !DILocation(line: 30, column: 38, scope: !64)
// CHECK:STDOUT: !77 = !DILocation(line: 30, column: 16, scope: !64)
// CHECK:STDOUT: !78 = !DILocation(line: 30, column: 9, scope: !64)
// CHECK:STDOUT: !79 = !DILocation(line: 32, column: 16, scope: !64)
// CHECK:STDOUT: !80 = !DILocation(line: 32, column: 9, scope: !64)
// CHECK:STDOUT: !81 = distinct !DISubprogram(name: "HasValue", linkageName: "_CHasValue.Optional.Core.95ac245efc335d3f", scope: null, file: !82, line: 36, type: !83, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !85)
// CHECK:STDOUT: !82 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !83 = !DISubroutineType(types: !84)
// CHECK:STDOUT: !84 = !{!29, !29}
// CHECK:STDOUT: !85 = !{!86}
// CHECK:STDOUT: !86 = !DILocalVariable(arg: 1, scope: !81, type: !29)
// CHECK:STDOUT: !87 = !DILocation(line: 37, column: 12, scope: !81)
// CHECK:STDOUT: !88 = !DILocation(line: 37, column: 5, scope: !81)
// CHECK:STDOUT: !89 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.Optional.Core.95ac245efc335d3f", scope: null, file: !82, line: 39, type: !58, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !90)
// CHECK:STDOUT: !90 = !{!91}
// CHECK:STDOUT: !91 = !DILocalVariable(arg: 1, scope: !89, type: !29)
// CHECK:STDOUT: !92 = !DILocation(line: 40, column: 12, scope: !89)
// CHECK:STDOUT: !93 = !DILocation(line: 40, column: 5, scope: !89)
// CHECK:STDOUT: !94 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8", scope: null, file: !95, line: 370, type: !20, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !96)
// CHECK:STDOUT: !95 = !DIFile(filename: "{{.*}}/prelude/types/int.carbon", directory: "")
// CHECK:STDOUT: !96 = !{!97}
// CHECK:STDOUT: !97 = !DILocalVariable(arg: 1, scope: !94, type: !22)
// CHECK:STDOUT: !98 = !DILocation(line: 372, column: 5, scope: !94)
// CHECK:STDOUT: !99 = !DILocation(line: 370, column: 3, scope: !94)
// CHECK:STDOUT: !100 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.95ac245efc335d3f", scope: null, file: !82, line: 33, type: !101, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !103)
// CHECK:STDOUT: !101 = !DISubroutineType(types: !102)
// CHECK:STDOUT: !102 = !{!29, !22}
// CHECK:STDOUT: !103 = !{!104}
// CHECK:STDOUT: !104 = !DILocalVariable(arg: 1, scope: !100, type: !22)
// CHECK:STDOUT: !105 = !DILocation(line: 34, column: 12, scope: !100)
// CHECK:STDOUT: !106 = !DILocation(line: 34, column: 5, scope: !100)
// CHECK:STDOUT: !107 = distinct !DISubprogram(name: "None", linkageName: "_CNone.Optional.Core.95ac245efc335d3f", scope: null, file: !82, line: 30, type: !108, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !108 = !DISubroutineType(types: !109)
// CHECK:STDOUT: !109 = !{!29}
// CHECK:STDOUT: !110 = !DILocation(line: 31, column: 12, scope: !107)
// CHECK:STDOUT: !111 = !DILocation(line: 31, column: 5, scope: !107)
// CHECK:STDOUT: !112 = distinct !DISubprogram(name: "Has", linkageName: "_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.39417379e302314a", scope: null, file: !82, line: 141, type: !83, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !113)
// CHECK:STDOUT: !113 = !{!114}
// CHECK:STDOUT: !114 = !DILocalVariable(arg: 1, scope: !112, type: !29)
// CHECK:STDOUT: !115 = !DILocation(line: 142, column: 12, scope: !112)
// CHECK:STDOUT: !116 = !DILocation(line: 142, column: 5, scope: !112)
// CHECK:STDOUT: !117 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.5d6f3f3a4b44a8f1:OptionalStorage.Core.39417379e302314a", scope: null, file: !82, line: 144, type: !58, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !118)
// CHECK:STDOUT: !118 = !{!119}
// CHECK:STDOUT: !119 = !DILocalVariable(arg: 1, scope: !117, type: !29)
// CHECK:STDOUT: !120 = !DILocation(line: 145, column: 12, scope: !117)
// CHECK:STDOUT: !121 = !DILocation(line: 145, column: 5, scope: !117)
// CHECK:STDOUT: !122 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:AddAssignWith.d395a13f81aa0e29.Core:Int.9452f4c51951679b.Core:AddAssignWith.bcc2a64c00f3554f.Core.44914b6ff6901661", scope: null, file: !95, line: 306, type: !123, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !125)
// CHECK:STDOUT: !123 = !DISubroutineType(types: !124)
// CHECK:STDOUT: !124 = !{null, !22, !22}
// CHECK:STDOUT: !125 = !{!126, !127}
// CHECK:STDOUT: !126 = !DILocalVariable(arg: 1, scope: !122, type: !22)
// CHECK:STDOUT: !127 = !DILocalVariable(arg: 2, scope: !122, type: !22)
// CHECK:STDOUT: !128 = !DILocation(line: 4294967295, scope: !122)
// CHECK:STDOUT: !129 = !DILocation(line: 306, column: 3, scope: !122)
// CHECK:STDOUT: !130 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.5d6f3f3a4b44a8f1:OptionalStorage.Core.39417379e302314a", scope: null, file: !82, line: 131, type: !101, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !131)
// CHECK:STDOUT: !131 = !{!132}
// CHECK:STDOUT: !132 = !DILocalVariable(arg: 1, scope: !130, type: !22)
// CHECK:STDOUT: !133 = !DILocation(line: 137, column: 5, scope: !130)
// CHECK:STDOUT: !134 = !DILocation(line: 138, column: 5, scope: !130)
// CHECK:STDOUT: !135 = !DILocation(line: 139, column: 5, scope: !130)
// CHECK:STDOUT: !136 = distinct !DISubprogram(name: "None", linkageName: "_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.39417379e302314a", scope: null, file: !82, line: 125, type: !108, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !137 = !DILocation(line: 128, column: 5, scope: !136)
// CHECK:STDOUT: !138 = !DILocation(line: 129, column: 5, scope: !136)
// CHECK:STDOUT: !139 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.40372ba0ea03ef47", scope: null, file: !95, line: 63, type: !140, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !142)
// CHECK:STDOUT: !140 = !DISubroutineType(types: !141)
// CHECK:STDOUT: !141 = !{!22, !22}
// CHECK:STDOUT: !142 = !{!143}
// CHECK:STDOUT: !143 = !DILocalVariable(arg: 1, scope: !139, type: !22)
// CHECK:STDOUT: !144 = !DILocation(line: 65, column: 17, scope: !139)
// CHECK:STDOUT: !145 = !DILocation(line: 65, column: 5, scope: !139)
// CHECK:STDOUT: !146 = distinct !DISubprogram(name: "AsInt", linkageName: "_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8", scope: null, file: !95, line: 58, type: !140, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !147)
// CHECK:STDOUT: !147 = !{!148}
// CHECK:STDOUT: !148 = !DILocalVariable(arg: 1, scope: !146, type: !22)
// CHECK:STDOUT: !149 = !DILocation(line: 58, column: 28, scope: !146)
