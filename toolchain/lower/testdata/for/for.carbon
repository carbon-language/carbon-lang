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
// CHECK:STDOUT: define void @_CFor.Main() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc19_32.1.temp = alloca { i32, i32 }, align 4, !dbg !7
// CHECK:STDOUT:   %var = alloca i32, align 4, !dbg !8
// CHECK:STDOUT:   %.loc19_33.1.temp = alloca <{ i32, i1 }>, align 4, !dbg !8
// CHECK:STDOUT:   call void @_CF.Main(), !dbg !9
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc19_32.1.temp), !dbg !7
// CHECK:STDOUT:   call void @_CRange.Core(ptr %.loc19_32.1.temp, i32 100), !dbg !7
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.NewCursor.call = call i32 @"_CNewCursor.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8"(ptr %.loc19_32.1.temp), !dbg !8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %var), !dbg !8
// CHECK:STDOUT:   store i32 %IntRange.as.Iterate.impl.NewCursor.call, ptr %var, align 4, !dbg !8
// CHECK:STDOUT:   br label %for.next, !dbg !8
// CHECK:STDOUT:
// CHECK:STDOUT: for.next:                                         ; preds = %for.body, %entry
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc19_33.1.temp), !dbg !8
// CHECK:STDOUT:   call void @"_CNext.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8"(ptr %.loc19_33.1.temp, ptr %.loc19_32.1.temp, ptr %var), !dbg !8
// CHECK:STDOUT:   %Optional.HasValue.call = call i1 @_CHasValue.Optional.Core.63bc0bf740b17859(ptr %.loc19_33.1.temp), !dbg !8
// CHECK:STDOUT:   br i1 %Optional.HasValue.call, label %for.body, label %for.done, !dbg !8
// CHECK:STDOUT:
// CHECK:STDOUT: for.body:                                         ; preds = %for.next
// CHECK:STDOUT:   %Optional.Get.call = call i32 @_CGet.Optional.Core.63bc0bf740b17859(ptr %.loc19_33.1.temp), !dbg !8
// CHECK:STDOUT:   call void @_CG.Main(), !dbg !10
// CHECK:STDOUT:   call void @"_COp.1423d794a4867d77:core.Destroy.Core"(ptr %.loc19_33.1.temp), !dbg !8
// CHECK:STDOUT:   br label %for.next, !dbg !11
// CHECK:STDOUT:
// CHECK:STDOUT: for.done:                                         ; preds = %for.next
// CHECK:STDOUT:   call void @"_COp.1423d794a4867d77:core.Destroy.Core"(ptr %.loc19_33.1.temp), !dbg !8
// CHECK:STDOUT:   call void @"_COp.f58524831c37807e:core.Destroy.Core"(ptr %var), !dbg !8
// CHECK:STDOUT:   call void @"_COp.c9184c281ed4335d:core.Destroy.Core"(ptr %.loc19_32.1.temp), !dbg !7
// CHECK:STDOUT:   call void @_CH.Main(), !dbg !12
// CHECK:STDOUT:   ret void, !dbg !13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CRange.Core(ptr sret({ i32, i32 }), i32)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.f58524831c37807e:core.Destroy.Core"(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.1d6a7672bf2f14c6:core.Destroy.Core"(ptr %self) #0 !dbg !14 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.c1dffdf44e31057b:core.Destroy.Core"(ptr %self) #0 !dbg !21 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.271d79a4f33942b3:core.Destroy.Core"(ptr %self) #0 !dbg !28 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.aef98596d3a9e1eb:core.Destroy.Core"(ptr %self) #0 !dbg !32 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.1423d794a4867d77:core.Destroy.Core"(ptr %self) #0 !dbg !36 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.e933d5ff8212e171:core.Destroy.Core"(ptr %self) #0 !dbg !40 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !43
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.c9184c281ed4335d:core.Destroy.Core"(ptr %self) #0 !dbg !44 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !47
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CNewCursor.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8"(ptr %self) #0 !dbg !48 {
// CHECK:STDOUT:   %start = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 0, !dbg !54
// CHECK:STDOUT:   %1 = load i32, ptr %start, align 4, !dbg !54
// CHECK:STDOUT:   ret i32 %1, !dbg !55
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNext.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8"(ptr sret(<{ i32, i1 }>) %return, ptr %self, ptr %cursor) #0 !dbg !56 {
// CHECK:STDOUT:   %1 = alloca i32, align 4, !dbg !62
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %1), !dbg !62
// CHECK:STDOUT:   %2 = load i32, ptr %cursor, align 4, !dbg !63
// CHECK:STDOUT:   store i32 %2, ptr %1, align 4, !dbg !62
// CHECK:STDOUT:   %end = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 1, !dbg !64
// CHECK:STDOUT:   %3 = load i32, ptr %end, align 4, !dbg !64
// CHECK:STDOUT:   %4 = load i32, ptr %1, align 4, !dbg !65
// CHECK:STDOUT:   %5 = icmp slt i32 %4, %3, !dbg !65
// CHECK:STDOUT:   br i1 %5, label %6, label %8, !dbg !66
// CHECK:STDOUT:
// CHECK:STDOUT: 6:                                                ; preds = %0
// CHECK:STDOUT:   call void @"_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8"(ptr %cursor), !dbg !67
// CHECK:STDOUT:   %7 = load i32, ptr %1, align 4, !dbg !68
// CHECK:STDOUT:   call void @_CSome.Optional.Core.63bc0bf740b17859(ptr %return, i32 %7), !dbg !69
// CHECK:STDOUT:   call void @"_COp.f58524831c37807e:core.Destroy.Core"(ptr %1), !dbg !62
// CHECK:STDOUT:   ret void, !dbg !70
// CHECK:STDOUT:
// CHECK:STDOUT: 8:                                                ; preds = %0
// CHECK:STDOUT:   call void @_CNone.Optional.Core.63bc0bf740b17859(ptr %return), !dbg !71
// CHECK:STDOUT:   call void @"_COp.f58524831c37807e:core.Destroy.Core"(ptr %1), !dbg !62
// CHECK:STDOUT:   ret void, !dbg !72
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CHasValue.Optional.Core.63bc0bf740b17859(ptr %self) #0 !dbg !73 {
// CHECK:STDOUT:   %1 = call i1 @"_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.89a3e87df6fdafe1"(ptr %self), !dbg !79
// CHECK:STDOUT:   ret i1 %1, !dbg !80
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CGet.Optional.Core.63bc0bf740b17859(ptr %self) #0 !dbg !81 {
// CHECK:STDOUT:   %1 = call i32 @"_CGet.5d6f3f3a4b44a8f1:OptionalStorage.Core.89a3e87df6fdafe1"(ptr %self), !dbg !84
// CHECK:STDOUT:   ret i32 %1, !dbg !85
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CInclusiveRange.Core(ptr sret({ i32, i32 }), i32, i32)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8"(ptr %self) #0 !dbg !86 {
// CHECK:STDOUT:   call void @"_COp:thunk:AddAssignWith.39e2e78a3c2e036d.Core:Int.9452f4c51951679b.Core:AddAssignWith.bcc2a64c00f3554f.Core.83481f1872d67895"(ptr %self, i32 1), !dbg !90
// CHECK:STDOUT:   ret void, !dbg !91
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSome.Optional.Core.63bc0bf740b17859(ptr sret(<{ i32, i1 }>) %return, i32 %value) #0 !dbg !92 {
// CHECK:STDOUT:   call void @"_CSome.5d6f3f3a4b44a8f1:OptionalStorage.Core.89a3e87df6fdafe1"(ptr %return, i32 %value), !dbg !97
// CHECK:STDOUT:   ret void, !dbg !98
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CNone.Optional.Core.63bc0bf740b17859(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !99 {
// CHECK:STDOUT:   call void @"_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.89a3e87df6fdafe1"(ptr %return), !dbg !102
// CHECK:STDOUT:   ret void, !dbg !103
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @"_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.89a3e87df6fdafe1"(ptr %value) #0 !dbg !104 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %value, i32 0, i32 1, !dbg !107
// CHECK:STDOUT:   %1 = load i8, ptr %has_value, align 1, !dbg !107
// CHECK:STDOUT:   %2 = trunc i8 %1 to i1, !dbg !107
// CHECK:STDOUT:   ret i1 %2, !dbg !108
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CGet.5d6f3f3a4b44a8f1:OptionalStorage.Core.89a3e87df6fdafe1"(ptr %value) #0 !dbg !109 {
// CHECK:STDOUT:   %value1 = getelementptr inbounds nuw <{ i32, i1 }>, ptr %value, i32 0, i32 0, !dbg !112
// CHECK:STDOUT:   %1 = load i32, ptr %value1, align 4, !dbg !112
// CHECK:STDOUT:   ret i32 %1, !dbg !113
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp:thunk:AddAssignWith.39e2e78a3c2e036d.Core:Int.9452f4c51951679b.Core:AddAssignWith.bcc2a64c00f3554f.Core.83481f1872d67895"(ptr %self, i32 %other) #2 !dbg !114 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.bb469f8924cce044"(i32 %other), !dbg !120
// CHECK:STDOUT:   %2 = load i32, ptr %self, align 4, !dbg !121
// CHECK:STDOUT:   %3 = add i32 %2, %1, !dbg !121
// CHECK:STDOUT:   store i32 %3, ptr %self, align 4, !dbg !121
// CHECK:STDOUT:   ret void, !dbg !121
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.5d6f3f3a4b44a8f1:OptionalStorage.Core.89a3e87df6fdafe1"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !122 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 0, !dbg !125
// CHECK:STDOUT:   store i32 %self, ptr %value, align 4, !dbg !125
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !126
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !126
// CHECK:STDOUT:   ret void, !dbg !127
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.89a3e87df6fdafe1"(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !128 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !129
// CHECK:STDOUT:   store i8 0, ptr %has_value, align 1, !dbg !129
// CHECK:STDOUT:   ret void, !dbg !130
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.bb469f8924cce044"(i32 %self) #0 !dbg !131 {
// CHECK:STDOUT:   %1 = call i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self), !dbg !136
// CHECK:STDOUT:   ret i32 %1, !dbg !137
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self) #0 !dbg !138 {
// CHECK:STDOUT:   ret i32 %self, !dbg !141
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @"_COp.1423d794a4867d77:core.Destroy.Core", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 3, 2, 1 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #2 = { alwaysinline nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "for.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "For", linkageName: "_CFor.Main", scope: null, file: !1, line: 17, type: !5, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DILocation(line: 19, column: 18, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 19, column: 7, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 18, column: 3, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 20, column: 5, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 19, column: 3, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 22, column: 3, scope: !4)
// CHECK:STDOUT: !13 = !DILocation(line: 17, column: 1, scope: !4)
// CHECK:STDOUT: !14 = distinct !DISubprogram(name: "Op", linkageName: "_COp.1d6a7672bf2f14c6:core.Destroy.Core", scope: null, file: !1, line: 19, type: !15, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !18)
// CHECK:STDOUT: !15 = !DISubroutineType(types: !16)
// CHECK:STDOUT: !16 = !{null, !17}
// CHECK:STDOUT: !17 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !18 = !{!19}
// CHECK:STDOUT: !19 = !DILocalVariable(arg: 1, scope: !14, type: !17)
// CHECK:STDOUT: !20 = !DILocation(line: 19, column: 7, scope: !14)
// CHECK:STDOUT: !21 = distinct !DISubprogram(name: "Op", linkageName: "_COp.c1dffdf44e31057b:core.Destroy.Core", scope: null, file: !1, line: 19, type: !22, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !25)
// CHECK:STDOUT: !22 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !23 = !{null, !24}
// CHECK:STDOUT: !24 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !25 = !{!26}
// CHECK:STDOUT: !26 = !DILocalVariable(arg: 1, scope: !21, type: !24)
// CHECK:STDOUT: !27 = !DILocation(line: 19, column: 7, scope: !21)
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "Op", linkageName: "_COp.271d79a4f33942b3:core.Destroy.Core", scope: null, file: !1, line: 19, type: !15, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !29)
// CHECK:STDOUT: !29 = !{!30}
// CHECK:STDOUT: !30 = !DILocalVariable(arg: 1, scope: !28, type: !17)
// CHECK:STDOUT: !31 = !DILocation(line: 19, column: 7, scope: !28)
// CHECK:STDOUT: !32 = distinct !DISubprogram(name: "Op", linkageName: "_COp.aef98596d3a9e1eb:core.Destroy.Core", scope: null, file: !1, line: 19, type: !15, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !33)
// CHECK:STDOUT: !33 = !{!34}
// CHECK:STDOUT: !34 = !DILocalVariable(arg: 1, scope: !32, type: !17)
// CHECK:STDOUT: !35 = !DILocation(line: 19, column: 7, scope: !32)
// CHECK:STDOUT: !36 = distinct !DISubprogram(name: "Op", linkageName: "_COp.1423d794a4867d77:core.Destroy.Core", scope: null, file: !1, line: 19, type: !15, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !37)
// CHECK:STDOUT: !37 = !{!38}
// CHECK:STDOUT: !38 = !DILocalVariable(arg: 1, scope: !36, type: !17)
// CHECK:STDOUT: !39 = !DILocation(line: 19, column: 7, scope: !36)
// CHECK:STDOUT: !40 = distinct !DISubprogram(name: "Op", linkageName: "_COp.e933d5ff8212e171:core.Destroy.Core", scope: null, file: !1, line: 19, type: !15, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !41)
// CHECK:STDOUT: !41 = !{!42}
// CHECK:STDOUT: !42 = !DILocalVariable(arg: 1, scope: !40, type: !17)
// CHECK:STDOUT: !43 = !DILocation(line: 19, column: 18, scope: !40)
// CHECK:STDOUT: !44 = distinct !DISubprogram(name: "Op", linkageName: "_COp.c9184c281ed4335d:core.Destroy.Core", scope: null, file: !1, line: 19, type: !15, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !45)
// CHECK:STDOUT: !45 = !{!46}
// CHECK:STDOUT: !46 = !DILocalVariable(arg: 1, scope: !44, type: !17)
// CHECK:STDOUT: !47 = !DILocation(line: 19, column: 18, scope: !44)
// CHECK:STDOUT: !48 = distinct !DISubprogram(name: "NewCursor", linkageName: "_CNewCursor.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8", scope: null, file: !49, line: 19, type: !50, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !52)
// CHECK:STDOUT: !49 = !DIFile(filename: "{{.*}}/prelude/range.carbon", directory: "")
// CHECK:STDOUT: !50 = !DISubroutineType(types: !51)
// CHECK:STDOUT: !51 = !{!24, !17}
// CHECK:STDOUT: !52 = !{!53}
// CHECK:STDOUT: !53 = !DILocalVariable(arg: 1, scope: !48, type: !17)
// CHECK:STDOUT: !54 = !DILocation(line: 19, column: 43, scope: !48)
// CHECK:STDOUT: !55 = !DILocation(line: 19, column: 36, scope: !48)
// CHECK:STDOUT: !56 = distinct !DISubprogram(name: "Next", linkageName: "_CNext.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8", scope: null, file: !49, line: 20, type: !57, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !59)
// CHECK:STDOUT: !57 = !DISubroutineType(types: !58)
// CHECK:STDOUT: !58 = !{!17, !17, !17}
// CHECK:STDOUT: !59 = !{!60, !61}
// CHECK:STDOUT: !60 = !DILocalVariable(arg: 1, scope: !56, type: !17)
// CHECK:STDOUT: !61 = !DILocalVariable(arg: 2, scope: !56, type: !17)
// CHECK:STDOUT: !62 = !DILocation(line: 21, column: 7, scope: !56)
// CHECK:STDOUT: !63 = !DILocation(line: 21, column: 27, scope: !56)
// CHECK:STDOUT: !64 = !DILocation(line: 22, column: 19, scope: !56)
// CHECK:STDOUT: !65 = !DILocation(line: 22, column: 11, scope: !56)
// CHECK:STDOUT: !66 = !DILocation(line: 22, column: 10, scope: !56)
// CHECK:STDOUT: !67 = !DILocation(line: 23, column: 9, scope: !56)
// CHECK:STDOUT: !68 = !DILocation(line: 24, column: 38, scope: !56)
// CHECK:STDOUT: !69 = !DILocation(line: 24, column: 16, scope: !56)
// CHECK:STDOUT: !70 = !DILocation(line: 24, column: 9, scope: !56)
// CHECK:STDOUT: !71 = !DILocation(line: 26, column: 16, scope: !56)
// CHECK:STDOUT: !72 = !DILocation(line: 26, column: 9, scope: !56)
// CHECK:STDOUT: !73 = distinct !DISubprogram(name: "HasValue", linkageName: "_CHasValue.Optional.Core.63bc0bf740b17859", scope: null, file: !74, line: 36, type: !75, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !77)
// CHECK:STDOUT: !74 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !75 = !DISubroutineType(types: !76)
// CHECK:STDOUT: !76 = !{!17, !17}
// CHECK:STDOUT: !77 = !{!78}
// CHECK:STDOUT: !78 = !DILocalVariable(arg: 1, scope: !73, type: !17)
// CHECK:STDOUT: !79 = !DILocation(line: 37, column: 12, scope: !73)
// CHECK:STDOUT: !80 = !DILocation(line: 37, column: 5, scope: !73)
// CHECK:STDOUT: !81 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.Optional.Core.63bc0bf740b17859", scope: null, file: !74, line: 39, type: !50, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !82)
// CHECK:STDOUT: !82 = !{!83}
// CHECK:STDOUT: !83 = !DILocalVariable(arg: 1, scope: !81, type: !17)
// CHECK:STDOUT: !84 = !DILocation(line: 40, column: 12, scope: !81)
// CHECK:STDOUT: !85 = !DILocation(line: 40, column: 5, scope: !81)
// CHECK:STDOUT: !86 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8", scope: null, file: !87, line: 396, type: !22, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !88)
// CHECK:STDOUT: !87 = !DIFile(filename: "{{.*}}/prelude/types/int.carbon", directory: "")
// CHECK:STDOUT: !88 = !{!89}
// CHECK:STDOUT: !89 = !DILocalVariable(arg: 1, scope: !86, type: !24)
// CHECK:STDOUT: !90 = !DILocation(line: 398, column: 5, scope: !86)
// CHECK:STDOUT: !91 = !DILocation(line: 396, column: 3, scope: !86)
// CHECK:STDOUT: !92 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.63bc0bf740b17859", scope: null, file: !74, line: 33, type: !93, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !95)
// CHECK:STDOUT: !93 = !DISubroutineType(types: !94)
// CHECK:STDOUT: !94 = !{!17, !24}
// CHECK:STDOUT: !95 = !{!96}
// CHECK:STDOUT: !96 = !DILocalVariable(arg: 1, scope: !92, type: !24)
// CHECK:STDOUT: !97 = !DILocation(line: 34, column: 12, scope: !92)
// CHECK:STDOUT: !98 = !DILocation(line: 34, column: 5, scope: !92)
// CHECK:STDOUT: !99 = distinct !DISubprogram(name: "None", linkageName: "_CNone.Optional.Core.63bc0bf740b17859", scope: null, file: !74, line: 30, type: !100, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !100 = !DISubroutineType(types: !101)
// CHECK:STDOUT: !101 = !{!17}
// CHECK:STDOUT: !102 = !DILocation(line: 31, column: 12, scope: !99)
// CHECK:STDOUT: !103 = !DILocation(line: 31, column: 5, scope: !99)
// CHECK:STDOUT: !104 = distinct !DISubprogram(name: "Has", linkageName: "_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.89a3e87df6fdafe1", scope: null, file: !74, line: 141, type: !75, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !105)
// CHECK:STDOUT: !105 = !{!106}
// CHECK:STDOUT: !106 = !DILocalVariable(arg: 1, scope: !104, type: !17)
// CHECK:STDOUT: !107 = !DILocation(line: 142, column: 12, scope: !104)
// CHECK:STDOUT: !108 = !DILocation(line: 142, column: 5, scope: !104)
// CHECK:STDOUT: !109 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.5d6f3f3a4b44a8f1:OptionalStorage.Core.89a3e87df6fdafe1", scope: null, file: !74, line: 144, type: !50, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !110)
// CHECK:STDOUT: !110 = !{!111}
// CHECK:STDOUT: !111 = !DILocalVariable(arg: 1, scope: !109, type: !17)
// CHECK:STDOUT: !112 = !DILocation(line: 145, column: 12, scope: !109)
// CHECK:STDOUT: !113 = !DILocation(line: 145, column: 5, scope: !109)
// CHECK:STDOUT: !114 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:AddAssignWith.39e2e78a3c2e036d.Core:Int.9452f4c51951679b.Core:AddAssignWith.bcc2a64c00f3554f.Core.83481f1872d67895", scope: null, file: !87, line: 331, type: !115, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !117)
// CHECK:STDOUT: !115 = !DISubroutineType(types: !116)
// CHECK:STDOUT: !116 = !{null, !24, !24}
// CHECK:STDOUT: !117 = !{!118, !119}
// CHECK:STDOUT: !118 = !DILocalVariable(arg: 1, scope: !114, type: !24)
// CHECK:STDOUT: !119 = !DILocalVariable(arg: 2, scope: !114, type: !24)
// CHECK:STDOUT: !120 = !DILocation(line: 0, scope: !114)
// CHECK:STDOUT: !121 = !DILocation(line: 331, column: 3, scope: !114)
// CHECK:STDOUT: !122 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.5d6f3f3a4b44a8f1:OptionalStorage.Core.89a3e87df6fdafe1", scope: null, file: !74, line: 131, type: !93, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !123)
// CHECK:STDOUT: !123 = !{!124}
// CHECK:STDOUT: !124 = !DILocalVariable(arg: 1, scope: !122, type: !24)
// CHECK:STDOUT: !125 = !DILocation(line: 137, column: 5, scope: !122)
// CHECK:STDOUT: !126 = !DILocation(line: 138, column: 5, scope: !122)
// CHECK:STDOUT: !127 = !DILocation(line: 139, column: 5, scope: !122)
// CHECK:STDOUT: !128 = distinct !DISubprogram(name: "None", linkageName: "_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.89a3e87df6fdafe1", scope: null, file: !74, line: 125, type: !100, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !129 = !DILocation(line: 128, column: 5, scope: !128)
// CHECK:STDOUT: !130 = !DILocation(line: 129, column: 5, scope: !128)
// CHECK:STDOUT: !131 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.bb469f8924cce044", scope: null, file: !87, line: 64, type: !132, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !134)
// CHECK:STDOUT: !132 = !DISubroutineType(types: !133)
// CHECK:STDOUT: !133 = !{!24, !24}
// CHECK:STDOUT: !134 = !{!135}
// CHECK:STDOUT: !135 = !DILocalVariable(arg: 1, scope: !131, type: !24)
// CHECK:STDOUT: !136 = !DILocation(line: 66, column: 17, scope: !131)
// CHECK:STDOUT: !137 = !DILocation(line: 66, column: 5, scope: !131)
// CHECK:STDOUT: !138 = distinct !DISubprogram(name: "AsInt", linkageName: "_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8", scope: null, file: !87, line: 59, type: !132, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !139)
// CHECK:STDOUT: !139 = !{!140}
// CHECK:STDOUT: !140 = !DILocalVariable(arg: 1, scope: !138, type: !24)
// CHECK:STDOUT: !141 = !DILocation(line: 59, column: 28, scope: !138)
// CHECK:STDOUT:
