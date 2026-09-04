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
// CHECK:STDOUT:   %Optional.HasValue.call = call i1 @_CHasValue.Optional.Core.a5214d4ceffb522b(ptr %.loc19_33.1.temp), !dbg !8
// CHECK:STDOUT:   br i1 %Optional.HasValue.call, label %for.body, label %for.done, !dbg !8
// CHECK:STDOUT:
// CHECK:STDOUT: for.body:                                         ; preds = %for.next
// CHECK:STDOUT:   %Optional.Get.call = call i32 @_CGet.Optional.Core.a5214d4ceffb522b(ptr %.loc19_33.1.temp), !dbg !8
// CHECK:STDOUT:   call void @_CG.Main(), !dbg !10
// CHECK:STDOUT:   call void @"_COp.891c4207e57d0f40:core.Destroy.Core"(ptr %.loc19_33.1.temp), !dbg !8
// CHECK:STDOUT:   br label %for.next, !dbg !11
// CHECK:STDOUT:
// CHECK:STDOUT: for.done:                                         ; preds = %for.next
// CHECK:STDOUT:   call void @"_COp.891c4207e57d0f40:core.Destroy.Core"(ptr %.loc19_33.1.temp), !dbg !8
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
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:Destroy.935345caab7693bf.Core:Destroy.Core"(ptr %self) #1 !dbg !14 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_COp.f58524831c37807e:core.Destroy.Core"(ptr %self), !dbg !21
// CHECK:STDOUT:   ret void, !dbg !21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.1d6a7672bf2f14c6:core.Destroy.Core"(ptr %self) #0 !dbg !22 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.0239a84bcf0315a7:core.Destroy.Core"(ptr %self) #0 !dbg !29 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.ad05ff27ec7895c0:core.Destroy.Core"(ptr %self) #0 !dbg !33 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.ae16cb2db38d93af:core.Destroy.Core"(ptr %self) #0 !dbg !37 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.891c4207e57d0f40:core.Destroy.Core"(ptr %self) #0 !dbg !41 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.e933d5ff8212e171:core.Destroy.Core"(ptr %self) #0 !dbg !45 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.c9184c281ed4335d:core.Destroy.Core"(ptr %self) #0 !dbg !49 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !52
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CNewCursor.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8"(ptr %self) #0 !dbg !53 {
// CHECK:STDOUT:   %start = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 0, !dbg !59
// CHECK:STDOUT:   %1 = load i32, ptr %start, align 4, !dbg !59
// CHECK:STDOUT:   ret i32 %1, !dbg !60
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNext.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8"(ptr sret(<{ i32, i1 }>) %return, ptr %self, ptr %cursor) #0 !dbg !61 {
// CHECK:STDOUT:   %1 = alloca i32, align 4, !dbg !67
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %1), !dbg !67
// CHECK:STDOUT:   %2 = load i32, ptr %cursor, align 4, !dbg !68
// CHECK:STDOUT:   store i32 %2, ptr %1, align 4, !dbg !67
// CHECK:STDOUT:   %end = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 1, !dbg !69
// CHECK:STDOUT:   %3 = load i32, ptr %end, align 4, !dbg !69
// CHECK:STDOUT:   %4 = load i32, ptr %1, align 4, !dbg !70
// CHECK:STDOUT:   %5 = icmp slt i32 %4, %3, !dbg !70
// CHECK:STDOUT:   br i1 %5, label %6, label %8, !dbg !71
// CHECK:STDOUT:
// CHECK:STDOUT: 6:                                                ; preds = %0
// CHECK:STDOUT:   call void @"_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8"(ptr %cursor), !dbg !72
// CHECK:STDOUT:   %7 = load i32, ptr %1, align 4, !dbg !73
// CHECK:STDOUT:   call void @_CSome.Optional.Core.a5214d4ceffb522b(ptr %return, i32 %7), !dbg !74
// CHECK:STDOUT:   call void @"_COp:thunk:Destroy.935345caab7693bf.Core:Destroy.Core"(ptr %1), !dbg !67
// CHECK:STDOUT:   ret void, !dbg !75
// CHECK:STDOUT:
// CHECK:STDOUT: 8:                                                ; preds = %0
// CHECK:STDOUT:   call void @_CNone.Optional.Core.a5214d4ceffb522b(ptr %return), !dbg !76
// CHECK:STDOUT:   call void @"_COp:thunk:Destroy.935345caab7693bf.Core:Destroy.Core"(ptr %1), !dbg !67
// CHECK:STDOUT:   ret void, !dbg !77
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CHasValue.Optional.Core.a5214d4ceffb522b(ptr %self) #0 !dbg !78 {
// CHECK:STDOUT:   %1 = call i1 @"_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.553a1f59d1dd8381"(ptr %self), !dbg !84
// CHECK:STDOUT:   ret i1 %1, !dbg !85
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CGet.Optional.Core.a5214d4ceffb522b(ptr %self) #0 !dbg !86 {
// CHECK:STDOUT:   %1 = call i32 @"_CGet.5d6f3f3a4b44a8f1:OptionalStorage.Core.553a1f59d1dd8381"(ptr %self), !dbg !89
// CHECK:STDOUT:   ret i32 %1, !dbg !90
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CInclusiveRange.Core(ptr sret({ i32, i32 }), i32, i32)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8"(ptr %self) #0 !dbg !91 {
// CHECK:STDOUT:   call void @"_COp:thunk:AddAssignWith.39e2e78a3c2e036d.Core:Int.9452f4c51951679b.Core:AddAssignWith.bcc2a64c00f3554f.Core.83481f1872d67895"(ptr %self, i32 1), !dbg !95
// CHECK:STDOUT:   ret void, !dbg !96
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSome.Optional.Core.a5214d4ceffb522b(ptr sret(<{ i32, i1 }>) %return, i32 %value) #0 !dbg !97 {
// CHECK:STDOUT:   call void @"_CSome.5d6f3f3a4b44a8f1:OptionalStorage.Core.553a1f59d1dd8381"(ptr %return, i32 %value), !dbg !102
// CHECK:STDOUT:   ret void, !dbg !103
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CNone.Optional.Core.a5214d4ceffb522b(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !104 {
// CHECK:STDOUT:   call void @"_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.553a1f59d1dd8381"(ptr %return), !dbg !107
// CHECK:STDOUT:   ret void, !dbg !108
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @"_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.553a1f59d1dd8381"(ptr %value) #0 !dbg !109 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %value, i32 0, i32 1, !dbg !112
// CHECK:STDOUT:   %1 = load i8, ptr %has_value, align 1, !dbg !112
// CHECK:STDOUT:   %2 = trunc i8 %1 to i1, !dbg !112
// CHECK:STDOUT:   ret i1 %2, !dbg !113
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CGet.5d6f3f3a4b44a8f1:OptionalStorage.Core.553a1f59d1dd8381"(ptr %value) #0 !dbg !114 {
// CHECK:STDOUT:   %value1 = getelementptr inbounds nuw <{ i32, i1 }>, ptr %value, i32 0, i32 0, !dbg !117
// CHECK:STDOUT:   %1 = load i32, ptr %value1, align 4, !dbg !117
// CHECK:STDOUT:   ret i32 %1, !dbg !118
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp:thunk:AddAssignWith.39e2e78a3c2e036d.Core:Int.9452f4c51951679b.Core:AddAssignWith.bcc2a64c00f3554f.Core.83481f1872d67895"(ptr %self, i32 %other) #1 !dbg !119 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.bb469f8924cce044"(i32 %other), !dbg !125
// CHECK:STDOUT:   %2 = load i32, ptr %self, align 4, !dbg !126
// CHECK:STDOUT:   %3 = add i32 %2, %1, !dbg !126
// CHECK:STDOUT:   store i32 %3, ptr %self, align 4, !dbg !126
// CHECK:STDOUT:   ret void, !dbg !126
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.5d6f3f3a4b44a8f1:OptionalStorage.Core.553a1f59d1dd8381"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !127 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 0, !dbg !130
// CHECK:STDOUT:   store i32 %self, ptr %value, align 4, !dbg !130
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !131
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !131
// CHECK:STDOUT:   ret void, !dbg !132
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.553a1f59d1dd8381"(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !133 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !134
// CHECK:STDOUT:   store i8 0, ptr %has_value, align 1, !dbg !134
// CHECK:STDOUT:   ret void, !dbg !135
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.bb469f8924cce044"(i32 %self) #0 !dbg !136 {
// CHECK:STDOUT:   %1 = call i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self), !dbg !141
// CHECK:STDOUT:   ret i32 %1, !dbg !142
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self) #0 !dbg !143 {
// CHECK:STDOUT:   ret i32 %self, !dbg !146
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @"_COp.891c4207e57d0f40:core.Destroy.Core", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 3, 2, 1 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
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
// CHECK:STDOUT: !14 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:Destroy.935345caab7693bf.Core:Destroy.Core", scope: null, file: !15, line: 22, type: !16, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !19)
// CHECK:STDOUT: !15 = !DIFile(filename: "{{.*}}/prelude/iterate.carbon", directory: "")
// CHECK:STDOUT: !16 = !DISubroutineType(types: !17)
// CHECK:STDOUT: !17 = !{null, !18}
// CHECK:STDOUT: !18 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !19 = !{!20}
// CHECK:STDOUT: !20 = !DILocalVariable(arg: 1, scope: !14, type: !18)
// CHECK:STDOUT: !21 = !DILocation(line: 22, column: 46, scope: !14)
// CHECK:STDOUT: !22 = distinct !DISubprogram(name: "Op", linkageName: "_COp.1d6a7672bf2f14c6:core.Destroy.Core", scope: null, file: !1, line: 19, type: !23, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !26)
// CHECK:STDOUT: !23 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !24 = !{null, !25}
// CHECK:STDOUT: !25 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !26 = !{!27}
// CHECK:STDOUT: !27 = !DILocalVariable(arg: 1, scope: !22, type: !25)
// CHECK:STDOUT: !28 = !DILocation(line: 19, column: 7, scope: !22)
// CHECK:STDOUT: !29 = distinct !DISubprogram(name: "Op", linkageName: "_COp.0239a84bcf0315a7:core.Destroy.Core", scope: null, file: !1, line: 19, type: !16, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !30)
// CHECK:STDOUT: !30 = !{!31}
// CHECK:STDOUT: !31 = !DILocalVariable(arg: 1, scope: !29, type: !18)
// CHECK:STDOUT: !32 = !DILocation(line: 19, column: 7, scope: !29)
// CHECK:STDOUT: !33 = distinct !DISubprogram(name: "Op", linkageName: "_COp.ad05ff27ec7895c0:core.Destroy.Core", scope: null, file: !1, line: 19, type: !23, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !34)
// CHECK:STDOUT: !34 = !{!35}
// CHECK:STDOUT: !35 = !DILocalVariable(arg: 1, scope: !33, type: !25)
// CHECK:STDOUT: !36 = !DILocation(line: 19, column: 7, scope: !33)
// CHECK:STDOUT: !37 = distinct !DISubprogram(name: "Op", linkageName: "_COp.ae16cb2db38d93af:core.Destroy.Core", scope: null, file: !1, line: 19, type: !23, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !38)
// CHECK:STDOUT: !38 = !{!39}
// CHECK:STDOUT: !39 = !DILocalVariable(arg: 1, scope: !37, type: !25)
// CHECK:STDOUT: !40 = !DILocation(line: 19, column: 7, scope: !37)
// CHECK:STDOUT: !41 = distinct !DISubprogram(name: "Op", linkageName: "_COp.891c4207e57d0f40:core.Destroy.Core", scope: null, file: !1, line: 19, type: !23, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !42)
// CHECK:STDOUT: !42 = !{!43}
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 1, scope: !41, type: !25)
// CHECK:STDOUT: !44 = !DILocation(line: 19, column: 7, scope: !41)
// CHECK:STDOUT: !45 = distinct !DISubprogram(name: "Op", linkageName: "_COp.e933d5ff8212e171:core.Destroy.Core", scope: null, file: !1, line: 19, type: !23, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !46)
// CHECK:STDOUT: !46 = !{!47}
// CHECK:STDOUT: !47 = !DILocalVariable(arg: 1, scope: !45, type: !25)
// CHECK:STDOUT: !48 = !DILocation(line: 19, column: 18, scope: !45)
// CHECK:STDOUT: !49 = distinct !DISubprogram(name: "Op", linkageName: "_COp.c9184c281ed4335d:core.Destroy.Core", scope: null, file: !1, line: 19, type: !23, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !50)
// CHECK:STDOUT: !50 = !{!51}
// CHECK:STDOUT: !51 = !DILocalVariable(arg: 1, scope: !49, type: !25)
// CHECK:STDOUT: !52 = !DILocation(line: 19, column: 18, scope: !49)
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "NewCursor", linkageName: "_CNewCursor.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8", scope: null, file: !54, line: 19, type: !55, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !57)
// CHECK:STDOUT: !54 = !DIFile(filename: "{{.*}}/prelude/range.carbon", directory: "")
// CHECK:STDOUT: !55 = !DISubroutineType(types: !56)
// CHECK:STDOUT: !56 = !{!18, !25}
// CHECK:STDOUT: !57 = !{!58}
// CHECK:STDOUT: !58 = !DILocalVariable(arg: 1, scope: !53, type: !25)
// CHECK:STDOUT: !59 = !DILocation(line: 19, column: 43, scope: !53)
// CHECK:STDOUT: !60 = !DILocation(line: 19, column: 36, scope: !53)
// CHECK:STDOUT: !61 = distinct !DISubprogram(name: "Next", linkageName: "_CNext.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8", scope: null, file: !54, line: 20, type: !62, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !64)
// CHECK:STDOUT: !62 = !DISubroutineType(types: !63)
// CHECK:STDOUT: !63 = !{!25, !25, !25}
// CHECK:STDOUT: !64 = !{!65, !66}
// CHECK:STDOUT: !65 = !DILocalVariable(arg: 1, scope: !61, type: !25)
// CHECK:STDOUT: !66 = !DILocalVariable(arg: 2, scope: !61, type: !25)
// CHECK:STDOUT: !67 = !DILocation(line: 21, column: 7, scope: !61)
// CHECK:STDOUT: !68 = !DILocation(line: 21, column: 27, scope: !61)
// CHECK:STDOUT: !69 = !DILocation(line: 22, column: 19, scope: !61)
// CHECK:STDOUT: !70 = !DILocation(line: 22, column: 11, scope: !61)
// CHECK:STDOUT: !71 = !DILocation(line: 22, column: 10, scope: !61)
// CHECK:STDOUT: !72 = !DILocation(line: 23, column: 9, scope: !61)
// CHECK:STDOUT: !73 = !DILocation(line: 24, column: 38, scope: !61)
// CHECK:STDOUT: !74 = !DILocation(line: 24, column: 16, scope: !61)
// CHECK:STDOUT: !75 = !DILocation(line: 24, column: 9, scope: !61)
// CHECK:STDOUT: !76 = !DILocation(line: 26, column: 16, scope: !61)
// CHECK:STDOUT: !77 = !DILocation(line: 26, column: 9, scope: !61)
// CHECK:STDOUT: !78 = distinct !DISubprogram(name: "HasValue", linkageName: "_CHasValue.Optional.Core.a5214d4ceffb522b", scope: null, file: !79, line: 36, type: !80, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !82)
// CHECK:STDOUT: !79 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !80 = !DISubroutineType(types: !81)
// CHECK:STDOUT: !81 = !{!25, !25}
// CHECK:STDOUT: !82 = !{!83}
// CHECK:STDOUT: !83 = !DILocalVariable(arg: 1, scope: !78, type: !25)
// CHECK:STDOUT: !84 = !DILocation(line: 37, column: 12, scope: !78)
// CHECK:STDOUT: !85 = !DILocation(line: 37, column: 5, scope: !78)
// CHECK:STDOUT: !86 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.Optional.Core.a5214d4ceffb522b", scope: null, file: !79, line: 39, type: !55, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !87)
// CHECK:STDOUT: !87 = !{!88}
// CHECK:STDOUT: !88 = !DILocalVariable(arg: 1, scope: !86, type: !25)
// CHECK:STDOUT: !89 = !DILocation(line: 40, column: 12, scope: !86)
// CHECK:STDOUT: !90 = !DILocation(line: 40, column: 5, scope: !86)
// CHECK:STDOUT: !91 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8", scope: null, file: !92, line: 396, type: !16, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !93)
// CHECK:STDOUT: !92 = !DIFile(filename: "{{.*}}/prelude/types/int.carbon", directory: "")
// CHECK:STDOUT: !93 = !{!94}
// CHECK:STDOUT: !94 = !DILocalVariable(arg: 1, scope: !91, type: !18)
// CHECK:STDOUT: !95 = !DILocation(line: 398, column: 5, scope: !91)
// CHECK:STDOUT: !96 = !DILocation(line: 396, column: 3, scope: !91)
// CHECK:STDOUT: !97 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.a5214d4ceffb522b", scope: null, file: !79, line: 33, type: !98, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !100)
// CHECK:STDOUT: !98 = !DISubroutineType(types: !99)
// CHECK:STDOUT: !99 = !{!25, !18}
// CHECK:STDOUT: !100 = !{!101}
// CHECK:STDOUT: !101 = !DILocalVariable(arg: 1, scope: !97, type: !18)
// CHECK:STDOUT: !102 = !DILocation(line: 34, column: 12, scope: !97)
// CHECK:STDOUT: !103 = !DILocation(line: 34, column: 5, scope: !97)
// CHECK:STDOUT: !104 = distinct !DISubprogram(name: "None", linkageName: "_CNone.Optional.Core.a5214d4ceffb522b", scope: null, file: !79, line: 30, type: !105, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !105 = !DISubroutineType(types: !106)
// CHECK:STDOUT: !106 = !{!25}
// CHECK:STDOUT: !107 = !DILocation(line: 31, column: 12, scope: !104)
// CHECK:STDOUT: !108 = !DILocation(line: 31, column: 5, scope: !104)
// CHECK:STDOUT: !109 = distinct !DISubprogram(name: "Has", linkageName: "_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.553a1f59d1dd8381", scope: null, file: !79, line: 141, type: !80, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !110)
// CHECK:STDOUT: !110 = !{!111}
// CHECK:STDOUT: !111 = !DILocalVariable(arg: 1, scope: !109, type: !25)
// CHECK:STDOUT: !112 = !DILocation(line: 142, column: 12, scope: !109)
// CHECK:STDOUT: !113 = !DILocation(line: 142, column: 5, scope: !109)
// CHECK:STDOUT: !114 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.5d6f3f3a4b44a8f1:OptionalStorage.Core.553a1f59d1dd8381", scope: null, file: !79, line: 144, type: !55, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !115)
// CHECK:STDOUT: !115 = !{!116}
// CHECK:STDOUT: !116 = !DILocalVariable(arg: 1, scope: !114, type: !25)
// CHECK:STDOUT: !117 = !DILocation(line: 145, column: 12, scope: !114)
// CHECK:STDOUT: !118 = !DILocation(line: 145, column: 5, scope: !114)
// CHECK:STDOUT: !119 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:AddAssignWith.39e2e78a3c2e036d.Core:Int.9452f4c51951679b.Core:AddAssignWith.bcc2a64c00f3554f.Core.83481f1872d67895", scope: null, file: !92, line: 331, type: !120, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !122)
// CHECK:STDOUT: !120 = !DISubroutineType(types: !121)
// CHECK:STDOUT: !121 = !{null, !18, !18}
// CHECK:STDOUT: !122 = !{!123, !124}
// CHECK:STDOUT: !123 = !DILocalVariable(arg: 1, scope: !119, type: !18)
// CHECK:STDOUT: !124 = !DILocalVariable(arg: 2, scope: !119, type: !18)
// CHECK:STDOUT: !125 = !DILocation(line: 0, scope: !119)
// CHECK:STDOUT: !126 = !DILocation(line: 331, column: 3, scope: !119)
// CHECK:STDOUT: !127 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.5d6f3f3a4b44a8f1:OptionalStorage.Core.553a1f59d1dd8381", scope: null, file: !79, line: 131, type: !98, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !128)
// CHECK:STDOUT: !128 = !{!129}
// CHECK:STDOUT: !129 = !DILocalVariable(arg: 1, scope: !127, type: !18)
// CHECK:STDOUT: !130 = !DILocation(line: 137, column: 5, scope: !127)
// CHECK:STDOUT: !131 = !DILocation(line: 138, column: 5, scope: !127)
// CHECK:STDOUT: !132 = !DILocation(line: 139, column: 5, scope: !127)
// CHECK:STDOUT: !133 = distinct !DISubprogram(name: "None", linkageName: "_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.553a1f59d1dd8381", scope: null, file: !79, line: 125, type: !105, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !134 = !DILocation(line: 128, column: 5, scope: !133)
// CHECK:STDOUT: !135 = !DILocation(line: 129, column: 5, scope: !133)
// CHECK:STDOUT: !136 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.bb469f8924cce044", scope: null, file: !92, line: 64, type: !137, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !139)
// CHECK:STDOUT: !137 = !DISubroutineType(types: !138)
// CHECK:STDOUT: !138 = !{!18, !18}
// CHECK:STDOUT: !139 = !{!140}
// CHECK:STDOUT: !140 = !DILocalVariable(arg: 1, scope: !136, type: !18)
// CHECK:STDOUT: !141 = !DILocation(line: 66, column: 17, scope: !136)
// CHECK:STDOUT: !142 = !DILocation(line: 66, column: 5, scope: !136)
// CHECK:STDOUT: !143 = distinct !DISubprogram(name: "AsInt", linkageName: "_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8", scope: null, file: !92, line: 59, type: !137, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !144)
// CHECK:STDOUT: !144 = !{!145}
// CHECK:STDOUT: !145 = !DILocalVariable(arg: 1, scope: !143, type: !18)
// CHECK:STDOUT: !146 = !DILocation(line: 59, column: 28, scope: !143)
// CHECK:STDOUT:
