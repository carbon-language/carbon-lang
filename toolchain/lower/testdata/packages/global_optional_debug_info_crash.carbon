// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
// EXTRA-ARGS: --expect-full-prelude
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/packages/global_optional_debug_info_crash.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/packages/global_optional_debug_info_crash.carbon

// --- file1.carbon
package TestPackage;

var globalVal1: Core.Optional(i32) = Core.Optional(i32).None();

fn GenericFn[T:! Core.Destroy](ref x: T) {
  x.Op();
}

// --- file2.impl.carbon
impl package TestPackage;

fn Helper() {
  // CHECK:STDERR: file2.impl.carbon:[[@LINE+4]]:7: warning: binding `localVal` unused [UnusedBinding]
  // CHECK:STDERR:   var localVal: Core.Optional(i32) = Core.Optional(i32).None();
  // CHECK:STDERR:       ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var localVal: Core.Optional(i32) = Core.Optional(i32).None();
  GenericFn(ref globalVal1);
}

// CHECK:STDOUT: ; ModuleID = 'file1.carbon'
// CHECK:STDOUT: source_filename = "file1.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @_CglobalVal1.TestPackage = global <{ i32, i1 }> zeroinitializer
// CHECK:STDOUT: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 0, ptr @_C__global_init.TestPackage, ptr null }]
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.5e27612b9dd31a14:core.Destroy.Core"(ptr %self) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define internal void @_C__global_init.TestPackage() #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CNone.Optional.Core.0dd6752b29c80544(ptr @_CglobalVal1.TestPackage), !dbg !14
// CHECK:STDOUT:   ret void, !dbg !15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CNone.Optional.Core.0dd6752b29c80544(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !16 {
// CHECK:STDOUT:   call void @"_CNone.74becf39533834ee:OptionalStorage.Core.011c8227b9ce853f"(ptr %return), !dbg !21
// CHECK:STDOUT:   ret void, !dbg !22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNone.74becf39533834ee:OptionalStorage.Core.011c8227b9ce853f"(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !23 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !24
// CHECK:STDOUT:   store i8 0, ptr %has_value, align 1, !dbg !24
// CHECK:STDOUT:   ret void, !dbg !25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "file1.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Op", linkageName: "_COp.5e27612b9dd31a14:core.Destroy.Core", scope: null, file: !3, line: 3, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !8)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null, !7}
// CHECK:STDOUT: !7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !8 = !{!9}
// CHECK:STDOUT: !9 = !DILocalVariable(arg: 1, scope: !4, type: !7)
// CHECK:STDOUT: !10 = !DILocation(line: 3, column: 17, scope: !4)
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "__global_init", linkageName: "_C__global_init.TestPackage", scope: null, file: !3, type: !12, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !12 = !DISubroutineType(types: !13)
// CHECK:STDOUT: !13 = !{null}
// CHECK:STDOUT: !14 = !DILocation(line: 3, column: 38, scope: !11)
// CHECK:STDOUT: !15 = !DILocation(line: 0, scope: !11)
// CHECK:STDOUT: !16 = distinct !DISubprogram(name: "None", linkageName: "_CNone.Optional.Core.0dd6752b29c80544", scope: null, file: !17, line: 30, type: !18, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !17 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !18 = !DISubroutineType(types: !19)
// CHECK:STDOUT: !19 = !{!20}
// CHECK:STDOUT: !20 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !21 = !DILocation(line: 31, column: 12, scope: !16)
// CHECK:STDOUT: !22 = !DILocation(line: 31, column: 5, scope: !16)
// CHECK:STDOUT: !23 = distinct !DISubprogram(name: "None", linkageName: "_CNone.74becf39533834ee:OptionalStorage.Core.011c8227b9ce853f", scope: null, file: !17, line: 125, type: !18, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !24 = !DILocation(line: 128, column: 5, scope: !23)
// CHECK:STDOUT: !25 = !DILocation(line: 129, column: 5, scope: !23)
// CHECK:STDOUT: ; ModuleID = 'file2.impl.carbon'
// CHECK:STDOUT: source_filename = "file2.impl.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @_CglobalVal1.TestPackage = external global <{ i32, i1 }>
// CHECK:STDOUT: @_CglobalVal1.TestPackage.1 = external global <{ i32, i1 }>
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CHelper.TestPackage() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %localVal.var = alloca <{ i32, i1 }>, align 4, !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %localVal.var), !dbg !7
// CHECK:STDOUT:   call void @_CNone.Optional.Core.2e41ac48200427bd(ptr %localVal.var), !dbg !8
// CHECK:STDOUT:   call void @_CGenericFn.TestPackage.24c6b5468c18afc1(ptr @_CglobalVal1.TestPackage), !dbg !9
// CHECK:STDOUT:   call void @"_COp.1e398a2e7d739d84:core.Destroy.Core"(ptr %localVal.var), !dbg !7
// CHECK:STDOUT:   ret void, !dbg !10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.5e27612b9dd31a14:core.Destroy.Core"(ptr %self) #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.0fc9634857315db2:core.Destroy.Core"(ptr %self) #0 !dbg !18 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.69700c1fc1b2744c:core.Destroy.Core"(ptr %self) #0 !dbg !22 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.07d96243c887cec8:core.Destroy.Core"(ptr %self) #0 !dbg !29 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.1e398a2e7d739d84:core.Destroy.Core"(ptr %self) #0 !dbg !33 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CNone.Optional.Core.2e41ac48200427bd(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !37 {
// CHECK:STDOUT:   call void @"_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.4eb825cacec05555"(ptr %return), !dbg !41
// CHECK:STDOUT:   ret void, !dbg !42
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CGenericFn.TestPackage.24c6b5468c18afc1(ptr %x) #0 !dbg !43 {
// CHECK:STDOUT:   call void @"_COp.1e398a2e7d739d84:core.Destroy.Core"(ptr %x), !dbg !47
// CHECK:STDOUT:   ret void, !dbg !48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.4eb825cacec05555"(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !49 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !50
// CHECK:STDOUT:   store i8 0, ptr %has_value, align 1, !dbg !50
// CHECK:STDOUT:   ret void, !dbg !51
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "file2.impl.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Helper", linkageName: "_CHelper.TestPackage", scope: null, file: !3, line: 3, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DILocation(line: 8, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 8, column: 38, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 9, column: 3, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 3, column: 1, scope: !4)
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "Op", linkageName: "_COp.5e27612b9dd31a14:core.Destroy.Core", scope: null, file: !3, line: 8, type: !12, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !15)
// CHECK:STDOUT: !12 = !DISubroutineType(types: !13)
// CHECK:STDOUT: !13 = !{null, !14}
// CHECK:STDOUT: !14 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !15 = !{!16}
// CHECK:STDOUT: !16 = !DILocalVariable(arg: 1, scope: !11, type: !14)
// CHECK:STDOUT: !17 = !DILocation(line: 8, column: 17, scope: !11)
// CHECK:STDOUT: !18 = distinct !DISubprogram(name: "Op", linkageName: "_COp.0fc9634857315db2:core.Destroy.Core", scope: null, file: !3, line: 9, type: !12, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !19)
// CHECK:STDOUT: !19 = !{!20}
// CHECK:STDOUT: !20 = !DILocalVariable(arg: 1, scope: !18, type: !14)
// CHECK:STDOUT: !21 = !DILocation(line: 9, column: 3, scope: !18)
// CHECK:STDOUT: !22 = distinct !DISubprogram(name: "Op", linkageName: "_COp.69700c1fc1b2744c:core.Destroy.Core", scope: null, file: !3, line: 9, type: !23, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !26)
// CHECK:STDOUT: !23 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !24 = !{null, !25}
// CHECK:STDOUT: !25 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !26 = !{!27}
// CHECK:STDOUT: !27 = !DILocalVariable(arg: 1, scope: !22, type: !25)
// CHECK:STDOUT: !28 = !DILocation(line: 9, column: 3, scope: !22)
// CHECK:STDOUT: !29 = distinct !DISubprogram(name: "Op", linkageName: "_COp.07d96243c887cec8:core.Destroy.Core", scope: null, file: !3, line: 9, type: !23, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !30)
// CHECK:STDOUT: !30 = !{!31}
// CHECK:STDOUT: !31 = !DILocalVariable(arg: 1, scope: !29, type: !25)
// CHECK:STDOUT: !32 = !DILocation(line: 9, column: 3, scope: !29)
// CHECK:STDOUT: !33 = distinct !DISubprogram(name: "Op", linkageName: "_COp.1e398a2e7d739d84:core.Destroy.Core", scope: null, file: !3, line: 9, type: !23, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !34)
// CHECK:STDOUT: !34 = !{!35}
// CHECK:STDOUT: !35 = !DILocalVariable(arg: 1, scope: !33, type: !25)
// CHECK:STDOUT: !36 = !DILocation(line: 9, column: 3, scope: !33)
// CHECK:STDOUT: !37 = distinct !DISubprogram(name: "None", linkageName: "_CNone.Optional.Core.2e41ac48200427bd", scope: null, file: !38, line: 30, type: !39, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !38 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !39 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !40 = !{!25}
// CHECK:STDOUT: !41 = !DILocation(line: 31, column: 12, scope: !37)
// CHECK:STDOUT: !42 = !DILocation(line: 31, column: 5, scope: !37)
// CHECK:STDOUT: !43 = distinct !DISubprogram(name: "GenericFn", linkageName: "_CGenericFn.TestPackage.24c6b5468c18afc1", scope: null, file: !44, line: 5, type: !23, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !45)
// CHECK:STDOUT: !44 = !DIFile(filename: "file1.carbon", directory: "")
// CHECK:STDOUT: !45 = !{!46}
// CHECK:STDOUT: !46 = !DILocalVariable(arg: 1, scope: !43, type: !25)
// CHECK:STDOUT: !47 = !DILocation(line: 6, column: 3, scope: !43)
// CHECK:STDOUT: !48 = !DILocation(line: 5, column: 1, scope: !43)
// CHECK:STDOUT: !49 = distinct !DISubprogram(name: "None", linkageName: "_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.4eb825cacec05555", scope: null, file: !38, line: 125, type: !39, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !50 = !DILocation(line: 128, column: 5, scope: !49)
// CHECK:STDOUT: !51 = !DILocation(line: 129, column: 5, scope: !49)
