// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/impl/import_thunk.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/impl/import_thunk.carbon

// --- thunk.carbon

library "[[@TEST_NAME]]";

class A { var a: i32; }
class B { var b: i32; }
class C { var c: i32; }

impl A as Core.ImplicitAs(B) {
  fn Convert(self: A) -> B { return {.b = self.a}; }
}
impl B as Core.ImplicitAs(C) {
  fn Convert(self: B) -> C { return {.c = self.b}; }
}

interface I {
  fn F(a: A) -> C;
}

impl () as I {
  fn F(b: B) -> B { return {.b = b.b}; }
}

// --- call_thunk.carbon

library "[[@TEST_NAME]]";

import library "thunk";

fn Test(a: A) -> C {
  return ().(I.F)(a);
}

// --- thunk_for_imported_interface.carbon

library "[[@TEST_NAME]]";

import library "thunk";

class X {}

impl X as I {
  fn F(b: B) -> B { return {.b = b.b}; }
}

// --- call_thunk_for_imported_interface.carbon

library "[[@TEST_NAME]]";

import library "thunk";
import library "thunk_for_imported_interface";

fn Test(a: A) -> C {
  return X.(I.F)(a);
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'thunk.carbon'
// CHECK:STDOUT: source_filename = "thunk.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CConvert.A.Main:ImplicitAs.a6a60dfee2d7d273.Core"(ptr sret({ i32 }) %return, ptr %self) #0 !dbg !26 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_47.1.a = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !31
// CHECK:STDOUT:   %.loc9_47.2 = load i32, ptr %.loc9_47.1.a, align 4, !dbg !31
// CHECK:STDOUT:   %.loc9_49.2.b = getelementptr inbounds nuw { i32 }, ptr %return, i32 0, i32 0, !dbg !32
// CHECK:STDOUT:   store i32 %.loc9_47.2, ptr %.loc9_49.2.b, align 4, !dbg !32
// CHECK:STDOUT:   ret void, !dbg !33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CConvert.B.Main:ImplicitAs.9522ff908e97e267.Core"(ptr sret({ i32 }) %return, ptr %self) #0 !dbg !35 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc12_47.1.b = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !40
// CHECK:STDOUT:   %.loc12_47.2 = load i32, ptr %.loc12_47.1.b, align 4, !dbg !40
// CHECK:STDOUT:   %.loc12_49.2.c = getelementptr inbounds nuw { i32 }, ptr %return, i32 0, i32 0, !dbg !41
// CHECK:STDOUT:   store i32 %.loc12_47.2, ptr %.loc12_49.2.c, align 4, !dbg !41
// CHECK:STDOUT:   ret void, !dbg !42
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CF.61ea2aba74ab3bf1:I.Main"(ptr sret({ i32 }) %return, ptr %b) #0 !dbg !44 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc20_35.1.b = getelementptr inbounds nuw { i32 }, ptr %b, i32 0, i32 0, !dbg !49
// CHECK:STDOUT:   %.loc20_35.2 = load i32, ptr %.loc20_35.1.b, align 4, !dbg !49
// CHECK:STDOUT:   %.loc20_37.2.b = getelementptr inbounds nuw { i32 }, ptr %return, i32 0, i32 0, !dbg !50
// CHECK:STDOUT:   store i32 %.loc20_35.2, ptr %.loc20_37.2.b, align 4, !dbg !50
// CHECK:STDOUT:   ret void, !dbg !51
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_CF:thunk:I.6a2feaaa6331eb92.Main:61ea2aba74ab3bf1:I.Main"(ptr sret({ i32 }) %return, ptr %a) #1 !dbg !53 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc20_19.1.temp = alloca { i32 }, align 4, !dbg !56
// CHECK:STDOUT:   %.4.temp = alloca { i32 }, align 4, !dbg !55
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc20_19.1.temp), !dbg !56
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.4.temp), !dbg !55
// CHECK:STDOUT:   call void @"_CConvert.A.Main:ImplicitAs.a6a60dfee2d7d273.Core"(ptr %.4.temp, ptr %a), !dbg !55
// CHECK:STDOUT:   call void @"_CF.61ea2aba74ab3bf1:I.Main"(ptr %.loc20_19.1.temp, ptr %.4.temp), !dbg !56
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.9522ff908e97e267.Core"(ptr %return, ptr %.loc20_19.1.temp), !dbg !56
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.c8fa36f2e394941a(ptr %.loc20_19.1.temp), !dbg !56
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.c8fa36f2e394941a(ptr %.4.temp), !dbg !55
// CHECK:STDOUT:   ret void, !dbg !56
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.0bfe8bb35a2876b2:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !58 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc20_19.1.b = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !60
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.3b18dc1d3d647932(ptr %.loc20_19.1.b), !dbg !60
// CHECK:STDOUT:   ret void, !dbg !60
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.8c4b83ecd900f1ea:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !64 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc20_19.1.b = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !66
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.3b18dc1d3d647932(ptr %.loc20_19.1.b), !dbg !66
// CHECK:STDOUT:   ret void, !dbg !66
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.c8fa36f2e394941a(ptr %t) #0 !dbg !68 {
// CHECK:STDOUT:   call void @"_COp.8c4b83ecd900f1ea:core.SubobjectDestroy.Core"(ptr %t), !dbg !72
// CHECK:STDOUT:   ret void, !dbg !73
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.3b18dc1d3d647932(ptr %t) #0 !dbg !75 {
// CHECK:STDOUT:   ret void, !dbg !80
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.c8fa36f2e394941a, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.3b18dc1d3d647932, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !18 = !DIFile(filename: "thunk.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{!21, !21}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !24 = !{null, !21}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.A.Main:ImplicitAs.a6a60dfee2d7d273.Core", scope: null, file: !18, line: 9, type: !23, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !34)
// CHECK:STDOUT: !27 = !DILocalVariable(arg: 1, scope: !26, type: !21)
// CHECK:STDOUT: !31 = !DILocation(line: 9, column: 43, scope: !26)
// CHECK:STDOUT: !32 = !DILocation(line: 9, column: 37, scope: !26)
// CHECK:STDOUT: !33 = !DILocation(line: 9, column: 30, scope: !26)
// CHECK:STDOUT: !34 = !{!27}
// CHECK:STDOUT: !35 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.B.Main:ImplicitAs.9522ff908e97e267.Core", scope: null, file: !18, line: 12, type: !23, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !43)
// CHECK:STDOUT: !36 = !DILocalVariable(arg: 1, scope: !35, type: !21)
// CHECK:STDOUT: !40 = !DILocation(line: 12, column: 43, scope: !35)
// CHECK:STDOUT: !41 = !DILocation(line: 12, column: 37, scope: !35)
// CHECK:STDOUT: !42 = !DILocation(line: 12, column: 30, scope: !35)
// CHECK:STDOUT: !43 = !{!36}
// CHECK:STDOUT: !44 = distinct !DISubprogram(name: "F", linkageName: "_CF.61ea2aba74ab3bf1:I.Main", scope: null, file: !18, line: 20, type: !23, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !52)
// CHECK:STDOUT: !45 = !DILocalVariable(arg: 1, scope: !44, type: !21)
// CHECK:STDOUT: !49 = !DILocation(line: 20, column: 34, scope: !44)
// CHECK:STDOUT: !50 = !DILocation(line: 20, column: 28, scope: !44)
// CHECK:STDOUT: !51 = !DILocation(line: 20, column: 21, scope: !44)
// CHECK:STDOUT: !52 = !{!45}
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "F", linkageName: "_CF:thunk:I.6a2feaaa6331eb92.Main:61ea2aba74ab3bf1:I.Main", scope: null, file: !18, line: 20, type: !23, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !57)
// CHECK:STDOUT: !54 = !DILocalVariable(arg: 1, scope: !53, type: !21)
// CHECK:STDOUT: !55 = !DILocation(line: 0, scope: !53)
// CHECK:STDOUT: !56 = !DILocation(line: 20, column: 3, scope: !53)
// CHECK:STDOUT: !57 = !{!54}
// CHECK:STDOUT: !58 = distinct !DISubprogram(name: "Op", linkageName: "_COp.0bfe8bb35a2876b2:core.SubobjectDestroy.Core", scope: null, file: !18, line: 20, type: !25, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !63)
// CHECK:STDOUT: !59 = !DILocalVariable(arg: 1, scope: !58, type: !21)
// CHECK:STDOUT: !60 = !DILocation(line: 20, column: 3, scope: !58)
// CHECK:STDOUT: !61 = !{null, !20}
// CHECK:STDOUT: !62 = !DISubroutineType(types: !61)
// CHECK:STDOUT: !63 = !{!59}
// CHECK:STDOUT: !64 = distinct !DISubprogram(name: "Op", linkageName: "_COp.8c4b83ecd900f1ea:core.SubobjectDestroy.Core", scope: null, file: !18, line: 20, type: !25, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !67)
// CHECK:STDOUT: !65 = !DILocalVariable(arg: 1, scope: !64, type: !21)
// CHECK:STDOUT: !66 = !DILocation(line: 20, column: 3, scope: !64)
// CHECK:STDOUT: !67 = !{!65}
// CHECK:STDOUT: !68 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.c8fa36f2e394941a", scope: null, file: !10, line: 26, type: !25, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !74)
// CHECK:STDOUT: !69 = !DILocalVariable(arg: 1, scope: !68, type: !21)
// CHECK:STDOUT: !72 = !DILocation(line: 28, column: 5, scope: !68)
// CHECK:STDOUT: !73 = !DILocation(line: 26, column: 1, scope: !68)
// CHECK:STDOUT: !74 = !{!69}
// CHECK:STDOUT: !75 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.3b18dc1d3d647932", scope: null, file: !10, line: 26, type: !62, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !81)
// CHECK:STDOUT: !76 = !DILocalVariable(arg: 1, scope: !75, type: !20)
// CHECK:STDOUT: !80 = !DILocation(line: 26, column: 1, scope: !75)
// CHECK:STDOUT: !81 = !{!76}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'call_thunk.carbon'
// CHECK:STDOUT: source_filename = "call_thunk.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTest.Main(ptr sret({ i32 }) %return, ptr %a) #0 !dbg !84 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_20.1.temp = alloca { i32 }, align 4, !dbg !90
// CHECK:STDOUT:   %.loc7_20.2.temp = alloca { i32 }, align 4, !dbg !90
// CHECK:STDOUT:   %.loc7_19.1.temp = alloca { i32 }, align 4, !dbg !89
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_20.1.temp), !dbg !90
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_20.2.temp), !dbg !90
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_19.1.temp), !dbg !89
// CHECK:STDOUT:   call void @"_CConvert.A.Main:ImplicitAs.a6a60dfee2d7d273.Core"(ptr %.loc7_19.1.temp, ptr %a), !dbg !89
// CHECK:STDOUT:   call void @"_CF.61ea2aba74ab3bf1:I.Main"(ptr %.loc7_20.2.temp, ptr %.loc7_19.1.temp), !dbg !90
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.9522ff908e97e267.Core"(ptr %return, ptr %.loc7_20.2.temp), !dbg !91
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.c8fa36f2e394941a(ptr %.loc7_20.2.temp), !dbg !90
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.c8fa36f2e394941a(ptr %.loc7_19.1.temp), !dbg !89
// CHECK:STDOUT:   ret void, !dbg !91
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_CF:thunk:I.f5faeaa774ca256e.Main:61ea2aba74ab3bf1:I.Main"(ptr sret({ i32 }), ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_CF.61ea2aba74ab3bf1:I.Main"(ptr sret({ i32 }), ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_CConvert.A.Main:ImplicitAs.a6a60dfee2d7d273.Core"(ptr sret({ i32 }), ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_CConvert.B.Main:ImplicitAs.9522ff908e97e267.Core"(ptr sret({ i32 }), ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.0bfe8bb35a2876b2:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !93 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_20.1.b = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !95
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.3b18dc1d3d647932(ptr %.loc7_20.1.b), !dbg !95
// CHECK:STDOUT:   ret void, !dbg !95
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.8c4b83ecd900f1ea:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !97 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_20.1.b = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !99
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.3b18dc1d3d647932(ptr %.loc7_20.1.b), !dbg !99
// CHECK:STDOUT:   ret void, !dbg !99
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.c8fa36f2e394941a(ptr %t) #0 !dbg !101 {
// CHECK:STDOUT:   call void @"_COp.8c4b83ecd900f1ea:core.SubobjectDestroy.Core"(ptr %t), !dbg !105
// CHECK:STDOUT:   ret void, !dbg !106
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.3b18dc1d3d647932(ptr %t) #0 !dbg !108 {
// CHECK:STDOUT:   ret void, !dbg !113
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.c8fa36f2e394941a, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.3b18dc1d3d647932, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!83}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{!21, !21}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !24 = !{null, !21}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !61 = !{null, !20}
// CHECK:STDOUT: !62 = !DISubroutineType(types: !61)
// CHECK:STDOUT: !82 = !DIFile(filename: "call_thunk.carbon", directory: "")
// CHECK:STDOUT: !83 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !82, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !84 = distinct !DISubprogram(name: "Test", linkageName: "_CTest.Main", scope: null, file: !82, line: 6, type: !23, spFlags: DISPFlagDefinition, unit: !83, retainedNodes: !92)
// CHECK:STDOUT: !85 = !DILocalVariable(arg: 1, scope: !84, type: !21)
// CHECK:STDOUT: !89 = !DILocation(line: 7, column: 19, scope: !84)
// CHECK:STDOUT: !90 = !DILocation(line: 7, column: 10, scope: !84)
// CHECK:STDOUT: !91 = !DILocation(line: 7, column: 3, scope: !84)
// CHECK:STDOUT: !92 = !{!85}
// CHECK:STDOUT: !93 = distinct !DISubprogram(name: "Op", linkageName: "_COp.0bfe8bb35a2876b2:core.SubobjectDestroy.Core", scope: null, file: !82, line: 7, type: !25, spFlags: DISPFlagDefinition, unit: !83, retainedNodes: !96)
// CHECK:STDOUT: !94 = !DILocalVariable(arg: 1, scope: !93, type: !21)
// CHECK:STDOUT: !95 = !DILocation(line: 7, column: 10, scope: !93)
// CHECK:STDOUT: !96 = !{!94}
// CHECK:STDOUT: !97 = distinct !DISubprogram(name: "Op", linkageName: "_COp.8c4b83ecd900f1ea:core.SubobjectDestroy.Core", scope: null, file: !82, line: 7, type: !25, spFlags: DISPFlagDefinition, unit: !83, retainedNodes: !100)
// CHECK:STDOUT: !98 = !DILocalVariable(arg: 1, scope: !97, type: !21)
// CHECK:STDOUT: !99 = !DILocation(line: 7, column: 10, scope: !97)
// CHECK:STDOUT: !100 = !{!98}
// CHECK:STDOUT: !101 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.c8fa36f2e394941a", scope: null, file: !10, line: 26, type: !25, spFlags: DISPFlagDefinition, unit: !83, retainedNodes: !107)
// CHECK:STDOUT: !102 = !DILocalVariable(arg: 1, scope: !101, type: !21)
// CHECK:STDOUT: !105 = !DILocation(line: 28, column: 5, scope: !101)
// CHECK:STDOUT: !106 = !DILocation(line: 26, column: 1, scope: !101)
// CHECK:STDOUT: !107 = !{!102}
// CHECK:STDOUT: !108 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.3b18dc1d3d647932", scope: null, file: !10, line: 26, type: !62, spFlags: DISPFlagDefinition, unit: !83, retainedNodes: !114)
// CHECK:STDOUT: !109 = !DILocalVariable(arg: 1, scope: !108, type: !20)
// CHECK:STDOUT: !113 = !DILocation(line: 26, column: 1, scope: !108)
// CHECK:STDOUT: !114 = !{!109}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'thunk_for_imported_interface.carbon'
// CHECK:STDOUT: source_filename = "thunk_for_imported_interface.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CF.X.Main:I.Main"(ptr sret({ i32 }) %return, ptr %b) #0 !dbg !117 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_35.1.b = getelementptr inbounds nuw { i32 }, ptr %b, i32 0, i32 0, !dbg !122
// CHECK:STDOUT:   %.loc9_35.2 = load i32, ptr %.loc9_35.1.b, align 4, !dbg !122
// CHECK:STDOUT:   %.loc9_37.2.b = getelementptr inbounds nuw { i32 }, ptr %return, i32 0, i32 0, !dbg !123
// CHECK:STDOUT:   store i32 %.loc9_35.2, ptr %.loc9_37.2.b, align 4, !dbg !123
// CHECK:STDOUT:   ret void, !dbg !124
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_CF:thunk:I.d0cc85b6c89517e3.Main:X.Main:I.Main"(ptr sret({ i32 }) %return, ptr %a) #1 !dbg !126 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_19.1.temp = alloca { i32 }, align 4, !dbg !129
// CHECK:STDOUT:   %.4.temp = alloca { i32 }, align 4, !dbg !128
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc9_19.1.temp), !dbg !129
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.4.temp), !dbg !128
// CHECK:STDOUT:   call void @"_CConvert.A.Main:ImplicitAs.a6a60dfee2d7d273.Core"(ptr %.4.temp, ptr %a), !dbg !128
// CHECK:STDOUT:   call void @"_CF.X.Main:I.Main"(ptr %.loc9_19.1.temp, ptr %.4.temp), !dbg !129
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.9522ff908e97e267.Core"(ptr %return, ptr %.loc9_19.1.temp), !dbg !129
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.c8fa36f2e394941a(ptr %.loc9_19.1.temp), !dbg !129
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.c8fa36f2e394941a(ptr %.4.temp), !dbg !128
// CHECK:STDOUT:   ret void, !dbg !129
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_CConvert.A.Main:ImplicitAs.a6a60dfee2d7d273.Core"(ptr sret({ i32 }), ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_CConvert.B.Main:ImplicitAs.9522ff908e97e267.Core"(ptr sret({ i32 }), ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.0bfe8bb35a2876b2:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !131 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_19.1.b = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !133
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.3b18dc1d3d647932(ptr %.loc9_19.1.b), !dbg !133
// CHECK:STDOUT:   ret void, !dbg !133
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.8c4b83ecd900f1ea:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !135 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_19.1.b = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !137
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.3b18dc1d3d647932(ptr %.loc9_19.1.b), !dbg !137
// CHECK:STDOUT:   ret void, !dbg !137
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.c8fa36f2e394941a(ptr %t) #0 !dbg !139 {
// CHECK:STDOUT:   call void @"_COp.8c4b83ecd900f1ea:core.SubobjectDestroy.Core"(ptr %t), !dbg !143
// CHECK:STDOUT:   ret void, !dbg !144
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.3b18dc1d3d647932(ptr %t) #0 !dbg !146 {
// CHECK:STDOUT:   ret void, !dbg !151
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.c8fa36f2e394941a, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.3b18dc1d3d647932, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!116}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{!21, !21}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !24 = !{null, !21}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !61 = !{null, !20}
// CHECK:STDOUT: !62 = !DISubroutineType(types: !61)
// CHECK:STDOUT: !115 = !DIFile(filename: "thunk_for_imported_interface.carbon", directory: "")
// CHECK:STDOUT: !116 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !115, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !117 = distinct !DISubprogram(name: "F", linkageName: "_CF.X.Main:I.Main", scope: null, file: !115, line: 9, type: !23, spFlags: DISPFlagDefinition, unit: !116, retainedNodes: !125)
// CHECK:STDOUT: !118 = !DILocalVariable(arg: 1, scope: !117, type: !21)
// CHECK:STDOUT: !122 = !DILocation(line: 9, column: 34, scope: !117)
// CHECK:STDOUT: !123 = !DILocation(line: 9, column: 28, scope: !117)
// CHECK:STDOUT: !124 = !DILocation(line: 9, column: 21, scope: !117)
// CHECK:STDOUT: !125 = !{!118}
// CHECK:STDOUT: !126 = distinct !DISubprogram(name: "F", linkageName: "_CF:thunk:I.d0cc85b6c89517e3.Main:X.Main:I.Main", scope: null, file: !115, line: 9, type: !23, spFlags: DISPFlagDefinition, unit: !116, retainedNodes: !130)
// CHECK:STDOUT: !127 = !DILocalVariable(arg: 1, scope: !126, type: !21)
// CHECK:STDOUT: !128 = !DILocation(line: 0, scope: !126)
// CHECK:STDOUT: !129 = !DILocation(line: 9, column: 3, scope: !126)
// CHECK:STDOUT: !130 = !{!127}
// CHECK:STDOUT: !131 = distinct !DISubprogram(name: "Op", linkageName: "_COp.0bfe8bb35a2876b2:core.SubobjectDestroy.Core", scope: null, file: !115, line: 9, type: !25, spFlags: DISPFlagDefinition, unit: !116, retainedNodes: !134)
// CHECK:STDOUT: !132 = !DILocalVariable(arg: 1, scope: !131, type: !21)
// CHECK:STDOUT: !133 = !DILocation(line: 9, column: 3, scope: !131)
// CHECK:STDOUT: !134 = !{!132}
// CHECK:STDOUT: !135 = distinct !DISubprogram(name: "Op", linkageName: "_COp.8c4b83ecd900f1ea:core.SubobjectDestroy.Core", scope: null, file: !115, line: 9, type: !25, spFlags: DISPFlagDefinition, unit: !116, retainedNodes: !138)
// CHECK:STDOUT: !136 = !DILocalVariable(arg: 1, scope: !135, type: !21)
// CHECK:STDOUT: !137 = !DILocation(line: 9, column: 3, scope: !135)
// CHECK:STDOUT: !138 = !{!136}
// CHECK:STDOUT: !139 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.c8fa36f2e394941a", scope: null, file: !10, line: 26, type: !25, spFlags: DISPFlagDefinition, unit: !116, retainedNodes: !145)
// CHECK:STDOUT: !140 = !DILocalVariable(arg: 1, scope: !139, type: !21)
// CHECK:STDOUT: !143 = !DILocation(line: 28, column: 5, scope: !139)
// CHECK:STDOUT: !144 = !DILocation(line: 26, column: 1, scope: !139)
// CHECK:STDOUT: !145 = !{!140}
// CHECK:STDOUT: !146 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.3b18dc1d3d647932", scope: null, file: !10, line: 26, type: !62, spFlags: DISPFlagDefinition, unit: !116, retainedNodes: !152)
// CHECK:STDOUT: !147 = !DILocalVariable(arg: 1, scope: !146, type: !20)
// CHECK:STDOUT: !151 = !DILocation(line: 26, column: 1, scope: !146)
// CHECK:STDOUT: !152 = !{!147}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'call_thunk_for_imported_interface.carbon'
// CHECK:STDOUT: source_filename = "call_thunk_for_imported_interface.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTest.Main(ptr sret({ i32 }) %return, ptr %a) #0 !dbg !155 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc8_19.1.temp = alloca { i32 }, align 4, !dbg !161
// CHECK:STDOUT:   %.loc8_19.2.temp = alloca { i32 }, align 4, !dbg !161
// CHECK:STDOUT:   %.loc8_18.1.temp = alloca { i32 }, align 4, !dbg !160
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc8_19.1.temp), !dbg !161
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc8_19.2.temp), !dbg !161
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc8_18.1.temp), !dbg !160
// CHECK:STDOUT:   call void @"_CConvert.A.Main:ImplicitAs.a6a60dfee2d7d273.Core"(ptr %.loc8_18.1.temp, ptr %a), !dbg !160
// CHECK:STDOUT:   call void @"_CF.X.Main:I.Main"(ptr %.loc8_19.2.temp, ptr %.loc8_18.1.temp), !dbg !161
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.9522ff908e97e267.Core"(ptr %return, ptr %.loc8_19.2.temp), !dbg !162
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.c8fa36f2e394941a(ptr %.loc8_19.2.temp), !dbg !161
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.c8fa36f2e394941a(ptr %.loc8_18.1.temp), !dbg !160
// CHECK:STDOUT:   ret void, !dbg !162
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_CF:thunk:I.4980a17f6b368c72.Main:X.Main:I.Main"(ptr sret({ i32 }), ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_CF.X.Main:I.Main"(ptr sret({ i32 }), ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_CConvert.A.Main:ImplicitAs.a6a60dfee2d7d273.Core"(ptr sret({ i32 }), ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_CConvert.B.Main:ImplicitAs.9522ff908e97e267.Core"(ptr sret({ i32 }), ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.0bfe8bb35a2876b2:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !164 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc8_19.1.b = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !166
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.3b18dc1d3d647932(ptr %.loc8_19.1.b), !dbg !166
// CHECK:STDOUT:   ret void, !dbg !166
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.8c4b83ecd900f1ea:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !168 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc8_19.1.b = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !170
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.3b18dc1d3d647932(ptr %.loc8_19.1.b), !dbg !170
// CHECK:STDOUT:   ret void, !dbg !170
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.c8fa36f2e394941a(ptr %t) #0 !dbg !172 {
// CHECK:STDOUT:   call void @"_COp.8c4b83ecd900f1ea:core.SubobjectDestroy.Core"(ptr %t), !dbg !176
// CHECK:STDOUT:   ret void, !dbg !177
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.3b18dc1d3d647932(ptr %t) #0 !dbg !179 {
// CHECK:STDOUT:   ret void, !dbg !184
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.c8fa36f2e394941a, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.3b18dc1d3d647932, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!154}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{!21, !21}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !24 = !{null, !21}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !61 = !{null, !20}
// CHECK:STDOUT: !62 = !DISubroutineType(types: !61)
// CHECK:STDOUT: !153 = !DIFile(filename: "call_thunk_for_imported_interface.carbon", directory: "")
// CHECK:STDOUT: !154 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !153, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !155 = distinct !DISubprogram(name: "Test", linkageName: "_CTest.Main", scope: null, file: !153, line: 7, type: !23, spFlags: DISPFlagDefinition, unit: !154, retainedNodes: !163)
// CHECK:STDOUT: !156 = !DILocalVariable(arg: 1, scope: !155, type: !21)
// CHECK:STDOUT: !160 = !DILocation(line: 8, column: 18, scope: !155)
// CHECK:STDOUT: !161 = !DILocation(line: 8, column: 10, scope: !155)
// CHECK:STDOUT: !162 = !DILocation(line: 8, column: 3, scope: !155)
// CHECK:STDOUT: !163 = !{!156}
// CHECK:STDOUT: !164 = distinct !DISubprogram(name: "Op", linkageName: "_COp.0bfe8bb35a2876b2:core.SubobjectDestroy.Core", scope: null, file: !153, line: 8, type: !25, spFlags: DISPFlagDefinition, unit: !154, retainedNodes: !167)
// CHECK:STDOUT: !165 = !DILocalVariable(arg: 1, scope: !164, type: !21)
// CHECK:STDOUT: !166 = !DILocation(line: 8, column: 10, scope: !164)
// CHECK:STDOUT: !167 = !{!165}
// CHECK:STDOUT: !168 = distinct !DISubprogram(name: "Op", linkageName: "_COp.8c4b83ecd900f1ea:core.SubobjectDestroy.Core", scope: null, file: !153, line: 8, type: !25, spFlags: DISPFlagDefinition, unit: !154, retainedNodes: !171)
// CHECK:STDOUT: !169 = !DILocalVariable(arg: 1, scope: !168, type: !21)
// CHECK:STDOUT: !170 = !DILocation(line: 8, column: 10, scope: !168)
// CHECK:STDOUT: !171 = !{!169}
// CHECK:STDOUT: !172 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.c8fa36f2e394941a", scope: null, file: !10, line: 26, type: !25, spFlags: DISPFlagDefinition, unit: !154, retainedNodes: !178)
// CHECK:STDOUT: !173 = !DILocalVariable(arg: 1, scope: !172, type: !21)
// CHECK:STDOUT: !176 = !DILocation(line: 28, column: 5, scope: !172)
// CHECK:STDOUT: !177 = !DILocation(line: 26, column: 1, scope: !172)
// CHECK:STDOUT: !178 = !{!173}
// CHECK:STDOUT: !179 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.3b18dc1d3d647932", scope: null, file: !10, line: 26, type: !62, spFlags: DISPFlagDefinition, unit: !154, retainedNodes: !185)
// CHECK:STDOUT: !180 = !DILocalVariable(arg: 1, scope: !179, type: !20)
// CHECK:STDOUT: !184 = !DILocation(line: 26, column: 1, scope: !179)
// CHECK:STDOUT: !185 = !{!180}
// CHECK:STDOUT:
