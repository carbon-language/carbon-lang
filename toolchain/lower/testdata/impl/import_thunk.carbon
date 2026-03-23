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
  fn Convert[self: A]() -> B { return {.b = self.a}; }
}
impl B as Core.ImplicitAs(C) {
  fn Convert[self: B]() -> C { return {.c = self.b}; }
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

// CHECK:STDOUT: ; ModuleID = 'thunk.carbon'
// CHECK:STDOUT: source_filename = "thunk.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CConvert.A.Main:ImplicitAs.707dbe5ae2414d29.Core"(ptr sret({ i32 }) %return, ptr %self) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc59.1.a = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !10
// CHECK:STDOUT:   %.loc59.2 = load i32, ptr %.loc59.1.a, align 4, !dbg !10
// CHECK:STDOUT:   %.loc61.2.b = getelementptr inbounds nuw { i32 }, ptr %return, i32 0, i32 0, !dbg !11
// CHECK:STDOUT:   store i32 %.loc59.2, ptr %.loc61.2.b, align 4, !dbg !11
// CHECK:STDOUT:   ret void, !dbg !12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CConvert.B.Main:ImplicitAs.0c4ab795cec6cb27.Core"(ptr sret({ i32 }) %return, ptr %self) #0 !dbg !13 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc93.1.b = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !16
// CHECK:STDOUT:   %.loc93.2 = load i32, ptr %.loc93.1.b, align 4, !dbg !16
// CHECK:STDOUT:   %.loc95.2.c = getelementptr inbounds nuw { i32 }, ptr %return, i32 0, i32 0, !dbg !17
// CHECK:STDOUT:   store i32 %.loc93.2, ptr %.loc95.2.c, align 4, !dbg !17
// CHECK:STDOUT:   ret void, !dbg !18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CF.61ea2aba74ab3bf1:I.Main"(ptr sret({ i32 }) %return, ptr %b) #0 !dbg !19 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc135.1.b = getelementptr inbounds nuw { i32 }, ptr %b, i32 0, i32 0, !dbg !22
// CHECK:STDOUT:   %.loc135.2 = load i32, ptr %.loc135.1.b, align 4, !dbg !22
// CHECK:STDOUT:   %.loc137.2.b = getelementptr inbounds nuw { i32 }, ptr %return, i32 0, i32 0, !dbg !23
// CHECK:STDOUT:   store i32 %.loc135.2, ptr %.loc137.2.b, align 4, !dbg !23
// CHECK:STDOUT:   ret void, !dbg !24
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define void @"_CF:thunk.61ea2aba74ab3bf1:I.Main"(ptr sret({ i32 }) %return, ptr %a) #1 !dbg !25 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc128.1.temp = alloca { i32 }, align 8, !dbg !28
// CHECK:STDOUT:   %.loc106.1.temp = alloca { i32 }, align 8, !dbg !29
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc128.1.temp), !dbg !28
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc106.1.temp), !dbg !29
// CHECK:STDOUT:   call void @"_CConvert.A.Main:ImplicitAs.707dbe5ae2414d29.Core"(ptr %.loc106.1.temp, ptr %a), !dbg !29
// CHECK:STDOUT:   call void @"_CF.61ea2aba74ab3bf1:I.Main"(ptr %.loc128.1.temp, ptr %.loc106.1.temp), !dbg !28
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.0c4ab795cec6cb27.Core"(ptr %return, ptr %.loc128.1.temp), !dbg !28
// CHECK:STDOUT:   ret void, !dbg !28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "thunk.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.A.Main:ImplicitAs.707dbe5ae2414d29.Core", scope: null, file: !3, line: 22, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !8)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7, !7}
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 8)
// CHECK:STDOUT: !8 = !{!9}
// CHECK:STDOUT: !9 = !DILocalVariable(arg: 1, scope: !4, type: !7)
// CHECK:STDOUT: !10 = !DILocation(line: 22, column: 45, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 22, column: 39, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 22, column: 32, scope: !4)
// CHECK:STDOUT: !13 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.B.Main:ImplicitAs.0c4ab795cec6cb27.Core", scope: null, file: !3, line: 25, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !14)
// CHECK:STDOUT: !14 = !{!15}
// CHECK:STDOUT: !15 = !DILocalVariable(arg: 1, scope: !13, type: !7)
// CHECK:STDOUT: !16 = !DILocation(line: 25, column: 45, scope: !13)
// CHECK:STDOUT: !17 = !DILocation(line: 25, column: 39, scope: !13)
// CHECK:STDOUT: !18 = !DILocation(line: 25, column: 32, scope: !13)
// CHECK:STDOUT: !19 = distinct !DISubprogram(name: "F", linkageName: "_CF.61ea2aba74ab3bf1:I.Main", scope: null, file: !3, line: 33, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !20)
// CHECK:STDOUT: !20 = !{!21}
// CHECK:STDOUT: !21 = !DILocalVariable(arg: 1, scope: !19, type: !7)
// CHECK:STDOUT: !22 = !DILocation(line: 33, column: 34, scope: !19)
// CHECK:STDOUT: !23 = !DILocation(line: 33, column: 28, scope: !19)
// CHECK:STDOUT: !24 = !DILocation(line: 33, column: 21, scope: !19)
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "F", linkageName: "_CF:thunk.61ea2aba74ab3bf1:I.Main", scope: null, file: !3, line: 33, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !26)
// CHECK:STDOUT: !26 = !{!27}
// CHECK:STDOUT: !27 = !DILocalVariable(arg: 1, scope: !25, type: !7)
// CHECK:STDOUT: !28 = !DILocation(line: 33, column: 3, scope: !25)
// CHECK:STDOUT: !29 = !DILocation(line: 29, column: 8, scope: !25)
// CHECK:STDOUT: ; ModuleID = 'call_thunk.carbon'
// CHECK:STDOUT: source_filename = "call_thunk.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTest.Main(ptr sret({ i32 }) %return, ptr %a) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc29.1.temp = alloca { i32 }, align 8, !dbg !10
// CHECK:STDOUT:   %.loc29.2.temp = alloca { i32 }, align 8, !dbg !10
// CHECK:STDOUT:   %.loc28.1.temp = alloca { i32 }, align 8, !dbg !11
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc29.1.temp), !dbg !10
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc29.2.temp), !dbg !10
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc28.1.temp), !dbg !11
// CHECK:STDOUT:   call void @"_CConvert.A.Main:ImplicitAs.707dbe5ae2414d29.Core"(ptr %.loc28.1.temp, ptr %a), !dbg !11
// CHECK:STDOUT:   call void @"_CF.61ea2aba74ab3bf1:I.Main"(ptr %.loc29.2.temp, ptr %.loc28.1.temp), !dbg !10
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.0c4ab795cec6cb27.Core"(ptr %return, ptr %.loc29.2.temp), !dbg !12
// CHECK:STDOUT:   ret void, !dbg !12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_CF:thunk.61ea2aba74ab3bf1:I.Main"(ptr sret({ i32 }), ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_CF.61ea2aba74ab3bf1:I.Main"(ptr sret({ i32 }), ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_CConvert.A.Main:ImplicitAs.707dbe5ae2414d29.Core"(ptr sret({ i32 }), ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_CConvert.B.Main:ImplicitAs.0c4ab795cec6cb27.Core"(ptr sret({ i32 }), ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 2, 1, 0 }
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
// CHECK:STDOUT: !3 = !DIFile(filename: "call_thunk.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Test", linkageName: "_CTest.Main", scope: null, file: !3, line: 42, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !8)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7, !7}
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 8)
// CHECK:STDOUT: !8 = !{!9}
// CHECK:STDOUT: !9 = !DILocalVariable(arg: 1, scope: !4, type: !7)
// CHECK:STDOUT: !10 = !DILocation(line: 43, column: 10, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 43, column: 19, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 43, column: 3, scope: !4)
// CHECK:STDOUT: ; ModuleID = 'thunk_for_imported_interface.carbon'
// CHECK:STDOUT: source_filename = "thunk_for_imported_interface.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CF.X.Main:I.Main"(ptr sret({ i32 }) %return, ptr %b) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc33.1.b = getelementptr inbounds nuw { i32 }, ptr %b, i32 0, i32 0, !dbg !10
// CHECK:STDOUT:   %.loc33.2 = load i32, ptr %.loc33.1.b, align 4, !dbg !10
// CHECK:STDOUT:   %.loc35.2.b = getelementptr inbounds nuw { i32 }, ptr %return, i32 0, i32 0, !dbg !11
// CHECK:STDOUT:   store i32 %.loc33.2, ptr %.loc35.2.b, align 4, !dbg !11
// CHECK:STDOUT:   ret void, !dbg !12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define void @"_CF:thunk.X.Main:I.Main"(ptr sret({ i32 }) %return, ptr %a) #1 !dbg !13 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc26.1.temp = alloca { i32 }, align 8, !dbg !16
// CHECK:STDOUT:   %.2.temp = alloca { i32 }, align 8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc26.1.temp), !dbg !16
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.2.temp)
// CHECK:STDOUT:   call void @"_CConvert.A.Main:ImplicitAs.707dbe5ae2414d29.Core"(ptr %.2.temp, ptr %a), !dbg !17
// CHECK:STDOUT:   call void @"_CF.X.Main:I.Main"(ptr %.loc26.1.temp, ptr %.2.temp), !dbg !16
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.0c4ab795cec6cb27.Core"(ptr %return, ptr %.loc26.1.temp), !dbg !16
// CHECK:STDOUT:   ret void, !dbg !16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_CConvert.A.Main:ImplicitAs.707dbe5ae2414d29.Core"(ptr sret({ i32 }), ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_CConvert.B.Main:ImplicitAs.0c4ab795cec6cb27.Core"(ptr sret({ i32 }), ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "thunk_for_imported_interface.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "F", linkageName: "_CF.X.Main:I.Main", scope: null, file: !3, line: 55, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !8)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7, !7}
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 8)
// CHECK:STDOUT: !8 = !{!9}
// CHECK:STDOUT: !9 = !DILocalVariable(arg: 1, scope: !4, type: !7)
// CHECK:STDOUT: !10 = !DILocation(line: 55, column: 34, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 55, column: 28, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 55, column: 21, scope: !4)
// CHECK:STDOUT: !13 = distinct !DISubprogram(name: "F", linkageName: "_CF:thunk.X.Main:I.Main", scope: null, file: !3, line: 55, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !14)
// CHECK:STDOUT: !14 = !{!15}
// CHECK:STDOUT: !15 = !DILocalVariable(arg: 1, scope: !13, type: !7)
// CHECK:STDOUT: !16 = !DILocation(line: 55, column: 3, scope: !13)
// CHECK:STDOUT: !17 = !DILocation(line: 4294967295, scope: !13)
// CHECK:STDOUT: ; ModuleID = 'call_thunk_for_imported_interface.carbon'
// CHECK:STDOUT: source_filename = "call_thunk_for_imported_interface.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTest.Main(ptr sret({ i32 }) %return, ptr %a) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc32.1.temp = alloca { i32 }, align 8, !dbg !10
// CHECK:STDOUT:   %.loc32.2.temp = alloca { i32 }, align 8, !dbg !10
// CHECK:STDOUT:   %.loc31.1.temp = alloca { i32 }, align 8, !dbg !11
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc32.1.temp), !dbg !10
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc32.2.temp), !dbg !10
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc31.1.temp), !dbg !11
// CHECK:STDOUT:   call void @"_CConvert.A.Main:ImplicitAs.707dbe5ae2414d29.Core"(ptr %.loc31.1.temp, ptr %a), !dbg !11
// CHECK:STDOUT:   call void @"_CF.X.Main:I.Main"(ptr %.loc32.2.temp, ptr %.loc31.1.temp), !dbg !10
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.0c4ab795cec6cb27.Core"(ptr %return, ptr %.loc32.2.temp), !dbg !12
// CHECK:STDOUT:   ret void, !dbg !12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_CF:thunk.X.Main:I.Main"(ptr sret({ i32 }), ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_CF.X.Main:I.Main"(ptr sret({ i32 }), ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_CConvert.A.Main:ImplicitAs.707dbe5ae2414d29.Core"(ptr sret({ i32 }), ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_CConvert.B.Main:ImplicitAs.0c4ab795cec6cb27.Core"(ptr sret({ i32 }), ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 2, 1, 0 }
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
// CHECK:STDOUT: !3 = !DIFile(filename: "call_thunk_for_imported_interface.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Test", linkageName: "_CTest.Main", scope: null, file: !3, line: 65, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !8)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7, !7}
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 8)
// CHECK:STDOUT: !8 = !{!9}
// CHECK:STDOUT: !9 = !DILocalVariable(arg: 1, scope: !4, type: !7)
// CHECK:STDOUT: !10 = !DILocation(line: 66, column: 10, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 66, column: 18, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 66, column: 3, scope: !4)
