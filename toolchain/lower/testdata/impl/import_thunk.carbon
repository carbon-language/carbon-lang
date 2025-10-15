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
// CHECK:STDOUT: define void @"_CConvert.A.Main:ImplicitAs.Core"(ptr sret({ i32 }) %return, ptr %self) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_49.1.a = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !7
// CHECK:STDOUT:   %.loc9_49.2 = load i32, ptr %.loc9_49.1.a, align 4, !dbg !7
// CHECK:STDOUT:   %.loc9_51.2.b = getelementptr inbounds nuw { i32 }, ptr %return, i32 0, i32 0, !dbg !8
// CHECK:STDOUT:   store i32 %.loc9_49.2, ptr %.loc9_51.2.b, align 4, !dbg !8
// CHECK:STDOUT:   ret void, !dbg !9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CConvert.B.Main:ImplicitAs.Core"(ptr sret({ i32 }) %return, ptr %self) #0 !dbg !10 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc12_49.1.b = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !11
// CHECK:STDOUT:   %.loc12_49.2 = load i32, ptr %.loc12_49.1.b, align 4, !dbg !11
// CHECK:STDOUT:   %.loc12_51.2.c = getelementptr inbounds nuw { i32 }, ptr %return, i32 0, i32 0, !dbg !12
// CHECK:STDOUT:   store i32 %.loc12_49.2, ptr %.loc12_51.2.c, align 4, !dbg !12
// CHECK:STDOUT:   ret void, !dbg !13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CF.61ea2aba74ab3bf1:I.Main"(ptr sret({ i32 }) %return, ptr %b) #0 !dbg !14 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc20_35.1.b = getelementptr inbounds nuw { i32 }, ptr %b, i32 0, i32 0, !dbg !15
// CHECK:STDOUT:   %.loc20_35.2 = load i32, ptr %.loc20_35.1.b, align 4, !dbg !15
// CHECK:STDOUT:   %.loc20_37.2.b = getelementptr inbounds nuw { i32 }, ptr %return, i32 0, i32 0, !dbg !16
// CHECK:STDOUT:   store i32 %.loc20_35.2, ptr %.loc20_37.2.b, align 4, !dbg !16
// CHECK:STDOUT:   ret void, !dbg !17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define void @"_CF:thunk.61ea2aba74ab3bf1:I.Main"(ptr sret({ i32 }) %return, ptr %a) #1 !dbg !18 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc20_19.1.temp = alloca { i32 }, align 8, !dbg !19
// CHECK:STDOUT:   %.loc16_9.1.temp = alloca { i32 }, align 8, !dbg !20
// CHECK:STDOUT:   %.loc20_19.2.temp = alloca { i32 }, align 8, !dbg !19
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc20_19.1.temp), !dbg !19
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc16_9.1.temp), !dbg !20
// CHECK:STDOUT:   call void @"_CConvert.A.Main:ImplicitAs.Core"(ptr %.loc16_9.1.temp, ptr %a), !dbg !20
// CHECK:STDOUT:   call void @"_CF.61ea2aba74ab3bf1:I.Main"(ptr %.loc20_19.1.temp, ptr %.loc16_9.1.temp), !dbg !19
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc20_19.2.temp), !dbg !19
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.Core"(ptr %.loc20_19.2.temp, ptr %.loc20_19.1.temp), !dbg !19
// CHECK:STDOUT:   ret void, !dbg !19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 2, 1, 0 }
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
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "thunk.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.A.Main:ImplicitAs.Core", scope: null, file: !3, line: 9, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 9, column: 45, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 9, column: 39, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 9, column: 32, scope: !4)
// CHECK:STDOUT: !10 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.B.Main:ImplicitAs.Core", scope: null, file: !3, line: 12, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !11 = !DILocation(line: 12, column: 45, scope: !10)
// CHECK:STDOUT: !12 = !DILocation(line: 12, column: 39, scope: !10)
// CHECK:STDOUT: !13 = !DILocation(line: 12, column: 32, scope: !10)
// CHECK:STDOUT: !14 = distinct !DISubprogram(name: "F", linkageName: "_CF.61ea2aba74ab3bf1:I.Main", scope: null, file: !3, line: 20, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !15 = !DILocation(line: 20, column: 34, scope: !14)
// CHECK:STDOUT: !16 = !DILocation(line: 20, column: 28, scope: !14)
// CHECK:STDOUT: !17 = !DILocation(line: 20, column: 21, scope: !14)
// CHECK:STDOUT: !18 = distinct !DISubprogram(name: "F", linkageName: "_CF:thunk.61ea2aba74ab3bf1:I.Main", scope: null, file: !3, line: 20, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !19 = !DILocation(line: 20, column: 3, scope: !18)
// CHECK:STDOUT: !20 = !DILocation(line: 16, column: 8, scope: !18)
// CHECK:STDOUT: ; ModuleID = 'call_thunk.carbon'
// CHECK:STDOUT: source_filename = "call_thunk.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTest.Main(ptr sret({ i32 }) %return, ptr %a) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_20.1.temp = alloca { i32 }, align 8, !dbg !7
// CHECK:STDOUT:   %.loc7_20.2.temp = alloca { i32 }, align 8, !dbg !7
// CHECK:STDOUT:   %.loc7_19.1.temp = alloca { i32 }, align 8, !dbg !8
// CHECK:STDOUT:   %.loc7_21.1.temp = alloca { i32 }, align 8, !dbg !9
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_20.1.temp), !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_20.2.temp), !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_19.1.temp), !dbg !8
// CHECK:STDOUT:   call void @"_CConvert.A.Main:ImplicitAs.Core"(ptr %.loc7_19.1.temp, ptr %a), !dbg !8
// CHECK:STDOUT:   call void @"_CF.61ea2aba74ab3bf1:I.Main"(ptr %.loc7_20.2.temp, ptr %.loc7_19.1.temp), !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_21.1.temp), !dbg !9
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.Core"(ptr %.loc7_21.1.temp, ptr %.loc7_20.2.temp), !dbg !9
// CHECK:STDOUT:   ret void, !dbg !9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_CF:thunk.61ea2aba74ab3bf1:I.Main"(ptr sret({ i32 }), ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_CF.61ea2aba74ab3bf1:I.Main"(ptr sret({ i32 }), ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_CConvert.A.Main:ImplicitAs.Core"(ptr sret({ i32 }), ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_CConvert.B.Main:ImplicitAs.Core"(ptr sret({ i32 }), ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "call_thunk.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Test", linkageName: "_CTest.Main", scope: null, file: !3, line: 6, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 7, column: 10, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 7, column: 19, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 7, column: 3, scope: !4)
// CHECK:STDOUT: ; ModuleID = 'thunk_for_imported_interface.carbon'
// CHECK:STDOUT: source_filename = "thunk_for_imported_interface.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CF.X.Main:I.Main"(ptr sret({ i32 }) %return, ptr %b) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_35.1.b = getelementptr inbounds nuw { i32 }, ptr %b, i32 0, i32 0, !dbg !7
// CHECK:STDOUT:   %.loc9_35.2 = load i32, ptr %.loc9_35.1.b, align 4, !dbg !7
// CHECK:STDOUT:   %.loc9_37.2.b = getelementptr inbounds nuw { i32 }, ptr %return, i32 0, i32 0, !dbg !8
// CHECK:STDOUT:   store i32 %.loc9_35.2, ptr %.loc9_37.2.b, align 4, !dbg !8
// CHECK:STDOUT:   ret void, !dbg !9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define void @"_CF:thunk.X.Main:I.Main"(ptr sret({ i32 }) %return, ptr %a) #1 !dbg !10 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_19.1.temp = alloca { i32 }, align 8, !dbg !11
// CHECK:STDOUT:   %.2.temp = alloca { i32 }, align 8
// CHECK:STDOUT:   %.loc9_19.2.temp = alloca { i32 }, align 8, !dbg !11
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc9_19.1.temp), !dbg !11
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.2.temp)
// CHECK:STDOUT:   call void @"_CConvert.A.Main:ImplicitAs.Core"(ptr %.2.temp, ptr %a), !dbg !12
// CHECK:STDOUT:   call void @"_CF.X.Main:I.Main"(ptr %.loc9_19.1.temp, ptr %.2.temp), !dbg !11
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc9_19.2.temp), !dbg !11
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.Core"(ptr %.loc9_19.2.temp, ptr %.loc9_19.1.temp), !dbg !11
// CHECK:STDOUT:   ret void, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_CConvert.A.Main:ImplicitAs.Core"(ptr sret({ i32 }), ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_CConvert.B.Main:ImplicitAs.Core"(ptr sret({ i32 }), ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 2, 1, 0 }
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
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "thunk_for_imported_interface.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "F", linkageName: "_CF.X.Main:I.Main", scope: null, file: !3, line: 9, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 9, column: 34, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 9, column: 28, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 9, column: 21, scope: !4)
// CHECK:STDOUT: !10 = distinct !DISubprogram(name: "F", linkageName: "_CF:thunk.X.Main:I.Main", scope: null, file: !3, line: 9, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !11 = !DILocation(line: 9, column: 3, scope: !10)
// CHECK:STDOUT: !12 = !DILocation(line: 4294967295, scope: !10)
// CHECK:STDOUT: ; ModuleID = 'call_thunk_for_imported_interface.carbon'
// CHECK:STDOUT: source_filename = "call_thunk_for_imported_interface.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTest.Main(ptr sret({ i32 }) %return, ptr %a) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc8_19.1.temp = alloca { i32 }, align 8, !dbg !7
// CHECK:STDOUT:   %.loc8_19.2.temp = alloca { i32 }, align 8, !dbg !7
// CHECK:STDOUT:   %.loc8_18.1.temp = alloca { i32 }, align 8, !dbg !8
// CHECK:STDOUT:   %.loc8_20.1.temp = alloca { i32 }, align 8, !dbg !9
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc8_19.1.temp), !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc8_19.2.temp), !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc8_18.1.temp), !dbg !8
// CHECK:STDOUT:   call void @"_CConvert.A.Main:ImplicitAs.Core"(ptr %.loc8_18.1.temp, ptr %a), !dbg !8
// CHECK:STDOUT:   call void @"_CF.X.Main:I.Main"(ptr %.loc8_19.2.temp, ptr %.loc8_18.1.temp), !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc8_20.1.temp), !dbg !9
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.Core"(ptr %.loc8_20.1.temp, ptr %.loc8_19.2.temp), !dbg !9
// CHECK:STDOUT:   ret void, !dbg !9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_CF:thunk.X.Main:I.Main"(ptr sret({ i32 }), ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_CF.X.Main:I.Main"(ptr sret({ i32 }), ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_CConvert.A.Main:ImplicitAs.Core"(ptr sret({ i32 }), ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_CConvert.B.Main:ImplicitAs.Core"(ptr sret({ i32 }), ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "call_thunk_for_imported_interface.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Test", linkageName: "_CTest.Main", scope: null, file: !3, line: 7, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 8, column: 10, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 8, column: 18, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 8, column: 3, scope: !4)
