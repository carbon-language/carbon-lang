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
// CHECK:STDOUT:   %.loc20_19.2.temp = alloca { i32 }, align 4, !dbg !56
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc20_19.1.temp), !dbg !56
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc20_19.2.temp), !dbg !56
// CHECK:STDOUT:   call void @"_CConvert.A.Main:ImplicitAs.a6a60dfee2d7d273.Core"(ptr %.loc20_19.2.temp, ptr %a), !dbg !56
// CHECK:STDOUT:   call void @"_CF.61ea2aba74ab3bf1:I.Main"(ptr %.loc20_19.1.temp, ptr %.loc20_19.2.temp), !dbg !56
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.9522ff908e97e267.Core"(ptr %return, ptr %.loc20_19.1.temp), !dbg !56
// CHECK:STDOUT:   ret void, !dbg !56
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!21}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIFile(filename: "thunk.carbon", directory: "")
// CHECK:STDOUT: !21 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !20, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{!23, !23}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.A.Main:ImplicitAs.a6a60dfee2d7d273.Core", scope: null, file: !20, line: 9, type: !25, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !34)
// CHECK:STDOUT: !27 = !DILocalVariable(arg: 1, scope: !26, type: !23)
// CHECK:STDOUT: !31 = !DILocation(line: 9, column: 43, scope: !26)
// CHECK:STDOUT: !32 = !DILocation(line: 9, column: 37, scope: !26)
// CHECK:STDOUT: !33 = !DILocation(line: 9, column: 30, scope: !26)
// CHECK:STDOUT: !34 = !{!27}
// CHECK:STDOUT: !35 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.B.Main:ImplicitAs.9522ff908e97e267.Core", scope: null, file: !20, line: 12, type: !25, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !43)
// CHECK:STDOUT: !36 = !DILocalVariable(arg: 1, scope: !35, type: !23)
// CHECK:STDOUT: !40 = !DILocation(line: 12, column: 43, scope: !35)
// CHECK:STDOUT: !41 = !DILocation(line: 12, column: 37, scope: !35)
// CHECK:STDOUT: !42 = !DILocation(line: 12, column: 30, scope: !35)
// CHECK:STDOUT: !43 = !{!36}
// CHECK:STDOUT: !44 = distinct !DISubprogram(name: "F", linkageName: "_CF.61ea2aba74ab3bf1:I.Main", scope: null, file: !20, line: 20, type: !25, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !52)
// CHECK:STDOUT: !45 = !DILocalVariable(arg: 1, scope: !44, type: !23)
// CHECK:STDOUT: !49 = !DILocation(line: 20, column: 34, scope: !44)
// CHECK:STDOUT: !50 = !DILocation(line: 20, column: 28, scope: !44)
// CHECK:STDOUT: !51 = !DILocation(line: 20, column: 21, scope: !44)
// CHECK:STDOUT: !52 = !{!45}
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "F", linkageName: "_CF:thunk:I.6a2feaaa6331eb92.Main:61ea2aba74ab3bf1:I.Main", scope: null, file: !20, line: 20, type: !25, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !57)
// CHECK:STDOUT: !54 = !DILocalVariable(arg: 1, scope: !53, type: !23)
// CHECK:STDOUT: !56 = !DILocation(line: 20, column: 3, scope: !53)
// CHECK:STDOUT: !57 = !{!54}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'call_thunk.carbon'
// CHECK:STDOUT: source_filename = "call_thunk.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTest.Main(ptr sret({ i32 }) %return, ptr %a) #0 !dbg !60 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_20.1.temp = alloca { i32 }, align 4, !dbg !66
// CHECK:STDOUT:   %.loc7_20.2.temp = alloca { i32 }, align 4, !dbg !66
// CHECK:STDOUT:   %.loc7_19.1.temp = alloca { i32 }, align 4, !dbg !65
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_20.1.temp), !dbg !66
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_20.2.temp), !dbg !66
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_19.1.temp), !dbg !65
// CHECK:STDOUT:   call void @"_CConvert.A.Main:ImplicitAs.a6a60dfee2d7d273.Core"(ptr %.loc7_19.1.temp, ptr %a), !dbg !65
// CHECK:STDOUT:   call void @"_CF.61ea2aba74ab3bf1:I.Main"(ptr %.loc7_20.2.temp, ptr %.loc7_19.1.temp), !dbg !66
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.9522ff908e97e267.Core"(ptr %return, ptr %.loc7_20.2.temp), !dbg !67
// CHECK:STDOUT:   ret void, !dbg !67
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
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!59}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{!23, !23}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !58 = !DIFile(filename: "call_thunk.carbon", directory: "")
// CHECK:STDOUT: !59 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !58, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !60 = distinct !DISubprogram(name: "Test", linkageName: "_CTest.Main", scope: null, file: !58, line: 6, type: !25, spFlags: DISPFlagDefinition, unit: !59, retainedNodes: !68)
// CHECK:STDOUT: !61 = !DILocalVariable(arg: 1, scope: !60, type: !23)
// CHECK:STDOUT: !65 = !DILocation(line: 7, column: 19, scope: !60)
// CHECK:STDOUT: !66 = !DILocation(line: 7, column: 10, scope: !60)
// CHECK:STDOUT: !67 = !DILocation(line: 7, column: 3, scope: !60)
// CHECK:STDOUT: !68 = !{!61}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'thunk_for_imported_interface.carbon'
// CHECK:STDOUT: source_filename = "thunk_for_imported_interface.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CF.X.Main:I.Main"(ptr sret({ i32 }) %return, ptr %b) #0 !dbg !71 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_35.1.b = getelementptr inbounds nuw { i32 }, ptr %b, i32 0, i32 0, !dbg !76
// CHECK:STDOUT:   %.loc9_35.2 = load i32, ptr %.loc9_35.1.b, align 4, !dbg !76
// CHECK:STDOUT:   %.loc9_37.2.b = getelementptr inbounds nuw { i32 }, ptr %return, i32 0, i32 0, !dbg !77
// CHECK:STDOUT:   store i32 %.loc9_35.2, ptr %.loc9_37.2.b, align 4, !dbg !77
// CHECK:STDOUT:   ret void, !dbg !78
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_CF:thunk:I.d0cc85b6c89517e3.Main:X.Main:I.Main"(ptr sret({ i32 }) %return, ptr %a) #1 !dbg !80 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_19.1.temp = alloca { i32 }, align 4, !dbg !83
// CHECK:STDOUT:   %.loc9_19.2.temp = alloca { i32 }, align 4, !dbg !83
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc9_19.1.temp), !dbg !83
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc9_19.2.temp), !dbg !83
// CHECK:STDOUT:   call void @"_CConvert.A.Main:ImplicitAs.a6a60dfee2d7d273.Core"(ptr %.loc9_19.2.temp, ptr %a), !dbg !83
// CHECK:STDOUT:   call void @"_CF.X.Main:I.Main"(ptr %.loc9_19.1.temp, ptr %.loc9_19.2.temp), !dbg !83
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.9522ff908e97e267.Core"(ptr %return, ptr %.loc9_19.1.temp), !dbg !83
// CHECK:STDOUT:   ret void, !dbg !83
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_CConvert.A.Main:ImplicitAs.a6a60dfee2d7d273.Core"(ptr sret({ i32 }), ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_CConvert.B.Main:ImplicitAs.9522ff908e97e267.Core"(ptr sret({ i32 }), ptr)
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!70}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{!23, !23}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !69 = !DIFile(filename: "thunk_for_imported_interface.carbon", directory: "")
// CHECK:STDOUT: !70 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !69, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !71 = distinct !DISubprogram(name: "F", linkageName: "_CF.X.Main:I.Main", scope: null, file: !69, line: 9, type: !25, spFlags: DISPFlagDefinition, unit: !70, retainedNodes: !79)
// CHECK:STDOUT: !72 = !DILocalVariable(arg: 1, scope: !71, type: !23)
// CHECK:STDOUT: !76 = !DILocation(line: 9, column: 34, scope: !71)
// CHECK:STDOUT: !77 = !DILocation(line: 9, column: 28, scope: !71)
// CHECK:STDOUT: !78 = !DILocation(line: 9, column: 21, scope: !71)
// CHECK:STDOUT: !79 = !{!72}
// CHECK:STDOUT: !80 = distinct !DISubprogram(name: "F", linkageName: "_CF:thunk:I.d0cc85b6c89517e3.Main:X.Main:I.Main", scope: null, file: !69, line: 9, type: !25, spFlags: DISPFlagDefinition, unit: !70, retainedNodes: !84)
// CHECK:STDOUT: !81 = !DILocalVariable(arg: 1, scope: !80, type: !23)
// CHECK:STDOUT: !83 = !DILocation(line: 9, column: 3, scope: !80)
// CHECK:STDOUT: !84 = !{!81}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'call_thunk_for_imported_interface.carbon'
// CHECK:STDOUT: source_filename = "call_thunk_for_imported_interface.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTest.Main(ptr sret({ i32 }) %return, ptr %a) #0 !dbg !87 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc8_19.1.temp = alloca { i32 }, align 4, !dbg !93
// CHECK:STDOUT:   %.loc8_19.2.temp = alloca { i32 }, align 4, !dbg !93
// CHECK:STDOUT:   %.loc8_18.1.temp = alloca { i32 }, align 4, !dbg !92
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc8_19.1.temp), !dbg !93
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc8_19.2.temp), !dbg !93
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc8_18.1.temp), !dbg !92
// CHECK:STDOUT:   call void @"_CConvert.A.Main:ImplicitAs.a6a60dfee2d7d273.Core"(ptr %.loc8_18.1.temp, ptr %a), !dbg !92
// CHECK:STDOUT:   call void @"_CF.X.Main:I.Main"(ptr %.loc8_19.2.temp, ptr %.loc8_18.1.temp), !dbg !93
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.9522ff908e97e267.Core"(ptr %return, ptr %.loc8_19.2.temp), !dbg !94
// CHECK:STDOUT:   ret void, !dbg !94
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
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!86}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{!23, !23}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !85 = !DIFile(filename: "call_thunk_for_imported_interface.carbon", directory: "")
// CHECK:STDOUT: !86 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !85, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !87 = distinct !DISubprogram(name: "Test", linkageName: "_CTest.Main", scope: null, file: !85, line: 7, type: !25, spFlags: DISPFlagDefinition, unit: !86, retainedNodes: !95)
// CHECK:STDOUT: !88 = !DILocalVariable(arg: 1, scope: !87, type: !23)
// CHECK:STDOUT: !92 = !DILocation(line: 8, column: 18, scope: !87)
// CHECK:STDOUT: !93 = !DILocation(line: 8, column: 10, scope: !87)
// CHECK:STDOUT: !94 = !DILocation(line: 8, column: 3, scope: !87)
// CHECK:STDOUT: !95 = !{!88}
// CHECK:STDOUT:
