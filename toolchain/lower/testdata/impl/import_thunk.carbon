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
// CHECK:STDOUT: define void @"_CConvert.A.Main:ImplicitAs.a6a60dfee2d7d273.Core"(ptr sret({ i32 }) %return, ptr %self) #0 !dbg !24 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_47.1.a = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !29
// CHECK:STDOUT:   %.loc9_47.2 = load i32, ptr %.loc9_47.1.a, align 4, !dbg !29
// CHECK:STDOUT:   %.loc9_49.2.b = getelementptr inbounds nuw { i32 }, ptr %return, i32 0, i32 0, !dbg !30
// CHECK:STDOUT:   store i32 %.loc9_47.2, ptr %.loc9_49.2.b, align 4, !dbg !30
// CHECK:STDOUT:   ret void, !dbg !31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CConvert.B.Main:ImplicitAs.9522ff908e97e267.Core"(ptr sret({ i32 }) %return, ptr %self) #0 !dbg !33 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc12_47.1.b = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !38
// CHECK:STDOUT:   %.loc12_47.2 = load i32, ptr %.loc12_47.1.b, align 4, !dbg !38
// CHECK:STDOUT:   %.loc12_49.2.c = getelementptr inbounds nuw { i32 }, ptr %return, i32 0, i32 0, !dbg !39
// CHECK:STDOUT:   store i32 %.loc12_47.2, ptr %.loc12_49.2.c, align 4, !dbg !39
// CHECK:STDOUT:   ret void, !dbg !40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CF.61ea2aba74ab3bf1:I.Main"(ptr sret({ i32 }) %return, ptr %b) #0 !dbg !42 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc20_35.1.b = getelementptr inbounds nuw { i32 }, ptr %b, i32 0, i32 0, !dbg !47
// CHECK:STDOUT:   %.loc20_35.2 = load i32, ptr %.loc20_35.1.b, align 4, !dbg !47
// CHECK:STDOUT:   %.loc20_37.2.b = getelementptr inbounds nuw { i32 }, ptr %return, i32 0, i32 0, !dbg !48
// CHECK:STDOUT:   store i32 %.loc20_35.2, ptr %.loc20_37.2.b, align 4, !dbg !48
// CHECK:STDOUT:   ret void, !dbg !49
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_CF:thunk:I.6a2feaaa6331eb92.Main:61ea2aba74ab3bf1:I.Main"(ptr sret({ i32 }) %return, ptr %a) #1 !dbg !51 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc20_19.1.temp = alloca { i32 }, align 4, !dbg !54
// CHECK:STDOUT:   %.loc20_19.2.temp = alloca { i32 }, align 4, !dbg !54
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc20_19.1.temp), !dbg !54
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc20_19.2.temp), !dbg !54
// CHECK:STDOUT:   call void @"_CConvert.A.Main:ImplicitAs.a6a60dfee2d7d273.Core"(ptr %.loc20_19.2.temp, ptr %a), !dbg !54
// CHECK:STDOUT:   call void @"_CF.61ea2aba74ab3bf1:I.Main"(ptr %.loc20_19.1.temp, ptr %.loc20_19.2.temp), !dbg !54
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.9522ff908e97e267.Core"(ptr %return, ptr %.loc20_19.1.temp), !dbg !54
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.6579a5013ba3dcd2(ptr %.loc20_19.1.temp), !dbg !54
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.6579a5013ba3dcd2(ptr %.loc20_19.2.temp), !dbg !54
// CHECK:STDOUT:   ret void, !dbg !54
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.6579a5013ba3dcd2(ptr %t) #0 !dbg !58 {
// CHECK:STDOUT:   ret void, !dbg !63
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.6579a5013ba3dcd2, { 1, 0 }
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
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{!21, !21}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !24 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.A.Main:ImplicitAs.a6a60dfee2d7d273.Core", scope: null, file: !18, line: 9, type: !23, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !32)
// CHECK:STDOUT: !25 = !DILocalVariable(arg: 1, scope: !24, type: !21)
// CHECK:STDOUT: !29 = !DILocation(line: 9, column: 43, scope: !24)
// CHECK:STDOUT: !30 = !DILocation(line: 9, column: 37, scope: !24)
// CHECK:STDOUT: !31 = !DILocation(line: 9, column: 30, scope: !24)
// CHECK:STDOUT: !32 = !{!25}
// CHECK:STDOUT: !33 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.B.Main:ImplicitAs.9522ff908e97e267.Core", scope: null, file: !18, line: 12, type: !23, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !41)
// CHECK:STDOUT: !34 = !DILocalVariable(arg: 1, scope: !33, type: !21)
// CHECK:STDOUT: !38 = !DILocation(line: 12, column: 43, scope: !33)
// CHECK:STDOUT: !39 = !DILocation(line: 12, column: 37, scope: !33)
// CHECK:STDOUT: !40 = !DILocation(line: 12, column: 30, scope: !33)
// CHECK:STDOUT: !41 = !{!34}
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "F", linkageName: "_CF.61ea2aba74ab3bf1:I.Main", scope: null, file: !18, line: 20, type: !23, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !50)
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 1, scope: !42, type: !21)
// CHECK:STDOUT: !47 = !DILocation(line: 20, column: 34, scope: !42)
// CHECK:STDOUT: !48 = !DILocation(line: 20, column: 28, scope: !42)
// CHECK:STDOUT: !49 = !DILocation(line: 20, column: 21, scope: !42)
// CHECK:STDOUT: !50 = !{!43}
// CHECK:STDOUT: !51 = distinct !DISubprogram(name: "F", linkageName: "_CF:thunk:I.6a2feaaa6331eb92.Main:61ea2aba74ab3bf1:I.Main", scope: null, file: !18, line: 20, type: !23, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !57)
// CHECK:STDOUT: !52 = !DILocalVariable(arg: 1, scope: !51, type: !21)
// CHECK:STDOUT: !54 = !DILocation(line: 20, column: 3, scope: !51)
// CHECK:STDOUT: !55 = !{null, !21}
// CHECK:STDOUT: !56 = !DISubroutineType(types: !55)
// CHECK:STDOUT: !57 = !{!52}
// CHECK:STDOUT: !58 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.6579a5013ba3dcd2", scope: null, file: !10, line: 32, type: !56, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !64)
// CHECK:STDOUT: !59 = !DILocalVariable(arg: 1, scope: !58, type: !21)
// CHECK:STDOUT: !63 = !DILocation(line: 32, column: 1, scope: !58)
// CHECK:STDOUT: !64 = !{!59}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'call_thunk.carbon'
// CHECK:STDOUT: source_filename = "call_thunk.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTest.Main(ptr sret({ i32 }) %return, ptr %a) #0 !dbg !67 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_20.1.temp = alloca { i32 }, align 4, !dbg !73
// CHECK:STDOUT:   %.loc7_20.2.temp = alloca { i32 }, align 4, !dbg !73
// CHECK:STDOUT:   %.loc7_19.1.temp = alloca { i32 }, align 4, !dbg !72
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_20.1.temp), !dbg !73
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_20.2.temp), !dbg !73
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_19.1.temp), !dbg !72
// CHECK:STDOUT:   call void @"_CConvert.A.Main:ImplicitAs.a6a60dfee2d7d273.Core"(ptr %.loc7_19.1.temp, ptr %a), !dbg !72
// CHECK:STDOUT:   call void @"_CF.61ea2aba74ab3bf1:I.Main"(ptr %.loc7_20.2.temp, ptr %.loc7_19.1.temp), !dbg !73
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.9522ff908e97e267.Core"(ptr %return, ptr %.loc7_20.2.temp), !dbg !74
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.6579a5013ba3dcd2(ptr %.loc7_20.2.temp), !dbg !73
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.6579a5013ba3dcd2(ptr %.loc7_19.1.temp), !dbg !72
// CHECK:STDOUT:   ret void, !dbg !74
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
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.6579a5013ba3dcd2(ptr %t) #0 !dbg !76 {
// CHECK:STDOUT:   ret void, !dbg !81
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.6579a5013ba3dcd2, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!66}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{!21, !21}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !55 = !{null, !21}
// CHECK:STDOUT: !56 = !DISubroutineType(types: !55)
// CHECK:STDOUT: !65 = !DIFile(filename: "call_thunk.carbon", directory: "")
// CHECK:STDOUT: !66 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !65, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !67 = distinct !DISubprogram(name: "Test", linkageName: "_CTest.Main", scope: null, file: !65, line: 6, type: !23, spFlags: DISPFlagDefinition, unit: !66, retainedNodes: !75)
// CHECK:STDOUT: !68 = !DILocalVariable(arg: 1, scope: !67, type: !21)
// CHECK:STDOUT: !72 = !DILocation(line: 7, column: 19, scope: !67)
// CHECK:STDOUT: !73 = !DILocation(line: 7, column: 10, scope: !67)
// CHECK:STDOUT: !74 = !DILocation(line: 7, column: 3, scope: !67)
// CHECK:STDOUT: !75 = !{!68}
// CHECK:STDOUT: !76 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.6579a5013ba3dcd2", scope: null, file: !10, line: 32, type: !56, spFlags: DISPFlagDefinition, unit: !66, retainedNodes: !82)
// CHECK:STDOUT: !77 = !DILocalVariable(arg: 1, scope: !76, type: !21)
// CHECK:STDOUT: !81 = !DILocation(line: 32, column: 1, scope: !76)
// CHECK:STDOUT: !82 = !{!77}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'thunk_for_imported_interface.carbon'
// CHECK:STDOUT: source_filename = "thunk_for_imported_interface.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CF.X.Main:I.Main"(ptr sret({ i32 }) %return, ptr %b) #0 !dbg !85 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_35.1.b = getelementptr inbounds nuw { i32 }, ptr %b, i32 0, i32 0, !dbg !90
// CHECK:STDOUT:   %.loc9_35.2 = load i32, ptr %.loc9_35.1.b, align 4, !dbg !90
// CHECK:STDOUT:   %.loc9_37.2.b = getelementptr inbounds nuw { i32 }, ptr %return, i32 0, i32 0, !dbg !91
// CHECK:STDOUT:   store i32 %.loc9_35.2, ptr %.loc9_37.2.b, align 4, !dbg !91
// CHECK:STDOUT:   ret void, !dbg !92
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_CF:thunk:I.d0cc85b6c89517e3.Main:X.Main:I.Main"(ptr sret({ i32 }) %return, ptr %a) #1 !dbg !94 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_19.1.temp = alloca { i32 }, align 4, !dbg !97
// CHECK:STDOUT:   %.loc9_19.2.temp = alloca { i32 }, align 4, !dbg !97
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc9_19.1.temp), !dbg !97
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc9_19.2.temp), !dbg !97
// CHECK:STDOUT:   call void @"_CConvert.A.Main:ImplicitAs.a6a60dfee2d7d273.Core"(ptr %.loc9_19.2.temp, ptr %a), !dbg !97
// CHECK:STDOUT:   call void @"_CF.X.Main:I.Main"(ptr %.loc9_19.1.temp, ptr %.loc9_19.2.temp), !dbg !97
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.9522ff908e97e267.Core"(ptr %return, ptr %.loc9_19.1.temp), !dbg !97
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.6579a5013ba3dcd2(ptr %.loc9_19.1.temp), !dbg !97
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.6579a5013ba3dcd2(ptr %.loc9_19.2.temp), !dbg !97
// CHECK:STDOUT:   ret void, !dbg !97
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_CConvert.A.Main:ImplicitAs.a6a60dfee2d7d273.Core"(ptr sret({ i32 }), ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_CConvert.B.Main:ImplicitAs.9522ff908e97e267.Core"(ptr sret({ i32 }), ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.6579a5013ba3dcd2(ptr %t) #0 !dbg !99 {
// CHECK:STDOUT:   ret void, !dbg !104
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.6579a5013ba3dcd2, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!84}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{!21, !21}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !55 = !{null, !21}
// CHECK:STDOUT: !56 = !DISubroutineType(types: !55)
// CHECK:STDOUT: !83 = !DIFile(filename: "thunk_for_imported_interface.carbon", directory: "")
// CHECK:STDOUT: !84 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !83, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !85 = distinct !DISubprogram(name: "F", linkageName: "_CF.X.Main:I.Main", scope: null, file: !83, line: 9, type: !23, spFlags: DISPFlagDefinition, unit: !84, retainedNodes: !93)
// CHECK:STDOUT: !86 = !DILocalVariable(arg: 1, scope: !85, type: !21)
// CHECK:STDOUT: !90 = !DILocation(line: 9, column: 34, scope: !85)
// CHECK:STDOUT: !91 = !DILocation(line: 9, column: 28, scope: !85)
// CHECK:STDOUT: !92 = !DILocation(line: 9, column: 21, scope: !85)
// CHECK:STDOUT: !93 = !{!86}
// CHECK:STDOUT: !94 = distinct !DISubprogram(name: "F", linkageName: "_CF:thunk:I.d0cc85b6c89517e3.Main:X.Main:I.Main", scope: null, file: !83, line: 9, type: !23, spFlags: DISPFlagDefinition, unit: !84, retainedNodes: !98)
// CHECK:STDOUT: !95 = !DILocalVariable(arg: 1, scope: !94, type: !21)
// CHECK:STDOUT: !97 = !DILocation(line: 9, column: 3, scope: !94)
// CHECK:STDOUT: !98 = !{!95}
// CHECK:STDOUT: !99 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.6579a5013ba3dcd2", scope: null, file: !10, line: 32, type: !56, spFlags: DISPFlagDefinition, unit: !84, retainedNodes: !105)
// CHECK:STDOUT: !100 = !DILocalVariable(arg: 1, scope: !99, type: !21)
// CHECK:STDOUT: !104 = !DILocation(line: 32, column: 1, scope: !99)
// CHECK:STDOUT: !105 = !{!100}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'call_thunk_for_imported_interface.carbon'
// CHECK:STDOUT: source_filename = "call_thunk_for_imported_interface.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTest.Main(ptr sret({ i32 }) %return, ptr %a) #0 !dbg !108 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc8_19.1.temp = alloca { i32 }, align 4, !dbg !114
// CHECK:STDOUT:   %.loc8_19.2.temp = alloca { i32 }, align 4, !dbg !114
// CHECK:STDOUT:   %.loc8_18.1.temp = alloca { i32 }, align 4, !dbg !113
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc8_19.1.temp), !dbg !114
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc8_19.2.temp), !dbg !114
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc8_18.1.temp), !dbg !113
// CHECK:STDOUT:   call void @"_CConvert.A.Main:ImplicitAs.a6a60dfee2d7d273.Core"(ptr %.loc8_18.1.temp, ptr %a), !dbg !113
// CHECK:STDOUT:   call void @"_CF.X.Main:I.Main"(ptr %.loc8_19.2.temp, ptr %.loc8_18.1.temp), !dbg !114
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.9522ff908e97e267.Core"(ptr %return, ptr %.loc8_19.2.temp), !dbg !115
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.6579a5013ba3dcd2(ptr %.loc8_19.2.temp), !dbg !114
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.6579a5013ba3dcd2(ptr %.loc8_18.1.temp), !dbg !113
// CHECK:STDOUT:   ret void, !dbg !115
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
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.6579a5013ba3dcd2(ptr %t) #0 !dbg !117 {
// CHECK:STDOUT:   ret void, !dbg !122
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.6579a5013ba3dcd2, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!107}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{!21, !21}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !55 = !{null, !21}
// CHECK:STDOUT: !56 = !DISubroutineType(types: !55)
// CHECK:STDOUT: !106 = !DIFile(filename: "call_thunk_for_imported_interface.carbon", directory: "")
// CHECK:STDOUT: !107 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !106, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !108 = distinct !DISubprogram(name: "Test", linkageName: "_CTest.Main", scope: null, file: !106, line: 7, type: !23, spFlags: DISPFlagDefinition, unit: !107, retainedNodes: !116)
// CHECK:STDOUT: !109 = !DILocalVariable(arg: 1, scope: !108, type: !21)
// CHECK:STDOUT: !113 = !DILocation(line: 8, column: 18, scope: !108)
// CHECK:STDOUT: !114 = !DILocation(line: 8, column: 10, scope: !108)
// CHECK:STDOUT: !115 = !DILocation(line: 8, column: 3, scope: !108)
// CHECK:STDOUT: !116 = !{!109}
// CHECK:STDOUT: !117 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.6579a5013ba3dcd2", scope: null, file: !10, line: 32, type: !56, spFlags: DISPFlagDefinition, unit: !107, retainedNodes: !123)
// CHECK:STDOUT: !118 = !DILocalVariable(arg: 1, scope: !117, type: !21)
// CHECK:STDOUT: !122 = !DILocation(line: 32, column: 1, scope: !117)
// CHECK:STDOUT: !123 = !{!118}
// CHECK:STDOUT:
