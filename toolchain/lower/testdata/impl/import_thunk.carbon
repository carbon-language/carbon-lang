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
// CHECK:STDOUT: define void @"_CConvert.A.Main:ImplicitAs.a6a60dfee2d7d273.Core"(ptr sret({ i32 }) %return, ptr %self) #0 !dbg !28 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_47.1.a = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !33
// CHECK:STDOUT:   %.loc9_47.2 = load i32, ptr %.loc9_47.1.a, align 4, !dbg !33
// CHECK:STDOUT:   %.loc9_49.2.b = getelementptr inbounds nuw { i32 }, ptr %return, i32 0, i32 0, !dbg !34
// CHECK:STDOUT:   store i32 %.loc9_47.2, ptr %.loc9_49.2.b, align 4, !dbg !34
// CHECK:STDOUT:   ret void, !dbg !35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CConvert.B.Main:ImplicitAs.9522ff908e97e267.Core"(ptr sret({ i32 }) %return, ptr %self) #0 !dbg !37 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc12_47.1.b = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !42
// CHECK:STDOUT:   %.loc12_47.2 = load i32, ptr %.loc12_47.1.b, align 4, !dbg !42
// CHECK:STDOUT:   %.loc12_49.2.c = getelementptr inbounds nuw { i32 }, ptr %return, i32 0, i32 0, !dbg !43
// CHECK:STDOUT:   store i32 %.loc12_47.2, ptr %.loc12_49.2.c, align 4, !dbg !43
// CHECK:STDOUT:   ret void, !dbg !44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CF.61ea2aba74ab3bf1:I.Main"(ptr sret({ i32 }) %return, ptr %b) #0 !dbg !46 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc20_35.1.b = getelementptr inbounds nuw { i32 }, ptr %b, i32 0, i32 0, !dbg !51
// CHECK:STDOUT:   %.loc20_35.2 = load i32, ptr %.loc20_35.1.b, align 4, !dbg !51
// CHECK:STDOUT:   %.loc20_37.2.b = getelementptr inbounds nuw { i32 }, ptr %return, i32 0, i32 0, !dbg !52
// CHECK:STDOUT:   store i32 %.loc20_35.2, ptr %.loc20_37.2.b, align 4, !dbg !52
// CHECK:STDOUT:   ret void, !dbg !53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_CF:thunk:I.6a2feaaa6331eb92.Main:61ea2aba74ab3bf1:I.Main"(ptr sret({ i32 }) %return, ptr %a) #1 !dbg !55 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc20_19.1.temp = alloca { i32 }, align 4, !dbg !58
// CHECK:STDOUT:   %.4.temp = alloca { i32 }, align 4, !dbg !57
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc20_19.1.temp), !dbg !58
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.4.temp), !dbg !57
// CHECK:STDOUT:   call void @"_CConvert.A.Main:ImplicitAs.a6a60dfee2d7d273.Core"(ptr %.4.temp, ptr %a), !dbg !57
// CHECK:STDOUT:   call void @"_CF.61ea2aba74ab3bf1:I.Main"(ptr %.loc20_19.1.temp, ptr %.4.temp), !dbg !58
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.9522ff908e97e267.Core"(ptr %return, ptr %.loc20_19.1.temp), !dbg !58
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.6a6353dfb36934de(ptr %.loc20_19.1.temp), !dbg !58
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.6a6353dfb36934de(ptr %.4.temp), !dbg !57
// CHECK:STDOUT:   ret void, !dbg !58
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !60 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !62
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.0bfe8bb35a2876b2:core.Destroy.Core"(ptr %self) #0 !dbg !64 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !66
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.8c4b83ecd900f1ea:core.Destroy.Core"(ptr %self) #0 !dbg !68 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !70
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.6a6353dfb36934de(ptr %self) #0 !dbg !72 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.8c4b83ecd900f1ea:core.Destroy.Core"(ptr %self), !dbg !76
// CHECK:STDOUT:   ret void, !dbg !77
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Destroy.Core.6a6353dfb36934de, { 1, 0 }
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
// CHECK:STDOUT: !14 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !18 = !DIFile(filename: "thunk.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{!21, !21}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !24 = !{null, !20}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = !{null, !21}
// CHECK:STDOUT: !27 = !DISubroutineType(types: !26)
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.A.Main:ImplicitAs.a6a60dfee2d7d273.Core", scope: null, file: !18, line: 9, type: !23, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !36)
// CHECK:STDOUT: !29 = !DILocalVariable(arg: 1, scope: !28, type: !21)
// CHECK:STDOUT: !33 = !DILocation(line: 9, column: 43, scope: !28)
// CHECK:STDOUT: !34 = !DILocation(line: 9, column: 37, scope: !28)
// CHECK:STDOUT: !35 = !DILocation(line: 9, column: 30, scope: !28)
// CHECK:STDOUT: !36 = !{!29}
// CHECK:STDOUT: !37 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.B.Main:ImplicitAs.9522ff908e97e267.Core", scope: null, file: !18, line: 12, type: !23, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !45)
// CHECK:STDOUT: !38 = !DILocalVariable(arg: 1, scope: !37, type: !21)
// CHECK:STDOUT: !42 = !DILocation(line: 12, column: 43, scope: !37)
// CHECK:STDOUT: !43 = !DILocation(line: 12, column: 37, scope: !37)
// CHECK:STDOUT: !44 = !DILocation(line: 12, column: 30, scope: !37)
// CHECK:STDOUT: !45 = !{!38}
// CHECK:STDOUT: !46 = distinct !DISubprogram(name: "F", linkageName: "_CF.61ea2aba74ab3bf1:I.Main", scope: null, file: !18, line: 20, type: !23, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !54)
// CHECK:STDOUT: !47 = !DILocalVariable(arg: 1, scope: !46, type: !21)
// CHECK:STDOUT: !51 = !DILocation(line: 20, column: 34, scope: !46)
// CHECK:STDOUT: !52 = !DILocation(line: 20, column: 28, scope: !46)
// CHECK:STDOUT: !53 = !DILocation(line: 20, column: 21, scope: !46)
// CHECK:STDOUT: !54 = !{!47}
// CHECK:STDOUT: !55 = distinct !DISubprogram(name: "F", linkageName: "_CF:thunk:I.6a2feaaa6331eb92.Main:61ea2aba74ab3bf1:I.Main", scope: null, file: !18, line: 20, type: !23, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !59)
// CHECK:STDOUT: !56 = !DILocalVariable(arg: 1, scope: !55, type: !21)
// CHECK:STDOUT: !57 = !DILocation(line: 0, scope: !55)
// CHECK:STDOUT: !58 = !DILocation(line: 20, column: 3, scope: !55)
// CHECK:STDOUT: !59 = !{!56}
// CHECK:STDOUT: !60 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core", scope: null, file: !18, line: 20, type: !25, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !63)
// CHECK:STDOUT: !61 = !DILocalVariable(arg: 1, scope: !60, type: !20)
// CHECK:STDOUT: !62 = !DILocation(line: 20, column: 3, scope: !60)
// CHECK:STDOUT: !63 = !{!61}
// CHECK:STDOUT: !64 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.0bfe8bb35a2876b2:core.Destroy.Core", scope: null, file: !18, line: 20, type: !27, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !67)
// CHECK:STDOUT: !65 = !DILocalVariable(arg: 1, scope: !64, type: !21)
// CHECK:STDOUT: !66 = !DILocation(line: 20, column: 3, scope: !64)
// CHECK:STDOUT: !67 = !{!65}
// CHECK:STDOUT: !68 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.8c4b83ecd900f1ea:core.Destroy.Core", scope: null, file: !18, line: 20, type: !27, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !71)
// CHECK:STDOUT: !69 = !DILocalVariable(arg: 1, scope: !68, type: !21)
// CHECK:STDOUT: !70 = !DILocation(line: 20, column: 3, scope: !68)
// CHECK:STDOUT: !71 = !{!69}
// CHECK:STDOUT: !72 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.6a6353dfb36934de", scope: null, file: !14, line: 24, type: !27, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !78)
// CHECK:STDOUT: !73 = !DILocalVariable(arg: 1, scope: !72, type: !21)
// CHECK:STDOUT: !76 = !DILocation(line: 26, column: 5, scope: !72)
// CHECK:STDOUT: !77 = !DILocation(line: 24, column: 3, scope: !72)
// CHECK:STDOUT: !78 = !{!73}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'call_thunk.carbon'
// CHECK:STDOUT: source_filename = "call_thunk.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTest.Main(ptr sret({ i32 }) %return, ptr %a) #0 !dbg !81 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_20.1.temp = alloca { i32 }, align 4, !dbg !87
// CHECK:STDOUT:   %.loc7_20.2.temp = alloca { i32 }, align 4, !dbg !87
// CHECK:STDOUT:   %.loc7_19.1.temp = alloca { i32 }, align 4, !dbg !86
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_20.1.temp), !dbg !87
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_20.2.temp), !dbg !87
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_19.1.temp), !dbg !86
// CHECK:STDOUT:   call void @"_CConvert.A.Main:ImplicitAs.a6a60dfee2d7d273.Core"(ptr %.loc7_19.1.temp, ptr %a), !dbg !86
// CHECK:STDOUT:   call void @"_CF.61ea2aba74ab3bf1:I.Main"(ptr %.loc7_20.2.temp, ptr %.loc7_19.1.temp), !dbg !87
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.9522ff908e97e267.Core"(ptr %return, ptr %.loc7_20.2.temp), !dbg !88
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.6a6353dfb36934de(ptr %.loc7_20.2.temp), !dbg !87
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.6a6353dfb36934de(ptr %.loc7_19.1.temp), !dbg !86
// CHECK:STDOUT:   ret void, !dbg !88
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
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !90 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !92
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.0bfe8bb35a2876b2:core.Destroy.Core"(ptr %self) #0 !dbg !94 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !96
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.8c4b83ecd900f1ea:core.Destroy.Core"(ptr %self) #0 !dbg !98 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !100
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.6a6353dfb36934de(ptr %self) #0 !dbg !102 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.8c4b83ecd900f1ea:core.Destroy.Core"(ptr %self), !dbg !106
// CHECK:STDOUT:   ret void, !dbg !107
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Destroy.Core.6a6353dfb36934de, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!80}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !14 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{!21, !21}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !24 = !{null, !20}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = !{null, !21}
// CHECK:STDOUT: !27 = !DISubroutineType(types: !26)
// CHECK:STDOUT: !79 = !DIFile(filename: "call_thunk.carbon", directory: "")
// CHECK:STDOUT: !80 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !79, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !81 = distinct !DISubprogram(name: "Test", linkageName: "_CTest.Main", scope: null, file: !79, line: 6, type: !23, spFlags: DISPFlagDefinition, unit: !80, retainedNodes: !89)
// CHECK:STDOUT: !82 = !DILocalVariable(arg: 1, scope: !81, type: !21)
// CHECK:STDOUT: !86 = !DILocation(line: 7, column: 19, scope: !81)
// CHECK:STDOUT: !87 = !DILocation(line: 7, column: 10, scope: !81)
// CHECK:STDOUT: !88 = !DILocation(line: 7, column: 3, scope: !81)
// CHECK:STDOUT: !89 = !{!82}
// CHECK:STDOUT: !90 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core", scope: null, file: !79, line: 7, type: !25, spFlags: DISPFlagDefinition, unit: !80, retainedNodes: !93)
// CHECK:STDOUT: !91 = !DILocalVariable(arg: 1, scope: !90, type: !20)
// CHECK:STDOUT: !92 = !DILocation(line: 7, column: 10, scope: !90)
// CHECK:STDOUT: !93 = !{!91}
// CHECK:STDOUT: !94 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.0bfe8bb35a2876b2:core.Destroy.Core", scope: null, file: !79, line: 7, type: !27, spFlags: DISPFlagDefinition, unit: !80, retainedNodes: !97)
// CHECK:STDOUT: !95 = !DILocalVariable(arg: 1, scope: !94, type: !21)
// CHECK:STDOUT: !96 = !DILocation(line: 7, column: 10, scope: !94)
// CHECK:STDOUT: !97 = !{!95}
// CHECK:STDOUT: !98 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.8c4b83ecd900f1ea:core.Destroy.Core", scope: null, file: !79, line: 7, type: !27, spFlags: DISPFlagDefinition, unit: !80, retainedNodes: !101)
// CHECK:STDOUT: !99 = !DILocalVariable(arg: 1, scope: !98, type: !21)
// CHECK:STDOUT: !100 = !DILocation(line: 7, column: 10, scope: !98)
// CHECK:STDOUT: !101 = !{!99}
// CHECK:STDOUT: !102 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.6a6353dfb36934de", scope: null, file: !14, line: 24, type: !27, spFlags: DISPFlagDefinition, unit: !80, retainedNodes: !108)
// CHECK:STDOUT: !103 = !DILocalVariable(arg: 1, scope: !102, type: !21)
// CHECK:STDOUT: !106 = !DILocation(line: 26, column: 5, scope: !102)
// CHECK:STDOUT: !107 = !DILocation(line: 24, column: 3, scope: !102)
// CHECK:STDOUT: !108 = !{!103}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'thunk_for_imported_interface.carbon'
// CHECK:STDOUT: source_filename = "thunk_for_imported_interface.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CF.X.Main:I.Main"(ptr sret({ i32 }) %return, ptr %b) #0 !dbg !111 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_35.1.b = getelementptr inbounds nuw { i32 }, ptr %b, i32 0, i32 0, !dbg !116
// CHECK:STDOUT:   %.loc9_35.2 = load i32, ptr %.loc9_35.1.b, align 4, !dbg !116
// CHECK:STDOUT:   %.loc9_37.2.b = getelementptr inbounds nuw { i32 }, ptr %return, i32 0, i32 0, !dbg !117
// CHECK:STDOUT:   store i32 %.loc9_35.2, ptr %.loc9_37.2.b, align 4, !dbg !117
// CHECK:STDOUT:   ret void, !dbg !118
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_CF:thunk:I.d0cc85b6c89517e3.Main:X.Main:I.Main"(ptr sret({ i32 }) %return, ptr %a) #1 !dbg !120 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_19.1.temp = alloca { i32 }, align 4, !dbg !123
// CHECK:STDOUT:   %.4.temp = alloca { i32 }, align 4, !dbg !122
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc9_19.1.temp), !dbg !123
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.4.temp), !dbg !122
// CHECK:STDOUT:   call void @"_CConvert.A.Main:ImplicitAs.a6a60dfee2d7d273.Core"(ptr %.4.temp, ptr %a), !dbg !122
// CHECK:STDOUT:   call void @"_CF.X.Main:I.Main"(ptr %.loc9_19.1.temp, ptr %.4.temp), !dbg !123
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.9522ff908e97e267.Core"(ptr %return, ptr %.loc9_19.1.temp), !dbg !123
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.6a6353dfb36934de(ptr %.loc9_19.1.temp), !dbg !123
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.6a6353dfb36934de(ptr %.4.temp), !dbg !122
// CHECK:STDOUT:   ret void, !dbg !123
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_CConvert.A.Main:ImplicitAs.a6a60dfee2d7d273.Core"(ptr sret({ i32 }), ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_CConvert.B.Main:ImplicitAs.9522ff908e97e267.Core"(ptr sret({ i32 }), ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !125 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !127
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.0bfe8bb35a2876b2:core.Destroy.Core"(ptr %self) #0 !dbg !129 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !131
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.8c4b83ecd900f1ea:core.Destroy.Core"(ptr %self) #0 !dbg !133 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !135
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.6a6353dfb36934de(ptr %self) #0 !dbg !137 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.8c4b83ecd900f1ea:core.Destroy.Core"(ptr %self), !dbg !141
// CHECK:STDOUT:   ret void, !dbg !142
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Destroy.Core.6a6353dfb36934de, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!110}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !14 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{!21, !21}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !24 = !{null, !20}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = !{null, !21}
// CHECK:STDOUT: !27 = !DISubroutineType(types: !26)
// CHECK:STDOUT: !109 = !DIFile(filename: "thunk_for_imported_interface.carbon", directory: "")
// CHECK:STDOUT: !110 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !109, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !111 = distinct !DISubprogram(name: "F", linkageName: "_CF.X.Main:I.Main", scope: null, file: !109, line: 9, type: !23, spFlags: DISPFlagDefinition, unit: !110, retainedNodes: !119)
// CHECK:STDOUT: !112 = !DILocalVariable(arg: 1, scope: !111, type: !21)
// CHECK:STDOUT: !116 = !DILocation(line: 9, column: 34, scope: !111)
// CHECK:STDOUT: !117 = !DILocation(line: 9, column: 28, scope: !111)
// CHECK:STDOUT: !118 = !DILocation(line: 9, column: 21, scope: !111)
// CHECK:STDOUT: !119 = !{!112}
// CHECK:STDOUT: !120 = distinct !DISubprogram(name: "F", linkageName: "_CF:thunk:I.d0cc85b6c89517e3.Main:X.Main:I.Main", scope: null, file: !109, line: 9, type: !23, spFlags: DISPFlagDefinition, unit: !110, retainedNodes: !124)
// CHECK:STDOUT: !121 = !DILocalVariable(arg: 1, scope: !120, type: !21)
// CHECK:STDOUT: !122 = !DILocation(line: 0, scope: !120)
// CHECK:STDOUT: !123 = !DILocation(line: 9, column: 3, scope: !120)
// CHECK:STDOUT: !124 = !{!121}
// CHECK:STDOUT: !125 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core", scope: null, file: !109, line: 9, type: !25, spFlags: DISPFlagDefinition, unit: !110, retainedNodes: !128)
// CHECK:STDOUT: !126 = !DILocalVariable(arg: 1, scope: !125, type: !20)
// CHECK:STDOUT: !127 = !DILocation(line: 9, column: 3, scope: !125)
// CHECK:STDOUT: !128 = !{!126}
// CHECK:STDOUT: !129 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.0bfe8bb35a2876b2:core.Destroy.Core", scope: null, file: !109, line: 9, type: !27, spFlags: DISPFlagDefinition, unit: !110, retainedNodes: !132)
// CHECK:STDOUT: !130 = !DILocalVariable(arg: 1, scope: !129, type: !21)
// CHECK:STDOUT: !131 = !DILocation(line: 9, column: 3, scope: !129)
// CHECK:STDOUT: !132 = !{!130}
// CHECK:STDOUT: !133 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.8c4b83ecd900f1ea:core.Destroy.Core", scope: null, file: !109, line: 9, type: !27, spFlags: DISPFlagDefinition, unit: !110, retainedNodes: !136)
// CHECK:STDOUT: !134 = !DILocalVariable(arg: 1, scope: !133, type: !21)
// CHECK:STDOUT: !135 = !DILocation(line: 9, column: 3, scope: !133)
// CHECK:STDOUT: !136 = !{!134}
// CHECK:STDOUT: !137 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.6a6353dfb36934de", scope: null, file: !14, line: 24, type: !27, spFlags: DISPFlagDefinition, unit: !110, retainedNodes: !143)
// CHECK:STDOUT: !138 = !DILocalVariable(arg: 1, scope: !137, type: !21)
// CHECK:STDOUT: !141 = !DILocation(line: 26, column: 5, scope: !137)
// CHECK:STDOUT: !142 = !DILocation(line: 24, column: 3, scope: !137)
// CHECK:STDOUT: !143 = !{!138}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'call_thunk_for_imported_interface.carbon'
// CHECK:STDOUT: source_filename = "call_thunk_for_imported_interface.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTest.Main(ptr sret({ i32 }) %return, ptr %a) #0 !dbg !146 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc8_19.1.temp = alloca { i32 }, align 4, !dbg !152
// CHECK:STDOUT:   %.loc8_19.2.temp = alloca { i32 }, align 4, !dbg !152
// CHECK:STDOUT:   %.loc8_18.1.temp = alloca { i32 }, align 4, !dbg !151
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc8_19.1.temp), !dbg !152
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc8_19.2.temp), !dbg !152
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc8_18.1.temp), !dbg !151
// CHECK:STDOUT:   call void @"_CConvert.A.Main:ImplicitAs.a6a60dfee2d7d273.Core"(ptr %.loc8_18.1.temp, ptr %a), !dbg !151
// CHECK:STDOUT:   call void @"_CF.X.Main:I.Main"(ptr %.loc8_19.2.temp, ptr %.loc8_18.1.temp), !dbg !152
// CHECK:STDOUT:   call void @"_CConvert.B.Main:ImplicitAs.9522ff908e97e267.Core"(ptr %return, ptr %.loc8_19.2.temp), !dbg !153
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.6a6353dfb36934de(ptr %.loc8_19.2.temp), !dbg !152
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.6a6353dfb36934de(ptr %.loc8_18.1.temp), !dbg !151
// CHECK:STDOUT:   ret void, !dbg !153
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
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !155 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !157
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.0bfe8bb35a2876b2:core.Destroy.Core"(ptr %self) #0 !dbg !159 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !161
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.8c4b83ecd900f1ea:core.Destroy.Core"(ptr %self) #0 !dbg !163 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !165
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.6a6353dfb36934de(ptr %self) #0 !dbg !167 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.8c4b83ecd900f1ea:core.Destroy.Core"(ptr %self), !dbg !171
// CHECK:STDOUT:   ret void, !dbg !172
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Destroy.Core.6a6353dfb36934de, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!145}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !14 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{!21, !21}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !24 = !{null, !20}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = !{null, !21}
// CHECK:STDOUT: !27 = !DISubroutineType(types: !26)
// CHECK:STDOUT: !144 = !DIFile(filename: "call_thunk_for_imported_interface.carbon", directory: "")
// CHECK:STDOUT: !145 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !144, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !146 = distinct !DISubprogram(name: "Test", linkageName: "_CTest.Main", scope: null, file: !144, line: 7, type: !23, spFlags: DISPFlagDefinition, unit: !145, retainedNodes: !154)
// CHECK:STDOUT: !147 = !DILocalVariable(arg: 1, scope: !146, type: !21)
// CHECK:STDOUT: !151 = !DILocation(line: 8, column: 18, scope: !146)
// CHECK:STDOUT: !152 = !DILocation(line: 8, column: 10, scope: !146)
// CHECK:STDOUT: !153 = !DILocation(line: 8, column: 3, scope: !146)
// CHECK:STDOUT: !154 = !{!147}
// CHECK:STDOUT: !155 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core", scope: null, file: !144, line: 8, type: !25, spFlags: DISPFlagDefinition, unit: !145, retainedNodes: !158)
// CHECK:STDOUT: !156 = !DILocalVariable(arg: 1, scope: !155, type: !20)
// CHECK:STDOUT: !157 = !DILocation(line: 8, column: 10, scope: !155)
// CHECK:STDOUT: !158 = !{!156}
// CHECK:STDOUT: !159 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.0bfe8bb35a2876b2:core.Destroy.Core", scope: null, file: !144, line: 8, type: !27, spFlags: DISPFlagDefinition, unit: !145, retainedNodes: !162)
// CHECK:STDOUT: !160 = !DILocalVariable(arg: 1, scope: !159, type: !21)
// CHECK:STDOUT: !161 = !DILocation(line: 8, column: 10, scope: !159)
// CHECK:STDOUT: !162 = !{!160}
// CHECK:STDOUT: !163 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.8c4b83ecd900f1ea:core.Destroy.Core", scope: null, file: !144, line: 8, type: !27, spFlags: DISPFlagDefinition, unit: !145, retainedNodes: !166)
// CHECK:STDOUT: !164 = !DILocalVariable(arg: 1, scope: !163, type: !21)
// CHECK:STDOUT: !165 = !DILocation(line: 8, column: 10, scope: !163)
// CHECK:STDOUT: !166 = !{!164}
// CHECK:STDOUT: !167 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.6a6353dfb36934de", scope: null, file: !14, line: 24, type: !27, spFlags: DISPFlagDefinition, unit: !145, retainedNodes: !173)
// CHECK:STDOUT: !168 = !DILocalVariable(arg: 1, scope: !167, type: !21)
// CHECK:STDOUT: !171 = !DILocation(line: 26, column: 5, scope: !167)
// CHECK:STDOUT: !172 = !DILocation(line: 24, column: 3, scope: !167)
// CHECK:STDOUT: !173 = !{!168}
// CHECK:STDOUT:
