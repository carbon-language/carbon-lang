// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/definition/terse.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/definition/terse.carbon

// --- lib.carbon
library "[[@TEST_NAME]]";

fn ImportedId(x: i32) => x;

// --- main.carbon
library "[[@TEST_NAME]]";

import library "lib";

fn LocalId(x: i32) => x;

fn Unit() => ();

fn MakePair(x: i32, y: i32) => {.a = x, .b = y};

class C {
  fn Get(self: Self) => self.x;
  var x: i32;
}

fn Test(c: C) -> i32 {
  Unit();
  let p: {.a: i32, .b: i32} = MakePair(ImportedId(c.Get()), LocalId(2));
  return p.a;
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'lib.carbon'
// CHECK:STDOUT: source_filename = "lib.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CImportedId.Main(i32 %x) #0 !dbg !23 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %x, !dbg !26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !18 = !DIFile(filename: "lib.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !{!20, !20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = distinct !DISubprogram(name: "ImportedId", linkageName: "_CImportedId.Main", scope: null, file: !18, line: 3, type: !22, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !27)
// CHECK:STDOUT: !24 = !DILocalVariable(arg: 1, scope: !23, type: !20)
// CHECK:STDOUT: !26 = !DILocation(line: 3, column: 26, scope: !23)
// CHECK:STDOUT: !27 = !{!24}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'main.carbon'
// CHECK:STDOUT: source_filename = "main.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CLocalId.Main(i32 %x) #0 !dbg !39 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %x, !dbg !42
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CUnit.Main() #0 !dbg !44 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMakePair.Main(ptr sret({ i32, i32 }) %return, i32 %x, i32 %y) #0 !dbg !46 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_47.3.a = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 0, !dbg !51
// CHECK:STDOUT:   store i32 %x, ptr %.loc9_47.3.a, align 4, !dbg !51
// CHECK:STDOUT:   %.loc9_47.5.b = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 1, !dbg !51
// CHECK:STDOUT:   store i32 %y, ptr %.loc9_47.5.b, align 4, !dbg !51
// CHECK:STDOUT:   ret void, !dbg !51
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CGet.C.Main(ptr %self) #0 !dbg !55 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc12_29.1.x = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !58
// CHECK:STDOUT:   %.loc12_29.2 = load i32, ptr %.loc12_29.1.x, align 4, !dbg !58
// CHECK:STDOUT:   ret i32 %.loc12_29.2, !dbg !58
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CTest.Main(ptr %c) #0 !dbg !60 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc18_71.1.temp = alloca { i32, i32 }, align 4, !dbg !69
// CHECK:STDOUT:   %struct = alloca { i32, i32 }, align 4, !dbg !69
// CHECK:STDOUT:   call void @_CUnit.Main(), !dbg !65
// CHECK:STDOUT:   %C.Get.call = call i32 @_CGet.C.Main(ptr %c), !dbg !66
// CHECK:STDOUT:   %ImportedId.call = call i32 @_CImportedId.Main(i32 %C.Get.call), !dbg !67
// CHECK:STDOUT:   %LocalId.call = call i32 @_CLocalId.Main(i32 2), !dbg !68
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc18_71.1.temp), !dbg !69
// CHECK:STDOUT:   call void @_CMakePair.Main(ptr %.loc18_71.1.temp, i32 %ImportedId.call, i32 %LocalId.call), !dbg !69
// CHECK:STDOUT:   %.loc18_71.3.a = getelementptr inbounds nuw { i32, i32 }, ptr %.loc18_71.1.temp, i32 0, i32 0, !dbg !69
// CHECK:STDOUT:   %.loc18_71.4 = load i32, ptr %.loc18_71.3.a, align 4, !dbg !69
// CHECK:STDOUT:   %.loc18_71.5.b = getelementptr inbounds nuw { i32, i32 }, ptr %.loc18_71.1.temp, i32 0, i32 1, !dbg !69
// CHECK:STDOUT:   %.loc18_71.6 = load i32, ptr %.loc18_71.5.b, align 4, !dbg !69
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %struct), !dbg !69
// CHECK:STDOUT:   %struct1 = getelementptr inbounds nuw { i32, i32 }, ptr %struct, i32 0, i32 0, !dbg !69
// CHECK:STDOUT:   store i32 %.loc18_71.4, ptr %struct1, align 4, !dbg !69
// CHECK:STDOUT:   %struct2 = getelementptr inbounds nuw { i32, i32 }, ptr %struct, i32 0, i32 1, !dbg !69
// CHECK:STDOUT:   store i32 %.loc18_71.6, ptr %struct2, align 4, !dbg !69
// CHECK:STDOUT:   %.loc19.a = getelementptr inbounds nuw { i32, i32 }, ptr %struct, i32 0, i32 0, !dbg !72
// CHECK:STDOUT:   %.loc19.a.load = load i32, ptr %.loc19.a, align 4, !dbg !72
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.ad01ed096e1f14e2(ptr %.loc18_71.1.temp), !dbg !69
// CHECK:STDOUT:   ret i32 %.loc19.a.load, !dbg !73
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @_CImportedId.Main(i32)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.2508ed710019eba8:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !75 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc18_71.1.b = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 1, !dbg !77
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %.loc18_71.1.b), !dbg !77
// CHECK:STDOUT:   %.loc18_71.4.a = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 0, !dbg !77
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %.loc18_71.4.a), !dbg !77
// CHECK:STDOUT:   ret void, !dbg !77
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.ad01ed096e1f14e2(ptr %t) #0 !dbg !81 {
// CHECK:STDOUT:   call void @"_COp.2508ed710019eba8:core.SubobjectDestroy.Core"(ptr %t), !dbg !85
// CHECK:STDOUT:   ret void, !dbg !86
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.f127f30143696d51(ptr %t) #0 !dbg !88 {
// CHECK:STDOUT:   ret void, !dbg !93
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.f127f30143696d51, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!29}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !{!20, !20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !28 = !DIFile(filename: "main.carbon", directory: "")
// CHECK:STDOUT: !29 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !28, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !30 = !{null}
// CHECK:STDOUT: !31 = !DISubroutineType(types: !30)
// CHECK:STDOUT: !32 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !33 = !{!32, !20, !20}
// CHECK:STDOUT: !34 = !DISubroutineType(types: !33)
// CHECK:STDOUT: !35 = !{!20, !32}
// CHECK:STDOUT: !36 = !DISubroutineType(types: !35)
// CHECK:STDOUT: !37 = !{null, !32}
// CHECK:STDOUT: !38 = !DISubroutineType(types: !37)
// CHECK:STDOUT: !39 = distinct !DISubprogram(name: "LocalId", linkageName: "_CLocalId.Main", scope: null, file: !28, line: 5, type: !22, spFlags: DISPFlagDefinition, unit: !29, retainedNodes: !43)
// CHECK:STDOUT: !40 = !DILocalVariable(arg: 1, scope: !39, type: !20)
// CHECK:STDOUT: !42 = !DILocation(line: 5, column: 23, scope: !39)
// CHECK:STDOUT: !43 = !{!40}
// CHECK:STDOUT: !44 = distinct !DISubprogram(name: "Unit", linkageName: "_CUnit.Main", scope: null, file: !28, line: 7, type: !31, spFlags: DISPFlagDefinition, unit: !29)
// CHECK:STDOUT: !45 = !DILocation(line: 7, column: 14, scope: !44)
// CHECK:STDOUT: !46 = distinct !DISubprogram(name: "MakePair", linkageName: "_CMakePair.Main", scope: null, file: !28, line: 9, type: !34, spFlags: DISPFlagDefinition, unit: !29, retainedNodes: !54)
// CHECK:STDOUT: !47 = !DILocalVariable(arg: 1, scope: !46, type: !20)
// CHECK:STDOUT: !48 = !DILocalVariable(arg: 2, scope: !46, type: !20)
// CHECK:STDOUT: !51 = !DILocation(line: 9, column: 32, scope: !46)
// CHECK:STDOUT: !54 = !{!47, !48}
// CHECK:STDOUT: !55 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.C.Main", scope: null, file: !28, line: 12, type: !36, spFlags: DISPFlagDefinition, unit: !29, retainedNodes: !59)
// CHECK:STDOUT: !56 = !DILocalVariable(arg: 1, scope: !55, type: !32)
// CHECK:STDOUT: !58 = !DILocation(line: 12, column: 25, scope: !55)
// CHECK:STDOUT: !59 = !{!56}
// CHECK:STDOUT: !60 = distinct !DISubprogram(name: "Test", linkageName: "_CTest.Main", scope: null, file: !28, line: 16, type: !36, spFlags: DISPFlagDefinition, unit: !29, retainedNodes: !74)
// CHECK:STDOUT: !61 = !DILocalVariable(arg: 1, scope: !60, type: !32)
// CHECK:STDOUT: !65 = !DILocation(line: 17, column: 3, scope: !60)
// CHECK:STDOUT: !66 = !DILocation(line: 18, column: 51, scope: !60)
// CHECK:STDOUT: !67 = !DILocation(line: 18, column: 40, scope: !60)
// CHECK:STDOUT: !68 = !DILocation(line: 18, column: 61, scope: !60)
// CHECK:STDOUT: !69 = !DILocation(line: 18, column: 31, scope: !60)
// CHECK:STDOUT: !72 = !DILocation(line: 19, column: 10, scope: !60)
// CHECK:STDOUT: !73 = !DILocation(line: 19, column: 3, scope: !60)
// CHECK:STDOUT: !74 = !{!61}
// CHECK:STDOUT: !75 = distinct !DISubprogram(name: "Op", linkageName: "_COp.2508ed710019eba8:core.SubobjectDestroy.Core", scope: null, file: !28, line: 18, type: !38, spFlags: DISPFlagDefinition, unit: !29, retainedNodes: !80)
// CHECK:STDOUT: !76 = !DILocalVariable(arg: 1, scope: !75, type: !32)
// CHECK:STDOUT: !77 = !DILocation(line: 18, column: 31, scope: !75)
// CHECK:STDOUT: !78 = !{null, !20}
// CHECK:STDOUT: !79 = !DISubroutineType(types: !78)
// CHECK:STDOUT: !80 = !{!76}
// CHECK:STDOUT: !81 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.ad01ed096e1f14e2", scope: null, file: !10, line: 32, type: !38, spFlags: DISPFlagDefinition, unit: !29, retainedNodes: !87)
// CHECK:STDOUT: !82 = !DILocalVariable(arg: 1, scope: !81, type: !32)
// CHECK:STDOUT: !85 = !DILocation(line: 34, column: 5, scope: !81)
// CHECK:STDOUT: !86 = !DILocation(line: 32, column: 1, scope: !81)
// CHECK:STDOUT: !87 = !{!82}
// CHECK:STDOUT: !88 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.f127f30143696d51", scope: null, file: !10, line: 32, type: !79, spFlags: DISPFlagDefinition, unit: !29, retainedNodes: !94)
// CHECK:STDOUT: !89 = !DILocalVariable(arg: 1, scope: !88, type: !20)
// CHECK:STDOUT: !93 = !DILocation(line: 32, column: 1, scope: !88)
// CHECK:STDOUT: !94 = !{!89}
// CHECK:STDOUT:
