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
// CHECK:STDOUT: define i32 @_CLocalId.Main(i32 %x) #0 !dbg !37 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %x, !dbg !40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CUnit.Main() #0 !dbg !42 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !43
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMakePair.Main(ptr sret({ i32, i32 }) %return, i32 %x, i32 %y) #0 !dbg !44 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_47.3.a = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 0, !dbg !49
// CHECK:STDOUT:   store i32 %x, ptr %.loc9_47.3.a, align 4, !dbg !49
// CHECK:STDOUT:   %.loc9_47.5.b = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 1, !dbg !49
// CHECK:STDOUT:   store i32 %y, ptr %.loc9_47.5.b, align 4, !dbg !49
// CHECK:STDOUT:   ret void, !dbg !49
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CGet.C.Main(ptr %self) #0 !dbg !53 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc12_29.1.x = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !56
// CHECK:STDOUT:   %.loc12_29.2 = load i32, ptr %.loc12_29.1.x, align 4, !dbg !56
// CHECK:STDOUT:   ret i32 %.loc12_29.2, !dbg !56
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CTest.Main(ptr %c) #0 !dbg !58 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc18_71.1.temp = alloca { i32, i32 }, align 4, !dbg !67
// CHECK:STDOUT:   %struct = alloca { i32, i32 }, align 4, !dbg !67
// CHECK:STDOUT:   call void @_CUnit.Main(), !dbg !63
// CHECK:STDOUT:   %C.Get.call = call i32 @_CGet.C.Main(ptr %c), !dbg !64
// CHECK:STDOUT:   %ImportedId.call = call i32 @_CImportedId.Main(i32 %C.Get.call), !dbg !65
// CHECK:STDOUT:   %LocalId.call = call i32 @_CLocalId.Main(i32 2), !dbg !66
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc18_71.1.temp), !dbg !67
// CHECK:STDOUT:   call void @_CMakePair.Main(ptr %.loc18_71.1.temp, i32 %ImportedId.call, i32 %LocalId.call), !dbg !67
// CHECK:STDOUT:   %.loc18_71.3.a = getelementptr inbounds nuw { i32, i32 }, ptr %.loc18_71.1.temp, i32 0, i32 0, !dbg !67
// CHECK:STDOUT:   %.loc18_71.4 = load i32, ptr %.loc18_71.3.a, align 4, !dbg !67
// CHECK:STDOUT:   %.loc18_71.5.b = getelementptr inbounds nuw { i32, i32 }, ptr %.loc18_71.1.temp, i32 0, i32 1, !dbg !67
// CHECK:STDOUT:   %.loc18_71.6 = load i32, ptr %.loc18_71.5.b, align 4, !dbg !67
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %struct), !dbg !67
// CHECK:STDOUT:   %struct1 = getelementptr inbounds nuw { i32, i32 }, ptr %struct, i32 0, i32 0, !dbg !67
// CHECK:STDOUT:   store i32 %.loc18_71.4, ptr %struct1, align 4, !dbg !67
// CHECK:STDOUT:   %struct2 = getelementptr inbounds nuw { i32, i32 }, ptr %struct, i32 0, i32 1, !dbg !67
// CHECK:STDOUT:   store i32 %.loc18_71.6, ptr %struct2, align 4, !dbg !67
// CHECK:STDOUT:   %.loc19.a = getelementptr inbounds nuw { i32, i32 }, ptr %struct, i32 0, i32 0, !dbg !70
// CHECK:STDOUT:   %.loc19.a.load = load i32, ptr %.loc19.a, align 4, !dbg !70
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.ad01ed096e1f14e2(ptr %.loc18_71.1.temp), !dbg !67
// CHECK:STDOUT:   ret i32 %.loc19.a.load, !dbg !73
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @_CImportedId.Main(i32)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.ad01ed096e1f14e2(ptr %t) #0 !dbg !75 {
// CHECK:STDOUT:   ret void, !dbg !80
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
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
// CHECK:STDOUT: !37 = distinct !DISubprogram(name: "LocalId", linkageName: "_CLocalId.Main", scope: null, file: !28, line: 5, type: !22, spFlags: DISPFlagDefinition, unit: !29, retainedNodes: !41)
// CHECK:STDOUT: !38 = !DILocalVariable(arg: 1, scope: !37, type: !20)
// CHECK:STDOUT: !40 = !DILocation(line: 5, column: 23, scope: !37)
// CHECK:STDOUT: !41 = !{!38}
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "Unit", linkageName: "_CUnit.Main", scope: null, file: !28, line: 7, type: !31, spFlags: DISPFlagDefinition, unit: !29)
// CHECK:STDOUT: !43 = !DILocation(line: 7, column: 14, scope: !42)
// CHECK:STDOUT: !44 = distinct !DISubprogram(name: "MakePair", linkageName: "_CMakePair.Main", scope: null, file: !28, line: 9, type: !34, spFlags: DISPFlagDefinition, unit: !29, retainedNodes: !52)
// CHECK:STDOUT: !45 = !DILocalVariable(arg: 1, scope: !44, type: !20)
// CHECK:STDOUT: !46 = !DILocalVariable(arg: 2, scope: !44, type: !20)
// CHECK:STDOUT: !49 = !DILocation(line: 9, column: 32, scope: !44)
// CHECK:STDOUT: !52 = !{!45, !46}
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.C.Main", scope: null, file: !28, line: 12, type: !36, spFlags: DISPFlagDefinition, unit: !29, retainedNodes: !57)
// CHECK:STDOUT: !54 = !DILocalVariable(arg: 1, scope: !53, type: !32)
// CHECK:STDOUT: !56 = !DILocation(line: 12, column: 25, scope: !53)
// CHECK:STDOUT: !57 = !{!54}
// CHECK:STDOUT: !58 = distinct !DISubprogram(name: "Test", linkageName: "_CTest.Main", scope: null, file: !28, line: 16, type: !36, spFlags: DISPFlagDefinition, unit: !29, retainedNodes: !74)
// CHECK:STDOUT: !59 = !DILocalVariable(arg: 1, scope: !58, type: !32)
// CHECK:STDOUT: !63 = !DILocation(line: 17, column: 3, scope: !58)
// CHECK:STDOUT: !64 = !DILocation(line: 18, column: 51, scope: !58)
// CHECK:STDOUT: !65 = !DILocation(line: 18, column: 40, scope: !58)
// CHECK:STDOUT: !66 = !DILocation(line: 18, column: 61, scope: !58)
// CHECK:STDOUT: !67 = !DILocation(line: 18, column: 31, scope: !58)
// CHECK:STDOUT: !70 = !DILocation(line: 19, column: 10, scope: !58)
// CHECK:STDOUT: !71 = !{null, !32}
// CHECK:STDOUT: !72 = !DISubroutineType(types: !71)
// CHECK:STDOUT: !73 = !DILocation(line: 19, column: 3, scope: !58)
// CHECK:STDOUT: !74 = !{!59}
// CHECK:STDOUT: !75 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.ad01ed096e1f14e2", scope: null, file: !10, line: 32, type: !72, spFlags: DISPFlagDefinition, unit: !29, retainedNodes: !81)
// CHECK:STDOUT: !76 = !DILocalVariable(arg: 1, scope: !75, type: !32)
// CHECK:STDOUT: !80 = !DILocation(line: 32, column: 1, scope: !75)
// CHECK:STDOUT: !81 = !{!76}
// CHECK:STDOUT:
