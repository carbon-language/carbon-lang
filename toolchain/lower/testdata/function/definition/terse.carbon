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
fn ImportedGenericId[T: Core.Copy](x: T) => x;

// --- main.carbon
library "[[@TEST_NAME]]";

import library "lib";

fn LocalId(x: i32) => x;
fn LocalGenericId[T: Core.Copy](x: T) => x;

fn Unit() => ();

fn MakePair[T: Core.Copy, U: Core.Copy](x: T, y: U) => {.a = x, .b = y};

class C {
  fn Get(self: Self) => self.x;
  var x: i32;
}

fn Test(c: C) -> i32 {
  Unit();
  let p: {.a: i32, .b: i32} =
      MakePair(ImportedGenericId(ImportedId(c.Get())), LocalGenericId(LocalId(2)));
  return p.a;
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'lib.carbon'
// CHECK:STDOUT: source_filename = "lib.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CImportedId.Main(i32 %x) #0 !dbg !25 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %x, !dbg !28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!21}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIFile(filename: "lib.carbon", directory: "")
// CHECK:STDOUT: !21 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !20, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !22 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !23 = !{!22, !22}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "ImportedId", linkageName: "_CImportedId.Main", scope: null, file: !20, line: 3, type: !24, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !29)
// CHECK:STDOUT: !26 = !DILocalVariable(arg: 1, scope: !25, type: !22)
// CHECK:STDOUT: !28 = !DILocation(line: 3, column: 26, scope: !25)
// CHECK:STDOUT: !29 = !{!26}
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
// CHECK:STDOUT: define i32 @_CGet.C.Main(ptr %self) #0 !dbg !44 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc13_29.1.x = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !47
// CHECK:STDOUT:   %.loc13_29.2 = load i32, ptr %.loc13_29.1.x, align 4, !dbg !47
// CHECK:STDOUT:   ret i32 %.loc13_29.2, !dbg !47
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CTest.Main(ptr %c) #0 !dbg !49 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc20_82.5.temp = alloca { i32, i32 }, align 4, !dbg !60
// CHECK:STDOUT:   %struct = alloca { i32, i32 }, align 4, !dbg !60
// CHECK:STDOUT:   call void @_CUnit.Main(), !dbg !54
// CHECK:STDOUT:   %C.Get.call = call i32 @_CGet.C.Main(ptr %c), !dbg !55
// CHECK:STDOUT:   %ImportedId.call = call i32 @_CImportedId.Main(i32 %C.Get.call), !dbg !56
// CHECK:STDOUT:   %ImportedGenericId.call = call i32 @_CImportedGenericId.Main.2fad9574e050607e(i32 %ImportedId.call), !dbg !57
// CHECK:STDOUT:   %LocalId.call = call i32 @_CLocalId.Main(i32 2), !dbg !58
// CHECK:STDOUT:   %LocalGenericId.call = call i32 @_CLocalGenericId.Main.2fad9574e050607e(i32 %LocalId.call), !dbg !59
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc20_82.5.temp), !dbg !60
// CHECK:STDOUT:   call void @_CMakePair.Main.3eda18978e8f48e1(ptr %.loc20_82.5.temp, i32 %ImportedGenericId.call, i32 %LocalGenericId.call), !dbg !60
// CHECK:STDOUT:   %.loc20_82.7.a = getelementptr inbounds nuw { i32, i32 }, ptr %.loc20_82.5.temp, i32 0, i32 0, !dbg !60
// CHECK:STDOUT:   %.loc20_82.8 = load i32, ptr %.loc20_82.7.a, align 4, !dbg !60
// CHECK:STDOUT:   %.loc20_82.9.b = getelementptr inbounds nuw { i32, i32 }, ptr %.loc20_82.5.temp, i32 0, i32 1, !dbg !60
// CHECK:STDOUT:   %.loc20_82.10 = load i32, ptr %.loc20_82.9.b, align 4, !dbg !60
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %struct), !dbg !60
// CHECK:STDOUT:   %struct1 = getelementptr inbounds nuw { i32, i32 }, ptr %struct, i32 0, i32 0, !dbg !60
// CHECK:STDOUT:   store i32 %.loc20_82.8, ptr %struct1, align 4, !dbg !60
// CHECK:STDOUT:   %struct2 = getelementptr inbounds nuw { i32, i32 }, ptr %struct, i32 0, i32 1, !dbg !60
// CHECK:STDOUT:   store i32 %.loc20_82.10, ptr %struct2, align 4, !dbg !60
// CHECK:STDOUT:   %.loc21.a = getelementptr inbounds nuw { i32, i32 }, ptr %struct, i32 0, i32 0, !dbg !65
// CHECK:STDOUT:   %.loc21.a.load = load i32, ptr %.loc21.a, align 4, !dbg !65
// CHECK:STDOUT:   ret i32 %.loc21.a.load, !dbg !66
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @_CImportedId.Main(i32)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CImportedGenericId.Main.2fad9574e050607e(i32 %x) #0 !dbg !68 {
// CHECK:STDOUT:   ret i32 %x, !dbg !71
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CLocalGenericId.Main.2fad9574e050607e(i32 %x) #0 !dbg !73 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %x, !dbg !76
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CMakePair.Main.3eda18978e8f48e1(ptr sret({ i32, i32 }) %return, i32 %x, i32 %y) #0 !dbg !78 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc10_71.3.a = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 0, !dbg !83
// CHECK:STDOUT:   store i32 %x, ptr %.loc10_71.3.a, align 4, !dbg !83
// CHECK:STDOUT:   %.loc10_71.5.b = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 1, !dbg !83
// CHECK:STDOUT:   store i32 %y, ptr %.loc10_71.5.b, align 4, !dbg !83
// CHECK:STDOUT:   ret void, !dbg !83
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!31}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIFile(filename: "lib.carbon", directory: "")
// CHECK:STDOUT: !22 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !23 = !{!22, !22}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !30 = !DIFile(filename: "main.carbon", directory: "")
// CHECK:STDOUT: !31 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !30, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !32 = !{null}
// CHECK:STDOUT: !33 = !DISubroutineType(types: !32)
// CHECK:STDOUT: !34 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !35 = !{!22, !34}
// CHECK:STDOUT: !36 = !DISubroutineType(types: !35)
// CHECK:STDOUT: !37 = distinct !DISubprogram(name: "LocalId", linkageName: "_CLocalId.Main", scope: null, file: !30, line: 5, type: !24, spFlags: DISPFlagDefinition, unit: !31, retainedNodes: !41)
// CHECK:STDOUT: !38 = !DILocalVariable(arg: 1, scope: !37, type: !22)
// CHECK:STDOUT: !40 = !DILocation(line: 5, column: 23, scope: !37)
// CHECK:STDOUT: !41 = !{!38}
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "Unit", linkageName: "_CUnit.Main", scope: null, file: !30, line: 8, type: !33, spFlags: DISPFlagDefinition, unit: !31)
// CHECK:STDOUT: !43 = !DILocation(line: 8, column: 14, scope: !42)
// CHECK:STDOUT: !44 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.C.Main", scope: null, file: !30, line: 13, type: !36, spFlags: DISPFlagDefinition, unit: !31, retainedNodes: !48)
// CHECK:STDOUT: !45 = !DILocalVariable(arg: 1, scope: !44, type: !34)
// CHECK:STDOUT: !47 = !DILocation(line: 13, column: 25, scope: !44)
// CHECK:STDOUT: !48 = !{!45}
// CHECK:STDOUT: !49 = distinct !DISubprogram(name: "Test", linkageName: "_CTest.Main", scope: null, file: !30, line: 17, type: !36, spFlags: DISPFlagDefinition, unit: !31, retainedNodes: !67)
// CHECK:STDOUT: !50 = !DILocalVariable(arg: 1, scope: !49, type: !34)
// CHECK:STDOUT: !54 = !DILocation(line: 18, column: 3, scope: !49)
// CHECK:STDOUT: !55 = !DILocation(line: 20, column: 45, scope: !49)
// CHECK:STDOUT: !56 = !DILocation(line: 20, column: 34, scope: !49)
// CHECK:STDOUT: !57 = !DILocation(line: 20, column: 16, scope: !49)
// CHECK:STDOUT: !58 = !DILocation(line: 20, column: 71, scope: !49)
// CHECK:STDOUT: !59 = !DILocation(line: 20, column: 56, scope: !49)
// CHECK:STDOUT: !60 = !DILocation(line: 20, column: 7, scope: !49)
// CHECK:STDOUT: !61 = !{!34, !22, !22}
// CHECK:STDOUT: !62 = !DISubroutineType(types: !61)
// CHECK:STDOUT: !65 = !DILocation(line: 21, column: 10, scope: !49)
// CHECK:STDOUT: !66 = !DILocation(line: 21, column: 3, scope: !49)
// CHECK:STDOUT: !67 = !{!50}
// CHECK:STDOUT: !68 = distinct !DISubprogram(name: "ImportedGenericId", linkageName: "_CImportedGenericId.Main.2fad9574e050607e", scope: null, file: !20, line: 4, type: !24, spFlags: DISPFlagDefinition, unit: !31, retainedNodes: !72)
// CHECK:STDOUT: !69 = !DILocalVariable(arg: 1, scope: !68, type: !22)
// CHECK:STDOUT: !71 = !DILocation(line: 4, column: 45, scope: !68)
// CHECK:STDOUT: !72 = !{!69}
// CHECK:STDOUT: !73 = distinct !DISubprogram(name: "LocalGenericId", linkageName: "_CLocalGenericId.Main.2fad9574e050607e", scope: null, file: !30, line: 6, type: !24, spFlags: DISPFlagDefinition, unit: !31, retainedNodes: !77)
// CHECK:STDOUT: !74 = !DILocalVariable(arg: 1, scope: !73, type: !22)
// CHECK:STDOUT: !76 = !DILocation(line: 6, column: 42, scope: !73)
// CHECK:STDOUT: !77 = !{!74}
// CHECK:STDOUT: !78 = distinct !DISubprogram(name: "MakePair", linkageName: "_CMakePair.Main.3eda18978e8f48e1", scope: null, file: !30, line: 10, type: !62, spFlags: DISPFlagDefinition, unit: !31, retainedNodes: !86)
// CHECK:STDOUT: !79 = !DILocalVariable(arg: 1, scope: !78, type: !22)
// CHECK:STDOUT: !80 = !DILocalVariable(arg: 2, scope: !78, type: !22)
// CHECK:STDOUT: !83 = !DILocation(line: 10, column: 56, scope: !78)
// CHECK:STDOUT: !86 = !{!79, !80}
// CHECK:STDOUT:
