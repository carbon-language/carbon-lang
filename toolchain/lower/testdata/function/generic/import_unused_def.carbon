// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/generic/import_unused_def.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/generic/import_unused_def.carbon

// --- a.carbon
library "a";

class Foo {
  adapt i32;
}

// This will be instantiated in "a", but unused here. Meanwhile the
// instantiation of "F" in "b" can create a use, without emitting a definition.
// This test verifies how the emitted definition is marked. Right now it is:
// `_COp.a9f4dd835fff2161:core.Destroy.Core`
impl Foo as Core.Copy {
  fn Op(self) -> Self {
    return (self as i32).(Core.Copy.Op)() as Foo;
  }
}

fn GenericF[T: type](unused x: T) -> Foo {
  var f: Foo = (0 as i32) as Foo;
  return f;
}

// --- b.carbon
library "b";
import library "a";

fn Main() {
  GenericF(0);
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'a.carbon'
// CHECK:STDOUT: source_filename = "a.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @"_COp.Foo.Main:Copy.Core"(i32 %self) #0 !dbg !25 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %self, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.70db2d92ffddbef7:core.Destroy.Core"(ptr %self) #0 !dbg !34 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.cdca8898aabbc885:core.Destroy.Core"(ptr %self) #0 !dbg !38 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !18 = !DIFile(filename: "a.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !{!20, !20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = !{null, !20}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Foo.Main:Copy.Core", scope: null, file: !18, line: 12, type: !22, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !33)
// CHECK:STDOUT: !26 = !DILocalVariable(arg: 1, scope: !25, type: !20)
// CHECK:STDOUT: !32 = !DILocation(line: 13, column: 5, scope: !25)
// CHECK:STDOUT: !33 = !{!26}
// CHECK:STDOUT: !34 = distinct !DISubprogram(name: "Op", linkageName: "_COp.70db2d92ffddbef7:core.Destroy.Core", scope: null, file: !18, line: 18, type: !24, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !37)
// CHECK:STDOUT: !35 = !DILocalVariable(arg: 1, scope: !34, type: !20)
// CHECK:STDOUT: !36 = !DILocation(line: 18, column: 3, scope: !34)
// CHECK:STDOUT: !37 = !{!35}
// CHECK:STDOUT: !38 = distinct !DISubprogram(name: "Op", linkageName: "_COp.cdca8898aabbc885:core.Destroy.Core", scope: null, file: !18, line: 18, type: !24, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !41)
// CHECK:STDOUT: !39 = !DILocalVariable(arg: 1, scope: !38, type: !20)
// CHECK:STDOUT: !40 = !DILocation(line: 18, column: 3, scope: !38)
// CHECK:STDOUT: !41 = !{!39}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'b.carbon'
// CHECK:STDOUT: source_filename = "b.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMain.Main() #0 !dbg !46 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %GenericF.call = call i32 @_CGenericF.Main.084a2801e0f5d754({} zeroinitializer), !dbg !47
// CHECK:STDOUT:   ret void, !dbg !51
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CGenericF.Main.084a2801e0f5d754({} %x) #0 !dbg !52 {
// CHECK:STDOUT:   %1 = alloca i32, align 4, !dbg !57
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %1), !dbg !57
// CHECK:STDOUT:   %2 = call i32 @"_COp.Foo.Main:Copy.Core"(i32 0), !dbg !58
// CHECK:STDOUT:   store i32 %2, ptr %1, align 4, !dbg !57
// CHECK:STDOUT:   %3 = load i32, ptr %1, align 4, !dbg !60
// CHECK:STDOUT:   %4 = call i32 @"_COp.Foo.Main:Copy.Core"(i32 %3), !dbg !60
// CHECK:STDOUT:   call void @"_COp.cdca8898aabbc885:core.Destroy.Core"(ptr %1), !dbg !57
// CHECK:STDOUT:   ret i32 %4, !dbg !61
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @"_COp.Foo.Main:Copy.Core"(i32)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.70db2d92ffddbef7:core.Destroy.Core"(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.cdca8898aabbc885:core.Destroy.Core"(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @"_COp.Foo.Main:Copy.Core", { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!43}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !18 = !DIFile(filename: "a.carbon", directory: "")
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !42 = !DIFile(filename: "b.carbon", directory: "")
// CHECK:STDOUT: !43 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !42, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !44 = !{null}
// CHECK:STDOUT: !45 = !DISubroutineType(types: !44)
// CHECK:STDOUT: !46 = distinct !DISubprogram(name: "Main", linkageName: "_CMain.Main", scope: null, file: !42, line: 4, type: !45, spFlags: DISPFlagDefinition, unit: !43)
// CHECK:STDOUT: !47 = !DILocation(line: 5, column: 3, scope: !46)
// CHECK:STDOUT: !48 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !49 = !{!20, !48}
// CHECK:STDOUT: !50 = !DISubroutineType(types: !49)
// CHECK:STDOUT: !51 = !DILocation(line: 4, column: 1, scope: !46)
// CHECK:STDOUT: !52 = distinct !DISubprogram(name: "GenericF", linkageName: "_CGenericF.Main.084a2801e0f5d754", scope: null, file: !18, line: 17, type: !50, spFlags: DISPFlagDefinition, unit: !43, retainedNodes: !62)
// CHECK:STDOUT: !53 = !DILocalVariable(arg: 1, scope: !52, type: !48)
// CHECK:STDOUT: !57 = !DILocation(line: 18, column: 3, scope: !52)
// CHECK:STDOUT: !58 = !DILocation(line: 18, column: 16, scope: !52)
// CHECK:STDOUT: !60 = !DILocation(line: 19, column: 10, scope: !52)
// CHECK:STDOUT: !61 = !DILocation(line: 19, column: 3, scope: !52)
// CHECK:STDOUT: !62 = !{!53}
// CHECK:STDOUT:
