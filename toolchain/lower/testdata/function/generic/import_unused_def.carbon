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
// CHECK:STDOUT: define i32 @"_COp.Foo.Main:Copy.Core"(i32 %self) #0 !dbg !23 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %self, !dbg !30
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
// CHECK:STDOUT: !23 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Foo.Main:Copy.Core", scope: null, file: !18, line: 12, type: !22, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !31)
// CHECK:STDOUT: !24 = !DILocalVariable(arg: 1, scope: !23, type: !20)
// CHECK:STDOUT: !30 = !DILocation(line: 13, column: 5, scope: !23)
// CHECK:STDOUT: !31 = !{!24}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'b.carbon'
// CHECK:STDOUT: source_filename = "b.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMain.Main() #0 !dbg !36 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %GenericF.call = call i32 @_CGenericF.Main.084a2801e0f5d754({} zeroinitializer), !dbg !37
// CHECK:STDOUT:   ret void, !dbg !41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CGenericF.Main.084a2801e0f5d754({} %x) #0 !dbg !42 {
// CHECK:STDOUT:   %1 = alloca i32, align 4, !dbg !47
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %1), !dbg !47
// CHECK:STDOUT:   %2 = call i32 @"_COp.Foo.Main:Copy.Core"(i32 0), !dbg !48
// CHECK:STDOUT:   store i32 %2, ptr %1, align 4, !dbg !47
// CHECK:STDOUT:   %3 = load i32, ptr %1, align 4, !dbg !50
// CHECK:STDOUT:   %4 = call i32 @"_COp.Foo.Main:Copy.Core"(i32 %3), !dbg !50
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.6339be073e904d6a(ptr %1), !dbg !47
// CHECK:STDOUT:   ret i32 %4, !dbg !53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @"_COp.Foo.Main:Copy.Core"(i32)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.6339be073e904d6a(ptr %t) #0 !dbg !55 {
// CHECK:STDOUT:   ret void, !dbg !60
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @"_COp.Foo.Main:Copy.Core", { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!33}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !18 = !DIFile(filename: "a.carbon", directory: "")
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !32 = !DIFile(filename: "b.carbon", directory: "")
// CHECK:STDOUT: !33 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !32, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !34 = !{null}
// CHECK:STDOUT: !35 = !DISubroutineType(types: !34)
// CHECK:STDOUT: !36 = distinct !DISubprogram(name: "Main", linkageName: "_CMain.Main", scope: null, file: !32, line: 4, type: !35, spFlags: DISPFlagDefinition, unit: !33)
// CHECK:STDOUT: !37 = !DILocation(line: 5, column: 3, scope: !36)
// CHECK:STDOUT: !38 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !39 = !{!20, !38}
// CHECK:STDOUT: !40 = !DISubroutineType(types: !39)
// CHECK:STDOUT: !41 = !DILocation(line: 4, column: 1, scope: !36)
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "GenericF", linkageName: "_CGenericF.Main.084a2801e0f5d754", scope: null, file: !18, line: 17, type: !40, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !54)
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 1, scope: !42, type: !38)
// CHECK:STDOUT: !47 = !DILocation(line: 18, column: 3, scope: !42)
// CHECK:STDOUT: !48 = !DILocation(line: 18, column: 16, scope: !42)
// CHECK:STDOUT: !50 = !DILocation(line: 19, column: 10, scope: !42)
// CHECK:STDOUT: !51 = !{null, !20}
// CHECK:STDOUT: !52 = !DISubroutineType(types: !51)
// CHECK:STDOUT: !53 = !DILocation(line: 19, column: 3, scope: !42)
// CHECK:STDOUT: !54 = !{!43}
// CHECK:STDOUT: !55 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.6339be073e904d6a", scope: null, file: !10, line: 32, type: !52, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !61)
// CHECK:STDOUT: !56 = !DILocalVariable(arg: 1, scope: !55, type: !20)
// CHECK:STDOUT: !60 = !DILocation(line: 32, column: 1, scope: !55)
// CHECK:STDOUT: !61 = !{!56}
// CHECK:STDOUT:
