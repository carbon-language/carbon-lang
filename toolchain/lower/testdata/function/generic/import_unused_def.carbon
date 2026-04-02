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
  fn Op[self: Self]() -> Self {
    return (self as i32).(Core.Copy.Op)() as Foo;
  }
}

fn GenericF[T:! type](unused x: T) -> Foo {
  var f: Foo = (0 as i32) as Foo;
  return f;
}

// --- b.carbon
library "b";
import library "a";

fn Main() {
  GenericF(0);
}

// CHECK:STDOUT: ; ModuleID = 'a.carbon'
// CHECK:STDOUT: source_filename = "a.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @"_COp.Foo.Main:Copy.Core"(i32 %self) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %self, !dbg !10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.a9f4dd835fff2161:core.Destroy.Core"(ptr %self) #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "a.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Foo.Main:Copy.Core", scope: null, file: !3, line: 12, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !8)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7, !7}
// CHECK:STDOUT: !7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !8 = !{!9}
// CHECK:STDOUT: !9 = !DILocalVariable(arg: 1, scope: !4, type: !7)
// CHECK:STDOUT: !10 = !DILocation(line: 13, column: 5, scope: !4)
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "Op", linkageName: "_COp.a9f4dd835fff2161:core.Destroy.Core", scope: null, file: !3, line: 18, type: !12, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !14)
// CHECK:STDOUT: !12 = !DISubroutineType(types: !13)
// CHECK:STDOUT: !13 = !{null, !7}
// CHECK:STDOUT: !14 = !{!15}
// CHECK:STDOUT: !15 = !DILocalVariable(arg: 1, scope: !11, type: !7)
// CHECK:STDOUT: !16 = !DILocation(line: 18, column: 3, scope: !11)
// CHECK:STDOUT: ; ModuleID = 'b.carbon'
// CHECK:STDOUT: source_filename = "b.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMain.Main() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %GenericF.call = call i32 @_CGenericF.Main.084a2801e0f5d754({} zeroinitializer), !dbg !7
// CHECK:STDOUT:   ret void, !dbg !8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CGenericF.Main.084a2801e0f5d754({} %x) #0 !dbg !9 {
// CHECK:STDOUT:   %1 = alloca i32, align 4, !dbg !17
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %1), !dbg !17
// CHECK:STDOUT:   %2 = call i32 @"_COp.Foo.Main:Copy.Core"(i32 0), !dbg !18
// CHECK:STDOUT:   store i32 %2, ptr %1, align 4, !dbg !17
// CHECK:STDOUT:   %3 = load i32, ptr %1, align 4, !dbg !19
// CHECK:STDOUT:   %4 = call i32 @"_COp.Foo.Main:Copy.Core"(i32 %3), !dbg !19
// CHECK:STDOUT:   call void @"_COp.a9f4dd835fff2161:core.Destroy.Core"(ptr %1), !dbg !17
// CHECK:STDOUT:   ret i32 %4, !dbg !20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @"_COp.Foo.Main:Copy.Core"(i32)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.a9f4dd835fff2161:core.Destroy.Core"(ptr)
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
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "b.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Main", linkageName: "_CMain.Main", scope: null, file: !3, line: 4, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DILocation(line: 5, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 4, column: 1, scope: !4)
// CHECK:STDOUT: !9 = distinct !DISubprogram(name: "GenericF", linkageName: "_CGenericF.Main.084a2801e0f5d754", scope: null, file: !10, line: 17, type: !11, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !15)
// CHECK:STDOUT: !10 = !DIFile(filename: "a.carbon", directory: "")
// CHECK:STDOUT: !11 = !DISubroutineType(types: !12)
// CHECK:STDOUT: !12 = !{!13, !14}
// CHECK:STDOUT: !13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !15 = !{!16}
// CHECK:STDOUT: !16 = !DILocalVariable(arg: 1, scope: !9, type: !14)
// CHECK:STDOUT: !17 = !DILocation(line: 18, column: 3, scope: !9)
// CHECK:STDOUT: !18 = !DILocation(line: 18, column: 16, scope: !9)
// CHECK:STDOUT: !19 = !DILocation(line: 19, column: 10, scope: !9)
// CHECK:STDOUT: !20 = !DILocation(line: 19, column: 3, scope: !9)
