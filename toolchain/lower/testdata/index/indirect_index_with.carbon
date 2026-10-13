// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/index/indirect_index_with.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/index/indirect_index_with.carbon

class Span {
  var ptr: i32*;
}

impl Span as Core.IndirectIndexWith(i32) where .ElementType = i32 {
  fn Ref(self, unused subscript: i32) -> ref i32 {
    return *self.ptr;
  }
}

fn Store(s: Span, v: i32) {
  s.(Core.IndirectIndexWith(i32).Ref)(0) = v;
}

fn Load(s: Span) -> i32 {
  return s.(Core.IndirectIndexWith(i32).Ref)(0);
}

fn LoadViaAt(s: Span) -> i32 {
  return s.(Core.IndexWith(i32).At)(0);
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'indirect_index_with.carbon'
// CHECK:STDOUT: source_filename = "indirect_index_with.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @"_CRef.Span.Main:IndirectIndexWith.d727653b20be66f9.Core"(ptr %self, i32 %subscript) #0 !dbg !14 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc19_17.1.ptr = getelementptr inbounds nuw { ptr }, ptr %self, i32 0, i32 0, !dbg !21
// CHECK:STDOUT:   %.loc19_17.2 = load ptr, ptr %.loc19_17.1.ptr, align 8, !dbg !21
// CHECK:STDOUT:   ret ptr %.loc19_17.2, !dbg !23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CStore.Main(ptr %s, i32 %v) #0 !dbg !25 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Span.as.IndirectIndexWith.impl.Ref.call = call ptr @"_CRef.Span.Main:IndirectIndexWith.d727653b20be66f9.Core"(ptr %s, i32 0), !dbg !30
// CHECK:STDOUT:   store i32 %v, ptr %Span.as.IndirectIndexWith.impl.Ref.call, align 4, !dbg !30
// CHECK:STDOUT:   ret void, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CLoad.Main(ptr %s) #0 !dbg !34 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Span.as.IndirectIndexWith.impl.Ref.call = call ptr @"_CRef.Span.Main:IndirectIndexWith.d727653b20be66f9.Core"(ptr %s, i32 0), !dbg !39
// CHECK:STDOUT:   %.loc28_47 = load i32, ptr %Span.as.IndirectIndexWith.impl.Ref.call, align 4, !dbg !39
// CHECK:STDOUT:   ret i32 %.loc28_47, !dbg !40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CLoadViaAt.Main(ptr %s) #0 !dbg !42 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %T.as_type.as.IndexWith.impl.At.call = call i32 @"_CAt.58585cd072850e1a:IndexWith.eb057aa32837c84e.Core.cf86281281909702"(ptr %s, i32 0), !dbg !47
// CHECK:STDOUT:   ret i32 %T.as_type.as.IndexWith.impl.At.call, !dbg !50
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CAt.58585cd072850e1a:IndexWith.eb057aa32837c84e.Core.cf86281281909702"(ptr %self, i32 %subscript) #0 !dbg !53 {
// CHECK:STDOUT:   %1 = call ptr @"_CRef.Span.Main:IndirectIndexWith.d727653b20be66f9.Core"(ptr %self, i32 %subscript), !dbg !60
// CHECK:STDOUT:   %2 = load i32, ptr %1, align 4, !dbg !60
// CHECK:STDOUT:   ret i32 %2, !dbg !62
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "indirect_index_with.carbon", directory: "")
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{!7, !7, !6}
// CHECK:STDOUT: !9 = !DISubroutineType(types: !8)
// CHECK:STDOUT: !10 = !{null, !7, !6}
// CHECK:STDOUT: !11 = !DISubroutineType(types: !10)
// CHECK:STDOUT: !12 = !{!6, !7}
// CHECK:STDOUT: !13 = !DISubroutineType(types: !12)
// CHECK:STDOUT: !14 = distinct !DISubprogram(name: "Ref", linkageName: "_CRef.Span.Main:IndirectIndexWith.d727653b20be66f9.Core", scope: null, file: !4, line: 18, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !24)
// CHECK:STDOUT: !15 = !DILocalVariable(arg: 1, scope: !14, type: !7)
// CHECK:STDOUT: !16 = !DILocalVariable(arg: 2, scope: !14, type: !6)
// CHECK:STDOUT: !21 = !DILocation(line: 19, column: 13, scope: !14)
// CHECK:STDOUT: !23 = !DILocation(line: 19, column: 5, scope: !14)
// CHECK:STDOUT: !24 = !{!15, !16}
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "Store", linkageName: "_CStore.Main", scope: null, file: !4, line: 23, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !33)
// CHECK:STDOUT: !26 = !DILocalVariable(arg: 1, scope: !25, type: !7)
// CHECK:STDOUT: !27 = !DILocalVariable(arg: 2, scope: !25, type: !6)
// CHECK:STDOUT: !30 = !DILocation(line: 24, column: 3, scope: !25)
// CHECK:STDOUT: !32 = !DILocation(line: 23, column: 1, scope: !25)
// CHECK:STDOUT: !33 = !{!26, !27}
// CHECK:STDOUT: !34 = distinct !DISubprogram(name: "Load", linkageName: "_CLoad.Main", scope: null, file: !4, line: 27, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !41)
// CHECK:STDOUT: !35 = !DILocalVariable(arg: 1, scope: !34, type: !7)
// CHECK:STDOUT: !39 = !DILocation(line: 28, column: 10, scope: !34)
// CHECK:STDOUT: !40 = !DILocation(line: 28, column: 3, scope: !34)
// CHECK:STDOUT: !41 = !{!35}
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "LoadViaAt", linkageName: "_CLoadViaAt.Main", scope: null, file: !4, line: 31, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !51)
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 1, scope: !42, type: !7)
// CHECK:STDOUT: !47 = !DILocation(line: 32, column: 10, scope: !42)
// CHECK:STDOUT: !48 = !{!6, !7, !6}
// CHECK:STDOUT: !49 = !DISubroutineType(types: !48)
// CHECK:STDOUT: !50 = !DILocation(line: 32, column: 3, scope: !42)
// CHECK:STDOUT: !51 = !{!43}
// CHECK:STDOUT: !52 = !DIFile(filename: "{{.*}}/prelude/operators/index.carbon", directory: "")
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "At", linkageName: "_CAt.58585cd072850e1a:IndexWith.eb057aa32837c84e.Core.cf86281281909702", scope: null, file: !52, line: 25, type: !49, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !63)
// CHECK:STDOUT: !54 = !DILocalVariable(arg: 1, scope: !53, type: !7)
// CHECK:STDOUT: !55 = !DILocalVariable(arg: 2, scope: !53, type: !6)
// CHECK:STDOUT: !60 = !DILocation(line: 26, column: 12, scope: !53)
// CHECK:STDOUT: !62 = !DILocation(line: 26, column: 5, scope: !53)
// CHECK:STDOUT: !63 = !{!54, !55}
// CHECK:STDOUT:
