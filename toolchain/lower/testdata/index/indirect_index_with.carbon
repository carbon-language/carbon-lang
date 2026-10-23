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

fn StoreSubscript(s: Span, i: i32, v: i32) {
  s[i] = v;
}

fn LoadSubscript(s: Span, i: i32) -> i32 {
  return s[i];
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'indirect_index_with.carbon'
// CHECK:STDOUT: source_filename = "indirect_index_with.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @"_CRef.Span.Main:IndirectIndexWith.55456748a8a2eb70.Core"(ptr %self, i32 %subscript) #0 !dbg !18 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc19_17.1.ptr = getelementptr inbounds nuw { ptr }, ptr %self, i32 0, i32 0, !dbg !25
// CHECK:STDOUT:   %.loc19_17.2 = load ptr, ptr %.loc19_17.1.ptr, align 8, !dbg !25
// CHECK:STDOUT:   ret ptr %.loc19_17.2, !dbg !27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CStore.Main(ptr %s, i32 %v) #0 !dbg !29 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Span.as.IndirectIndexWith.impl.Ref.call = call ptr @"_CRef.Span.Main:IndirectIndexWith.55456748a8a2eb70.Core"(ptr %s, i32 0), !dbg !34
// CHECK:STDOUT:   store i32 %v, ptr %Span.as.IndirectIndexWith.impl.Ref.call, align 4, !dbg !34
// CHECK:STDOUT:   ret void, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CLoad.Main(ptr %s) #0 !dbg !38 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Span.as.IndirectIndexWith.impl.Ref.call = call ptr @"_CRef.Span.Main:IndirectIndexWith.55456748a8a2eb70.Core"(ptr %s, i32 0), !dbg !43
// CHECK:STDOUT:   %.loc28_47 = load i32, ptr %Span.as.IndirectIndexWith.impl.Ref.call, align 4, !dbg !43
// CHECK:STDOUT:   ret i32 %.loc28_47, !dbg !44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CLoadViaAt.Main(ptr %s) #0 !dbg !46 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %T.as_type.as.IndexWith.impl.At.call = call i32 @"_CAt.4b18d5ec6f2c7a47:IndexWith.eb057aa32837c84e.Core.7d2c1205e4a4eaa2"(ptr %s, i32 0), !dbg !51
// CHECK:STDOUT:   ret i32 %T.as_type.as.IndexWith.impl.At.call, !dbg !52
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CStoreSubscript.Main(ptr %s, i32 %i, i32 %v) #0 !dbg !54 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Span.as.IndirectIndexWith.impl.Ref.call = call ptr @"_CRef.Span.Main:IndirectIndexWith.55456748a8a2eb70.Core"(ptr %s, i32 %i), !dbg !61
// CHECK:STDOUT:   store i32 %v, ptr %Span.as.IndirectIndexWith.impl.Ref.call, align 4, !dbg !61
// CHECK:STDOUT:   ret void, !dbg !64
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CLoadSubscript.Main(ptr %s, i32 %i) #0 !dbg !66 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Span.as.IndirectIndexWith.impl.Ref.call = call ptr @"_CRef.Span.Main:IndirectIndexWith.55456748a8a2eb70.Core"(ptr %s, i32 %i), !dbg !73
// CHECK:STDOUT:   %.loc40 = load i32, ptr %Span.as.IndirectIndexWith.impl.Ref.call, align 4, !dbg !73
// CHECK:STDOUT:   ret i32 %.loc40, !dbg !75
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CAt.4b18d5ec6f2c7a47:IndexWith.eb057aa32837c84e.Core.7d2c1205e4a4eaa2"(ptr %self, i32 %subscript) #0 !dbg !78 {
// CHECK:STDOUT:   %1 = call ptr @"_CRef.Span.Main:IndirectIndexWith.55456748a8a2eb70.Core"(ptr %self, i32 %subscript), !dbg !85
// CHECK:STDOUT:   %2 = load i32, ptr %1, align 4, !dbg !85
// CHECK:STDOUT:   ret i32 %2, !dbg !87
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
// CHECK:STDOUT: !14 = !{null, !7, !6, !6}
// CHECK:STDOUT: !15 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !16 = !{!6, !7, !6}
// CHECK:STDOUT: !17 = !DISubroutineType(types: !16)
// CHECK:STDOUT: !18 = distinct !DISubprogram(name: "Ref", linkageName: "_CRef.Span.Main:IndirectIndexWith.55456748a8a2eb70.Core", scope: null, file: !4, line: 18, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !28)
// CHECK:STDOUT: !19 = !DILocalVariable(arg: 1, scope: !18, type: !7)
// CHECK:STDOUT: !20 = !DILocalVariable(arg: 2, scope: !18, type: !6)
// CHECK:STDOUT: !25 = !DILocation(line: 19, column: 13, scope: !18)
// CHECK:STDOUT: !27 = !DILocation(line: 19, column: 5, scope: !18)
// CHECK:STDOUT: !28 = !{!19, !20}
// CHECK:STDOUT: !29 = distinct !DISubprogram(name: "Store", linkageName: "_CStore.Main", scope: null, file: !4, line: 23, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !37)
// CHECK:STDOUT: !30 = !DILocalVariable(arg: 1, scope: !29, type: !7)
// CHECK:STDOUT: !31 = !DILocalVariable(arg: 2, scope: !29, type: !6)
// CHECK:STDOUT: !34 = !DILocation(line: 24, column: 3, scope: !29)
// CHECK:STDOUT: !36 = !DILocation(line: 23, column: 1, scope: !29)
// CHECK:STDOUT: !37 = !{!30, !31}
// CHECK:STDOUT: !38 = distinct !DISubprogram(name: "Load", linkageName: "_CLoad.Main", scope: null, file: !4, line: 27, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !45)
// CHECK:STDOUT: !39 = !DILocalVariable(arg: 1, scope: !38, type: !7)
// CHECK:STDOUT: !43 = !DILocation(line: 28, column: 10, scope: !38)
// CHECK:STDOUT: !44 = !DILocation(line: 28, column: 3, scope: !38)
// CHECK:STDOUT: !45 = !{!39}
// CHECK:STDOUT: !46 = distinct !DISubprogram(name: "LoadViaAt", linkageName: "_CLoadViaAt.Main", scope: null, file: !4, line: 31, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !53)
// CHECK:STDOUT: !47 = !DILocalVariable(arg: 1, scope: !46, type: !7)
// CHECK:STDOUT: !51 = !DILocation(line: 32, column: 10, scope: !46)
// CHECK:STDOUT: !52 = !DILocation(line: 32, column: 3, scope: !46)
// CHECK:STDOUT: !53 = !{!47}
// CHECK:STDOUT: !54 = distinct !DISubprogram(name: "StoreSubscript", linkageName: "_CStoreSubscript.Main", scope: null, file: !4, line: 35, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !65)
// CHECK:STDOUT: !55 = !DILocalVariable(arg: 1, scope: !54, type: !7)
// CHECK:STDOUT: !56 = !DILocalVariable(arg: 2, scope: !54, type: !6)
// CHECK:STDOUT: !57 = !DILocalVariable(arg: 3, scope: !54, type: !6)
// CHECK:STDOUT: !61 = !DILocation(line: 36, column: 3, scope: !54)
// CHECK:STDOUT: !64 = !DILocation(line: 35, column: 1, scope: !54)
// CHECK:STDOUT: !65 = !{!55, !56, !57}
// CHECK:STDOUT: !66 = distinct !DISubprogram(name: "LoadSubscript", linkageName: "_CLoadSubscript.Main", scope: null, file: !4, line: 39, type: !17, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !76)
// CHECK:STDOUT: !67 = !DILocalVariable(arg: 1, scope: !66, type: !7)
// CHECK:STDOUT: !68 = !DILocalVariable(arg: 2, scope: !66, type: !6)
// CHECK:STDOUT: !73 = !DILocation(line: 40, column: 10, scope: !66)
// CHECK:STDOUT: !75 = !DILocation(line: 40, column: 3, scope: !66)
// CHECK:STDOUT: !76 = !{!67, !68}
// CHECK:STDOUT: !77 = !DIFile(filename: "{{.*}}/prelude/operators/index.carbon", directory: "")
// CHECK:STDOUT: !78 = distinct !DISubprogram(name: "At", linkageName: "_CAt.4b18d5ec6f2c7a47:IndexWith.eb057aa32837c84e.Core.7d2c1205e4a4eaa2", scope: null, file: !77, line: 25, type: !17, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !88)
// CHECK:STDOUT: !79 = !DILocalVariable(arg: 1, scope: !78, type: !7)
// CHECK:STDOUT: !80 = !DILocalVariable(arg: 2, scope: !78, type: !6)
// CHECK:STDOUT: !85 = !DILocation(line: 26, column: 12, scope: !78)
// CHECK:STDOUT: !87 = !DILocation(line: 26, column: 5, scope: !78)
// CHECK:STDOUT: !88 = !{!79, !80}
// CHECK:STDOUT:
