// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/class/convert.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/class/convert.carbon

class IntWrapper {
  var n: i32;
}

impl IntWrapper as Core.ImplicitAs(i32) {
  fn Convert[self: Self]() -> i32 { return self.n; }
}

fn Consume(_: i32) {}

fn DoIt() {
  var w: IntWrapper = {.n = 42};
  Consume(w);
}

// CHECK:STDOUT: ; ModuleID = 'convert.carbon'
// CHECK:STDOUT: source_filename = "convert.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @IntWrapper.val.loc24_3 = internal constant { i32 } { i32 42 }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @"_CConvert.IntWrapper.Main:ImplicitAs.c8fd090a2e741f70.Core"(ptr %self) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc18_48.1.n = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !11
// CHECK:STDOUT:   %.loc18_48.2 = load i32, ptr %.loc18_48.1.n, align 4, !dbg !11
// CHECK:STDOUT:   ret i32 %.loc18_48.2, !dbg !12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConsume.Main(i32 %_) #0 !dbg !13 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CDoIt.Main() #0 !dbg !19 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %w.var = alloca { i32 }, align 4, !dbg !22
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %w.var), !dbg !22
// CHECK:STDOUT:   %.loc24_31.3.n = getelementptr inbounds nuw { i32 }, ptr %w.var, i32 0, i32 0, !dbg !23
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %w.var, ptr align 4 @IntWrapper.val.loc24_3, i64 4, i1 false), !dbg !22
// CHECK:STDOUT:   %IntWrapper.as.ImplicitAs.impl.Convert.call = call i32 @"_CConvert.IntWrapper.Main:ImplicitAs.c8fd090a2e741f70.Core"(ptr %w.var), !dbg !24
// CHECK:STDOUT:   call void @_CConsume.Main(i32 %IntWrapper.as.ImplicitAs.impl.Convert.call), !dbg !25
// CHECK:STDOUT:   call void @"_COp.4dd4d21939fc07f5:core.Destroy.Core"(ptr %w.var), !dbg !22
// CHECK:STDOUT:   ret void, !dbg !26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.94e3ca86614aba5d:core.Destroy.Core"(ptr %self) #0 !dbg !27 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.49ee0ef9455d99da:core.Destroy.Core"(ptr %self) #0 !dbg !31 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.4dd4d21939fc07f5:core.Destroy.Core"(ptr %self) #0 !dbg !37 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
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
// CHECK:STDOUT: !3 = !DIFile(filename: "convert.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.IntWrapper.Main:ImplicitAs.c8fd090a2e741f70.Core", scope: null, file: !3, line: 18, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !9)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7, !8}
// CHECK:STDOUT: !7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !9 = !{!10}
// CHECK:STDOUT: !10 = !DILocalVariable(arg: 1, scope: !4, type: !8)
// CHECK:STDOUT: !11 = !DILocation(line: 18, column: 44, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 18, column: 37, scope: !4)
// CHECK:STDOUT: !13 = distinct !DISubprogram(name: "Consume", linkageName: "_CConsume.Main", scope: null, file: !3, line: 21, type: !14, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !16)
// CHECK:STDOUT: !14 = !DISubroutineType(types: !15)
// CHECK:STDOUT: !15 = !{null, !7}
// CHECK:STDOUT: !16 = !{!17}
// CHECK:STDOUT: !17 = !DILocalVariable(arg: 1, scope: !13, type: !7)
// CHECK:STDOUT: !18 = !DILocation(line: 21, column: 1, scope: !13)
// CHECK:STDOUT: !19 = distinct !DISubprogram(name: "DoIt", linkageName: "_CDoIt.Main", scope: null, file: !3, line: 23, type: !20, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !20 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !21 = !{null}
// CHECK:STDOUT: !22 = !DILocation(line: 24, column: 3, scope: !19)
// CHECK:STDOUT: !23 = !DILocation(line: 24, column: 23, scope: !19)
// CHECK:STDOUT: !24 = !DILocation(line: 25, column: 11, scope: !19)
// CHECK:STDOUT: !25 = !DILocation(line: 25, column: 3, scope: !19)
// CHECK:STDOUT: !26 = !DILocation(line: 23, column: 1, scope: !19)
// CHECK:STDOUT: !27 = distinct !DISubprogram(name: "Op", linkageName: "_COp.94e3ca86614aba5d:core.Destroy.Core", scope: null, file: !3, line: 24, type: !14, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !28)
// CHECK:STDOUT: !28 = !{!29}
// CHECK:STDOUT: !29 = !DILocalVariable(arg: 1, scope: !27, type: !7)
// CHECK:STDOUT: !30 = !DILocation(line: 24, column: 3, scope: !27)
// CHECK:STDOUT: !31 = distinct !DISubprogram(name: "Op", linkageName: "_COp.49ee0ef9455d99da:core.Destroy.Core", scope: null, file: !3, line: 24, type: !32, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !34)
// CHECK:STDOUT: !32 = !DISubroutineType(types: !33)
// CHECK:STDOUT: !33 = !{null, !8}
// CHECK:STDOUT: !34 = !{!35}
// CHECK:STDOUT: !35 = !DILocalVariable(arg: 1, scope: !31, type: !8)
// CHECK:STDOUT: !36 = !DILocation(line: 24, column: 3, scope: !31)
// CHECK:STDOUT: !37 = distinct !DISubprogram(name: "Op", linkageName: "_COp.4dd4d21939fc07f5:core.Destroy.Core", scope: null, file: !3, line: 24, type: !32, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !38)
// CHECK:STDOUT: !38 = !{!39}
// CHECK:STDOUT: !39 = !DILocalVariable(arg: 1, scope: !37, type: !8)
// CHECK:STDOUT: !40 = !DILocation(line: 24, column: 3, scope: !37)
