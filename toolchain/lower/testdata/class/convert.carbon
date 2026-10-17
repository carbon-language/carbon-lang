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

// --- class_to_int.carbon
library "[[@TEST_NAME]]";

class IntWrapper {
  var n: i32;
}

impl IntWrapper as Core.ImplicitAs(i32) {
  fn Convert(self) -> i32 { return self.n; }
}

fn Consume(_: i32) {}

fn DoIt() {
  var w: IntWrapper = {.n = 42};
  Consume(w);
}

// --- partial_class_to_partial_base.carbon
library "[[@TEST_NAME]]";

base class B {}
base class D { extend base: B; }

fn F(ref d: partial D) -> ref partial B {
    return d;
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'class_to_int.carbon'
// CHECK:STDOUT: source_filename = "class_to_int.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @IntWrapper.val.loc14_3.1 = internal constant { i32 } { i32 42 }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @"_CConvert.IntWrapper.Main:ImplicitAs.55456748a8a2eb70.Core"(ptr %self) #0 !dbg !28 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc8_40.1.n = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !33
// CHECK:STDOUT:   %.loc8_40.2 = load i32, ptr %.loc8_40.1.n, align 4, !dbg !33
// CHECK:STDOUT:   ret i32 %.loc8_40.2, !dbg !34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConsume.Main(i32 %_) #0 !dbg !36 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CDoIt.Main() #0 !dbg !41 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %w.var = alloca { i32 }, align 4, !dbg !42
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %w.var), !dbg !42
// CHECK:STDOUT:   %.loc14_31.3.n = getelementptr inbounds nuw { i32 }, ptr %w.var, i32 0, i32 0, !dbg !43
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %w.var, ptr align 4 @IntWrapper.val.loc14_3.1, i64 4, i1 false), !dbg !42
// CHECK:STDOUT:   %IntWrapper.as.ImplicitAs.impl.Convert.call = call i32 @"_CConvert.IntWrapper.Main:ImplicitAs.55456748a8a2eb70.Core"(ptr %w.var), !dbg !45
// CHECK:STDOUT:   call void @_CConsume.Main(i32 %IntWrapper.as.ImplicitAs.impl.Convert.call), !dbg !46
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.bca7988aa43d3b33(ptr %w.var), !dbg !42
// CHECK:STDOUT:   ret void, !dbg !49
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.bca7988aa43d3b33(ptr %t) #0 !dbg !50 {
// CHECK:STDOUT:   ret void, !dbg !55
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !18 = !DIFile(filename: "class_to_int.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{!20, !21}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !24 = !{null, !20}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = !{null}
// CHECK:STDOUT: !27 = !DISubroutineType(types: !26)
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.IntWrapper.Main:ImplicitAs.55456748a8a2eb70.Core", scope: null, file: !18, line: 8, type: !23, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !35)
// CHECK:STDOUT: !29 = !DILocalVariable(arg: 1, scope: !28, type: !21)
// CHECK:STDOUT: !33 = !DILocation(line: 8, column: 36, scope: !28)
// CHECK:STDOUT: !34 = !DILocation(line: 8, column: 29, scope: !28)
// CHECK:STDOUT: !35 = !{!29}
// CHECK:STDOUT: !36 = distinct !DISubprogram(name: "Consume", linkageName: "_CConsume.Main", scope: null, file: !18, line: 11, type: !25, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !40)
// CHECK:STDOUT: !37 = !DILocalVariable(arg: 1, scope: !36, type: !20)
// CHECK:STDOUT: !39 = !DILocation(line: 11, column: 1, scope: !36)
// CHECK:STDOUT: !40 = !{!37}
// CHECK:STDOUT: !41 = distinct !DISubprogram(name: "DoIt", linkageName: "_CDoIt.Main", scope: null, file: !18, line: 13, type: !27, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !42 = !DILocation(line: 14, column: 3, scope: !41)
// CHECK:STDOUT: !43 = !DILocation(line: 14, column: 23, scope: !41)
// CHECK:STDOUT: !45 = !DILocation(line: 15, column: 11, scope: !41)
// CHECK:STDOUT: !46 = !DILocation(line: 15, column: 3, scope: !41)
// CHECK:STDOUT: !47 = !{null, !21}
// CHECK:STDOUT: !48 = !DISubroutineType(types: !47)
// CHECK:STDOUT: !49 = !DILocation(line: 13, column: 1, scope: !41)
// CHECK:STDOUT: !50 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.bca7988aa43d3b33", scope: null, file: !10, line: 30, type: !48, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !56)
// CHECK:STDOUT: !51 = !DILocalVariable(arg: 1, scope: !50, type: !21)
// CHECK:STDOUT: !55 = !DILocation(line: 30, column: 1, scope: !50)
// CHECK:STDOUT: !56 = !{!51}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'partial_class_to_partial_base.carbon'
// CHECK:STDOUT: source_filename = "partial_class_to_partial_base.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CF.Main(ptr %d) #0 !dbg !61 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_13.1.base = getelementptr inbounds nuw { {} }, ptr %d, i32 0, i32 0, !dbg !67
// CHECK:STDOUT:   ret ptr %.loc7_13.1.base, !dbg !67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!58}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !57 = !DIFile(filename: "partial_class_to_partial_base.carbon", directory: "")
// CHECK:STDOUT: !58 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !57, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !59 = !{!21, !21}
// CHECK:STDOUT: !60 = !DISubroutineType(types: !59)
// CHECK:STDOUT: !61 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !57, line: 6, type: !60, spFlags: DISPFlagDefinition, unit: !58, retainedNodes: !68)
// CHECK:STDOUT: !62 = !DILocalVariable(arg: 1, scope: !61, type: !21)
// CHECK:STDOUT: !67 = !DILocation(line: 7, column: 5, scope: !61)
// CHECK:STDOUT: !68 = !{!62}
// CHECK:STDOUT:
