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
// CHECK:STDOUT: define i32 @"_CConvert.IntWrapper.Main:ImplicitAs.d727653b20be66f9.Core"(ptr %self) #0 !dbg !30 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc8_40.1.n = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !35
// CHECK:STDOUT:   %.loc8_40.2 = load i32, ptr %.loc8_40.1.n, align 4, !dbg !35
// CHECK:STDOUT:   ret i32 %.loc8_40.2, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConsume.Main(i32 %_) #0 !dbg !38 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CDoIt.Main() #0 !dbg !43 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %w.var = alloca { i32 }, align 4, !dbg !44
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %w.var), !dbg !44
// CHECK:STDOUT:   %.loc14_31.3.n = getelementptr inbounds nuw { i32 }, ptr %w.var, i32 0, i32 0, !dbg !45
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %w.var, ptr align 4 @IntWrapper.val.loc14_3.1, i64 4, i1 false), !dbg !44
// CHECK:STDOUT:   %IntWrapper.as.ImplicitAs.impl.Convert.call = call i32 @"_CConvert.IntWrapper.Main:ImplicitAs.d727653b20be66f9.Core"(ptr %w.var), !dbg !47
// CHECK:STDOUT:   call void @_CConsume.Main(i32 %IntWrapper.as.ImplicitAs.impl.Convert.call), !dbg !48
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.7adcc7f7765ae468(ptr %w.var), !dbg !44
// CHECK:STDOUT:   ret void, !dbg !49
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.8aa07e9a9e75fb3b:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !50 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc14_3.1.n = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !52
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.0aefc5da911f5051(ptr %.loc14_3.1.n), !dbg !52
// CHECK:STDOUT:   ret void, !dbg !52
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.aecc0ff76a42614a:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !54 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc14_3.1.n = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !56
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.0aefc5da911f5051(ptr %.loc14_3.1.n), !dbg !56
// CHECK:STDOUT:   ret void, !dbg !56
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.7adcc7f7765ae468(ptr %t) #0 !dbg !58 {
// CHECK:STDOUT:   call void @"_COp.aecc0ff76a42614a:core.SubobjectDestroy.Core"(ptr %t), !dbg !62
// CHECK:STDOUT:   ret void, !dbg !63
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.0aefc5da911f5051(ptr %t) #0 !dbg !65 {
// CHECK:STDOUT:   ret void, !dbg !70
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.0aefc5da911f5051, { 1, 0 }
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
// CHECK:STDOUT: !28 = !{null, !21}
// CHECK:STDOUT: !29 = !DISubroutineType(types: !28)
// CHECK:STDOUT: !30 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.IntWrapper.Main:ImplicitAs.d727653b20be66f9.Core", scope: null, file: !18, line: 8, type: !23, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !37)
// CHECK:STDOUT: !31 = !DILocalVariable(arg: 1, scope: !30, type: !21)
// CHECK:STDOUT: !35 = !DILocation(line: 8, column: 36, scope: !30)
// CHECK:STDOUT: !36 = !DILocation(line: 8, column: 29, scope: !30)
// CHECK:STDOUT: !37 = !{!31}
// CHECK:STDOUT: !38 = distinct !DISubprogram(name: "Consume", linkageName: "_CConsume.Main", scope: null, file: !18, line: 11, type: !25, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !42)
// CHECK:STDOUT: !39 = !DILocalVariable(arg: 1, scope: !38, type: !20)
// CHECK:STDOUT: !41 = !DILocation(line: 11, column: 1, scope: !38)
// CHECK:STDOUT: !42 = !{!39}
// CHECK:STDOUT: !43 = distinct !DISubprogram(name: "DoIt", linkageName: "_CDoIt.Main", scope: null, file: !18, line: 13, type: !27, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !44 = !DILocation(line: 14, column: 3, scope: !43)
// CHECK:STDOUT: !45 = !DILocation(line: 14, column: 23, scope: !43)
// CHECK:STDOUT: !47 = !DILocation(line: 15, column: 11, scope: !43)
// CHECK:STDOUT: !48 = !DILocation(line: 15, column: 3, scope: !43)
// CHECK:STDOUT: !49 = !DILocation(line: 13, column: 1, scope: !43)
// CHECK:STDOUT: !50 = distinct !DISubprogram(name: "Op", linkageName: "_COp.8aa07e9a9e75fb3b:core.SubobjectDestroy.Core", scope: null, file: !18, line: 14, type: !29, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !53)
// CHECK:STDOUT: !51 = !DILocalVariable(arg: 1, scope: !50, type: !21)
// CHECK:STDOUT: !52 = !DILocation(line: 14, column: 3, scope: !50)
// CHECK:STDOUT: !53 = !{!51}
// CHECK:STDOUT: !54 = distinct !DISubprogram(name: "Op", linkageName: "_COp.aecc0ff76a42614a:core.SubobjectDestroy.Core", scope: null, file: !18, line: 14, type: !29, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !57)
// CHECK:STDOUT: !55 = !DILocalVariable(arg: 1, scope: !54, type: !21)
// CHECK:STDOUT: !56 = !DILocation(line: 14, column: 3, scope: !54)
// CHECK:STDOUT: !57 = !{!55}
// CHECK:STDOUT: !58 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.7adcc7f7765ae468", scope: null, file: !10, line: 26, type: !29, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !64)
// CHECK:STDOUT: !59 = !DILocalVariable(arg: 1, scope: !58, type: !21)
// CHECK:STDOUT: !62 = !DILocation(line: 28, column: 5, scope: !58)
// CHECK:STDOUT: !63 = !DILocation(line: 26, column: 1, scope: !58)
// CHECK:STDOUT: !64 = !{!59}
// CHECK:STDOUT: !65 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.0aefc5da911f5051", scope: null, file: !10, line: 26, type: !25, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !71)
// CHECK:STDOUT: !66 = !DILocalVariable(arg: 1, scope: !65, type: !20)
// CHECK:STDOUT: !70 = !DILocation(line: 26, column: 1, scope: !65)
// CHECK:STDOUT: !71 = !{!66}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'partial_class_to_partial_base.carbon'
// CHECK:STDOUT: source_filename = "partial_class_to_partial_base.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CF.Main(ptr %d) #0 !dbg !76 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_13.1.base = getelementptr inbounds nuw { {} }, ptr %d, i32 0, i32 0, !dbg !82
// CHECK:STDOUT:   ret ptr %.loc7_13.1.base, !dbg !82
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!73}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !72 = !DIFile(filename: "partial_class_to_partial_base.carbon", directory: "")
// CHECK:STDOUT: !73 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !72, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !74 = !{!21, !21}
// CHECK:STDOUT: !75 = !DISubroutineType(types: !74)
// CHECK:STDOUT: !76 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !72, line: 6, type: !75, spFlags: DISPFlagDefinition, unit: !73, retainedNodes: !83)
// CHECK:STDOUT: !77 = !DILocalVariable(arg: 1, scope: !76, type: !21)
// CHECK:STDOUT: !82 = !DILocation(line: 7, column: 5, scope: !76)
// CHECK:STDOUT: !83 = !{!77}
// CHECK:STDOUT:
