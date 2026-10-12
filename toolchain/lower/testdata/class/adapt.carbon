// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/class/adapt.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/class/adapt.carbon

// --- adapt_class.carbon

library "[[@TEST_NAME]]";

class PairOfInts {
  var a: i32;
  var b: i32;

  fn Make() -> Self {
    return {.a = 1, .b = 2};
  }
}

class PairAdapter {
  adapt PairOfInts;

  fn Make() -> Self {
    return PairOfInts.Make() as Self;
  }

  fn GetB(self) -> i32 {
    let pi: PairOfInts = self as PairOfInts;
    return pi.b;
  }
}

fn Use() -> i32 {
  var pa: PairAdapter = PairAdapter.Make();
  return pa.GetB();
}

// --- adapt_int.carbon

library "[[@TEST_NAME]]";

class Int {
  adapt i32;

  impl as Core.Copy {
    fn Op(self) -> Self { return (self as i32).(Core.Copy.Op)() as Int; }
  }
}

fn DoStuff(a: Int) -> Int {
  return a;
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'adapt_class.carbon'
// CHECK:STDOUT: source_filename = "adapt_class.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @PairOfInts.val.loc9_28 = internal constant { i32, i32 } { i32 1, i32 2 }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMake.PairOfInts.Main(ptr sret({ i32, i32 }) %return) #0 !dbg !30 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_27.3.a = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 0, !dbg !33
// CHECK:STDOUT:   %.loc9_27.6.b = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 1, !dbg !33
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %return, ptr align 4 @PairOfInts.val.loc9_28, i64 8, i1 false), !dbg !34
// CHECK:STDOUT:   ret void, !dbg !34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMake.PairAdapter.Main(ptr sret({ i32, i32 }) %return) #0 !dbg !35 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CMake.PairOfInts.Main(ptr %return), !dbg !38
// CHECK:STDOUT:   ret void, !dbg !39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CGetB.PairAdapter.Main(ptr %self) #0 !dbg !40 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc22_14.1.b = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 1, !dbg !48
// CHECK:STDOUT:   %.loc22_14.2 = load i32, ptr %.loc22_14.1.b, align 4, !dbg !48
// CHECK:STDOUT:   ret i32 %.loc22_14.2, !dbg !49
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CUse.Main() #0 !dbg !51 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %pa.var = alloca { i32, i32 }, align 4, !dbg !54
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %pa.var), !dbg !54
// CHECK:STDOUT:   call void @_CMake.PairAdapter.Main(ptr %pa.var), !dbg !55
// CHECK:STDOUT:   %PairAdapter.GetB.call = call i32 @_CGetB.PairAdapter.Main(ptr %pa.var), !dbg !57
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.d4f16d438e91257f(ptr %pa.var), !dbg !54
// CHECK:STDOUT:   ret i32 %PairAdapter.GetB.call, !dbg !58
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.1e8be051c65833cb:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !59 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc27_3.1.b = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 1, !dbg !61
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.0aefc5da911f5051(ptr %.loc27_3.1.b), !dbg !61
// CHECK:STDOUT:   %.loc27_3.4.a = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 0, !dbg !61
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.0aefc5da911f5051(ptr %.loc27_3.4.a), !dbg !61
// CHECK:STDOUT:   ret void, !dbg !61
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.a031f5a2c623d311:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !65 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc27_3.1.b = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 1, !dbg !67
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.0aefc5da911f5051(ptr %.loc27_3.1.b), !dbg !67
// CHECK:STDOUT:   %.loc27_3.4.a = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 0, !dbg !67
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.0aefc5da911f5051(ptr %.loc27_3.4.a), !dbg !67
// CHECK:STDOUT:   ret void, !dbg !67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.d4f16d438e91257f(ptr %t) #0 !dbg !69 {
// CHECK:STDOUT:   ret void, !dbg !74
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.0aefc5da911f5051(ptr %t) #0 !dbg !76 {
// CHECK:STDOUT:   ret void, !dbg !81
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.0aefc5da911f5051, { 3, 2, 1, 0 }
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
// CHECK:STDOUT: !18 = !DIFile(filename: "adapt_class.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{!21}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !24 = !{!20, !21}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = !{!20}
// CHECK:STDOUT: !27 = !DISubroutineType(types: !26)
// CHECK:STDOUT: !28 = !{null, !21}
// CHECK:STDOUT: !29 = !DISubroutineType(types: !28)
// CHECK:STDOUT: !30 = distinct !DISubprogram(name: "Make", linkageName: "_CMake.PairOfInts.Main", scope: null, file: !18, line: 8, type: !23, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !33 = !DILocation(line: 9, column: 12, scope: !30)
// CHECK:STDOUT: !34 = !DILocation(line: 9, column: 5, scope: !30)
// CHECK:STDOUT: !35 = distinct !DISubprogram(name: "Make", linkageName: "_CMake.PairAdapter.Main", scope: null, file: !18, line: 16, type: !23, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !38 = !DILocation(line: 17, column: 12, scope: !35)
// CHECK:STDOUT: !39 = !DILocation(line: 17, column: 5, scope: !35)
// CHECK:STDOUT: !40 = distinct !DISubprogram(name: "GetB", linkageName: "_CGetB.PairAdapter.Main", scope: null, file: !18, line: 20, type: !25, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !50)
// CHECK:STDOUT: !41 = !DILocalVariable(arg: 1, scope: !40, type: !21)
// CHECK:STDOUT: !48 = !DILocation(line: 22, column: 12, scope: !40)
// CHECK:STDOUT: !49 = !DILocation(line: 22, column: 5, scope: !40)
// CHECK:STDOUT: !50 = !{!41}
// CHECK:STDOUT: !51 = distinct !DISubprogram(name: "Use", linkageName: "_CUse.Main", scope: null, file: !18, line: 26, type: !27, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !54 = !DILocation(line: 27, column: 3, scope: !51)
// CHECK:STDOUT: !55 = !DILocation(line: 27, column: 25, scope: !51)
// CHECK:STDOUT: !57 = !DILocation(line: 28, column: 10, scope: !51)
// CHECK:STDOUT: !58 = !DILocation(line: 28, column: 3, scope: !51)
// CHECK:STDOUT: !59 = distinct !DISubprogram(name: "Op", linkageName: "_COp.1e8be051c65833cb:core.SubobjectDestroy.Core", scope: null, file: !18, line: 27, type: !29, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !64)
// CHECK:STDOUT: !60 = !DILocalVariable(arg: 1, scope: !59, type: !21)
// CHECK:STDOUT: !61 = !DILocation(line: 27, column: 3, scope: !59)
// CHECK:STDOUT: !62 = !{null, !20}
// CHECK:STDOUT: !63 = !DISubroutineType(types: !62)
// CHECK:STDOUT: !64 = !{!60}
// CHECK:STDOUT: !65 = distinct !DISubprogram(name: "Op", linkageName: "_COp.a031f5a2c623d311:core.SubobjectDestroy.Core", scope: null, file: !18, line: 27, type: !29, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !68)
// CHECK:STDOUT: !66 = !DILocalVariable(arg: 1, scope: !65, type: !21)
// CHECK:STDOUT: !67 = !DILocation(line: 27, column: 3, scope: !65)
// CHECK:STDOUT: !68 = !{!66}
// CHECK:STDOUT: !69 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.d4f16d438e91257f", scope: null, file: !10, line: 26, type: !29, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !75)
// CHECK:STDOUT: !70 = !DILocalVariable(arg: 1, scope: !69, type: !21)
// CHECK:STDOUT: !74 = !DILocation(line: 26, column: 1, scope: !69)
// CHECK:STDOUT: !75 = !{!70}
// CHECK:STDOUT: !76 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.0aefc5da911f5051", scope: null, file: !10, line: 26, type: !63, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !82)
// CHECK:STDOUT: !77 = !DILocalVariable(arg: 1, scope: !76, type: !20)
// CHECK:STDOUT: !81 = !DILocation(line: 26, column: 1, scope: !76)
// CHECK:STDOUT: !82 = !{!77}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'adapt_int.carbon'
// CHECK:STDOUT: source_filename = "adapt_int.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @"_COp.Int.Main:Copy.Core"(i32 %self) #0 !dbg !87 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %self, !dbg !94
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CDoStuff.Main(i32 %a) #0 !dbg !96 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call = call i32 @"_COp.Int.Main:Copy.Core"(i32 %a), !dbg !101
// CHECK:STDOUT:   ret i32 %Int.as.Copy.impl.Op.call, !dbg !102
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!84}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !83 = !DIFile(filename: "adapt_int.carbon", directory: "")
// CHECK:STDOUT: !84 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !83, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !85 = !{!20, !20}
// CHECK:STDOUT: !86 = !DISubroutineType(types: !85)
// CHECK:STDOUT: !87 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Int.Main:Copy.Core", scope: null, file: !83, line: 8, type: !86, spFlags: DISPFlagDefinition, unit: !84, retainedNodes: !95)
// CHECK:STDOUT: !88 = !DILocalVariable(arg: 1, scope: !87, type: !20)
// CHECK:STDOUT: !94 = !DILocation(line: 8, column: 27, scope: !87)
// CHECK:STDOUT: !95 = !{!88}
// CHECK:STDOUT: !96 = distinct !DISubprogram(name: "DoStuff", linkageName: "_CDoStuff.Main", scope: null, file: !83, line: 12, type: !86, spFlags: DISPFlagDefinition, unit: !84, retainedNodes: !103)
// CHECK:STDOUT: !97 = !DILocalVariable(arg: 1, scope: !96, type: !20)
// CHECK:STDOUT: !101 = !DILocation(line: 13, column: 10, scope: !96)
// CHECK:STDOUT: !102 = !DILocation(line: 13, column: 3, scope: !96)
// CHECK:STDOUT: !103 = !{!97}
// CHECK:STDOUT:
