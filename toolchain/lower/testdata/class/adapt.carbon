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

  fn GetB[self: Self]() -> i32 {
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
    fn Op[self: Self]() -> Self { return (self as i32).(Core.Copy.Op)() as Int; }
  }
}

fn DoStuff(a: Int) -> Int {
  return a;
}

// CHECK:STDOUT: ; ModuleID = 'adapt_class.carbon'
// CHECK:STDOUT: source_filename = "adapt_class.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @PairOfInts.val.loc9_28 = internal constant { i32, i32 } { i32 1, i32 2 }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMake.PairOfInts.Main(ptr sret({ i32, i32 }) %return) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_27.3.a = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 0, !dbg !8
// CHECK:STDOUT:   %.loc9_27.6.b = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 1, !dbg !8
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %return, ptr align 4 @PairOfInts.val.loc9_28, i64 8, i1 false), !dbg !9
// CHECK:STDOUT:   ret void, !dbg !9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMake.PairAdapter.Main(ptr sret({ i32, i32 }) %return) #0 !dbg !10 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CMake.PairOfInts.Main(ptr %return), !dbg !11
// CHECK:STDOUT:   ret void, !dbg !12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CGetB.PairAdapter.Main(ptr %self) #0 !dbg !13 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc22_14.1.b = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 1, !dbg !19
// CHECK:STDOUT:   %.loc22_14.2 = load i32, ptr %.loc22_14.1.b, align 4, !dbg !19
// CHECK:STDOUT:   ret i32 %.loc22_14.2, !dbg !20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CUse.Main() #0 !dbg !21 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %pa.var = alloca { i32, i32 }, align 8, !dbg !24
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %pa.var), !dbg !24
// CHECK:STDOUT:   call void @_CMake.PairAdapter.Main(ptr %pa.var), !dbg !25
// CHECK:STDOUT:   %PairAdapter.GetB.call = call i32 @_CGetB.PairAdapter.Main(ptr %pa.var), !dbg !26
// CHECK:STDOUT:   ret i32 %PairAdapter.GetB.call, !dbg !27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "adapt_class.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Make", linkageName: "_CMake.PairOfInts.Main", scope: null, file: !3, line: 8, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7}
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 8)
// CHECK:STDOUT: !8 = !DILocation(line: 9, column: 12, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 9, column: 5, scope: !4)
// CHECK:STDOUT: !10 = distinct !DISubprogram(name: "Make", linkageName: "_CMake.PairAdapter.Main", scope: null, file: !3, line: 16, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !11 = !DILocation(line: 17, column: 12, scope: !10)
// CHECK:STDOUT: !12 = !DILocation(line: 17, column: 5, scope: !10)
// CHECK:STDOUT: !13 = distinct !DISubprogram(name: "GetB", linkageName: "_CGetB.PairAdapter.Main", scope: null, file: !3, line: 20, type: !14, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !17)
// CHECK:STDOUT: !14 = !DISubroutineType(types: !15)
// CHECK:STDOUT: !15 = !{!16, !7}
// CHECK:STDOUT: !16 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !17 = !{!18}
// CHECK:STDOUT: !18 = !DILocalVariable(arg: 1, scope: !13, type: !7)
// CHECK:STDOUT: !19 = !DILocation(line: 22, column: 12, scope: !13)
// CHECK:STDOUT: !20 = !DILocation(line: 22, column: 5, scope: !13)
// CHECK:STDOUT: !21 = distinct !DISubprogram(name: "Use", linkageName: "_CUse.Main", scope: null, file: !3, line: 26, type: !22, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !22 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !23 = !{!16}
// CHECK:STDOUT: !24 = !DILocation(line: 27, column: 3, scope: !21)
// CHECK:STDOUT: !25 = !DILocation(line: 27, column: 25, scope: !21)
// CHECK:STDOUT: !26 = !DILocation(line: 28, column: 10, scope: !21)
// CHECK:STDOUT: !27 = !DILocation(line: 28, column: 3, scope: !21)
// CHECK:STDOUT: ; ModuleID = 'adapt_int.carbon'
// CHECK:STDOUT: source_filename = "adapt_int.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @"_COp.Int.Main:Copy.Core"(i32 %self) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %self, !dbg !10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CDoStuff.Main(i32 %a) #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call = call i32 @"_COp.Int.Main:Copy.Core"(i32 %a), !dbg !14
// CHECK:STDOUT:   ret i32 %Int.as.Copy.impl.Op.call, !dbg !15
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
// CHECK:STDOUT: !3 = !DIFile(filename: "adapt_int.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Int.Main:Copy.Core", scope: null, file: !3, line: 8, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !8)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7, !7}
// CHECK:STDOUT: !7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !8 = !{!9}
// CHECK:STDOUT: !9 = !DILocalVariable(arg: 1, scope: !4, type: !7)
// CHECK:STDOUT: !10 = !DILocation(line: 8, column: 35, scope: !4)
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "DoStuff", linkageName: "_CDoStuff.Main", scope: null, file: !3, line: 12, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !12)
// CHECK:STDOUT: !12 = !{!13}
// CHECK:STDOUT: !13 = !DILocalVariable(arg: 1, scope: !11, type: !7)
// CHECK:STDOUT: !14 = !DILocation(line: 13, column: 10, scope: !11)
// CHECK:STDOUT: !15 = !DILocation(line: 13, column: 3, scope: !11)
