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
// CHECK:STDOUT: define void @_CMake.PairOfInts.Main(ptr sret({ i32, i32 }) %return) #0 !dbg !28 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_27.3.a = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 0, !dbg !31
// CHECK:STDOUT:   %.loc9_27.6.b = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 1, !dbg !31
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %return, ptr align 4 @PairOfInts.val.loc9_28, i64 8, i1 false), !dbg !32
// CHECK:STDOUT:   ret void, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMake.PairAdapter.Main(ptr sret({ i32, i32 }) %return) #0 !dbg !33 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CMake.PairOfInts.Main(ptr %return), !dbg !36
// CHECK:STDOUT:   ret void, !dbg !37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CGetB.PairAdapter.Main(ptr %self) #0 !dbg !38 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc22_14.1.b = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 1, !dbg !46
// CHECK:STDOUT:   %.loc22_14.2 = load i32, ptr %.loc22_14.1.b, align 4, !dbg !46
// CHECK:STDOUT:   ret i32 %.loc22_14.2, !dbg !47
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CUse.Main() #0 !dbg !49 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %pa.var = alloca { i32, i32 }, align 4, !dbg !52
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %pa.var), !dbg !52
// CHECK:STDOUT:   call void @_CMake.PairAdapter.Main(ptr %pa.var), !dbg !53
// CHECK:STDOUT:   %PairAdapter.GetB.call = call i32 @_CGetB.PairAdapter.Main(ptr %pa.var), !dbg !55
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.fb94fb1298d84111(ptr %pa.var), !dbg !52
// CHECK:STDOUT:   ret i32 %PairAdapter.GetB.call, !dbg !58
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.fb94fb1298d84111(ptr %t) #0 !dbg !59 {
// CHECK:STDOUT:   ret void, !dbg !64
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
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "Make", linkageName: "_CMake.PairOfInts.Main", scope: null, file: !18, line: 8, type: !23, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !31 = !DILocation(line: 9, column: 12, scope: !28)
// CHECK:STDOUT: !32 = !DILocation(line: 9, column: 5, scope: !28)
// CHECK:STDOUT: !33 = distinct !DISubprogram(name: "Make", linkageName: "_CMake.PairAdapter.Main", scope: null, file: !18, line: 16, type: !23, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !36 = !DILocation(line: 17, column: 12, scope: !33)
// CHECK:STDOUT: !37 = !DILocation(line: 17, column: 5, scope: !33)
// CHECK:STDOUT: !38 = distinct !DISubprogram(name: "GetB", linkageName: "_CGetB.PairAdapter.Main", scope: null, file: !18, line: 20, type: !25, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !48)
// CHECK:STDOUT: !39 = !DILocalVariable(arg: 1, scope: !38, type: !21)
// CHECK:STDOUT: !46 = !DILocation(line: 22, column: 12, scope: !38)
// CHECK:STDOUT: !47 = !DILocation(line: 22, column: 5, scope: !38)
// CHECK:STDOUT: !48 = !{!39}
// CHECK:STDOUT: !49 = distinct !DISubprogram(name: "Use", linkageName: "_CUse.Main", scope: null, file: !18, line: 26, type: !27, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !52 = !DILocation(line: 27, column: 3, scope: !49)
// CHECK:STDOUT: !53 = !DILocation(line: 27, column: 25, scope: !49)
// CHECK:STDOUT: !55 = !DILocation(line: 28, column: 10, scope: !49)
// CHECK:STDOUT: !56 = !{null, !21}
// CHECK:STDOUT: !57 = !DISubroutineType(types: !56)
// CHECK:STDOUT: !58 = !DILocation(line: 28, column: 3, scope: !49)
// CHECK:STDOUT: !59 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.fb94fb1298d84111", scope: null, file: !10, line: 32, type: !57, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !65)
// CHECK:STDOUT: !60 = !DILocalVariable(arg: 1, scope: !59, type: !21)
// CHECK:STDOUT: !64 = !DILocation(line: 32, column: 1, scope: !59)
// CHECK:STDOUT: !65 = !{!60}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'adapt_int.carbon'
// CHECK:STDOUT: source_filename = "adapt_int.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @"_COp.Int.Main:Copy.Core"(i32 %self) #0 !dbg !70 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %self, !dbg !77
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CDoStuff.Main(i32 %a) #0 !dbg !79 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call = call i32 @"_COp.Int.Main:Copy.Core"(i32 %a), !dbg !84
// CHECK:STDOUT:   ret i32 %Int.as.Copy.impl.Op.call, !dbg !85
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!67}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !66 = !DIFile(filename: "adapt_int.carbon", directory: "")
// CHECK:STDOUT: !67 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !66, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !68 = !{!20, !20}
// CHECK:STDOUT: !69 = !DISubroutineType(types: !68)
// CHECK:STDOUT: !70 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Int.Main:Copy.Core", scope: null, file: !66, line: 8, type: !69, spFlags: DISPFlagDefinition, unit: !67, retainedNodes: !78)
// CHECK:STDOUT: !71 = !DILocalVariable(arg: 1, scope: !70, type: !20)
// CHECK:STDOUT: !77 = !DILocation(line: 8, column: 27, scope: !70)
// CHECK:STDOUT: !78 = !{!71}
// CHECK:STDOUT: !79 = distinct !DISubprogram(name: "DoStuff", linkageName: "_CDoStuff.Main", scope: null, file: !66, line: 12, type: !69, spFlags: DISPFlagDefinition, unit: !67, retainedNodes: !86)
// CHECK:STDOUT: !80 = !DILocalVariable(arg: 1, scope: !79, type: !20)
// CHECK:STDOUT: !84 = !DILocation(line: 13, column: 10, scope: !79)
// CHECK:STDOUT: !85 = !DILocation(line: 13, column: 3, scope: !79)
// CHECK:STDOUT: !86 = !{!80}
// CHECK:STDOUT:
