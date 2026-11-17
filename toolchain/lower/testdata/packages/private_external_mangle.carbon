// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/packages/private_external_mangle.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/packages/private_external_mangle.carbon

// --- private_call_a.carbon

library "[[@TEST_NAME]]";

fn A();

private fn Call() {
  A();
}

private class C {
  fn Call() {
    A();
  }
}

fn CallInA(unused generic T: type) {
  Call();
  C.Call();
}

// --- private_call_b.carbon

library "[[@TEST_NAME]]";

fn B();

private fn Call() {
  B();
}

private class C {
  fn Call() {
    B();
  }
}

fn CallInB(unused generic T: type) {
  Call();
  C.Call();
}

// --- use_a_and_b.carbon

library "[[@TEST_NAME]]";

import library "private_call_a";
import library "private_call_b";

fn F() {
  // These should call different `Call` functions.
  CallInA({});
  CallInB({});
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'private_call_a.carbon'
// CHECK:STDOUT: source_filename = "private_call_a.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CA.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCall.Main.c14f6fa2d8939391() #0 !dbg !24 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CA.Main(), !dbg !25
// CHECK:STDOUT:   ret void, !dbg !26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCall.C.Main.df01d416cc5a0873() #0 !dbg !27 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CA.Main(), !dbg !28
// CHECK:STDOUT:   ret void, !dbg !29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!21}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIFile(filename: "private_call_a.carbon", directory: "")
// CHECK:STDOUT: !21 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !20, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !22 = !{null}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !24 = distinct !DISubprogram(name: "Call", linkageName: "_CCall.Main.c14f6fa2d8939391", scope: null, file: !20, line: 6, type: !23, spFlags: DISPFlagDefinition, unit: !21)
// CHECK:STDOUT: !25 = !DILocation(line: 7, column: 3, scope: !24)
// CHECK:STDOUT: !26 = !DILocation(line: 6, column: 1, scope: !24)
// CHECK:STDOUT: !27 = distinct !DISubprogram(name: "Call", linkageName: "_CCall.C.Main.df01d416cc5a0873", scope: null, file: !20, line: 11, type: !23, spFlags: DISPFlagDefinition, unit: !21)
// CHECK:STDOUT: !28 = !DILocation(line: 12, column: 5, scope: !27)
// CHECK:STDOUT: !29 = !DILocation(line: 11, column: 3, scope: !27)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'private_call_b.carbon'
// CHECK:STDOUT: source_filename = "private_call_b.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CB.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCall.Main.d20b101f33acb508() #0 !dbg !32 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CB.Main(), !dbg !33
// CHECK:STDOUT:   ret void, !dbg !34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCall.C.Main.df01d416cc5a0873() #0 !dbg !35 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CB.Main(), !dbg !36
// CHECK:STDOUT:   ret void, !dbg !37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!31}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !22 = !{null}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !30 = !DIFile(filename: "private_call_b.carbon", directory: "")
// CHECK:STDOUT: !31 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !30, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !32 = distinct !DISubprogram(name: "Call", linkageName: "_CCall.Main.d20b101f33acb508", scope: null, file: !30, line: 6, type: !23, spFlags: DISPFlagDefinition, unit: !31)
// CHECK:STDOUT: !33 = !DILocation(line: 7, column: 3, scope: !32)
// CHECK:STDOUT: !34 = !DILocation(line: 6, column: 1, scope: !32)
// CHECK:STDOUT: !35 = distinct !DISubprogram(name: "Call", linkageName: "_CCall.C.Main.df01d416cc5a0873", scope: null, file: !30, line: 11, type: !23, spFlags: DISPFlagDefinition, unit: !31)
// CHECK:STDOUT: !36 = !DILocation(line: 12, column: 5, scope: !35)
// CHECK:STDOUT: !37 = !DILocation(line: 11, column: 3, scope: !35)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'use_a_and_b.carbon'
// CHECK:STDOUT: source_filename = "use_a_and_b.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !40 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CCallInA.Main.6944efcfbb6a8360(), !dbg !41
// CHECK:STDOUT:   call void @_CCallInB.Main.6944efcfbb6a8360(), !dbg !42
// CHECK:STDOUT:   ret void, !dbg !43
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCallInA.Main.6944efcfbb6a8360() #0 !dbg !44 {
// CHECK:STDOUT:   call void @_CCall.Main.c14f6fa2d8939391(), !dbg !45
// CHECK:STDOUT:   call void @_CCall.C.Main.df01d416cc5a0873(), !dbg !46
// CHECK:STDOUT:   ret void, !dbg !47
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCallInB.Main.6944efcfbb6a8360() #0 !dbg !48 {
// CHECK:STDOUT:   call void @_CCall.Main.d20b101f33acb508(), !dbg !49
// CHECK:STDOUT:   call void @_CCall.C.Main.df01d416cc5a0873(), !dbg !50
// CHECK:STDOUT:   ret void, !dbg !51
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CA.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CCall.Main.c14f6fa2d8939391()
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CCall.C.Main.df01d416cc5a0873()
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CB.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CCall.Main.d20b101f33acb508()
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CCall.C.Main.df01d416cc5a0873, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!39}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIFile(filename: "private_call_a.carbon", directory: "")
// CHECK:STDOUT: !22 = !{null}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !30 = !DIFile(filename: "private_call_b.carbon", directory: "")
// CHECK:STDOUT: !38 = !DIFile(filename: "use_a_and_b.carbon", directory: "")
// CHECK:STDOUT: !39 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !38, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !40 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !38, line: 7, type: !23, spFlags: DISPFlagDefinition, unit: !39)
// CHECK:STDOUT: !41 = !DILocation(line: 9, column: 3, scope: !40)
// CHECK:STDOUT: !42 = !DILocation(line: 10, column: 3, scope: !40)
// CHECK:STDOUT: !43 = !DILocation(line: 7, column: 1, scope: !40)
// CHECK:STDOUT: !44 = distinct !DISubprogram(name: "CallInA", linkageName: "_CCallInA.Main.6944efcfbb6a8360", scope: null, file: !20, line: 16, type: !23, spFlags: DISPFlagDefinition, unit: !39)
// CHECK:STDOUT: !45 = !DILocation(line: 17, column: 3, scope: !44)
// CHECK:STDOUT: !46 = !DILocation(line: 18, column: 3, scope: !44)
// CHECK:STDOUT: !47 = !DILocation(line: 16, column: 1, scope: !44)
// CHECK:STDOUT: !48 = distinct !DISubprogram(name: "CallInB", linkageName: "_CCallInB.Main.6944efcfbb6a8360", scope: null, file: !30, line: 16, type: !23, spFlags: DISPFlagDefinition, unit: !39)
// CHECK:STDOUT: !49 = !DILocation(line: 17, column: 3, scope: !48)
// CHECK:STDOUT: !50 = !DILocation(line: 18, column: 3, scope: !48)
// CHECK:STDOUT: !51 = !DILocation(line: 16, column: 1, scope: !48)
// CHECK:STDOUT:
