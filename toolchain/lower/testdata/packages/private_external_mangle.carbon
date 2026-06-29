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

fn CallInA(unused T:! type) {
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

fn CallInB(unused T:! type) {
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

// CHECK:STDOUT: ; ModuleID = 'private_call_a.carbon'
// CHECK:STDOUT: source_filename = "private_call_a.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CA.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCall.Main.c14f6fa2d8939391() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CA.Main(), !dbg !7
// CHECK:STDOUT:   ret void, !dbg !8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCall.C.Main.df01d416cc5a0873() #0 !dbg !9 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CA.Main(), !dbg !10
// CHECK:STDOUT:   ret void, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "private_call_a.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Call", linkageName: "_CCall.Main.c14f6fa2d8939391", scope: null, file: !1, line: 6, type: !5, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DILocation(line: 7, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 6, column: 1, scope: !4)
// CHECK:STDOUT: !9 = distinct !DISubprogram(name: "Call", linkageName: "_CCall.C.Main.df01d416cc5a0873", scope: null, file: !1, line: 11, type: !5, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !10 = !DILocation(line: 12, column: 5, scope: !9)
// CHECK:STDOUT: !11 = !DILocation(line: 11, column: 3, scope: !9)
// CHECK:STDOUT: ; ModuleID = 'private_call_b.carbon'
// CHECK:STDOUT: source_filename = "private_call_b.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CB.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCall.Main.d20b101f33acb508() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CB.Main(), !dbg !7
// CHECK:STDOUT:   ret void, !dbg !8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCall.C.Main.df01d416cc5a0873() #0 !dbg !9 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CB.Main(), !dbg !10
// CHECK:STDOUT:   ret void, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "private_call_b.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Call", linkageName: "_CCall.Main.d20b101f33acb508", scope: null, file: !1, line: 6, type: !5, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DILocation(line: 7, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 6, column: 1, scope: !4)
// CHECK:STDOUT: !9 = distinct !DISubprogram(name: "Call", linkageName: "_CCall.C.Main.df01d416cc5a0873", scope: null, file: !1, line: 11, type: !5, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !10 = !DILocation(line: 12, column: 5, scope: !9)
// CHECK:STDOUT: !11 = !DILocation(line: 11, column: 3, scope: !9)
// CHECK:STDOUT: ; ModuleID = 'use_a_and_b.carbon'
// CHECK:STDOUT: source_filename = "use_a_and_b.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CCallInA.Main.6944efcfbb6a8360(), !dbg !7
// CHECK:STDOUT:   call void @_CCallInB.Main.6944efcfbb6a8360(), !dbg !8
// CHECK:STDOUT:   ret void, !dbg !9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCallInA.Main.6944efcfbb6a8360() #0 !dbg !10 {
// CHECK:STDOUT:   call void @_CCall.Main.c14f6fa2d8939391(), !dbg !12
// CHECK:STDOUT:   call void @_CCall.C.Main.df01d416cc5a0873(), !dbg !13
// CHECK:STDOUT:   ret void, !dbg !14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCallInB.Main.6944efcfbb6a8360() #0 !dbg !15 {
// CHECK:STDOUT:   call void @_CCall.Main.d20b101f33acb508(), !dbg !17
// CHECK:STDOUT:   call void @_CCall.C.Main.df01d416cc5a0873(), !dbg !18
// CHECK:STDOUT:   ret void, !dbg !19
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "use_a_and_b.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !1, line: 7, type: !5, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DILocation(line: 9, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 10, column: 3, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 7, column: 1, scope: !4)
// CHECK:STDOUT: !10 = distinct !DISubprogram(name: "CallInA", linkageName: "_CCallInA.Main.6944efcfbb6a8360", scope: null, file: !11, line: 16, type: !5, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !11 = !DIFile(filename: "private_call_a.carbon", directory: "")
// CHECK:STDOUT: !12 = !DILocation(line: 17, column: 3, scope: !10)
// CHECK:STDOUT: !13 = !DILocation(line: 18, column: 3, scope: !10)
// CHECK:STDOUT: !14 = !DILocation(line: 16, column: 1, scope: !10)
// CHECK:STDOUT: !15 = distinct !DISubprogram(name: "CallInB", linkageName: "_CCallInB.Main.6944efcfbb6a8360", scope: null, file: !16, line: 16, type: !5, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !16 = !DIFile(filename: "private_call_b.carbon", directory: "")
// CHECK:STDOUT: !17 = !DILocation(line: 17, column: 3, scope: !15)
// CHECK:STDOUT: !18 = !DILocation(line: 18, column: 3, scope: !15)
// CHECK:STDOUT: !19 = !DILocation(line: 16, column: 1, scope: !15)
