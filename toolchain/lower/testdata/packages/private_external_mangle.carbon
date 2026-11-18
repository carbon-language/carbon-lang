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
// CHECK:STDOUT: define void @_CCall.Main.c14f6fa2d8939391() #0 !dbg !22 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CA.Main(), !dbg !23
// CHECK:STDOUT:   ret void, !dbg !24
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCall.C.Main.df01d416cc5a0873() #0 !dbg !25 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CA.Main(), !dbg !26
// CHECK:STDOUT:   ret void, !dbg !27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !18 = !DIFile(filename: "private_call_a.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !20 = !{null}
// CHECK:STDOUT: !21 = !DISubroutineType(types: !20)
// CHECK:STDOUT: !22 = distinct !DISubprogram(name: "Call", linkageName: "_CCall.Main.c14f6fa2d8939391", scope: null, file: !18, line: 6, type: !21, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !23 = !DILocation(line: 7, column: 3, scope: !22)
// CHECK:STDOUT: !24 = !DILocation(line: 6, column: 1, scope: !22)
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "Call", linkageName: "_CCall.C.Main.df01d416cc5a0873", scope: null, file: !18, line: 11, type: !21, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !26 = !DILocation(line: 12, column: 5, scope: !25)
// CHECK:STDOUT: !27 = !DILocation(line: 11, column: 3, scope: !25)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'private_call_b.carbon'
// CHECK:STDOUT: source_filename = "private_call_b.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CB.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCall.Main.d20b101f33acb508() #0 !dbg !30 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CB.Main(), !dbg !31
// CHECK:STDOUT:   ret void, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCall.C.Main.df01d416cc5a0873() #0 !dbg !33 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CB.Main(), !dbg !34
// CHECK:STDOUT:   ret void, !dbg !35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!29}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !{null}
// CHECK:STDOUT: !21 = !DISubroutineType(types: !20)
// CHECK:STDOUT: !28 = !DIFile(filename: "private_call_b.carbon", directory: "")
// CHECK:STDOUT: !29 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !28, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !30 = distinct !DISubprogram(name: "Call", linkageName: "_CCall.Main.d20b101f33acb508", scope: null, file: !28, line: 6, type: !21, spFlags: DISPFlagDefinition, unit: !29)
// CHECK:STDOUT: !31 = !DILocation(line: 7, column: 3, scope: !30)
// CHECK:STDOUT: !32 = !DILocation(line: 6, column: 1, scope: !30)
// CHECK:STDOUT: !33 = distinct !DISubprogram(name: "Call", linkageName: "_CCall.C.Main.df01d416cc5a0873", scope: null, file: !28, line: 11, type: !21, spFlags: DISPFlagDefinition, unit: !29)
// CHECK:STDOUT: !34 = !DILocation(line: 12, column: 5, scope: !33)
// CHECK:STDOUT: !35 = !DILocation(line: 11, column: 3, scope: !33)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'use_a_and_b.carbon'
// CHECK:STDOUT: source_filename = "use_a_and_b.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !38 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CCallInA.Main.6944efcfbb6a8360(), !dbg !39
// CHECK:STDOUT:   call void @_CCallInB.Main.6944efcfbb6a8360(), !dbg !40
// CHECK:STDOUT:   ret void, !dbg !41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCallInA.Main.6944efcfbb6a8360() #0 !dbg !42 {
// CHECK:STDOUT:   call void @_CCall.Main.c14f6fa2d8939391(), !dbg !43
// CHECK:STDOUT:   call void @_CCall.C.Main.df01d416cc5a0873(), !dbg !44
// CHECK:STDOUT:   ret void, !dbg !45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCallInB.Main.6944efcfbb6a8360() #0 !dbg !46 {
// CHECK:STDOUT:   call void @_CCall.Main.d20b101f33acb508(), !dbg !47
// CHECK:STDOUT:   call void @_CCall.C.Main.df01d416cc5a0873(), !dbg !48
// CHECK:STDOUT:   ret void, !dbg !49
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!37}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !18 = !DIFile(filename: "private_call_a.carbon", directory: "")
// CHECK:STDOUT: !20 = !{null}
// CHECK:STDOUT: !21 = !DISubroutineType(types: !20)
// CHECK:STDOUT: !28 = !DIFile(filename: "private_call_b.carbon", directory: "")
// CHECK:STDOUT: !36 = !DIFile(filename: "use_a_and_b.carbon", directory: "")
// CHECK:STDOUT: !37 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !36, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !38 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !36, line: 7, type: !21, spFlags: DISPFlagDefinition, unit: !37)
// CHECK:STDOUT: !39 = !DILocation(line: 9, column: 3, scope: !38)
// CHECK:STDOUT: !40 = !DILocation(line: 10, column: 3, scope: !38)
// CHECK:STDOUT: !41 = !DILocation(line: 7, column: 1, scope: !38)
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "CallInA", linkageName: "_CCallInA.Main.6944efcfbb6a8360", scope: null, file: !18, line: 16, type: !21, spFlags: DISPFlagDefinition, unit: !37)
// CHECK:STDOUT: !43 = !DILocation(line: 17, column: 3, scope: !42)
// CHECK:STDOUT: !44 = !DILocation(line: 18, column: 3, scope: !42)
// CHECK:STDOUT: !45 = !DILocation(line: 16, column: 1, scope: !42)
// CHECK:STDOUT: !46 = distinct !DISubprogram(name: "CallInB", linkageName: "_CCallInB.Main.6944efcfbb6a8360", scope: null, file: !28, line: 16, type: !21, spFlags: DISPFlagDefinition, unit: !37)
// CHECK:STDOUT: !47 = !DILocation(line: 17, column: 3, scope: !46)
// CHECK:STDOUT: !48 = !DILocation(line: 18, column: 3, scope: !46)
// CHECK:STDOUT: !49 = !DILocation(line: 16, column: 1, scope: !46)
// CHECK:STDOUT:
