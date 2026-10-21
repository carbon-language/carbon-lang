// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/impl/extend_impl.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/impl/extend_impl.carbon

interface I {
  fn F() -> i32;
}

class A {
  extend impl as I {
    fn F() -> i32 {
      return 0;
    }
  }
}

fn TypeAccess() -> i32 {
  return A.F();
}

fn InstanceAccess(a: A) -> i32 {
  return a.F();
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'extend_impl.carbon'
// CHECK:STDOUT: source_filename = "extend_impl.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @"_CF.A.Main:I.Main"() #0 !dbg !26 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 0, !dbg !29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CTypeAccess.Main() #0 !dbg !30 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %A.as.I.impl.F.call = call i32 @"_CF.A.Main:I.Main"(), !dbg !33
// CHECK:STDOUT:   ret i32 %A.as.I.impl.F.call, !dbg !34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CInstanceAccess.Main(ptr %a) #0 !dbg !35 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %A.as.I.impl.F.call = call i32 @"_CF.A.Main:I.Main"(), !dbg !40
// CHECK:STDOUT:   ret i32 %A.as.I.impl.F.call, !dbg !41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !18 = !DIFile(filename: "extend_impl.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !{!20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{!20, !23}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "F", linkageName: "_CF.A.Main:I.Main", scope: null, file: !18, line: 19, type: !22, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !29 = !DILocation(line: 20, column: 7, scope: !26)
// CHECK:STDOUT: !30 = distinct !DISubprogram(name: "TypeAccess", linkageName: "_CTypeAccess.Main", scope: null, file: !18, line: 25, type: !22, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !33 = !DILocation(line: 26, column: 10, scope: !30)
// CHECK:STDOUT: !34 = !DILocation(line: 26, column: 3, scope: !30)
// CHECK:STDOUT: !35 = distinct !DISubprogram(name: "InstanceAccess", linkageName: "_CInstanceAccess.Main", scope: null, file: !18, line: 29, type: !25, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !42)
// CHECK:STDOUT: !36 = !DILocalVariable(arg: 1, scope: !35, type: !23)
// CHECK:STDOUT: !40 = !DILocation(line: 30, column: 10, scope: !35)
// CHECK:STDOUT: !41 = !DILocation(line: 30, column: 3, scope: !35)
// CHECK:STDOUT: !42 = !{!36}
// CHECK:STDOUT:
