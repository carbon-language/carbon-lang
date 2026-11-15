// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/let/local.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/let/local.carbon

// --- local.carbon
library "[[@TEST_NAME]]";

fn Run() -> i32 {
  let n: i32 = 1;
  let m: i32 = n;
  return m;
}

// --- symbolic.carbon
library "[[@TEST_NAME]]";

class C {}

fn F(_: C) {}

fn CallF() {
  // TODO: Uncomment this once symbolic locals work in check. It's disabled so
  // that we still test lowering of non-symbolic locals.
  // let generic c: C = {};
  // F(c);
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'local.carbon'
// CHECK:STDOUT: source_filename = "local.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @main() #0 !dbg !25 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 1, !dbg !34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!21}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIFile(filename: "local.carbon", directory: "")
// CHECK:STDOUT: !21 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !20, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !22 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !23 = !{!22}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !20, line: 3, type: !24, spFlags: DISPFlagDefinition, unit: !21)
// CHECK:STDOUT: !34 = !DILocation(line: 6, column: 3, scope: !25)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'symbolic.carbon'
// CHECK:STDOUT: source_filename = "symbolic.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main(ptr %_) #0 !dbg !42 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCallF.Main() #0 !dbg !47 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!36}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !35 = !DIFile(filename: "symbolic.carbon", directory: "")
// CHECK:STDOUT: !36 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !35, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !37 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !38 = !{null, !37}
// CHECK:STDOUT: !39 = !DISubroutineType(types: !38)
// CHECK:STDOUT: !40 = !{null}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !35, line: 5, type: !39, spFlags: DISPFlagDefinition, unit: !36, retainedNodes: !46)
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 1, scope: !42, type: !37)
// CHECK:STDOUT: !45 = !DILocation(line: 5, column: 1, scope: !42)
// CHECK:STDOUT: !46 = !{!43}
// CHECK:STDOUT: !47 = distinct !DISubprogram(name: "CallF", linkageName: "_CCallF.Main", scope: null, file: !35, line: 7, type: !41, spFlags: DISPFlagDefinition, unit: !36)
// CHECK:STDOUT: !48 = !DILocation(line: 7, column: 1, scope: !47)
// CHECK:STDOUT:
