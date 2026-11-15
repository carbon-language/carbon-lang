// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/impl/assoc_fn_alias.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/impl/assoc_fn_alias.carbon

// --- class_method_defined_as_alias.carbon

library "[[@TEST_NAME]]";

interface I {
  fn F(self) -> i32;
}

class A {
  alias F = I.F;
  var n: i32;
}

impl A as I {
  fn F(self: A) -> i32 {
    return self.n;
  }
}

fn Call(a: A) -> i32 {
  return a.F();
}

// --- impl_method_defined_as_alias.carbon

library "[[@TEST_NAME]]";

interface I {
  fn F(self) -> i32;
}

class C {
  var i: i32;
  fn F(self) -> i32 { return self.i; }
}

impl C as I {
  alias F = C.F;
}

fn Call[I: I](i: I) -> i32 {
  return i.F();
}

fn Call2(c: C) {
  Call(c);
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'class_method_defined_as_alias.carbon'
// CHECK:STDOUT: source_filename = "class_method_defined_as_alias.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @"_CF.A.Main:I.Main"(ptr %self) #0 !dbg !26 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc15_16.1.n = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !31
// CHECK:STDOUT:   %.loc15_16.2 = load i32, ptr %.loc15_16.1.n, align 4, !dbg !31
// CHECK:STDOUT:   ret i32 %.loc15_16.2, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CCall.Main(ptr %a) #0 !dbg !34 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %A.as.I.impl.F.call = call i32 @"_CF.A.Main:I.Main"(ptr %a), !dbg !39
// CHECK:STDOUT:   ret i32 %A.as.I.impl.F.call, !dbg !40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!21}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIFile(filename: "class_method_defined_as_alias.carbon", directory: "")
// CHECK:STDOUT: !21 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !20, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !22 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{!22, !23}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "F", linkageName: "_CF.A.Main:I.Main", scope: null, file: !20, line: 14, type: !25, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !33)
// CHECK:STDOUT: !27 = !DILocalVariable(arg: 1, scope: !26, type: !23)
// CHECK:STDOUT: !31 = !DILocation(line: 15, column: 12, scope: !26)
// CHECK:STDOUT: !32 = !DILocation(line: 15, column: 5, scope: !26)
// CHECK:STDOUT: !33 = !{!27}
// CHECK:STDOUT: !34 = distinct !DISubprogram(name: "Call", linkageName: "_CCall.Main", scope: null, file: !20, line: 19, type: !25, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !41)
// CHECK:STDOUT: !35 = !DILocalVariable(arg: 1, scope: !34, type: !23)
// CHECK:STDOUT: !39 = !DILocation(line: 20, column: 10, scope: !34)
// CHECK:STDOUT: !40 = !DILocation(line: 20, column: 3, scope: !34)
// CHECK:STDOUT: !41 = !{!35}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'impl_method_defined_as_alias.carbon'
// CHECK:STDOUT: source_filename = "impl_method_defined_as_alias.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CF.C.Main(ptr %self) #0 !dbg !46 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc10_34.1.i = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !51
// CHECK:STDOUT:   %.loc10_34.2 = load i32, ptr %.loc10_34.1.i, align 4, !dbg !51
// CHECK:STDOUT:   ret i32 %.loc10_34.2, !dbg !52
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCall2.Main(ptr %c) #0 !dbg !54 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Call.call = call i32 @_CCall.Main.0676fa98bb7dbd66(ptr %c), !dbg !58
// CHECK:STDOUT:   ret void, !dbg !59
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CCall.Main.0676fa98bb7dbd66(ptr %i) #0 !dbg !61 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %I.WithSelf.F.call = call i32 @_CF.C.Main(ptr %i), !dbg !66
// CHECK:STDOUT:   ret i32 %I.WithSelf.F.call, !dbg !67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!43}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !22 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{!22, !23}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !42 = !DIFile(filename: "impl_method_defined_as_alias.carbon", directory: "")
// CHECK:STDOUT: !43 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !42, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !44 = !{null, !23}
// CHECK:STDOUT: !45 = !DISubroutineType(types: !44)
// CHECK:STDOUT: !46 = distinct !DISubprogram(name: "F", linkageName: "_CF.C.Main", scope: null, file: !42, line: 10, type: !25, spFlags: DISPFlagDefinition, unit: !43, retainedNodes: !53)
// CHECK:STDOUT: !47 = !DILocalVariable(arg: 1, scope: !46, type: !23)
// CHECK:STDOUT: !51 = !DILocation(line: 10, column: 30, scope: !46)
// CHECK:STDOUT: !52 = !DILocation(line: 10, column: 23, scope: !46)
// CHECK:STDOUT: !53 = !{!47}
// CHECK:STDOUT: !54 = distinct !DISubprogram(name: "Call2", linkageName: "_CCall2.Main", scope: null, file: !42, line: 21, type: !45, spFlags: DISPFlagDefinition, unit: !43, retainedNodes: !60)
// CHECK:STDOUT: !55 = !DILocalVariable(arg: 1, scope: !54, type: !23)
// CHECK:STDOUT: !58 = !DILocation(line: 22, column: 3, scope: !54)
// CHECK:STDOUT: !59 = !DILocation(line: 21, column: 1, scope: !54)
// CHECK:STDOUT: !60 = !{!55}
// CHECK:STDOUT: !61 = distinct !DISubprogram(name: "Call", linkageName: "_CCall.Main.0676fa98bb7dbd66", scope: null, file: !42, line: 17, type: !25, spFlags: DISPFlagDefinition, unit: !43, retainedNodes: !68)
// CHECK:STDOUT: !62 = !DILocalVariable(arg: 1, scope: !61, type: !23)
// CHECK:STDOUT: !66 = !DILocation(line: 18, column: 10, scope: !61)
// CHECK:STDOUT: !67 = !DILocation(line: 18, column: 3, scope: !61)
// CHECK:STDOUT: !68 = !{!62}
// CHECK:STDOUT:
