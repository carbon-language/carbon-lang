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
// CHECK:STDOUT: define i32 @"_CF.A.Main:I.Main"(ptr %self) #0 !dbg !24 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc15_16.1.n = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !29
// CHECK:STDOUT:   %.loc15_16.2 = load i32, ptr %.loc15_16.1.n, align 4, !dbg !29
// CHECK:STDOUT:   ret i32 %.loc15_16.2, !dbg !30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CCall.Main(ptr %a) #0 !dbg !32 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %A.as.I.impl.F.call = call i32 @"_CF.A.Main:I.Main"(ptr %a), !dbg !37
// CHECK:STDOUT:   ret i32 %A.as.I.impl.F.call, !dbg !38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !18 = !DIFile(filename: "class_method_defined_as_alias.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{!20, !21}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !24 = distinct !DISubprogram(name: "F", linkageName: "_CF.A.Main:I.Main", scope: null, file: !18, line: 14, type: !23, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !31)
// CHECK:STDOUT: !25 = !DILocalVariable(arg: 1, scope: !24, type: !21)
// CHECK:STDOUT: !29 = !DILocation(line: 15, column: 12, scope: !24)
// CHECK:STDOUT: !30 = !DILocation(line: 15, column: 5, scope: !24)
// CHECK:STDOUT: !31 = !{!25}
// CHECK:STDOUT: !32 = distinct !DISubprogram(name: "Call", linkageName: "_CCall.Main", scope: null, file: !18, line: 19, type: !23, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !39)
// CHECK:STDOUT: !33 = !DILocalVariable(arg: 1, scope: !32, type: !21)
// CHECK:STDOUT: !37 = !DILocation(line: 20, column: 10, scope: !32)
// CHECK:STDOUT: !38 = !DILocation(line: 20, column: 3, scope: !32)
// CHECK:STDOUT: !39 = !{!33}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'impl_method_defined_as_alias.carbon'
// CHECK:STDOUT: source_filename = "impl_method_defined_as_alias.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CF.C.Main(ptr %self) #0 !dbg !44 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc10_34.1.i = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !49
// CHECK:STDOUT:   %.loc10_34.2 = load i32, ptr %.loc10_34.1.i, align 4, !dbg !49
// CHECK:STDOUT:   ret i32 %.loc10_34.2, !dbg !50
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCall2.Main(ptr %c) #0 !dbg !52 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Call.call = call i32 @_CCall.Main.0676fa98bb7dbd66(ptr %c), !dbg !56
// CHECK:STDOUT:   ret void, !dbg !57
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CCall.Main.0676fa98bb7dbd66(ptr %i) #0 !dbg !59 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %I.WithSelf.F.call = call i32 @_CF.C.Main(ptr %i), !dbg !64
// CHECK:STDOUT:   ret i32 %I.WithSelf.F.call, !dbg !65
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!41}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{!20, !21}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !40 = !DIFile(filename: "impl_method_defined_as_alias.carbon", directory: "")
// CHECK:STDOUT: !41 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !40, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !42 = !{null, !21}
// CHECK:STDOUT: !43 = !DISubroutineType(types: !42)
// CHECK:STDOUT: !44 = distinct !DISubprogram(name: "F", linkageName: "_CF.C.Main", scope: null, file: !40, line: 10, type: !23, spFlags: DISPFlagDefinition, unit: !41, retainedNodes: !51)
// CHECK:STDOUT: !45 = !DILocalVariable(arg: 1, scope: !44, type: !21)
// CHECK:STDOUT: !49 = !DILocation(line: 10, column: 30, scope: !44)
// CHECK:STDOUT: !50 = !DILocation(line: 10, column: 23, scope: !44)
// CHECK:STDOUT: !51 = !{!45}
// CHECK:STDOUT: !52 = distinct !DISubprogram(name: "Call2", linkageName: "_CCall2.Main", scope: null, file: !40, line: 21, type: !43, spFlags: DISPFlagDefinition, unit: !41, retainedNodes: !58)
// CHECK:STDOUT: !53 = !DILocalVariable(arg: 1, scope: !52, type: !21)
// CHECK:STDOUT: !56 = !DILocation(line: 22, column: 3, scope: !52)
// CHECK:STDOUT: !57 = !DILocation(line: 21, column: 1, scope: !52)
// CHECK:STDOUT: !58 = !{!53}
// CHECK:STDOUT: !59 = distinct !DISubprogram(name: "Call", linkageName: "_CCall.Main.0676fa98bb7dbd66", scope: null, file: !40, line: 17, type: !23, spFlags: DISPFlagDefinition, unit: !41, retainedNodes: !66)
// CHECK:STDOUT: !60 = !DILocalVariable(arg: 1, scope: !59, type: !21)
// CHECK:STDOUT: !64 = !DILocation(line: 18, column: 10, scope: !59)
// CHECK:STDOUT: !65 = !DILocation(line: 18, column: 3, scope: !59)
// CHECK:STDOUT: !66 = !{!60}
// CHECK:STDOUT:
