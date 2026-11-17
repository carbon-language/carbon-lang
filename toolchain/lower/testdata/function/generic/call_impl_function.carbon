// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/generic/call_impl_function.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/generic/call_impl_function.carbon


interface SomeInterface(T: type) {
  fn F(x: T) -> T;
}

class SomeParam {}

class ImplsSomeInterface {}
impl ImplsSomeInterface as SomeInterface(i32) {
  fn F(x: i32) -> i32 {
    return -x;
  }
}

// TODO: This asserts with "Mangled name collision".
// impl ImplsSomeInterface as SomeInterface(SomeParam) {
  // fn F(x: SomeParam) -> SomeParam {
    // return {};
  // }
// }


fn CallGenericMethod(generic T: type, generic U: SomeInterface(T), x: T) -> T {
  return U.F(x);
}

fn G() {
  CallGenericMethod(i32, ImplsSomeInterface, 10);
  // TODO: Fix above
  // CallGenericMethod(SomeParam, ImplsSomeInterface, {});
}


// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'call_impl_function.carbon'
// CHECK:STDOUT: source_filename = "call_impl_function.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @"_CF.ImplsSomeInterface.Main:SomeInterface.55456748a8a2eb70.Main"(i32 %x) #0 !dbg !27 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.Negate.impl.Op.call = sub i32 0, %x, !dbg !33
// CHECK:STDOUT:   ret i32 %Int.as.Negate.impl.Op.call, !dbg !34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !36 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %CallGenericMethod.call = call i32 @_CCallGenericMethod.Main.d22799102cbf9a02(i32 10), !dbg !37
// CHECK:STDOUT:   ret void, !dbg !38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CCallGenericMethod.Main.d22799102cbf9a02(i32 %x) #0 !dbg !39 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %SomeInterface.WithSelf.F.call = call i32 @"_CF.ImplsSomeInterface.Main:SomeInterface.55456748a8a2eb70.Main"(i32 %x), !dbg !45
// CHECK:STDOUT:   ret i32 %SomeInterface.WithSelf.F.call, !dbg !46
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!21}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIFile(filename: "call_impl_function.carbon", directory: "")
// CHECK:STDOUT: !21 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !20, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !22 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !23 = !{!22, !22}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !25 = !{null}
// CHECK:STDOUT: !26 = !DISubroutineType(types: !25)
// CHECK:STDOUT: !27 = distinct !DISubprogram(name: "F", linkageName: "_CF.ImplsSomeInterface.Main:SomeInterface.55456748a8a2eb70.Main", scope: null, file: !20, line: 22, type: !24, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !35)
// CHECK:STDOUT: !28 = !DILocalVariable(arg: 1, scope: !27, type: !22)
// CHECK:STDOUT: !33 = !DILocation(line: 23, column: 12, scope: !27)
// CHECK:STDOUT: !34 = !DILocation(line: 23, column: 5, scope: !27)
// CHECK:STDOUT: !35 = !{!28}
// CHECK:STDOUT: !36 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !20, line: 39, type: !26, spFlags: DISPFlagDefinition, unit: !21)
// CHECK:STDOUT: !37 = !DILocation(line: 40, column: 3, scope: !36)
// CHECK:STDOUT: !38 = !DILocation(line: 39, column: 1, scope: !36)
// CHECK:STDOUT: !39 = distinct !DISubprogram(name: "CallGenericMethod", linkageName: "_CCallGenericMethod.Main.d22799102cbf9a02", scope: null, file: !20, line: 35, type: !24, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !47)
// CHECK:STDOUT: !40 = !DILocalVariable(arg: 1, scope: !39, type: !22)
// CHECK:STDOUT: !45 = !DILocation(line: 36, column: 10, scope: !39)
// CHECK:STDOUT: !46 = !DILocation(line: 36, column: 3, scope: !39)
// CHECK:STDOUT: !47 = !{!40}
// CHECK:STDOUT:
