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


interface SomeInterface(T:! type) {
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


fn CallGenericMethod(T:! type, U:! SomeInterface(T), x: T) -> T {
  return U.F(x);
}

fn G() {
  CallGenericMethod(i32, ImplsSomeInterface, 10);
  // TODO: Fix above
  // CallGenericMethod(SomeParam, ImplsSomeInterface, {});
}


// CHECK:STDOUT: ; ModuleID = 'call_impl_function.carbon'
// CHECK:STDOUT: source_filename = "call_impl_function.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @"_CF.ImplsSomeInterface.Main:SomeInterface.c8fd090a2e741f70.Main"(i32 %x) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.Negate.impl.Op.call = sub i32 0, %x, !dbg !10
// CHECK:STDOUT:   ret i32 %Int.as.Negate.impl.Op.call, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %CallGenericMethod.call = call i32 @_CCallGenericMethod.Main.ae0f2ca660e51227(i32 10), !dbg !15
// CHECK:STDOUT:   ret void, !dbg !16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CCallGenericMethod.Main.ae0f2ca660e51227(i32 %x) #0 !dbg !17 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %SomeInterface.WithSelf.F.call = call i32 @"_CF.ImplsSomeInterface.Main:SomeInterface.c8fd090a2e741f70.Main"(i32 %x), !dbg !20
// CHECK:STDOUT:   ret i32 %SomeInterface.WithSelf.F.call, !dbg !21
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
// CHECK:STDOUT: !3 = !DIFile(filename: "call_impl_function.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "F", linkageName: "_CF.ImplsSomeInterface.Main:SomeInterface.c8fd090a2e741f70.Main", scope: null, file: !3, line: 22, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !8)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7, !7}
// CHECK:STDOUT: !7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !8 = !{!9}
// CHECK:STDOUT: !9 = !DILocalVariable(arg: 1, scope: !4, type: !7)
// CHECK:STDOUT: !10 = !DILocation(line: 23, column: 12, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 23, column: 5, scope: !4)
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !3, line: 39, type: !13, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !13 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !14 = !{null}
// CHECK:STDOUT: !15 = !DILocation(line: 40, column: 3, scope: !12)
// CHECK:STDOUT: !16 = !DILocation(line: 39, column: 1, scope: !12)
// CHECK:STDOUT: !17 = distinct !DISubprogram(name: "CallGenericMethod", linkageName: "_CCallGenericMethod.Main.ae0f2ca660e51227", scope: null, file: !3, line: 35, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !18)
// CHECK:STDOUT: !18 = !{!19}
// CHECK:STDOUT: !19 = !DILocalVariable(arg: 1, scope: !17, type: !7)
// CHECK:STDOUT: !20 = !DILocation(line: 36, column: 10, scope: !17)
// CHECK:STDOUT: !21 = !DILocation(line: 36, column: 3, scope: !17)
