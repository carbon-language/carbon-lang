// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
// CHECK:STDOUT: define i32 @"_CF.ImplsSomeInterface.Main:SomeInterface.Main"(i32 %x) !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.snegate = sub i32 0, %x, !dbg !7
// CHECK:STDOUT:   ret i32 %int.snegate, !dbg !8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CG.Main() !dbg !9 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %CallGenericMethod.call = call i32 @_CCallGenericMethod.Main.b330a91dfda257ee(i32 10), !dbg !10
// CHECK:STDOUT:   ret void, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr i32 @_CCallGenericMethod.Main.b330a91dfda257ee(i32 %x) !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc34_15 = call i32 @"_CF.ImplsSomeInterface.Main:SomeInterface.Main"(i32 %x), !dbg !13
// CHECK:STDOUT:   ret i32 %.loc34_15, !dbg !14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "call_impl_function.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "F", linkageName: "_CF.ImplsSomeInterface.Main:SomeInterface.Main", scope: null, file: !3, line: 20, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 21, column: 12, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 21, column: 5, scope: !4)
// CHECK:STDOUT: !9 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !3, line: 37, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !10 = !DILocation(line: 38, column: 3, scope: !9)
// CHECK:STDOUT: !11 = !DILocation(line: 37, column: 1, scope: !9)
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "CallGenericMethod", linkageName: "_CCallGenericMethod.Main.b330a91dfda257ee", scope: null, file: !3, line: 33, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !13 = !DILocation(line: 34, column: 10, scope: !12)
// CHECK:STDOUT: !14 = !DILocation(line: 34, column: 3, scope: !12)
