// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/call/tuple_param_with_return_slot.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/call/tuple_param_with_return_slot.carbon

fn F(a: (), b: (i32,), c: (i32, i32)) -> (i32, i32, i32) {
  return (b.0, c.0, c.1);
}

fn Main() {
  F((), (1,), (2, 3));
}

// CHECK:STDOUT: ; ModuleID = 'tuple_param_with_return_slot.carbon'
// CHECK:STDOUT: source_filename = "tuple_param_with_return_slot.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @tuple.3.loc16_4.4 = internal constant { i32, i32 } { i32 2, i32 3 }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @F(ptr sret({ i32, i32, i32 }) %return, { i32 } %b, ptr %c) !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc12_12.tuple.index = extractvalue { i32 } %b, 0
// CHECK:STDOUT:   %.loc12_17.tuple.index = getelementptr inbounds nuw { i32, i32 }, ptr %c, i32 0, i32 0
// CHECK:STDOUT:   %.loc12_17.tuple.index.load = load i32, ptr %.loc12_17.tuple.index, align 4
// CHECK:STDOUT:   %.loc12_22.tuple.index = getelementptr inbounds nuw { i32, i32 }, ptr %c, i32 0, i32 1
// CHECK:STDOUT:   %.loc12_22.tuple.index.load = load i32, ptr %.loc12_22.tuple.index, align 4
// CHECK:STDOUT:   %.loc12_24.2.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %return, i32 0, i32 0
// CHECK:STDOUT:   store i32 %.loc12_12.tuple.index, ptr %.loc12_24.2.tuple.elem, align 4
// CHECK:STDOUT:   %.loc12_24.4.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %return, i32 0, i32 1
// CHECK:STDOUT:   store i32 %.loc12_17.tuple.index.load, ptr %.loc12_24.4.tuple.elem, align 4
// CHECK:STDOUT:   %.loc12_24.6.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %return, i32 0, i32 2
// CHECK:STDOUT:   store i32 %.loc12_22.tuple.index.load, ptr %.loc12_24.6.tuple.elem, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @Main() !dbg !7 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc16_4.1.temp = alloca { i32, i32, i32 }, align 8
// CHECK:STDOUT:   call void @F(ptr %.loc16_4.1.temp, { i32 } { i32 1 }, ptr @tuple.3.loc16_4.4)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder i32 1, { 3, 0, 1, 2 }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "tuple_param_with_return_slot.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "F", scope: null, file: !3, line: 11, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = distinct !DISubprogram(name: "Main", scope: null, file: !3, line: 15, type: !5, spFlags: DISPFlagDefinition, unit: !2)
