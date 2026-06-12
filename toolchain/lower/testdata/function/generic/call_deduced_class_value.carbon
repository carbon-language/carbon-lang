// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/parts/int.carbon
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/parts/destroy.carbon
// EXTRA-ARGS: --custom-core --exclude-dump-file-prefix=min_prelude/
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/generic/call_deduced_class_value.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/generic/call_deduced_class_value.carbon

// --- min_prelude/core.carbon

package Core library "prelude";

export import library "prelude/parts/int";
export import library "prelude/parts/destroy";

// --- call_deduced_class_value.carbon

library "[[@TEST_NAME]]";

class Data {
  var a: i32;
  var b: i32;

  // TODO: Change to Core.Destructor when possible.
  impl as Core.Destroy {
    fn Op(ref self);
  }
}

// CHECK:STDERR: call_deduced_class_value.carbon:[[@LINE+4]]:6: warning: binding `d` unused [UnusedBinding]
// CHECK:STDERR: fn G(d:! Data) {}
// CHECK:STDERR:      ^~~~~~~~
// CHECK:STDERR:
fn G(d:! Data) {}

// Deducing `d` converts the struct literal to a `Data` value, which
// materializes a temporary that is destroyed at the end of the statement. The
// temporary's storage and field accesses are runtime instructions generated
// during deduction; they must survive deduction's discarding of dead constant
// instructions and lower to the temporary's alloca and field GEPs, alongside
// the call to the deduced specific and the destroy call.
fn Run() {
  G({.a = 1, .b = 2});
}

// CHECK:STDOUT: ; ModuleID = 'call_deduced_class_value.carbon'
// CHECK:STDOUT: source_filename = "call_deduced_class_value.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @Data.val = internal constant { i32, i32 } { i32 1, i32 2 }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Data.Main:Destroy.Core"(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @main() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc27_20.2.temp = alloca { i32, i32 }, align 4, !dbg !8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc27_20.2.temp), !dbg !8
// CHECK:STDOUT:   %.loc27_20.3.a = getelementptr inbounds nuw { i32, i32 }, ptr %.loc27_20.2.temp, i32 0, i32 0, !dbg !8
// CHECK:STDOUT:   %.loc27_20.4.b = getelementptr inbounds nuw { i32, i32 }, ptr %.loc27_20.2.temp, i32 0, i32 1, !dbg !8
// CHECK:STDOUT:   call void @_CG.Main.63ce70a0a5b9c75e(), !dbg !9
// CHECK:STDOUT:   call void @"_COp.9e6fec8ec8e188d4:core.Destroy.Core"(ptr @Data.val), !dbg !9
// CHECK:STDOUT:   ret i32 0, !dbg !10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.5e27612b9dd31a14:core.Destroy.Core"(ptr %self) #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.f30036b66b652106:core.Destroy.Core"(ptr %self) #0 !dbg !17 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.9e6fec8ec8e188d4:core.Destroy.Core"(ptr %self) #0 !dbg !24 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CG.Main.63ce70a0a5b9c75e() #0 !dbg !28 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "call_deduced_class_value.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !3, line: 26, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7}
// CHECK:STDOUT: !7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !8 = !DILocation(line: 27, column: 5, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 27, column: 3, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 26, column: 1, scope: !4)
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "Op", linkageName: "_COp.5e27612b9dd31a14:core.Destroy.Core", scope: null, file: !3, line: 27, type: !12, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !14)
// CHECK:STDOUT: !12 = !DISubroutineType(types: !13)
// CHECK:STDOUT: !13 = !{null, !7}
// CHECK:STDOUT: !14 = !{!15}
// CHECK:STDOUT: !15 = !DILocalVariable(arg: 1, scope: !11, type: !7)
// CHECK:STDOUT: !16 = !DILocation(line: 27, column: 3, scope: !11)
// CHECK:STDOUT: !17 = distinct !DISubprogram(name: "Op", linkageName: "_COp.f30036b66b652106:core.Destroy.Core", scope: null, file: !3, line: 27, type: !18, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !21)
// CHECK:STDOUT: !18 = !DISubroutineType(types: !19)
// CHECK:STDOUT: !19 = !{null, !20}
// CHECK:STDOUT: !20 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !21 = !{!22}
// CHECK:STDOUT: !22 = !DILocalVariable(arg: 1, scope: !17, type: !20)
// CHECK:STDOUT: !23 = !DILocation(line: 27, column: 3, scope: !17)
// CHECK:STDOUT: !24 = distinct !DISubprogram(name: "Op", linkageName: "_COp.9e6fec8ec8e188d4:core.Destroy.Core", scope: null, file: !3, line: 27, type: !18, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !25)
// CHECK:STDOUT: !25 = !{!26}
// CHECK:STDOUT: !26 = !DILocalVariable(arg: 1, scope: !24, type: !20)
// CHECK:STDOUT: !27 = !DILocation(line: 27, column: 3, scope: !24)
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.63ce70a0a5b9c75e", scope: null, file: !3, line: 18, type: !29, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !29 = !DISubroutineType(types: !30)
// CHECK:STDOUT: !30 = !{null}
// CHECK:STDOUT: !31 = !DILocation(line: 18, column: 1, scope: !28)
