// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/packages/imported_type_completeness.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/packages/imported_type_completeness.carbon

// --- require_completion_in_call.carbon

library "[[@TEST_NAME]]";

class C(T:! type) {
  // Core.Int(123) is required to be complete here.
  fn F() -> Core.Int(123) {
    return 123;
  }

  // Core.Int(456) is required to be complete here.
  fn G(unused t: T) -> Core.Int(456) {
    return 456;
  }
}

fn G(T:! type, t: T) {
  // Core.Int(123) is also required to be complete here. But because it's not
  // dependent, we don't require it to be complete when forming a specific for G.
  ({} as C(T)).F();

  // Same for Core.Int(456).
  ({} as C(T)).G(t);
}

// --- use_without_completion_in_call.carbon

library "[[@TEST_NAME]]";

import library "require_completion_in_call";

fn I() {
  // OK, even though nothing in the compilation of this file requires i123 or
  // i456 to be complete.
  G((), ());
}

// --- require_completion_in_vtable.carbon

library "[[@TEST_NAME]]";

base class C(T:! type) {
  // Require Core.Int(11) and Core.Int(12) to be complete.
  virtual fn F[unused self: Self]() -> Core.Int(11) { return 11; }
  virtual fn G[unused self: Self](unused t: T) -> Core.Int(12) { return 12; }
}

// --- use_without_completion_in_vtable.carbon

library "[[@TEST_NAME]]";

import library "require_completion_in_vtable";

// Trigger emission of C({})'s vtable without requiring Core.Int(N) to be
// complete in this file.
// TODO: This currently emits an undefined external reference to the vtable!
var v: C({}) = {};

// --- require_completion_in_virtual_call.carbon

library "[[@TEST_NAME]]";

import library "require_completion_in_vtable";

fn F(T:! type, t: T) {
  var c: C(T) = {};
  c.F();
  c.G(t);
}

// --- use_without_completion_in_virtual_call.carbon

library "[[@TEST_NAME]]";

import library "require_completion_in_virtual_call";

class C {}

fn CallF() {
  F(C, {} as C);
}

// CHECK:STDOUT: ; ModuleID = 'require_completion_in_call.carbon'
// CHECK:STDOUT: source_filename = "require_completion_in_call.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "require_completion_in_call.carbon", directory: "")
// CHECK:STDOUT: ; ModuleID = 'use_without_completion_in_call.carbon'
// CHECK:STDOUT: source_filename = "use_without_completion_in_call.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @C.val.7d0 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CI.Main() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CG.Main.e43630e9a6c38c3f(), !dbg !7
// CHECK:STDOUT:   ret void, !dbg !8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CG.Main.e43630e9a6c38c3f() #0 !dbg !9 {
// CHECK:STDOUT:   %temp = alloca {}, align 8, !dbg !11
// CHECK:STDOUT:   %temp1 = alloca {}, align 8, !dbg !12
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp), !dbg !11
// CHECK:STDOUT:   %1 = call i123 @_CF.C.Main.e43630e9a6c38c3f(), !dbg !13
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp1), !dbg !12
// CHECK:STDOUT:   %2 = call i456 @_CG.C.Main.e43630e9a6c38c3f(), !dbg !14
// CHECK:STDOUT:   ret void, !dbg !15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i123 @_CF.C.Main.e43630e9a6c38c3f() #0 !dbg !16 {
// CHECK:STDOUT:   ret i123 123, !dbg !20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i456 @_CG.C.Main.e43630e9a6c38c3f() #0 !dbg !21 {
// CHECK:STDOUT:   ret i456 456, !dbg !25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
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
// CHECK:STDOUT: !3 = !DIFile(filename: "use_without_completion_in_call.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "I", linkageName: "_CI.Main", scope: null, file: !3, line: 6, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DILocation(line: 9, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 6, column: 1, scope: !4)
// CHECK:STDOUT: !9 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.e43630e9a6c38c3f", scope: null, file: !10, line: 16, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !10 = !DIFile(filename: "require_completion_in_call.carbon", directory: "")
// CHECK:STDOUT: !11 = !DILocation(line: 19, column: 4, scope: !9)
// CHECK:STDOUT: !12 = !DILocation(line: 22, column: 4, scope: !9)
// CHECK:STDOUT: !13 = !DILocation(line: 19, column: 3, scope: !9)
// CHECK:STDOUT: !14 = !DILocation(line: 22, column: 3, scope: !9)
// CHECK:STDOUT: !15 = !DILocation(line: 16, column: 1, scope: !9)
// CHECK:STDOUT: !16 = distinct !DISubprogram(name: "F", linkageName: "_CF.C.Main.e43630e9a6c38c3f", scope: null, file: !10, line: 6, type: !17, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !17 = !DISubroutineType(types: !18)
// CHECK:STDOUT: !18 = !{!19}
// CHECK:STDOUT: !19 = !DIBasicType(name: "int", size: 123, encoding: DW_ATE_signed)
// CHECK:STDOUT: !20 = !DILocation(line: 7, column: 5, scope: !16)
// CHECK:STDOUT: !21 = distinct !DISubprogram(name: "G", linkageName: "_CG.C.Main.e43630e9a6c38c3f", scope: null, file: !10, line: 11, type: !22, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !22 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !23 = !{!24}
// CHECK:STDOUT: !24 = !DIBasicType(name: "int", size: 456, encoding: DW_ATE_signed)
// CHECK:STDOUT: !25 = !DILocation(line: 12, column: 5, scope: !21)
// CHECK:STDOUT: ; ModuleID = 'require_completion_in_vtable.carbon'
// CHECK:STDOUT: source_filename = "require_completion_in_vtable.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "require_completion_in_vtable.carbon", directory: "")
// CHECK:STDOUT: ; ModuleID = 'use_without_completion_in_vtable.carbon'
// CHECK:STDOUT: source_filename = "use_without_completion_in_vtable.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @_Cv.Main = global { ptr } zeroinitializer
// CHECK:STDOUT: @"_CC.Main.$vtable.6944efcfbb6a8360" = external unnamed_addr constant ptr
// CHECK:STDOUT: @C.val.loc9_1 = internal constant { ptr } { ptr @"_CC.Main.$vtable.6944efcfbb6a8360" }
// CHECK:STDOUT: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 0, ptr @_C__global_init.Main, ptr null }]
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define internal void @_C__global_init.Main() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 @_Cv.Main, ptr align 8 @C.val.loc9_1, i64 8, i1 false), !dbg !7
// CHECK:STDOUT:   ret void, !dbg !8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "use_without_completion_in_vtable.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "__global_init", linkageName: "_C__global_init.Main", scope: null, file: !3, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DILocation(line: 9, column: 1, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 0, scope: !4)
// CHECK:STDOUT: ; ModuleID = 'require_completion_in_virtual_call.carbon'
// CHECK:STDOUT: source_filename = "require_completion_in_virtual_call.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "require_completion_in_virtual_call.carbon", directory: "")
// CHECK:STDOUT: ; ModuleID = 'use_without_completion_in_virtual_call.carbon'
// CHECK:STDOUT: source_filename = "use_without_completion_in_virtual_call.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @C.val.4a3 = internal constant {} zeroinitializer
// CHECK:STDOUT: @"_CC.Main.$vtable.0c4ab795cec6cb27" = external unnamed_addr constant ptr
// CHECK:STDOUT: @C.val.4a3.loc9_11.3 = internal constant {} zeroinitializer
// CHECK:STDOUT: @C.val.011.anon = internal constant { ptr } { ptr @"_CC.Main.$vtable.0c4ab795cec6cb27" }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCallF.Main() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_9.2.temp = alloca {}, align 8, !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc9_9.2.temp), !dbg !7
// CHECK:STDOUT:   call void @_CF.Main.0c4ab795cec6cb27(ptr @C.val.4a3.loc9_11.3), !dbg !8
// CHECK:STDOUT:   ret void, !dbg !9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.0c4ab795cec6cb27(ptr %t) #0 !dbg !10 {
// CHECK:STDOUT:   %1 = alloca { ptr }, align 8, !dbg !17
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %1), !dbg !17
// CHECK:STDOUT:   %vptr = getelementptr inbounds nuw { ptr }, ptr %1, i32 0, i32 0, !dbg !18
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %1, ptr align 8 @C.val.011.anon, i64 8, i1 false), !dbg !17
// CHECK:STDOUT:   %vtable = load ptr, ptr %1, align 8, !dbg !19
// CHECK:STDOUT:   %2 = call ptr @llvm.load.relative.i32(ptr %vtable, i32 0), !dbg !19
// CHECK:STDOUT:   %3 = call i11 %2(ptr %1), !dbg !19
// CHECK:STDOUT:   %vtable1 = load ptr, ptr %1, align 8, !dbg !20
// CHECK:STDOUT:   %4 = call ptr @llvm.load.relative.i32(ptr %vtable1, i32 4), !dbg !20
// CHECK:STDOUT:   %5 = call i12 %4(ptr %1, ptr %t), !dbg !20
// CHECK:STDOUT:   ret void, !dbg !21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i11 @_CF.C.Main.0c4ab795cec6cb27(ptr %self) #0 !dbg !22 {
// CHECK:STDOUT:   ret i11 11, !dbg !29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
// CHECK:STDOUT: declare ptr @llvm.load.relative.i32(ptr, i32) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i12 @_CG.C.Main.0c4ab795cec6cb27(ptr %self, ptr %t) #0 !dbg !30 {
// CHECK:STDOUT:   ret i12 12, !dbg !37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.load.relative.i32, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "use_without_completion_in_virtual_call.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "CallF", linkageName: "_CCallF.Main", scope: null, file: !3, line: 8, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DILocation(line: 9, column: 8, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 9, column: 3, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 8, column: 1, scope: !4)
// CHECK:STDOUT: !10 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.0c4ab795cec6cb27", scope: null, file: !11, line: 6, type: !12, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !15)
// CHECK:STDOUT: !11 = !DIFile(filename: "require_completion_in_virtual_call.carbon", directory: "")
// CHECK:STDOUT: !12 = !DISubroutineType(types: !13)
// CHECK:STDOUT: !13 = !{null, !14}
// CHECK:STDOUT: !14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !15 = !{!16}
// CHECK:STDOUT: !16 = !DILocalVariable(arg: 1, scope: !10, type: !14)
// CHECK:STDOUT: !17 = !DILocation(line: 7, column: 3, scope: !10)
// CHECK:STDOUT: !18 = !DILocation(line: 7, column: 17, scope: !10)
// CHECK:STDOUT: !19 = !DILocation(line: 8, column: 3, scope: !10)
// CHECK:STDOUT: !20 = !DILocation(line: 9, column: 3, scope: !10)
// CHECK:STDOUT: !21 = !DILocation(line: 6, column: 1, scope: !10)
// CHECK:STDOUT: !22 = distinct !DISubprogram(name: "F", linkageName: "_CF.C.Main.0c4ab795cec6cb27", scope: null, file: !23, line: 6, type: !24, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !27)
// CHECK:STDOUT: !23 = !DIFile(filename: "require_completion_in_vtable.carbon", directory: "")
// CHECK:STDOUT: !24 = !DISubroutineType(types: !25)
// CHECK:STDOUT: !25 = !{!26, !14}
// CHECK:STDOUT: !26 = !DIBasicType(name: "int", size: 11, encoding: DW_ATE_signed)
// CHECK:STDOUT: !27 = !{!28}
// CHECK:STDOUT: !28 = !DILocalVariable(arg: 1, scope: !22, type: !14)
// CHECK:STDOUT: !29 = !DILocation(line: 6, column: 55, scope: !22)
// CHECK:STDOUT: !30 = distinct !DISubprogram(name: "G", linkageName: "_CG.C.Main.0c4ab795cec6cb27", scope: null, file: !23, line: 7, type: !31, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !34)
// CHECK:STDOUT: !31 = !DISubroutineType(types: !32)
// CHECK:STDOUT: !32 = !{!33, !14, !14}
// CHECK:STDOUT: !33 = !DIBasicType(name: "int", size: 12, encoding: DW_ATE_signed)
// CHECK:STDOUT: !34 = !{!35, !36}
// CHECK:STDOUT: !35 = !DILocalVariable(arg: 1, scope: !30, type: !14)
// CHECK:STDOUT: !36 = !DILocalVariable(arg: 2, scope: !30, type: !14)
// CHECK:STDOUT: !37 = !DILocation(line: 7, column: 66, scope: !30)
