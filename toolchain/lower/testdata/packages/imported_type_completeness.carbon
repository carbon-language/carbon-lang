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

class C(T: type) {
  // Core.Int(123) is required to be complete here.
  fn F() -> Core.Int(123) {
    return 123;
  }

  // Core.Int(456) is required to be complete here.
  fn G(unused t: T) -> Core.Int(456) {
    return 456;
  }
}

fn G(generic T: type, t: T) {
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

base class C(T: type) {
  // Require Core.Int(11) and Core.Int(12) to be complete.
  virtual fn F(unused self) -> Core.Int(11) { return 11; }
  virtual fn G(unused self, unused t: T) -> Core.Int(12) { return 12; }
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

fn F(generic T: type, t: T) {
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

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'require_completion_in_call.carbon'
// CHECK:STDOUT: source_filename = "require_completion_in_call.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !18 = !DIFile(filename: "require_completion_in_call.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'use_without_completion_in_call.carbon'
// CHECK:STDOUT: source_filename = "use_without_completion_in_call.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @C.val.eeb = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CI.Main() #0 !dbg !29 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CG.Main.e43630e9a6c38c3f(), !dbg !30
// CHECK:STDOUT:   ret void, !dbg !31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.b7a0a8bb33ac84ff:core.Destroy.Core"(ptr %self) #0 !dbg !32 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CG.Main.e43630e9a6c38c3f() #0 !dbg !36 {
// CHECK:STDOUT:   %temp = alloca {}, align 1, !dbg !38
// CHECK:STDOUT:   %temp1 = alloca {}, align 1, !dbg !42
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp), !dbg !38
// CHECK:STDOUT:   %1 = call i123 @_CF.C.Main.e43630e9a6c38c3f(), !dbg !39
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.27484138f50a4c75(ptr @C.val.eeb), !dbg !38
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp1), !dbg !42
// CHECK:STDOUT:   %2 = call i456 @_CG.C.Main.e43630e9a6c38c3f(), !dbg !44
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.27484138f50a4c75(ptr @C.val.eeb), !dbg !42
// CHECK:STDOUT:   ret void, !dbg !47
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i123 @_CF.C.Main.e43630e9a6c38c3f() #0 !dbg !48 {
// CHECK:STDOUT:   ret i123 123, !dbg !51
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.27484138f50a4c75(ptr %self) #0 !dbg !52 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.b7a0a8bb33ac84ff:core.Destroy.Core"(ptr %self), !dbg !56
// CHECK:STDOUT:   ret void, !dbg !57
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i456 @_CG.C.Main.e43630e9a6c38c3f() #0 !dbg !59 {
// CHECK:STDOUT:   ret i456 456, !dbg !63
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Destroy.Core.27484138f50a4c75, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!23}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !14 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !18 = !DIFile(filename: "require_completion_in_call.carbon", directory: "")
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 123, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !DIBasicType(name: "int", size: 456, encoding: DW_ATE_signed)
// CHECK:STDOUT: !22 = !DIFile(filename: "use_without_completion_in_call.carbon", directory: "")
// CHECK:STDOUT: !23 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !22, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !24 = !{null}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !27 = !{null, !26}
// CHECK:STDOUT: !28 = !DISubroutineType(types: !27)
// CHECK:STDOUT: !29 = distinct !DISubprogram(name: "I", linkageName: "_CI.Main", scope: null, file: !22, line: 6, type: !25, spFlags: DISPFlagDefinition, unit: !23)
// CHECK:STDOUT: !30 = !DILocation(line: 9, column: 3, scope: !29)
// CHECK:STDOUT: !31 = !DILocation(line: 6, column: 1, scope: !29)
// CHECK:STDOUT: !32 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.b7a0a8bb33ac84ff:core.Destroy.Core", scope: null, file: !18, line: 19, type: !28, spFlags: DISPFlagDefinition, unit: !23, retainedNodes: !35)
// CHECK:STDOUT: !33 = !DILocalVariable(arg: 1, scope: !32, type: !26)
// CHECK:STDOUT: !34 = !DILocation(line: 19, column: 4, scope: !32)
// CHECK:STDOUT: !35 = !{!33}
// CHECK:STDOUT: !36 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.e43630e9a6c38c3f", scope: null, file: !18, line: 16, type: !25, spFlags: DISPFlagDefinition, unit: !23)
// CHECK:STDOUT: !38 = !DILocation(line: 19, column: 4, scope: !36)
// CHECK:STDOUT: !39 = !DILocation(line: 19, column: 3, scope: !36)
// CHECK:STDOUT: !40 = !{!20}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !42 = !DILocation(line: 22, column: 4, scope: !36)
// CHECK:STDOUT: !44 = !DILocation(line: 22, column: 3, scope: !36)
// CHECK:STDOUT: !45 = !{!21}
// CHECK:STDOUT: !46 = !DISubroutineType(types: !45)
// CHECK:STDOUT: !47 = !DILocation(line: 16, column: 1, scope: !36)
// CHECK:STDOUT: !48 = distinct !DISubprogram(name: "F", linkageName: "_CF.C.Main.e43630e9a6c38c3f", scope: null, file: !18, line: 6, type: !41, spFlags: DISPFlagDefinition, unit: !23)
// CHECK:STDOUT: !51 = !DILocation(line: 7, column: 5, scope: !48)
// CHECK:STDOUT: !52 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.27484138f50a4c75", scope: null, file: !14, line: 24, type: !28, spFlags: DISPFlagDefinition, unit: !23, retainedNodes: !58)
// CHECK:STDOUT: !53 = !DILocalVariable(arg: 1, scope: !52, type: !26)
// CHECK:STDOUT: !56 = !DILocation(line: 26, column: 5, scope: !52)
// CHECK:STDOUT: !57 = !DILocation(line: 24, column: 3, scope: !52)
// CHECK:STDOUT: !58 = !{!53}
// CHECK:STDOUT: !59 = distinct !DISubprogram(name: "G", linkageName: "_CG.C.Main.e43630e9a6c38c3f", scope: null, file: !18, line: 11, type: !46, spFlags: DISPFlagDefinition, unit: !23)
// CHECK:STDOUT: !63 = !DILocation(line: 12, column: 5, scope: !59)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'require_completion_in_vtable.carbon'
// CHECK:STDOUT: source_filename = "require_completion_in_vtable.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!65}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !64 = !DIFile(filename: "require_completion_in_vtable.carbon", directory: "")
// CHECK:STDOUT: !65 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !64, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'use_without_completion_in_vtable.carbon'
// CHECK:STDOUT: source_filename = "use_without_completion_in_vtable.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @_Cv.Main = global { ptr } zeroinitializer
// CHECK:STDOUT: @"_CC.Main.$vtable.6944efcfbb6a8360" = external unnamed_addr constant ptr
// CHECK:STDOUT: @C.val.loc9_1 = internal constant { ptr } { ptr @"_CC.Main.$vtable.6944efcfbb6a8360" }
// CHECK:STDOUT: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 0, ptr @_C__global_init.Main, ptr null }]
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define internal void @_C__global_init.Main() #0 !dbg !70 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 @_Cv.Main, ptr align 8 @C.val.loc9_1, i64 8, i1 false), !dbg !71
// CHECK:STDOUT:   ret void, !dbg !72
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!69}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !24 = !{null}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !68 = !DIFile(filename: "use_without_completion_in_vtable.carbon", directory: "")
// CHECK:STDOUT: !69 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !68, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !70 = distinct !DISubprogram(name: "__global_init", linkageName: "_C__global_init.Main", scope: null, file: !68, type: !25, spFlags: DISPFlagDefinition, unit: !69)
// CHECK:STDOUT: !71 = !DILocation(line: 9, column: 1, scope: !70)
// CHECK:STDOUT: !72 = !DILocation(line: 0, scope: !70)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'require_completion_in_virtual_call.carbon'
// CHECK:STDOUT: source_filename = "require_completion_in_virtual_call.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.95528a86e1127e2d:core.Destroy.Core"(ptr %self) #0 !dbg !75 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !77
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!74}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !26 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !27 = !{null, !26}
// CHECK:STDOUT: !28 = !DISubroutineType(types: !27)
// CHECK:STDOUT: !73 = !DIFile(filename: "require_completion_in_virtual_call.carbon", directory: "")
// CHECK:STDOUT: !74 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !73, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !75 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.95528a86e1127e2d:core.Destroy.Core", scope: null, file: !73, line: 7, type: !28, spFlags: DISPFlagDefinition, unit: !74, retainedNodes: !78)
// CHECK:STDOUT: !76 = !DILocalVariable(arg: 1, scope: !75, type: !26)
// CHECK:STDOUT: !77 = !DILocation(line: 7, column: 3, scope: !75)
// CHECK:STDOUT: !78 = !{!76}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'use_without_completion_in_virtual_call.carbon'
// CHECK:STDOUT: source_filename = "use_without_completion_in_virtual_call.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @C.val.d1d = internal constant {} zeroinitializer
// CHECK:STDOUT: @"_CC.Main.$vtable.9522ff908e97e267" = external unnamed_addr constant ptr
// CHECK:STDOUT: @C.val.d1d.loc9_11.3 = internal constant {} zeroinitializer
// CHECK:STDOUT: @C.val.9ff.anon = internal constant { ptr } { ptr @"_CC.Main.$vtable.9522ff908e97e267" }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCallF.Main() #0 !dbg !81 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_9.2.temp = alloca {}, align 1, !dbg !82
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc9_9.2.temp), !dbg !82
// CHECK:STDOUT:   call void @_CF.Main.9522ff908e97e267(ptr @C.val.d1d.loc9_11.3), !dbg !83
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.2eb69dd0c58be63c(ptr @C.val.d1d), !dbg !82
// CHECK:STDOUT:   ret void, !dbg !84
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.08d9e38b928ba575:core.Destroy.Core"(ptr %self) #0 !dbg !85 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !87
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.95528a86e1127e2d:core.Destroy.Core"(ptr %self) #0 !dbg !89 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !91
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.119330419006a03e:core.Destroy.Core"(ptr %self) #0 !dbg !93 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !95
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.9522ff908e97e267(ptr %t) #0 !dbg !97 {
// CHECK:STDOUT:   %1 = alloca { ptr }, align 8, !dbg !100
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %1), !dbg !100
// CHECK:STDOUT:   %vptr = getelementptr inbounds nuw { ptr }, ptr %1, i32 0, i32 0, !dbg !101
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %1, ptr align 8 @C.val.9ff.anon, i64 8, i1 false), !dbg !100
// CHECK:STDOUT:   %vtable = load ptr, ptr %1, align 8, !dbg !103
// CHECK:STDOUT:   %2 = call ptr @llvm.load.relative.i32(ptr %vtable, i32 0), !dbg !103
// CHECK:STDOUT:   %3 = call i11 %2(ptr %1), !dbg !103
// CHECK:STDOUT:   %vtable1 = load ptr, ptr %1, align 8, !dbg !106
// CHECK:STDOUT:   %4 = call ptr @llvm.load.relative.i32(ptr %vtable1, i32 4), !dbg !106
// CHECK:STDOUT:   %5 = call i12 %4(ptr %1, ptr %t), !dbg !106
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.c0567936d7d00ee6(ptr %1), !dbg !100
// CHECK:STDOUT:   ret void, !dbg !110
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.2eb69dd0c58be63c(ptr %self) #0 !dbg !112 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.08d9e38b928ba575:core.Destroy.Core"(ptr %self), !dbg !116
// CHECK:STDOUT:   ret void, !dbg !117
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i11 @_CF.C.Main.9522ff908e97e267(ptr %self) #0 !dbg !119 {
// CHECK:STDOUT:   ret i11 11, !dbg !124
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
// CHECK:STDOUT: declare ptr @llvm.load.relative.i32(ptr, i32) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i12 @_CG.C.Main.9522ff908e97e267(ptr %self, ptr %t) #0 !dbg !126 {
// CHECK:STDOUT:   ret i12 12, !dbg !133
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.c0567936d7d00ee6(ptr %self) #0 !dbg !135 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.119330419006a03e:core.Destroy.Core"(ptr %self), !dbg !139
// CHECK:STDOUT:   ret void, !dbg !140
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.load.relative.i32, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!80}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !14 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !24 = !{null}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !27 = !{null, !26}
// CHECK:STDOUT: !28 = !DISubroutineType(types: !27)
// CHECK:STDOUT: !64 = !DIFile(filename: "require_completion_in_vtable.carbon", directory: "")
// CHECK:STDOUT: !66 = !DIBasicType(name: "int", size: 11, encoding: DW_ATE_signed)
// CHECK:STDOUT: !67 = !DIBasicType(name: "int", size: 12, encoding: DW_ATE_signed)
// CHECK:STDOUT: !73 = !DIFile(filename: "require_completion_in_virtual_call.carbon", directory: "")
// CHECK:STDOUT: !79 = !DIFile(filename: "use_without_completion_in_virtual_call.carbon", directory: "")
// CHECK:STDOUT: !80 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !79, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !81 = distinct !DISubprogram(name: "CallF", linkageName: "_CCallF.Main", scope: null, file: !79, line: 8, type: !25, spFlags: DISPFlagDefinition, unit: !80)
// CHECK:STDOUT: !82 = !DILocation(line: 9, column: 8, scope: !81)
// CHECK:STDOUT: !83 = !DILocation(line: 9, column: 3, scope: !81)
// CHECK:STDOUT: !84 = !DILocation(line: 8, column: 1, scope: !81)
// CHECK:STDOUT: !85 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.08d9e38b928ba575:core.Destroy.Core", scope: null, file: !79, line: 9, type: !28, spFlags: DISPFlagDefinition, unit: !80, retainedNodes: !88)
// CHECK:STDOUT: !86 = !DILocalVariable(arg: 1, scope: !85, type: !26)
// CHECK:STDOUT: !87 = !DILocation(line: 9, column: 8, scope: !85)
// CHECK:STDOUT: !88 = !{!86}
// CHECK:STDOUT: !89 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.95528a86e1127e2d:core.Destroy.Core", scope: null, file: !73, line: 7, type: !28, spFlags: DISPFlagDefinition, unit: !80, retainedNodes: !92)
// CHECK:STDOUT: !90 = !DILocalVariable(arg: 1, scope: !89, type: !26)
// CHECK:STDOUT: !91 = !DILocation(line: 7, column: 3, scope: !89)
// CHECK:STDOUT: !92 = !{!90}
// CHECK:STDOUT: !93 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.119330419006a03e:core.Destroy.Core", scope: null, file: !73, line: 7, type: !28, spFlags: DISPFlagDefinition, unit: !80, retainedNodes: !96)
// CHECK:STDOUT: !94 = !DILocalVariable(arg: 1, scope: !93, type: !26)
// CHECK:STDOUT: !95 = !DILocation(line: 7, column: 3, scope: !93)
// CHECK:STDOUT: !96 = !{!94}
// CHECK:STDOUT: !97 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.9522ff908e97e267", scope: null, file: !73, line: 6, type: !28, spFlags: DISPFlagDefinition, unit: !80, retainedNodes: !111)
// CHECK:STDOUT: !98 = !DILocalVariable(arg: 1, scope: !97, type: !26)
// CHECK:STDOUT: !100 = !DILocation(line: 7, column: 3, scope: !97)
// CHECK:STDOUT: !101 = !DILocation(line: 7, column: 17, scope: !97)
// CHECK:STDOUT: !103 = !DILocation(line: 8, column: 3, scope: !97)
// CHECK:STDOUT: !104 = !{!66, !26}
// CHECK:STDOUT: !105 = !DISubroutineType(types: !104)
// CHECK:STDOUT: !106 = !DILocation(line: 9, column: 3, scope: !97)
// CHECK:STDOUT: !108 = !{!67, !26, !26}
// CHECK:STDOUT: !109 = !DISubroutineType(types: !108)
// CHECK:STDOUT: !110 = !DILocation(line: 6, column: 1, scope: !97)
// CHECK:STDOUT: !111 = !{!98}
// CHECK:STDOUT: !112 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.2eb69dd0c58be63c", scope: null, file: !14, line: 24, type: !28, spFlags: DISPFlagDefinition, unit: !80, retainedNodes: !118)
// CHECK:STDOUT: !113 = !DILocalVariable(arg: 1, scope: !112, type: !26)
// CHECK:STDOUT: !116 = !DILocation(line: 26, column: 5, scope: !112)
// CHECK:STDOUT: !117 = !DILocation(line: 24, column: 3, scope: !112)
// CHECK:STDOUT: !118 = !{!113}
// CHECK:STDOUT: !119 = distinct !DISubprogram(name: "F", linkageName: "_CF.C.Main.9522ff908e97e267", scope: null, file: !64, line: 6, type: !105, spFlags: DISPFlagDefinition, unit: !80, retainedNodes: !125)
// CHECK:STDOUT: !120 = !DILocalVariable(arg: 1, scope: !119, type: !26)
// CHECK:STDOUT: !124 = !DILocation(line: 6, column: 47, scope: !119)
// CHECK:STDOUT: !125 = !{!120}
// CHECK:STDOUT: !126 = distinct !DISubprogram(name: "G", linkageName: "_CG.C.Main.9522ff908e97e267", scope: null, file: !64, line: 7, type: !109, spFlags: DISPFlagDefinition, unit: !80, retainedNodes: !134)
// CHECK:STDOUT: !127 = !DILocalVariable(arg: 1, scope: !126, type: !26)
// CHECK:STDOUT: !128 = !DILocalVariable(arg: 2, scope: !126, type: !26)
// CHECK:STDOUT: !133 = !DILocation(line: 7, column: 60, scope: !126)
// CHECK:STDOUT: !134 = !{!127, !128}
// CHECK:STDOUT: !135 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.c0567936d7d00ee6", scope: null, file: !14, line: 24, type: !28, spFlags: DISPFlagDefinition, unit: !80, retainedNodes: !141)
// CHECK:STDOUT: !136 = !DILocalVariable(arg: 1, scope: !135, type: !26)
// CHECK:STDOUT: !139 = !DILocation(line: 26, column: 5, scope: !135)
// CHECK:STDOUT: !140 = !DILocation(line: 24, column: 3, scope: !135)
// CHECK:STDOUT: !141 = !{!136}
// CHECK:STDOUT:
