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
// CHECK:STDOUT: @C.val.c84 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CI.Main() #0 !dbg !26 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CG.Main.e43630e9a6c38c3f(), !dbg !27
// CHECK:STDOUT:   ret void, !dbg !28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CG.Main.e43630e9a6c38c3f() #0 !dbg !29 {
// CHECK:STDOUT:   %temp = alloca {}, align 1, !dbg !31
// CHECK:STDOUT:   %temp1 = alloca {}, align 1, !dbg !38
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp), !dbg !31
// CHECK:STDOUT:   %1 = call i123 @_CF.C.Main.e43630e9a6c38c3f(), !dbg !32
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.7021b132e5818ff2(ptr @C.val.c84), !dbg !31
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp1), !dbg !38
// CHECK:STDOUT:   %2 = call i456 @_CG.C.Main.e43630e9a6c38c3f(), !dbg !40
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.7021b132e5818ff2(ptr @C.val.c84), !dbg !38
// CHECK:STDOUT:   ret void, !dbg !43
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i123 @_CF.C.Main.e43630e9a6c38c3f() #0 !dbg !44 {
// CHECK:STDOUT:   ret i123 123, !dbg !47
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.7021b132e5818ff2(ptr %t) #0 !dbg !48 {
// CHECK:STDOUT:   ret void, !dbg !53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i456 @_CG.C.Main.e43630e9a6c38c3f() #0 !dbg !55 {
// CHECK:STDOUT:   ret i456 456, !dbg !59
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.7021b132e5818ff2, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!23}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !18 = !DIFile(filename: "require_completion_in_call.carbon", directory: "")
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 123, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !DIBasicType(name: "int", size: 456, encoding: DW_ATE_signed)
// CHECK:STDOUT: !22 = !DIFile(filename: "use_without_completion_in_call.carbon", directory: "")
// CHECK:STDOUT: !23 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !22, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !24 = !{null}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "I", linkageName: "_CI.Main", scope: null, file: !22, line: 6, type: !25, spFlags: DISPFlagDefinition, unit: !23)
// CHECK:STDOUT: !27 = !DILocation(line: 9, column: 3, scope: !26)
// CHECK:STDOUT: !28 = !DILocation(line: 6, column: 1, scope: !26)
// CHECK:STDOUT: !29 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.e43630e9a6c38c3f", scope: null, file: !18, line: 16, type: !25, spFlags: DISPFlagDefinition, unit: !23)
// CHECK:STDOUT: !31 = !DILocation(line: 19, column: 4, scope: !29)
// CHECK:STDOUT: !32 = !DILocation(line: 19, column: 3, scope: !29)
// CHECK:STDOUT: !33 = !{!20}
// CHECK:STDOUT: !34 = !DISubroutineType(types: !33)
// CHECK:STDOUT: !35 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !36 = !{null, !35}
// CHECK:STDOUT: !37 = !DISubroutineType(types: !36)
// CHECK:STDOUT: !38 = !DILocation(line: 22, column: 4, scope: !29)
// CHECK:STDOUT: !40 = !DILocation(line: 22, column: 3, scope: !29)
// CHECK:STDOUT: !41 = !{!21}
// CHECK:STDOUT: !42 = !DISubroutineType(types: !41)
// CHECK:STDOUT: !43 = !DILocation(line: 16, column: 1, scope: !29)
// CHECK:STDOUT: !44 = distinct !DISubprogram(name: "F", linkageName: "_CF.C.Main.e43630e9a6c38c3f", scope: null, file: !18, line: 6, type: !34, spFlags: DISPFlagDefinition, unit: !23)
// CHECK:STDOUT: !47 = !DILocation(line: 7, column: 5, scope: !44)
// CHECK:STDOUT: !48 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.7021b132e5818ff2", scope: null, file: !10, line: 30, type: !37, spFlags: DISPFlagDefinition, unit: !23, retainedNodes: !54)
// CHECK:STDOUT: !49 = !DILocalVariable(arg: 1, scope: !48, type: !35)
// CHECK:STDOUT: !53 = !DILocation(line: 30, column: 1, scope: !48)
// CHECK:STDOUT: !54 = !{!49}
// CHECK:STDOUT: !55 = distinct !DISubprogram(name: "G", linkageName: "_CG.C.Main.e43630e9a6c38c3f", scope: null, file: !18, line: 11, type: !42, spFlags: DISPFlagDefinition, unit: !23)
// CHECK:STDOUT: !59 = !DILocation(line: 12, column: 5, scope: !55)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'require_completion_in_vtable.carbon'
// CHECK:STDOUT: source_filename = "require_completion_in_vtable.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!61}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !60 = !DIFile(filename: "require_completion_in_vtable.carbon", directory: "")
// CHECK:STDOUT: !61 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !60, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
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
// CHECK:STDOUT: define internal void @_C__global_init.Main() #0 !dbg !66 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 @_Cv.Main, ptr align 8 @C.val.loc9_1, i64 8, i1 false), !dbg !67
// CHECK:STDOUT:   ret void, !dbg !68
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!65}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !24 = !{null}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !64 = !DIFile(filename: "use_without_completion_in_vtable.carbon", directory: "")
// CHECK:STDOUT: !65 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !64, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !66 = distinct !DISubprogram(name: "__global_init", linkageName: "_C__global_init.Main", scope: null, file: !64, type: !25, spFlags: DISPFlagDefinition, unit: !65)
// CHECK:STDOUT: !67 = !DILocation(line: 9, column: 1, scope: !66)
// CHECK:STDOUT: !68 = !DILocation(line: 0, scope: !66)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'require_completion_in_virtual_call.carbon'
// CHECK:STDOUT: source_filename = "require_completion_in_virtual_call.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.95528a86e1127e2d:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !71 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_3.1.vptr = getelementptr inbounds nuw { ptr }, ptr %self, i32 0, i32 0, !dbg !73
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.648208d69a8e7988(ptr %.loc7_3.1.vptr), !dbg !73
// CHECK:STDOUT:   ret void, !dbg !73
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.648208d69a8e7988(ptr %t) #0 !dbg !75 {
// CHECK:STDOUT:   ret void, !dbg !80
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!70}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !35 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !36 = !{null, !35}
// CHECK:STDOUT: !37 = !DISubroutineType(types: !36)
// CHECK:STDOUT: !69 = !DIFile(filename: "require_completion_in_virtual_call.carbon", directory: "")
// CHECK:STDOUT: !70 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !69, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !71 = distinct !DISubprogram(name: "Op", linkageName: "_COp.95528a86e1127e2d:core.SubobjectDestroy.Core", scope: null, file: !69, line: 7, type: !37, spFlags: DISPFlagDefinition, unit: !70, retainedNodes: !74)
// CHECK:STDOUT: !72 = !DILocalVariable(arg: 1, scope: !71, type: !35)
// CHECK:STDOUT: !73 = !DILocation(line: 7, column: 3, scope: !71)
// CHECK:STDOUT: !74 = !{!72}
// CHECK:STDOUT: !75 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.648208d69a8e7988", scope: null, file: !10, line: 30, type: !37, spFlags: DISPFlagDefinition, unit: !70, retainedNodes: !81)
// CHECK:STDOUT: !76 = !DILocalVariable(arg: 1, scope: !75, type: !35)
// CHECK:STDOUT: !80 = !DILocation(line: 30, column: 1, scope: !75)
// CHECK:STDOUT: !81 = !{!76}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'use_without_completion_in_virtual_call.carbon'
// CHECK:STDOUT: source_filename = "use_without_completion_in_virtual_call.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @C.val.d1d = internal constant {} zeroinitializer
// CHECK:STDOUT: @"_CC.Main.$vtable.9522ff908e97e267" = external unnamed_addr constant ptr
// CHECK:STDOUT: @C.val.d1d.loc9_11.3 = internal constant {} zeroinitializer
// CHECK:STDOUT: @C.val.99f.anon = internal constant { ptr } { ptr @"_CC.Main.$vtable.9522ff908e97e267" }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCallF.Main() #0 !dbg !84 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_9.2.temp = alloca {}, align 1, !dbg !85
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc9_9.2.temp), !dbg !85
// CHECK:STDOUT:   call void @_CF.Main.9522ff908e97e267(ptr @C.val.d1d.loc9_11.3), !dbg !86
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.e25577acf5e5e7da(ptr @C.val.d1d), !dbg !85
// CHECK:STDOUT:   ret void, !dbg !87
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.95528a86e1127e2d:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !88 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.1.vptr = getelementptr inbounds nuw { ptr }, ptr %self, i32 0, i32 0, !dbg !90
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.34f1b6b4cafb3f13(ptr %.1.vptr), !dbg !90
// CHECK:STDOUT:   ret void, !dbg !90
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.18d6342c4f35ccd3:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !92 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !94
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.9522ff908e97e267(ptr %t) #0 !dbg !96 {
// CHECK:STDOUT:   %1 = alloca { ptr }, align 8, !dbg !99
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %1), !dbg !99
// CHECK:STDOUT:   %vptr = getelementptr inbounds nuw { ptr }, ptr %1, i32 0, i32 0, !dbg !100
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %1, ptr align 8 @C.val.99f.anon, i64 8, i1 false), !dbg !99
// CHECK:STDOUT:   %vtable = load ptr, ptr %1, align 8, !dbg !102
// CHECK:STDOUT:   %2 = call ptr @llvm.load.relative.i32(ptr %vtable, i32 0), !dbg !102
// CHECK:STDOUT:   %3 = call i11 %2(ptr %1), !dbg !102
// CHECK:STDOUT:   %vtable1 = load ptr, ptr %1, align 8, !dbg !105
// CHECK:STDOUT:   %4 = call ptr @llvm.load.relative.i32(ptr %vtable1, i32 4), !dbg !105
// CHECK:STDOUT:   %5 = call i12 %4(ptr %1, ptr %t), !dbg !105
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.2256c270ea7be63c(ptr %1), !dbg !99
// CHECK:STDOUT:   ret void, !dbg !109
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.e25577acf5e5e7da(ptr %t) #0 !dbg !111 {
// CHECK:STDOUT:   ret void, !dbg !116
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.34f1b6b4cafb3f13(ptr %t) #0 !dbg !118 {
// CHECK:STDOUT:   ret void, !dbg !123
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i11 @_CF.C.Main.9522ff908e97e267(ptr %self) #0 !dbg !125 {
// CHECK:STDOUT:   ret i11 11, !dbg !130
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
// CHECK:STDOUT: declare ptr @llvm.load.relative.i32(ptr, i32) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i12 @_CG.C.Main.9522ff908e97e267(ptr %self, ptr %t) #0 !dbg !132 {
// CHECK:STDOUT:   ret i12 12, !dbg !139
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.2256c270ea7be63c(ptr %t) #0 !dbg !141 {
// CHECK:STDOUT:   call void @"_COp.18d6342c4f35ccd3:core.SubobjectDestroy.Core"(ptr %t), !dbg !145
// CHECK:STDOUT:   ret void, !dbg !146
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.load.relative.i32, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!83}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !24 = !{null}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !35 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !36 = !{null, !35}
// CHECK:STDOUT: !37 = !DISubroutineType(types: !36)
// CHECK:STDOUT: !60 = !DIFile(filename: "require_completion_in_vtable.carbon", directory: "")
// CHECK:STDOUT: !62 = !DIBasicType(name: "int", size: 11, encoding: DW_ATE_signed)
// CHECK:STDOUT: !63 = !DIBasicType(name: "int", size: 12, encoding: DW_ATE_signed)
// CHECK:STDOUT: !69 = !DIFile(filename: "require_completion_in_virtual_call.carbon", directory: "")
// CHECK:STDOUT: !82 = !DIFile(filename: "use_without_completion_in_virtual_call.carbon", directory: "")
// CHECK:STDOUT: !83 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !82, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !84 = distinct !DISubprogram(name: "CallF", linkageName: "_CCallF.Main", scope: null, file: !82, line: 8, type: !25, spFlags: DISPFlagDefinition, unit: !83)
// CHECK:STDOUT: !85 = !DILocation(line: 9, column: 8, scope: !84)
// CHECK:STDOUT: !86 = !DILocation(line: 9, column: 3, scope: !84)
// CHECK:STDOUT: !87 = !DILocation(line: 8, column: 1, scope: !84)
// CHECK:STDOUT: !88 = distinct !DISubprogram(name: "Op", linkageName: "_COp.95528a86e1127e2d:core.SubobjectDestroy.Core", scope: null, file: !69, line: 7, type: !37, spFlags: DISPFlagDefinition, unit: !83, retainedNodes: !91)
// CHECK:STDOUT: !89 = !DILocalVariable(arg: 1, scope: !88, type: !35)
// CHECK:STDOUT: !90 = !DILocation(line: 7, column: 3, scope: !88)
// CHECK:STDOUT: !91 = !{!89}
// CHECK:STDOUT: !92 = distinct !DISubprogram(name: "Op", linkageName: "_COp.18d6342c4f35ccd3:core.SubobjectDestroy.Core", scope: null, file: !69, line: 7, type: !37, spFlags: DISPFlagDefinition, unit: !83, retainedNodes: !95)
// CHECK:STDOUT: !93 = !DILocalVariable(arg: 1, scope: !92, type: !35)
// CHECK:STDOUT: !94 = !DILocation(line: 7, column: 3, scope: !92)
// CHECK:STDOUT: !95 = !{!93}
// CHECK:STDOUT: !96 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.9522ff908e97e267", scope: null, file: !69, line: 6, type: !37, spFlags: DISPFlagDefinition, unit: !83, retainedNodes: !110)
// CHECK:STDOUT: !97 = !DILocalVariable(arg: 1, scope: !96, type: !35)
// CHECK:STDOUT: !99 = !DILocation(line: 7, column: 3, scope: !96)
// CHECK:STDOUT: !100 = !DILocation(line: 7, column: 17, scope: !96)
// CHECK:STDOUT: !102 = !DILocation(line: 8, column: 3, scope: !96)
// CHECK:STDOUT: !103 = !{!62, !35}
// CHECK:STDOUT: !104 = !DISubroutineType(types: !103)
// CHECK:STDOUT: !105 = !DILocation(line: 9, column: 3, scope: !96)
// CHECK:STDOUT: !107 = !{!63, !35, !35}
// CHECK:STDOUT: !108 = !DISubroutineType(types: !107)
// CHECK:STDOUT: !109 = !DILocation(line: 6, column: 1, scope: !96)
// CHECK:STDOUT: !110 = !{!97}
// CHECK:STDOUT: !111 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.e25577acf5e5e7da", scope: null, file: !10, line: 30, type: !37, spFlags: DISPFlagDefinition, unit: !83, retainedNodes: !117)
// CHECK:STDOUT: !112 = !DILocalVariable(arg: 1, scope: !111, type: !35)
// CHECK:STDOUT: !116 = !DILocation(line: 30, column: 1, scope: !111)
// CHECK:STDOUT: !117 = !{!112}
// CHECK:STDOUT: !118 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.34f1b6b4cafb3f13", scope: null, file: !10, line: 30, type: !37, spFlags: DISPFlagDefinition, unit: !83, retainedNodes: !124)
// CHECK:STDOUT: !119 = !DILocalVariable(arg: 1, scope: !118, type: !35)
// CHECK:STDOUT: !123 = !DILocation(line: 30, column: 1, scope: !118)
// CHECK:STDOUT: !124 = !{!119}
// CHECK:STDOUT: !125 = distinct !DISubprogram(name: "F", linkageName: "_CF.C.Main.9522ff908e97e267", scope: null, file: !60, line: 6, type: !104, spFlags: DISPFlagDefinition, unit: !83, retainedNodes: !131)
// CHECK:STDOUT: !126 = !DILocalVariable(arg: 1, scope: !125, type: !35)
// CHECK:STDOUT: !130 = !DILocation(line: 6, column: 47, scope: !125)
// CHECK:STDOUT: !131 = !{!126}
// CHECK:STDOUT: !132 = distinct !DISubprogram(name: "G", linkageName: "_CG.C.Main.9522ff908e97e267", scope: null, file: !60, line: 7, type: !108, spFlags: DISPFlagDefinition, unit: !83, retainedNodes: !140)
// CHECK:STDOUT: !133 = !DILocalVariable(arg: 1, scope: !132, type: !35)
// CHECK:STDOUT: !134 = !DILocalVariable(arg: 2, scope: !132, type: !35)
// CHECK:STDOUT: !139 = !DILocation(line: 7, column: 60, scope: !132)
// CHECK:STDOUT: !140 = !{!133, !134}
// CHECK:STDOUT: !141 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.2256c270ea7be63c", scope: null, file: !10, line: 30, type: !37, spFlags: DISPFlagDefinition, unit: !83, retainedNodes: !147)
// CHECK:STDOUT: !142 = !DILocalVariable(arg: 1, scope: !141, type: !35)
// CHECK:STDOUT: !145 = !DILocation(line: 32, column: 5, scope: !141)
// CHECK:STDOUT: !146 = !DILocation(line: 30, column: 1, scope: !141)
// CHECK:STDOUT: !147 = !{!142}
// CHECK:STDOUT:
