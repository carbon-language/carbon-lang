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
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.cb8adced19392b99:core.Destroy.Core"(ptr %self) #0 !dbg !25 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !18 = !DIFile(filename: "require_completion_in_call.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !22 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !23 = !{null, !22}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.cb8adced19392b99:core.Destroy.Core", scope: null, file: !18, line: 19, type: !24, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !28)
// CHECK:STDOUT: !26 = !DILocalVariable(arg: 1, scope: !25, type: !22)
// CHECK:STDOUT: !27 = !DILocation(line: 19, column: 4, scope: !25)
// CHECK:STDOUT: !28 = !{!26}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'use_without_completion_in_call.carbon'
// CHECK:STDOUT: source_filename = "use_without_completion_in_call.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @C.val.eeb = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CI.Main() #0 !dbg !33 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CG.Main.e43630e9a6c38c3f(), !dbg !34
// CHECK:STDOUT:   ret void, !dbg !35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.cb8adced19392b99:core.Destroy.Core"(ptr %self) #0 !dbg !36 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.b7a0a8bb33ac84ff:core.Destroy.Core"(ptr %self) #0 !dbg !40 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !42
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.b7a0a8bb33ac84ff:core.Destroy.Core"(ptr %self) #0 !dbg !44 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.b7a0a8bb33ac84ff:core.Destroy.Core"(ptr %self), !dbg !46
// CHECK:STDOUT:   ret void, !dbg !46
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CG.Main.e43630e9a6c38c3f() #0 !dbg !48 {
// CHECK:STDOUT:   %temp = alloca {}, align 1, !dbg !50
// CHECK:STDOUT:   %temp1 = alloca {}, align 1, !dbg !54
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp), !dbg !50
// CHECK:STDOUT:   %1 = call i123 @_CF.C.Main.e43630e9a6c38c3f(), !dbg !51
// CHECK:STDOUT:   call void @"_CSelfDestruct.b7a0a8bb33ac84ff:core.Destroy.Core"(ptr @C.val.eeb), !dbg !50
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp1), !dbg !54
// CHECK:STDOUT:   %2 = call i456 @_CG.C.Main.e43630e9a6c38c3f(), !dbg !56
// CHECK:STDOUT:   call void @"_CSelfDestruct.b7a0a8bb33ac84ff:core.Destroy.Core"(ptr @C.val.eeb), !dbg !54
// CHECK:STDOUT:   ret void, !dbg !59
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i123 @_CF.C.Main.e43630e9a6c38c3f() #0 !dbg !60 {
// CHECK:STDOUT:   ret i123 123, !dbg !63
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i456 @_CG.C.Main.e43630e9a6c38c3f() #0 !dbg !64 {
// CHECK:STDOUT:   ret i456 456, !dbg !68
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!30}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !18 = !DIFile(filename: "require_completion_in_call.carbon", directory: "")
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 123, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !DIBasicType(name: "int", size: 456, encoding: DW_ATE_signed)
// CHECK:STDOUT: !22 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !23 = !{null, !22}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !29 = !DIFile(filename: "use_without_completion_in_call.carbon", directory: "")
// CHECK:STDOUT: !30 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !29, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !31 = !{null}
// CHECK:STDOUT: !32 = !DISubroutineType(types: !31)
// CHECK:STDOUT: !33 = distinct !DISubprogram(name: "I", linkageName: "_CI.Main", scope: null, file: !29, line: 6, type: !32, spFlags: DISPFlagDefinition, unit: !30)
// CHECK:STDOUT: !34 = !DILocation(line: 9, column: 3, scope: !33)
// CHECK:STDOUT: !35 = !DILocation(line: 6, column: 1, scope: !33)
// CHECK:STDOUT: !36 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.cb8adced19392b99:core.Destroy.Core", scope: null, file: !18, line: 19, type: !24, spFlags: DISPFlagDefinition, unit: !30, retainedNodes: !39)
// CHECK:STDOUT: !37 = !DILocalVariable(arg: 1, scope: !36, type: !22)
// CHECK:STDOUT: !38 = !DILocation(line: 19, column: 4, scope: !36)
// CHECK:STDOUT: !39 = !{!37}
// CHECK:STDOUT: !40 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.b7a0a8bb33ac84ff:core.Destroy.Core", scope: null, file: !18, line: 19, type: !24, spFlags: DISPFlagDefinition, unit: !30, retainedNodes: !43)
// CHECK:STDOUT: !41 = !DILocalVariable(arg: 1, scope: !40, type: !22)
// CHECK:STDOUT: !42 = !DILocation(line: 19, column: 4, scope: !40)
// CHECK:STDOUT: !43 = !{!41}
// CHECK:STDOUT: !44 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.b7a0a8bb33ac84ff:core.Destroy.Core", scope: null, file: !18, line: 19, type: !24, spFlags: DISPFlagDefinition, unit: !30, retainedNodes: !47)
// CHECK:STDOUT: !45 = !DILocalVariable(arg: 1, scope: !44, type: !22)
// CHECK:STDOUT: !46 = !DILocation(line: 19, column: 4, scope: !44)
// CHECK:STDOUT: !47 = !{!45}
// CHECK:STDOUT: !48 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.e43630e9a6c38c3f", scope: null, file: !18, line: 16, type: !32, spFlags: DISPFlagDefinition, unit: !30)
// CHECK:STDOUT: !50 = !DILocation(line: 19, column: 4, scope: !48)
// CHECK:STDOUT: !51 = !DILocation(line: 19, column: 3, scope: !48)
// CHECK:STDOUT: !52 = !{!20}
// CHECK:STDOUT: !53 = !DISubroutineType(types: !52)
// CHECK:STDOUT: !54 = !DILocation(line: 22, column: 4, scope: !48)
// CHECK:STDOUT: !56 = !DILocation(line: 22, column: 3, scope: !48)
// CHECK:STDOUT: !57 = !{!21}
// CHECK:STDOUT: !58 = !DISubroutineType(types: !57)
// CHECK:STDOUT: !59 = !DILocation(line: 16, column: 1, scope: !48)
// CHECK:STDOUT: !60 = distinct !DISubprogram(name: "F", linkageName: "_CF.C.Main.e43630e9a6c38c3f", scope: null, file: !18, line: 6, type: !53, spFlags: DISPFlagDefinition, unit: !30)
// CHECK:STDOUT: !63 = !DILocation(line: 7, column: 5, scope: !60)
// CHECK:STDOUT: !64 = distinct !DISubprogram(name: "G", linkageName: "_CG.C.Main.e43630e9a6c38c3f", scope: null, file: !18, line: 11, type: !58, spFlags: DISPFlagDefinition, unit: !30)
// CHECK:STDOUT: !68 = !DILocation(line: 12, column: 5, scope: !64)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'require_completion_in_vtable.carbon'
// CHECK:STDOUT: source_filename = "require_completion_in_vtable.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!70}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !69 = !DIFile(filename: "require_completion_in_vtable.carbon", directory: "")
// CHECK:STDOUT: !70 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !69, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
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
// CHECK:STDOUT: define internal void @_C__global_init.Main() #0 !dbg !75 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 @_Cv.Main, ptr align 8 @C.val.loc9_1, i64 8, i1 false), !dbg !76
// CHECK:STDOUT:   ret void, !dbg !77
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!74}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !31 = !{null}
// CHECK:STDOUT: !32 = !DISubroutineType(types: !31)
// CHECK:STDOUT: !73 = !DIFile(filename: "use_without_completion_in_vtable.carbon", directory: "")
// CHECK:STDOUT: !74 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !73, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !75 = distinct !DISubprogram(name: "__global_init", linkageName: "_C__global_init.Main", scope: null, file: !73, type: !32, spFlags: DISPFlagDefinition, unit: !74)
// CHECK:STDOUT: !76 = !DILocation(line: 9, column: 1, scope: !75)
// CHECK:STDOUT: !77 = !DILocation(line: 0, scope: !75)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'require_completion_in_virtual_call.carbon'
// CHECK:STDOUT: source_filename = "require_completion_in_virtual_call.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.454fecbcedcd86e1:core.Destroy.Core"(ptr %self) #0 !dbg !80 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !82
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.95528a86e1127e2d:core.Destroy.Core"(ptr %self) #0 !dbg !84 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7.vptr = getelementptr inbounds nuw { ptr }, ptr %self, i32 0, i32 0, !dbg !86
// CHECK:STDOUT:   call void @"_CSelfDestruct.454fecbcedcd86e1:core.Destroy.Core"(ptr %.loc7.vptr), !dbg !86
// CHECK:STDOUT:   ret void, !dbg !86
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.95528a86e1127e2d:core.Destroy.Core"(ptr %self) #0 !dbg !88 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.95528a86e1127e2d:core.Destroy.Core"(ptr %self), !dbg !90
// CHECK:STDOUT:   ret void, !dbg !90
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!79}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !22 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !23 = !{null, !22}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !78 = !DIFile(filename: "require_completion_in_virtual_call.carbon", directory: "")
// CHECK:STDOUT: !79 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !78, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !80 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.454fecbcedcd86e1:core.Destroy.Core", scope: null, file: !78, line: 7, type: !24, spFlags: DISPFlagDefinition, unit: !79, retainedNodes: !83)
// CHECK:STDOUT: !81 = !DILocalVariable(arg: 1, scope: !80, type: !22)
// CHECK:STDOUT: !82 = !DILocation(line: 7, column: 3, scope: !80)
// CHECK:STDOUT: !83 = !{!81}
// CHECK:STDOUT: !84 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.95528a86e1127e2d:core.Destroy.Core", scope: null, file: !78, line: 7, type: !24, spFlags: DISPFlagDefinition, unit: !79, retainedNodes: !87)
// CHECK:STDOUT: !85 = !DILocalVariable(arg: 1, scope: !84, type: !22)
// CHECK:STDOUT: !86 = !DILocation(line: 7, column: 3, scope: !84)
// CHECK:STDOUT: !87 = !{!85}
// CHECK:STDOUT: !88 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.95528a86e1127e2d:core.Destroy.Core", scope: null, file: !78, line: 7, type: !24, spFlags: DISPFlagDefinition, unit: !79, retainedNodes: !91)
// CHECK:STDOUT: !89 = !DILocalVariable(arg: 1, scope: !88, type: !22)
// CHECK:STDOUT: !90 = !DILocation(line: 7, column: 3, scope: !88)
// CHECK:STDOUT: !91 = !{!89}
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
// CHECK:STDOUT: define void @_CCallF.Main() #0 !dbg !94 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_9.2.temp = alloca {}, align 1, !dbg !95
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc9_9.2.temp), !dbg !95
// CHECK:STDOUT:   call void @_CF.Main.9522ff908e97e267(ptr @C.val.d1d.loc9_11.3), !dbg !96
// CHECK:STDOUT:   call void @"_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core"(ptr @C.val.d1d), !dbg !95
// CHECK:STDOUT:   ret void, !dbg !97
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.cb8adced19392b99:core.Destroy.Core"(ptr %self) #0 !dbg !98 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !100
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.08d9e38b928ba575:core.Destroy.Core"(ptr %self) #0 !dbg !102 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !104
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core"(ptr %self) #0 !dbg !106 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.08d9e38b928ba575:core.Destroy.Core"(ptr %self), !dbg !108
// CHECK:STDOUT:   ret void, !dbg !108
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.454fecbcedcd86e1:core.Destroy.Core"(ptr %self) #0 !dbg !110 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !112
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.95528a86e1127e2d:core.Destroy.Core"(ptr %self) #0 !dbg !114 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.1.vptr = getelementptr inbounds nuw { ptr }, ptr %self, i32 0, i32 0, !dbg !116
// CHECK:STDOUT:   call void @"_CSelfDestruct.454fecbcedcd86e1:core.Destroy.Core"(ptr %.1.vptr), !dbg !116
// CHECK:STDOUT:   ret void, !dbg !116
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.95528a86e1127e2d:core.Destroy.Core"(ptr %self) #0 !dbg !118 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.95528a86e1127e2d:core.Destroy.Core"(ptr %self), !dbg !120
// CHECK:STDOUT:   ret void, !dbg !120
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.119330419006a03e:core.Destroy.Core"(ptr %self) #0 !dbg !122 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !124
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.119330419006a03e:core.Destroy.Core"(ptr %self) #0 !dbg !126 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.119330419006a03e:core.Destroy.Core"(ptr %self), !dbg !128
// CHECK:STDOUT:   ret void, !dbg !128
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.9522ff908e97e267(ptr %t) #0 !dbg !130 {
// CHECK:STDOUT:   %1 = alloca { ptr }, align 8, !dbg !133
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %1), !dbg !133
// CHECK:STDOUT:   %vptr = getelementptr inbounds nuw { ptr }, ptr %1, i32 0, i32 0, !dbg !134
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %1, ptr align 8 @C.val.9ff.anon, i64 8, i1 false), !dbg !133
// CHECK:STDOUT:   %vtable = load ptr, ptr %1, align 8, !dbg !136
// CHECK:STDOUT:   %2 = call ptr @llvm.load.relative.i32(ptr %vtable, i32 0), !dbg !136
// CHECK:STDOUT:   %3 = call i11 %2(ptr %1), !dbg !136
// CHECK:STDOUT:   %vtable1 = load ptr, ptr %1, align 8, !dbg !139
// CHECK:STDOUT:   %4 = call ptr @llvm.load.relative.i32(ptr %vtable1, i32 4), !dbg !139
// CHECK:STDOUT:   %5 = call i12 %4(ptr %1, ptr %t), !dbg !139
// CHECK:STDOUT:   call void @"_CSelfDestruct.119330419006a03e:core.Destroy.Core"(ptr %1), !dbg !133
// CHECK:STDOUT:   ret void, !dbg !143
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i11 @_CF.C.Main.9522ff908e97e267(ptr %self) #0 !dbg !145 {
// CHECK:STDOUT:   ret i11 11, !dbg !150
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
// CHECK:STDOUT: declare ptr @llvm.load.relative.i32(ptr, i32) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i12 @_CG.C.Main.9522ff908e97e267(ptr %self, ptr %t) #0 !dbg !152 {
// CHECK:STDOUT:   ret i12 12, !dbg !159
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.load.relative.i32, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!93}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !22 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !23 = !{null, !22}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !31 = !{null}
// CHECK:STDOUT: !32 = !DISubroutineType(types: !31)
// CHECK:STDOUT: !69 = !DIFile(filename: "require_completion_in_vtable.carbon", directory: "")
// CHECK:STDOUT: !71 = !DIBasicType(name: "int", size: 11, encoding: DW_ATE_signed)
// CHECK:STDOUT: !72 = !DIBasicType(name: "int", size: 12, encoding: DW_ATE_signed)
// CHECK:STDOUT: !78 = !DIFile(filename: "require_completion_in_virtual_call.carbon", directory: "")
// CHECK:STDOUT: !92 = !DIFile(filename: "use_without_completion_in_virtual_call.carbon", directory: "")
// CHECK:STDOUT: !93 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !92, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !94 = distinct !DISubprogram(name: "CallF", linkageName: "_CCallF.Main", scope: null, file: !92, line: 8, type: !32, spFlags: DISPFlagDefinition, unit: !93)
// CHECK:STDOUT: !95 = !DILocation(line: 9, column: 8, scope: !94)
// CHECK:STDOUT: !96 = !DILocation(line: 9, column: 3, scope: !94)
// CHECK:STDOUT: !97 = !DILocation(line: 8, column: 1, scope: !94)
// CHECK:STDOUT: !98 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.cb8adced19392b99:core.Destroy.Core", scope: null, file: !92, line: 9, type: !24, spFlags: DISPFlagDefinition, unit: !93, retainedNodes: !101)
// CHECK:STDOUT: !99 = !DILocalVariable(arg: 1, scope: !98, type: !22)
// CHECK:STDOUT: !100 = !DILocation(line: 9, column: 8, scope: !98)
// CHECK:STDOUT: !101 = !{!99}
// CHECK:STDOUT: !102 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.08d9e38b928ba575:core.Destroy.Core", scope: null, file: !92, line: 9, type: !24, spFlags: DISPFlagDefinition, unit: !93, retainedNodes: !105)
// CHECK:STDOUT: !103 = !DILocalVariable(arg: 1, scope: !102, type: !22)
// CHECK:STDOUT: !104 = !DILocation(line: 9, column: 8, scope: !102)
// CHECK:STDOUT: !105 = !{!103}
// CHECK:STDOUT: !106 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core", scope: null, file: !92, line: 9, type: !24, spFlags: DISPFlagDefinition, unit: !93, retainedNodes: !109)
// CHECK:STDOUT: !107 = !DILocalVariable(arg: 1, scope: !106, type: !22)
// CHECK:STDOUT: !108 = !DILocation(line: 9, column: 8, scope: !106)
// CHECK:STDOUT: !109 = !{!107}
// CHECK:STDOUT: !110 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.454fecbcedcd86e1:core.Destroy.Core", scope: null, file: !78, line: 7, type: !24, spFlags: DISPFlagDefinition, unit: !93, retainedNodes: !113)
// CHECK:STDOUT: !111 = !DILocalVariable(arg: 1, scope: !110, type: !22)
// CHECK:STDOUT: !112 = !DILocation(line: 7, column: 3, scope: !110)
// CHECK:STDOUT: !113 = !{!111}
// CHECK:STDOUT: !114 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.95528a86e1127e2d:core.Destroy.Core", scope: null, file: !78, line: 7, type: !24, spFlags: DISPFlagDefinition, unit: !93, retainedNodes: !117)
// CHECK:STDOUT: !115 = !DILocalVariable(arg: 1, scope: !114, type: !22)
// CHECK:STDOUT: !116 = !DILocation(line: 7, column: 3, scope: !114)
// CHECK:STDOUT: !117 = !{!115}
// CHECK:STDOUT: !118 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.95528a86e1127e2d:core.Destroy.Core", scope: null, file: !78, line: 7, type: !24, spFlags: DISPFlagDefinition, unit: !93, retainedNodes: !121)
// CHECK:STDOUT: !119 = !DILocalVariable(arg: 1, scope: !118, type: !22)
// CHECK:STDOUT: !120 = !DILocation(line: 7, column: 3, scope: !118)
// CHECK:STDOUT: !121 = !{!119}
// CHECK:STDOUT: !122 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.119330419006a03e:core.Destroy.Core", scope: null, file: !78, line: 7, type: !24, spFlags: DISPFlagDefinition, unit: !93, retainedNodes: !125)
// CHECK:STDOUT: !123 = !DILocalVariable(arg: 1, scope: !122, type: !22)
// CHECK:STDOUT: !124 = !DILocation(line: 7, column: 3, scope: !122)
// CHECK:STDOUT: !125 = !{!123}
// CHECK:STDOUT: !126 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.119330419006a03e:core.Destroy.Core", scope: null, file: !78, line: 7, type: !24, spFlags: DISPFlagDefinition, unit: !93, retainedNodes: !129)
// CHECK:STDOUT: !127 = !DILocalVariable(arg: 1, scope: !126, type: !22)
// CHECK:STDOUT: !128 = !DILocation(line: 7, column: 3, scope: !126)
// CHECK:STDOUT: !129 = !{!127}
// CHECK:STDOUT: !130 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.9522ff908e97e267", scope: null, file: !78, line: 6, type: !24, spFlags: DISPFlagDefinition, unit: !93, retainedNodes: !144)
// CHECK:STDOUT: !131 = !DILocalVariable(arg: 1, scope: !130, type: !22)
// CHECK:STDOUT: !133 = !DILocation(line: 7, column: 3, scope: !130)
// CHECK:STDOUT: !134 = !DILocation(line: 7, column: 17, scope: !130)
// CHECK:STDOUT: !136 = !DILocation(line: 8, column: 3, scope: !130)
// CHECK:STDOUT: !137 = !{!71, !22}
// CHECK:STDOUT: !138 = !DISubroutineType(types: !137)
// CHECK:STDOUT: !139 = !DILocation(line: 9, column: 3, scope: !130)
// CHECK:STDOUT: !141 = !{!72, !22, !22}
// CHECK:STDOUT: !142 = !DISubroutineType(types: !141)
// CHECK:STDOUT: !143 = !DILocation(line: 6, column: 1, scope: !130)
// CHECK:STDOUT: !144 = !{!131}
// CHECK:STDOUT: !145 = distinct !DISubprogram(name: "F", linkageName: "_CF.C.Main.9522ff908e97e267", scope: null, file: !69, line: 6, type: !138, spFlags: DISPFlagDefinition, unit: !93, retainedNodes: !151)
// CHECK:STDOUT: !146 = !DILocalVariable(arg: 1, scope: !145, type: !22)
// CHECK:STDOUT: !150 = !DILocation(line: 6, column: 47, scope: !145)
// CHECK:STDOUT: !151 = !{!146}
// CHECK:STDOUT: !152 = distinct !DISubprogram(name: "G", linkageName: "_CG.C.Main.9522ff908e97e267", scope: null, file: !69, line: 7, type: !142, spFlags: DISPFlagDefinition, unit: !93, retainedNodes: !160)
// CHECK:STDOUT: !153 = !DILocalVariable(arg: 1, scope: !152, type: !22)
// CHECK:STDOUT: !154 = !DILocalVariable(arg: 2, scope: !152, type: !22)
// CHECK:STDOUT: !159 = !DILocation(line: 7, column: 60, scope: !152)
// CHECK:STDOUT: !160 = !{!153, !154}
// CHECK:STDOUT:
