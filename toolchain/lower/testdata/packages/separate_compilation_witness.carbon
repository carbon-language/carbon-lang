// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/packages/separate_compilation_witness.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/packages/separate_compilation_witness.carbon

// This test ensures that we get the same mangled name for the below `(Vector as
// Add).Op` and `(Vector as MulWith(i32)).Op` functions for various calls from
// different contexts.

// --- owner.carbon
package OwningPackage library "[[@TEST_NAME]]";

class Vector {
  var x: i32;
}

impl Vector as Core.AddWith(Vector) where .Result = Vector {
  fn Op(self, other: Vector) -> Vector {
    return {.x = self.x + other.x};
  }
}

impl Vector as Core.MulWith(i32) where .Result = Vector {
  fn Op(self, n: i32) -> Vector {
    return {.x = self.x * n};
  }
}

// --- owner.impl.carbon
impl package OwningPackage library "[[@TEST_NAME]]";

fn DirectInImplFile(v1: Vector, v2: Vector) -> Vector {
  // Direct call, same package, same library.
  return v1 + v2 * (2 as i32);
}

// --- direct_same_package.carbon
package OwningPackage library "[[@TEST_NAME]]";

import library "owner";

fn Direct(v1: Vector, v2: Vector) -> Vector {
  // Direct call, same package, different library.
  return v1 + v2 * (2 as i32);
}

// --- generic_same_package.carbon
package OwningPackage library "[[@TEST_NAME]]";

import library "owner";

fn Generic(unused generic T: type, v1: Vector, v2: Vector) -> Vector {
  return v1 + v2 * (2 as i32);
}

fn Make() -> Vector;

// --- indirect_same_package.carbon
package OwningPackage library "[[@TEST_NAME]]";

import library "generic_same_package";

fn CallGeneric() {
  // Indirect call, same package, different library.
  Generic((), Make(), Make());
}

// --- direct_different_package.carbon
package OtherPackage library "[[@TEST_NAME]]";

import OwningPackage library "owner";

fn Direct(v1: OwningPackage.Vector, v2: OwningPackage.Vector) -> OwningPackage.Vector {
  // Direct call, different package.
  return v1 + v2 * (2 as i32);
}

// --- indirect_different_package_via_generic_in_same_package.carbon
package OtherPackage library "[[@TEST_NAME]]";

import OwningPackage library "generic_same_package";

fn CallGenericInOwningPackage() {
  // Indirect call, different package.
  OwningPackage.Generic((), OwningPackage.Make(), OwningPackage.Make());
}

// --- generic_different_package.carbon
package OtherPackage library "[[@TEST_NAME]]";

import OwningPackage library "owner";

fn Generic(unused generic T: type, v1: OwningPackage.Vector, v2: OwningPackage.Vector) -> OwningPackage.Vector {
  return v1 + v2 * (2 as i32);
}

fn Make() -> OwningPackage.Vector;

// --- indirect_different_package_via_generic_in_different_package.carbon
package OtherPackage library "[[@TEST_NAME]]";

import library "generic_different_package";

fn CallGeneric() {
  // Indirect call, different package.
  Generic((), Make(), Make());
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'owner.carbon'
// CHECK:STDOUT: source_filename = "owner.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_COp.Vector.OwningPackage:AddWith.f941828044620367.Core"(ptr sret({ i32 }) %return, ptr %self, ptr %other) #0 !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_22.1.x = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !19
// CHECK:STDOUT:   %.loc9_22.2 = load i32, ptr %.loc9_22.1.x, align 4, !dbg !19
// CHECK:STDOUT:   %.loc9_32.1.x = getelementptr inbounds nuw { i32 }, ptr %other, i32 0, i32 0, !dbg !20
// CHECK:STDOUT:   %.loc9_32.2 = load i32, ptr %.loc9_32.1.x, align 4, !dbg !20
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %.loc9_22.2, %.loc9_32.2, !dbg !19
// CHECK:STDOUT:   %.loc9_34.2.x = getelementptr inbounds nuw { i32 }, ptr %return, i32 0, i32 0, !dbg !21
// CHECK:STDOUT:   store i32 %Int.as.AddWith.impl.Op.call, ptr %.loc9_34.2.x, align 4, !dbg !21
// CHECK:STDOUT:   ret void, !dbg !22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_COp.Vector.OwningPackage:MulWith.55456748a8a2eb70.Core"(ptr sret({ i32 }) %return, ptr %self, i32 %n) #0 !dbg !24 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc15_22.1.x = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !31
// CHECK:STDOUT:   %.loc15_22.2 = load i32, ptr %.loc15_22.1.x, align 4, !dbg !31
// CHECK:STDOUT:   %Int.as.MulWith.impl.Op.call = mul i32 %.loc15_22.2, %n, !dbg !31
// CHECK:STDOUT:   %.loc15_28.2.x = getelementptr inbounds nuw { i32 }, ptr %return, i32 0, i32 0, !dbg !33
// CHECK:STDOUT:   store i32 %Int.as.MulWith.impl.Op.call, ptr %.loc15_28.2.x, align 4, !dbg !33
// CHECK:STDOUT:   ret void, !dbg !34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "owner.carbon", directory: "")
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{!7, !7, !7}
// CHECK:STDOUT: !9 = !DISubroutineType(types: !8)
// CHECK:STDOUT: !10 = !{!7, !7, !6}
// CHECK:STDOUT: !11 = !DISubroutineType(types: !10)
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Vector.OwningPackage:AddWith.f941828044620367.Core", scope: null, file: !4, line: 8, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !23)
// CHECK:STDOUT: !13 = !DILocalVariable(arg: 1, scope: !12, type: !7)
// CHECK:STDOUT: !14 = !DILocalVariable(arg: 2, scope: !12, type: !7)
// CHECK:STDOUT: !19 = !DILocation(line: 9, column: 18, scope: !12)
// CHECK:STDOUT: !20 = !DILocation(line: 9, column: 27, scope: !12)
// CHECK:STDOUT: !21 = !DILocation(line: 9, column: 12, scope: !12)
// CHECK:STDOUT: !22 = !DILocation(line: 9, column: 5, scope: !12)
// CHECK:STDOUT: !23 = !{!13, !14}
// CHECK:STDOUT: !24 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Vector.OwningPackage:MulWith.55456748a8a2eb70.Core", scope: null, file: !4, line: 14, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !35)
// CHECK:STDOUT: !25 = !DILocalVariable(arg: 1, scope: !24, type: !7)
// CHECK:STDOUT: !26 = !DILocalVariable(arg: 2, scope: !24, type: !6)
// CHECK:STDOUT: !31 = !DILocation(line: 15, column: 18, scope: !24)
// CHECK:STDOUT: !33 = !DILocation(line: 15, column: 12, scope: !24)
// CHECK:STDOUT: !34 = !DILocation(line: 15, column: 5, scope: !24)
// CHECK:STDOUT: !35 = !{!25, !26}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'owner.impl.carbon'
// CHECK:STDOUT: source_filename = "owner.impl.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CDirectInImplFile.OwningPackage(ptr sret({ i32 }) %return, ptr %v1, ptr %v2) #0 !dbg !38 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc5_18.1.temp = alloca { i32 }, align 4, !dbg !46
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc5_18.1.temp), !dbg !46
// CHECK:STDOUT:   call void @"_COp.Vector.OwningPackage:MulWith.55456748a8a2eb70.Core"(ptr %.loc5_18.1.temp, ptr %v2, i32 2), !dbg !46
// CHECK:STDOUT:   call void @"_COp.Vector.OwningPackage:AddWith.f941828044620367.Core"(ptr %return, ptr %v1, ptr %.loc5_18.1.temp), !dbg !45
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.7e1cc1683e7a3817(ptr %.loc5_18.1.temp), !dbg !46
// CHECK:STDOUT:   ret void, !dbg !49
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Vector.OwningPackage:MulWith.55456748a8a2eb70.Core"(ptr sret({ i32 }), ptr, i32)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Vector.OwningPackage:AddWith.f941828044620367.Core"(ptr sret({ i32 }), ptr, ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.7e1cc1683e7a3817(ptr %t) #0 !dbg !52 {
// CHECK:STDOUT:   ret void, !dbg !57
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!37}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{!7, !7, !7}
// CHECK:STDOUT: !9 = !DISubroutineType(types: !8)
// CHECK:STDOUT: !36 = !DIFile(filename: "owner.impl.carbon", directory: "")
// CHECK:STDOUT: !37 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !36, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !38 = distinct !DISubprogram(name: "DirectInImplFile", linkageName: "_CDirectInImplFile.OwningPackage", scope: null, file: !36, line: 3, type: !9, spFlags: DISPFlagDefinition, unit: !37, retainedNodes: !50)
// CHECK:STDOUT: !39 = !DILocalVariable(arg: 1, scope: !38, type: !7)
// CHECK:STDOUT: !40 = !DILocalVariable(arg: 2, scope: !38, type: !7)
// CHECK:STDOUT: !45 = !DILocation(line: 5, column: 10, scope: !38)
// CHECK:STDOUT: !46 = !DILocation(line: 5, column: 15, scope: !38)
// CHECK:STDOUT: !47 = !{null, !7}
// CHECK:STDOUT: !48 = !DISubroutineType(types: !47)
// CHECK:STDOUT: !49 = !DILocation(line: 5, column: 3, scope: !38)
// CHECK:STDOUT: !50 = !{!39, !40}
// CHECK:STDOUT: !51 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !52 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.7e1cc1683e7a3817", scope: null, file: !51, line: 35, type: !48, spFlags: DISPFlagDefinition, unit: !37, retainedNodes: !58)
// CHECK:STDOUT: !53 = !DILocalVariable(arg: 1, scope: !52, type: !7)
// CHECK:STDOUT: !57 = !DILocation(line: 35, column: 1, scope: !52)
// CHECK:STDOUT: !58 = !{!53}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'direct_same_package.carbon'
// CHECK:STDOUT: source_filename = "direct_same_package.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CDirect.OwningPackage(ptr sret({ i32 }) %return, ptr %v1, ptr %v2) #0 !dbg !61 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_18.1.temp = alloca { i32 }, align 4, !dbg !69
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_18.1.temp), !dbg !69
// CHECK:STDOUT:   call void @"_COp.Vector.OwningPackage:MulWith.55456748a8a2eb70.Core"(ptr %.loc7_18.1.temp, ptr %v2, i32 2), !dbg !69
// CHECK:STDOUT:   call void @"_COp.Vector.OwningPackage:AddWith.f941828044620367.Core"(ptr %return, ptr %v1, ptr %.loc7_18.1.temp), !dbg !68
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.7e1cc1683e7a3817(ptr %.loc7_18.1.temp), !dbg !69
// CHECK:STDOUT:   ret void, !dbg !70
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Vector.OwningPackage:MulWith.55456748a8a2eb70.Core"(ptr sret({ i32 }), ptr, i32)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Vector.OwningPackage:AddWith.f941828044620367.Core"(ptr sret({ i32 }), ptr, ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.7e1cc1683e7a3817(ptr %t) #0 !dbg !72 {
// CHECK:STDOUT:   ret void, !dbg !77
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!60}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{!7, !7, !7}
// CHECK:STDOUT: !9 = !DISubroutineType(types: !8)
// CHECK:STDOUT: !47 = !{null, !7}
// CHECK:STDOUT: !48 = !DISubroutineType(types: !47)
// CHECK:STDOUT: !51 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !59 = !DIFile(filename: "direct_same_package.carbon", directory: "")
// CHECK:STDOUT: !60 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !59, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !61 = distinct !DISubprogram(name: "Direct", linkageName: "_CDirect.OwningPackage", scope: null, file: !59, line: 5, type: !9, spFlags: DISPFlagDefinition, unit: !60, retainedNodes: !71)
// CHECK:STDOUT: !62 = !DILocalVariable(arg: 1, scope: !61, type: !7)
// CHECK:STDOUT: !63 = !DILocalVariable(arg: 2, scope: !61, type: !7)
// CHECK:STDOUT: !68 = !DILocation(line: 7, column: 10, scope: !61)
// CHECK:STDOUT: !69 = !DILocation(line: 7, column: 15, scope: !61)
// CHECK:STDOUT: !70 = !DILocation(line: 7, column: 3, scope: !61)
// CHECK:STDOUT: !71 = !{!62, !63}
// CHECK:STDOUT: !72 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.7e1cc1683e7a3817", scope: null, file: !51, line: 35, type: !48, spFlags: DISPFlagDefinition, unit: !60, retainedNodes: !78)
// CHECK:STDOUT: !73 = !DILocalVariable(arg: 1, scope: !72, type: !7)
// CHECK:STDOUT: !77 = !DILocation(line: 35, column: 1, scope: !72)
// CHECK:STDOUT: !78 = !{!73}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'generic_same_package.carbon'
// CHECK:STDOUT: source_filename = "generic_same_package.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Vector.OwningPackage:MulWith.55456748a8a2eb70.Core"(ptr sret({ i32 }), ptr, i32)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Vector.OwningPackage:AddWith.f941828044620367.Core"(ptr sret({ i32 }), ptr, ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CMake.OwningPackage(ptr sret({ i32 }))
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!80}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !79 = !DIFile(filename: "generic_same_package.carbon", directory: "")
// CHECK:STDOUT: !80 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !79, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'indirect_same_package.carbon'
// CHECK:STDOUT: source_filename = "indirect_same_package.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCallGeneric.OwningPackage() #0 !dbg !87 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_20.1.temp = alloca { i32 }, align 4, !dbg !88
// CHECK:STDOUT:   %.loc7_28.1.temp = alloca { i32 }, align 4, !dbg !89
// CHECK:STDOUT:   %.loc7_29.2.temp = alloca { i32 }, align 4, !dbg !90
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_20.1.temp), !dbg !88
// CHECK:STDOUT:   call void @_CMake.OwningPackage(ptr %.loc7_20.1.temp), !dbg !88
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_28.1.temp), !dbg !89
// CHECK:STDOUT:   call void @_CMake.OwningPackage(ptr %.loc7_28.1.temp), !dbg !89
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_29.2.temp), !dbg !90
// CHECK:STDOUT:   call void @_CGeneric.OwningPackage.e43630e9a6c38c3f(ptr %.loc7_29.2.temp, ptr %.loc7_20.1.temp, ptr %.loc7_28.1.temp), !dbg !90
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.7e1cc1683e7a3817(ptr %.loc7_29.2.temp), !dbg !90
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.7e1cc1683e7a3817(ptr %.loc7_28.1.temp), !dbg !89
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.7e1cc1683e7a3817(ptr %.loc7_20.1.temp), !dbg !88
// CHECK:STDOUT:   ret void, !dbg !91
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CMake.OwningPackage(ptr sret({ i32 }))
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CGeneric.OwningPackage.e43630e9a6c38c3f(ptr sret({ i32 }) %return, ptr %v1, ptr %v2) #0 !dbg !92 {
// CHECK:STDOUT:   %temp = alloca { i32 }, align 4, !dbg !100
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp), !dbg !100
// CHECK:STDOUT:   call void @"_COp.Vector.OwningPackage:MulWith.55456748a8a2eb70.Core"(ptr %temp, ptr %v2, i32 2), !dbg !100
// CHECK:STDOUT:   call void @"_COp.Vector.OwningPackage:AddWith.f941828044620367.Core"(ptr %return, ptr %v1, ptr %temp), !dbg !99
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.7e1cc1683e7a3817(ptr %temp), !dbg !100
// CHECK:STDOUT:   ret void, !dbg !101
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.7e1cc1683e7a3817(ptr %t) #0 !dbg !103 {
// CHECK:STDOUT:   ret void, !dbg !108
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Vector.OwningPackage:MulWith.55456748a8a2eb70.Core"(ptr sret({ i32 }), ptr, i32)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Vector.OwningPackage:AddWith.f941828044620367.Core"(ptr sret({ i32 }), ptr, ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CMake.OwningPackage, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 3, 2, 1 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.7e1cc1683e7a3817, { 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!84}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{!7, !7, !7}
// CHECK:STDOUT: !9 = !DISubroutineType(types: !8)
// CHECK:STDOUT: !47 = !{null, !7}
// CHECK:STDOUT: !48 = !DISubroutineType(types: !47)
// CHECK:STDOUT: !51 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !79 = !DIFile(filename: "generic_same_package.carbon", directory: "")
// CHECK:STDOUT: !83 = !DIFile(filename: "indirect_same_package.carbon", directory: "")
// CHECK:STDOUT: !84 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !83, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !85 = !{null}
// CHECK:STDOUT: !86 = !DISubroutineType(types: !85)
// CHECK:STDOUT: !87 = distinct !DISubprogram(name: "CallGeneric", linkageName: "_CCallGeneric.OwningPackage", scope: null, file: !83, line: 5, type: !86, spFlags: DISPFlagDefinition, unit: !84)
// CHECK:STDOUT: !88 = !DILocation(line: 7, column: 15, scope: !87)
// CHECK:STDOUT: !89 = !DILocation(line: 7, column: 23, scope: !87)
// CHECK:STDOUT: !90 = !DILocation(line: 7, column: 3, scope: !87)
// CHECK:STDOUT: !91 = !DILocation(line: 5, column: 1, scope: !87)
// CHECK:STDOUT: !92 = distinct !DISubprogram(name: "Generic", linkageName: "_CGeneric.OwningPackage.e43630e9a6c38c3f", scope: null, file: !79, line: 5, type: !9, spFlags: DISPFlagDefinition, unit: !84, retainedNodes: !102)
// CHECK:STDOUT: !93 = !DILocalVariable(arg: 1, scope: !92, type: !7)
// CHECK:STDOUT: !94 = !DILocalVariable(arg: 2, scope: !92, type: !7)
// CHECK:STDOUT: !99 = !DILocation(line: 6, column: 10, scope: !92)
// CHECK:STDOUT: !100 = !DILocation(line: 6, column: 15, scope: !92)
// CHECK:STDOUT: !101 = !DILocation(line: 6, column: 3, scope: !92)
// CHECK:STDOUT: !102 = !{!93, !94}
// CHECK:STDOUT: !103 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.7e1cc1683e7a3817", scope: null, file: !51, line: 35, type: !48, spFlags: DISPFlagDefinition, unit: !84, retainedNodes: !109)
// CHECK:STDOUT: !104 = !DILocalVariable(arg: 1, scope: !103, type: !7)
// CHECK:STDOUT: !108 = !DILocation(line: 35, column: 1, scope: !103)
// CHECK:STDOUT: !109 = !{!104}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'direct_different_package.carbon'
// CHECK:STDOUT: source_filename = "direct_different_package.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CDirect.OtherPackage(ptr sret({ i32 }) %return, ptr %v1, ptr %v2) #0 !dbg !112 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_18.1.temp = alloca { i32 }, align 4, !dbg !120
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_18.1.temp), !dbg !120
// CHECK:STDOUT:   call void @"_COp.Vector.OwningPackage:MulWith.55456748a8a2eb70.Core"(ptr %.loc7_18.1.temp, ptr %v2, i32 2), !dbg !120
// CHECK:STDOUT:   call void @"_COp.Vector.OwningPackage:AddWith.f941828044620367.Core"(ptr %return, ptr %v1, ptr %.loc7_18.1.temp), !dbg !119
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.7e1cc1683e7a3817(ptr %.loc7_18.1.temp), !dbg !120
// CHECK:STDOUT:   ret void, !dbg !121
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Vector.OwningPackage:MulWith.55456748a8a2eb70.Core"(ptr sret({ i32 }), ptr, i32)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Vector.OwningPackage:AddWith.f941828044620367.Core"(ptr sret({ i32 }), ptr, ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.7e1cc1683e7a3817(ptr %t) #0 !dbg !123 {
// CHECK:STDOUT:   ret void, !dbg !128
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!111}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{!7, !7, !7}
// CHECK:STDOUT: !9 = !DISubroutineType(types: !8)
// CHECK:STDOUT: !47 = !{null, !7}
// CHECK:STDOUT: !48 = !DISubroutineType(types: !47)
// CHECK:STDOUT: !51 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !110 = !DIFile(filename: "direct_different_package.carbon", directory: "")
// CHECK:STDOUT: !111 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !110, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !112 = distinct !DISubprogram(name: "Direct", linkageName: "_CDirect.OtherPackage", scope: null, file: !110, line: 5, type: !9, spFlags: DISPFlagDefinition, unit: !111, retainedNodes: !122)
// CHECK:STDOUT: !113 = !DILocalVariable(arg: 1, scope: !112, type: !7)
// CHECK:STDOUT: !114 = !DILocalVariable(arg: 2, scope: !112, type: !7)
// CHECK:STDOUT: !119 = !DILocation(line: 7, column: 10, scope: !112)
// CHECK:STDOUT: !120 = !DILocation(line: 7, column: 15, scope: !112)
// CHECK:STDOUT: !121 = !DILocation(line: 7, column: 3, scope: !112)
// CHECK:STDOUT: !122 = !{!113, !114}
// CHECK:STDOUT: !123 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.7e1cc1683e7a3817", scope: null, file: !51, line: 35, type: !48, spFlags: DISPFlagDefinition, unit: !111, retainedNodes: !129)
// CHECK:STDOUT: !124 = !DILocalVariable(arg: 1, scope: !123, type: !7)
// CHECK:STDOUT: !128 = !DILocation(line: 35, column: 1, scope: !123)
// CHECK:STDOUT: !129 = !{!124}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'indirect_different_package_via_generic_in_same_package.carbon'
// CHECK:STDOUT: source_filename = "indirect_different_package_via_generic_in_same_package.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCallGenericInOwningPackage.OtherPackage() #0 !dbg !132 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_48.1.temp = alloca { i32 }, align 4, !dbg !133
// CHECK:STDOUT:   %.loc7_70.1.temp = alloca { i32 }, align 4, !dbg !134
// CHECK:STDOUT:   %.loc7_71.2.temp = alloca { i32 }, align 4, !dbg !135
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_48.1.temp), !dbg !133
// CHECK:STDOUT:   call void @_CMake.OwningPackage(ptr %.loc7_48.1.temp), !dbg !133
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_70.1.temp), !dbg !134
// CHECK:STDOUT:   call void @_CMake.OwningPackage(ptr %.loc7_70.1.temp), !dbg !134
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_71.2.temp), !dbg !135
// CHECK:STDOUT:   call void @_CGeneric.OwningPackage.e43630e9a6c38c3f(ptr %.loc7_71.2.temp, ptr %.loc7_48.1.temp, ptr %.loc7_70.1.temp), !dbg !135
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.7e1cc1683e7a3817(ptr %.loc7_71.2.temp), !dbg !135
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.7e1cc1683e7a3817(ptr %.loc7_70.1.temp), !dbg !134
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.7e1cc1683e7a3817(ptr %.loc7_48.1.temp), !dbg !133
// CHECK:STDOUT:   ret void, !dbg !136
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CMake.OwningPackage(ptr sret({ i32 }))
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CGeneric.OwningPackage.e43630e9a6c38c3f(ptr sret({ i32 }) %return, ptr %v1, ptr %v2) #0 !dbg !137 {
// CHECK:STDOUT:   %temp = alloca { i32 }, align 4, !dbg !145
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp), !dbg !145
// CHECK:STDOUT:   call void @"_COp.Vector.OwningPackage:MulWith.55456748a8a2eb70.Core"(ptr %temp, ptr %v2, i32 2), !dbg !145
// CHECK:STDOUT:   call void @"_COp.Vector.OwningPackage:AddWith.f941828044620367.Core"(ptr %return, ptr %v1, ptr %temp), !dbg !144
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.7e1cc1683e7a3817(ptr %temp), !dbg !145
// CHECK:STDOUT:   ret void, !dbg !146
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.7e1cc1683e7a3817(ptr %t) #0 !dbg !148 {
// CHECK:STDOUT:   ret void, !dbg !153
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Vector.OwningPackage:MulWith.55456748a8a2eb70.Core"(ptr sret({ i32 }), ptr, i32)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Vector.OwningPackage:AddWith.f941828044620367.Core"(ptr sret({ i32 }), ptr, ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CMake.OwningPackage, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 3, 2, 1 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.7e1cc1683e7a3817, { 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!131}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{!7, !7, !7}
// CHECK:STDOUT: !9 = !DISubroutineType(types: !8)
// CHECK:STDOUT: !47 = !{null, !7}
// CHECK:STDOUT: !48 = !DISubroutineType(types: !47)
// CHECK:STDOUT: !51 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !79 = !DIFile(filename: "generic_same_package.carbon", directory: "")
// CHECK:STDOUT: !85 = !{null}
// CHECK:STDOUT: !86 = !DISubroutineType(types: !85)
// CHECK:STDOUT: !130 = !DIFile(filename: "indirect_different_package_via_generic_in_same_package.carbon", directory: "")
// CHECK:STDOUT: !131 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !130, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !132 = distinct !DISubprogram(name: "CallGenericInOwningPackage", linkageName: "_CCallGenericInOwningPackage.OtherPackage", scope: null, file: !130, line: 5, type: !86, spFlags: DISPFlagDefinition, unit: !131)
// CHECK:STDOUT: !133 = !DILocation(line: 7, column: 29, scope: !132)
// CHECK:STDOUT: !134 = !DILocation(line: 7, column: 51, scope: !132)
// CHECK:STDOUT: !135 = !DILocation(line: 7, column: 3, scope: !132)
// CHECK:STDOUT: !136 = !DILocation(line: 5, column: 1, scope: !132)
// CHECK:STDOUT: !137 = distinct !DISubprogram(name: "Generic", linkageName: "_CGeneric.OwningPackage.e43630e9a6c38c3f", scope: null, file: !79, line: 5, type: !9, spFlags: DISPFlagDefinition, unit: !131, retainedNodes: !147)
// CHECK:STDOUT: !138 = !DILocalVariable(arg: 1, scope: !137, type: !7)
// CHECK:STDOUT: !139 = !DILocalVariable(arg: 2, scope: !137, type: !7)
// CHECK:STDOUT: !144 = !DILocation(line: 6, column: 10, scope: !137)
// CHECK:STDOUT: !145 = !DILocation(line: 6, column: 15, scope: !137)
// CHECK:STDOUT: !146 = !DILocation(line: 6, column: 3, scope: !137)
// CHECK:STDOUT: !147 = !{!138, !139}
// CHECK:STDOUT: !148 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.7e1cc1683e7a3817", scope: null, file: !51, line: 35, type: !48, spFlags: DISPFlagDefinition, unit: !131, retainedNodes: !154)
// CHECK:STDOUT: !149 = !DILocalVariable(arg: 1, scope: !148, type: !7)
// CHECK:STDOUT: !153 = !DILocation(line: 35, column: 1, scope: !148)
// CHECK:STDOUT: !154 = !{!149}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'generic_different_package.carbon'
// CHECK:STDOUT: source_filename = "generic_different_package.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Vector.OwningPackage:MulWith.55456748a8a2eb70.Core"(ptr sret({ i32 }), ptr, i32)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Vector.OwningPackage:AddWith.f941828044620367.Core"(ptr sret({ i32 }), ptr, ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CMake.OtherPackage(ptr sret({ i32 }))
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!156}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !155 = !DIFile(filename: "generic_different_package.carbon", directory: "")
// CHECK:STDOUT: !156 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !155, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'indirect_different_package_via_generic_in_different_package.carbon'
// CHECK:STDOUT: source_filename = "indirect_different_package_via_generic_in_different_package.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCallGeneric.OtherPackage() #0 !dbg !159 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_20.1.temp = alloca { i32 }, align 4, !dbg !160
// CHECK:STDOUT:   %.loc7_28.1.temp = alloca { i32 }, align 4, !dbg !161
// CHECK:STDOUT:   %.loc7_29.2.temp = alloca { i32 }, align 4, !dbg !162
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_20.1.temp), !dbg !160
// CHECK:STDOUT:   call void @_CMake.OtherPackage(ptr %.loc7_20.1.temp), !dbg !160
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_28.1.temp), !dbg !161
// CHECK:STDOUT:   call void @_CMake.OtherPackage(ptr %.loc7_28.1.temp), !dbg !161
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_29.2.temp), !dbg !162
// CHECK:STDOUT:   call void @_CGeneric.OtherPackage.e43630e9a6c38c3f(ptr %.loc7_29.2.temp, ptr %.loc7_20.1.temp, ptr %.loc7_28.1.temp), !dbg !162
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.7e1cc1683e7a3817(ptr %.loc7_29.2.temp), !dbg !162
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.7e1cc1683e7a3817(ptr %.loc7_28.1.temp), !dbg !161
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.7e1cc1683e7a3817(ptr %.loc7_20.1.temp), !dbg !160
// CHECK:STDOUT:   ret void, !dbg !163
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CMake.OtherPackage(ptr sret({ i32 }))
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CGeneric.OtherPackage.e43630e9a6c38c3f(ptr sret({ i32 }) %return, ptr %v1, ptr %v2) #0 !dbg !164 {
// CHECK:STDOUT:   %temp = alloca { i32 }, align 4, !dbg !172
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp), !dbg !172
// CHECK:STDOUT:   call void @"_COp.Vector.OwningPackage:MulWith.55456748a8a2eb70.Core"(ptr %temp, ptr %v2, i32 2), !dbg !172
// CHECK:STDOUT:   call void @"_COp.Vector.OwningPackage:AddWith.f941828044620367.Core"(ptr %return, ptr %v1, ptr %temp), !dbg !171
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.7e1cc1683e7a3817(ptr %temp), !dbg !172
// CHECK:STDOUT:   ret void, !dbg !173
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.7e1cc1683e7a3817(ptr %t) #0 !dbg !175 {
// CHECK:STDOUT:   ret void, !dbg !180
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Vector.OwningPackage:MulWith.55456748a8a2eb70.Core"(ptr sret({ i32 }), ptr, i32)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Vector.OwningPackage:AddWith.f941828044620367.Core"(ptr sret({ i32 }), ptr, ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CMake.OtherPackage, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 3, 2, 1 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.7e1cc1683e7a3817, { 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!158}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{!7, !7, !7}
// CHECK:STDOUT: !9 = !DISubroutineType(types: !8)
// CHECK:STDOUT: !47 = !{null, !7}
// CHECK:STDOUT: !48 = !DISubroutineType(types: !47)
// CHECK:STDOUT: !51 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !85 = !{null}
// CHECK:STDOUT: !86 = !DISubroutineType(types: !85)
// CHECK:STDOUT: !155 = !DIFile(filename: "generic_different_package.carbon", directory: "")
// CHECK:STDOUT: !157 = !DIFile(filename: "indirect_different_package_via_generic_in_different_package.carbon", directory: "")
// CHECK:STDOUT: !158 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !157, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !159 = distinct !DISubprogram(name: "CallGeneric", linkageName: "_CCallGeneric.OtherPackage", scope: null, file: !157, line: 5, type: !86, spFlags: DISPFlagDefinition, unit: !158)
// CHECK:STDOUT: !160 = !DILocation(line: 7, column: 15, scope: !159)
// CHECK:STDOUT: !161 = !DILocation(line: 7, column: 23, scope: !159)
// CHECK:STDOUT: !162 = !DILocation(line: 7, column: 3, scope: !159)
// CHECK:STDOUT: !163 = !DILocation(line: 5, column: 1, scope: !159)
// CHECK:STDOUT: !164 = distinct !DISubprogram(name: "Generic", linkageName: "_CGeneric.OtherPackage.e43630e9a6c38c3f", scope: null, file: !155, line: 5, type: !9, spFlags: DISPFlagDefinition, unit: !158, retainedNodes: !174)
// CHECK:STDOUT: !165 = !DILocalVariable(arg: 1, scope: !164, type: !7)
// CHECK:STDOUT: !166 = !DILocalVariable(arg: 2, scope: !164, type: !7)
// CHECK:STDOUT: !171 = !DILocation(line: 6, column: 10, scope: !164)
// CHECK:STDOUT: !172 = !DILocation(line: 6, column: 15, scope: !164)
// CHECK:STDOUT: !173 = !DILocation(line: 6, column: 3, scope: !164)
// CHECK:STDOUT: !174 = !{!165, !166}
// CHECK:STDOUT: !175 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.7e1cc1683e7a3817", scope: null, file: !51, line: 35, type: !48, spFlags: DISPFlagDefinition, unit: !158, retainedNodes: !181)
// CHECK:STDOUT: !176 = !DILocalVariable(arg: 1, scope: !175, type: !7)
// CHECK:STDOUT: !180 = !DILocation(line: 35, column: 1, scope: !175)
// CHECK:STDOUT: !181 = !{!176}
// CHECK:STDOUT:
