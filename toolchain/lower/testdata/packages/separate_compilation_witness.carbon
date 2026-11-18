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
// CHECK:STDOUT: define void @_CDirectInImplFile.OwningPackage(ptr sret({ i32 }) %return, ptr %v1, ptr %v2) #0 !dbg !40 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc5_18.1.temp = alloca { i32 }, align 4, !dbg !48
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc5_18.1.temp), !dbg !48
// CHECK:STDOUT:   call void @"_COp.Vector.OwningPackage:MulWith.55456748a8a2eb70.Core"(ptr %.loc5_18.1.temp, ptr %v2, i32 2), !dbg !48
// CHECK:STDOUT:   call void @"_COp.Vector.OwningPackage:AddWith.f941828044620367.Core"(ptr %return, ptr %v1, ptr %.loc5_18.1.temp), !dbg !47
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.7e1cc1683e7a3817(ptr %.loc5_18.1.temp), !dbg !48
// CHECK:STDOUT:   ret void, !dbg !49
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Vector.OwningPackage:MulWith.55456748a8a2eb70.Core"(ptr sret({ i32 }), ptr, i32)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Vector.OwningPackage:AddWith.f941828044620367.Core"(ptr sret({ i32 }), ptr, ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.991ec6fea7332ad5:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !51 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc5_18.1.x = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !53
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %.loc5_18.1.x), !dbg !53
// CHECK:STDOUT:   ret void, !dbg !53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.d9636b4f2fbd64a7:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !57 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc5_18.1.x = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !59
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %.loc5_18.1.x), !dbg !59
// CHECK:STDOUT:   ret void, !dbg !59
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.7e1cc1683e7a3817(ptr %t) #0 !dbg !62 {
// CHECK:STDOUT:   call void @"_COp.d9636b4f2fbd64a7:core.SubobjectDestroy.Core"(ptr %t), !dbg !66
// CHECK:STDOUT:   ret void, !dbg !67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.f127f30143696d51(ptr %t) #0 !dbg !69 {
// CHECK:STDOUT:   ret void, !dbg !74
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.f127f30143696d51, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!37}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{!7, !7, !7}
// CHECK:STDOUT: !9 = !DISubroutineType(types: !8)
// CHECK:STDOUT: !36 = !DIFile(filename: "owner.impl.carbon", directory: "")
// CHECK:STDOUT: !37 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !36, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !38 = !{null, !7}
// CHECK:STDOUT: !39 = !DISubroutineType(types: !38)
// CHECK:STDOUT: !40 = distinct !DISubprogram(name: "DirectInImplFile", linkageName: "_CDirectInImplFile.OwningPackage", scope: null, file: !36, line: 3, type: !9, spFlags: DISPFlagDefinition, unit: !37, retainedNodes: !50)
// CHECK:STDOUT: !41 = !DILocalVariable(arg: 1, scope: !40, type: !7)
// CHECK:STDOUT: !42 = !DILocalVariable(arg: 2, scope: !40, type: !7)
// CHECK:STDOUT: !47 = !DILocation(line: 5, column: 10, scope: !40)
// CHECK:STDOUT: !48 = !DILocation(line: 5, column: 15, scope: !40)
// CHECK:STDOUT: !49 = !DILocation(line: 5, column: 3, scope: !40)
// CHECK:STDOUT: !50 = !{!41, !42}
// CHECK:STDOUT: !51 = distinct !DISubprogram(name: "Op", linkageName: "_COp.991ec6fea7332ad5:core.SubobjectDestroy.Core", scope: null, file: !36, line: 5, type: !39, spFlags: DISPFlagDefinition, unit: !37, retainedNodes: !56)
// CHECK:STDOUT: !52 = !DILocalVariable(arg: 1, scope: !51, type: !7)
// CHECK:STDOUT: !53 = !DILocation(line: 5, column: 15, scope: !51)
// CHECK:STDOUT: !54 = !{null, !6}
// CHECK:STDOUT: !55 = !DISubroutineType(types: !54)
// CHECK:STDOUT: !56 = !{!52}
// CHECK:STDOUT: !57 = distinct !DISubprogram(name: "Op", linkageName: "_COp.d9636b4f2fbd64a7:core.SubobjectDestroy.Core", scope: null, file: !36, line: 5, type: !39, spFlags: DISPFlagDefinition, unit: !37, retainedNodes: !60)
// CHECK:STDOUT: !58 = !DILocalVariable(arg: 1, scope: !57, type: !7)
// CHECK:STDOUT: !59 = !DILocation(line: 5, column: 15, scope: !57)
// CHECK:STDOUT: !60 = !{!58}
// CHECK:STDOUT: !61 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !62 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.7e1cc1683e7a3817", scope: null, file: !61, line: 36, type: !39, spFlags: DISPFlagDefinition, unit: !37, retainedNodes: !68)
// CHECK:STDOUT: !63 = !DILocalVariable(arg: 1, scope: !62, type: !7)
// CHECK:STDOUT: !66 = !DILocation(line: 38, column: 5, scope: !62)
// CHECK:STDOUT: !67 = !DILocation(line: 36, column: 1, scope: !62)
// CHECK:STDOUT: !68 = !{!63}
// CHECK:STDOUT: !69 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.f127f30143696d51", scope: null, file: !61, line: 36, type: !55, spFlags: DISPFlagDefinition, unit: !37, retainedNodes: !75)
// CHECK:STDOUT: !70 = !DILocalVariable(arg: 1, scope: !69, type: !6)
// CHECK:STDOUT: !74 = !DILocation(line: 36, column: 1, scope: !69)
// CHECK:STDOUT: !75 = !{!70}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'direct_same_package.carbon'
// CHECK:STDOUT: source_filename = "direct_same_package.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CDirect.OwningPackage(ptr sret({ i32 }) %return, ptr %v1, ptr %v2) #0 !dbg !78 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_18.1.temp = alloca { i32 }, align 4, !dbg !86
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_18.1.temp), !dbg !86
// CHECK:STDOUT:   call void @"_COp.Vector.OwningPackage:MulWith.55456748a8a2eb70.Core"(ptr %.loc7_18.1.temp, ptr %v2, i32 2), !dbg !86
// CHECK:STDOUT:   call void @"_COp.Vector.OwningPackage:AddWith.f941828044620367.Core"(ptr %return, ptr %v1, ptr %.loc7_18.1.temp), !dbg !85
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.7e1cc1683e7a3817(ptr %.loc7_18.1.temp), !dbg !86
// CHECK:STDOUT:   ret void, !dbg !87
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Vector.OwningPackage:MulWith.55456748a8a2eb70.Core"(ptr sret({ i32 }), ptr, i32)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Vector.OwningPackage:AddWith.f941828044620367.Core"(ptr sret({ i32 }), ptr, ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.991ec6fea7332ad5:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !89 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_18.1.x = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !91
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %.loc7_18.1.x), !dbg !91
// CHECK:STDOUT:   ret void, !dbg !91
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.d9636b4f2fbd64a7:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !93 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_18.1.x = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !95
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %.loc7_18.1.x), !dbg !95
// CHECK:STDOUT:   ret void, !dbg !95
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.7e1cc1683e7a3817(ptr %t) #0 !dbg !97 {
// CHECK:STDOUT:   call void @"_COp.d9636b4f2fbd64a7:core.SubobjectDestroy.Core"(ptr %t), !dbg !101
// CHECK:STDOUT:   ret void, !dbg !102
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.f127f30143696d51(ptr %t) #0 !dbg !104 {
// CHECK:STDOUT:   ret void, !dbg !109
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.f127f30143696d51, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!77}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{!7, !7, !7}
// CHECK:STDOUT: !9 = !DISubroutineType(types: !8)
// CHECK:STDOUT: !38 = !{null, !7}
// CHECK:STDOUT: !39 = !DISubroutineType(types: !38)
// CHECK:STDOUT: !54 = !{null, !6}
// CHECK:STDOUT: !55 = !DISubroutineType(types: !54)
// CHECK:STDOUT: !61 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !76 = !DIFile(filename: "direct_same_package.carbon", directory: "")
// CHECK:STDOUT: !77 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !76, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !78 = distinct !DISubprogram(name: "Direct", linkageName: "_CDirect.OwningPackage", scope: null, file: !76, line: 5, type: !9, spFlags: DISPFlagDefinition, unit: !77, retainedNodes: !88)
// CHECK:STDOUT: !79 = !DILocalVariable(arg: 1, scope: !78, type: !7)
// CHECK:STDOUT: !80 = !DILocalVariable(arg: 2, scope: !78, type: !7)
// CHECK:STDOUT: !85 = !DILocation(line: 7, column: 10, scope: !78)
// CHECK:STDOUT: !86 = !DILocation(line: 7, column: 15, scope: !78)
// CHECK:STDOUT: !87 = !DILocation(line: 7, column: 3, scope: !78)
// CHECK:STDOUT: !88 = !{!79, !80}
// CHECK:STDOUT: !89 = distinct !DISubprogram(name: "Op", linkageName: "_COp.991ec6fea7332ad5:core.SubobjectDestroy.Core", scope: null, file: !76, line: 7, type: !39, spFlags: DISPFlagDefinition, unit: !77, retainedNodes: !92)
// CHECK:STDOUT: !90 = !DILocalVariable(arg: 1, scope: !89, type: !7)
// CHECK:STDOUT: !91 = !DILocation(line: 7, column: 15, scope: !89)
// CHECK:STDOUT: !92 = !{!90}
// CHECK:STDOUT: !93 = distinct !DISubprogram(name: "Op", linkageName: "_COp.d9636b4f2fbd64a7:core.SubobjectDestroy.Core", scope: null, file: !76, line: 7, type: !39, spFlags: DISPFlagDefinition, unit: !77, retainedNodes: !96)
// CHECK:STDOUT: !94 = !DILocalVariable(arg: 1, scope: !93, type: !7)
// CHECK:STDOUT: !95 = !DILocation(line: 7, column: 15, scope: !93)
// CHECK:STDOUT: !96 = !{!94}
// CHECK:STDOUT: !97 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.7e1cc1683e7a3817", scope: null, file: !61, line: 36, type: !39, spFlags: DISPFlagDefinition, unit: !77, retainedNodes: !103)
// CHECK:STDOUT: !98 = !DILocalVariable(arg: 1, scope: !97, type: !7)
// CHECK:STDOUT: !101 = !DILocation(line: 38, column: 5, scope: !97)
// CHECK:STDOUT: !102 = !DILocation(line: 36, column: 1, scope: !97)
// CHECK:STDOUT: !103 = !{!98}
// CHECK:STDOUT: !104 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.f127f30143696d51", scope: null, file: !61, line: 36, type: !55, spFlags: DISPFlagDefinition, unit: !77, retainedNodes: !110)
// CHECK:STDOUT: !105 = !DILocalVariable(arg: 1, scope: !104, type: !6)
// CHECK:STDOUT: !109 = !DILocation(line: 36, column: 1, scope: !104)
// CHECK:STDOUT: !110 = !{!105}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'generic_same_package.carbon'
// CHECK:STDOUT: source_filename = "generic_same_package.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Vector.OwningPackage:MulWith.55456748a8a2eb70.Core"(ptr sret({ i32 }), ptr, i32)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Vector.OwningPackage:AddWith.f941828044620367.Core"(ptr sret({ i32 }), ptr, ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.991ec6fea7332ad5:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !115 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc6_18.1.x = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !117
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %.loc6_18.1.x), !dbg !117
// CHECK:STDOUT:   ret void, !dbg !117
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.d9636b4f2fbd64a7:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !119 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc6_18.1.x = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !121
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %.loc6_18.1.x), !dbg !121
// CHECK:STDOUT:   ret void, !dbg !121
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CMake.OwningPackage(ptr sret({ i32 }))
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.f127f30143696d51(ptr %t) #0 !dbg !123 {
// CHECK:STDOUT:   ret void, !dbg !128
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.f127f30143696d51, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!112}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !38 = !{null, !7}
// CHECK:STDOUT: !39 = !DISubroutineType(types: !38)
// CHECK:STDOUT: !54 = !{null, !6}
// CHECK:STDOUT: !55 = !DISubroutineType(types: !54)
// CHECK:STDOUT: !61 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !111 = !DIFile(filename: "generic_same_package.carbon", directory: "")
// CHECK:STDOUT: !112 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !111, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !115 = distinct !DISubprogram(name: "Op", linkageName: "_COp.991ec6fea7332ad5:core.SubobjectDestroy.Core", scope: null, file: !111, line: 6, type: !39, spFlags: DISPFlagDefinition, unit: !112, retainedNodes: !118)
// CHECK:STDOUT: !116 = !DILocalVariable(arg: 1, scope: !115, type: !7)
// CHECK:STDOUT: !117 = !DILocation(line: 6, column: 15, scope: !115)
// CHECK:STDOUT: !118 = !{!116}
// CHECK:STDOUT: !119 = distinct !DISubprogram(name: "Op", linkageName: "_COp.d9636b4f2fbd64a7:core.SubobjectDestroy.Core", scope: null, file: !111, line: 6, type: !39, spFlags: DISPFlagDefinition, unit: !112, retainedNodes: !122)
// CHECK:STDOUT: !120 = !DILocalVariable(arg: 1, scope: !119, type: !7)
// CHECK:STDOUT: !121 = !DILocation(line: 6, column: 15, scope: !119)
// CHECK:STDOUT: !122 = !{!120}
// CHECK:STDOUT: !123 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.f127f30143696d51", scope: null, file: !61, line: 36, type: !55, spFlags: DISPFlagDefinition, unit: !112, retainedNodes: !129)
// CHECK:STDOUT: !124 = !DILocalVariable(arg: 1, scope: !123, type: !6)
// CHECK:STDOUT: !128 = !DILocation(line: 36, column: 1, scope: !123)
// CHECK:STDOUT: !129 = !{!124}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'indirect_same_package.carbon'
// CHECK:STDOUT: source_filename = "indirect_same_package.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCallGeneric.OwningPackage() #0 !dbg !134 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_20.1.temp = alloca { i32 }, align 4, !dbg !135
// CHECK:STDOUT:   %.loc7_28.1.temp = alloca { i32 }, align 4, !dbg !136
// CHECK:STDOUT:   %.loc7_29.2.temp = alloca { i32 }, align 4, !dbg !137
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_20.1.temp), !dbg !135
// CHECK:STDOUT:   call void @_CMake.OwningPackage(ptr %.loc7_20.1.temp), !dbg !135
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_28.1.temp), !dbg !136
// CHECK:STDOUT:   call void @_CMake.OwningPackage(ptr %.loc7_28.1.temp), !dbg !136
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_29.2.temp), !dbg !137
// CHECK:STDOUT:   call void @_CGeneric.OwningPackage.e43630e9a6c38c3f(ptr %.loc7_29.2.temp, ptr %.loc7_20.1.temp, ptr %.loc7_28.1.temp), !dbg !137
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.7e1cc1683e7a3817(ptr %.loc7_29.2.temp), !dbg !137
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.7e1cc1683e7a3817(ptr %.loc7_28.1.temp), !dbg !136
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.7e1cc1683e7a3817(ptr %.loc7_20.1.temp), !dbg !135
// CHECK:STDOUT:   ret void, !dbg !138
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CMake.OwningPackage(ptr sret({ i32 }))
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.991ec6fea7332ad5:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !139 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_29.1.x = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !141
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %.loc7_29.1.x), !dbg !141
// CHECK:STDOUT:   ret void, !dbg !141
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.d9636b4f2fbd64a7:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !143 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_29.1.x = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !145
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %.loc7_29.1.x), !dbg !145
// CHECK:STDOUT:   ret void, !dbg !145
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CGeneric.OwningPackage.e43630e9a6c38c3f(ptr sret({ i32 }) %return, ptr %v1, ptr %v2) #0 !dbg !147 {
// CHECK:STDOUT:   %temp = alloca { i32 }, align 4, !dbg !155
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp), !dbg !155
// CHECK:STDOUT:   call void @"_COp.Vector.OwningPackage:MulWith.55456748a8a2eb70.Core"(ptr %temp, ptr %v2, i32 2), !dbg !155
// CHECK:STDOUT:   call void @"_COp.Vector.OwningPackage:AddWith.f941828044620367.Core"(ptr %return, ptr %v1, ptr %temp), !dbg !154
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.7e1cc1683e7a3817(ptr %temp), !dbg !155
// CHECK:STDOUT:   ret void, !dbg !156
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.7e1cc1683e7a3817(ptr %t) #0 !dbg !158 {
// CHECK:STDOUT:   call void @"_COp.d9636b4f2fbd64a7:core.SubobjectDestroy.Core"(ptr %t), !dbg !162
// CHECK:STDOUT:   ret void, !dbg !163
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.f127f30143696d51(ptr %t) #0 !dbg !165 {
// CHECK:STDOUT:   ret void, !dbg !170
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
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.f127f30143696d51, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!131}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{!7, !7, !7}
// CHECK:STDOUT: !9 = !DISubroutineType(types: !8)
// CHECK:STDOUT: !38 = !{null, !7}
// CHECK:STDOUT: !39 = !DISubroutineType(types: !38)
// CHECK:STDOUT: !54 = !{null, !6}
// CHECK:STDOUT: !55 = !DISubroutineType(types: !54)
// CHECK:STDOUT: !61 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !111 = !DIFile(filename: "generic_same_package.carbon", directory: "")
// CHECK:STDOUT: !130 = !DIFile(filename: "indirect_same_package.carbon", directory: "")
// CHECK:STDOUT: !131 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !130, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !132 = !{null}
// CHECK:STDOUT: !133 = !DISubroutineType(types: !132)
// CHECK:STDOUT: !134 = distinct !DISubprogram(name: "CallGeneric", linkageName: "_CCallGeneric.OwningPackage", scope: null, file: !130, line: 5, type: !133, spFlags: DISPFlagDefinition, unit: !131)
// CHECK:STDOUT: !135 = !DILocation(line: 7, column: 15, scope: !134)
// CHECK:STDOUT: !136 = !DILocation(line: 7, column: 23, scope: !134)
// CHECK:STDOUT: !137 = !DILocation(line: 7, column: 3, scope: !134)
// CHECK:STDOUT: !138 = !DILocation(line: 5, column: 1, scope: !134)
// CHECK:STDOUT: !139 = distinct !DISubprogram(name: "Op", linkageName: "_COp.991ec6fea7332ad5:core.SubobjectDestroy.Core", scope: null, file: !130, line: 7, type: !39, spFlags: DISPFlagDefinition, unit: !131, retainedNodes: !142)
// CHECK:STDOUT: !140 = !DILocalVariable(arg: 1, scope: !139, type: !7)
// CHECK:STDOUT: !141 = !DILocation(line: 7, column: 3, scope: !139)
// CHECK:STDOUT: !142 = !{!140}
// CHECK:STDOUT: !143 = distinct !DISubprogram(name: "Op", linkageName: "_COp.d9636b4f2fbd64a7:core.SubobjectDestroy.Core", scope: null, file: !130, line: 7, type: !39, spFlags: DISPFlagDefinition, unit: !131, retainedNodes: !146)
// CHECK:STDOUT: !144 = !DILocalVariable(arg: 1, scope: !143, type: !7)
// CHECK:STDOUT: !145 = !DILocation(line: 7, column: 3, scope: !143)
// CHECK:STDOUT: !146 = !{!144}
// CHECK:STDOUT: !147 = distinct !DISubprogram(name: "Generic", linkageName: "_CGeneric.OwningPackage.e43630e9a6c38c3f", scope: null, file: !111, line: 5, type: !9, spFlags: DISPFlagDefinition, unit: !131, retainedNodes: !157)
// CHECK:STDOUT: !148 = !DILocalVariable(arg: 1, scope: !147, type: !7)
// CHECK:STDOUT: !149 = !DILocalVariable(arg: 2, scope: !147, type: !7)
// CHECK:STDOUT: !154 = !DILocation(line: 6, column: 10, scope: !147)
// CHECK:STDOUT: !155 = !DILocation(line: 6, column: 15, scope: !147)
// CHECK:STDOUT: !156 = !DILocation(line: 6, column: 3, scope: !147)
// CHECK:STDOUT: !157 = !{!148, !149}
// CHECK:STDOUT: !158 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.7e1cc1683e7a3817", scope: null, file: !61, line: 36, type: !39, spFlags: DISPFlagDefinition, unit: !131, retainedNodes: !164)
// CHECK:STDOUT: !159 = !DILocalVariable(arg: 1, scope: !158, type: !7)
// CHECK:STDOUT: !162 = !DILocation(line: 38, column: 5, scope: !158)
// CHECK:STDOUT: !163 = !DILocation(line: 36, column: 1, scope: !158)
// CHECK:STDOUT: !164 = !{!159}
// CHECK:STDOUT: !165 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.f127f30143696d51", scope: null, file: !61, line: 36, type: !55, spFlags: DISPFlagDefinition, unit: !131, retainedNodes: !171)
// CHECK:STDOUT: !166 = !DILocalVariable(arg: 1, scope: !165, type: !6)
// CHECK:STDOUT: !170 = !DILocation(line: 36, column: 1, scope: !165)
// CHECK:STDOUT: !171 = !{!166}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'direct_different_package.carbon'
// CHECK:STDOUT: source_filename = "direct_different_package.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CDirect.OtherPackage(ptr sret({ i32 }) %return, ptr %v1, ptr %v2) #0 !dbg !174 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_18.1.temp = alloca { i32 }, align 4, !dbg !182
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_18.1.temp), !dbg !182
// CHECK:STDOUT:   call void @"_COp.Vector.OwningPackage:MulWith.55456748a8a2eb70.Core"(ptr %.loc7_18.1.temp, ptr %v2, i32 2), !dbg !182
// CHECK:STDOUT:   call void @"_COp.Vector.OwningPackage:AddWith.f941828044620367.Core"(ptr %return, ptr %v1, ptr %.loc7_18.1.temp), !dbg !181
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.7e1cc1683e7a3817(ptr %.loc7_18.1.temp), !dbg !182
// CHECK:STDOUT:   ret void, !dbg !183
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Vector.OwningPackage:MulWith.55456748a8a2eb70.Core"(ptr sret({ i32 }), ptr, i32)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Vector.OwningPackage:AddWith.f941828044620367.Core"(ptr sret({ i32 }), ptr, ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.991ec6fea7332ad5:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !185 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_18.1.x = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !187
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %.loc7_18.1.x), !dbg !187
// CHECK:STDOUT:   ret void, !dbg !187
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.d9636b4f2fbd64a7:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !189 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_18.1.x = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !191
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %.loc7_18.1.x), !dbg !191
// CHECK:STDOUT:   ret void, !dbg !191
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.7e1cc1683e7a3817(ptr %t) #0 !dbg !193 {
// CHECK:STDOUT:   call void @"_COp.d9636b4f2fbd64a7:core.SubobjectDestroy.Core"(ptr %t), !dbg !197
// CHECK:STDOUT:   ret void, !dbg !198
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.f127f30143696d51(ptr %t) #0 !dbg !200 {
// CHECK:STDOUT:   ret void, !dbg !205
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.f127f30143696d51, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!173}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{!7, !7, !7}
// CHECK:STDOUT: !9 = !DISubroutineType(types: !8)
// CHECK:STDOUT: !38 = !{null, !7}
// CHECK:STDOUT: !39 = !DISubroutineType(types: !38)
// CHECK:STDOUT: !54 = !{null, !6}
// CHECK:STDOUT: !55 = !DISubroutineType(types: !54)
// CHECK:STDOUT: !61 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !172 = !DIFile(filename: "direct_different_package.carbon", directory: "")
// CHECK:STDOUT: !173 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !172, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !174 = distinct !DISubprogram(name: "Direct", linkageName: "_CDirect.OtherPackage", scope: null, file: !172, line: 5, type: !9, spFlags: DISPFlagDefinition, unit: !173, retainedNodes: !184)
// CHECK:STDOUT: !175 = !DILocalVariable(arg: 1, scope: !174, type: !7)
// CHECK:STDOUT: !176 = !DILocalVariable(arg: 2, scope: !174, type: !7)
// CHECK:STDOUT: !181 = !DILocation(line: 7, column: 10, scope: !174)
// CHECK:STDOUT: !182 = !DILocation(line: 7, column: 15, scope: !174)
// CHECK:STDOUT: !183 = !DILocation(line: 7, column: 3, scope: !174)
// CHECK:STDOUT: !184 = !{!175, !176}
// CHECK:STDOUT: !185 = distinct !DISubprogram(name: "Op", linkageName: "_COp.991ec6fea7332ad5:core.SubobjectDestroy.Core", scope: null, file: !172, line: 7, type: !39, spFlags: DISPFlagDefinition, unit: !173, retainedNodes: !188)
// CHECK:STDOUT: !186 = !DILocalVariable(arg: 1, scope: !185, type: !7)
// CHECK:STDOUT: !187 = !DILocation(line: 7, column: 15, scope: !185)
// CHECK:STDOUT: !188 = !{!186}
// CHECK:STDOUT: !189 = distinct !DISubprogram(name: "Op", linkageName: "_COp.d9636b4f2fbd64a7:core.SubobjectDestroy.Core", scope: null, file: !172, line: 7, type: !39, spFlags: DISPFlagDefinition, unit: !173, retainedNodes: !192)
// CHECK:STDOUT: !190 = !DILocalVariable(arg: 1, scope: !189, type: !7)
// CHECK:STDOUT: !191 = !DILocation(line: 7, column: 15, scope: !189)
// CHECK:STDOUT: !192 = !{!190}
// CHECK:STDOUT: !193 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.7e1cc1683e7a3817", scope: null, file: !61, line: 36, type: !39, spFlags: DISPFlagDefinition, unit: !173, retainedNodes: !199)
// CHECK:STDOUT: !194 = !DILocalVariable(arg: 1, scope: !193, type: !7)
// CHECK:STDOUT: !197 = !DILocation(line: 38, column: 5, scope: !193)
// CHECK:STDOUT: !198 = !DILocation(line: 36, column: 1, scope: !193)
// CHECK:STDOUT: !199 = !{!194}
// CHECK:STDOUT: !200 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.f127f30143696d51", scope: null, file: !61, line: 36, type: !55, spFlags: DISPFlagDefinition, unit: !173, retainedNodes: !206)
// CHECK:STDOUT: !201 = !DILocalVariable(arg: 1, scope: !200, type: !6)
// CHECK:STDOUT: !205 = !DILocation(line: 36, column: 1, scope: !200)
// CHECK:STDOUT: !206 = !{!201}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'indirect_different_package_via_generic_in_same_package.carbon'
// CHECK:STDOUT: source_filename = "indirect_different_package_via_generic_in_same_package.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCallGenericInOwningPackage.OtherPackage() #0 !dbg !209 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_48.1.temp = alloca { i32 }, align 4, !dbg !210
// CHECK:STDOUT:   %.loc7_70.1.temp = alloca { i32 }, align 4, !dbg !211
// CHECK:STDOUT:   %.loc7_71.2.temp = alloca { i32 }, align 4, !dbg !212
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_48.1.temp), !dbg !210
// CHECK:STDOUT:   call void @_CMake.OwningPackage(ptr %.loc7_48.1.temp), !dbg !210
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_70.1.temp), !dbg !211
// CHECK:STDOUT:   call void @_CMake.OwningPackage(ptr %.loc7_70.1.temp), !dbg !211
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_71.2.temp), !dbg !212
// CHECK:STDOUT:   call void @_CGeneric.OwningPackage.e43630e9a6c38c3f(ptr %.loc7_71.2.temp, ptr %.loc7_48.1.temp, ptr %.loc7_70.1.temp), !dbg !212
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.7e1cc1683e7a3817(ptr %.loc7_71.2.temp), !dbg !212
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.7e1cc1683e7a3817(ptr %.loc7_70.1.temp), !dbg !211
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.7e1cc1683e7a3817(ptr %.loc7_48.1.temp), !dbg !210
// CHECK:STDOUT:   ret void, !dbg !213
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CMake.OwningPackage(ptr sret({ i32 }))
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.991ec6fea7332ad5:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !214 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_71.1.x = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !216
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %.loc7_71.1.x), !dbg !216
// CHECK:STDOUT:   ret void, !dbg !216
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.d9636b4f2fbd64a7:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !218 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_71.1.x = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !220
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %.loc7_71.1.x), !dbg !220
// CHECK:STDOUT:   ret void, !dbg !220
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CGeneric.OwningPackage.e43630e9a6c38c3f(ptr sret({ i32 }) %return, ptr %v1, ptr %v2) #0 !dbg !222 {
// CHECK:STDOUT:   %temp = alloca { i32 }, align 4, !dbg !230
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp), !dbg !230
// CHECK:STDOUT:   call void @"_COp.Vector.OwningPackage:MulWith.55456748a8a2eb70.Core"(ptr %temp, ptr %v2, i32 2), !dbg !230
// CHECK:STDOUT:   call void @"_COp.Vector.OwningPackage:AddWith.f941828044620367.Core"(ptr %return, ptr %v1, ptr %temp), !dbg !229
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.7e1cc1683e7a3817(ptr %temp), !dbg !230
// CHECK:STDOUT:   ret void, !dbg !231
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.7e1cc1683e7a3817(ptr %t) #0 !dbg !233 {
// CHECK:STDOUT:   call void @"_COp.d9636b4f2fbd64a7:core.SubobjectDestroy.Core"(ptr %t), !dbg !237
// CHECK:STDOUT:   ret void, !dbg !238
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.f127f30143696d51(ptr %t) #0 !dbg !240 {
// CHECK:STDOUT:   ret void, !dbg !245
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
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.f127f30143696d51, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!208}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{!7, !7, !7}
// CHECK:STDOUT: !9 = !DISubroutineType(types: !8)
// CHECK:STDOUT: !38 = !{null, !7}
// CHECK:STDOUT: !39 = !DISubroutineType(types: !38)
// CHECK:STDOUT: !54 = !{null, !6}
// CHECK:STDOUT: !55 = !DISubroutineType(types: !54)
// CHECK:STDOUT: !61 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !111 = !DIFile(filename: "generic_same_package.carbon", directory: "")
// CHECK:STDOUT: !132 = !{null}
// CHECK:STDOUT: !133 = !DISubroutineType(types: !132)
// CHECK:STDOUT: !207 = !DIFile(filename: "indirect_different_package_via_generic_in_same_package.carbon", directory: "")
// CHECK:STDOUT: !208 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !207, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !209 = distinct !DISubprogram(name: "CallGenericInOwningPackage", linkageName: "_CCallGenericInOwningPackage.OtherPackage", scope: null, file: !207, line: 5, type: !133, spFlags: DISPFlagDefinition, unit: !208)
// CHECK:STDOUT: !210 = !DILocation(line: 7, column: 29, scope: !209)
// CHECK:STDOUT: !211 = !DILocation(line: 7, column: 51, scope: !209)
// CHECK:STDOUT: !212 = !DILocation(line: 7, column: 3, scope: !209)
// CHECK:STDOUT: !213 = !DILocation(line: 5, column: 1, scope: !209)
// CHECK:STDOUT: !214 = distinct !DISubprogram(name: "Op", linkageName: "_COp.991ec6fea7332ad5:core.SubobjectDestroy.Core", scope: null, file: !207, line: 7, type: !39, spFlags: DISPFlagDefinition, unit: !208, retainedNodes: !217)
// CHECK:STDOUT: !215 = !DILocalVariable(arg: 1, scope: !214, type: !7)
// CHECK:STDOUT: !216 = !DILocation(line: 7, column: 3, scope: !214)
// CHECK:STDOUT: !217 = !{!215}
// CHECK:STDOUT: !218 = distinct !DISubprogram(name: "Op", linkageName: "_COp.d9636b4f2fbd64a7:core.SubobjectDestroy.Core", scope: null, file: !207, line: 7, type: !39, spFlags: DISPFlagDefinition, unit: !208, retainedNodes: !221)
// CHECK:STDOUT: !219 = !DILocalVariable(arg: 1, scope: !218, type: !7)
// CHECK:STDOUT: !220 = !DILocation(line: 7, column: 3, scope: !218)
// CHECK:STDOUT: !221 = !{!219}
// CHECK:STDOUT: !222 = distinct !DISubprogram(name: "Generic", linkageName: "_CGeneric.OwningPackage.e43630e9a6c38c3f", scope: null, file: !111, line: 5, type: !9, spFlags: DISPFlagDefinition, unit: !208, retainedNodes: !232)
// CHECK:STDOUT: !223 = !DILocalVariable(arg: 1, scope: !222, type: !7)
// CHECK:STDOUT: !224 = !DILocalVariable(arg: 2, scope: !222, type: !7)
// CHECK:STDOUT: !229 = !DILocation(line: 6, column: 10, scope: !222)
// CHECK:STDOUT: !230 = !DILocation(line: 6, column: 15, scope: !222)
// CHECK:STDOUT: !231 = !DILocation(line: 6, column: 3, scope: !222)
// CHECK:STDOUT: !232 = !{!223, !224}
// CHECK:STDOUT: !233 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.7e1cc1683e7a3817", scope: null, file: !61, line: 36, type: !39, spFlags: DISPFlagDefinition, unit: !208, retainedNodes: !239)
// CHECK:STDOUT: !234 = !DILocalVariable(arg: 1, scope: !233, type: !7)
// CHECK:STDOUT: !237 = !DILocation(line: 38, column: 5, scope: !233)
// CHECK:STDOUT: !238 = !DILocation(line: 36, column: 1, scope: !233)
// CHECK:STDOUT: !239 = !{!234}
// CHECK:STDOUT: !240 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.f127f30143696d51", scope: null, file: !61, line: 36, type: !55, spFlags: DISPFlagDefinition, unit: !208, retainedNodes: !246)
// CHECK:STDOUT: !241 = !DILocalVariable(arg: 1, scope: !240, type: !6)
// CHECK:STDOUT: !245 = !DILocation(line: 36, column: 1, scope: !240)
// CHECK:STDOUT: !246 = !{!241}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'generic_different_package.carbon'
// CHECK:STDOUT: source_filename = "generic_different_package.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Vector.OwningPackage:MulWith.55456748a8a2eb70.Core"(ptr sret({ i32 }), ptr, i32)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Vector.OwningPackage:AddWith.f941828044620367.Core"(ptr sret({ i32 }), ptr, ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.991ec6fea7332ad5:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !249 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc6_18.1.x = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !251
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %.loc6_18.1.x), !dbg !251
// CHECK:STDOUT:   ret void, !dbg !251
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.d9636b4f2fbd64a7:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !253 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc6_18.1.x = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !255
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %.loc6_18.1.x), !dbg !255
// CHECK:STDOUT:   ret void, !dbg !255
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CMake.OtherPackage(ptr sret({ i32 }))
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.f127f30143696d51(ptr %t) #0 !dbg !257 {
// CHECK:STDOUT:   ret void, !dbg !262
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.f127f30143696d51, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!248}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !38 = !{null, !7}
// CHECK:STDOUT: !39 = !DISubroutineType(types: !38)
// CHECK:STDOUT: !54 = !{null, !6}
// CHECK:STDOUT: !55 = !DISubroutineType(types: !54)
// CHECK:STDOUT: !61 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !247 = !DIFile(filename: "generic_different_package.carbon", directory: "")
// CHECK:STDOUT: !248 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !247, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !249 = distinct !DISubprogram(name: "Op", linkageName: "_COp.991ec6fea7332ad5:core.SubobjectDestroy.Core", scope: null, file: !247, line: 6, type: !39, spFlags: DISPFlagDefinition, unit: !248, retainedNodes: !252)
// CHECK:STDOUT: !250 = !DILocalVariable(arg: 1, scope: !249, type: !7)
// CHECK:STDOUT: !251 = !DILocation(line: 6, column: 15, scope: !249)
// CHECK:STDOUT: !252 = !{!250}
// CHECK:STDOUT: !253 = distinct !DISubprogram(name: "Op", linkageName: "_COp.d9636b4f2fbd64a7:core.SubobjectDestroy.Core", scope: null, file: !247, line: 6, type: !39, spFlags: DISPFlagDefinition, unit: !248, retainedNodes: !256)
// CHECK:STDOUT: !254 = !DILocalVariable(arg: 1, scope: !253, type: !7)
// CHECK:STDOUT: !255 = !DILocation(line: 6, column: 15, scope: !253)
// CHECK:STDOUT: !256 = !{!254}
// CHECK:STDOUT: !257 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.f127f30143696d51", scope: null, file: !61, line: 36, type: !55, spFlags: DISPFlagDefinition, unit: !248, retainedNodes: !263)
// CHECK:STDOUT: !258 = !DILocalVariable(arg: 1, scope: !257, type: !6)
// CHECK:STDOUT: !262 = !DILocation(line: 36, column: 1, scope: !257)
// CHECK:STDOUT: !263 = !{!258}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'indirect_different_package_via_generic_in_different_package.carbon'
// CHECK:STDOUT: source_filename = "indirect_different_package_via_generic_in_different_package.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCallGeneric.OtherPackage() #0 !dbg !266 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_20.1.temp = alloca { i32 }, align 4, !dbg !267
// CHECK:STDOUT:   %.loc7_28.1.temp = alloca { i32 }, align 4, !dbg !268
// CHECK:STDOUT:   %.loc7_29.2.temp = alloca { i32 }, align 4, !dbg !269
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_20.1.temp), !dbg !267
// CHECK:STDOUT:   call void @_CMake.OtherPackage(ptr %.loc7_20.1.temp), !dbg !267
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_28.1.temp), !dbg !268
// CHECK:STDOUT:   call void @_CMake.OtherPackage(ptr %.loc7_28.1.temp), !dbg !268
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_29.2.temp), !dbg !269
// CHECK:STDOUT:   call void @_CGeneric.OtherPackage.e43630e9a6c38c3f(ptr %.loc7_29.2.temp, ptr %.loc7_20.1.temp, ptr %.loc7_28.1.temp), !dbg !269
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.7e1cc1683e7a3817(ptr %.loc7_29.2.temp), !dbg !269
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.7e1cc1683e7a3817(ptr %.loc7_28.1.temp), !dbg !268
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.7e1cc1683e7a3817(ptr %.loc7_20.1.temp), !dbg !267
// CHECK:STDOUT:   ret void, !dbg !270
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CMake.OtherPackage(ptr sret({ i32 }))
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.991ec6fea7332ad5:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !271 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_29.1.x = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !273
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %.loc7_29.1.x), !dbg !273
// CHECK:STDOUT:   ret void, !dbg !273
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.d9636b4f2fbd64a7:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !275 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_29.1.x = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !277
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %.loc7_29.1.x), !dbg !277
// CHECK:STDOUT:   ret void, !dbg !277
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CGeneric.OtherPackage.e43630e9a6c38c3f(ptr sret({ i32 }) %return, ptr %v1, ptr %v2) #0 !dbg !279 {
// CHECK:STDOUT:   %temp = alloca { i32 }, align 4, !dbg !287
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp), !dbg !287
// CHECK:STDOUT:   call void @"_COp.Vector.OwningPackage:MulWith.55456748a8a2eb70.Core"(ptr %temp, ptr %v2, i32 2), !dbg !287
// CHECK:STDOUT:   call void @"_COp.Vector.OwningPackage:AddWith.f941828044620367.Core"(ptr %return, ptr %v1, ptr %temp), !dbg !286
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.7e1cc1683e7a3817(ptr %temp), !dbg !287
// CHECK:STDOUT:   ret void, !dbg !288
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.7e1cc1683e7a3817(ptr %t) #0 !dbg !290 {
// CHECK:STDOUT:   call void @"_COp.d9636b4f2fbd64a7:core.SubobjectDestroy.Core"(ptr %t), !dbg !294
// CHECK:STDOUT:   ret void, !dbg !295
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.f127f30143696d51(ptr %t) #0 !dbg !297 {
// CHECK:STDOUT:   ret void, !dbg !302
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
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.f127f30143696d51, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!265}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{!7, !7, !7}
// CHECK:STDOUT: !9 = !DISubroutineType(types: !8)
// CHECK:STDOUT: !38 = !{null, !7}
// CHECK:STDOUT: !39 = !DISubroutineType(types: !38)
// CHECK:STDOUT: !54 = !{null, !6}
// CHECK:STDOUT: !55 = !DISubroutineType(types: !54)
// CHECK:STDOUT: !61 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !132 = !{null}
// CHECK:STDOUT: !133 = !DISubroutineType(types: !132)
// CHECK:STDOUT: !247 = !DIFile(filename: "generic_different_package.carbon", directory: "")
// CHECK:STDOUT: !264 = !DIFile(filename: "indirect_different_package_via_generic_in_different_package.carbon", directory: "")
// CHECK:STDOUT: !265 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !264, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !266 = distinct !DISubprogram(name: "CallGeneric", linkageName: "_CCallGeneric.OtherPackage", scope: null, file: !264, line: 5, type: !133, spFlags: DISPFlagDefinition, unit: !265)
// CHECK:STDOUT: !267 = !DILocation(line: 7, column: 15, scope: !266)
// CHECK:STDOUT: !268 = !DILocation(line: 7, column: 23, scope: !266)
// CHECK:STDOUT: !269 = !DILocation(line: 7, column: 3, scope: !266)
// CHECK:STDOUT: !270 = !DILocation(line: 5, column: 1, scope: !266)
// CHECK:STDOUT: !271 = distinct !DISubprogram(name: "Op", linkageName: "_COp.991ec6fea7332ad5:core.SubobjectDestroy.Core", scope: null, file: !264, line: 7, type: !39, spFlags: DISPFlagDefinition, unit: !265, retainedNodes: !274)
// CHECK:STDOUT: !272 = !DILocalVariable(arg: 1, scope: !271, type: !7)
// CHECK:STDOUT: !273 = !DILocation(line: 7, column: 3, scope: !271)
// CHECK:STDOUT: !274 = !{!272}
// CHECK:STDOUT: !275 = distinct !DISubprogram(name: "Op", linkageName: "_COp.d9636b4f2fbd64a7:core.SubobjectDestroy.Core", scope: null, file: !264, line: 7, type: !39, spFlags: DISPFlagDefinition, unit: !265, retainedNodes: !278)
// CHECK:STDOUT: !276 = !DILocalVariable(arg: 1, scope: !275, type: !7)
// CHECK:STDOUT: !277 = !DILocation(line: 7, column: 3, scope: !275)
// CHECK:STDOUT: !278 = !{!276}
// CHECK:STDOUT: !279 = distinct !DISubprogram(name: "Generic", linkageName: "_CGeneric.OtherPackage.e43630e9a6c38c3f", scope: null, file: !247, line: 5, type: !9, spFlags: DISPFlagDefinition, unit: !265, retainedNodes: !289)
// CHECK:STDOUT: !280 = !DILocalVariable(arg: 1, scope: !279, type: !7)
// CHECK:STDOUT: !281 = !DILocalVariable(arg: 2, scope: !279, type: !7)
// CHECK:STDOUT: !286 = !DILocation(line: 6, column: 10, scope: !279)
// CHECK:STDOUT: !287 = !DILocation(line: 6, column: 15, scope: !279)
// CHECK:STDOUT: !288 = !DILocation(line: 6, column: 3, scope: !279)
// CHECK:STDOUT: !289 = !{!280, !281}
// CHECK:STDOUT: !290 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.7e1cc1683e7a3817", scope: null, file: !61, line: 36, type: !39, spFlags: DISPFlagDefinition, unit: !265, retainedNodes: !296)
// CHECK:STDOUT: !291 = !DILocalVariable(arg: 1, scope: !290, type: !7)
// CHECK:STDOUT: !294 = !DILocation(line: 38, column: 5, scope: !290)
// CHECK:STDOUT: !295 = !DILocation(line: 36, column: 1, scope: !290)
// CHECK:STDOUT: !296 = !{!291}
// CHECK:STDOUT: !297 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.f127f30143696d51", scope: null, file: !61, line: 36, type: !55, spFlags: DISPFlagDefinition, unit: !265, retainedNodes: !303)
// CHECK:STDOUT: !298 = !DILocalVariable(arg: 1, scope: !297, type: !6)
// CHECK:STDOUT: !302 = !DILocation(line: 36, column: 1, scope: !297)
// CHECK:STDOUT: !303 = !{!298}
// CHECK:STDOUT:
