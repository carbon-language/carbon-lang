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
// CHECK:STDOUT: define void @"_COp.Vector.OwningPackage:MulWith.84588f41d61dafba.Core"(ptr sret({ i32 }) %return, ptr %self, i32 %n) #0 !dbg !24 {
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
// CHECK:STDOUT: !24 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Vector.OwningPackage:MulWith.84588f41d61dafba.Core", scope: null, file: !4, line: 14, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !35)
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
// CHECK:STDOUT: define void @_CDirectInImplFile.OwningPackage(ptr sret({ i32 }) %return, ptr %v1, ptr %v2) #0 !dbg !42 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc5_18.1.temp = alloca { i32 }, align 4, !dbg !50
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc5_18.1.temp), !dbg !50
// CHECK:STDOUT:   call void @"_COp.Vector.OwningPackage:MulWith.84588f41d61dafba.Core"(ptr %.loc5_18.1.temp, ptr %v2, i32 2), !dbg !50
// CHECK:STDOUT:   call void @"_COp.Vector.OwningPackage:AddWith.f941828044620367.Core"(ptr %return, ptr %v1, ptr %.loc5_18.1.temp), !dbg !49
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.0c08d107d887b38d(ptr %.loc5_18.1.temp), !dbg !50
// CHECK:STDOUT:   ret void, !dbg !51
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Vector.OwningPackage:MulWith.84588f41d61dafba.Core"(ptr sret({ i32 }), ptr, i32)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Vector.OwningPackage:AddWith.f941828044620367.Core"(ptr sret({ i32 }), ptr, ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !53 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !55
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.a15077f50f3020c3:core.Destroy.Core"(ptr %self) #0 !dbg !57 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !59
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.d9636b4f2fbd64a7:core.Destroy.Core"(ptr %self) #0 !dbg !61 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !63
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.0c08d107d887b38d(ptr %self) #0 !dbg !66 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.d9636b4f2fbd64a7:core.Destroy.Core"(ptr %self), !dbg !70
// CHECK:STDOUT:   ret void, !dbg !71
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
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{!7, !7, !7}
// CHECK:STDOUT: !9 = !DISubroutineType(types: !8)
// CHECK:STDOUT: !36 = !DIFile(filename: "owner.impl.carbon", directory: "")
// CHECK:STDOUT: !37 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !36, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !38 = !{null, !6}
// CHECK:STDOUT: !39 = !DISubroutineType(types: !38)
// CHECK:STDOUT: !40 = !{null, !7}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "DirectInImplFile", linkageName: "_CDirectInImplFile.OwningPackage", scope: null, file: !36, line: 3, type: !9, spFlags: DISPFlagDefinition, unit: !37, retainedNodes: !52)
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 1, scope: !42, type: !7)
// CHECK:STDOUT: !44 = !DILocalVariable(arg: 2, scope: !42, type: !7)
// CHECK:STDOUT: !49 = !DILocation(line: 5, column: 10, scope: !42)
// CHECK:STDOUT: !50 = !DILocation(line: 5, column: 15, scope: !42)
// CHECK:STDOUT: !51 = !DILocation(line: 5, column: 3, scope: !42)
// CHECK:STDOUT: !52 = !{!43, !44}
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core", scope: null, file: !36, line: 5, type: !39, spFlags: DISPFlagDefinition, unit: !37, retainedNodes: !56)
// CHECK:STDOUT: !54 = !DILocalVariable(arg: 1, scope: !53, type: !6)
// CHECK:STDOUT: !55 = !DILocation(line: 5, column: 15, scope: !53)
// CHECK:STDOUT: !56 = !{!54}
// CHECK:STDOUT: !57 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.a15077f50f3020c3:core.Destroy.Core", scope: null, file: !36, line: 5, type: !41, spFlags: DISPFlagDefinition, unit: !37, retainedNodes: !60)
// CHECK:STDOUT: !58 = !DILocalVariable(arg: 1, scope: !57, type: !7)
// CHECK:STDOUT: !59 = !DILocation(line: 5, column: 15, scope: !57)
// CHECK:STDOUT: !60 = !{!58}
// CHECK:STDOUT: !61 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.d9636b4f2fbd64a7:core.Destroy.Core", scope: null, file: !36, line: 5, type: !41, spFlags: DISPFlagDefinition, unit: !37, retainedNodes: !64)
// CHECK:STDOUT: !62 = !DILocalVariable(arg: 1, scope: !61, type: !7)
// CHECK:STDOUT: !63 = !DILocation(line: 5, column: 15, scope: !61)
// CHECK:STDOUT: !64 = !{!62}
// CHECK:STDOUT: !65 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !66 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.0c08d107d887b38d", scope: null, file: !65, line: 26, type: !41, spFlags: DISPFlagDefinition, unit: !37, retainedNodes: !72)
// CHECK:STDOUT: !67 = !DILocalVariable(arg: 1, scope: !66, type: !7)
// CHECK:STDOUT: !70 = !DILocation(line: 28, column: 5, scope: !66)
// CHECK:STDOUT: !71 = !DILocation(line: 26, column: 3, scope: !66)
// CHECK:STDOUT: !72 = !{!67}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'direct_same_package.carbon'
// CHECK:STDOUT: source_filename = "direct_same_package.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CDirect.OwningPackage(ptr sret({ i32 }) %return, ptr %v1, ptr %v2) #0 !dbg !75 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_18.1.temp = alloca { i32 }, align 4, !dbg !83
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_18.1.temp), !dbg !83
// CHECK:STDOUT:   call void @"_COp.Vector.OwningPackage:MulWith.84588f41d61dafba.Core"(ptr %.loc7_18.1.temp, ptr %v2, i32 2), !dbg !83
// CHECK:STDOUT:   call void @"_COp.Vector.OwningPackage:AddWith.f941828044620367.Core"(ptr %return, ptr %v1, ptr %.loc7_18.1.temp), !dbg !82
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.0c08d107d887b38d(ptr %.loc7_18.1.temp), !dbg !83
// CHECK:STDOUT:   ret void, !dbg !84
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Vector.OwningPackage:MulWith.84588f41d61dafba.Core"(ptr sret({ i32 }), ptr, i32)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Vector.OwningPackage:AddWith.f941828044620367.Core"(ptr sret({ i32 }), ptr, ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !86 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !88
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.a15077f50f3020c3:core.Destroy.Core"(ptr %self) #0 !dbg !90 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !92
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.d9636b4f2fbd64a7:core.Destroy.Core"(ptr %self) #0 !dbg !94 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !96
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.0c08d107d887b38d(ptr %self) #0 !dbg !98 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.d9636b4f2fbd64a7:core.Destroy.Core"(ptr %self), !dbg !102
// CHECK:STDOUT:   ret void, !dbg !103
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!74}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{!7, !7, !7}
// CHECK:STDOUT: !9 = !DISubroutineType(types: !8)
// CHECK:STDOUT: !38 = !{null, !6}
// CHECK:STDOUT: !39 = !DISubroutineType(types: !38)
// CHECK:STDOUT: !40 = !{null, !7}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !65 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !73 = !DIFile(filename: "direct_same_package.carbon", directory: "")
// CHECK:STDOUT: !74 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !73, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !75 = distinct !DISubprogram(name: "Direct", linkageName: "_CDirect.OwningPackage", scope: null, file: !73, line: 5, type: !9, spFlags: DISPFlagDefinition, unit: !74, retainedNodes: !85)
// CHECK:STDOUT: !76 = !DILocalVariable(arg: 1, scope: !75, type: !7)
// CHECK:STDOUT: !77 = !DILocalVariable(arg: 2, scope: !75, type: !7)
// CHECK:STDOUT: !82 = !DILocation(line: 7, column: 10, scope: !75)
// CHECK:STDOUT: !83 = !DILocation(line: 7, column: 15, scope: !75)
// CHECK:STDOUT: !84 = !DILocation(line: 7, column: 3, scope: !75)
// CHECK:STDOUT: !85 = !{!76, !77}
// CHECK:STDOUT: !86 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core", scope: null, file: !73, line: 7, type: !39, spFlags: DISPFlagDefinition, unit: !74, retainedNodes: !89)
// CHECK:STDOUT: !87 = !DILocalVariable(arg: 1, scope: !86, type: !6)
// CHECK:STDOUT: !88 = !DILocation(line: 7, column: 15, scope: !86)
// CHECK:STDOUT: !89 = !{!87}
// CHECK:STDOUT: !90 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.a15077f50f3020c3:core.Destroy.Core", scope: null, file: !73, line: 7, type: !41, spFlags: DISPFlagDefinition, unit: !74, retainedNodes: !93)
// CHECK:STDOUT: !91 = !DILocalVariable(arg: 1, scope: !90, type: !7)
// CHECK:STDOUT: !92 = !DILocation(line: 7, column: 15, scope: !90)
// CHECK:STDOUT: !93 = !{!91}
// CHECK:STDOUT: !94 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.d9636b4f2fbd64a7:core.Destroy.Core", scope: null, file: !73, line: 7, type: !41, spFlags: DISPFlagDefinition, unit: !74, retainedNodes: !97)
// CHECK:STDOUT: !95 = !DILocalVariable(arg: 1, scope: !94, type: !7)
// CHECK:STDOUT: !96 = !DILocation(line: 7, column: 15, scope: !94)
// CHECK:STDOUT: !97 = !{!95}
// CHECK:STDOUT: !98 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.0c08d107d887b38d", scope: null, file: !65, line: 26, type: !41, spFlags: DISPFlagDefinition, unit: !74, retainedNodes: !104)
// CHECK:STDOUT: !99 = !DILocalVariable(arg: 1, scope: !98, type: !7)
// CHECK:STDOUT: !102 = !DILocation(line: 28, column: 5, scope: !98)
// CHECK:STDOUT: !103 = !DILocation(line: 26, column: 3, scope: !98)
// CHECK:STDOUT: !104 = !{!99}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'generic_same_package.carbon'
// CHECK:STDOUT: source_filename = "generic_same_package.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Vector.OwningPackage:MulWith.84588f41d61dafba.Core"(ptr sret({ i32 }), ptr, i32)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Vector.OwningPackage:AddWith.f941828044620367.Core"(ptr sret({ i32 }), ptr, ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !109 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !111
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.a15077f50f3020c3:core.Destroy.Core"(ptr %self) #0 !dbg !113 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !115
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.d9636b4f2fbd64a7:core.Destroy.Core"(ptr %self) #0 !dbg !117 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !119
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CMake.OwningPackage(ptr sret({ i32 }))
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!106}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !38 = !{null, !6}
// CHECK:STDOUT: !39 = !DISubroutineType(types: !38)
// CHECK:STDOUT: !40 = !{null, !7}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !105 = !DIFile(filename: "generic_same_package.carbon", directory: "")
// CHECK:STDOUT: !106 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !105, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !109 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core", scope: null, file: !105, line: 6, type: !39, spFlags: DISPFlagDefinition, unit: !106, retainedNodes: !112)
// CHECK:STDOUT: !110 = !DILocalVariable(arg: 1, scope: !109, type: !6)
// CHECK:STDOUT: !111 = !DILocation(line: 6, column: 15, scope: !109)
// CHECK:STDOUT: !112 = !{!110}
// CHECK:STDOUT: !113 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.a15077f50f3020c3:core.Destroy.Core", scope: null, file: !105, line: 6, type: !41, spFlags: DISPFlagDefinition, unit: !106, retainedNodes: !116)
// CHECK:STDOUT: !114 = !DILocalVariable(arg: 1, scope: !113, type: !7)
// CHECK:STDOUT: !115 = !DILocation(line: 6, column: 15, scope: !113)
// CHECK:STDOUT: !116 = !{!114}
// CHECK:STDOUT: !117 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.d9636b4f2fbd64a7:core.Destroy.Core", scope: null, file: !105, line: 6, type: !41, spFlags: DISPFlagDefinition, unit: !106, retainedNodes: !120)
// CHECK:STDOUT: !118 = !DILocalVariable(arg: 1, scope: !117, type: !7)
// CHECK:STDOUT: !119 = !DILocation(line: 6, column: 15, scope: !117)
// CHECK:STDOUT: !120 = !{!118}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'indirect_same_package.carbon'
// CHECK:STDOUT: source_filename = "indirect_same_package.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCallGeneric.OwningPackage() #0 !dbg !125 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_20.1.temp = alloca { i32 }, align 4, !dbg !126
// CHECK:STDOUT:   %.loc7_28.1.temp = alloca { i32 }, align 4, !dbg !127
// CHECK:STDOUT:   %.loc7_29.2.temp = alloca { i32 }, align 4, !dbg !128
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_20.1.temp), !dbg !126
// CHECK:STDOUT:   call void @_CMake.OwningPackage(ptr %.loc7_20.1.temp), !dbg !126
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_28.1.temp), !dbg !127
// CHECK:STDOUT:   call void @_CMake.OwningPackage(ptr %.loc7_28.1.temp), !dbg !127
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_29.2.temp), !dbg !128
// CHECK:STDOUT:   call void @_CGeneric.OwningPackage.e43630e9a6c38c3f(ptr %.loc7_29.2.temp, ptr %.loc7_20.1.temp, ptr %.loc7_28.1.temp), !dbg !128
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.0c08d107d887b38d(ptr %.loc7_29.2.temp), !dbg !128
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.0c08d107d887b38d(ptr %.loc7_28.1.temp), !dbg !127
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.0c08d107d887b38d(ptr %.loc7_20.1.temp), !dbg !126
// CHECK:STDOUT:   ret void, !dbg !129
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CMake.OwningPackage(ptr sret({ i32 }))
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !130 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !132
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.a15077f50f3020c3:core.Destroy.Core"(ptr %self) #0 !dbg !134 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !136
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.d9636b4f2fbd64a7:core.Destroy.Core"(ptr %self) #0 !dbg !138 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !140
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CGeneric.OwningPackage.e43630e9a6c38c3f(ptr sret({ i32 }) %return, ptr %v1, ptr %v2) #0 !dbg !142 {
// CHECK:STDOUT:   %temp = alloca { i32 }, align 4, !dbg !150
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp), !dbg !150
// CHECK:STDOUT:   call void @"_COp.Vector.OwningPackage:MulWith.84588f41d61dafba.Core"(ptr %temp, ptr %v2, i32 2), !dbg !150
// CHECK:STDOUT:   call void @"_COp.Vector.OwningPackage:AddWith.f941828044620367.Core"(ptr %return, ptr %v1, ptr %temp), !dbg !149
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.0c08d107d887b38d(ptr %temp), !dbg !150
// CHECK:STDOUT:   ret void, !dbg !151
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.0c08d107d887b38d(ptr %self) #0 !dbg !153 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.d9636b4f2fbd64a7:core.Destroy.Core"(ptr %self), !dbg !157
// CHECK:STDOUT:   ret void, !dbg !158
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Vector.OwningPackage:MulWith.84588f41d61dafba.Core"(ptr sret({ i32 }), ptr, i32)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Vector.OwningPackage:AddWith.f941828044620367.Core"(ptr sret({ i32 }), ptr, ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CMake.OwningPackage, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 3, 2, 1 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Destroy.Core.0c08d107d887b38d, { 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!122}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{!7, !7, !7}
// CHECK:STDOUT: !9 = !DISubroutineType(types: !8)
// CHECK:STDOUT: !38 = !{null, !6}
// CHECK:STDOUT: !39 = !DISubroutineType(types: !38)
// CHECK:STDOUT: !40 = !{null, !7}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !65 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !105 = !DIFile(filename: "generic_same_package.carbon", directory: "")
// CHECK:STDOUT: !121 = !DIFile(filename: "indirect_same_package.carbon", directory: "")
// CHECK:STDOUT: !122 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !121, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !123 = !{null}
// CHECK:STDOUT: !124 = !DISubroutineType(types: !123)
// CHECK:STDOUT: !125 = distinct !DISubprogram(name: "CallGeneric", linkageName: "_CCallGeneric.OwningPackage", scope: null, file: !121, line: 5, type: !124, spFlags: DISPFlagDefinition, unit: !122)
// CHECK:STDOUT: !126 = !DILocation(line: 7, column: 15, scope: !125)
// CHECK:STDOUT: !127 = !DILocation(line: 7, column: 23, scope: !125)
// CHECK:STDOUT: !128 = !DILocation(line: 7, column: 3, scope: !125)
// CHECK:STDOUT: !129 = !DILocation(line: 5, column: 1, scope: !125)
// CHECK:STDOUT: !130 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core", scope: null, file: !121, line: 7, type: !39, spFlags: DISPFlagDefinition, unit: !122, retainedNodes: !133)
// CHECK:STDOUT: !131 = !DILocalVariable(arg: 1, scope: !130, type: !6)
// CHECK:STDOUT: !132 = !DILocation(line: 7, column: 3, scope: !130)
// CHECK:STDOUT: !133 = !{!131}
// CHECK:STDOUT: !134 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.a15077f50f3020c3:core.Destroy.Core", scope: null, file: !121, line: 7, type: !41, spFlags: DISPFlagDefinition, unit: !122, retainedNodes: !137)
// CHECK:STDOUT: !135 = !DILocalVariable(arg: 1, scope: !134, type: !7)
// CHECK:STDOUT: !136 = !DILocation(line: 7, column: 3, scope: !134)
// CHECK:STDOUT: !137 = !{!135}
// CHECK:STDOUT: !138 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.d9636b4f2fbd64a7:core.Destroy.Core", scope: null, file: !121, line: 7, type: !41, spFlags: DISPFlagDefinition, unit: !122, retainedNodes: !141)
// CHECK:STDOUT: !139 = !DILocalVariable(arg: 1, scope: !138, type: !7)
// CHECK:STDOUT: !140 = !DILocation(line: 7, column: 3, scope: !138)
// CHECK:STDOUT: !141 = !{!139}
// CHECK:STDOUT: !142 = distinct !DISubprogram(name: "Generic", linkageName: "_CGeneric.OwningPackage.e43630e9a6c38c3f", scope: null, file: !105, line: 5, type: !9, spFlags: DISPFlagDefinition, unit: !122, retainedNodes: !152)
// CHECK:STDOUT: !143 = !DILocalVariable(arg: 1, scope: !142, type: !7)
// CHECK:STDOUT: !144 = !DILocalVariable(arg: 2, scope: !142, type: !7)
// CHECK:STDOUT: !149 = !DILocation(line: 6, column: 10, scope: !142)
// CHECK:STDOUT: !150 = !DILocation(line: 6, column: 15, scope: !142)
// CHECK:STDOUT: !151 = !DILocation(line: 6, column: 3, scope: !142)
// CHECK:STDOUT: !152 = !{!143, !144}
// CHECK:STDOUT: !153 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.0c08d107d887b38d", scope: null, file: !65, line: 26, type: !41, spFlags: DISPFlagDefinition, unit: !122, retainedNodes: !159)
// CHECK:STDOUT: !154 = !DILocalVariable(arg: 1, scope: !153, type: !7)
// CHECK:STDOUT: !157 = !DILocation(line: 28, column: 5, scope: !153)
// CHECK:STDOUT: !158 = !DILocation(line: 26, column: 3, scope: !153)
// CHECK:STDOUT: !159 = !{!154}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'direct_different_package.carbon'
// CHECK:STDOUT: source_filename = "direct_different_package.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CDirect.OtherPackage(ptr sret({ i32 }) %return, ptr %v1, ptr %v2) #0 !dbg !162 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_18.1.temp = alloca { i32 }, align 4, !dbg !170
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_18.1.temp), !dbg !170
// CHECK:STDOUT:   call void @"_COp.Vector.OwningPackage:MulWith.84588f41d61dafba.Core"(ptr %.loc7_18.1.temp, ptr %v2, i32 2), !dbg !170
// CHECK:STDOUT:   call void @"_COp.Vector.OwningPackage:AddWith.f941828044620367.Core"(ptr %return, ptr %v1, ptr %.loc7_18.1.temp), !dbg !169
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.0c08d107d887b38d(ptr %.loc7_18.1.temp), !dbg !170
// CHECK:STDOUT:   ret void, !dbg !171
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Vector.OwningPackage:MulWith.84588f41d61dafba.Core"(ptr sret({ i32 }), ptr, i32)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Vector.OwningPackage:AddWith.f941828044620367.Core"(ptr sret({ i32 }), ptr, ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !173 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !175
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.a15077f50f3020c3:core.Destroy.Core"(ptr %self) #0 !dbg !177 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !179
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.d9636b4f2fbd64a7:core.Destroy.Core"(ptr %self) #0 !dbg !181 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !183
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.0c08d107d887b38d(ptr %self) #0 !dbg !185 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.d9636b4f2fbd64a7:core.Destroy.Core"(ptr %self), !dbg !189
// CHECK:STDOUT:   ret void, !dbg !190
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!161}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{!7, !7, !7}
// CHECK:STDOUT: !9 = !DISubroutineType(types: !8)
// CHECK:STDOUT: !38 = !{null, !6}
// CHECK:STDOUT: !39 = !DISubroutineType(types: !38)
// CHECK:STDOUT: !40 = !{null, !7}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !65 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !160 = !DIFile(filename: "direct_different_package.carbon", directory: "")
// CHECK:STDOUT: !161 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !160, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !162 = distinct !DISubprogram(name: "Direct", linkageName: "_CDirect.OtherPackage", scope: null, file: !160, line: 5, type: !9, spFlags: DISPFlagDefinition, unit: !161, retainedNodes: !172)
// CHECK:STDOUT: !163 = !DILocalVariable(arg: 1, scope: !162, type: !7)
// CHECK:STDOUT: !164 = !DILocalVariable(arg: 2, scope: !162, type: !7)
// CHECK:STDOUT: !169 = !DILocation(line: 7, column: 10, scope: !162)
// CHECK:STDOUT: !170 = !DILocation(line: 7, column: 15, scope: !162)
// CHECK:STDOUT: !171 = !DILocation(line: 7, column: 3, scope: !162)
// CHECK:STDOUT: !172 = !{!163, !164}
// CHECK:STDOUT: !173 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core", scope: null, file: !160, line: 7, type: !39, spFlags: DISPFlagDefinition, unit: !161, retainedNodes: !176)
// CHECK:STDOUT: !174 = !DILocalVariable(arg: 1, scope: !173, type: !6)
// CHECK:STDOUT: !175 = !DILocation(line: 7, column: 15, scope: !173)
// CHECK:STDOUT: !176 = !{!174}
// CHECK:STDOUT: !177 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.a15077f50f3020c3:core.Destroy.Core", scope: null, file: !160, line: 7, type: !41, spFlags: DISPFlagDefinition, unit: !161, retainedNodes: !180)
// CHECK:STDOUT: !178 = !DILocalVariable(arg: 1, scope: !177, type: !7)
// CHECK:STDOUT: !179 = !DILocation(line: 7, column: 15, scope: !177)
// CHECK:STDOUT: !180 = !{!178}
// CHECK:STDOUT: !181 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.d9636b4f2fbd64a7:core.Destroy.Core", scope: null, file: !160, line: 7, type: !41, spFlags: DISPFlagDefinition, unit: !161, retainedNodes: !184)
// CHECK:STDOUT: !182 = !DILocalVariable(arg: 1, scope: !181, type: !7)
// CHECK:STDOUT: !183 = !DILocation(line: 7, column: 15, scope: !181)
// CHECK:STDOUT: !184 = !{!182}
// CHECK:STDOUT: !185 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.0c08d107d887b38d", scope: null, file: !65, line: 26, type: !41, spFlags: DISPFlagDefinition, unit: !161, retainedNodes: !191)
// CHECK:STDOUT: !186 = !DILocalVariable(arg: 1, scope: !185, type: !7)
// CHECK:STDOUT: !189 = !DILocation(line: 28, column: 5, scope: !185)
// CHECK:STDOUT: !190 = !DILocation(line: 26, column: 3, scope: !185)
// CHECK:STDOUT: !191 = !{!186}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'indirect_different_package_via_generic_in_same_package.carbon'
// CHECK:STDOUT: source_filename = "indirect_different_package_via_generic_in_same_package.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCallGenericInOwningPackage.OtherPackage() #0 !dbg !194 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_48.1.temp = alloca { i32 }, align 4, !dbg !195
// CHECK:STDOUT:   %.loc7_70.1.temp = alloca { i32 }, align 4, !dbg !196
// CHECK:STDOUT:   %.loc7_71.2.temp = alloca { i32 }, align 4, !dbg !197
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_48.1.temp), !dbg !195
// CHECK:STDOUT:   call void @_CMake.OwningPackage(ptr %.loc7_48.1.temp), !dbg !195
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_70.1.temp), !dbg !196
// CHECK:STDOUT:   call void @_CMake.OwningPackage(ptr %.loc7_70.1.temp), !dbg !196
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_71.2.temp), !dbg !197
// CHECK:STDOUT:   call void @_CGeneric.OwningPackage.e43630e9a6c38c3f(ptr %.loc7_71.2.temp, ptr %.loc7_48.1.temp, ptr %.loc7_70.1.temp), !dbg !197
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.0c08d107d887b38d(ptr %.loc7_71.2.temp), !dbg !197
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.0c08d107d887b38d(ptr %.loc7_70.1.temp), !dbg !196
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.0c08d107d887b38d(ptr %.loc7_48.1.temp), !dbg !195
// CHECK:STDOUT:   ret void, !dbg !198
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CMake.OwningPackage(ptr sret({ i32 }))
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !199 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !201
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.a15077f50f3020c3:core.Destroy.Core"(ptr %self) #0 !dbg !203 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !205
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.d9636b4f2fbd64a7:core.Destroy.Core"(ptr %self) #0 !dbg !207 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !209
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CGeneric.OwningPackage.e43630e9a6c38c3f(ptr sret({ i32 }) %return, ptr %v1, ptr %v2) #0 !dbg !211 {
// CHECK:STDOUT:   %temp = alloca { i32 }, align 4, !dbg !219
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp), !dbg !219
// CHECK:STDOUT:   call void @"_COp.Vector.OwningPackage:MulWith.84588f41d61dafba.Core"(ptr %temp, ptr %v2, i32 2), !dbg !219
// CHECK:STDOUT:   call void @"_COp.Vector.OwningPackage:AddWith.f941828044620367.Core"(ptr %return, ptr %v1, ptr %temp), !dbg !218
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.0c08d107d887b38d(ptr %temp), !dbg !219
// CHECK:STDOUT:   ret void, !dbg !220
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.0c08d107d887b38d(ptr %self) #0 !dbg !222 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.d9636b4f2fbd64a7:core.Destroy.Core"(ptr %self), !dbg !226
// CHECK:STDOUT:   ret void, !dbg !227
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Vector.OwningPackage:MulWith.84588f41d61dafba.Core"(ptr sret({ i32 }), ptr, i32)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Vector.OwningPackage:AddWith.f941828044620367.Core"(ptr sret({ i32 }), ptr, ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CMake.OwningPackage, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 3, 2, 1 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Destroy.Core.0c08d107d887b38d, { 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!193}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{!7, !7, !7}
// CHECK:STDOUT: !9 = !DISubroutineType(types: !8)
// CHECK:STDOUT: !38 = !{null, !6}
// CHECK:STDOUT: !39 = !DISubroutineType(types: !38)
// CHECK:STDOUT: !40 = !{null, !7}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !65 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !105 = !DIFile(filename: "generic_same_package.carbon", directory: "")
// CHECK:STDOUT: !123 = !{null}
// CHECK:STDOUT: !124 = !DISubroutineType(types: !123)
// CHECK:STDOUT: !192 = !DIFile(filename: "indirect_different_package_via_generic_in_same_package.carbon", directory: "")
// CHECK:STDOUT: !193 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !192, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !194 = distinct !DISubprogram(name: "CallGenericInOwningPackage", linkageName: "_CCallGenericInOwningPackage.OtherPackage", scope: null, file: !192, line: 5, type: !124, spFlags: DISPFlagDefinition, unit: !193)
// CHECK:STDOUT: !195 = !DILocation(line: 7, column: 29, scope: !194)
// CHECK:STDOUT: !196 = !DILocation(line: 7, column: 51, scope: !194)
// CHECK:STDOUT: !197 = !DILocation(line: 7, column: 3, scope: !194)
// CHECK:STDOUT: !198 = !DILocation(line: 5, column: 1, scope: !194)
// CHECK:STDOUT: !199 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core", scope: null, file: !192, line: 7, type: !39, spFlags: DISPFlagDefinition, unit: !193, retainedNodes: !202)
// CHECK:STDOUT: !200 = !DILocalVariable(arg: 1, scope: !199, type: !6)
// CHECK:STDOUT: !201 = !DILocation(line: 7, column: 3, scope: !199)
// CHECK:STDOUT: !202 = !{!200}
// CHECK:STDOUT: !203 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.a15077f50f3020c3:core.Destroy.Core", scope: null, file: !192, line: 7, type: !41, spFlags: DISPFlagDefinition, unit: !193, retainedNodes: !206)
// CHECK:STDOUT: !204 = !DILocalVariable(arg: 1, scope: !203, type: !7)
// CHECK:STDOUT: !205 = !DILocation(line: 7, column: 3, scope: !203)
// CHECK:STDOUT: !206 = !{!204}
// CHECK:STDOUT: !207 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.d9636b4f2fbd64a7:core.Destroy.Core", scope: null, file: !192, line: 7, type: !41, spFlags: DISPFlagDefinition, unit: !193, retainedNodes: !210)
// CHECK:STDOUT: !208 = !DILocalVariable(arg: 1, scope: !207, type: !7)
// CHECK:STDOUT: !209 = !DILocation(line: 7, column: 3, scope: !207)
// CHECK:STDOUT: !210 = !{!208}
// CHECK:STDOUT: !211 = distinct !DISubprogram(name: "Generic", linkageName: "_CGeneric.OwningPackage.e43630e9a6c38c3f", scope: null, file: !105, line: 5, type: !9, spFlags: DISPFlagDefinition, unit: !193, retainedNodes: !221)
// CHECK:STDOUT: !212 = !DILocalVariable(arg: 1, scope: !211, type: !7)
// CHECK:STDOUT: !213 = !DILocalVariable(arg: 2, scope: !211, type: !7)
// CHECK:STDOUT: !218 = !DILocation(line: 6, column: 10, scope: !211)
// CHECK:STDOUT: !219 = !DILocation(line: 6, column: 15, scope: !211)
// CHECK:STDOUT: !220 = !DILocation(line: 6, column: 3, scope: !211)
// CHECK:STDOUT: !221 = !{!212, !213}
// CHECK:STDOUT: !222 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.0c08d107d887b38d", scope: null, file: !65, line: 26, type: !41, spFlags: DISPFlagDefinition, unit: !193, retainedNodes: !228)
// CHECK:STDOUT: !223 = !DILocalVariable(arg: 1, scope: !222, type: !7)
// CHECK:STDOUT: !226 = !DILocation(line: 28, column: 5, scope: !222)
// CHECK:STDOUT: !227 = !DILocation(line: 26, column: 3, scope: !222)
// CHECK:STDOUT: !228 = !{!223}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'generic_different_package.carbon'
// CHECK:STDOUT: source_filename = "generic_different_package.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Vector.OwningPackage:MulWith.84588f41d61dafba.Core"(ptr sret({ i32 }), ptr, i32)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Vector.OwningPackage:AddWith.f941828044620367.Core"(ptr sret({ i32 }), ptr, ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !231 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !233
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.a15077f50f3020c3:core.Destroy.Core"(ptr %self) #0 !dbg !235 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !237
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.d9636b4f2fbd64a7:core.Destroy.Core"(ptr %self) #0 !dbg !239 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !241
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CMake.OtherPackage(ptr sret({ i32 }))
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!230}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !38 = !{null, !6}
// CHECK:STDOUT: !39 = !DISubroutineType(types: !38)
// CHECK:STDOUT: !40 = !{null, !7}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !229 = !DIFile(filename: "generic_different_package.carbon", directory: "")
// CHECK:STDOUT: !230 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !229, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !231 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core", scope: null, file: !229, line: 6, type: !39, spFlags: DISPFlagDefinition, unit: !230, retainedNodes: !234)
// CHECK:STDOUT: !232 = !DILocalVariable(arg: 1, scope: !231, type: !6)
// CHECK:STDOUT: !233 = !DILocation(line: 6, column: 15, scope: !231)
// CHECK:STDOUT: !234 = !{!232}
// CHECK:STDOUT: !235 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.a15077f50f3020c3:core.Destroy.Core", scope: null, file: !229, line: 6, type: !41, spFlags: DISPFlagDefinition, unit: !230, retainedNodes: !238)
// CHECK:STDOUT: !236 = !DILocalVariable(arg: 1, scope: !235, type: !7)
// CHECK:STDOUT: !237 = !DILocation(line: 6, column: 15, scope: !235)
// CHECK:STDOUT: !238 = !{!236}
// CHECK:STDOUT: !239 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.d9636b4f2fbd64a7:core.Destroy.Core", scope: null, file: !229, line: 6, type: !41, spFlags: DISPFlagDefinition, unit: !230, retainedNodes: !242)
// CHECK:STDOUT: !240 = !DILocalVariable(arg: 1, scope: !239, type: !7)
// CHECK:STDOUT: !241 = !DILocation(line: 6, column: 15, scope: !239)
// CHECK:STDOUT: !242 = !{!240}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'indirect_different_package_via_generic_in_different_package.carbon'
// CHECK:STDOUT: source_filename = "indirect_different_package_via_generic_in_different_package.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCallGeneric.OtherPackage() #0 !dbg !245 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_20.1.temp = alloca { i32 }, align 4, !dbg !246
// CHECK:STDOUT:   %.loc7_28.1.temp = alloca { i32 }, align 4, !dbg !247
// CHECK:STDOUT:   %.loc7_29.2.temp = alloca { i32 }, align 4, !dbg !248
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_20.1.temp), !dbg !246
// CHECK:STDOUT:   call void @_CMake.OtherPackage(ptr %.loc7_20.1.temp), !dbg !246
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_28.1.temp), !dbg !247
// CHECK:STDOUT:   call void @_CMake.OtherPackage(ptr %.loc7_28.1.temp), !dbg !247
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_29.2.temp), !dbg !248
// CHECK:STDOUT:   call void @_CGeneric.OtherPackage.e43630e9a6c38c3f(ptr %.loc7_29.2.temp, ptr %.loc7_20.1.temp, ptr %.loc7_28.1.temp), !dbg !248
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.0c08d107d887b38d(ptr %.loc7_29.2.temp), !dbg !248
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.0c08d107d887b38d(ptr %.loc7_28.1.temp), !dbg !247
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.0c08d107d887b38d(ptr %.loc7_20.1.temp), !dbg !246
// CHECK:STDOUT:   ret void, !dbg !249
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CMake.OtherPackage(ptr sret({ i32 }))
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !250 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !252
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.a15077f50f3020c3:core.Destroy.Core"(ptr %self) #0 !dbg !254 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !256
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.d9636b4f2fbd64a7:core.Destroy.Core"(ptr %self) #0 !dbg !258 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !260
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CGeneric.OtherPackage.e43630e9a6c38c3f(ptr sret({ i32 }) %return, ptr %v1, ptr %v2) #0 !dbg !262 {
// CHECK:STDOUT:   %temp = alloca { i32 }, align 4, !dbg !270
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp), !dbg !270
// CHECK:STDOUT:   call void @"_COp.Vector.OwningPackage:MulWith.84588f41d61dafba.Core"(ptr %temp, ptr %v2, i32 2), !dbg !270
// CHECK:STDOUT:   call void @"_COp.Vector.OwningPackage:AddWith.f941828044620367.Core"(ptr %return, ptr %v1, ptr %temp), !dbg !269
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.0c08d107d887b38d(ptr %temp), !dbg !270
// CHECK:STDOUT:   ret void, !dbg !271
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.0c08d107d887b38d(ptr %self) #0 !dbg !273 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.d9636b4f2fbd64a7:core.Destroy.Core"(ptr %self), !dbg !277
// CHECK:STDOUT:   ret void, !dbg !278
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Vector.OwningPackage:MulWith.84588f41d61dafba.Core"(ptr sret({ i32 }), ptr, i32)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Vector.OwningPackage:AddWith.f941828044620367.Core"(ptr sret({ i32 }), ptr, ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CMake.OtherPackage, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 3, 2, 1 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Destroy.Core.0c08d107d887b38d, { 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!244}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{!7, !7, !7}
// CHECK:STDOUT: !9 = !DISubroutineType(types: !8)
// CHECK:STDOUT: !38 = !{null, !6}
// CHECK:STDOUT: !39 = !DISubroutineType(types: !38)
// CHECK:STDOUT: !40 = !{null, !7}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !65 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !123 = !{null}
// CHECK:STDOUT: !124 = !DISubroutineType(types: !123)
// CHECK:STDOUT: !229 = !DIFile(filename: "generic_different_package.carbon", directory: "")
// CHECK:STDOUT: !243 = !DIFile(filename: "indirect_different_package_via_generic_in_different_package.carbon", directory: "")
// CHECK:STDOUT: !244 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !243, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !245 = distinct !DISubprogram(name: "CallGeneric", linkageName: "_CCallGeneric.OtherPackage", scope: null, file: !243, line: 5, type: !124, spFlags: DISPFlagDefinition, unit: !244)
// CHECK:STDOUT: !246 = !DILocation(line: 7, column: 15, scope: !245)
// CHECK:STDOUT: !247 = !DILocation(line: 7, column: 23, scope: !245)
// CHECK:STDOUT: !248 = !DILocation(line: 7, column: 3, scope: !245)
// CHECK:STDOUT: !249 = !DILocation(line: 5, column: 1, scope: !245)
// CHECK:STDOUT: !250 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core", scope: null, file: !243, line: 7, type: !39, spFlags: DISPFlagDefinition, unit: !244, retainedNodes: !253)
// CHECK:STDOUT: !251 = !DILocalVariable(arg: 1, scope: !250, type: !6)
// CHECK:STDOUT: !252 = !DILocation(line: 7, column: 3, scope: !250)
// CHECK:STDOUT: !253 = !{!251}
// CHECK:STDOUT: !254 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.a15077f50f3020c3:core.Destroy.Core", scope: null, file: !243, line: 7, type: !41, spFlags: DISPFlagDefinition, unit: !244, retainedNodes: !257)
// CHECK:STDOUT: !255 = !DILocalVariable(arg: 1, scope: !254, type: !7)
// CHECK:STDOUT: !256 = !DILocation(line: 7, column: 3, scope: !254)
// CHECK:STDOUT: !257 = !{!255}
// CHECK:STDOUT: !258 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.d9636b4f2fbd64a7:core.Destroy.Core", scope: null, file: !243, line: 7, type: !41, spFlags: DISPFlagDefinition, unit: !244, retainedNodes: !261)
// CHECK:STDOUT: !259 = !DILocalVariable(arg: 1, scope: !258, type: !7)
// CHECK:STDOUT: !260 = !DILocation(line: 7, column: 3, scope: !258)
// CHECK:STDOUT: !261 = !{!259}
// CHECK:STDOUT: !262 = distinct !DISubprogram(name: "Generic", linkageName: "_CGeneric.OtherPackage.e43630e9a6c38c3f", scope: null, file: !229, line: 5, type: !9, spFlags: DISPFlagDefinition, unit: !244, retainedNodes: !272)
// CHECK:STDOUT: !263 = !DILocalVariable(arg: 1, scope: !262, type: !7)
// CHECK:STDOUT: !264 = !DILocalVariable(arg: 2, scope: !262, type: !7)
// CHECK:STDOUT: !269 = !DILocation(line: 6, column: 10, scope: !262)
// CHECK:STDOUT: !270 = !DILocation(line: 6, column: 15, scope: !262)
// CHECK:STDOUT: !271 = !DILocation(line: 6, column: 3, scope: !262)
// CHECK:STDOUT: !272 = !{!263, !264}
// CHECK:STDOUT: !273 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.0c08d107d887b38d", scope: null, file: !65, line: 26, type: !41, spFlags: DISPFlagDefinition, unit: !244, retainedNodes: !279)
// CHECK:STDOUT: !274 = !DILocalVariable(arg: 1, scope: !273, type: !7)
// CHECK:STDOUT: !277 = !DILocation(line: 28, column: 5, scope: !273)
// CHECK:STDOUT: !278 = !DILocation(line: 26, column: 3, scope: !273)
// CHECK:STDOUT: !279 = !{!274}
// CHECK:STDOUT:
