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
// CHECK:STDOUT:   call void @"_CSelfDestruct.d9636b4f2fbd64a7:core.Destroy.Core"(ptr %.loc5_18.1.temp), !dbg !50
// CHECK:STDOUT:   ret void, !dbg !51
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Vector.OwningPackage:MulWith.84588f41d61dafba.Core"(ptr sret({ i32 }), ptr, i32)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Vector.OwningPackage:AddWith.f941828044620367.Core"(ptr sret({ i32 }), ptr, ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.6dae29ece9a708c4:core.Destroy.Core"(ptr %self) #0 !dbg !53 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !55
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !57 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !59
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.a15077f50f3020c3:core.Destroy.Core"(ptr %self) #0 !dbg !61 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !63
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.a15077f50f3020c3:core.Destroy.Core"(ptr %self) #0 !dbg !65 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.a15077f50f3020c3:core.Destroy.Core"(ptr %self), !dbg !67
// CHECK:STDOUT:   ret void, !dbg !67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.d9636b4f2fbd64a7:core.Destroy.Core"(ptr %self) #0 !dbg !69 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !71
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.d9636b4f2fbd64a7:core.Destroy.Core"(ptr %self) #0 !dbg !73 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.d9636b4f2fbd64a7:core.Destroy.Core"(ptr %self), !dbg !75
// CHECK:STDOUT:   ret void, !dbg !75
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
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
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.6dae29ece9a708c4:core.Destroy.Core", scope: null, file: !36, line: 5, type: !39, spFlags: DISPFlagDefinition, unit: !37, retainedNodes: !56)
// CHECK:STDOUT: !54 = !DILocalVariable(arg: 1, scope: !53, type: !6)
// CHECK:STDOUT: !55 = !DILocation(line: 5, column: 15, scope: !53)
// CHECK:STDOUT: !56 = !{!54}
// CHECK:STDOUT: !57 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f58524831c37807e:core.Destroy.Core", scope: null, file: !36, line: 5, type: !39, spFlags: DISPFlagDefinition, unit: !37, retainedNodes: !60)
// CHECK:STDOUT: !58 = !DILocalVariable(arg: 1, scope: !57, type: !6)
// CHECK:STDOUT: !59 = !DILocation(line: 5, column: 15, scope: !57)
// CHECK:STDOUT: !60 = !{!58}
// CHECK:STDOUT: !61 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.a15077f50f3020c3:core.Destroy.Core", scope: null, file: !36, line: 5, type: !41, spFlags: DISPFlagDefinition, unit: !37, retainedNodes: !64)
// CHECK:STDOUT: !62 = !DILocalVariable(arg: 1, scope: !61, type: !7)
// CHECK:STDOUT: !63 = !DILocation(line: 5, column: 15, scope: !61)
// CHECK:STDOUT: !64 = !{!62}
// CHECK:STDOUT: !65 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.a15077f50f3020c3:core.Destroy.Core", scope: null, file: !36, line: 5, type: !41, spFlags: DISPFlagDefinition, unit: !37, retainedNodes: !68)
// CHECK:STDOUT: !66 = !DILocalVariable(arg: 1, scope: !65, type: !7)
// CHECK:STDOUT: !67 = !DILocation(line: 5, column: 15, scope: !65)
// CHECK:STDOUT: !68 = !{!66}
// CHECK:STDOUT: !69 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.d9636b4f2fbd64a7:core.Destroy.Core", scope: null, file: !36, line: 5, type: !41, spFlags: DISPFlagDefinition, unit: !37, retainedNodes: !72)
// CHECK:STDOUT: !70 = !DILocalVariable(arg: 1, scope: !69, type: !7)
// CHECK:STDOUT: !71 = !DILocation(line: 5, column: 15, scope: !69)
// CHECK:STDOUT: !72 = !{!70}
// CHECK:STDOUT: !73 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.d9636b4f2fbd64a7:core.Destroy.Core", scope: null, file: !36, line: 5, type: !41, spFlags: DISPFlagDefinition, unit: !37, retainedNodes: !76)
// CHECK:STDOUT: !74 = !DILocalVariable(arg: 1, scope: !73, type: !7)
// CHECK:STDOUT: !75 = !DILocation(line: 5, column: 15, scope: !73)
// CHECK:STDOUT: !76 = !{!74}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'direct_same_package.carbon'
// CHECK:STDOUT: source_filename = "direct_same_package.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CDirect.OwningPackage(ptr sret({ i32 }) %return, ptr %v1, ptr %v2) #0 !dbg !79 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_18.1.temp = alloca { i32 }, align 4, !dbg !87
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_18.1.temp), !dbg !87
// CHECK:STDOUT:   call void @"_COp.Vector.OwningPackage:MulWith.84588f41d61dafba.Core"(ptr %.loc7_18.1.temp, ptr %v2, i32 2), !dbg !87
// CHECK:STDOUT:   call void @"_COp.Vector.OwningPackage:AddWith.f941828044620367.Core"(ptr %return, ptr %v1, ptr %.loc7_18.1.temp), !dbg !86
// CHECK:STDOUT:   call void @"_CSelfDestruct.d9636b4f2fbd64a7:core.Destroy.Core"(ptr %.loc7_18.1.temp), !dbg !87
// CHECK:STDOUT:   ret void, !dbg !88
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Vector.OwningPackage:MulWith.84588f41d61dafba.Core"(ptr sret({ i32 }), ptr, i32)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Vector.OwningPackage:AddWith.f941828044620367.Core"(ptr sret({ i32 }), ptr, ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.6dae29ece9a708c4:core.Destroy.Core"(ptr %self) #0 !dbg !90 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !92
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !94 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !96
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.a15077f50f3020c3:core.Destroy.Core"(ptr %self) #0 !dbg !98 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !100
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.a15077f50f3020c3:core.Destroy.Core"(ptr %self) #0 !dbg !102 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.a15077f50f3020c3:core.Destroy.Core"(ptr %self), !dbg !104
// CHECK:STDOUT:   ret void, !dbg !104
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.d9636b4f2fbd64a7:core.Destroy.Core"(ptr %self) #0 !dbg !106 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !108
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.d9636b4f2fbd64a7:core.Destroy.Core"(ptr %self) #0 !dbg !110 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.d9636b4f2fbd64a7:core.Destroy.Core"(ptr %self), !dbg !112
// CHECK:STDOUT:   ret void, !dbg !112
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!78}
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
// CHECK:STDOUT: !77 = !DIFile(filename: "direct_same_package.carbon", directory: "")
// CHECK:STDOUT: !78 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !77, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !79 = distinct !DISubprogram(name: "Direct", linkageName: "_CDirect.OwningPackage", scope: null, file: !77, line: 5, type: !9, spFlags: DISPFlagDefinition, unit: !78, retainedNodes: !89)
// CHECK:STDOUT: !80 = !DILocalVariable(arg: 1, scope: !79, type: !7)
// CHECK:STDOUT: !81 = !DILocalVariable(arg: 2, scope: !79, type: !7)
// CHECK:STDOUT: !86 = !DILocation(line: 7, column: 10, scope: !79)
// CHECK:STDOUT: !87 = !DILocation(line: 7, column: 15, scope: !79)
// CHECK:STDOUT: !88 = !DILocation(line: 7, column: 3, scope: !79)
// CHECK:STDOUT: !89 = !{!80, !81}
// CHECK:STDOUT: !90 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.6dae29ece9a708c4:core.Destroy.Core", scope: null, file: !77, line: 7, type: !39, spFlags: DISPFlagDefinition, unit: !78, retainedNodes: !93)
// CHECK:STDOUT: !91 = !DILocalVariable(arg: 1, scope: !90, type: !6)
// CHECK:STDOUT: !92 = !DILocation(line: 7, column: 15, scope: !90)
// CHECK:STDOUT: !93 = !{!91}
// CHECK:STDOUT: !94 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f58524831c37807e:core.Destroy.Core", scope: null, file: !77, line: 7, type: !39, spFlags: DISPFlagDefinition, unit: !78, retainedNodes: !97)
// CHECK:STDOUT: !95 = !DILocalVariable(arg: 1, scope: !94, type: !6)
// CHECK:STDOUT: !96 = !DILocation(line: 7, column: 15, scope: !94)
// CHECK:STDOUT: !97 = !{!95}
// CHECK:STDOUT: !98 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.a15077f50f3020c3:core.Destroy.Core", scope: null, file: !77, line: 7, type: !41, spFlags: DISPFlagDefinition, unit: !78, retainedNodes: !101)
// CHECK:STDOUT: !99 = !DILocalVariable(arg: 1, scope: !98, type: !7)
// CHECK:STDOUT: !100 = !DILocation(line: 7, column: 15, scope: !98)
// CHECK:STDOUT: !101 = !{!99}
// CHECK:STDOUT: !102 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.a15077f50f3020c3:core.Destroy.Core", scope: null, file: !77, line: 7, type: !41, spFlags: DISPFlagDefinition, unit: !78, retainedNodes: !105)
// CHECK:STDOUT: !103 = !DILocalVariable(arg: 1, scope: !102, type: !7)
// CHECK:STDOUT: !104 = !DILocation(line: 7, column: 15, scope: !102)
// CHECK:STDOUT: !105 = !{!103}
// CHECK:STDOUT: !106 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.d9636b4f2fbd64a7:core.Destroy.Core", scope: null, file: !77, line: 7, type: !41, spFlags: DISPFlagDefinition, unit: !78, retainedNodes: !109)
// CHECK:STDOUT: !107 = !DILocalVariable(arg: 1, scope: !106, type: !7)
// CHECK:STDOUT: !108 = !DILocation(line: 7, column: 15, scope: !106)
// CHECK:STDOUT: !109 = !{!107}
// CHECK:STDOUT: !110 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.d9636b4f2fbd64a7:core.Destroy.Core", scope: null, file: !77, line: 7, type: !41, spFlags: DISPFlagDefinition, unit: !78, retainedNodes: !113)
// CHECK:STDOUT: !111 = !DILocalVariable(arg: 1, scope: !110, type: !7)
// CHECK:STDOUT: !112 = !DILocation(line: 7, column: 15, scope: !110)
// CHECK:STDOUT: !113 = !{!111}
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
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.6dae29ece9a708c4:core.Destroy.Core"(ptr %self) #0 !dbg !118 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !120
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !122 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !124
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.a15077f50f3020c3:core.Destroy.Core"(ptr %self) #0 !dbg !126 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !128
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.a15077f50f3020c3:core.Destroy.Core"(ptr %self) #0 !dbg !130 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.a15077f50f3020c3:core.Destroy.Core"(ptr %self), !dbg !132
// CHECK:STDOUT:   ret void, !dbg !132
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.d9636b4f2fbd64a7:core.Destroy.Core"(ptr %self) #0 !dbg !134 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !136
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.d9636b4f2fbd64a7:core.Destroy.Core"(ptr %self) #0 !dbg !138 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.d9636b4f2fbd64a7:core.Destroy.Core"(ptr %self), !dbg !140
// CHECK:STDOUT:   ret void, !dbg !140
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CMake.OwningPackage(ptr sret({ i32 }))
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!115}
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
// CHECK:STDOUT: !114 = !DIFile(filename: "generic_same_package.carbon", directory: "")
// CHECK:STDOUT: !115 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !114, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !118 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.6dae29ece9a708c4:core.Destroy.Core", scope: null, file: !114, line: 6, type: !39, spFlags: DISPFlagDefinition, unit: !115, retainedNodes: !121)
// CHECK:STDOUT: !119 = !DILocalVariable(arg: 1, scope: !118, type: !6)
// CHECK:STDOUT: !120 = !DILocation(line: 6, column: 15, scope: !118)
// CHECK:STDOUT: !121 = !{!119}
// CHECK:STDOUT: !122 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f58524831c37807e:core.Destroy.Core", scope: null, file: !114, line: 6, type: !39, spFlags: DISPFlagDefinition, unit: !115, retainedNodes: !125)
// CHECK:STDOUT: !123 = !DILocalVariable(arg: 1, scope: !122, type: !6)
// CHECK:STDOUT: !124 = !DILocation(line: 6, column: 15, scope: !122)
// CHECK:STDOUT: !125 = !{!123}
// CHECK:STDOUT: !126 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.a15077f50f3020c3:core.Destroy.Core", scope: null, file: !114, line: 6, type: !41, spFlags: DISPFlagDefinition, unit: !115, retainedNodes: !129)
// CHECK:STDOUT: !127 = !DILocalVariable(arg: 1, scope: !126, type: !7)
// CHECK:STDOUT: !128 = !DILocation(line: 6, column: 15, scope: !126)
// CHECK:STDOUT: !129 = !{!127}
// CHECK:STDOUT: !130 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.a15077f50f3020c3:core.Destroy.Core", scope: null, file: !114, line: 6, type: !41, spFlags: DISPFlagDefinition, unit: !115, retainedNodes: !133)
// CHECK:STDOUT: !131 = !DILocalVariable(arg: 1, scope: !130, type: !7)
// CHECK:STDOUT: !132 = !DILocation(line: 6, column: 15, scope: !130)
// CHECK:STDOUT: !133 = !{!131}
// CHECK:STDOUT: !134 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.d9636b4f2fbd64a7:core.Destroy.Core", scope: null, file: !114, line: 6, type: !41, spFlags: DISPFlagDefinition, unit: !115, retainedNodes: !137)
// CHECK:STDOUT: !135 = !DILocalVariable(arg: 1, scope: !134, type: !7)
// CHECK:STDOUT: !136 = !DILocation(line: 6, column: 15, scope: !134)
// CHECK:STDOUT: !137 = !{!135}
// CHECK:STDOUT: !138 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.d9636b4f2fbd64a7:core.Destroy.Core", scope: null, file: !114, line: 6, type: !41, spFlags: DISPFlagDefinition, unit: !115, retainedNodes: !141)
// CHECK:STDOUT: !139 = !DILocalVariable(arg: 1, scope: !138, type: !7)
// CHECK:STDOUT: !140 = !DILocation(line: 6, column: 15, scope: !138)
// CHECK:STDOUT: !141 = !{!139}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'indirect_same_package.carbon'
// CHECK:STDOUT: source_filename = "indirect_same_package.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCallGeneric.OwningPackage() #0 !dbg !146 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_20.1.temp = alloca { i32 }, align 4, !dbg !147
// CHECK:STDOUT:   %.loc7_28.1.temp = alloca { i32 }, align 4, !dbg !148
// CHECK:STDOUT:   %.loc7_29.2.temp = alloca { i32 }, align 4, !dbg !149
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_20.1.temp), !dbg !147
// CHECK:STDOUT:   call void @_CMake.OwningPackage(ptr %.loc7_20.1.temp), !dbg !147
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_28.1.temp), !dbg !148
// CHECK:STDOUT:   call void @_CMake.OwningPackage(ptr %.loc7_28.1.temp), !dbg !148
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_29.2.temp), !dbg !149
// CHECK:STDOUT:   call void @_CGeneric.OwningPackage.e43630e9a6c38c3f(ptr %.loc7_29.2.temp, ptr %.loc7_20.1.temp, ptr %.loc7_28.1.temp), !dbg !149
// CHECK:STDOUT:   call void @"_CSelfDestruct.d9636b4f2fbd64a7:core.Destroy.Core"(ptr %.loc7_29.2.temp), !dbg !149
// CHECK:STDOUT:   call void @"_CSelfDestruct.d9636b4f2fbd64a7:core.Destroy.Core"(ptr %.loc7_28.1.temp), !dbg !148
// CHECK:STDOUT:   call void @"_CSelfDestruct.d9636b4f2fbd64a7:core.Destroy.Core"(ptr %.loc7_20.1.temp), !dbg !147
// CHECK:STDOUT:   ret void, !dbg !150
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CMake.OwningPackage(ptr sret({ i32 }))
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.6dae29ece9a708c4:core.Destroy.Core"(ptr %self) #0 !dbg !151 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !153
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !155 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !157
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.a15077f50f3020c3:core.Destroy.Core"(ptr %self) #0 !dbg !159 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !161
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.a15077f50f3020c3:core.Destroy.Core"(ptr %self) #0 !dbg !163 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.a15077f50f3020c3:core.Destroy.Core"(ptr %self), !dbg !165
// CHECK:STDOUT:   ret void, !dbg !165
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.d9636b4f2fbd64a7:core.Destroy.Core"(ptr %self) #0 !dbg !167 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !169
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.d9636b4f2fbd64a7:core.Destroy.Core"(ptr %self) #0 !dbg !171 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.d9636b4f2fbd64a7:core.Destroy.Core"(ptr %self), !dbg !173
// CHECK:STDOUT:   ret void, !dbg !173
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CGeneric.OwningPackage.e43630e9a6c38c3f(ptr sret({ i32 }) %return, ptr %v1, ptr %v2) #0 !dbg !175 {
// CHECK:STDOUT:   %temp = alloca { i32 }, align 4, !dbg !183
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp), !dbg !183
// CHECK:STDOUT:   call void @"_COp.Vector.OwningPackage:MulWith.84588f41d61dafba.Core"(ptr %temp, ptr %v2, i32 2), !dbg !183
// CHECK:STDOUT:   call void @"_COp.Vector.OwningPackage:AddWith.f941828044620367.Core"(ptr %return, ptr %v1, ptr %temp), !dbg !182
// CHECK:STDOUT:   call void @"_CSelfDestruct.d9636b4f2fbd64a7:core.Destroy.Core"(ptr %temp), !dbg !183
// CHECK:STDOUT:   ret void, !dbg !184
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Vector.OwningPackage:MulWith.84588f41d61dafba.Core"(ptr sret({ i32 }), ptr, i32)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Vector.OwningPackage:AddWith.f941828044620367.Core"(ptr sret({ i32 }), ptr, ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CMake.OwningPackage, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @"_CSelfDestruct.d9636b4f2fbd64a7:core.Destroy.Core", { 0, 3, 2, 1 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 3, 2, 1 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!143}
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
// CHECK:STDOUT: !114 = !DIFile(filename: "generic_same_package.carbon", directory: "")
// CHECK:STDOUT: !142 = !DIFile(filename: "indirect_same_package.carbon", directory: "")
// CHECK:STDOUT: !143 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !142, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !144 = !{null}
// CHECK:STDOUT: !145 = !DISubroutineType(types: !144)
// CHECK:STDOUT: !146 = distinct !DISubprogram(name: "CallGeneric", linkageName: "_CCallGeneric.OwningPackage", scope: null, file: !142, line: 5, type: !145, spFlags: DISPFlagDefinition, unit: !143)
// CHECK:STDOUT: !147 = !DILocation(line: 7, column: 15, scope: !146)
// CHECK:STDOUT: !148 = !DILocation(line: 7, column: 23, scope: !146)
// CHECK:STDOUT: !149 = !DILocation(line: 7, column: 3, scope: !146)
// CHECK:STDOUT: !150 = !DILocation(line: 5, column: 1, scope: !146)
// CHECK:STDOUT: !151 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.6dae29ece9a708c4:core.Destroy.Core", scope: null, file: !142, line: 7, type: !39, spFlags: DISPFlagDefinition, unit: !143, retainedNodes: !154)
// CHECK:STDOUT: !152 = !DILocalVariable(arg: 1, scope: !151, type: !6)
// CHECK:STDOUT: !153 = !DILocation(line: 7, column: 3, scope: !151)
// CHECK:STDOUT: !154 = !{!152}
// CHECK:STDOUT: !155 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f58524831c37807e:core.Destroy.Core", scope: null, file: !142, line: 7, type: !39, spFlags: DISPFlagDefinition, unit: !143, retainedNodes: !158)
// CHECK:STDOUT: !156 = !DILocalVariable(arg: 1, scope: !155, type: !6)
// CHECK:STDOUT: !157 = !DILocation(line: 7, column: 3, scope: !155)
// CHECK:STDOUT: !158 = !{!156}
// CHECK:STDOUT: !159 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.a15077f50f3020c3:core.Destroy.Core", scope: null, file: !142, line: 7, type: !41, spFlags: DISPFlagDefinition, unit: !143, retainedNodes: !162)
// CHECK:STDOUT: !160 = !DILocalVariable(arg: 1, scope: !159, type: !7)
// CHECK:STDOUT: !161 = !DILocation(line: 7, column: 3, scope: !159)
// CHECK:STDOUT: !162 = !{!160}
// CHECK:STDOUT: !163 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.a15077f50f3020c3:core.Destroy.Core", scope: null, file: !142, line: 7, type: !41, spFlags: DISPFlagDefinition, unit: !143, retainedNodes: !166)
// CHECK:STDOUT: !164 = !DILocalVariable(arg: 1, scope: !163, type: !7)
// CHECK:STDOUT: !165 = !DILocation(line: 7, column: 3, scope: !163)
// CHECK:STDOUT: !166 = !{!164}
// CHECK:STDOUT: !167 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.d9636b4f2fbd64a7:core.Destroy.Core", scope: null, file: !142, line: 7, type: !41, spFlags: DISPFlagDefinition, unit: !143, retainedNodes: !170)
// CHECK:STDOUT: !168 = !DILocalVariable(arg: 1, scope: !167, type: !7)
// CHECK:STDOUT: !169 = !DILocation(line: 7, column: 3, scope: !167)
// CHECK:STDOUT: !170 = !{!168}
// CHECK:STDOUT: !171 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.d9636b4f2fbd64a7:core.Destroy.Core", scope: null, file: !142, line: 7, type: !41, spFlags: DISPFlagDefinition, unit: !143, retainedNodes: !174)
// CHECK:STDOUT: !172 = !DILocalVariable(arg: 1, scope: !171, type: !7)
// CHECK:STDOUT: !173 = !DILocation(line: 7, column: 3, scope: !171)
// CHECK:STDOUT: !174 = !{!172}
// CHECK:STDOUT: !175 = distinct !DISubprogram(name: "Generic", linkageName: "_CGeneric.OwningPackage.e43630e9a6c38c3f", scope: null, file: !114, line: 5, type: !9, spFlags: DISPFlagDefinition, unit: !143, retainedNodes: !185)
// CHECK:STDOUT: !176 = !DILocalVariable(arg: 1, scope: !175, type: !7)
// CHECK:STDOUT: !177 = !DILocalVariable(arg: 2, scope: !175, type: !7)
// CHECK:STDOUT: !182 = !DILocation(line: 6, column: 10, scope: !175)
// CHECK:STDOUT: !183 = !DILocation(line: 6, column: 15, scope: !175)
// CHECK:STDOUT: !184 = !DILocation(line: 6, column: 3, scope: !175)
// CHECK:STDOUT: !185 = !{!176, !177}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'direct_different_package.carbon'
// CHECK:STDOUT: source_filename = "direct_different_package.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CDirect.OtherPackage(ptr sret({ i32 }) %return, ptr %v1, ptr %v2) #0 !dbg !188 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_18.1.temp = alloca { i32 }, align 4, !dbg !196
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_18.1.temp), !dbg !196
// CHECK:STDOUT:   call void @"_COp.Vector.OwningPackage:MulWith.84588f41d61dafba.Core"(ptr %.loc7_18.1.temp, ptr %v2, i32 2), !dbg !196
// CHECK:STDOUT:   call void @"_COp.Vector.OwningPackage:AddWith.f941828044620367.Core"(ptr %return, ptr %v1, ptr %.loc7_18.1.temp), !dbg !195
// CHECK:STDOUT:   call void @"_CSelfDestruct.d9636b4f2fbd64a7:core.Destroy.Core"(ptr %.loc7_18.1.temp), !dbg !196
// CHECK:STDOUT:   ret void, !dbg !197
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Vector.OwningPackage:MulWith.84588f41d61dafba.Core"(ptr sret({ i32 }), ptr, i32)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Vector.OwningPackage:AddWith.f941828044620367.Core"(ptr sret({ i32 }), ptr, ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.6dae29ece9a708c4:core.Destroy.Core"(ptr %self) #0 !dbg !199 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !201
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !203 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !205
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.a15077f50f3020c3:core.Destroy.Core"(ptr %self) #0 !dbg !207 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !209
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.a15077f50f3020c3:core.Destroy.Core"(ptr %self) #0 !dbg !211 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.a15077f50f3020c3:core.Destroy.Core"(ptr %self), !dbg !213
// CHECK:STDOUT:   ret void, !dbg !213
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.d9636b4f2fbd64a7:core.Destroy.Core"(ptr %self) #0 !dbg !215 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !217
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.d9636b4f2fbd64a7:core.Destroy.Core"(ptr %self) #0 !dbg !219 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.d9636b4f2fbd64a7:core.Destroy.Core"(ptr %self), !dbg !221
// CHECK:STDOUT:   ret void, !dbg !221
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!187}
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
// CHECK:STDOUT: !186 = !DIFile(filename: "direct_different_package.carbon", directory: "")
// CHECK:STDOUT: !187 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !186, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !188 = distinct !DISubprogram(name: "Direct", linkageName: "_CDirect.OtherPackage", scope: null, file: !186, line: 5, type: !9, spFlags: DISPFlagDefinition, unit: !187, retainedNodes: !198)
// CHECK:STDOUT: !189 = !DILocalVariable(arg: 1, scope: !188, type: !7)
// CHECK:STDOUT: !190 = !DILocalVariable(arg: 2, scope: !188, type: !7)
// CHECK:STDOUT: !195 = !DILocation(line: 7, column: 10, scope: !188)
// CHECK:STDOUT: !196 = !DILocation(line: 7, column: 15, scope: !188)
// CHECK:STDOUT: !197 = !DILocation(line: 7, column: 3, scope: !188)
// CHECK:STDOUT: !198 = !{!189, !190}
// CHECK:STDOUT: !199 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.6dae29ece9a708c4:core.Destroy.Core", scope: null, file: !186, line: 7, type: !39, spFlags: DISPFlagDefinition, unit: !187, retainedNodes: !202)
// CHECK:STDOUT: !200 = !DILocalVariable(arg: 1, scope: !199, type: !6)
// CHECK:STDOUT: !201 = !DILocation(line: 7, column: 15, scope: !199)
// CHECK:STDOUT: !202 = !{!200}
// CHECK:STDOUT: !203 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f58524831c37807e:core.Destroy.Core", scope: null, file: !186, line: 7, type: !39, spFlags: DISPFlagDefinition, unit: !187, retainedNodes: !206)
// CHECK:STDOUT: !204 = !DILocalVariable(arg: 1, scope: !203, type: !6)
// CHECK:STDOUT: !205 = !DILocation(line: 7, column: 15, scope: !203)
// CHECK:STDOUT: !206 = !{!204}
// CHECK:STDOUT: !207 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.a15077f50f3020c3:core.Destroy.Core", scope: null, file: !186, line: 7, type: !41, spFlags: DISPFlagDefinition, unit: !187, retainedNodes: !210)
// CHECK:STDOUT: !208 = !DILocalVariable(arg: 1, scope: !207, type: !7)
// CHECK:STDOUT: !209 = !DILocation(line: 7, column: 15, scope: !207)
// CHECK:STDOUT: !210 = !{!208}
// CHECK:STDOUT: !211 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.a15077f50f3020c3:core.Destroy.Core", scope: null, file: !186, line: 7, type: !41, spFlags: DISPFlagDefinition, unit: !187, retainedNodes: !214)
// CHECK:STDOUT: !212 = !DILocalVariable(arg: 1, scope: !211, type: !7)
// CHECK:STDOUT: !213 = !DILocation(line: 7, column: 15, scope: !211)
// CHECK:STDOUT: !214 = !{!212}
// CHECK:STDOUT: !215 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.d9636b4f2fbd64a7:core.Destroy.Core", scope: null, file: !186, line: 7, type: !41, spFlags: DISPFlagDefinition, unit: !187, retainedNodes: !218)
// CHECK:STDOUT: !216 = !DILocalVariable(arg: 1, scope: !215, type: !7)
// CHECK:STDOUT: !217 = !DILocation(line: 7, column: 15, scope: !215)
// CHECK:STDOUT: !218 = !{!216}
// CHECK:STDOUT: !219 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.d9636b4f2fbd64a7:core.Destroy.Core", scope: null, file: !186, line: 7, type: !41, spFlags: DISPFlagDefinition, unit: !187, retainedNodes: !222)
// CHECK:STDOUT: !220 = !DILocalVariable(arg: 1, scope: !219, type: !7)
// CHECK:STDOUT: !221 = !DILocation(line: 7, column: 15, scope: !219)
// CHECK:STDOUT: !222 = !{!220}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'indirect_different_package_via_generic_in_same_package.carbon'
// CHECK:STDOUT: source_filename = "indirect_different_package_via_generic_in_same_package.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCallGenericInOwningPackage.OtherPackage() #0 !dbg !225 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_48.1.temp = alloca { i32 }, align 4, !dbg !226
// CHECK:STDOUT:   %.loc7_70.1.temp = alloca { i32 }, align 4, !dbg !227
// CHECK:STDOUT:   %.loc7_71.2.temp = alloca { i32 }, align 4, !dbg !228
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_48.1.temp), !dbg !226
// CHECK:STDOUT:   call void @_CMake.OwningPackage(ptr %.loc7_48.1.temp), !dbg !226
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_70.1.temp), !dbg !227
// CHECK:STDOUT:   call void @_CMake.OwningPackage(ptr %.loc7_70.1.temp), !dbg !227
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_71.2.temp), !dbg !228
// CHECK:STDOUT:   call void @_CGeneric.OwningPackage.e43630e9a6c38c3f(ptr %.loc7_71.2.temp, ptr %.loc7_48.1.temp, ptr %.loc7_70.1.temp), !dbg !228
// CHECK:STDOUT:   call void @"_CSelfDestruct.d9636b4f2fbd64a7:core.Destroy.Core"(ptr %.loc7_71.2.temp), !dbg !228
// CHECK:STDOUT:   call void @"_CSelfDestruct.d9636b4f2fbd64a7:core.Destroy.Core"(ptr %.loc7_70.1.temp), !dbg !227
// CHECK:STDOUT:   call void @"_CSelfDestruct.d9636b4f2fbd64a7:core.Destroy.Core"(ptr %.loc7_48.1.temp), !dbg !226
// CHECK:STDOUT:   ret void, !dbg !229
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CMake.OwningPackage(ptr sret({ i32 }))
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.6dae29ece9a708c4:core.Destroy.Core"(ptr %self) #0 !dbg !230 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !232
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !234 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !236
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.a15077f50f3020c3:core.Destroy.Core"(ptr %self) #0 !dbg !238 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !240
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.a15077f50f3020c3:core.Destroy.Core"(ptr %self) #0 !dbg !242 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.a15077f50f3020c3:core.Destroy.Core"(ptr %self), !dbg !244
// CHECK:STDOUT:   ret void, !dbg !244
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.d9636b4f2fbd64a7:core.Destroy.Core"(ptr %self) #0 !dbg !246 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !248
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.d9636b4f2fbd64a7:core.Destroy.Core"(ptr %self) #0 !dbg !250 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.d9636b4f2fbd64a7:core.Destroy.Core"(ptr %self), !dbg !252
// CHECK:STDOUT:   ret void, !dbg !252
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CGeneric.OwningPackage.e43630e9a6c38c3f(ptr sret({ i32 }) %return, ptr %v1, ptr %v2) #0 !dbg !254 {
// CHECK:STDOUT:   %temp = alloca { i32 }, align 4, !dbg !262
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp), !dbg !262
// CHECK:STDOUT:   call void @"_COp.Vector.OwningPackage:MulWith.84588f41d61dafba.Core"(ptr %temp, ptr %v2, i32 2), !dbg !262
// CHECK:STDOUT:   call void @"_COp.Vector.OwningPackage:AddWith.f941828044620367.Core"(ptr %return, ptr %v1, ptr %temp), !dbg !261
// CHECK:STDOUT:   call void @"_CSelfDestruct.d9636b4f2fbd64a7:core.Destroy.Core"(ptr %temp), !dbg !262
// CHECK:STDOUT:   ret void, !dbg !263
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Vector.OwningPackage:MulWith.84588f41d61dafba.Core"(ptr sret({ i32 }), ptr, i32)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Vector.OwningPackage:AddWith.f941828044620367.Core"(ptr sret({ i32 }), ptr, ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CMake.OwningPackage, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @"_CSelfDestruct.d9636b4f2fbd64a7:core.Destroy.Core", { 0, 3, 2, 1 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 3, 2, 1 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!224}
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
// CHECK:STDOUT: !114 = !DIFile(filename: "generic_same_package.carbon", directory: "")
// CHECK:STDOUT: !144 = !{null}
// CHECK:STDOUT: !145 = !DISubroutineType(types: !144)
// CHECK:STDOUT: !223 = !DIFile(filename: "indirect_different_package_via_generic_in_same_package.carbon", directory: "")
// CHECK:STDOUT: !224 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !223, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !225 = distinct !DISubprogram(name: "CallGenericInOwningPackage", linkageName: "_CCallGenericInOwningPackage.OtherPackage", scope: null, file: !223, line: 5, type: !145, spFlags: DISPFlagDefinition, unit: !224)
// CHECK:STDOUT: !226 = !DILocation(line: 7, column: 29, scope: !225)
// CHECK:STDOUT: !227 = !DILocation(line: 7, column: 51, scope: !225)
// CHECK:STDOUT: !228 = !DILocation(line: 7, column: 3, scope: !225)
// CHECK:STDOUT: !229 = !DILocation(line: 5, column: 1, scope: !225)
// CHECK:STDOUT: !230 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.6dae29ece9a708c4:core.Destroy.Core", scope: null, file: !223, line: 7, type: !39, spFlags: DISPFlagDefinition, unit: !224, retainedNodes: !233)
// CHECK:STDOUT: !231 = !DILocalVariable(arg: 1, scope: !230, type: !6)
// CHECK:STDOUT: !232 = !DILocation(line: 7, column: 3, scope: !230)
// CHECK:STDOUT: !233 = !{!231}
// CHECK:STDOUT: !234 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f58524831c37807e:core.Destroy.Core", scope: null, file: !223, line: 7, type: !39, spFlags: DISPFlagDefinition, unit: !224, retainedNodes: !237)
// CHECK:STDOUT: !235 = !DILocalVariable(arg: 1, scope: !234, type: !6)
// CHECK:STDOUT: !236 = !DILocation(line: 7, column: 3, scope: !234)
// CHECK:STDOUT: !237 = !{!235}
// CHECK:STDOUT: !238 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.a15077f50f3020c3:core.Destroy.Core", scope: null, file: !223, line: 7, type: !41, spFlags: DISPFlagDefinition, unit: !224, retainedNodes: !241)
// CHECK:STDOUT: !239 = !DILocalVariable(arg: 1, scope: !238, type: !7)
// CHECK:STDOUT: !240 = !DILocation(line: 7, column: 3, scope: !238)
// CHECK:STDOUT: !241 = !{!239}
// CHECK:STDOUT: !242 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.a15077f50f3020c3:core.Destroy.Core", scope: null, file: !223, line: 7, type: !41, spFlags: DISPFlagDefinition, unit: !224, retainedNodes: !245)
// CHECK:STDOUT: !243 = !DILocalVariable(arg: 1, scope: !242, type: !7)
// CHECK:STDOUT: !244 = !DILocation(line: 7, column: 3, scope: !242)
// CHECK:STDOUT: !245 = !{!243}
// CHECK:STDOUT: !246 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.d9636b4f2fbd64a7:core.Destroy.Core", scope: null, file: !223, line: 7, type: !41, spFlags: DISPFlagDefinition, unit: !224, retainedNodes: !249)
// CHECK:STDOUT: !247 = !DILocalVariable(arg: 1, scope: !246, type: !7)
// CHECK:STDOUT: !248 = !DILocation(line: 7, column: 3, scope: !246)
// CHECK:STDOUT: !249 = !{!247}
// CHECK:STDOUT: !250 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.d9636b4f2fbd64a7:core.Destroy.Core", scope: null, file: !223, line: 7, type: !41, spFlags: DISPFlagDefinition, unit: !224, retainedNodes: !253)
// CHECK:STDOUT: !251 = !DILocalVariable(arg: 1, scope: !250, type: !7)
// CHECK:STDOUT: !252 = !DILocation(line: 7, column: 3, scope: !250)
// CHECK:STDOUT: !253 = !{!251}
// CHECK:STDOUT: !254 = distinct !DISubprogram(name: "Generic", linkageName: "_CGeneric.OwningPackage.e43630e9a6c38c3f", scope: null, file: !114, line: 5, type: !9, spFlags: DISPFlagDefinition, unit: !224, retainedNodes: !264)
// CHECK:STDOUT: !255 = !DILocalVariable(arg: 1, scope: !254, type: !7)
// CHECK:STDOUT: !256 = !DILocalVariable(arg: 2, scope: !254, type: !7)
// CHECK:STDOUT: !261 = !DILocation(line: 6, column: 10, scope: !254)
// CHECK:STDOUT: !262 = !DILocation(line: 6, column: 15, scope: !254)
// CHECK:STDOUT: !263 = !DILocation(line: 6, column: 3, scope: !254)
// CHECK:STDOUT: !264 = !{!255, !256}
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
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.6dae29ece9a708c4:core.Destroy.Core"(ptr %self) #0 !dbg !267 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !269
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !271 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !273
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.a15077f50f3020c3:core.Destroy.Core"(ptr %self) #0 !dbg !275 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !277
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.a15077f50f3020c3:core.Destroy.Core"(ptr %self) #0 !dbg !279 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.a15077f50f3020c3:core.Destroy.Core"(ptr %self), !dbg !281
// CHECK:STDOUT:   ret void, !dbg !281
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.d9636b4f2fbd64a7:core.Destroy.Core"(ptr %self) #0 !dbg !283 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !285
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.d9636b4f2fbd64a7:core.Destroy.Core"(ptr %self) #0 !dbg !287 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.d9636b4f2fbd64a7:core.Destroy.Core"(ptr %self), !dbg !289
// CHECK:STDOUT:   ret void, !dbg !289
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CMake.OtherPackage(ptr sret({ i32 }))
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!266}
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
// CHECK:STDOUT: !265 = !DIFile(filename: "generic_different_package.carbon", directory: "")
// CHECK:STDOUT: !266 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !265, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !267 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.6dae29ece9a708c4:core.Destroy.Core", scope: null, file: !265, line: 6, type: !39, spFlags: DISPFlagDefinition, unit: !266, retainedNodes: !270)
// CHECK:STDOUT: !268 = !DILocalVariable(arg: 1, scope: !267, type: !6)
// CHECK:STDOUT: !269 = !DILocation(line: 6, column: 15, scope: !267)
// CHECK:STDOUT: !270 = !{!268}
// CHECK:STDOUT: !271 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f58524831c37807e:core.Destroy.Core", scope: null, file: !265, line: 6, type: !39, spFlags: DISPFlagDefinition, unit: !266, retainedNodes: !274)
// CHECK:STDOUT: !272 = !DILocalVariable(arg: 1, scope: !271, type: !6)
// CHECK:STDOUT: !273 = !DILocation(line: 6, column: 15, scope: !271)
// CHECK:STDOUT: !274 = !{!272}
// CHECK:STDOUT: !275 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.a15077f50f3020c3:core.Destroy.Core", scope: null, file: !265, line: 6, type: !41, spFlags: DISPFlagDefinition, unit: !266, retainedNodes: !278)
// CHECK:STDOUT: !276 = !DILocalVariable(arg: 1, scope: !275, type: !7)
// CHECK:STDOUT: !277 = !DILocation(line: 6, column: 15, scope: !275)
// CHECK:STDOUT: !278 = !{!276}
// CHECK:STDOUT: !279 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.a15077f50f3020c3:core.Destroy.Core", scope: null, file: !265, line: 6, type: !41, spFlags: DISPFlagDefinition, unit: !266, retainedNodes: !282)
// CHECK:STDOUT: !280 = !DILocalVariable(arg: 1, scope: !279, type: !7)
// CHECK:STDOUT: !281 = !DILocation(line: 6, column: 15, scope: !279)
// CHECK:STDOUT: !282 = !{!280}
// CHECK:STDOUT: !283 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.d9636b4f2fbd64a7:core.Destroy.Core", scope: null, file: !265, line: 6, type: !41, spFlags: DISPFlagDefinition, unit: !266, retainedNodes: !286)
// CHECK:STDOUT: !284 = !DILocalVariable(arg: 1, scope: !283, type: !7)
// CHECK:STDOUT: !285 = !DILocation(line: 6, column: 15, scope: !283)
// CHECK:STDOUT: !286 = !{!284}
// CHECK:STDOUT: !287 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.d9636b4f2fbd64a7:core.Destroy.Core", scope: null, file: !265, line: 6, type: !41, spFlags: DISPFlagDefinition, unit: !266, retainedNodes: !290)
// CHECK:STDOUT: !288 = !DILocalVariable(arg: 1, scope: !287, type: !7)
// CHECK:STDOUT: !289 = !DILocation(line: 6, column: 15, scope: !287)
// CHECK:STDOUT: !290 = !{!288}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'indirect_different_package_via_generic_in_different_package.carbon'
// CHECK:STDOUT: source_filename = "indirect_different_package_via_generic_in_different_package.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCallGeneric.OtherPackage() #0 !dbg !293 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_20.1.temp = alloca { i32 }, align 4, !dbg !294
// CHECK:STDOUT:   %.loc7_28.1.temp = alloca { i32 }, align 4, !dbg !295
// CHECK:STDOUT:   %.loc7_29.2.temp = alloca { i32 }, align 4, !dbg !296
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_20.1.temp), !dbg !294
// CHECK:STDOUT:   call void @_CMake.OtherPackage(ptr %.loc7_20.1.temp), !dbg !294
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_28.1.temp), !dbg !295
// CHECK:STDOUT:   call void @_CMake.OtherPackage(ptr %.loc7_28.1.temp), !dbg !295
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_29.2.temp), !dbg !296
// CHECK:STDOUT:   call void @_CGeneric.OtherPackage.e43630e9a6c38c3f(ptr %.loc7_29.2.temp, ptr %.loc7_20.1.temp, ptr %.loc7_28.1.temp), !dbg !296
// CHECK:STDOUT:   call void @"_CSelfDestruct.d9636b4f2fbd64a7:core.Destroy.Core"(ptr %.loc7_29.2.temp), !dbg !296
// CHECK:STDOUT:   call void @"_CSelfDestruct.d9636b4f2fbd64a7:core.Destroy.Core"(ptr %.loc7_28.1.temp), !dbg !295
// CHECK:STDOUT:   call void @"_CSelfDestruct.d9636b4f2fbd64a7:core.Destroy.Core"(ptr %.loc7_20.1.temp), !dbg !294
// CHECK:STDOUT:   ret void, !dbg !297
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CMake.OtherPackage(ptr sret({ i32 }))
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.6dae29ece9a708c4:core.Destroy.Core"(ptr %self) #0 !dbg !298 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !300
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !302 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !304
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.a15077f50f3020c3:core.Destroy.Core"(ptr %self) #0 !dbg !306 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !308
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.a15077f50f3020c3:core.Destroy.Core"(ptr %self) #0 !dbg !310 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.a15077f50f3020c3:core.Destroy.Core"(ptr %self), !dbg !312
// CHECK:STDOUT:   ret void, !dbg !312
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.d9636b4f2fbd64a7:core.Destroy.Core"(ptr %self) #0 !dbg !314 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !316
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.d9636b4f2fbd64a7:core.Destroy.Core"(ptr %self) #0 !dbg !318 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.d9636b4f2fbd64a7:core.Destroy.Core"(ptr %self), !dbg !320
// CHECK:STDOUT:   ret void, !dbg !320
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CGeneric.OtherPackage.e43630e9a6c38c3f(ptr sret({ i32 }) %return, ptr %v1, ptr %v2) #0 !dbg !322 {
// CHECK:STDOUT:   %temp = alloca { i32 }, align 4, !dbg !330
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp), !dbg !330
// CHECK:STDOUT:   call void @"_COp.Vector.OwningPackage:MulWith.84588f41d61dafba.Core"(ptr %temp, ptr %v2, i32 2), !dbg !330
// CHECK:STDOUT:   call void @"_COp.Vector.OwningPackage:AddWith.f941828044620367.Core"(ptr %return, ptr %v1, ptr %temp), !dbg !329
// CHECK:STDOUT:   call void @"_CSelfDestruct.d9636b4f2fbd64a7:core.Destroy.Core"(ptr %temp), !dbg !330
// CHECK:STDOUT:   ret void, !dbg !331
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Vector.OwningPackage:MulWith.84588f41d61dafba.Core"(ptr sret({ i32 }), ptr, i32)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Vector.OwningPackage:AddWith.f941828044620367.Core"(ptr sret({ i32 }), ptr, ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CMake.OtherPackage, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @"_CSelfDestruct.d9636b4f2fbd64a7:core.Destroy.Core", { 0, 3, 2, 1 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 3, 2, 1 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!292}
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
// CHECK:STDOUT: !144 = !{null}
// CHECK:STDOUT: !145 = !DISubroutineType(types: !144)
// CHECK:STDOUT: !265 = !DIFile(filename: "generic_different_package.carbon", directory: "")
// CHECK:STDOUT: !291 = !DIFile(filename: "indirect_different_package_via_generic_in_different_package.carbon", directory: "")
// CHECK:STDOUT: !292 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !291, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !293 = distinct !DISubprogram(name: "CallGeneric", linkageName: "_CCallGeneric.OtherPackage", scope: null, file: !291, line: 5, type: !145, spFlags: DISPFlagDefinition, unit: !292)
// CHECK:STDOUT: !294 = !DILocation(line: 7, column: 15, scope: !293)
// CHECK:STDOUT: !295 = !DILocation(line: 7, column: 23, scope: !293)
// CHECK:STDOUT: !296 = !DILocation(line: 7, column: 3, scope: !293)
// CHECK:STDOUT: !297 = !DILocation(line: 5, column: 1, scope: !293)
// CHECK:STDOUT: !298 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.6dae29ece9a708c4:core.Destroy.Core", scope: null, file: !291, line: 7, type: !39, spFlags: DISPFlagDefinition, unit: !292, retainedNodes: !301)
// CHECK:STDOUT: !299 = !DILocalVariable(arg: 1, scope: !298, type: !6)
// CHECK:STDOUT: !300 = !DILocation(line: 7, column: 3, scope: !298)
// CHECK:STDOUT: !301 = !{!299}
// CHECK:STDOUT: !302 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f58524831c37807e:core.Destroy.Core", scope: null, file: !291, line: 7, type: !39, spFlags: DISPFlagDefinition, unit: !292, retainedNodes: !305)
// CHECK:STDOUT: !303 = !DILocalVariable(arg: 1, scope: !302, type: !6)
// CHECK:STDOUT: !304 = !DILocation(line: 7, column: 3, scope: !302)
// CHECK:STDOUT: !305 = !{!303}
// CHECK:STDOUT: !306 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.a15077f50f3020c3:core.Destroy.Core", scope: null, file: !291, line: 7, type: !41, spFlags: DISPFlagDefinition, unit: !292, retainedNodes: !309)
// CHECK:STDOUT: !307 = !DILocalVariable(arg: 1, scope: !306, type: !7)
// CHECK:STDOUT: !308 = !DILocation(line: 7, column: 3, scope: !306)
// CHECK:STDOUT: !309 = !{!307}
// CHECK:STDOUT: !310 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.a15077f50f3020c3:core.Destroy.Core", scope: null, file: !291, line: 7, type: !41, spFlags: DISPFlagDefinition, unit: !292, retainedNodes: !313)
// CHECK:STDOUT: !311 = !DILocalVariable(arg: 1, scope: !310, type: !7)
// CHECK:STDOUT: !312 = !DILocation(line: 7, column: 3, scope: !310)
// CHECK:STDOUT: !313 = !{!311}
// CHECK:STDOUT: !314 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.d9636b4f2fbd64a7:core.Destroy.Core", scope: null, file: !291, line: 7, type: !41, spFlags: DISPFlagDefinition, unit: !292, retainedNodes: !317)
// CHECK:STDOUT: !315 = !DILocalVariable(arg: 1, scope: !314, type: !7)
// CHECK:STDOUT: !316 = !DILocation(line: 7, column: 3, scope: !314)
// CHECK:STDOUT: !317 = !{!315}
// CHECK:STDOUT: !318 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.d9636b4f2fbd64a7:core.Destroy.Core", scope: null, file: !291, line: 7, type: !41, spFlags: DISPFlagDefinition, unit: !292, retainedNodes: !321)
// CHECK:STDOUT: !319 = !DILocalVariable(arg: 1, scope: !318, type: !7)
// CHECK:STDOUT: !320 = !DILocation(line: 7, column: 3, scope: !318)
// CHECK:STDOUT: !321 = !{!319}
// CHECK:STDOUT: !322 = distinct !DISubprogram(name: "Generic", linkageName: "_CGeneric.OtherPackage.e43630e9a6c38c3f", scope: null, file: !265, line: 5, type: !9, spFlags: DISPFlagDefinition, unit: !292, retainedNodes: !332)
// CHECK:STDOUT: !323 = !DILocalVariable(arg: 1, scope: !322, type: !7)
// CHECK:STDOUT: !324 = !DILocalVariable(arg: 2, scope: !322, type: !7)
// CHECK:STDOUT: !329 = !DILocation(line: 6, column: 10, scope: !322)
// CHECK:STDOUT: !330 = !DILocation(line: 6, column: 15, scope: !322)
// CHECK:STDOUT: !331 = !DILocation(line: 6, column: 3, scope: !322)
// CHECK:STDOUT: !332 = !{!323, !324}
// CHECK:STDOUT:
