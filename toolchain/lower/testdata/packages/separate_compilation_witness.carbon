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
// CHECK:STDOUT:   call void @"_COp.3e88a108c7e00533:core.Destroy.Core"(ptr %.loc5_18.1.temp), !dbg !50
// CHECK:STDOUT:   ret void, !dbg !51
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Vector.OwningPackage:MulWith.84588f41d61dafba.Core"(ptr sret({ i32 }), ptr, i32)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Vector.OwningPackage:AddWith.f941828044620367.Core"(ptr sret({ i32 }), ptr, ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.70db2d92ffddbef7:core.Destroy.Core"(ptr %self) #0 !dbg !53 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !55
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.7ec4ffd904220f7b:core.Destroy.Core"(ptr %self) #0 !dbg !57 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !59
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.3e88a108c7e00533:core.Destroy.Core"(ptr %self) #0 !dbg !61 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !63
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
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "Op", linkageName: "_COp.70db2d92ffddbef7:core.Destroy.Core", scope: null, file: !36, line: 5, type: !39, spFlags: DISPFlagDefinition, unit: !37, retainedNodes: !56)
// CHECK:STDOUT: !54 = !DILocalVariable(arg: 1, scope: !53, type: !6)
// CHECK:STDOUT: !55 = !DILocation(line: 5, column: 15, scope: !53)
// CHECK:STDOUT: !56 = !{!54}
// CHECK:STDOUT: !57 = distinct !DISubprogram(name: "Op", linkageName: "_COp.7ec4ffd904220f7b:core.Destroy.Core", scope: null, file: !36, line: 5, type: !41, spFlags: DISPFlagDefinition, unit: !37, retainedNodes: !60)
// CHECK:STDOUT: !58 = !DILocalVariable(arg: 1, scope: !57, type: !7)
// CHECK:STDOUT: !59 = !DILocation(line: 5, column: 15, scope: !57)
// CHECK:STDOUT: !60 = !{!58}
// CHECK:STDOUT: !61 = distinct !DISubprogram(name: "Op", linkageName: "_COp.3e88a108c7e00533:core.Destroy.Core", scope: null, file: !36, line: 5, type: !41, spFlags: DISPFlagDefinition, unit: !37, retainedNodes: !64)
// CHECK:STDOUT: !62 = !DILocalVariable(arg: 1, scope: !61, type: !7)
// CHECK:STDOUT: !63 = !DILocation(line: 5, column: 15, scope: !61)
// CHECK:STDOUT: !64 = !{!62}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'direct_same_package.carbon'
// CHECK:STDOUT: source_filename = "direct_same_package.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CDirect.OwningPackage(ptr sret({ i32 }) %return, ptr %v1, ptr %v2) #0 !dbg !67 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_18.1.temp = alloca { i32 }, align 4, !dbg !75
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_18.1.temp), !dbg !75
// CHECK:STDOUT:   call void @"_COp.Vector.OwningPackage:MulWith.84588f41d61dafba.Core"(ptr %.loc7_18.1.temp, ptr %v2, i32 2), !dbg !75
// CHECK:STDOUT:   call void @"_COp.Vector.OwningPackage:AddWith.f941828044620367.Core"(ptr %return, ptr %v1, ptr %.loc7_18.1.temp), !dbg !74
// CHECK:STDOUT:   call void @"_COp.3e88a108c7e00533:core.Destroy.Core"(ptr %.loc7_18.1.temp), !dbg !75
// CHECK:STDOUT:   ret void, !dbg !76
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Vector.OwningPackage:MulWith.84588f41d61dafba.Core"(ptr sret({ i32 }), ptr, i32)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Vector.OwningPackage:AddWith.f941828044620367.Core"(ptr sret({ i32 }), ptr, ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.70db2d92ffddbef7:core.Destroy.Core"(ptr %self) #0 !dbg !78 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !80
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.7ec4ffd904220f7b:core.Destroy.Core"(ptr %self) #0 !dbg !82 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !84
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.3e88a108c7e00533:core.Destroy.Core"(ptr %self) #0 !dbg !86 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !88
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!66}
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
// CHECK:STDOUT: !65 = !DIFile(filename: "direct_same_package.carbon", directory: "")
// CHECK:STDOUT: !66 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !65, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !67 = distinct !DISubprogram(name: "Direct", linkageName: "_CDirect.OwningPackage", scope: null, file: !65, line: 5, type: !9, spFlags: DISPFlagDefinition, unit: !66, retainedNodes: !77)
// CHECK:STDOUT: !68 = !DILocalVariable(arg: 1, scope: !67, type: !7)
// CHECK:STDOUT: !69 = !DILocalVariable(arg: 2, scope: !67, type: !7)
// CHECK:STDOUT: !74 = !DILocation(line: 7, column: 10, scope: !67)
// CHECK:STDOUT: !75 = !DILocation(line: 7, column: 15, scope: !67)
// CHECK:STDOUT: !76 = !DILocation(line: 7, column: 3, scope: !67)
// CHECK:STDOUT: !77 = !{!68, !69}
// CHECK:STDOUT: !78 = distinct !DISubprogram(name: "Op", linkageName: "_COp.70db2d92ffddbef7:core.Destroy.Core", scope: null, file: !65, line: 7, type: !39, spFlags: DISPFlagDefinition, unit: !66, retainedNodes: !81)
// CHECK:STDOUT: !79 = !DILocalVariable(arg: 1, scope: !78, type: !6)
// CHECK:STDOUT: !80 = !DILocation(line: 7, column: 15, scope: !78)
// CHECK:STDOUT: !81 = !{!79}
// CHECK:STDOUT: !82 = distinct !DISubprogram(name: "Op", linkageName: "_COp.7ec4ffd904220f7b:core.Destroy.Core", scope: null, file: !65, line: 7, type: !41, spFlags: DISPFlagDefinition, unit: !66, retainedNodes: !85)
// CHECK:STDOUT: !83 = !DILocalVariable(arg: 1, scope: !82, type: !7)
// CHECK:STDOUT: !84 = !DILocation(line: 7, column: 15, scope: !82)
// CHECK:STDOUT: !85 = !{!83}
// CHECK:STDOUT: !86 = distinct !DISubprogram(name: "Op", linkageName: "_COp.3e88a108c7e00533:core.Destroy.Core", scope: null, file: !65, line: 7, type: !41, spFlags: DISPFlagDefinition, unit: !66, retainedNodes: !89)
// CHECK:STDOUT: !87 = !DILocalVariable(arg: 1, scope: !86, type: !7)
// CHECK:STDOUT: !88 = !DILocation(line: 7, column: 15, scope: !86)
// CHECK:STDOUT: !89 = !{!87}
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
// CHECK:STDOUT: define weak_odr void @"_COp.70db2d92ffddbef7:core.Destroy.Core"(ptr %self) #0 !dbg !94 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !96
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.7ec4ffd904220f7b:core.Destroy.Core"(ptr %self) #0 !dbg !98 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !100
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.3e88a108c7e00533:core.Destroy.Core"(ptr %self) #0 !dbg !102 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !104
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CMake.OwningPackage(ptr sret({ i32 }))
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!91}
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
// CHECK:STDOUT: !90 = !DIFile(filename: "generic_same_package.carbon", directory: "")
// CHECK:STDOUT: !91 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !90, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !94 = distinct !DISubprogram(name: "Op", linkageName: "_COp.70db2d92ffddbef7:core.Destroy.Core", scope: null, file: !90, line: 6, type: !39, spFlags: DISPFlagDefinition, unit: !91, retainedNodes: !97)
// CHECK:STDOUT: !95 = !DILocalVariable(arg: 1, scope: !94, type: !6)
// CHECK:STDOUT: !96 = !DILocation(line: 6, column: 15, scope: !94)
// CHECK:STDOUT: !97 = !{!95}
// CHECK:STDOUT: !98 = distinct !DISubprogram(name: "Op", linkageName: "_COp.7ec4ffd904220f7b:core.Destroy.Core", scope: null, file: !90, line: 6, type: !41, spFlags: DISPFlagDefinition, unit: !91, retainedNodes: !101)
// CHECK:STDOUT: !99 = !DILocalVariable(arg: 1, scope: !98, type: !7)
// CHECK:STDOUT: !100 = !DILocation(line: 6, column: 15, scope: !98)
// CHECK:STDOUT: !101 = !{!99}
// CHECK:STDOUT: !102 = distinct !DISubprogram(name: "Op", linkageName: "_COp.3e88a108c7e00533:core.Destroy.Core", scope: null, file: !90, line: 6, type: !41, spFlags: DISPFlagDefinition, unit: !91, retainedNodes: !105)
// CHECK:STDOUT: !103 = !DILocalVariable(arg: 1, scope: !102, type: !7)
// CHECK:STDOUT: !104 = !DILocation(line: 6, column: 15, scope: !102)
// CHECK:STDOUT: !105 = !{!103}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'indirect_same_package.carbon'
// CHECK:STDOUT: source_filename = "indirect_same_package.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCallGeneric.OwningPackage() #0 !dbg !110 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_20.1.temp = alloca { i32 }, align 4, !dbg !111
// CHECK:STDOUT:   %.loc7_28.1.temp = alloca { i32 }, align 4, !dbg !112
// CHECK:STDOUT:   %.loc7_29.2.temp = alloca { i32 }, align 4, !dbg !113
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_20.1.temp), !dbg !111
// CHECK:STDOUT:   call void @_CMake.OwningPackage(ptr %.loc7_20.1.temp), !dbg !111
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_28.1.temp), !dbg !112
// CHECK:STDOUT:   call void @_CMake.OwningPackage(ptr %.loc7_28.1.temp), !dbg !112
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_29.2.temp), !dbg !113
// CHECK:STDOUT:   call void @_CGeneric.OwningPackage.e43630e9a6c38c3f(ptr %.loc7_29.2.temp, ptr %.loc7_20.1.temp, ptr %.loc7_28.1.temp), !dbg !113
// CHECK:STDOUT:   call void @"_COp.3e88a108c7e00533:core.Destroy.Core"(ptr %.loc7_29.2.temp), !dbg !113
// CHECK:STDOUT:   call void @"_COp.3e88a108c7e00533:core.Destroy.Core"(ptr %.loc7_28.1.temp), !dbg !112
// CHECK:STDOUT:   call void @"_COp.3e88a108c7e00533:core.Destroy.Core"(ptr %.loc7_20.1.temp), !dbg !111
// CHECK:STDOUT:   ret void, !dbg !114
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CMake.OwningPackage(ptr sret({ i32 }))
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.70db2d92ffddbef7:core.Destroy.Core"(ptr %self) #0 !dbg !115 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !117
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.7ec4ffd904220f7b:core.Destroy.Core"(ptr %self) #0 !dbg !119 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !121
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.3e88a108c7e00533:core.Destroy.Core"(ptr %self) #0 !dbg !123 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !125
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CGeneric.OwningPackage.e43630e9a6c38c3f(ptr sret({ i32 }) %return, ptr %v1, ptr %v2) #0 !dbg !127 {
// CHECK:STDOUT:   %temp = alloca { i32 }, align 4, !dbg !135
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp), !dbg !135
// CHECK:STDOUT:   call void @"_COp.Vector.OwningPackage:MulWith.84588f41d61dafba.Core"(ptr %temp, ptr %v2, i32 2), !dbg !135
// CHECK:STDOUT:   call void @"_COp.Vector.OwningPackage:AddWith.f941828044620367.Core"(ptr %return, ptr %v1, ptr %temp), !dbg !134
// CHECK:STDOUT:   call void @"_COp.3e88a108c7e00533:core.Destroy.Core"(ptr %temp), !dbg !135
// CHECK:STDOUT:   ret void, !dbg !136
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Vector.OwningPackage:MulWith.84588f41d61dafba.Core"(ptr sret({ i32 }), ptr, i32)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Vector.OwningPackage:AddWith.f941828044620367.Core"(ptr sret({ i32 }), ptr, ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CMake.OwningPackage, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @"_COp.3e88a108c7e00533:core.Destroy.Core", { 0, 3, 2, 1 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 3, 2, 1 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!107}
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
// CHECK:STDOUT: !90 = !DIFile(filename: "generic_same_package.carbon", directory: "")
// CHECK:STDOUT: !106 = !DIFile(filename: "indirect_same_package.carbon", directory: "")
// CHECK:STDOUT: !107 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !106, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !108 = !{null}
// CHECK:STDOUT: !109 = !DISubroutineType(types: !108)
// CHECK:STDOUT: !110 = distinct !DISubprogram(name: "CallGeneric", linkageName: "_CCallGeneric.OwningPackage", scope: null, file: !106, line: 5, type: !109, spFlags: DISPFlagDefinition, unit: !107)
// CHECK:STDOUT: !111 = !DILocation(line: 7, column: 15, scope: !110)
// CHECK:STDOUT: !112 = !DILocation(line: 7, column: 23, scope: !110)
// CHECK:STDOUT: !113 = !DILocation(line: 7, column: 3, scope: !110)
// CHECK:STDOUT: !114 = !DILocation(line: 5, column: 1, scope: !110)
// CHECK:STDOUT: !115 = distinct !DISubprogram(name: "Op", linkageName: "_COp.70db2d92ffddbef7:core.Destroy.Core", scope: null, file: !106, line: 7, type: !39, spFlags: DISPFlagDefinition, unit: !107, retainedNodes: !118)
// CHECK:STDOUT: !116 = !DILocalVariable(arg: 1, scope: !115, type: !6)
// CHECK:STDOUT: !117 = !DILocation(line: 7, column: 3, scope: !115)
// CHECK:STDOUT: !118 = !{!116}
// CHECK:STDOUT: !119 = distinct !DISubprogram(name: "Op", linkageName: "_COp.7ec4ffd904220f7b:core.Destroy.Core", scope: null, file: !106, line: 7, type: !41, spFlags: DISPFlagDefinition, unit: !107, retainedNodes: !122)
// CHECK:STDOUT: !120 = !DILocalVariable(arg: 1, scope: !119, type: !7)
// CHECK:STDOUT: !121 = !DILocation(line: 7, column: 3, scope: !119)
// CHECK:STDOUT: !122 = !{!120}
// CHECK:STDOUT: !123 = distinct !DISubprogram(name: "Op", linkageName: "_COp.3e88a108c7e00533:core.Destroy.Core", scope: null, file: !106, line: 7, type: !41, spFlags: DISPFlagDefinition, unit: !107, retainedNodes: !126)
// CHECK:STDOUT: !124 = !DILocalVariable(arg: 1, scope: !123, type: !7)
// CHECK:STDOUT: !125 = !DILocation(line: 7, column: 3, scope: !123)
// CHECK:STDOUT: !126 = !{!124}
// CHECK:STDOUT: !127 = distinct !DISubprogram(name: "Generic", linkageName: "_CGeneric.OwningPackage.e43630e9a6c38c3f", scope: null, file: !90, line: 5, type: !9, spFlags: DISPFlagDefinition, unit: !107, retainedNodes: !137)
// CHECK:STDOUT: !128 = !DILocalVariable(arg: 1, scope: !127, type: !7)
// CHECK:STDOUT: !129 = !DILocalVariable(arg: 2, scope: !127, type: !7)
// CHECK:STDOUT: !134 = !DILocation(line: 6, column: 10, scope: !127)
// CHECK:STDOUT: !135 = !DILocation(line: 6, column: 15, scope: !127)
// CHECK:STDOUT: !136 = !DILocation(line: 6, column: 3, scope: !127)
// CHECK:STDOUT: !137 = !{!128, !129}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'direct_different_package.carbon'
// CHECK:STDOUT: source_filename = "direct_different_package.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CDirect.OtherPackage(ptr sret({ i32 }) %return, ptr %v1, ptr %v2) #0 !dbg !140 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_18.1.temp = alloca { i32 }, align 4, !dbg !148
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_18.1.temp), !dbg !148
// CHECK:STDOUT:   call void @"_COp.Vector.OwningPackage:MulWith.84588f41d61dafba.Core"(ptr %.loc7_18.1.temp, ptr %v2, i32 2), !dbg !148
// CHECK:STDOUT:   call void @"_COp.Vector.OwningPackage:AddWith.f941828044620367.Core"(ptr %return, ptr %v1, ptr %.loc7_18.1.temp), !dbg !147
// CHECK:STDOUT:   call void @"_COp.3e88a108c7e00533:core.Destroy.Core"(ptr %.loc7_18.1.temp), !dbg !148
// CHECK:STDOUT:   ret void, !dbg !149
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Vector.OwningPackage:MulWith.84588f41d61dafba.Core"(ptr sret({ i32 }), ptr, i32)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Vector.OwningPackage:AddWith.f941828044620367.Core"(ptr sret({ i32 }), ptr, ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.70db2d92ffddbef7:core.Destroy.Core"(ptr %self) #0 !dbg !151 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !153
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.7ec4ffd904220f7b:core.Destroy.Core"(ptr %self) #0 !dbg !155 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !157
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.3e88a108c7e00533:core.Destroy.Core"(ptr %self) #0 !dbg !159 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !161
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!139}
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
// CHECK:STDOUT: !138 = !DIFile(filename: "direct_different_package.carbon", directory: "")
// CHECK:STDOUT: !139 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !138, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !140 = distinct !DISubprogram(name: "Direct", linkageName: "_CDirect.OtherPackage", scope: null, file: !138, line: 5, type: !9, spFlags: DISPFlagDefinition, unit: !139, retainedNodes: !150)
// CHECK:STDOUT: !141 = !DILocalVariable(arg: 1, scope: !140, type: !7)
// CHECK:STDOUT: !142 = !DILocalVariable(arg: 2, scope: !140, type: !7)
// CHECK:STDOUT: !147 = !DILocation(line: 7, column: 10, scope: !140)
// CHECK:STDOUT: !148 = !DILocation(line: 7, column: 15, scope: !140)
// CHECK:STDOUT: !149 = !DILocation(line: 7, column: 3, scope: !140)
// CHECK:STDOUT: !150 = !{!141, !142}
// CHECK:STDOUT: !151 = distinct !DISubprogram(name: "Op", linkageName: "_COp.70db2d92ffddbef7:core.Destroy.Core", scope: null, file: !138, line: 7, type: !39, spFlags: DISPFlagDefinition, unit: !139, retainedNodes: !154)
// CHECK:STDOUT: !152 = !DILocalVariable(arg: 1, scope: !151, type: !6)
// CHECK:STDOUT: !153 = !DILocation(line: 7, column: 15, scope: !151)
// CHECK:STDOUT: !154 = !{!152}
// CHECK:STDOUT: !155 = distinct !DISubprogram(name: "Op", linkageName: "_COp.7ec4ffd904220f7b:core.Destroy.Core", scope: null, file: !138, line: 7, type: !41, spFlags: DISPFlagDefinition, unit: !139, retainedNodes: !158)
// CHECK:STDOUT: !156 = !DILocalVariable(arg: 1, scope: !155, type: !7)
// CHECK:STDOUT: !157 = !DILocation(line: 7, column: 15, scope: !155)
// CHECK:STDOUT: !158 = !{!156}
// CHECK:STDOUT: !159 = distinct !DISubprogram(name: "Op", linkageName: "_COp.3e88a108c7e00533:core.Destroy.Core", scope: null, file: !138, line: 7, type: !41, spFlags: DISPFlagDefinition, unit: !139, retainedNodes: !162)
// CHECK:STDOUT: !160 = !DILocalVariable(arg: 1, scope: !159, type: !7)
// CHECK:STDOUT: !161 = !DILocation(line: 7, column: 15, scope: !159)
// CHECK:STDOUT: !162 = !{!160}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'indirect_different_package_via_generic_in_same_package.carbon'
// CHECK:STDOUT: source_filename = "indirect_different_package_via_generic_in_same_package.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCallGenericInOwningPackage.OtherPackage() #0 !dbg !165 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_48.1.temp = alloca { i32 }, align 4, !dbg !166
// CHECK:STDOUT:   %.loc7_70.1.temp = alloca { i32 }, align 4, !dbg !167
// CHECK:STDOUT:   %.loc7_71.2.temp = alloca { i32 }, align 4, !dbg !168
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_48.1.temp), !dbg !166
// CHECK:STDOUT:   call void @_CMake.OwningPackage(ptr %.loc7_48.1.temp), !dbg !166
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_70.1.temp), !dbg !167
// CHECK:STDOUT:   call void @_CMake.OwningPackage(ptr %.loc7_70.1.temp), !dbg !167
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_71.2.temp), !dbg !168
// CHECK:STDOUT:   call void @_CGeneric.OwningPackage.e43630e9a6c38c3f(ptr %.loc7_71.2.temp, ptr %.loc7_48.1.temp, ptr %.loc7_70.1.temp), !dbg !168
// CHECK:STDOUT:   call void @"_COp.3e88a108c7e00533:core.Destroy.Core"(ptr %.loc7_71.2.temp), !dbg !168
// CHECK:STDOUT:   call void @"_COp.3e88a108c7e00533:core.Destroy.Core"(ptr %.loc7_70.1.temp), !dbg !167
// CHECK:STDOUT:   call void @"_COp.3e88a108c7e00533:core.Destroy.Core"(ptr %.loc7_48.1.temp), !dbg !166
// CHECK:STDOUT:   ret void, !dbg !169
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CMake.OwningPackage(ptr sret({ i32 }))
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.70db2d92ffddbef7:core.Destroy.Core"(ptr %self) #0 !dbg !170 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !172
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.7ec4ffd904220f7b:core.Destroy.Core"(ptr %self) #0 !dbg !174 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !176
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.3e88a108c7e00533:core.Destroy.Core"(ptr %self) #0 !dbg !178 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !180
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CGeneric.OwningPackage.e43630e9a6c38c3f(ptr sret({ i32 }) %return, ptr %v1, ptr %v2) #0 !dbg !182 {
// CHECK:STDOUT:   %temp = alloca { i32 }, align 4, !dbg !190
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp), !dbg !190
// CHECK:STDOUT:   call void @"_COp.Vector.OwningPackage:MulWith.84588f41d61dafba.Core"(ptr %temp, ptr %v2, i32 2), !dbg !190
// CHECK:STDOUT:   call void @"_COp.Vector.OwningPackage:AddWith.f941828044620367.Core"(ptr %return, ptr %v1, ptr %temp), !dbg !189
// CHECK:STDOUT:   call void @"_COp.3e88a108c7e00533:core.Destroy.Core"(ptr %temp), !dbg !190
// CHECK:STDOUT:   ret void, !dbg !191
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Vector.OwningPackage:MulWith.84588f41d61dafba.Core"(ptr sret({ i32 }), ptr, i32)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Vector.OwningPackage:AddWith.f941828044620367.Core"(ptr sret({ i32 }), ptr, ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CMake.OwningPackage, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @"_COp.3e88a108c7e00533:core.Destroy.Core", { 0, 3, 2, 1 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 3, 2, 1 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!164}
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
// CHECK:STDOUT: !90 = !DIFile(filename: "generic_same_package.carbon", directory: "")
// CHECK:STDOUT: !108 = !{null}
// CHECK:STDOUT: !109 = !DISubroutineType(types: !108)
// CHECK:STDOUT: !163 = !DIFile(filename: "indirect_different_package_via_generic_in_same_package.carbon", directory: "")
// CHECK:STDOUT: !164 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !163, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !165 = distinct !DISubprogram(name: "CallGenericInOwningPackage", linkageName: "_CCallGenericInOwningPackage.OtherPackage", scope: null, file: !163, line: 5, type: !109, spFlags: DISPFlagDefinition, unit: !164)
// CHECK:STDOUT: !166 = !DILocation(line: 7, column: 29, scope: !165)
// CHECK:STDOUT: !167 = !DILocation(line: 7, column: 51, scope: !165)
// CHECK:STDOUT: !168 = !DILocation(line: 7, column: 3, scope: !165)
// CHECK:STDOUT: !169 = !DILocation(line: 5, column: 1, scope: !165)
// CHECK:STDOUT: !170 = distinct !DISubprogram(name: "Op", linkageName: "_COp.70db2d92ffddbef7:core.Destroy.Core", scope: null, file: !163, line: 7, type: !39, spFlags: DISPFlagDefinition, unit: !164, retainedNodes: !173)
// CHECK:STDOUT: !171 = !DILocalVariable(arg: 1, scope: !170, type: !6)
// CHECK:STDOUT: !172 = !DILocation(line: 7, column: 3, scope: !170)
// CHECK:STDOUT: !173 = !{!171}
// CHECK:STDOUT: !174 = distinct !DISubprogram(name: "Op", linkageName: "_COp.7ec4ffd904220f7b:core.Destroy.Core", scope: null, file: !163, line: 7, type: !41, spFlags: DISPFlagDefinition, unit: !164, retainedNodes: !177)
// CHECK:STDOUT: !175 = !DILocalVariable(arg: 1, scope: !174, type: !7)
// CHECK:STDOUT: !176 = !DILocation(line: 7, column: 3, scope: !174)
// CHECK:STDOUT: !177 = !{!175}
// CHECK:STDOUT: !178 = distinct !DISubprogram(name: "Op", linkageName: "_COp.3e88a108c7e00533:core.Destroy.Core", scope: null, file: !163, line: 7, type: !41, spFlags: DISPFlagDefinition, unit: !164, retainedNodes: !181)
// CHECK:STDOUT: !179 = !DILocalVariable(arg: 1, scope: !178, type: !7)
// CHECK:STDOUT: !180 = !DILocation(line: 7, column: 3, scope: !178)
// CHECK:STDOUT: !181 = !{!179}
// CHECK:STDOUT: !182 = distinct !DISubprogram(name: "Generic", linkageName: "_CGeneric.OwningPackage.e43630e9a6c38c3f", scope: null, file: !90, line: 5, type: !9, spFlags: DISPFlagDefinition, unit: !164, retainedNodes: !192)
// CHECK:STDOUT: !183 = !DILocalVariable(arg: 1, scope: !182, type: !7)
// CHECK:STDOUT: !184 = !DILocalVariable(arg: 2, scope: !182, type: !7)
// CHECK:STDOUT: !189 = !DILocation(line: 6, column: 10, scope: !182)
// CHECK:STDOUT: !190 = !DILocation(line: 6, column: 15, scope: !182)
// CHECK:STDOUT: !191 = !DILocation(line: 6, column: 3, scope: !182)
// CHECK:STDOUT: !192 = !{!183, !184}
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
// CHECK:STDOUT: define weak_odr void @"_COp.70db2d92ffddbef7:core.Destroy.Core"(ptr %self) #0 !dbg !195 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !197
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.7ec4ffd904220f7b:core.Destroy.Core"(ptr %self) #0 !dbg !199 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !201
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.3e88a108c7e00533:core.Destroy.Core"(ptr %self) #0 !dbg !203 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !205
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CMake.OtherPackage(ptr sret({ i32 }))
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!194}
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
// CHECK:STDOUT: !193 = !DIFile(filename: "generic_different_package.carbon", directory: "")
// CHECK:STDOUT: !194 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !193, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !195 = distinct !DISubprogram(name: "Op", linkageName: "_COp.70db2d92ffddbef7:core.Destroy.Core", scope: null, file: !193, line: 6, type: !39, spFlags: DISPFlagDefinition, unit: !194, retainedNodes: !198)
// CHECK:STDOUT: !196 = !DILocalVariable(arg: 1, scope: !195, type: !6)
// CHECK:STDOUT: !197 = !DILocation(line: 6, column: 15, scope: !195)
// CHECK:STDOUT: !198 = !{!196}
// CHECK:STDOUT: !199 = distinct !DISubprogram(name: "Op", linkageName: "_COp.7ec4ffd904220f7b:core.Destroy.Core", scope: null, file: !193, line: 6, type: !41, spFlags: DISPFlagDefinition, unit: !194, retainedNodes: !202)
// CHECK:STDOUT: !200 = !DILocalVariable(arg: 1, scope: !199, type: !7)
// CHECK:STDOUT: !201 = !DILocation(line: 6, column: 15, scope: !199)
// CHECK:STDOUT: !202 = !{!200}
// CHECK:STDOUT: !203 = distinct !DISubprogram(name: "Op", linkageName: "_COp.3e88a108c7e00533:core.Destroy.Core", scope: null, file: !193, line: 6, type: !41, spFlags: DISPFlagDefinition, unit: !194, retainedNodes: !206)
// CHECK:STDOUT: !204 = !DILocalVariable(arg: 1, scope: !203, type: !7)
// CHECK:STDOUT: !205 = !DILocation(line: 6, column: 15, scope: !203)
// CHECK:STDOUT: !206 = !{!204}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'indirect_different_package_via_generic_in_different_package.carbon'
// CHECK:STDOUT: source_filename = "indirect_different_package_via_generic_in_different_package.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCallGeneric.OtherPackage() #0 !dbg !209 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_20.1.temp = alloca { i32 }, align 4, !dbg !210
// CHECK:STDOUT:   %.loc7_28.1.temp = alloca { i32 }, align 4, !dbg !211
// CHECK:STDOUT:   %.loc7_29.2.temp = alloca { i32 }, align 4, !dbg !212
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_20.1.temp), !dbg !210
// CHECK:STDOUT:   call void @_CMake.OtherPackage(ptr %.loc7_20.1.temp), !dbg !210
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_28.1.temp), !dbg !211
// CHECK:STDOUT:   call void @_CMake.OtherPackage(ptr %.loc7_28.1.temp), !dbg !211
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_29.2.temp), !dbg !212
// CHECK:STDOUT:   call void @_CGeneric.OtherPackage.e43630e9a6c38c3f(ptr %.loc7_29.2.temp, ptr %.loc7_20.1.temp, ptr %.loc7_28.1.temp), !dbg !212
// CHECK:STDOUT:   call void @"_COp.3e88a108c7e00533:core.Destroy.Core"(ptr %.loc7_29.2.temp), !dbg !212
// CHECK:STDOUT:   call void @"_COp.3e88a108c7e00533:core.Destroy.Core"(ptr %.loc7_28.1.temp), !dbg !211
// CHECK:STDOUT:   call void @"_COp.3e88a108c7e00533:core.Destroy.Core"(ptr %.loc7_20.1.temp), !dbg !210
// CHECK:STDOUT:   ret void, !dbg !213
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CMake.OtherPackage(ptr sret({ i32 }))
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.70db2d92ffddbef7:core.Destroy.Core"(ptr %self) #0 !dbg !214 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !216
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.7ec4ffd904220f7b:core.Destroy.Core"(ptr %self) #0 !dbg !218 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !220
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.3e88a108c7e00533:core.Destroy.Core"(ptr %self) #0 !dbg !222 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !224
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CGeneric.OtherPackage.e43630e9a6c38c3f(ptr sret({ i32 }) %return, ptr %v1, ptr %v2) #0 !dbg !226 {
// CHECK:STDOUT:   %temp = alloca { i32 }, align 4, !dbg !234
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp), !dbg !234
// CHECK:STDOUT:   call void @"_COp.Vector.OwningPackage:MulWith.84588f41d61dafba.Core"(ptr %temp, ptr %v2, i32 2), !dbg !234
// CHECK:STDOUT:   call void @"_COp.Vector.OwningPackage:AddWith.f941828044620367.Core"(ptr %return, ptr %v1, ptr %temp), !dbg !233
// CHECK:STDOUT:   call void @"_COp.3e88a108c7e00533:core.Destroy.Core"(ptr %temp), !dbg !234
// CHECK:STDOUT:   ret void, !dbg !235
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Vector.OwningPackage:MulWith.84588f41d61dafba.Core"(ptr sret({ i32 }), ptr, i32)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Vector.OwningPackage:AddWith.f941828044620367.Core"(ptr sret({ i32 }), ptr, ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CMake.OtherPackage, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @"_COp.3e88a108c7e00533:core.Destroy.Core", { 0, 3, 2, 1 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 3, 2, 1 }
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
// CHECK:STDOUT: !38 = !{null, !6}
// CHECK:STDOUT: !39 = !DISubroutineType(types: !38)
// CHECK:STDOUT: !40 = !{null, !7}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !108 = !{null}
// CHECK:STDOUT: !109 = !DISubroutineType(types: !108)
// CHECK:STDOUT: !193 = !DIFile(filename: "generic_different_package.carbon", directory: "")
// CHECK:STDOUT: !207 = !DIFile(filename: "indirect_different_package_via_generic_in_different_package.carbon", directory: "")
// CHECK:STDOUT: !208 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !207, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !209 = distinct !DISubprogram(name: "CallGeneric", linkageName: "_CCallGeneric.OtherPackage", scope: null, file: !207, line: 5, type: !109, spFlags: DISPFlagDefinition, unit: !208)
// CHECK:STDOUT: !210 = !DILocation(line: 7, column: 15, scope: !209)
// CHECK:STDOUT: !211 = !DILocation(line: 7, column: 23, scope: !209)
// CHECK:STDOUT: !212 = !DILocation(line: 7, column: 3, scope: !209)
// CHECK:STDOUT: !213 = !DILocation(line: 5, column: 1, scope: !209)
// CHECK:STDOUT: !214 = distinct !DISubprogram(name: "Op", linkageName: "_COp.70db2d92ffddbef7:core.Destroy.Core", scope: null, file: !207, line: 7, type: !39, spFlags: DISPFlagDefinition, unit: !208, retainedNodes: !217)
// CHECK:STDOUT: !215 = !DILocalVariable(arg: 1, scope: !214, type: !6)
// CHECK:STDOUT: !216 = !DILocation(line: 7, column: 3, scope: !214)
// CHECK:STDOUT: !217 = !{!215}
// CHECK:STDOUT: !218 = distinct !DISubprogram(name: "Op", linkageName: "_COp.7ec4ffd904220f7b:core.Destroy.Core", scope: null, file: !207, line: 7, type: !41, spFlags: DISPFlagDefinition, unit: !208, retainedNodes: !221)
// CHECK:STDOUT: !219 = !DILocalVariable(arg: 1, scope: !218, type: !7)
// CHECK:STDOUT: !220 = !DILocation(line: 7, column: 3, scope: !218)
// CHECK:STDOUT: !221 = !{!219}
// CHECK:STDOUT: !222 = distinct !DISubprogram(name: "Op", linkageName: "_COp.3e88a108c7e00533:core.Destroy.Core", scope: null, file: !207, line: 7, type: !41, spFlags: DISPFlagDefinition, unit: !208, retainedNodes: !225)
// CHECK:STDOUT: !223 = !DILocalVariable(arg: 1, scope: !222, type: !7)
// CHECK:STDOUT: !224 = !DILocation(line: 7, column: 3, scope: !222)
// CHECK:STDOUT: !225 = !{!223}
// CHECK:STDOUT: !226 = distinct !DISubprogram(name: "Generic", linkageName: "_CGeneric.OtherPackage.e43630e9a6c38c3f", scope: null, file: !193, line: 5, type: !9, spFlags: DISPFlagDefinition, unit: !208, retainedNodes: !236)
// CHECK:STDOUT: !227 = !DILocalVariable(arg: 1, scope: !226, type: !7)
// CHECK:STDOUT: !228 = !DILocalVariable(arg: 2, scope: !226, type: !7)
// CHECK:STDOUT: !233 = !DILocation(line: 6, column: 10, scope: !226)
// CHECK:STDOUT: !234 = !DILocation(line: 6, column: 15, scope: !226)
// CHECK:STDOUT: !235 = !DILocation(line: 6, column: 3, scope: !226)
// CHECK:STDOUT: !236 = !{!227, !228}
// CHECK:STDOUT:
