// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/nullptr.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/nullptr.carbon

// --- simple.carbon

import Cpp inline '''
using nullptr_t = decltype(nullptr);

void TakePtr(int *p);
void TakeNullptr(nullptr_t n);

nullptr_t ReturnNullptr();
''';

fn PassNullptr() {
  Cpp.TakePtr(Cpp.nullptr);

  Cpp.TakeNullptr(Cpp.nullptr);
}

fn ReturnNullptr() -> Cpp.nullptr_t {
  return Cpp.nullptr;
}

fn ReturnNullptrCopy() -> Cpp.nullptr_t {
  var a: Cpp.nullptr_t = Cpp.nullptr;
  return a;
}

fn ReturnConvertedNullptrIndirectly() -> Core.Optional(i32*) {
  var a: Cpp.nullptr_t = Cpp.ReturnNullptr();
  return a;
}

fn ReturnConvertedNullptrDirectly() -> Core.Optional(i32*) {
  return Cpp.ReturnNullptr();
}

fn ConvertNullptrConstant() -> Core.Optional(i32*) {
  return Cpp.nullptr;
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'simple.carbon'
// CHECK:STDOUT: source_filename = "simple.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassNullptr.Main() #0 !dbg !16 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc12_18.2.temp = alloca ptr, align 8, !dbg !17
// CHECK:STDOUT:   %.loc14_22.1.temp = alloca ptr, align 8, !dbg !19
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.NullptrT.CppCompat.Core:ImplicitAs.49279a15ecfc119b.Core.84588f41d61dafba"(ptr poison), !dbg !17
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc12_18.2.temp), !dbg !17
// CHECK:STDOUT:   store ptr %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.call, ptr %.loc12_18.2.temp, align 8, !dbg !17
// CHECK:STDOUT:   %.loc12_18.4 = load ptr, ptr %.loc12_18.2.temp, align 8, !dbg !17
// CHECK:STDOUT:   call void @_Z7TakePtrPi(ptr %.loc12_18.4), !dbg !18
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.2696e2361851b1c6(ptr %.loc12_18.2.temp), !dbg !17
// CHECK:STDOUT:   %Cpp.nullptr_t.as.Copy.impl.Op.call = call ptr @"_COp.NullptrT.CppCompat.Core:Copy.Core"(ptr poison), !dbg !19
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc14_22.1.temp), !dbg !19
// CHECK:STDOUT:   store ptr %Cpp.nullptr_t.as.Copy.impl.Op.call, ptr %.loc14_22.1.temp, align 8, !dbg !19
// CHECK:STDOUT:   call void @_Z11TakeNullptrDn.carbon_thunk._(ptr %.loc14_22.1.temp), !dbg !20
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.dda34febf79fada3(ptr %.loc14_22.1.temp), !dbg !19
// CHECK:STDOUT:   ret void, !dbg !21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z7TakePtrPi(ptr noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.0f866e2c9731ba7d:core.Destroy.Core"(ptr %self) #0 !dbg !22 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !24
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.13a43c95de1d136f:core.Destroy.Core"(ptr %self) #0 !dbg !26 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.5b309895020f658e:core.Destroy.Core"(ptr %self) #0 !dbg !30 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z11TakeNullptrDn.carbon_thunk._(ptr noundef %n) #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %n.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %n, ptr %n.addr, align 8, !tbaa !111
// CHECK:STDOUT:   %0 = load ptr, ptr %n.addr, align 8, !tbaa !111
// CHECK:STDOUT:   call void @_Z11TakeNullptrDn(ptr null)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare ptr @"_COp.NullptrT.CppCompat.Core:Copy.Core"(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.a7d1b77eb1279818:core.Destroy.Core"(ptr %self) #0 !dbg !34 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.1ac0350a118ac030:core.Destroy.Core"(ptr %self) #0 !dbg !38 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.826ebaace97974ae:core.Destroy.Core"(ptr %self) #0 !dbg !42 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CReturnNullptr.Main() #0 !dbg !46 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Cpp.nullptr_t.as.Copy.impl.Op.call = call ptr @"_COp.NullptrT.CppCompat.Core:Copy.Core"(ptr poison), !dbg !49
// CHECK:STDOUT:   ret ptr %Cpp.nullptr_t.as.Copy.impl.Op.call, !dbg !50
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CReturnNullptrCopy.Main() #0 !dbg !51 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca ptr, align 8, !dbg !54
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !54
// CHECK:STDOUT:   %Cpp.nullptr_t.as.Copy.impl.Op.call.loc22 = call ptr @"_COp.NullptrT.CppCompat.Core:Copy.Core"(ptr poison), !dbg !55
// CHECK:STDOUT:   store ptr %Cpp.nullptr_t.as.Copy.impl.Op.call.loc22, ptr %a.var, align 8, !dbg !54
// CHECK:STDOUT:   %.loc23 = load ptr, ptr %a.var, align 8, !dbg !57
// CHECK:STDOUT:   %Cpp.nullptr_t.as.Copy.impl.Op.call.loc23 = call ptr @"_COp.NullptrT.CppCompat.Core:Copy.Core"(ptr %.loc23), !dbg !57
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.dda34febf79fada3(ptr %a.var), !dbg !54
// CHECK:STDOUT:   ret ptr %Cpp.nullptr_t.as.Copy.impl.Op.call.loc23, !dbg !58
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CReturnConvertedNullptrIndirectly.Main() #0 !dbg !59 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca ptr, align 8, !dbg !62
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !62
// CHECK:STDOUT:   call void @_Z13ReturnNullptrv.carbon_thunk.(ptr %a.var), !dbg !63
// CHECK:STDOUT:   %.loc28_10 = load ptr, ptr %a.var, align 8, !dbg !65
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.NullptrT.CppCompat.Core:ImplicitAs.49279a15ecfc119b.Core.84588f41d61dafba"(ptr %.loc28_10), !dbg !66
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.dda34febf79fada3(ptr %a.var), !dbg !62
// CHECK:STDOUT:   ret ptr %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.call, !dbg !66
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z13ReturnNullptrv.carbon_thunk.(ptr noundef %return) #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !111
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !111
// CHECK:STDOUT:   %call = call ptr @_Z13ReturnNullptrv()
// CHECK:STDOUT:   store ptr %call, ptr %0, align 8, !tbaa !112
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CReturnConvertedNullptrDirectly.Main() #0 !dbg !67 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc32_28.1.temp = alloca ptr, align 8, !dbg !70
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc32_28.1.temp), !dbg !70
// CHECK:STDOUT:   call void @_Z13ReturnNullptrv.carbon_thunk.(ptr %.loc32_28.1.temp), !dbg !70
// CHECK:STDOUT:   %.loc32_28.4 = load ptr, ptr %.loc32_28.1.temp, align 8, !dbg !70
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.NullptrT.CppCompat.Core:ImplicitAs.49279a15ecfc119b.Core.84588f41d61dafba"(ptr %.loc32_28.4), !dbg !71
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.dda34febf79fada3(ptr %.loc32_28.1.temp), !dbg !70
// CHECK:STDOUT:   ret ptr %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.call, !dbg !71
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CConvertNullptrConstant.Main() #0 !dbg !72 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.NullptrT.CppCompat.Core:ImplicitAs.49279a15ecfc119b.Core.84588f41d61dafba"(ptr poison), !dbg !75
// CHECK:STDOUT:   ret ptr %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.call, !dbg !75
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.NullptrT.CppCompat.Core:ImplicitAs.49279a15ecfc119b.Core.84588f41d61dafba"(ptr %self) #0 !dbg !77 {
// CHECK:STDOUT:   %1 = call ptr @_CNone.Optional.Core.6e9e957697159b53(), !dbg !82
// CHECK:STDOUT:   ret ptr %1, !dbg !83
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.2696e2361851b1c6(ptr %self) #0 !dbg !86 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.5b309895020f658e:core.Destroy.Core"(ptr %self), !dbg !90
// CHECK:STDOUT:   ret void, !dbg !91
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.dda34febf79fada3(ptr %self) #0 !dbg !93 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.826ebaace97974ae:core.Destroy.Core"(ptr %self), !dbg !97
// CHECK:STDOUT:   ret void, !dbg !98
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare ptr @_CMake.NullptrT.CppCompat.Core()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CNone.Optional.Core.6e9e957697159b53() #0 !dbg !101 {
// CHECK:STDOUT:   ret ptr null, !dbg !105
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z11TakeNullptrDn(ptr) #1
// CHECK:STDOUT:
// CHECK:STDOUT: declare ptr @_Z13ReturnNullptrv() #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @"_CConvert.NullptrT.CppCompat.Core:ImplicitAs.49279a15ecfc119b.Core.84588f41d61dafba", { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Destroy.Core.dda34febf79fada3, { 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.module.flags = !{!113, !114, !115, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!118}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "simple.carbon", directory: "")
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !7 = !{null}
// CHECK:STDOUT: !8 = !DISubroutineType(types: !7)
// CHECK:STDOUT: !9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !10 = !{null, !9}
// CHECK:STDOUT: !11 = !DISubroutineType(types: !10)
// CHECK:STDOUT: !12 = !{!9, !9}
// CHECK:STDOUT: !13 = !DISubroutineType(types: !12)
// CHECK:STDOUT: !14 = !{!9}
// CHECK:STDOUT: !15 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !16 = distinct !DISubprogram(name: "PassNullptr", linkageName: "_CPassNullptr.Main", scope: null, file: !4, line: 11, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !17 = !DILocation(line: 12, column: 15, scope: !16)
// CHECK:STDOUT: !18 = !DILocation(line: 12, column: 3, scope: !16)
// CHECK:STDOUT: !19 = !DILocation(line: 14, column: 19, scope: !16)
// CHECK:STDOUT: !20 = !DILocation(line: 14, column: 3, scope: !16)
// CHECK:STDOUT: !21 = !DILocation(line: 11, column: 1, scope: !16)
// CHECK:STDOUT: !22 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.0f866e2c9731ba7d:core.Destroy.Core", scope: null, file: !4, line: 12, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !25)
// CHECK:STDOUT: !23 = !DILocalVariable(arg: 1, scope: !22, type: !9)
// CHECK:STDOUT: !24 = !DILocation(line: 12, column: 15, scope: !22)
// CHECK:STDOUT: !25 = !{!23}
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.13a43c95de1d136f:core.Destroy.Core", scope: null, file: !4, line: 12, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !29)
// CHECK:STDOUT: !27 = !DILocalVariable(arg: 1, scope: !26, type: !9)
// CHECK:STDOUT: !28 = !DILocation(line: 12, column: 15, scope: !26)
// CHECK:STDOUT: !29 = !{!27}
// CHECK:STDOUT: !30 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.5b309895020f658e:core.Destroy.Core", scope: null, file: !4, line: 12, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !33)
// CHECK:STDOUT: !31 = !DILocalVariable(arg: 1, scope: !30, type: !9)
// CHECK:STDOUT: !32 = !DILocation(line: 12, column: 15, scope: !30)
// CHECK:STDOUT: !33 = !{!31}
// CHECK:STDOUT: !34 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.a7d1b77eb1279818:core.Destroy.Core", scope: null, file: !4, line: 14, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !37)
// CHECK:STDOUT: !35 = !DILocalVariable(arg: 1, scope: !34, type: !9)
// CHECK:STDOUT: !36 = !DILocation(line: 14, column: 19, scope: !34)
// CHECK:STDOUT: !37 = !{!35}
// CHECK:STDOUT: !38 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.1ac0350a118ac030:core.Destroy.Core", scope: null, file: !4, line: 14, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !41)
// CHECK:STDOUT: !39 = !DILocalVariable(arg: 1, scope: !38, type: !9)
// CHECK:STDOUT: !40 = !DILocation(line: 14, column: 19, scope: !38)
// CHECK:STDOUT: !41 = !{!39}
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.826ebaace97974ae:core.Destroy.Core", scope: null, file: !4, line: 14, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !45)
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 1, scope: !42, type: !9)
// CHECK:STDOUT: !44 = !DILocation(line: 14, column: 19, scope: !42)
// CHECK:STDOUT: !45 = !{!43}
// CHECK:STDOUT: !46 = distinct !DISubprogram(name: "ReturnNullptr", linkageName: "_CReturnNullptr.Main", scope: null, file: !4, line: 17, type: !15, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !49 = !DILocation(line: 18, column: 10, scope: !46)
// CHECK:STDOUT: !50 = !DILocation(line: 18, column: 3, scope: !46)
// CHECK:STDOUT: !51 = distinct !DISubprogram(name: "ReturnNullptrCopy", linkageName: "_CReturnNullptrCopy.Main", scope: null, file: !4, line: 21, type: !15, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !54 = !DILocation(line: 22, column: 3, scope: !51)
// CHECK:STDOUT: !55 = !DILocation(line: 22, column: 26, scope: !51)
// CHECK:STDOUT: !57 = !DILocation(line: 23, column: 10, scope: !51)
// CHECK:STDOUT: !58 = !DILocation(line: 23, column: 3, scope: !51)
// CHECK:STDOUT: !59 = distinct !DISubprogram(name: "ReturnConvertedNullptrIndirectly", linkageName: "_CReturnConvertedNullptrIndirectly.Main", scope: null, file: !4, line: 26, type: !15, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !62 = !DILocation(line: 27, column: 3, scope: !59)
// CHECK:STDOUT: !63 = !DILocation(line: 27, column: 26, scope: !59)
// CHECK:STDOUT: !65 = !DILocation(line: 28, column: 10, scope: !59)
// CHECK:STDOUT: !66 = !DILocation(line: 28, column: 3, scope: !59)
// CHECK:STDOUT: !67 = distinct !DISubprogram(name: "ReturnConvertedNullptrDirectly", linkageName: "_CReturnConvertedNullptrDirectly.Main", scope: null, file: !4, line: 31, type: !15, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !70 = !DILocation(line: 32, column: 10, scope: !67)
// CHECK:STDOUT: !71 = !DILocation(line: 32, column: 3, scope: !67)
// CHECK:STDOUT: !72 = distinct !DISubprogram(name: "ConvertNullptrConstant", linkageName: "_CConvertNullptrConstant.Main", scope: null, file: !4, line: 35, type: !15, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !75 = !DILocation(line: 36, column: 3, scope: !72)
// CHECK:STDOUT: !76 = !DIFile(filename: "{{.*}}/prelude/types/cpp/nullptr.carbon", directory: "")
// CHECK:STDOUT: !77 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.NullptrT.CppCompat.Core:ImplicitAs.49279a15ecfc119b.Core.84588f41d61dafba", scope: null, file: !76, line: 51, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !84)
// CHECK:STDOUT: !78 = !DILocalVariable(arg: 1, scope: !77, type: !9)
// CHECK:STDOUT: !82 = !DILocation(line: 52, column: 14, scope: !77)
// CHECK:STDOUT: !83 = !DILocation(line: 52, column: 7, scope: !77)
// CHECK:STDOUT: !84 = !{!78}
// CHECK:STDOUT: !85 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !86 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.2696e2361851b1c6", scope: null, file: !85, line: 26, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !92)
// CHECK:STDOUT: !87 = !DILocalVariable(arg: 1, scope: !86, type: !9)
// CHECK:STDOUT: !90 = !DILocation(line: 28, column: 5, scope: !86)
// CHECK:STDOUT: !91 = !DILocation(line: 26, column: 3, scope: !86)
// CHECK:STDOUT: !92 = !{!87}
// CHECK:STDOUT: !93 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.dda34febf79fada3", scope: null, file: !85, line: 26, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !99)
// CHECK:STDOUT: !94 = !DILocalVariable(arg: 1, scope: !93, type: !9)
// CHECK:STDOUT: !97 = !DILocation(line: 28, column: 5, scope: !93)
// CHECK:STDOUT: !98 = !DILocation(line: 26, column: 3, scope: !93)
// CHECK:STDOUT: !99 = !{!94}
// CHECK:STDOUT: !100 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !101 = distinct !DISubprogram(name: "None", linkageName: "_CNone.Optional.Core.6e9e957697159b53", scope: null, file: !100, line: 30, type: !15, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !105 = !DILocation(line: 31, column: 5, scope: !101)
// CHECK:STDOUT: !106 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !107 = !{!"omnipotent char", !106, i64 0}
// CHECK:STDOUT: !108 = !{!"std::nullptr_t", !107, i64 0}
// CHECK:STDOUT: !109 = !{!"any pointer", !107, i64 0}
// CHECK:STDOUT: !110 = !{!"p1 std::nullptr_t", !109, i64 0}
// CHECK:STDOUT: !111 = !{!110, !110, i64 0}
// CHECK:STDOUT: !112 = !{!108, !108, i64 0}
// CHECK:STDOUT: !113 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !114 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !115 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !116 = !{!"int", !107, i64 0}
// CHECK:STDOUT: !117 = !{!"__libc_errno", !116, i64 0}
// CHECK:STDOUT: !118 = !{!117, !116, i64 0}
// CHECK:STDOUT:
