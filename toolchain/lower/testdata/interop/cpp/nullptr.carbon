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
// CHECK:STDOUT:   call void @"_CSelfDestruct.5b309895020f658e:core.Destroy.Core"(ptr %.loc12_18.2.temp), !dbg !17
// CHECK:STDOUT:   %Cpp.nullptr_t.as.Copy.impl.Op.call = call ptr @"_COp.NullptrT.CppCompat.Core:Copy.Core"(ptr poison), !dbg !19
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc14_22.1.temp), !dbg !19
// CHECK:STDOUT:   store ptr %Cpp.nullptr_t.as.Copy.impl.Op.call, ptr %.loc14_22.1.temp, align 8, !dbg !19
// CHECK:STDOUT:   call void @_Z11TakeNullptrDn.carbon_thunk._(ptr %.loc14_22.1.temp), !dbg !20
// CHECK:STDOUT:   call void @"_CSelfDestruct.826ebaace97974ae:core.Destroy.Core"(ptr %.loc14_22.1.temp), !dbg !19
// CHECK:STDOUT:   ret void, !dbg !21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_CSelfDestruct.00ac8ef5084ea134:core.Destroy.Core"(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.d088c7eb746488be:core.Destroy.Core"(ptr %self) #0 !dbg !23 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z7TakePtrPi(ptr noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.0f866e2c9731ba7d:core.Destroy.Core"(ptr %self) #0 !dbg !27 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.0f866e2c9731ba7d:core.Destroy.Core"(ptr %self) #0 !dbg !31 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.0f866e2c9731ba7d:core.Destroy.Core"(ptr %self), !dbg !33
// CHECK:STDOUT:   ret void, !dbg !33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.729fca203337f92a:core.Destroy.Core"(ptr %self) #0 !dbg !35 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.729fca203337f92a:core.Destroy.Core"(ptr %self) #0 !dbg !39 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.729fca203337f92a:core.Destroy.Core"(ptr %self), !dbg !41
// CHECK:STDOUT:   ret void, !dbg !41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.5b309895020f658e:core.Destroy.Core"(ptr %self) #0 !dbg !43 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.5b309895020f658e:core.Destroy.Core"(ptr %self) #0 !dbg !47 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.5b309895020f658e:core.Destroy.Core"(ptr %self), !dbg !49
// CHECK:STDOUT:   ret void, !dbg !49
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z11TakeNullptrDn.carbon_thunk._(ptr noundef %n) #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %n.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %n, ptr %n.addr, align 8, !tbaa !124
// CHECK:STDOUT:   %0 = load ptr, ptr %n.addr, align 8, !tbaa !124
// CHECK:STDOUT:   call void @_Z11TakeNullptrDn(ptr null)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare ptr @"_COp.NullptrT.CppCompat.Core:Copy.Core"(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.a7d1b77eb1279818:core.Destroy.Core"(ptr %self) #0 !dbg !51 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.a7d1b77eb1279818:core.Destroy.Core"(ptr %self) #0 !dbg !55 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.a7d1b77eb1279818:core.Destroy.Core"(ptr %self), !dbg !57
// CHECK:STDOUT:   ret void, !dbg !57
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.174120d4d03791b0:core.Destroy.Core"(ptr %self) #0 !dbg !59 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !61
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.174120d4d03791b0:core.Destroy.Core"(ptr %self) #0 !dbg !63 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.174120d4d03791b0:core.Destroy.Core"(ptr %self), !dbg !65
// CHECK:STDOUT:   ret void, !dbg !65
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.826ebaace97974ae:core.Destroy.Core"(ptr %self) #0 !dbg !67 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !69
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.826ebaace97974ae:core.Destroy.Core"(ptr %self) #0 !dbg !71 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.826ebaace97974ae:core.Destroy.Core"(ptr %self), !dbg !73
// CHECK:STDOUT:   ret void, !dbg !73
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CReturnNullptr.Main() #0 !dbg !75 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Cpp.nullptr_t.as.Copy.impl.Op.call = call ptr @"_COp.NullptrT.CppCompat.Core:Copy.Core"(ptr poison), !dbg !78
// CHECK:STDOUT:   ret ptr %Cpp.nullptr_t.as.Copy.impl.Op.call, !dbg !79
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CReturnNullptrCopy.Main() #0 !dbg !80 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca ptr, align 8, !dbg !83
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !83
// CHECK:STDOUT:   %Cpp.nullptr_t.as.Copy.impl.Op.call.loc22 = call ptr @"_COp.NullptrT.CppCompat.Core:Copy.Core"(ptr poison), !dbg !84
// CHECK:STDOUT:   store ptr %Cpp.nullptr_t.as.Copy.impl.Op.call.loc22, ptr %a.var, align 8, !dbg !83
// CHECK:STDOUT:   %.loc23 = load ptr, ptr %a.var, align 8, !dbg !86
// CHECK:STDOUT:   %Cpp.nullptr_t.as.Copy.impl.Op.call.loc23 = call ptr @"_COp.NullptrT.CppCompat.Core:Copy.Core"(ptr %.loc23), !dbg !86
// CHECK:STDOUT:   call void @"_CSelfDestruct.826ebaace97974ae:core.Destroy.Core"(ptr %a.var), !dbg !83
// CHECK:STDOUT:   ret ptr %Cpp.nullptr_t.as.Copy.impl.Op.call.loc23, !dbg !87
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CReturnConvertedNullptrIndirectly.Main() #0 !dbg !88 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca ptr, align 8, !dbg !91
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !91
// CHECK:STDOUT:   call void @_Z13ReturnNullptrv.carbon_thunk.(ptr %a.var), !dbg !92
// CHECK:STDOUT:   %.loc28_10 = load ptr, ptr %a.var, align 8, !dbg !94
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.NullptrT.CppCompat.Core:ImplicitAs.49279a15ecfc119b.Core.84588f41d61dafba"(ptr %.loc28_10), !dbg !95
// CHECK:STDOUT:   call void @"_CSelfDestruct.826ebaace97974ae:core.Destroy.Core"(ptr %a.var), !dbg !91
// CHECK:STDOUT:   ret ptr %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.call, !dbg !95
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z13ReturnNullptrv.carbon_thunk.(ptr noundef %return) #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !124
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !124
// CHECK:STDOUT:   %call = call ptr @_Z13ReturnNullptrv()
// CHECK:STDOUT:   store ptr %call, ptr %0, align 8, !tbaa !125
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CReturnConvertedNullptrDirectly.Main() #0 !dbg !96 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc32_28.1.temp = alloca ptr, align 8, !dbg !99
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc32_28.1.temp), !dbg !99
// CHECK:STDOUT:   call void @_Z13ReturnNullptrv.carbon_thunk.(ptr %.loc32_28.1.temp), !dbg !99
// CHECK:STDOUT:   %.loc32_28.4 = load ptr, ptr %.loc32_28.1.temp, align 8, !dbg !99
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.NullptrT.CppCompat.Core:ImplicitAs.49279a15ecfc119b.Core.84588f41d61dafba"(ptr %.loc32_28.4), !dbg !100
// CHECK:STDOUT:   call void @"_CSelfDestruct.826ebaace97974ae:core.Destroy.Core"(ptr %.loc32_28.1.temp), !dbg !99
// CHECK:STDOUT:   ret ptr %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.call, !dbg !100
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CConvertNullptrConstant.Main() #0 !dbg !101 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.NullptrT.CppCompat.Core:ImplicitAs.49279a15ecfc119b.Core.84588f41d61dafba"(ptr poison), !dbg !104
// CHECK:STDOUT:   ret ptr %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.call, !dbg !104
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.NullptrT.CppCompat.Core:ImplicitAs.49279a15ecfc119b.Core.84588f41d61dafba"(ptr %self) #0 !dbg !106 {
// CHECK:STDOUT:   %1 = call ptr @_CNone.Optional.Core.6e9e957697159b53(), !dbg !111
// CHECK:STDOUT:   ret ptr %1, !dbg !112
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: declare ptr @_CMake.NullptrT.CppCompat.Core()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CNone.Optional.Core.6e9e957697159b53() #0 !dbg !114 {
// CHECK:STDOUT:   ret ptr null, !dbg !118
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z11TakeNullptrDn(ptr) #1
// CHECK:STDOUT:
// CHECK:STDOUT: declare ptr @_Z13ReturnNullptrv() #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @"_CConvert.NullptrT.CppCompat.Core:ImplicitAs.49279a15ecfc119b.Core.84588f41d61dafba", { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 4, 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.module.flags = !{!126, !127, !128, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!131}
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
// CHECK:STDOUT: !22 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !23 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.d088c7eb746488be:core.Destroy.Core", scope: null, file: !22, line: 164, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !26)
// CHECK:STDOUT: !24 = !DILocalVariable(arg: 1, scope: !23, type: !9)
// CHECK:STDOUT: !25 = !DILocation(line: 164, column: 19, scope: !23)
// CHECK:STDOUT: !26 = !{!24}
// CHECK:STDOUT: !27 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.0f866e2c9731ba7d:core.Destroy.Core", scope: null, file: !4, line: 12, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !30)
// CHECK:STDOUT: !28 = !DILocalVariable(arg: 1, scope: !27, type: !9)
// CHECK:STDOUT: !29 = !DILocation(line: 12, column: 15, scope: !27)
// CHECK:STDOUT: !30 = !{!28}
// CHECK:STDOUT: !31 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.0f866e2c9731ba7d:core.Destroy.Core", scope: null, file: !4, line: 12, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !34)
// CHECK:STDOUT: !32 = !DILocalVariable(arg: 1, scope: !31, type: !9)
// CHECK:STDOUT: !33 = !DILocation(line: 12, column: 15, scope: !31)
// CHECK:STDOUT: !34 = !{!32}
// CHECK:STDOUT: !35 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.729fca203337f92a:core.Destroy.Core", scope: null, file: !4, line: 12, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !38)
// CHECK:STDOUT: !36 = !DILocalVariable(arg: 1, scope: !35, type: !9)
// CHECK:STDOUT: !37 = !DILocation(line: 12, column: 15, scope: !35)
// CHECK:STDOUT: !38 = !{!36}
// CHECK:STDOUT: !39 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.729fca203337f92a:core.Destroy.Core", scope: null, file: !4, line: 12, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !42)
// CHECK:STDOUT: !40 = !DILocalVariable(arg: 1, scope: !39, type: !9)
// CHECK:STDOUT: !41 = !DILocation(line: 12, column: 15, scope: !39)
// CHECK:STDOUT: !42 = !{!40}
// CHECK:STDOUT: !43 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.5b309895020f658e:core.Destroy.Core", scope: null, file: !4, line: 12, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !46)
// CHECK:STDOUT: !44 = !DILocalVariable(arg: 1, scope: !43, type: !9)
// CHECK:STDOUT: !45 = !DILocation(line: 12, column: 15, scope: !43)
// CHECK:STDOUT: !46 = !{!44}
// CHECK:STDOUT: !47 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.5b309895020f658e:core.Destroy.Core", scope: null, file: !4, line: 12, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !50)
// CHECK:STDOUT: !48 = !DILocalVariable(arg: 1, scope: !47, type: !9)
// CHECK:STDOUT: !49 = !DILocation(line: 12, column: 15, scope: !47)
// CHECK:STDOUT: !50 = !{!48}
// CHECK:STDOUT: !51 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.a7d1b77eb1279818:core.Destroy.Core", scope: null, file: !4, line: 14, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !54)
// CHECK:STDOUT: !52 = !DILocalVariable(arg: 1, scope: !51, type: !9)
// CHECK:STDOUT: !53 = !DILocation(line: 14, column: 19, scope: !51)
// CHECK:STDOUT: !54 = !{!52}
// CHECK:STDOUT: !55 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.a7d1b77eb1279818:core.Destroy.Core", scope: null, file: !4, line: 14, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !58)
// CHECK:STDOUT: !56 = !DILocalVariable(arg: 1, scope: !55, type: !9)
// CHECK:STDOUT: !57 = !DILocation(line: 14, column: 19, scope: !55)
// CHECK:STDOUT: !58 = !{!56}
// CHECK:STDOUT: !59 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.174120d4d03791b0:core.Destroy.Core", scope: null, file: !4, line: 14, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !62)
// CHECK:STDOUT: !60 = !DILocalVariable(arg: 1, scope: !59, type: !9)
// CHECK:STDOUT: !61 = !DILocation(line: 14, column: 19, scope: !59)
// CHECK:STDOUT: !62 = !{!60}
// CHECK:STDOUT: !63 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.174120d4d03791b0:core.Destroy.Core", scope: null, file: !4, line: 14, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !66)
// CHECK:STDOUT: !64 = !DILocalVariable(arg: 1, scope: !63, type: !9)
// CHECK:STDOUT: !65 = !DILocation(line: 14, column: 19, scope: !63)
// CHECK:STDOUT: !66 = !{!64}
// CHECK:STDOUT: !67 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.826ebaace97974ae:core.Destroy.Core", scope: null, file: !4, line: 14, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !70)
// CHECK:STDOUT: !68 = !DILocalVariable(arg: 1, scope: !67, type: !9)
// CHECK:STDOUT: !69 = !DILocation(line: 14, column: 19, scope: !67)
// CHECK:STDOUT: !70 = !{!68}
// CHECK:STDOUT: !71 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.826ebaace97974ae:core.Destroy.Core", scope: null, file: !4, line: 14, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !74)
// CHECK:STDOUT: !72 = !DILocalVariable(arg: 1, scope: !71, type: !9)
// CHECK:STDOUT: !73 = !DILocation(line: 14, column: 19, scope: !71)
// CHECK:STDOUT: !74 = !{!72}
// CHECK:STDOUT: !75 = distinct !DISubprogram(name: "ReturnNullptr", linkageName: "_CReturnNullptr.Main", scope: null, file: !4, line: 17, type: !15, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !78 = !DILocation(line: 18, column: 10, scope: !75)
// CHECK:STDOUT: !79 = !DILocation(line: 18, column: 3, scope: !75)
// CHECK:STDOUT: !80 = distinct !DISubprogram(name: "ReturnNullptrCopy", linkageName: "_CReturnNullptrCopy.Main", scope: null, file: !4, line: 21, type: !15, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !83 = !DILocation(line: 22, column: 3, scope: !80)
// CHECK:STDOUT: !84 = !DILocation(line: 22, column: 26, scope: !80)
// CHECK:STDOUT: !86 = !DILocation(line: 23, column: 10, scope: !80)
// CHECK:STDOUT: !87 = !DILocation(line: 23, column: 3, scope: !80)
// CHECK:STDOUT: !88 = distinct !DISubprogram(name: "ReturnConvertedNullptrIndirectly", linkageName: "_CReturnConvertedNullptrIndirectly.Main", scope: null, file: !4, line: 26, type: !15, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !91 = !DILocation(line: 27, column: 3, scope: !88)
// CHECK:STDOUT: !92 = !DILocation(line: 27, column: 26, scope: !88)
// CHECK:STDOUT: !94 = !DILocation(line: 28, column: 10, scope: !88)
// CHECK:STDOUT: !95 = !DILocation(line: 28, column: 3, scope: !88)
// CHECK:STDOUT: !96 = distinct !DISubprogram(name: "ReturnConvertedNullptrDirectly", linkageName: "_CReturnConvertedNullptrDirectly.Main", scope: null, file: !4, line: 31, type: !15, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !99 = !DILocation(line: 32, column: 10, scope: !96)
// CHECK:STDOUT: !100 = !DILocation(line: 32, column: 3, scope: !96)
// CHECK:STDOUT: !101 = distinct !DISubprogram(name: "ConvertNullptrConstant", linkageName: "_CConvertNullptrConstant.Main", scope: null, file: !4, line: 35, type: !15, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !104 = !DILocation(line: 36, column: 3, scope: !101)
// CHECK:STDOUT: !105 = !DIFile(filename: "{{.*}}/prelude/types/cpp/nullptr.carbon", directory: "")
// CHECK:STDOUT: !106 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.NullptrT.CppCompat.Core:ImplicitAs.49279a15ecfc119b.Core.84588f41d61dafba", scope: null, file: !105, line: 51, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !113)
// CHECK:STDOUT: !107 = !DILocalVariable(arg: 1, scope: !106, type: !9)
// CHECK:STDOUT: !111 = !DILocation(line: 52, column: 14, scope: !106)
// CHECK:STDOUT: !112 = !DILocation(line: 52, column: 7, scope: !106)
// CHECK:STDOUT: !113 = !{!107}
// CHECK:STDOUT: !114 = distinct !DISubprogram(name: "None", linkageName: "_CNone.Optional.Core.6e9e957697159b53", scope: null, file: !22, line: 30, type: !15, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !118 = !DILocation(line: 31, column: 5, scope: !114)
// CHECK:STDOUT: !119 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !120 = !{!"omnipotent char", !119, i64 0}
// CHECK:STDOUT: !121 = !{!"std::nullptr_t", !120, i64 0}
// CHECK:STDOUT: !122 = !{!"any pointer", !120, i64 0}
// CHECK:STDOUT: !123 = !{!"p1 std::nullptr_t", !122, i64 0}
// CHECK:STDOUT: !124 = !{!123, !123, i64 0}
// CHECK:STDOUT: !125 = !{!121, !121, i64 0}
// CHECK:STDOUT: !126 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !127 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !128 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !129 = !{!"int", !120, i64 0}
// CHECK:STDOUT: !130 = !{!"__libc_errno", !129, i64 0}
// CHECK:STDOUT: !131 = !{!130, !129, i64 0}
// CHECK:STDOUT:
