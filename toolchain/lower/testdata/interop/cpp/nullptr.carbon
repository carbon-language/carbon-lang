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

// CHECK:STDOUT: ; ModuleID = 'simple.carbon'
// CHECK:STDOUT: source_filename = "simple.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassNullptr.Main() #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc12_18.2.temp = alloca ptr, align 8, !dbg !14
// CHECK:STDOUT:   %.loc14_22.1.temp = alloca ptr, align 8, !dbg !15
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.NullptrT.CppCompat.Core:ImplicitAs.269b9b75d033aabb.Core.84588f41d61dafba"(ptr poison), !dbg !14
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc12_18.2.temp), !dbg !14
// CHECK:STDOUT:   store ptr %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.call, ptr %.loc12_18.2.temp, align 8, !dbg !14
// CHECK:STDOUT:   %.loc12_18.4 = load ptr, ptr %.loc12_18.2.temp, align 8, !dbg !14
// CHECK:STDOUT:   call void @_Z7TakePtrPi(ptr %.loc12_18.4), !dbg !16
// CHECK:STDOUT:   %Cpp.nullptr_t.as.Copy.impl.Op.call = call ptr @"_COp.NullptrT.CppCompat.Core:Copy.Core"(ptr poison), !dbg !15
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc14_22.1.temp), !dbg !15
// CHECK:STDOUT:   store ptr %Cpp.nullptr_t.as.Copy.impl.Op.call, ptr %.loc14_22.1.temp, align 8, !dbg !15
// CHECK:STDOUT:   call void @_Z11TakeNullptrDn.carbon_thunk._(ptr %.loc14_22.1.temp), !dbg !17
// CHECK:STDOUT:   call void @"_COp.37a0be3fec6961f4:core.Destroy.Core"(ptr %.loc14_22.1.temp), !dbg !15
// CHECK:STDOUT:   call void @"_COp.197d85846f87d79f:core.Destroy.Core"(ptr %.loc12_18.2.temp), !dbg !14
// CHECK:STDOUT:   ret void, !dbg !18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z7TakePtrPi(ptr noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z11TakeNullptrDn.carbon_thunk._(ptr noundef %n) #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %n.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %n, ptr %n.addr, align 8, !tbaa !19
// CHECK:STDOUT:   %0 = load ptr, ptr %n.addr, align 8, !tbaa !19
// CHECK:STDOUT:   call void @_Z11TakeNullptrDn(ptr null)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare ptr @"_COp.NullptrT.CppCompat.Core:Copy.Core"(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.fd39a15e7d61228d:core.Destroy.Core"(ptr %self) #0 !dbg !22 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.125695b29f2cf97c:core.Destroy.Core"(ptr %self) #0 !dbg !29 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.37a0be3fec6961f4:core.Destroy.Core"(ptr %self) #0 !dbg !33 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.32a070744df634d9:core.Destroy.Core"(ptr %self) #0 !dbg !37 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.958634845f762cfe:core.Destroy.Core"(ptr %self) #0 !dbg !41 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.197d85846f87d79f:core.Destroy.Core"(ptr %self) #0 !dbg !45 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CReturnNullptr.Main() #0 !dbg !49 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Cpp.nullptr_t.as.Copy.impl.Op.call = call ptr @"_COp.NullptrT.CppCompat.Core:Copy.Core"(ptr poison), !dbg !52
// CHECK:STDOUT:   ret ptr %Cpp.nullptr_t.as.Copy.impl.Op.call, !dbg !53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CReturnNullptrCopy.Main() #0 !dbg !54 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca ptr, align 8, !dbg !55
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !55
// CHECK:STDOUT:   %Cpp.nullptr_t.as.Copy.impl.Op.call.loc22 = call ptr @"_COp.NullptrT.CppCompat.Core:Copy.Core"(ptr poison), !dbg !56
// CHECK:STDOUT:   store ptr %Cpp.nullptr_t.as.Copy.impl.Op.call.loc22, ptr %a.var, align 8, !dbg !55
// CHECK:STDOUT:   %.loc23 = load ptr, ptr %a.var, align 8, !dbg !57
// CHECK:STDOUT:   %Cpp.nullptr_t.as.Copy.impl.Op.call.loc23 = call ptr @"_COp.NullptrT.CppCompat.Core:Copy.Core"(ptr %.loc23), !dbg !57
// CHECK:STDOUT:   call void @"_COp.37a0be3fec6961f4:core.Destroy.Core"(ptr %a.var), !dbg !55
// CHECK:STDOUT:   ret ptr %Cpp.nullptr_t.as.Copy.impl.Op.call.loc23, !dbg !58
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CReturnConvertedNullptrIndirectly.Main() #0 !dbg !59 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca ptr, align 8, !dbg !60
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !60
// CHECK:STDOUT:   call void @_Z13ReturnNullptrv.carbon_thunk.(ptr %a.var), !dbg !61
// CHECK:STDOUT:   %.loc28_10 = load ptr, ptr %a.var, align 8, !dbg !62
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.NullptrT.CppCompat.Core:ImplicitAs.269b9b75d033aabb.Core.84588f41d61dafba"(ptr %.loc28_10), !dbg !63
// CHECK:STDOUT:   call void @"_COp.37a0be3fec6961f4:core.Destroy.Core"(ptr %a.var), !dbg !60
// CHECK:STDOUT:   ret ptr %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.call, !dbg !63
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z13ReturnNullptrv.carbon_thunk.(ptr noundef %return) #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !19
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !19
// CHECK:STDOUT:   %call = call ptr @_Z13ReturnNullptrv()
// CHECK:STDOUT:   store ptr %call, ptr %0, align 8, !tbaa !64
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CReturnConvertedNullptrDirectly.Main() #0 !dbg !66 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc32_28.1.temp = alloca ptr, align 8, !dbg !67
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc32_28.1.temp), !dbg !67
// CHECK:STDOUT:   call void @_Z13ReturnNullptrv.carbon_thunk.(ptr %.loc32_28.1.temp), !dbg !67
// CHECK:STDOUT:   %.loc32_28.4 = load ptr, ptr %.loc32_28.1.temp, align 8, !dbg !67
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.NullptrT.CppCompat.Core:ImplicitAs.269b9b75d033aabb.Core.84588f41d61dafba"(ptr %.loc32_28.4), !dbg !68
// CHECK:STDOUT:   call void @"_COp.37a0be3fec6961f4:core.Destroy.Core"(ptr %.loc32_28.1.temp), !dbg !67
// CHECK:STDOUT:   ret ptr %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.call, !dbg !68
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CConvertNullptrConstant.Main() #0 !dbg !69 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.NullptrT.CppCompat.Core:ImplicitAs.269b9b75d033aabb.Core.84588f41d61dafba"(ptr poison), !dbg !70
// CHECK:STDOUT:   ret ptr %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.call, !dbg !70
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.NullptrT.CppCompat.Core:ImplicitAs.269b9b75d033aabb.Core.84588f41d61dafba"(ptr %self) #0 !dbg !71 {
// CHECK:STDOUT:   %1 = call ptr @_CNone.Optional.Core.a636f8d928e21310(), !dbg !77
// CHECK:STDOUT:   ret ptr %1, !dbg !78
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: declare ptr @_CMake.NullptrT.CppCompat.Core()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CNone.Optional.Core.a636f8d928e21310() #0 !dbg !79 {
// CHECK:STDOUT:   ret ptr null, !dbg !81
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z11TakeNullptrDn(ptr) #1
// CHECK:STDOUT:
// CHECK:STDOUT: declare ptr @_Z13ReturnNullptrv() #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @"_CConvert.NullptrT.CppCompat.Core:ImplicitAs.269b9b75d033aabb.Core.84588f41d61dafba", { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 4, 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3, !4, !5, !6}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!7}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "simple.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !3 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !4 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !5 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !6 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !7 = !{!8, !8, i64 0}
// CHECK:STDOUT: !8 = !{!"int", !9, i64 0}
// CHECK:STDOUT: !9 = !{!"omnipotent char", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "PassNullptr", linkageName: "_CPassNullptr.Main", scope: null, file: !1, line: 11, type: !12, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !12 = !DISubroutineType(types: !13)
// CHECK:STDOUT: !13 = !{null}
// CHECK:STDOUT: !14 = !DILocation(line: 12, column: 15, scope: !11)
// CHECK:STDOUT: !15 = !DILocation(line: 14, column: 19, scope: !11)
// CHECK:STDOUT: !16 = !DILocation(line: 12, column: 3, scope: !11)
// CHECK:STDOUT: !17 = !DILocation(line: 14, column: 3, scope: !11)
// CHECK:STDOUT: !18 = !DILocation(line: 11, column: 1, scope: !11)
// CHECK:STDOUT: !19 = !{!20, !20, i64 0}
// CHECK:STDOUT: !20 = !{!"p1 std::nullptr_t", !21, i64 0}
// CHECK:STDOUT: !21 = !{!"any pointer", !9, i64 0}
// CHECK:STDOUT: !22 = distinct !DISubprogram(name: "Op", linkageName: "_COp.fd39a15e7d61228d:core.Destroy.Core", scope: null, file: !1, line: 14, type: !23, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !26)
// CHECK:STDOUT: !23 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !24 = !{null, !25}
// CHECK:STDOUT: !25 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !26 = !{!27}
// CHECK:STDOUT: !27 = !DILocalVariable(arg: 1, scope: !22, type: !25)
// CHECK:STDOUT: !28 = !DILocation(line: 14, column: 19, scope: !22)
// CHECK:STDOUT: !29 = distinct !DISubprogram(name: "Op", linkageName: "_COp.125695b29f2cf97c:core.Destroy.Core", scope: null, file: !1, line: 14, type: !23, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !30)
// CHECK:STDOUT: !30 = !{!31}
// CHECK:STDOUT: !31 = !DILocalVariable(arg: 1, scope: !29, type: !25)
// CHECK:STDOUT: !32 = !DILocation(line: 14, column: 19, scope: !29)
// CHECK:STDOUT: !33 = distinct !DISubprogram(name: "Op", linkageName: "_COp.37a0be3fec6961f4:core.Destroy.Core", scope: null, file: !1, line: 14, type: !23, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !34)
// CHECK:STDOUT: !34 = !{!35}
// CHECK:STDOUT: !35 = !DILocalVariable(arg: 1, scope: !33, type: !25)
// CHECK:STDOUT: !36 = !DILocation(line: 14, column: 19, scope: !33)
// CHECK:STDOUT: !37 = distinct !DISubprogram(name: "Op", linkageName: "_COp.32a070744df634d9:core.Destroy.Core", scope: null, file: !1, line: 12, type: !23, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !38)
// CHECK:STDOUT: !38 = !{!39}
// CHECK:STDOUT: !39 = !DILocalVariable(arg: 1, scope: !37, type: !25)
// CHECK:STDOUT: !40 = !DILocation(line: 12, column: 15, scope: !37)
// CHECK:STDOUT: !41 = distinct !DISubprogram(name: "Op", linkageName: "_COp.958634845f762cfe:core.Destroy.Core", scope: null, file: !1, line: 12, type: !23, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !42)
// CHECK:STDOUT: !42 = !{!43}
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 1, scope: !41, type: !25)
// CHECK:STDOUT: !44 = !DILocation(line: 12, column: 15, scope: !41)
// CHECK:STDOUT: !45 = distinct !DISubprogram(name: "Op", linkageName: "_COp.197d85846f87d79f:core.Destroy.Core", scope: null, file: !1, line: 12, type: !23, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !46)
// CHECK:STDOUT: !46 = !{!47}
// CHECK:STDOUT: !47 = !DILocalVariable(arg: 1, scope: !45, type: !25)
// CHECK:STDOUT: !48 = !DILocation(line: 12, column: 15, scope: !45)
// CHECK:STDOUT: !49 = distinct !DISubprogram(name: "ReturnNullptr", linkageName: "_CReturnNullptr.Main", scope: null, file: !1, line: 17, type: !50, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !50 = !DISubroutineType(types: !51)
// CHECK:STDOUT: !51 = !{!25}
// CHECK:STDOUT: !52 = !DILocation(line: 18, column: 10, scope: !49)
// CHECK:STDOUT: !53 = !DILocation(line: 18, column: 3, scope: !49)
// CHECK:STDOUT: !54 = distinct !DISubprogram(name: "ReturnNullptrCopy", linkageName: "_CReturnNullptrCopy.Main", scope: null, file: !1, line: 21, type: !50, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !55 = !DILocation(line: 22, column: 3, scope: !54)
// CHECK:STDOUT: !56 = !DILocation(line: 22, column: 26, scope: !54)
// CHECK:STDOUT: !57 = !DILocation(line: 23, column: 10, scope: !54)
// CHECK:STDOUT: !58 = !DILocation(line: 23, column: 3, scope: !54)
// CHECK:STDOUT: !59 = distinct !DISubprogram(name: "ReturnConvertedNullptrIndirectly", linkageName: "_CReturnConvertedNullptrIndirectly.Main", scope: null, file: !1, line: 26, type: !50, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !60 = !DILocation(line: 27, column: 3, scope: !59)
// CHECK:STDOUT: !61 = !DILocation(line: 27, column: 26, scope: !59)
// CHECK:STDOUT: !62 = !DILocation(line: 28, column: 10, scope: !59)
// CHECK:STDOUT: !63 = !DILocation(line: 28, column: 3, scope: !59)
// CHECK:STDOUT: !64 = !{!65, !65, i64 0}
// CHECK:STDOUT: !65 = !{!"std::nullptr_t", !9, i64 0}
// CHECK:STDOUT: !66 = distinct !DISubprogram(name: "ReturnConvertedNullptrDirectly", linkageName: "_CReturnConvertedNullptrDirectly.Main", scope: null, file: !1, line: 31, type: !50, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !67 = !DILocation(line: 32, column: 10, scope: !66)
// CHECK:STDOUT: !68 = !DILocation(line: 32, column: 3, scope: !66)
// CHECK:STDOUT: !69 = distinct !DISubprogram(name: "ConvertNullptrConstant", linkageName: "_CConvertNullptrConstant.Main", scope: null, file: !1, line: 35, type: !50, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !70 = !DILocation(line: 36, column: 3, scope: !69)
// CHECK:STDOUT: !71 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.NullptrT.CppCompat.Core:ImplicitAs.269b9b75d033aabb.Core.84588f41d61dafba", scope: null, file: !72, line: 51, type: !73, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !75)
// CHECK:STDOUT: !72 = !DIFile(filename: "{{.*}}/prelude/types/cpp/nullptr.carbon", directory: "")
// CHECK:STDOUT: !73 = !DISubroutineType(types: !74)
// CHECK:STDOUT: !74 = !{!25, !25}
// CHECK:STDOUT: !75 = !{!76}
// CHECK:STDOUT: !76 = !DILocalVariable(arg: 1, scope: !71, type: !25)
// CHECK:STDOUT: !77 = !DILocation(line: 52, column: 14, scope: !71)
// CHECK:STDOUT: !78 = !DILocation(line: 52, column: 7, scope: !71)
// CHECK:STDOUT: !79 = distinct !DISubprogram(name: "None", linkageName: "_CNone.Optional.Core.a636f8d928e21310", scope: null, file: !80, line: 30, type: !50, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !80 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !81 = !DILocation(line: 31, column: 5, scope: !79)
