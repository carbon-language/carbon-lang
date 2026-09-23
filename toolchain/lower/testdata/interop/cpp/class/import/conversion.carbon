// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/class/import/conversion.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/class/import/conversion.carbon

// --- class_to_class.carbon

library "[[@TEST_NAME]]";

import Cpp;
inline Cpp '''
struct A {
  int n;
  ~A();
};
struct B {
  operator A() const;
};
''';

fn ConvertToVar(b: Cpp.B) {
  var unused a: Cpp.A = b;
}

fn ConvertToValue(b: Cpp.B) {
  let unused a: Cpp.A = b;
}

// --- class_to_enum.carbon

library "[[@TEST_NAME]]";

import Cpp;
inline Cpp '''
enum S : short {};
enum I : short {};
struct EnumB {
  operator S() const;
  operator I() const;
};
''';

fn ConvertToVar(b: Cpp.EnumB) {
  var unused s: Cpp.S = b;
  var unused i: Cpp.I = b;
}

fn ConvertToValue(b: Cpp.EnumB) {
  let unused s: Cpp.S = b;
  let unused i: Cpp.I = b;
}

// --- class_to_int.carbon

library "[[@TEST_NAME]]";

import Cpp;
inline Cpp '''
struct IntB {
  operator int() const;
};
''';

fn ConvertToVar(b: Cpp.IntB) {
  var unused a: Cpp.int = b;
}

fn ConvertToValue(b: Cpp.IntB) {
  let unused a: Cpp.int = b;
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'class_to_class.carbon'
// CHECK:STDOUT: source_filename = "class_to_class.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %struct.A = type { i32 }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvertToVar.Main(ptr %b) #0 !dbg !26 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca [4 x i8], align 4, !dbg !29
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !29
// CHECK:STDOUT:   call void @_ZNK1Bcv1AEv.carbon_thunk._(ptr %b, ptr %a.var), !dbg !29
// CHECK:STDOUT:   call void @"_CSelfDestruct.f2cd4feee22137a3:core.Destroy.Core"(ptr %a.var), !dbg !29
// CHECK:STDOUT:   ret void, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZNK1Bcv1AEv.carbon_thunk._(ptr noundef nonnull align 1 dereferenceable(1) %this, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !50
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !52
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !52
// CHECK:STDOUT:   %1 = load ptr, ptr %this.addr, align 8, !tbaa !50, !nonnull !54
// CHECK:STDOUT:   call void @_ZNK1Bcv1AEv(ptr dead_on_unwind writable sret(%struct.A) align 4 %0, ptr noundef nonnull align 1 dereferenceable(1) %1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: declare void @_ZN1AD1Ev(ptr noundef nonnull align 4 dead_on_return(4) dereferenceable(4)) unnamed_addr #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f2cd4feee22137a3:core.Destroy.Core"(ptr %self) #0 !dbg !34 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN1AD1Ev(ptr %self), !dbg !36
// CHECK:STDOUT:   ret void, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvertToValue.Main(ptr %b) #0 !dbg !38 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc20_25.1.temp = alloca [4 x i8], align 4, !dbg !41
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc20_25.1.temp), !dbg !41
// CHECK:STDOUT:   call void @_ZNK1Bcv1AEv.carbon_thunk._(ptr %b, ptr %.loc20_25.1.temp), !dbg !41
// CHECK:STDOUT:   call void @"_CSelfDestruct.f2cd4feee22137a3:core.Destroy.Core"(ptr %.loc20_25.1.temp), !dbg !41
// CHECK:STDOUT:   ret void, !dbg !44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_ZNK1Bcv1AEv(ptr dead_on_unwind writable sret(%struct.A) align 4, ptr noundef nonnull align 1 dereferenceable(1)) #4
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #4 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!57, !58, !59, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!61}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !18 = !DIFile(filename: "class_to_class.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{null, !21}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "ConvertToVar", linkageName: "_CConvertToVar.Main", scope: null, file: !18, line: 15, type: !23, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !33)
// CHECK:STDOUT: !27 = !DILocalVariable(arg: 1, scope: !26, type: !21)
// CHECK:STDOUT: !29 = !DILocation(line: 16, column: 3, scope: !26)
// CHECK:STDOUT: !32 = !DILocation(line: 15, column: 1, scope: !26)
// CHECK:STDOUT: !33 = !{!27}
// CHECK:STDOUT: !34 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f2cd4feee22137a3:core.Destroy.Core", scope: null, file: !18, line: 16, type: !23, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !37)
// CHECK:STDOUT: !35 = !DILocalVariable(arg: 1, scope: !34, type: !21)
// CHECK:STDOUT: !36 = !DILocation(line: 16, column: 3, scope: !34)
// CHECK:STDOUT: !37 = !{!35}
// CHECK:STDOUT: !38 = distinct !DISubprogram(name: "ConvertToValue", linkageName: "_CConvertToValue.Main", scope: null, file: !18, line: 19, type: !23, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !45)
// CHECK:STDOUT: !39 = !DILocalVariable(arg: 1, scope: !38, type: !21)
// CHECK:STDOUT: !41 = !DILocation(line: 20, column: 25, scope: !38)
// CHECK:STDOUT: !44 = !DILocation(line: 19, column: 1, scope: !38)
// CHECK:STDOUT: !45 = !{!39}
// CHECK:STDOUT: !46 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !47 = !{!"omnipotent char", !46, i64 0}
// CHECK:STDOUT: !48 = !{!"any pointer", !47, i64 0}
// CHECK:STDOUT: !49 = !{!"p1 _ZTS1B", !48, i64 0}
// CHECK:STDOUT: !50 = !{!49, !49, i64 0}
// CHECK:STDOUT: !51 = !{!"p1 _ZTS1A", !48, i64 0}
// CHECK:STDOUT: !52 = !{!51, !51, i64 0}
// CHECK:STDOUT: !54 = !{}
// CHECK:STDOUT: !55 = !{!"int", !47, i64 0}
// CHECK:STDOUT: !57 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !58 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !59 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !60 = !{!"__libc_errno", !55, i64 0}
// CHECK:STDOUT: !61 = !{!60, !55, i64 0}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'class_to_enum.carbon'
// CHECK:STDOUT: source_filename = "class_to_enum.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvertToVar.Main(ptr %b) #0 !dbg !67 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %s.var = alloca i16, align 2, !dbg !70
// CHECK:STDOUT:   %i.var = alloca i16, align 2, !dbg !73
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %s.var), !dbg !70
// CHECK:STDOUT:   call void @_ZNK5EnumBcv1SEv.carbon_thunk._(ptr %b, ptr %s.var), !dbg !70
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %i.var), !dbg !73
// CHECK:STDOUT:   call void @_ZNK5EnumBcv1IEv.carbon_thunk._(ptr %b, ptr %i.var), !dbg !73
// CHECK:STDOUT:   call void @"_CSelfDestruct.93ce72f55a28222b:core.Destroy.Core"(ptr %i.var), !dbg !73
// CHECK:STDOUT:   call void @"_CSelfDestruct.7c9bfa05445fab48:core.Destroy.Core"(ptr %s.var), !dbg !70
// CHECK:STDOUT:   ret void, !dbg !76
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZNK5EnumBcv1SEv.carbon_thunk._(ptr noundef nonnull align 1 dereferenceable(1) %this, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !98
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !99
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !99
// CHECK:STDOUT:   %1 = load ptr, ptr %this.addr, align 8, !tbaa !98, !nonnull !54
// CHECK:STDOUT:   %call = call noundef signext i16 @_ZNK5EnumBcv1SEv(ptr noundef nonnull align 1 dereferenceable(1) %1)
// CHECK:STDOUT:   store i16 %call, ptr %0, align 2, !tbaa !102
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZNK5EnumBcv1IEv.carbon_thunk._(ptr noundef nonnull align 1 dereferenceable(1) %this, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !98
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !99
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !99
// CHECK:STDOUT:   %1 = load ptr, ptr %this.addr, align 8, !tbaa !98, !nonnull !54
// CHECK:STDOUT:   %call = call noundef signext i16 @_ZNK5EnumBcv1IEv(ptr noundef nonnull align 1 dereferenceable(1) %1)
// CHECK:STDOUT:   store i16 %call, ptr %0, align 2, !tbaa !104
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.93ce72f55a28222b:core.Destroy.Core"(ptr %self) #0 !dbg !78 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !80
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.7c9bfa05445fab48:core.Destroy.Core"(ptr %self) #0 !dbg !82 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !84
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvertToValue.Main(ptr %b) #0 !dbg !86 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc20_25.1.temp = alloca i16, align 2, !dbg !89
// CHECK:STDOUT:   %.loc21_25.1.temp = alloca i16, align 2, !dbg !92
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc20_25.1.temp), !dbg !89
// CHECK:STDOUT:   call void @_ZNK5EnumBcv1SEv.carbon_thunk._(ptr %b, ptr %.loc20_25.1.temp), !dbg !89
// CHECK:STDOUT:   %.loc20_25.5 = load i16, ptr %.loc20_25.1.temp, align 2, !dbg !89
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc21_25.1.temp), !dbg !92
// CHECK:STDOUT:   call void @_ZNK5EnumBcv1IEv.carbon_thunk._(ptr %b, ptr %.loc21_25.1.temp), !dbg !92
// CHECK:STDOUT:   %.loc21_25.5 = load i16, ptr %.loc21_25.1.temp, align 2, !dbg !92
// CHECK:STDOUT:   call void @"_CSelfDestruct.93ce72f55a28222b:core.Destroy.Core"(ptr %.loc21_25.1.temp), !dbg !92
// CHECK:STDOUT:   call void @"_CSelfDestruct.7c9bfa05445fab48:core.Destroy.Core"(ptr %.loc20_25.1.temp), !dbg !89
// CHECK:STDOUT:   ret void, !dbg !95
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef signext i16 @_ZNK5EnumBcv1SEv(ptr noundef nonnull align 1 dereferenceable(1)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef signext i16 @_ZNK5EnumBcv1IEv(ptr noundef nonnull align 1 dereferenceable(1)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!63}
// CHECK:STDOUT: !llvm.module.flags = !{!57, !58, !59, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!61}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{null, !21}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !46 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !47 = !{!"omnipotent char", !46, i64 0}
// CHECK:STDOUT: !48 = !{!"any pointer", !47, i64 0}
// CHECK:STDOUT: !54 = !{}
// CHECK:STDOUT: !55 = !{!"int", !47, i64 0}
// CHECK:STDOUT: !57 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !58 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !59 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !60 = !{!"__libc_errno", !55, i64 0}
// CHECK:STDOUT: !61 = !{!60, !55, i64 0}
// CHECK:STDOUT: !62 = !DIFile(filename: "class_to_enum.carbon", directory: "")
// CHECK:STDOUT: !63 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !62, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !64 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_signed)
// CHECK:STDOUT: !65 = !{null, !64}
// CHECK:STDOUT: !66 = !DISubroutineType(types: !65)
// CHECK:STDOUT: !67 = distinct !DISubprogram(name: "ConvertToVar", linkageName: "_CConvertToVar.Main", scope: null, file: !62, line: 14, type: !23, spFlags: DISPFlagDefinition, unit: !63, retainedNodes: !77)
// CHECK:STDOUT: !68 = !DILocalVariable(arg: 1, scope: !67, type: !21)
// CHECK:STDOUT: !70 = !DILocation(line: 15, column: 3, scope: !67)
// CHECK:STDOUT: !73 = !DILocation(line: 16, column: 3, scope: !67)
// CHECK:STDOUT: !76 = !DILocation(line: 14, column: 1, scope: !67)
// CHECK:STDOUT: !77 = !{!68}
// CHECK:STDOUT: !78 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.93ce72f55a28222b:core.Destroy.Core", scope: null, file: !62, line: 16, type: !66, spFlags: DISPFlagDefinition, unit: !63, retainedNodes: !81)
// CHECK:STDOUT: !79 = !DILocalVariable(arg: 1, scope: !78, type: !64)
// CHECK:STDOUT: !80 = !DILocation(line: 16, column: 3, scope: !78)
// CHECK:STDOUT: !81 = !{!79}
// CHECK:STDOUT: !82 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.7c9bfa05445fab48:core.Destroy.Core", scope: null, file: !62, line: 15, type: !66, spFlags: DISPFlagDefinition, unit: !63, retainedNodes: !85)
// CHECK:STDOUT: !83 = !DILocalVariable(arg: 1, scope: !82, type: !64)
// CHECK:STDOUT: !84 = !DILocation(line: 15, column: 3, scope: !82)
// CHECK:STDOUT: !85 = !{!83}
// CHECK:STDOUT: !86 = distinct !DISubprogram(name: "ConvertToValue", linkageName: "_CConvertToValue.Main", scope: null, file: !62, line: 19, type: !23, spFlags: DISPFlagDefinition, unit: !63, retainedNodes: !96)
// CHECK:STDOUT: !87 = !DILocalVariable(arg: 1, scope: !86, type: !21)
// CHECK:STDOUT: !89 = !DILocation(line: 20, column: 25, scope: !86)
// CHECK:STDOUT: !92 = !DILocation(line: 21, column: 25, scope: !86)
// CHECK:STDOUT: !95 = !DILocation(line: 19, column: 1, scope: !86)
// CHECK:STDOUT: !96 = !{!87}
// CHECK:STDOUT: !97 = !{!"p1 _ZTS5EnumB", !48, i64 0}
// CHECK:STDOUT: !98 = !{!97, !97, i64 0}
// CHECK:STDOUT: !99 = !{!48, !48, i64 0}
// CHECK:STDOUT: !100 = !{!"_ZTS1S", !47, i64 0}
// CHECK:STDOUT: !102 = !{!100, !100, i64 0}
// CHECK:STDOUT: !103 = !{!"_ZTS1I", !47, i64 0}
// CHECK:STDOUT: !104 = !{!103, !103, i64 0}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'class_to_int.carbon'
// CHECK:STDOUT: source_filename = "class_to_int.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvertToVar.Main(ptr %b) #0 !dbg !111 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca i32, align 4, !dbg !114
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !114
// CHECK:STDOUT:   %__carbon_thunk.call = call i32 @_ZNK4IntBcviEv.carbon_thunk._(ptr %b), !dbg !114
// CHECK:STDOUT:   store i32 %__carbon_thunk.call, ptr %a.var, align 4, !dbg !114
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %a.var), !dbg !114
// CHECK:STDOUT:   ret void, !dbg !117
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal noundef i32 @_ZNK4IntBcviEv.carbon_thunk._(ptr noundef nonnull align 1 dereferenceable(1) %this) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !140
// CHECK:STDOUT:   %0 = load ptr, ptr %this.addr, align 8, !tbaa !140, !nonnull !54
// CHECK:STDOUT:   %call = call noundef i32 @_ZNK4IntBcviEv(ptr noundef nonnull align 1 dereferenceable(1) %0)
// CHECK:STDOUT:   ret i32 %call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.6dae29ece9a708c4:core.Destroy.Core"(ptr %self) #0 !dbg !119 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !121
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !123 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !125
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !127 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self), !dbg !129
// CHECK:STDOUT:   ret void, !dbg !129
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvertToValue.Main(ptr %b) #0 !dbg !131 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %__carbon_thunk.call = call i32 @_ZNK4IntBcviEv.carbon_thunk._(ptr %b), !dbg !134
// CHECK:STDOUT:   ret void, !dbg !137
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef i32 @_ZNK4IntBcviEv(ptr noundef nonnull align 1 dereferenceable(1)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!106}
// CHECK:STDOUT: !llvm.module.flags = !{!57, !58, !59, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!61}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{null, !21}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !46 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !47 = !{!"omnipotent char", !46, i64 0}
// CHECK:STDOUT: !48 = !{!"any pointer", !47, i64 0}
// CHECK:STDOUT: !54 = !{}
// CHECK:STDOUT: !55 = !{!"int", !47, i64 0}
// CHECK:STDOUT: !57 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !58 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !59 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !60 = !{!"__libc_errno", !55, i64 0}
// CHECK:STDOUT: !61 = !{!60, !55, i64 0}
// CHECK:STDOUT: !105 = !DIFile(filename: "class_to_int.carbon", directory: "")
// CHECK:STDOUT: !106 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !105, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !109 = !{null, !20}
// CHECK:STDOUT: !110 = !DISubroutineType(types: !109)
// CHECK:STDOUT: !111 = distinct !DISubprogram(name: "ConvertToVar", linkageName: "_CConvertToVar.Main", scope: null, file: !105, line: 11, type: !23, spFlags: DISPFlagDefinition, unit: !106, retainedNodes: !118)
// CHECK:STDOUT: !112 = !DILocalVariable(arg: 1, scope: !111, type: !21)
// CHECK:STDOUT: !114 = !DILocation(line: 12, column: 3, scope: !111)
// CHECK:STDOUT: !117 = !DILocation(line: 11, column: 1, scope: !111)
// CHECK:STDOUT: !118 = !{!112}
// CHECK:STDOUT: !119 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.6dae29ece9a708c4:core.Destroy.Core", scope: null, file: !105, line: 12, type: !110, spFlags: DISPFlagDefinition, unit: !106, retainedNodes: !122)
// CHECK:STDOUT: !120 = !DILocalVariable(arg: 1, scope: !119, type: !20)
// CHECK:STDOUT: !121 = !DILocation(line: 12, column: 3, scope: !119)
// CHECK:STDOUT: !122 = !{!120}
// CHECK:STDOUT: !123 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core", scope: null, file: !105, line: 12, type: !110, spFlags: DISPFlagDefinition, unit: !106, retainedNodes: !126)
// CHECK:STDOUT: !124 = !DILocalVariable(arg: 1, scope: !123, type: !20)
// CHECK:STDOUT: !125 = !DILocation(line: 12, column: 3, scope: !123)
// CHECK:STDOUT: !126 = !{!124}
// CHECK:STDOUT: !127 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f58524831c37807e:core.Destroy.Core", scope: null, file: !105, line: 12, type: !110, spFlags: DISPFlagDefinition, unit: !106, retainedNodes: !130)
// CHECK:STDOUT: !128 = !DILocalVariable(arg: 1, scope: !127, type: !20)
// CHECK:STDOUT: !129 = !DILocation(line: 12, column: 3, scope: !127)
// CHECK:STDOUT: !130 = !{!128}
// CHECK:STDOUT: !131 = distinct !DISubprogram(name: "ConvertToValue", linkageName: "_CConvertToValue.Main", scope: null, file: !105, line: 15, type: !23, spFlags: DISPFlagDefinition, unit: !106, retainedNodes: !138)
// CHECK:STDOUT: !132 = !DILocalVariable(arg: 1, scope: !131, type: !21)
// CHECK:STDOUT: !134 = !DILocation(line: 16, column: 27, scope: !131)
// CHECK:STDOUT: !137 = !DILocation(line: 15, column: 1, scope: !131)
// CHECK:STDOUT: !138 = !{!132}
// CHECK:STDOUT: !139 = !{!"p1 _ZTS4IntB", !48, i64 0}
// CHECK:STDOUT: !140 = !{!139, !139, i64 0}
// CHECK:STDOUT:
