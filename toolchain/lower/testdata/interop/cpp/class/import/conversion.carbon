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
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.6191dcb71c708f97(ptr %a.var), !dbg !29
// CHECK:STDOUT:   ret void, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZNK1Bcv1AEv.carbon_thunk._(ptr noundef nonnull align 1 dereferenceable(1) %this, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !57
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !59
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !59
// CHECK:STDOUT:   %1 = load ptr, ptr %this.addr, align 8, !tbaa !57, !nonnull !61
// CHECK:STDOUT:   call void @_ZNK1Bcv1AEv(ptr dead_on_unwind writable sret(%struct.A) align 4 %0, ptr noundef nonnull align 1 dereferenceable(1) %1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:DestroyImpl.20eb678cb24c67a6.Core:DestroyImpl.Core"(ptr %self) #2 !dbg !34 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: declare void @_ZN1AD1Ev(ptr noundef nonnull align 4 dead_on_return(4) dereferenceable(4)) unnamed_addr #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvertToValue.Main(ptr %b) #0 !dbg !38 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc20_25.1.temp = alloca [4 x i8], align 4, !dbg !41
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc20_25.1.temp), !dbg !41
// CHECK:STDOUT:   call void @_ZNK1Bcv1AEv.carbon_thunk._(ptr %b, ptr %.loc20_25.1.temp), !dbg !41
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.6191dcb71c708f97(ptr %.loc20_25.1.temp), !dbg !41
// CHECK:STDOUT:   ret void, !dbg !44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #4
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.6191dcb71c708f97(ptr %t) #0 !dbg !46 {
// CHECK:STDOUT:   call void @"_COp:thunk:DestroyImpl.20eb678cb24c67a6.Core:DestroyImpl.Core"(ptr %t), !dbg !49
// CHECK:STDOUT:   call void @_ZN1AD1Ev(ptr %t), !dbg !50
// CHECK:STDOUT:   ret void, !dbg !51
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_ZNK1Bcv1AEv(ptr dead_on_unwind writable sret(%struct.A) align 4, ptr noundef nonnull align 1 dereferenceable(1)) #5
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.6191dcb71c708f97, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #3 = { nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #4 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #5 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!64, !65, !66, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!68}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
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
// CHECK:STDOUT: !34 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:DestroyImpl.20eb678cb24c67a6.Core:DestroyImpl.Core", scope: null, file: !18, type: !23, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !37)
// CHECK:STDOUT: !35 = !DILocalVariable(arg: 1, scope: !34, type: !21)
// CHECK:STDOUT: !36 = !DILocation(line: 0, scope: !34)
// CHECK:STDOUT: !37 = !{!35}
// CHECK:STDOUT: !38 = distinct !DISubprogram(name: "ConvertToValue", linkageName: "_CConvertToValue.Main", scope: null, file: !18, line: 19, type: !23, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !45)
// CHECK:STDOUT: !39 = !DILocalVariable(arg: 1, scope: !38, type: !21)
// CHECK:STDOUT: !41 = !DILocation(line: 20, column: 25, scope: !38)
// CHECK:STDOUT: !44 = !DILocation(line: 19, column: 1, scope: !38)
// CHECK:STDOUT: !45 = !{!39}
// CHECK:STDOUT: !46 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.6191dcb71c708f97", scope: null, file: !10, line: 26, type: !23, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !52)
// CHECK:STDOUT: !47 = !DILocalVariable(arg: 1, scope: !46, type: !21)
// CHECK:STDOUT: !49 = !DILocation(line: 27, column: 5, scope: !46)
// CHECK:STDOUT: !50 = !DILocation(line: 28, column: 5, scope: !46)
// CHECK:STDOUT: !51 = !DILocation(line: 26, column: 1, scope: !46)
// CHECK:STDOUT: !52 = !{!47}
// CHECK:STDOUT: !53 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !54 = !{!"omnipotent char", !53, i64 0}
// CHECK:STDOUT: !55 = !{!"any pointer", !54, i64 0}
// CHECK:STDOUT: !56 = !{!"p1 _ZTS1B", !55, i64 0}
// CHECK:STDOUT: !57 = !{!56, !56, i64 0}
// CHECK:STDOUT: !58 = !{!"p1 _ZTS1A", !55, i64 0}
// CHECK:STDOUT: !59 = !{!58, !58, i64 0}
// CHECK:STDOUT: !61 = !{}
// CHECK:STDOUT: !62 = !{!"int", !54, i64 0}
// CHECK:STDOUT: !64 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !65 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !66 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !67 = !{!"__libc_errno", !62, i64 0}
// CHECK:STDOUT: !68 = !{!67, !62, i64 0}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'class_to_enum.carbon'
// CHECK:STDOUT: source_filename = "class_to_enum.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvertToVar.Main(ptr %b) #0 !dbg !74 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %s.var = alloca i16, align 2, !dbg !77
// CHECK:STDOUT:   %i.var = alloca i16, align 2, !dbg !80
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %s.var), !dbg !77
// CHECK:STDOUT:   call void @_ZNK5EnumBcv1SEv.carbon_thunk._(ptr %b, ptr %s.var), !dbg !77
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %i.var), !dbg !80
// CHECK:STDOUT:   call void @_ZNK5EnumBcv1IEv.carbon_thunk._(ptr %b, ptr %i.var), !dbg !80
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.c37502bfbac78a47(ptr %i.var), !dbg !80
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.d2b3e68d186d04ec(ptr %s.var), !dbg !77
// CHECK:STDOUT:   ret void, !dbg !83
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZNK5EnumBcv1SEv.carbon_thunk._(ptr noundef nonnull align 1 dereferenceable(1) %this, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !119
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !120
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !120
// CHECK:STDOUT:   %1 = load ptr, ptr %this.addr, align 8, !tbaa !119, !nonnull !61
// CHECK:STDOUT:   %call = call noundef signext i16 @_ZNK5EnumBcv1SEv(ptr noundef nonnull align 1 dereferenceable(1) %1)
// CHECK:STDOUT:   store i16 %call, ptr %0, align 2, !tbaa !123
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZNK5EnumBcv1IEv.carbon_thunk._(ptr noundef nonnull align 1 dereferenceable(1) %this, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !119
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !120
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !120
// CHECK:STDOUT:   %1 = load ptr, ptr %this.addr, align 8, !tbaa !119, !nonnull !61
// CHECK:STDOUT:   %call = call noundef signext i16 @_ZNK5EnumBcv1IEv(ptr noundef nonnull align 1 dereferenceable(1) %1)
// CHECK:STDOUT:   store i16 %call, ptr %0, align 2, !tbaa !125
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:DestroyImpl.28a2429900bb1060.Core:DestroyImpl.Core"(ptr %self) #2 !dbg !85 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.2 = load i16, ptr %self, align 2, !dbg !87
// CHECK:STDOUT:   ret void, !dbg !87
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:DestroyImpl.565367360be60f20.Core:DestroyImpl.Core"(ptr %self) #2 !dbg !89 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.2 = load i16, ptr %self, align 2, !dbg !91
// CHECK:STDOUT:   ret void, !dbg !91
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvertToValue.Main(ptr %b) #0 !dbg !93 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc20_25.1.temp = alloca i16, align 2, !dbg !96
// CHECK:STDOUT:   %.loc21_25.1.temp = alloca i16, align 2, !dbg !99
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc20_25.1.temp), !dbg !96
// CHECK:STDOUT:   call void @_ZNK5EnumBcv1SEv.carbon_thunk._(ptr %b, ptr %.loc20_25.1.temp), !dbg !96
// CHECK:STDOUT:   %.loc20_25.5 = load i16, ptr %.loc20_25.1.temp, align 2, !dbg !96
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc21_25.1.temp), !dbg !99
// CHECK:STDOUT:   call void @_ZNK5EnumBcv1IEv.carbon_thunk._(ptr %b, ptr %.loc21_25.1.temp), !dbg !99
// CHECK:STDOUT:   %.loc21_25.5 = load i16, ptr %.loc21_25.1.temp, align 2, !dbg !99
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.c37502bfbac78a47(ptr %.loc21_25.1.temp), !dbg !99
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.d2b3e68d186d04ec(ptr %.loc20_25.1.temp), !dbg !96
// CHECK:STDOUT:   ret void, !dbg !102
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.c37502bfbac78a47(ptr %t) #0 !dbg !104 {
// CHECK:STDOUT:   call void @"_COp:thunk:DestroyImpl.28a2429900bb1060.Core:DestroyImpl.Core"(ptr %t), !dbg !107
// CHECK:STDOUT:   ret void, !dbg !109
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.d2b3e68d186d04ec(ptr %t) #0 !dbg !111 {
// CHECK:STDOUT:   call void @"_COp:thunk:DestroyImpl.565367360be60f20.Core:DestroyImpl.Core"(ptr %t), !dbg !114
// CHECK:STDOUT:   ret void, !dbg !116
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef signext i16 @_ZNK5EnumBcv1SEv(ptr noundef nonnull align 1 dereferenceable(1)) #4
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef signext i16 @_ZNK5EnumBcv1IEv(ptr noundef nonnull align 1 dereferenceable(1)) #4
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.c37502bfbac78a47, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.d2b3e68d186d04ec, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #4 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!70}
// CHECK:STDOUT: !llvm.module.flags = !{!64, !65, !66, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!68}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{null, !21}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !53 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !54 = !{!"omnipotent char", !53, i64 0}
// CHECK:STDOUT: !55 = !{!"any pointer", !54, i64 0}
// CHECK:STDOUT: !61 = !{}
// CHECK:STDOUT: !62 = !{!"int", !54, i64 0}
// CHECK:STDOUT: !64 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !65 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !66 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !67 = !{!"__libc_errno", !62, i64 0}
// CHECK:STDOUT: !68 = !{!67, !62, i64 0}
// CHECK:STDOUT: !69 = !DIFile(filename: "class_to_enum.carbon", directory: "")
// CHECK:STDOUT: !70 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !69, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !71 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_signed)
// CHECK:STDOUT: !72 = !{null, !71}
// CHECK:STDOUT: !73 = !DISubroutineType(types: !72)
// CHECK:STDOUT: !74 = distinct !DISubprogram(name: "ConvertToVar", linkageName: "_CConvertToVar.Main", scope: null, file: !69, line: 14, type: !23, spFlags: DISPFlagDefinition, unit: !70, retainedNodes: !84)
// CHECK:STDOUT: !75 = !DILocalVariable(arg: 1, scope: !74, type: !21)
// CHECK:STDOUT: !77 = !DILocation(line: 15, column: 3, scope: !74)
// CHECK:STDOUT: !80 = !DILocation(line: 16, column: 3, scope: !74)
// CHECK:STDOUT: !83 = !DILocation(line: 14, column: 1, scope: !74)
// CHECK:STDOUT: !84 = !{!75}
// CHECK:STDOUT: !85 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:DestroyImpl.28a2429900bb1060.Core:DestroyImpl.Core", scope: null, file: !69, type: !73, spFlags: DISPFlagDefinition, unit: !70, retainedNodes: !88)
// CHECK:STDOUT: !86 = !DILocalVariable(arg: 1, scope: !85, type: !71)
// CHECK:STDOUT: !87 = !DILocation(line: 0, scope: !85)
// CHECK:STDOUT: !88 = !{!86}
// CHECK:STDOUT: !89 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:DestroyImpl.565367360be60f20.Core:DestroyImpl.Core", scope: null, file: !69, type: !73, spFlags: DISPFlagDefinition, unit: !70, retainedNodes: !92)
// CHECK:STDOUT: !90 = !DILocalVariable(arg: 1, scope: !89, type: !71)
// CHECK:STDOUT: !91 = !DILocation(line: 0, scope: !89)
// CHECK:STDOUT: !92 = !{!90}
// CHECK:STDOUT: !93 = distinct !DISubprogram(name: "ConvertToValue", linkageName: "_CConvertToValue.Main", scope: null, file: !69, line: 19, type: !23, spFlags: DISPFlagDefinition, unit: !70, retainedNodes: !103)
// CHECK:STDOUT: !94 = !DILocalVariable(arg: 1, scope: !93, type: !21)
// CHECK:STDOUT: !96 = !DILocation(line: 20, column: 25, scope: !93)
// CHECK:STDOUT: !99 = !DILocation(line: 21, column: 25, scope: !93)
// CHECK:STDOUT: !102 = !DILocation(line: 19, column: 1, scope: !93)
// CHECK:STDOUT: !103 = !{!94}
// CHECK:STDOUT: !104 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.c37502bfbac78a47", scope: null, file: !10, line: 26, type: !73, spFlags: DISPFlagDefinition, unit: !70, retainedNodes: !110)
// CHECK:STDOUT: !105 = !DILocalVariable(arg: 1, scope: !104, type: !71)
// CHECK:STDOUT: !107 = !DILocation(line: 27, column: 5, scope: !104)
// CHECK:STDOUT: !109 = !DILocation(line: 26, column: 1, scope: !104)
// CHECK:STDOUT: !110 = !{!105}
// CHECK:STDOUT: !111 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.d2b3e68d186d04ec", scope: null, file: !10, line: 26, type: !73, spFlags: DISPFlagDefinition, unit: !70, retainedNodes: !117)
// CHECK:STDOUT: !112 = !DILocalVariable(arg: 1, scope: !111, type: !71)
// CHECK:STDOUT: !114 = !DILocation(line: 27, column: 5, scope: !111)
// CHECK:STDOUT: !116 = !DILocation(line: 26, column: 1, scope: !111)
// CHECK:STDOUT: !117 = !{!112}
// CHECK:STDOUT: !118 = !{!"p1 _ZTS5EnumB", !55, i64 0}
// CHECK:STDOUT: !119 = !{!118, !118, i64 0}
// CHECK:STDOUT: !120 = !{!55, !55, i64 0}
// CHECK:STDOUT: !121 = !{!"_ZTS1S", !54, i64 0}
// CHECK:STDOUT: !123 = !{!121, !121, i64 0}
// CHECK:STDOUT: !124 = !{!"_ZTS1I", !54, i64 0}
// CHECK:STDOUT: !125 = !{!124, !124, i64 0}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'class_to_int.carbon'
// CHECK:STDOUT: source_filename = "class_to_int.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvertToVar.Main(ptr %b) #0 !dbg !130 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca i32, align 4, !dbg !133
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !133
// CHECK:STDOUT:   %__carbon_thunk.call = call i32 @_ZNK4IntBcviEv.carbon_thunk._(ptr %b), !dbg !133
// CHECK:STDOUT:   store i32 %__carbon_thunk.call, ptr %a.var, align 4, !dbg !133
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.0aefc5da911f5051(ptr %a.var), !dbg !133
// CHECK:STDOUT:   ret void, !dbg !138
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal noundef i32 @_ZNK4IntBcviEv.carbon_thunk._(ptr noundef nonnull align 1 dereferenceable(1) %this) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !156
// CHECK:STDOUT:   %0 = load ptr, ptr %this.addr, align 8, !tbaa !156, !nonnull !61
// CHECK:STDOUT:   %call = call noundef i32 @_ZNK4IntBcviEv(ptr noundef nonnull align 1 dereferenceable(1) %0)
// CHECK:STDOUT:   ret i32 %call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvertToValue.Main(ptr %b) #0 !dbg !140 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %__carbon_thunk.call = call i32 @_ZNK4IntBcviEv.carbon_thunk._(ptr %b), !dbg !143
// CHECK:STDOUT:   ret void, !dbg !146
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.0aefc5da911f5051(ptr %t) #0 !dbg !148 {
// CHECK:STDOUT:   ret void, !dbg !153
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef i32 @_ZNK4IntBcviEv(ptr noundef nonnull align 1 dereferenceable(1)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!127}
// CHECK:STDOUT: !llvm.module.flags = !{!64, !65, !66, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!68}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{null, !21}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !53 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !54 = !{!"omnipotent char", !53, i64 0}
// CHECK:STDOUT: !55 = !{!"any pointer", !54, i64 0}
// CHECK:STDOUT: !61 = !{}
// CHECK:STDOUT: !62 = !{!"int", !54, i64 0}
// CHECK:STDOUT: !64 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !65 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !66 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !67 = !{!"__libc_errno", !62, i64 0}
// CHECK:STDOUT: !68 = !{!67, !62, i64 0}
// CHECK:STDOUT: !126 = !DIFile(filename: "class_to_int.carbon", directory: "")
// CHECK:STDOUT: !127 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !126, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !130 = distinct !DISubprogram(name: "ConvertToVar", linkageName: "_CConvertToVar.Main", scope: null, file: !126, line: 11, type: !23, spFlags: DISPFlagDefinition, unit: !127, retainedNodes: !139)
// CHECK:STDOUT: !131 = !DILocalVariable(arg: 1, scope: !130, type: !21)
// CHECK:STDOUT: !133 = !DILocation(line: 12, column: 3, scope: !130)
// CHECK:STDOUT: !136 = !{null, !20}
// CHECK:STDOUT: !137 = !DISubroutineType(types: !136)
// CHECK:STDOUT: !138 = !DILocation(line: 11, column: 1, scope: !130)
// CHECK:STDOUT: !139 = !{!131}
// CHECK:STDOUT: !140 = distinct !DISubprogram(name: "ConvertToValue", linkageName: "_CConvertToValue.Main", scope: null, file: !126, line: 15, type: !23, spFlags: DISPFlagDefinition, unit: !127, retainedNodes: !147)
// CHECK:STDOUT: !141 = !DILocalVariable(arg: 1, scope: !140, type: !21)
// CHECK:STDOUT: !143 = !DILocation(line: 16, column: 27, scope: !140)
// CHECK:STDOUT: !146 = !DILocation(line: 15, column: 1, scope: !140)
// CHECK:STDOUT: !147 = !{!141}
// CHECK:STDOUT: !148 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.0aefc5da911f5051", scope: null, file: !10, line: 26, type: !137, spFlags: DISPFlagDefinition, unit: !127, retainedNodes: !154)
// CHECK:STDOUT: !149 = !DILocalVariable(arg: 1, scope: !148, type: !20)
// CHECK:STDOUT: !153 = !DILocation(line: 26, column: 1, scope: !148)
// CHECK:STDOUT: !154 = !{!149}
// CHECK:STDOUT: !155 = !{!"p1 _ZTS4IntB", !55, i64 0}
// CHECK:STDOUT: !156 = !{!155, !155, i64 0}
// CHECK:STDOUT:
