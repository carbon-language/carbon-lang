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
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.ad394d08bf7f4546(ptr %a.var), !dbg !29
// CHECK:STDOUT:   ret void, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZNK1Bcv1AEv.carbon_thunk._(ptr noundef nonnull align 1 dereferenceable(1) %this, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !53
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !55
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !55
// CHECK:STDOUT:   %1 = load ptr, ptr %this.addr, align 8, !tbaa !53, !nonnull !57
// CHECK:STDOUT:   call void @_ZNK1Bcv1AEv(ptr dead_on_unwind writable sret(%struct.A) align 4 %0, ptr noundef nonnull align 1 dereferenceable(1) %1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: declare void @_ZN1AD1Ev(ptr noundef nonnull align 4 dead_on_return(4) dereferenceable(4)) unnamed_addr #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvertToValue.Main(ptr %b) #0 !dbg !34 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc20_25.1.temp = alloca [4 x i8], align 4, !dbg !37
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc20_25.1.temp), !dbg !37
// CHECK:STDOUT:   call void @_ZNK1Bcv1AEv.carbon_thunk._(ptr %b, ptr %.loc20_25.1.temp), !dbg !37
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.ad394d08bf7f4546(ptr %.loc20_25.1.temp), !dbg !37
// CHECK:STDOUT:   ret void, !dbg !40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.ad394d08bf7f4546(ptr %self) #0 !dbg !42 {
// CHECK:STDOUT:   call void @_ZN1AD1Ev(ptr %self), !dbg !46
// CHECK:STDOUT:   ret void, !dbg !47
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_ZNK1Bcv1AEv(ptr dead_on_unwind writable sret(%struct.A) align 4, ptr noundef nonnull align 1 dereferenceable(1)) #4
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Destroy.Core.ad394d08bf7f4546, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #4 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!60, !61, !62, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!64}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !14 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
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
// CHECK:STDOUT: !34 = distinct !DISubprogram(name: "ConvertToValue", linkageName: "_CConvertToValue.Main", scope: null, file: !18, line: 19, type: !23, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !41)
// CHECK:STDOUT: !35 = !DILocalVariable(arg: 1, scope: !34, type: !21)
// CHECK:STDOUT: !37 = !DILocation(line: 20, column: 25, scope: !34)
// CHECK:STDOUT: !40 = !DILocation(line: 19, column: 1, scope: !34)
// CHECK:STDOUT: !41 = !{!35}
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.ad394d08bf7f4546", scope: null, file: !14, line: 24, type: !23, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !48)
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 1, scope: !42, type: !21)
// CHECK:STDOUT: !46 = !DILocation(line: 26, column: 5, scope: !42)
// CHECK:STDOUT: !47 = !DILocation(line: 24, column: 3, scope: !42)
// CHECK:STDOUT: !48 = !{!43}
// CHECK:STDOUT: !49 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !50 = !{!"omnipotent char", !49, i64 0}
// CHECK:STDOUT: !51 = !{!"any pointer", !50, i64 0}
// CHECK:STDOUT: !52 = !{!"p1 _ZTS1B", !51, i64 0}
// CHECK:STDOUT: !53 = !{!52, !52, i64 0}
// CHECK:STDOUT: !54 = !{!"p1 _ZTS1A", !51, i64 0}
// CHECK:STDOUT: !55 = !{!54, !54, i64 0}
// CHECK:STDOUT: !57 = !{}
// CHECK:STDOUT: !58 = !{!"int", !50, i64 0}
// CHECK:STDOUT: !60 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !61 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !62 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !63 = !{!"__libc_errno", !58, i64 0}
// CHECK:STDOUT: !64 = !{!63, !58, i64 0}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'class_to_enum.carbon'
// CHECK:STDOUT: source_filename = "class_to_enum.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvertToVar.Main(ptr %b) #0 !dbg !68 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %s.var = alloca i16, align 2, !dbg !71
// CHECK:STDOUT:   %i.var = alloca i16, align 2, !dbg !74
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %s.var), !dbg !71
// CHECK:STDOUT:   call void @_ZNK5EnumBcv1SEv.carbon_thunk._(ptr %b, ptr %s.var), !dbg !71
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %i.var), !dbg !74
// CHECK:STDOUT:   call void @_ZNK5EnumBcv1IEv.carbon_thunk._(ptr %b, ptr %i.var), !dbg !74
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.db33789924763105(ptr %i.var), !dbg !74
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.7283d87e25af0531(ptr %s.var), !dbg !71
// CHECK:STDOUT:   ret void, !dbg !79
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZNK5EnumBcv1SEv.carbon_thunk._(ptr noundef nonnull align 1 dereferenceable(1) %this, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !107
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !108
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !108
// CHECK:STDOUT:   %1 = load ptr, ptr %this.addr, align 8, !tbaa !107, !nonnull !57
// CHECK:STDOUT:   %call = call noundef signext i16 @_ZNK5EnumBcv1SEv(ptr noundef nonnull align 1 dereferenceable(1) %1)
// CHECK:STDOUT:   store i16 %call, ptr %0, align 2, !tbaa !111
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZNK5EnumBcv1IEv.carbon_thunk._(ptr noundef nonnull align 1 dereferenceable(1) %this, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !107
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !108
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !108
// CHECK:STDOUT:   %1 = load ptr, ptr %this.addr, align 8, !tbaa !107, !nonnull !57
// CHECK:STDOUT:   %call = call noundef signext i16 @_ZNK5EnumBcv1IEv(ptr noundef nonnull align 1 dereferenceable(1) %1)
// CHECK:STDOUT:   store i16 %call, ptr %0, align 2, !tbaa !113
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvertToValue.Main(ptr %b) #0 !dbg !81 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc20_25.1.temp = alloca i16, align 2, !dbg !84
// CHECK:STDOUT:   %.loc21_25.1.temp = alloca i16, align 2, !dbg !87
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc20_25.1.temp), !dbg !84
// CHECK:STDOUT:   call void @_ZNK5EnumBcv1SEv.carbon_thunk._(ptr %b, ptr %.loc20_25.1.temp), !dbg !84
// CHECK:STDOUT:   %.loc20_25.5 = load i16, ptr %.loc20_25.1.temp, align 2, !dbg !84
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc21_25.1.temp), !dbg !87
// CHECK:STDOUT:   call void @_ZNK5EnumBcv1IEv.carbon_thunk._(ptr %b, ptr %.loc21_25.1.temp), !dbg !87
// CHECK:STDOUT:   %.loc21_25.5 = load i16, ptr %.loc21_25.1.temp, align 2, !dbg !87
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.db33789924763105(ptr %.loc21_25.1.temp), !dbg !87
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.7283d87e25af0531(ptr %.loc20_25.1.temp), !dbg !84
// CHECK:STDOUT:   ret void, !dbg !90
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.db33789924763105(ptr %self) #0 !dbg !92 {
// CHECK:STDOUT:   ret void, !dbg !97
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.7283d87e25af0531(ptr %self) #0 !dbg !99 {
// CHECK:STDOUT:   ret void, !dbg !104
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef signext i16 @_ZNK5EnumBcv1SEv(ptr noundef nonnull align 1 dereferenceable(1)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef signext i16 @_ZNK5EnumBcv1IEv(ptr noundef nonnull align 1 dereferenceable(1)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Destroy.Core.db33789924763105, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Destroy.Core.7283d87e25af0531, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!66}
// CHECK:STDOUT: !llvm.module.flags = !{!60, !61, !62, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!64}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !14 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{null, !21}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !49 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !50 = !{!"omnipotent char", !49, i64 0}
// CHECK:STDOUT: !51 = !{!"any pointer", !50, i64 0}
// CHECK:STDOUT: !57 = !{}
// CHECK:STDOUT: !58 = !{!"int", !50, i64 0}
// CHECK:STDOUT: !60 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !61 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !62 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !63 = !{!"__libc_errno", !58, i64 0}
// CHECK:STDOUT: !64 = !{!63, !58, i64 0}
// CHECK:STDOUT: !65 = !DIFile(filename: "class_to_enum.carbon", directory: "")
// CHECK:STDOUT: !66 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !65, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !67 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_signed)
// CHECK:STDOUT: !68 = distinct !DISubprogram(name: "ConvertToVar", linkageName: "_CConvertToVar.Main", scope: null, file: !65, line: 14, type: !23, spFlags: DISPFlagDefinition, unit: !66, retainedNodes: !80)
// CHECK:STDOUT: !69 = !DILocalVariable(arg: 1, scope: !68, type: !21)
// CHECK:STDOUT: !71 = !DILocation(line: 15, column: 3, scope: !68)
// CHECK:STDOUT: !74 = !DILocation(line: 16, column: 3, scope: !68)
// CHECK:STDOUT: !77 = !{null, !67}
// CHECK:STDOUT: !78 = !DISubroutineType(types: !77)
// CHECK:STDOUT: !79 = !DILocation(line: 14, column: 1, scope: !68)
// CHECK:STDOUT: !80 = !{!69}
// CHECK:STDOUT: !81 = distinct !DISubprogram(name: "ConvertToValue", linkageName: "_CConvertToValue.Main", scope: null, file: !65, line: 19, type: !23, spFlags: DISPFlagDefinition, unit: !66, retainedNodes: !91)
// CHECK:STDOUT: !82 = !DILocalVariable(arg: 1, scope: !81, type: !21)
// CHECK:STDOUT: !84 = !DILocation(line: 20, column: 25, scope: !81)
// CHECK:STDOUT: !87 = !DILocation(line: 21, column: 25, scope: !81)
// CHECK:STDOUT: !90 = !DILocation(line: 19, column: 1, scope: !81)
// CHECK:STDOUT: !91 = !{!82}
// CHECK:STDOUT: !92 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.db33789924763105", scope: null, file: !14, line: 24, type: !78, spFlags: DISPFlagDefinition, unit: !66, retainedNodes: !98)
// CHECK:STDOUT: !93 = !DILocalVariable(arg: 1, scope: !92, type: !67)
// CHECK:STDOUT: !97 = !DILocation(line: 24, column: 3, scope: !92)
// CHECK:STDOUT: !98 = !{!93}
// CHECK:STDOUT: !99 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.7283d87e25af0531", scope: null, file: !14, line: 24, type: !78, spFlags: DISPFlagDefinition, unit: !66, retainedNodes: !105)
// CHECK:STDOUT: !100 = !DILocalVariable(arg: 1, scope: !99, type: !67)
// CHECK:STDOUT: !104 = !DILocation(line: 24, column: 3, scope: !99)
// CHECK:STDOUT: !105 = !{!100}
// CHECK:STDOUT: !106 = !{!"p1 _ZTS5EnumB", !51, i64 0}
// CHECK:STDOUT: !107 = !{!106, !106, i64 0}
// CHECK:STDOUT: !108 = !{!51, !51, i64 0}
// CHECK:STDOUT: !109 = !{!"_ZTS1S", !50, i64 0}
// CHECK:STDOUT: !111 = !{!109, !109, i64 0}
// CHECK:STDOUT: !112 = !{!"_ZTS1I", !50, i64 0}
// CHECK:STDOUT: !113 = !{!112, !112, i64 0}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'class_to_int.carbon'
// CHECK:STDOUT: source_filename = "class_to_int.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvertToVar.Main(ptr %b) #0 !dbg !120 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca i32, align 4, !dbg !123
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !123
// CHECK:STDOUT:   %__carbon_thunk.call = call i32 @_ZNK4IntBcviEv.carbon_thunk._(ptr %b), !dbg !123
// CHECK:STDOUT:   store i32 %__carbon_thunk.call, ptr %a.var, align 4, !dbg !123
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48(ptr %a.var), !dbg !123
// CHECK:STDOUT:   ret void, !dbg !126
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal noundef i32 @_ZNK4IntBcviEv.carbon_thunk._(ptr noundef nonnull align 1 dereferenceable(1) %this) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !148
// CHECK:STDOUT:   %0 = load ptr, ptr %this.addr, align 8, !tbaa !148, !nonnull !57
// CHECK:STDOUT:   %call = call noundef i32 @_ZNK4IntBcviEv(ptr noundef nonnull align 1 dereferenceable(1) %0)
// CHECK:STDOUT:   ret i32 %call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !128 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !130
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvertToValue.Main(ptr %b) #0 !dbg !132 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %__carbon_thunk.call = call i32 @_ZNK4IntBcviEv.carbon_thunk._(ptr %b), !dbg !135
// CHECK:STDOUT:   ret void, !dbg !138
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48(ptr %self) #0 !dbg !140 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self), !dbg !144
// CHECK:STDOUT:   ret void, !dbg !145
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef i32 @_ZNK4IntBcviEv(ptr noundef nonnull align 1 dereferenceable(1)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!115}
// CHECK:STDOUT: !llvm.module.flags = !{!60, !61, !62, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!64}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !14 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{null, !21}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !49 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !50 = !{!"omnipotent char", !49, i64 0}
// CHECK:STDOUT: !51 = !{!"any pointer", !50, i64 0}
// CHECK:STDOUT: !57 = !{}
// CHECK:STDOUT: !58 = !{!"int", !50, i64 0}
// CHECK:STDOUT: !60 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !61 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !62 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !63 = !{!"__libc_errno", !58, i64 0}
// CHECK:STDOUT: !64 = !{!63, !58, i64 0}
// CHECK:STDOUT: !114 = !DIFile(filename: "class_to_int.carbon", directory: "")
// CHECK:STDOUT: !115 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !114, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !118 = !{null, !20}
// CHECK:STDOUT: !119 = !DISubroutineType(types: !118)
// CHECK:STDOUT: !120 = distinct !DISubprogram(name: "ConvertToVar", linkageName: "_CConvertToVar.Main", scope: null, file: !114, line: 11, type: !23, spFlags: DISPFlagDefinition, unit: !115, retainedNodes: !127)
// CHECK:STDOUT: !121 = !DILocalVariable(arg: 1, scope: !120, type: !21)
// CHECK:STDOUT: !123 = !DILocation(line: 12, column: 3, scope: !120)
// CHECK:STDOUT: !126 = !DILocation(line: 11, column: 1, scope: !120)
// CHECK:STDOUT: !127 = !{!121}
// CHECK:STDOUT: !128 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core", scope: null, file: !114, line: 12, type: !119, spFlags: DISPFlagDefinition, unit: !115, retainedNodes: !131)
// CHECK:STDOUT: !129 = !DILocalVariable(arg: 1, scope: !128, type: !20)
// CHECK:STDOUT: !130 = !DILocation(line: 12, column: 3, scope: !128)
// CHECK:STDOUT: !131 = !{!129}
// CHECK:STDOUT: !132 = distinct !DISubprogram(name: "ConvertToValue", linkageName: "_CConvertToValue.Main", scope: null, file: !114, line: 15, type: !23, spFlags: DISPFlagDefinition, unit: !115, retainedNodes: !139)
// CHECK:STDOUT: !133 = !DILocalVariable(arg: 1, scope: !132, type: !21)
// CHECK:STDOUT: !135 = !DILocation(line: 16, column: 27, scope: !132)
// CHECK:STDOUT: !138 = !DILocation(line: 15, column: 1, scope: !132)
// CHECK:STDOUT: !139 = !{!133}
// CHECK:STDOUT: !140 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48", scope: null, file: !14, line: 24, type: !119, spFlags: DISPFlagDefinition, unit: !115, retainedNodes: !146)
// CHECK:STDOUT: !141 = !DILocalVariable(arg: 1, scope: !140, type: !20)
// CHECK:STDOUT: !144 = !DILocation(line: 26, column: 5, scope: !140)
// CHECK:STDOUT: !145 = !DILocation(line: 24, column: 3, scope: !140)
// CHECK:STDOUT: !146 = !{!141}
// CHECK:STDOUT: !147 = !{!"p1 _ZTS4IntB", !51, i64 0}
// CHECK:STDOUT: !148 = !{!147, !147, i64 0}
// CHECK:STDOUT:
