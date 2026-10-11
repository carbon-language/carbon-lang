// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/class/import/constructor.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/class/import/constructor.carbon

// ============================================================================
// Default constructor
// ============================================================================

// --- default.h

class C {
 public:
  C() : x_(8), y_(9) {}

 private:
  int x_;
  int y_;
};

// --- import_default.carbon

library "[[@TEST_NAME]]";

import Cpp library "default.h";

fn F() {
  let _: Cpp.C = Cpp.C.C();
}

fn G() {
  let _: Cpp.C = ();
}

// --- implicit_default.carbon

library "[[@TEST_NAME]]";

import Cpp library "default.h";

fn F() {
  var _: Cpp.C;
}

// --- defaulted_default.h

struct ImplicitlyDefaultedDefault {
  int x_ = 1;
};

struct ExplicitlyDefaultedDefault {
  ExplicitlyDefaultedDefault() = default;
  int x_ = 1;
};

// --- defaulted_default.carbon

library "[[@TEST_NAME]]";

import Cpp library "defaulted_default.h";

fn F() {
  let _: Cpp.ImplicitlyDefaultedDefault =
      Cpp.ImplicitlyDefaultedDefault.ImplicitlyDefaultedDefault();
  var _: Cpp.ImplicitlyDefaultedDefault;

  let _: Cpp.ExplicitlyDefaultedDefault =
      Cpp.ExplicitlyDefaultedDefault.ExplicitlyDefaultedDefault();
  var _: Cpp.ExplicitlyDefaultedDefault;
}

// ============================================================================
// Copy constructor
// ============================================================================

// --- copy.h

class Copy {
 public:
  Copy();
  Copy(const Copy&);

 private:
  int n;
};

// --- call_copy.carbon

library "[[@TEST_NAME]]";

import Cpp library "copy.h";

fn Copy(c: Cpp.Copy) -> Cpp.Copy {
  return c;
}

// ============================================================================
// C1 constructor
// ============================================================================

// --- header.h

struct Base { };
struct Derived: virtual Base {
};

// --- call_c1_in_thunk.carbon

library "[[@TEST_NAME]]";

import Cpp library "header.h";

fn F() {
  let _: Cpp.Derived = Cpp.Derived.Derived();
}

// ============================================================================
// Multiple arguments
// ============================================================================

// --- multi_argument.h

class MultiArgC {
 public:
  MultiArgC() {}
  MultiArgC(int) {}
  MultiArgC(int, int) {}
  MultiArgC(int, int, int, int = 0) {}
};

// --- call_multi_argument.carbon

library "[[@TEST_NAME]]";

import Cpp library "multi_argument.h";

fn Zero() {
  let _: Cpp.MultiArgC = Cpp.MultiArgC.MultiArgC();
  let _: Cpp.MultiArgC = ();
}

fn One() {
  let _: Cpp.MultiArgC = Cpp.MultiArgC.MultiArgC(1);
  let _: Cpp.MultiArgC = 2;
  let _: Cpp.MultiArgC = (3,);
}

fn Two() {
  let _: Cpp.MultiArgC = Cpp.MultiArgC.MultiArgC(1, 2);
  let _: Cpp.MultiArgC = (3, 4);
}

fn Three() {
  let _: Cpp.MultiArgC = Cpp.MultiArgC.MultiArgC(1, 2, 3);
  let _: Cpp.MultiArgC = (4, 5, 6);
}

fn Four() {
  let _: Cpp.MultiArgC = Cpp.MultiArgC.MultiArgC(1, 2, 3, 4);
  let _: Cpp.MultiArgC = (5, 6, 7, 8);
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'import_default.carbon'
// CHECK:STDOUT: source_filename = "import_default.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %class.C = type { i32, i32 }
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN1CC2Ev = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !26 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_26.1.temp = alloca [8 x i8], align 4, !dbg !27
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_26.1.temp), !dbg !27
// CHECK:STDOUT:   call void @_ZN1CC1Ev.carbon_thunk.(ptr %.loc7_26.1.temp), !dbg !27
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.075fdbe7169ee367(ptr %.loc7_26.1.temp), !dbg !27
// CHECK:STDOUT:   ret void, !dbg !30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN1CC1Ev.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !51
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !51
// CHECK:STDOUT:   call void @_ZN1CC2Ev(ptr noundef nonnull align 4 dereferenceable(8) %0)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:DestroyImpl.75f7c69b6a29abf8.Core:DestroyImpl.Core"(ptr %self) #2 !dbg !31 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !35 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc11_19.2.temp = alloca [8 x i8], align 4, !dbg !36
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc11_19.2.temp), !dbg !36
// CHECK:STDOUT:   call void @_ZN1CC1Ev.carbon_thunk_tuple.(ptr %.loc11_19.2.temp), !dbg !36
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.075fdbe7169ee367(ptr %.loc11_19.2.temp), !dbg !36
// CHECK:STDOUT:   ret void, !dbg !39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN1CC1Ev.carbon_thunk_tuple.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !51
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !51
// CHECK:STDOUT:   call void @_ZN1CC2Ev(ptr noundef nonnull align 4 dereferenceable(8) %0)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.075fdbe7169ee367(ptr %t) #0 !dbg !40 {
// CHECK:STDOUT:   call void @"_COp:thunk:DestroyImpl.75f7c69b6a29abf8.Core:DestroyImpl.Core"(ptr %t), !dbg !43
// CHECK:STDOUT:   ret void, !dbg !45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN1CC2Ev(ptr noundef nonnull align 4 dereferenceable(8) %this) unnamed_addr #4 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !51
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   %x_ = getelementptr inbounds nuw %class.C, ptr %this1, i32 0, i32 0
// CHECK:STDOUT:   store i32 8, ptr %x_, align 4, !tbaa !54
// CHECK:STDOUT:   %y_ = getelementptr inbounds nuw %class.C, ptr %this1, i32 0, i32 1
// CHECK:STDOUT:   store i32 9, ptr %y_, align 4, !tbaa !55
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.075fdbe7169ee367, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #4 = { mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!56, !57, !58, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!60}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !18 = !DIFile(filename: "import_default.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !21 = !{null}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{null, !23}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !18, line: 6, type: !22, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !27 = !DILocation(line: 7, column: 18, scope: !26)
// CHECK:STDOUT: !30 = !DILocation(line: 6, column: 1, scope: !26)
// CHECK:STDOUT: !31 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:DestroyImpl.75f7c69b6a29abf8.Core:DestroyImpl.Core", scope: null, file: !18, type: !25, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !34)
// CHECK:STDOUT: !32 = !DILocalVariable(arg: 1, scope: !31, type: !23)
// CHECK:STDOUT: !33 = !DILocation(line: 0, scope: !31)
// CHECK:STDOUT: !34 = !{!32}
// CHECK:STDOUT: !35 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !18, line: 10, type: !22, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !36 = !DILocation(line: 11, column: 18, scope: !35)
// CHECK:STDOUT: !39 = !DILocation(line: 10, column: 1, scope: !35)
// CHECK:STDOUT: !40 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.075fdbe7169ee367", scope: null, file: !10, line: 26, type: !25, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !46)
// CHECK:STDOUT: !41 = !DILocalVariable(arg: 1, scope: !40, type: !23)
// CHECK:STDOUT: !43 = !DILocation(line: 27, column: 5, scope: !40)
// CHECK:STDOUT: !45 = !DILocation(line: 26, column: 1, scope: !40)
// CHECK:STDOUT: !46 = !{!41}
// CHECK:STDOUT: !47 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !48 = !{!"omnipotent char", !47, i64 0}
// CHECK:STDOUT: !49 = !{!"any pointer", !48, i64 0}
// CHECK:STDOUT: !50 = !{!"p1 _ZTS1C", !49, i64 0}
// CHECK:STDOUT: !51 = !{!50, !50, i64 0}
// CHECK:STDOUT: !52 = !{!"int", !48, i64 0}
// CHECK:STDOUT: !53 = !{!"_ZTS1C", !52, i64 0, !52, i64 4}
// CHECK:STDOUT: !54 = !{!53, !52, i64 0}
// CHECK:STDOUT: !55 = !{!53, !52, i64 4}
// CHECK:STDOUT: !56 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !57 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !58 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !59 = !{!"__libc_errno", !52, i64 0}
// CHECK:STDOUT: !60 = !{!59, !52, i64 0}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'implicit_default.carbon'
// CHECK:STDOUT: source_filename = "implicit_default.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %class.C.18 = type { i32, i32 }
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN1CC2Ev = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !65 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca [8 x i8], align 4, !dbg !66
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !66
// CHECK:STDOUT:   call void @"_COp.51f3fb62cb986374:DefaultOrUnformed.Core.67b28f83c7cfd60c"(ptr %_.var), !dbg !66
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.075fdbe7169ee367(ptr %_.var), !dbg !66
// CHECK:STDOUT:   ret void, !dbg !68
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN1CC1Ev.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !51
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !51
// CHECK:STDOUT:   call void @_ZN1CC2Ev(ptr noundef nonnull align 4 dereferenceable(8) %0)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:Default.dc3dd9c349162a91.Core:C.Cpp"(ptr sret([8 x i8]) %return) #2 !dbg !70 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN1CC1Ev.carbon_thunk.(ptr %return), !dbg !71
// CHECK:STDOUT:   ret void, !dbg !71
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:DestroyImpl.75f7c69b6a29abf8.Core:DestroyImpl.Core"(ptr %self) #2 !dbg !72 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !74
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.51f3fb62cb986374:DefaultOrUnformed.Core.67b28f83c7cfd60c"(ptr sret([8 x i8]) %return) #0 !dbg !76 {
// CHECK:STDOUT:   call void @"_COp:thunk:Default.dc3dd9c349162a91.Core:C.Cpp"(ptr %return), !dbg !79
// CHECK:STDOUT:   ret void, !dbg !80
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.075fdbe7169ee367(ptr %t) #0 !dbg !81 {
// CHECK:STDOUT:   call void @"_COp:thunk:DestroyImpl.75f7c69b6a29abf8.Core:DestroyImpl.Core"(ptr %t), !dbg !84
// CHECK:STDOUT:   ret void, !dbg !86
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN1CC2Ev(ptr noundef nonnull align 4 dereferenceable(8) %this) unnamed_addr #4 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !51
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   %x_ = getelementptr inbounds nuw %class.C.18, ptr %this1, i32 0, i32 0
// CHECK:STDOUT:   store i32 8, ptr %x_, align 4, !tbaa !54
// CHECK:STDOUT:   %y_ = getelementptr inbounds nuw %class.C.18, ptr %this1, i32 0, i32 1
// CHECK:STDOUT:   store i32 9, ptr %y_, align 4, !tbaa !55
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #4 = { mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!62}
// CHECK:STDOUT: !llvm.module.flags = !{!56, !57, !58, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!60}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !8 = !DIFile(filename: "min_prelude/parts/default.carbon", directory: "")
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !21 = !{null}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{null, !23}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !47 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !48 = !{!"omnipotent char", !47, i64 0}
// CHECK:STDOUT: !49 = !{!"any pointer", !48, i64 0}
// CHECK:STDOUT: !50 = !{!"p1 _ZTS1C", !49, i64 0}
// CHECK:STDOUT: !51 = !{!50, !50, i64 0}
// CHECK:STDOUT: !52 = !{!"int", !48, i64 0}
// CHECK:STDOUT: !53 = !{!"_ZTS1C", !52, i64 0, !52, i64 4}
// CHECK:STDOUT: !54 = !{!53, !52, i64 0}
// CHECK:STDOUT: !55 = !{!53, !52, i64 4}
// CHECK:STDOUT: !56 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !57 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !58 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !59 = !{!"__libc_errno", !52, i64 0}
// CHECK:STDOUT: !60 = !{!59, !52, i64 0}
// CHECK:STDOUT: !61 = !DIFile(filename: "implicit_default.carbon", directory: "")
// CHECK:STDOUT: !62 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !61, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !63 = !{!23}
// CHECK:STDOUT: !64 = !DISubroutineType(types: !63)
// CHECK:STDOUT: !65 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !61, line: 6, type: !22, spFlags: DISPFlagDefinition, unit: !62)
// CHECK:STDOUT: !66 = !DILocation(line: 7, column: 3, scope: !65)
// CHECK:STDOUT: !68 = !DILocation(line: 6, column: 1, scope: !65)
// CHECK:STDOUT: !69 = !DIFile(filename: "./default.h", directory: "")
// CHECK:STDOUT: !70 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:Default.dc3dd9c349162a91.Core:C.Cpp", scope: null, file: !69, line: 4, type: !64, spFlags: DISPFlagDefinition, unit: !62)
// CHECK:STDOUT: !71 = !DILocation(line: 4, column: 3, scope: !70)
// CHECK:STDOUT: !72 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:DestroyImpl.75f7c69b6a29abf8.Core:DestroyImpl.Core", scope: null, file: !61, type: !25, spFlags: DISPFlagDefinition, unit: !62, retainedNodes: !75)
// CHECK:STDOUT: !73 = !DILocalVariable(arg: 1, scope: !72, type: !23)
// CHECK:STDOUT: !74 = !DILocation(line: 0, scope: !72)
// CHECK:STDOUT: !75 = !{!73}
// CHECK:STDOUT: !76 = distinct !DISubprogram(name: "Op", linkageName: "_COp.51f3fb62cb986374:DefaultOrUnformed.Core.67b28f83c7cfd60c", scope: null, file: !8, line: 9, type: !64, spFlags: DISPFlagDefinition, unit: !62)
// CHECK:STDOUT: !79 = !DILocation(line: 9, column: 28, scope: !76)
// CHECK:STDOUT: !80 = !DILocation(line: 9, column: 21, scope: !76)
// CHECK:STDOUT: !81 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.075fdbe7169ee367", scope: null, file: !10, line: 26, type: !25, spFlags: DISPFlagDefinition, unit: !62, retainedNodes: !87)
// CHECK:STDOUT: !82 = !DILocalVariable(arg: 1, scope: !81, type: !23)
// CHECK:STDOUT: !84 = !DILocation(line: 27, column: 5, scope: !81)
// CHECK:STDOUT: !86 = !DILocation(line: 26, column: 1, scope: !81)
// CHECK:STDOUT: !87 = !{!82}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'defaulted_default.carbon'
// CHECK:STDOUT: source_filename = "defaulted_default.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %struct.ImplicitlyDefaultedDefault = type { i32 }
// CHECK:STDOUT: %struct.ExplicitlyDefaultedDefault = type { i32 }
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN26ImplicitlyDefaultedDefaultC2Ev = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN26ExplicitlyDefaultedDefaultC2Ev = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !90 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc8_65.1.temp = alloca [4 x i8], align 4, !dbg !91
// CHECK:STDOUT:   %_.var.loc9 = alloca [4 x i8], align 4, !dbg !94
// CHECK:STDOUT:   %.loc12_65.1.temp = alloca [4 x i8], align 4, !dbg !96
// CHECK:STDOUT:   %_.var.loc13 = alloca [4 x i8], align 4, !dbg !99
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc8_65.1.temp), !dbg !91
// CHECK:STDOUT:   call void @_ZN26ImplicitlyDefaultedDefaultC1Ev.carbon_thunk.(ptr %.loc8_65.1.temp), !dbg !91
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc9), !dbg !94
// CHECK:STDOUT:   call void @"_COp.51f3fb62cb986374:DefaultOrUnformed.Core.c20bae04b665a8c9"(ptr %_.var.loc9), !dbg !94
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc12_65.1.temp), !dbg !96
// CHECK:STDOUT:   call void @_ZN26ExplicitlyDefaultedDefaultC1Ev.carbon_thunk.(ptr %.loc12_65.1.temp), !dbg !96
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc13), !dbg !99
// CHECK:STDOUT:   call void @"_COp.51f3fb62cb986374:DefaultOrUnformed.Core.055ae7235dade271"(ptr %_.var.loc13), !dbg !99
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.3d7f655e9f020542(ptr %_.var.loc13), !dbg !99
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.3d7f655e9f020542(ptr %.loc12_65.1.temp), !dbg !96
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.2968cd7107a26841(ptr %_.var.loc9), !dbg !94
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.2968cd7107a26841(ptr %.loc8_65.1.temp), !dbg !91
// CHECK:STDOUT:   ret void, !dbg !101
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN26ImplicitlyDefaultedDefaultC1Ev.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !140
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !140
// CHECK:STDOUT:   call void @_ZN26ImplicitlyDefaultedDefaultC2Ev(ptr noundef nonnull align 4 dereferenceable(4) %0) #0
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:Default.48fb4c4d31b7d0bf.Core:ImplicitlyDefaultedDefault.Cpp"(ptr sret([4 x i8]) %return) #2 !dbg !103 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN26ImplicitlyDefaultedDefaultC1Ev.carbon_thunk.(ptr %return), !dbg !104
// CHECK:STDOUT:   ret void, !dbg !104
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN26ExplicitlyDefaultedDefaultC1Ev.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !144
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !144
// CHECK:STDOUT:   call void @_ZN26ExplicitlyDefaultedDefaultC2Ev(ptr noundef nonnull align 4 dereferenceable(4) %0) #0
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:Default.f7ca6602362234ee.Core:ExplicitlyDefaultedDefault.Cpp"(ptr sret([4 x i8]) %return) #2 !dbg !105 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN26ExplicitlyDefaultedDefaultC1Ev.carbon_thunk.(ptr %return), !dbg !106
// CHECK:STDOUT:   ret void, !dbg !106
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:DestroyImpl.40d7fdd10b56fa91.Core:DestroyImpl.Core"(ptr %self) #2 !dbg !107 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !109
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:DestroyImpl.a73ffbfc339da3af.Core:DestroyImpl.Core"(ptr %self) #2 !dbg !111 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !113
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.51f3fb62cb986374:DefaultOrUnformed.Core.c20bae04b665a8c9"(ptr sret([4 x i8]) %return) #0 !dbg !115 {
// CHECK:STDOUT:   call void @"_COp:thunk:Default.48fb4c4d31b7d0bf.Core:ImplicitlyDefaultedDefault.Cpp"(ptr %return), !dbg !118
// CHECK:STDOUT:   ret void, !dbg !119
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.51f3fb62cb986374:DefaultOrUnformed.Core.055ae7235dade271"(ptr sret([4 x i8]) %return) #0 !dbg !120 {
// CHECK:STDOUT:   call void @"_COp:thunk:Default.f7ca6602362234ee.Core:ExplicitlyDefaultedDefault.Cpp"(ptr %return), !dbg !123
// CHECK:STDOUT:   ret void, !dbg !124
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.3d7f655e9f020542(ptr %t) #0 !dbg !125 {
// CHECK:STDOUT:   call void @"_COp:thunk:DestroyImpl.40d7fdd10b56fa91.Core:DestroyImpl.Core"(ptr %t), !dbg !128
// CHECK:STDOUT:   ret void, !dbg !130
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.2968cd7107a26841(ptr %t) #0 !dbg !132 {
// CHECK:STDOUT:   call void @"_COp:thunk:DestroyImpl.a73ffbfc339da3af.Core:DestroyImpl.Core"(ptr %t), !dbg !135
// CHECK:STDOUT:   ret void, !dbg !137
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN26ImplicitlyDefaultedDefaultC2Ev(ptr noundef nonnull align 4 dereferenceable(4) %this) unnamed_addr #4 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !140
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   %x_ = getelementptr inbounds nuw %struct.ImplicitlyDefaultedDefault, ptr %this1, i32 0, i32 0
// CHECK:STDOUT:   store i32 1, ptr %x_, align 4, !tbaa !142
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN26ExplicitlyDefaultedDefaultC2Ev(ptr noundef nonnull align 4 dereferenceable(4) %this) unnamed_addr #5 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !144
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   %x_ = getelementptr inbounds nuw %struct.ExplicitlyDefaultedDefault, ptr %this1, i32 0, i32 0
// CHECK:STDOUT:   store i32 1, ptr %x_, align 4, !tbaa !146
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.3d7f655e9f020542, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.2968cd7107a26841, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #4 = { inlinehint mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #5 = { mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!89}
// CHECK:STDOUT: !llvm.module.flags = !{!56, !57, !58, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!60}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !8 = !DIFile(filename: "min_prelude/parts/default.carbon", directory: "")
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !21 = !{null}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{null, !23}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !47 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !48 = !{!"omnipotent char", !47, i64 0}
// CHECK:STDOUT: !49 = !{!"any pointer", !48, i64 0}
// CHECK:STDOUT: !52 = !{!"int", !48, i64 0}
// CHECK:STDOUT: !56 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !57 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !58 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !59 = !{!"__libc_errno", !52, i64 0}
// CHECK:STDOUT: !60 = !{!59, !52, i64 0}
// CHECK:STDOUT: !63 = !{!23}
// CHECK:STDOUT: !64 = !DISubroutineType(types: !63)
// CHECK:STDOUT: !88 = !DIFile(filename: "defaulted_default.carbon", directory: "")
// CHECK:STDOUT: !89 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !88, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !90 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !88, line: 6, type: !22, spFlags: DISPFlagDefinition, unit: !89)
// CHECK:STDOUT: !91 = !DILocation(line: 8, column: 7, scope: !90)
// CHECK:STDOUT: !94 = !DILocation(line: 9, column: 3, scope: !90)
// CHECK:STDOUT: !96 = !DILocation(line: 12, column: 7, scope: !90)
// CHECK:STDOUT: !99 = !DILocation(line: 13, column: 3, scope: !90)
// CHECK:STDOUT: !101 = !DILocation(line: 6, column: 1, scope: !90)
// CHECK:STDOUT: !102 = !DIFile(filename: "./defaulted_default.h", directory: "")
// CHECK:STDOUT: !103 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:Default.48fb4c4d31b7d0bf.Core:ImplicitlyDefaultedDefault.Cpp", scope: null, file: !102, line: 2, type: !64, spFlags: DISPFlagDefinition, unit: !89)
// CHECK:STDOUT: !104 = !DILocation(line: 2, column: 8, scope: !103)
// CHECK:STDOUT: !105 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:Default.f7ca6602362234ee.Core:ExplicitlyDefaultedDefault.Cpp", scope: null, file: !102, line: 7, type: !64, spFlags: DISPFlagDefinition, unit: !89)
// CHECK:STDOUT: !106 = !DILocation(line: 7, column: 3, scope: !105)
// CHECK:STDOUT: !107 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:DestroyImpl.40d7fdd10b56fa91.Core:DestroyImpl.Core", scope: null, file: !88, type: !25, spFlags: DISPFlagDefinition, unit: !89, retainedNodes: !110)
// CHECK:STDOUT: !108 = !DILocalVariable(arg: 1, scope: !107, type: !23)
// CHECK:STDOUT: !109 = !DILocation(line: 0, scope: !107)
// CHECK:STDOUT: !110 = !{!108}
// CHECK:STDOUT: !111 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:DestroyImpl.a73ffbfc339da3af.Core:DestroyImpl.Core", scope: null, file: !88, type: !25, spFlags: DISPFlagDefinition, unit: !89, retainedNodes: !114)
// CHECK:STDOUT: !112 = !DILocalVariable(arg: 1, scope: !111, type: !23)
// CHECK:STDOUT: !113 = !DILocation(line: 0, scope: !111)
// CHECK:STDOUT: !114 = !{!112}
// CHECK:STDOUT: !115 = distinct !DISubprogram(name: "Op", linkageName: "_COp.51f3fb62cb986374:DefaultOrUnformed.Core.c20bae04b665a8c9", scope: null, file: !8, line: 9, type: !64, spFlags: DISPFlagDefinition, unit: !89)
// CHECK:STDOUT: !118 = !DILocation(line: 9, column: 28, scope: !115)
// CHECK:STDOUT: !119 = !DILocation(line: 9, column: 21, scope: !115)
// CHECK:STDOUT: !120 = distinct !DISubprogram(name: "Op", linkageName: "_COp.51f3fb62cb986374:DefaultOrUnformed.Core.055ae7235dade271", scope: null, file: !8, line: 9, type: !64, spFlags: DISPFlagDefinition, unit: !89)
// CHECK:STDOUT: !123 = !DILocation(line: 9, column: 28, scope: !120)
// CHECK:STDOUT: !124 = !DILocation(line: 9, column: 21, scope: !120)
// CHECK:STDOUT: !125 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.3d7f655e9f020542", scope: null, file: !10, line: 26, type: !25, spFlags: DISPFlagDefinition, unit: !89, retainedNodes: !131)
// CHECK:STDOUT: !126 = !DILocalVariable(arg: 1, scope: !125, type: !23)
// CHECK:STDOUT: !128 = !DILocation(line: 27, column: 5, scope: !125)
// CHECK:STDOUT: !130 = !DILocation(line: 26, column: 1, scope: !125)
// CHECK:STDOUT: !131 = !{!126}
// CHECK:STDOUT: !132 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.2968cd7107a26841", scope: null, file: !10, line: 26, type: !25, spFlags: DISPFlagDefinition, unit: !89, retainedNodes: !138)
// CHECK:STDOUT: !133 = !DILocalVariable(arg: 1, scope: !132, type: !23)
// CHECK:STDOUT: !135 = !DILocation(line: 27, column: 5, scope: !132)
// CHECK:STDOUT: !137 = !DILocation(line: 26, column: 1, scope: !132)
// CHECK:STDOUT: !138 = !{!133}
// CHECK:STDOUT: !139 = !{!"p1 _ZTS26ImplicitlyDefaultedDefault", !49, i64 0}
// CHECK:STDOUT: !140 = !{!139, !139, i64 0}
// CHECK:STDOUT: !141 = !{!"_ZTS26ImplicitlyDefaultedDefault", !52, i64 0}
// CHECK:STDOUT: !142 = !{!141, !52, i64 0}
// CHECK:STDOUT: !143 = !{!"p1 _ZTS26ExplicitlyDefaultedDefault", !49, i64 0}
// CHECK:STDOUT: !144 = !{!143, !143, i64 0}
// CHECK:STDOUT: !145 = !{!"_ZTS26ExplicitlyDefaultedDefault", !52, i64 0}
// CHECK:STDOUT: !146 = !{!145, !52, i64 0}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'call_copy.carbon'
// CHECK:STDOUT: source_filename = "call_copy.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCopy.Main(ptr sret([4 x i8]) %return, ptr %c) #0 !dbg !153 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_10.1.temp = alloca [4 x i8], align 4, !dbg !158
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_10.1.temp), !dbg !158
// CHECK:STDOUT:   call void @_ZN4CopyC1ERKS_.carbon_thunk._(ptr %c, ptr %return), !dbg !158
// CHECK:STDOUT:   ret void, !dbg !159
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN4CopyC1ERKS_.carbon_thunk._(ptr noundef %0, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !167
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !167
// CHECK:STDOUT:   %1 = load ptr, ptr %return.addr, align 8, !tbaa !167
// CHECK:STDOUT:   %2 = load ptr, ptr %.addr, align 8, !tbaa !167
// CHECK:STDOUT:   call void @_ZN4CopyC1ERKS_(ptr noundef nonnull align 4 dereferenceable(4) %1, ptr noundef nonnull align 4 dereferenceable(4) %2)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:Copy.58fe21e3ef78b0d7.Core:Copy.Cpp"(ptr sret([4 x i8]) %return, ptr %self) #2 !dbg !162 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN4CopyC1ERKS_.carbon_thunk._(ptr %self, ptr %return), !dbg !164
// CHECK:STDOUT:   ret void, !dbg !164
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_ZN4CopyC1ERKS_(ptr noundef nonnull align 4 dereferenceable(4), ptr noundef nonnull align 4 dereferenceable(4)) unnamed_addr #4
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #4 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!148}
// CHECK:STDOUT: !llvm.module.flags = !{!56, !57, !58, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!60}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !47 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !48 = !{!"omnipotent char", !47, i64 0}
// CHECK:STDOUT: !49 = !{!"any pointer", !48, i64 0}
// CHECK:STDOUT: !52 = !{!"int", !48, i64 0}
// CHECK:STDOUT: !56 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !57 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !58 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !59 = !{!"__libc_errno", !52, i64 0}
// CHECK:STDOUT: !60 = !{!59, !52, i64 0}
// CHECK:STDOUT: !147 = !DIFile(filename: "call_copy.carbon", directory: "")
// CHECK:STDOUT: !148 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !147, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !149 = !{!23, !23}
// CHECK:STDOUT: !150 = !DISubroutineType(types: !149)
// CHECK:STDOUT: !153 = distinct !DISubprogram(name: "Copy", linkageName: "_CCopy.Main", scope: null, file: !147, line: 6, type: !150, spFlags: DISPFlagDefinition, unit: !148, retainedNodes: !160)
// CHECK:STDOUT: !154 = !DILocalVariable(arg: 1, scope: !153, type: !23)
// CHECK:STDOUT: !158 = !DILocation(line: 7, column: 10, scope: !153)
// CHECK:STDOUT: !159 = !DILocation(line: 7, column: 3, scope: !153)
// CHECK:STDOUT: !160 = !{!154}
// CHECK:STDOUT: !161 = !DIFile(filename: "./copy.h", directory: "")
// CHECK:STDOUT: !162 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:Copy.58fe21e3ef78b0d7.Core:Copy.Cpp", scope: null, file: !161, line: 5, type: !150, spFlags: DISPFlagDefinition, unit: !148, retainedNodes: !165)
// CHECK:STDOUT: !163 = !DILocalVariable(arg: 1, scope: !162, type: !23)
// CHECK:STDOUT: !164 = !DILocation(line: 5, column: 3, scope: !162)
// CHECK:STDOUT: !165 = !{!163}
// CHECK:STDOUT: !166 = !{!"p1 _ZTS4Copy", !49, i64 0}
// CHECK:STDOUT: !167 = !{!166, !166, i64 0}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'call_c1_in_thunk.carbon'
// CHECK:STDOUT: source_filename = "call_c1_in_thunk.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN7DerivedC1Ev = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZTV7Derived = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZTT7Derived = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZTI7Derived = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZTS7Derived = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZTI4Base = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZTS4Base = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: @_ZTV7Derived = linkonce_odr dso_local constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr null, ptr @_ZTI7Derived] }, comdat, align 8
// CHECK:STDOUT: @_ZTT7Derived = linkonce_odr dso_local unnamed_addr constant [1 x ptr] [ptr getelementptr inbounds inrange(-24, 0) (i8, ptr @_ZTV7Derived, i64 24)], comdat, align 8
// CHECK:STDOUT: @_ZTI7Derived = linkonce_odr dso_local constant { ptr, ptr, i32, i32, ptr, i64 } { ptr getelementptr inbounds (i8, ptr @_ZTVN10__cxxabiv121__vmi_class_type_infoE, i64 16), ptr @_ZTS7Derived, i32 0, i32 1, ptr @_ZTI4Base, i64 -6141 }, comdat, align 8
// CHECK:STDOUT: @_ZTVN10__cxxabiv121__vmi_class_type_infoE = external global [0 x ptr]
// CHECK:STDOUT: @_ZTS7Derived = linkonce_odr dso_local constant [9 x i8] c"7Derived\00", comdat, align 1
// CHECK:STDOUT: @_ZTI4Base = linkonce_odr dso_local constant { ptr, ptr } { ptr getelementptr inbounds (i8, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 16), ptr @_ZTS4Base }, comdat, align 8
// CHECK:STDOUT: @_ZTVN10__cxxabiv117__class_type_infoE = external global [0 x ptr]
// CHECK:STDOUT: @_ZTS4Base = linkonce_odr dso_local constant [6 x i8] c"4Base\00", comdat, align 1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !171 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_44.1.temp = alloca [8 x i8], align 8, !dbg !172
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_44.1.temp), !dbg !172
// CHECK:STDOUT:   call void @_ZN7DerivedC1Ev.carbon_thunk.(ptr %.loc7_44.1.temp), !dbg !172
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.7e3ba9cb5f45b57c(ptr %.loc7_44.1.temp), !dbg !172
// CHECK:STDOUT:   ret void, !dbg !175
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN7DerivedC1Ev.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !188
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !188
// CHECK:STDOUT:   call void @_ZN7DerivedC1Ev(ptr noundef nonnull align 8 dereferenceable(8) %0) #0
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:DestroyImpl.c4a81c42849ddfa3.Core:DestroyImpl.Core"(ptr %self) #2 !dbg !176 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !178
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.7e3ba9cb5f45b57c(ptr %t) #0 !dbg !180 {
// CHECK:STDOUT:   call void @"_COp:thunk:DestroyImpl.c4a81c42849ddfa3.Core:DestroyImpl.Core"(ptr %t), !dbg !183
// CHECK:STDOUT:   ret void, !dbg !185
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN7DerivedC1Ev(ptr noundef nonnull align 8 dereferenceable(8) %this) unnamed_addr #4 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !188
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   store ptr getelementptr inbounds inrange(-24, 0) (i8, ptr @_ZTV7Derived, i64 24), ptr %this1, align 8, !tbaa !190
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #4 = { inlinehint mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!170}
// CHECK:STDOUT: !llvm.module.flags = !{!56, !57, !58, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!60}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !21 = !{null}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{null, !23}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !47 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !48 = !{!"omnipotent char", !47, i64 0}
// CHECK:STDOUT: !49 = !{!"any pointer", !48, i64 0}
// CHECK:STDOUT: !52 = !{!"int", !48, i64 0}
// CHECK:STDOUT: !56 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !57 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !58 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !59 = !{!"__libc_errno", !52, i64 0}
// CHECK:STDOUT: !60 = !{!59, !52, i64 0}
// CHECK:STDOUT: !169 = !DIFile(filename: "call_c1_in_thunk.carbon", directory: "")
// CHECK:STDOUT: !170 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !169, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !171 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !169, line: 6, type: !22, spFlags: DISPFlagDefinition, unit: !170)
// CHECK:STDOUT: !172 = !DILocation(line: 7, column: 24, scope: !171)
// CHECK:STDOUT: !175 = !DILocation(line: 6, column: 1, scope: !171)
// CHECK:STDOUT: !176 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:DestroyImpl.c4a81c42849ddfa3.Core:DestroyImpl.Core", scope: null, file: !169, type: !25, spFlags: DISPFlagDefinition, unit: !170, retainedNodes: !179)
// CHECK:STDOUT: !177 = !DILocalVariable(arg: 1, scope: !176, type: !23)
// CHECK:STDOUT: !178 = !DILocation(line: 0, scope: !176)
// CHECK:STDOUT: !179 = !{!177}
// CHECK:STDOUT: !180 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.7e3ba9cb5f45b57c", scope: null, file: !10, line: 26, type: !25, spFlags: DISPFlagDefinition, unit: !170, retainedNodes: !186)
// CHECK:STDOUT: !181 = !DILocalVariable(arg: 1, scope: !180, type: !23)
// CHECK:STDOUT: !183 = !DILocation(line: 27, column: 5, scope: !180)
// CHECK:STDOUT: !185 = !DILocation(line: 26, column: 1, scope: !180)
// CHECK:STDOUT: !186 = !{!181}
// CHECK:STDOUT: !187 = !{!"p1 _ZTS7Derived", !49, i64 0}
// CHECK:STDOUT: !188 = !{!187, !187, i64 0}
// CHECK:STDOUT: !189 = !{!"vtable pointer", !47, i64 0}
// CHECK:STDOUT: !190 = !{!189, !189, i64 0}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'call_multi_argument.carbon'
// CHECK:STDOUT: source_filename = "call_multi_argument.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN9MultiArgCC2Ev = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN9MultiArgCC2Ei = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN9MultiArgCC2Eii = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN9MultiArgCC2Eiiii = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CZero.Main() #0 !dbg !201 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_50.1.temp = alloca [1 x i8], align 1, !dbg !202
// CHECK:STDOUT:   %.loc8_27.2.temp = alloca [1 x i8], align 1, !dbg !205
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_50.1.temp), !dbg !202
// CHECK:STDOUT:   call void @_ZN9MultiArgCC1Ev.carbon_thunk.(ptr %.loc7_50.1.temp), !dbg !202
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc8_27.2.temp), !dbg !205
// CHECK:STDOUT:   call void @_ZN9MultiArgCC1Ev.carbon_thunk_tuple.(ptr %.loc8_27.2.temp), !dbg !205
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.4509b4e1a7960bb4(ptr %.loc8_27.2.temp), !dbg !205
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.4509b4e1a7960bb4(ptr %.loc7_50.1.temp), !dbg !202
// CHECK:STDOUT:   ret void, !dbg !208
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN9MultiArgCC1Ev.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !256
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !256
// CHECK:STDOUT:   call void @_ZN9MultiArgCC2Ev(ptr noundef nonnull align 1 dereferenceable(1) %0)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN9MultiArgCC1Ev.carbon_thunk_tuple.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !256
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !256
// CHECK:STDOUT:   call void @_ZN9MultiArgCC2Ev(ptr noundef nonnull align 1 dereferenceable(1) %0)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:DestroyImpl.80399c54d9740c53.Core:DestroyImpl.Core"(ptr %self) #2 !dbg !209 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !211
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_COne.Main() #0 !dbg !213 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc12_51.1.temp = alloca [1 x i8], align 1, !dbg !214
// CHECK:STDOUT:   %.loc13_26.1.temp = alloca [1 x i8], align 1, !dbg !217
// CHECK:STDOUT:   %.loc14_29.2.temp = alloca [1 x i8], align 1, !dbg !220
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc12_51.1.temp), !dbg !214
// CHECK:STDOUT:   call void @_ZN9MultiArgCC1Ei.carbon_thunk._(i32 1, ptr %.loc12_51.1.temp), !dbg !214
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc13_26.1.temp), !dbg !217
// CHECK:STDOUT:   call void @_ZN9MultiArgCC1Ei.carbon_thunk._(i32 2, ptr %.loc13_26.1.temp), !dbg !217
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc14_29.2.temp), !dbg !220
// CHECK:STDOUT:   call void @_ZN9MultiArgCC1Ei.carbon_thunk_tuple._(i32 3, ptr %.loc14_29.2.temp), !dbg !220
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.4509b4e1a7960bb4(ptr %.loc14_29.2.temp), !dbg !220
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.4509b4e1a7960bb4(ptr %.loc13_26.1.temp), !dbg !217
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.4509b4e1a7960bb4(ptr %.loc12_51.1.temp), !dbg !214
// CHECK:STDOUT:   ret void, !dbg !223
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN9MultiArgCC1Ei.carbon_thunk._(i32 noundef %0, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca i32, align 4
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store i32 %0, ptr %.addr, align 4, !tbaa !257
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !256
// CHECK:STDOUT:   %1 = load ptr, ptr %return.addr, align 8, !tbaa !256
// CHECK:STDOUT:   %2 = load i32, ptr %.addr, align 4, !tbaa !257
// CHECK:STDOUT:   call void @_ZN9MultiArgCC2Ei(ptr noundef nonnull align 1 dereferenceable(1) %1, i32 noundef %2)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN9MultiArgCC1Ei.carbon_thunk_tuple._(i32 noundef %0, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca i32, align 4
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store i32 %0, ptr %.addr, align 4, !tbaa !257
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !256
// CHECK:STDOUT:   %1 = load ptr, ptr %return.addr, align 8, !tbaa !256
// CHECK:STDOUT:   %2 = load i32, ptr %.addr, align 4, !tbaa !257
// CHECK:STDOUT:   call void @_ZN9MultiArgCC2Ei(ptr noundef nonnull align 1 dereferenceable(1) %1, i32 noundef %2)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTwo.Main() #0 !dbg !224 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc18_54.1.temp = alloca [1 x i8], align 1, !dbg !225
// CHECK:STDOUT:   %.loc19_31.2.temp = alloca [1 x i8], align 1, !dbg !228
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc18_54.1.temp), !dbg !225
// CHECK:STDOUT:   call void @_ZN9MultiArgCC1Eii.carbon_thunk.__(i32 1, i32 2, ptr %.loc18_54.1.temp), !dbg !225
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc19_31.2.temp), !dbg !228
// CHECK:STDOUT:   call void @_ZN9MultiArgCC1Eii.carbon_thunk_tuple.__(i32 3, i32 4, ptr %.loc19_31.2.temp), !dbg !228
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.4509b4e1a7960bb4(ptr %.loc19_31.2.temp), !dbg !228
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.4509b4e1a7960bb4(ptr %.loc18_54.1.temp), !dbg !225
// CHECK:STDOUT:   ret void, !dbg !231
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN9MultiArgCC1Eii.carbon_thunk.__(i32 noundef %0, i32 noundef %1, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca i32, align 4
// CHECK:STDOUT:   %.addr1 = alloca i32, align 4
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store i32 %0, ptr %.addr, align 4, !tbaa !257
// CHECK:STDOUT:   store i32 %1, ptr %.addr1, align 4, !tbaa !257
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !256
// CHECK:STDOUT:   %2 = load ptr, ptr %return.addr, align 8, !tbaa !256
// CHECK:STDOUT:   %3 = load i32, ptr %.addr, align 4, !tbaa !257
// CHECK:STDOUT:   %4 = load i32, ptr %.addr1, align 4, !tbaa !257
// CHECK:STDOUT:   call void @_ZN9MultiArgCC2Eii(ptr noundef nonnull align 1 dereferenceable(1) %2, i32 noundef %3, i32 noundef %4)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN9MultiArgCC1Eii.carbon_thunk_tuple.__(i32 noundef %0, i32 noundef %1, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca i32, align 4
// CHECK:STDOUT:   %.addr1 = alloca i32, align 4
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store i32 %0, ptr %.addr, align 4, !tbaa !257
// CHECK:STDOUT:   store i32 %1, ptr %.addr1, align 4, !tbaa !257
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !256
// CHECK:STDOUT:   %2 = load ptr, ptr %return.addr, align 8, !tbaa !256
// CHECK:STDOUT:   %3 = load i32, ptr %.addr, align 4, !tbaa !257
// CHECK:STDOUT:   %4 = load i32, ptr %.addr1, align 4, !tbaa !257
// CHECK:STDOUT:   call void @_ZN9MultiArgCC2Eii(ptr noundef nonnull align 1 dereferenceable(1) %2, i32 noundef %3, i32 noundef %4)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CThree.Main() #0 !dbg !232 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc23_57.1.temp = alloca [1 x i8], align 1, !dbg !233
// CHECK:STDOUT:   %.loc24_34.2.temp = alloca [1 x i8], align 1, !dbg !236
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc23_57.1.temp), !dbg !233
// CHECK:STDOUT:   call void @_ZN9MultiArgCC1Eiiii.carbon_thunk.___(i32 1, i32 2, i32 3, ptr %.loc23_57.1.temp), !dbg !233
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc24_34.2.temp), !dbg !236
// CHECK:STDOUT:   call void @_ZN9MultiArgCC1Eiiii.carbon_thunk_tuple.___(i32 4, i32 5, i32 6, ptr %.loc24_34.2.temp), !dbg !236
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.4509b4e1a7960bb4(ptr %.loc24_34.2.temp), !dbg !236
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.4509b4e1a7960bb4(ptr %.loc23_57.1.temp), !dbg !233
// CHECK:STDOUT:   ret void, !dbg !239
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN9MultiArgCC1Eiiii.carbon_thunk.___(i32 noundef %0, i32 noundef %1, i32 noundef %2, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca i32, align 4
// CHECK:STDOUT:   %.addr1 = alloca i32, align 4
// CHECK:STDOUT:   %.addr2 = alloca i32, align 4
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store i32 %0, ptr %.addr, align 4, !tbaa !257
// CHECK:STDOUT:   store i32 %1, ptr %.addr1, align 4, !tbaa !257
// CHECK:STDOUT:   store i32 %2, ptr %.addr2, align 4, !tbaa !257
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !256
// CHECK:STDOUT:   %3 = load ptr, ptr %return.addr, align 8, !tbaa !256
// CHECK:STDOUT:   %4 = load i32, ptr %.addr, align 4, !tbaa !257
// CHECK:STDOUT:   %5 = load i32, ptr %.addr1, align 4, !tbaa !257
// CHECK:STDOUT:   %6 = load i32, ptr %.addr2, align 4, !tbaa !257
// CHECK:STDOUT:   call void @_ZN9MultiArgCC2Eiiii(ptr noundef nonnull align 1 dereferenceable(1) %3, i32 noundef %4, i32 noundef %5, i32 noundef %6, i32 noundef 0)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN9MultiArgCC1Eiiii.carbon_thunk_tuple.___(i32 noundef %0, i32 noundef %1, i32 noundef %2, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca i32, align 4
// CHECK:STDOUT:   %.addr1 = alloca i32, align 4
// CHECK:STDOUT:   %.addr2 = alloca i32, align 4
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store i32 %0, ptr %.addr, align 4, !tbaa !257
// CHECK:STDOUT:   store i32 %1, ptr %.addr1, align 4, !tbaa !257
// CHECK:STDOUT:   store i32 %2, ptr %.addr2, align 4, !tbaa !257
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !256
// CHECK:STDOUT:   %3 = load ptr, ptr %return.addr, align 8, !tbaa !256
// CHECK:STDOUT:   %4 = load i32, ptr %.addr, align 4, !tbaa !257
// CHECK:STDOUT:   %5 = load i32, ptr %.addr1, align 4, !tbaa !257
// CHECK:STDOUT:   %6 = load i32, ptr %.addr2, align 4, !tbaa !257
// CHECK:STDOUT:   call void @_ZN9MultiArgCC2Eiiii(ptr noundef nonnull align 1 dereferenceable(1) %3, i32 noundef %4, i32 noundef %5, i32 noundef %6, i32 noundef 0)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CFour.Main() #0 !dbg !240 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc28_60.1.temp = alloca [1 x i8], align 1, !dbg !241
// CHECK:STDOUT:   %.loc29_37.2.temp = alloca [1 x i8], align 1, !dbg !244
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc28_60.1.temp), !dbg !241
// CHECK:STDOUT:   call void @_ZN9MultiArgCC1Eiiii.carbon_thunk.____(i32 1, i32 2, i32 3, i32 4, ptr %.loc28_60.1.temp), !dbg !241
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc29_37.2.temp), !dbg !244
// CHECK:STDOUT:   call void @_ZN9MultiArgCC1Eiiii.carbon_thunk_tuple.____(i32 5, i32 6, i32 7, i32 8, ptr %.loc29_37.2.temp), !dbg !244
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.4509b4e1a7960bb4(ptr %.loc29_37.2.temp), !dbg !244
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.4509b4e1a7960bb4(ptr %.loc28_60.1.temp), !dbg !241
// CHECK:STDOUT:   ret void, !dbg !247
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN9MultiArgCC1Eiiii.carbon_thunk.____(i32 noundef %0, i32 noundef %1, i32 noundef %2, i32 noundef %3, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca i32, align 4
// CHECK:STDOUT:   %.addr1 = alloca i32, align 4
// CHECK:STDOUT:   %.addr2 = alloca i32, align 4
// CHECK:STDOUT:   %.addr3 = alloca i32, align 4
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store i32 %0, ptr %.addr, align 4, !tbaa !257
// CHECK:STDOUT:   store i32 %1, ptr %.addr1, align 4, !tbaa !257
// CHECK:STDOUT:   store i32 %2, ptr %.addr2, align 4, !tbaa !257
// CHECK:STDOUT:   store i32 %3, ptr %.addr3, align 4, !tbaa !257
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !256
// CHECK:STDOUT:   %4 = load ptr, ptr %return.addr, align 8, !tbaa !256
// CHECK:STDOUT:   %5 = load i32, ptr %.addr, align 4, !tbaa !257
// CHECK:STDOUT:   %6 = load i32, ptr %.addr1, align 4, !tbaa !257
// CHECK:STDOUT:   %7 = load i32, ptr %.addr2, align 4, !tbaa !257
// CHECK:STDOUT:   %8 = load i32, ptr %.addr3, align 4, !tbaa !257
// CHECK:STDOUT:   call void @_ZN9MultiArgCC2Eiiii(ptr noundef nonnull align 1 dereferenceable(1) %4, i32 noundef %5, i32 noundef %6, i32 noundef %7, i32 noundef %8)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN9MultiArgCC1Eiiii.carbon_thunk_tuple.____(i32 noundef %0, i32 noundef %1, i32 noundef %2, i32 noundef %3, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca i32, align 4
// CHECK:STDOUT:   %.addr1 = alloca i32, align 4
// CHECK:STDOUT:   %.addr2 = alloca i32, align 4
// CHECK:STDOUT:   %.addr3 = alloca i32, align 4
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store i32 %0, ptr %.addr, align 4, !tbaa !257
// CHECK:STDOUT:   store i32 %1, ptr %.addr1, align 4, !tbaa !257
// CHECK:STDOUT:   store i32 %2, ptr %.addr2, align 4, !tbaa !257
// CHECK:STDOUT:   store i32 %3, ptr %.addr3, align 4, !tbaa !257
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !256
// CHECK:STDOUT:   %4 = load ptr, ptr %return.addr, align 8, !tbaa !256
// CHECK:STDOUT:   %5 = load i32, ptr %.addr, align 4, !tbaa !257
// CHECK:STDOUT:   %6 = load i32, ptr %.addr1, align 4, !tbaa !257
// CHECK:STDOUT:   %7 = load i32, ptr %.addr2, align 4, !tbaa !257
// CHECK:STDOUT:   %8 = load i32, ptr %.addr3, align 4, !tbaa !257
// CHECK:STDOUT:   call void @_ZN9MultiArgCC2Eiiii(ptr noundef nonnull align 1 dereferenceable(1) %4, i32 noundef %5, i32 noundef %6, i32 noundef %7, i32 noundef %8)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.4509b4e1a7960bb4(ptr %t) #0 !dbg !248 {
// CHECK:STDOUT:   call void @"_COp:thunk:DestroyImpl.80399c54d9740c53.Core:DestroyImpl.Core"(ptr %t), !dbg !251
// CHECK:STDOUT:   ret void, !dbg !253
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN9MultiArgCC2Ev(ptr noundef nonnull align 1 dereferenceable(1) %this) unnamed_addr #4 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !256
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN9MultiArgCC2Ei(ptr noundef nonnull align 1 dereferenceable(1) %this, i32 noundef %0) unnamed_addr #4 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   %.addr = alloca i32, align 4
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !256
// CHECK:STDOUT:   store i32 %0, ptr %.addr, align 4, !tbaa !257
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN9MultiArgCC2Eii(ptr noundef nonnull align 1 dereferenceable(1) %this, i32 noundef %0, i32 noundef %1) unnamed_addr #4 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   %.addr = alloca i32, align 4
// CHECK:STDOUT:   %.addr1 = alloca i32, align 4
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !256
// CHECK:STDOUT:   store i32 %0, ptr %.addr, align 4, !tbaa !257
// CHECK:STDOUT:   store i32 %1, ptr %.addr1, align 4, !tbaa !257
// CHECK:STDOUT:   %this2 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN9MultiArgCC2Eiiii(ptr noundef nonnull align 1 dereferenceable(1) %this, i32 noundef %0, i32 noundef %1, i32 noundef %2, i32 noundef %3) unnamed_addr #4 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   %.addr = alloca i32, align 4
// CHECK:STDOUT:   %.addr1 = alloca i32, align 4
// CHECK:STDOUT:   %.addr2 = alloca i32, align 4
// CHECK:STDOUT:   %.addr3 = alloca i32, align 4
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !256
// CHECK:STDOUT:   store i32 %0, ptr %.addr, align 4, !tbaa !257
// CHECK:STDOUT:   store i32 %1, ptr %.addr1, align 4, !tbaa !257
// CHECK:STDOUT:   store i32 %2, ptr %.addr2, align 4, !tbaa !257
// CHECK:STDOUT:   store i32 %3, ptr %.addr3, align 4, !tbaa !257
// CHECK:STDOUT:   %this4 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_ZN9MultiArgCC1Ei.carbon_thunk._, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.4509b4e1a7960bb4, { 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #4 = { mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!192}
// CHECK:STDOUT: !llvm.module.flags = !{!56, !57, !58, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!60}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !21 = !{null}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{null, !23}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !47 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !48 = !{!"omnipotent char", !47, i64 0}
// CHECK:STDOUT: !49 = !{!"any pointer", !48, i64 0}
// CHECK:STDOUT: !52 = !{!"int", !48, i64 0}
// CHECK:STDOUT: !56 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !57 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !58 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !59 = !{!"__libc_errno", !52, i64 0}
// CHECK:STDOUT: !60 = !{!59, !52, i64 0}
// CHECK:STDOUT: !191 = !DIFile(filename: "call_multi_argument.carbon", directory: "")
// CHECK:STDOUT: !192 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !191, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !201 = distinct !DISubprogram(name: "Zero", linkageName: "_CZero.Main", scope: null, file: !191, line: 6, type: !22, spFlags: DISPFlagDefinition, unit: !192)
// CHECK:STDOUT: !202 = !DILocation(line: 7, column: 26, scope: !201)
// CHECK:STDOUT: !205 = !DILocation(line: 8, column: 26, scope: !201)
// CHECK:STDOUT: !208 = !DILocation(line: 6, column: 1, scope: !201)
// CHECK:STDOUT: !209 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:DestroyImpl.80399c54d9740c53.Core:DestroyImpl.Core", scope: null, file: !191, type: !25, spFlags: DISPFlagDefinition, unit: !192, retainedNodes: !212)
// CHECK:STDOUT: !210 = !DILocalVariable(arg: 1, scope: !209, type: !23)
// CHECK:STDOUT: !211 = !DILocation(line: 0, scope: !209)
// CHECK:STDOUT: !212 = !{!210}
// CHECK:STDOUT: !213 = distinct !DISubprogram(name: "One", linkageName: "_COne.Main", scope: null, file: !191, line: 11, type: !22, spFlags: DISPFlagDefinition, unit: !192)
// CHECK:STDOUT: !214 = !DILocation(line: 12, column: 26, scope: !213)
// CHECK:STDOUT: !217 = !DILocation(line: 13, column: 26, scope: !213)
// CHECK:STDOUT: !220 = !DILocation(line: 14, column: 26, scope: !213)
// CHECK:STDOUT: !223 = !DILocation(line: 11, column: 1, scope: !213)
// CHECK:STDOUT: !224 = distinct !DISubprogram(name: "Two", linkageName: "_CTwo.Main", scope: null, file: !191, line: 17, type: !22, spFlags: DISPFlagDefinition, unit: !192)
// CHECK:STDOUT: !225 = !DILocation(line: 18, column: 26, scope: !224)
// CHECK:STDOUT: !228 = !DILocation(line: 19, column: 26, scope: !224)
// CHECK:STDOUT: !231 = !DILocation(line: 17, column: 1, scope: !224)
// CHECK:STDOUT: !232 = distinct !DISubprogram(name: "Three", linkageName: "_CThree.Main", scope: null, file: !191, line: 22, type: !22, spFlags: DISPFlagDefinition, unit: !192)
// CHECK:STDOUT: !233 = !DILocation(line: 23, column: 26, scope: !232)
// CHECK:STDOUT: !236 = !DILocation(line: 24, column: 26, scope: !232)
// CHECK:STDOUT: !239 = !DILocation(line: 22, column: 1, scope: !232)
// CHECK:STDOUT: !240 = distinct !DISubprogram(name: "Four", linkageName: "_CFour.Main", scope: null, file: !191, line: 27, type: !22, spFlags: DISPFlagDefinition, unit: !192)
// CHECK:STDOUT: !241 = !DILocation(line: 28, column: 26, scope: !240)
// CHECK:STDOUT: !244 = !DILocation(line: 29, column: 26, scope: !240)
// CHECK:STDOUT: !247 = !DILocation(line: 27, column: 1, scope: !240)
// CHECK:STDOUT: !248 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.4509b4e1a7960bb4", scope: null, file: !10, line: 26, type: !25, spFlags: DISPFlagDefinition, unit: !192, retainedNodes: !254)
// CHECK:STDOUT: !249 = !DILocalVariable(arg: 1, scope: !248, type: !23)
// CHECK:STDOUT: !251 = !DILocation(line: 27, column: 5, scope: !248)
// CHECK:STDOUT: !253 = !DILocation(line: 26, column: 1, scope: !248)
// CHECK:STDOUT: !254 = !{!249}
// CHECK:STDOUT: !255 = !{!"p1 _ZTS9MultiArgC", !49, i64 0}
// CHECK:STDOUT: !256 = !{!255, !255, i64 0}
// CHECK:STDOUT: !257 = !{!52, !52, i64 0}
// CHECK:STDOUT:
