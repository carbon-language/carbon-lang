// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/constructor.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/constructor.carbon

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

class C {
 public:
  C() {}
  C(int) {}
  C(int, int) {}
  C(int, int, int, int = 0) {}
};

// --- call_multi_argument.carbon

library "[[@TEST_NAME]]";

import Cpp library "multi_argument.h";

fn Zero() {
  let _: Cpp.C = Cpp.C.C();
  let _: Cpp.C = ();
}

fn One() {
  let _: Cpp.C = Cpp.C.C(1);
  let _: Cpp.C = 2;
  let _: Cpp.C = (3,);
}

fn Two() {
  let _: Cpp.C = Cpp.C.C(1, 2);
  let _: Cpp.C = (3, 4);
}

fn Three() {
  let _: Cpp.C = Cpp.C.C(1, 2, 3);
  let _: Cpp.C = (4, 5, 6);
}

fn Four() {
  let _: Cpp.C = Cpp.C.C(1, 2, 3, 4);
  let _: Cpp.C = (5, 6, 7, 8);
}

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
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_26.1.temp = alloca [8 x i8], align 1, !dbg !14
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_26.1.temp), !dbg !14
// CHECK:STDOUT:   call void @_ZN1CC1Ev.carbon_thunk.(ptr %.loc7_26.1.temp), !dbg !14
// CHECK:STDOUT:   ret void, !dbg !15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN1CC1Ev.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !16
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !16
// CHECK:STDOUT:   call void @_ZN1CC2Ev(ptr noundef nonnull align 4 dereferenceable(8) %0)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !19 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc11_19.2.temp = alloca [8 x i8], align 1, !dbg !20
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc11_19.2.temp), !dbg !20
// CHECK:STDOUT:   call void @_ZN1CC1Ev.carbon_thunk_tuple.(ptr %.loc11_19.2.temp), !dbg !20
// CHECK:STDOUT:   ret void, !dbg !21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN1CC1Ev.carbon_thunk_tuple.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !16
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !16
// CHECK:STDOUT:   call void @_ZN1CC2Ev(ptr noundef nonnull align 4 dereferenceable(8) %0)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN1CC2Ev(ptr noundef nonnull align 4 dereferenceable(8) %this) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !16
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   %x_ = getelementptr inbounds nuw %class.C, ptr %this1, i32 0, i32 0
// CHECK:STDOUT:   store i32 8, ptr %x_, align 4, !tbaa !22
// CHECK:STDOUT:   %y_ = getelementptr inbounds nuw %class.C, ptr %this1, i32 0, i32 1
// CHECK:STDOUT:   store i32 9, ptr %y_, align 4, !tbaa !24
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1, !2, !3, !4}
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!7}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !3 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !4 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !6, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIFile(filename: "import_default.carbon", directory: "")
// CHECK:STDOUT: !7 = !{!8, !8, i64 0}
// CHECK:STDOUT: !8 = !{!"int", !9, i64 0}
// CHECK:STDOUT: !9 = !{!"omnipotent char", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !6, line: 6, type: !12, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !12 = !DISubroutineType(types: !13)
// CHECK:STDOUT: !13 = !{null}
// CHECK:STDOUT: !14 = !DILocation(line: 7, column: 18, scope: !11)
// CHECK:STDOUT: !15 = !DILocation(line: 6, column: 1, scope: !11)
// CHECK:STDOUT: !16 = !{!17, !17, i64 0}
// CHECK:STDOUT: !17 = !{!"p1 _ZTS1C", !18, i64 0}
// CHECK:STDOUT: !18 = !{!"any pointer", !9, i64 0}
// CHECK:STDOUT: !19 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !6, line: 10, type: !12, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !20 = !DILocation(line: 11, column: 18, scope: !19)
// CHECK:STDOUT: !21 = !DILocation(line: 10, column: 1, scope: !19)
// CHECK:STDOUT: !22 = !{!23, !8, i64 0}
// CHECK:STDOUT: !23 = !{!"_ZTS1C", !8, i64 0, !8, i64 4}
// CHECK:STDOUT: !24 = !{!23, !8, i64 4}
// CHECK:STDOUT: ; ModuleID = 'implicit_default.carbon'
// CHECK:STDOUT: source_filename = "implicit_default.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %class.C = type { i32, i32 }
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN1CC2Ev = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca [8 x i8], align 1, !dbg !14
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !14
// CHECK:STDOUT:   call void @"_COp.d288d62be0e5d791:DefaultOrUnformed.Core.93349b0fe912a29b"(ptr %_.var), !dbg !14
// CHECK:STDOUT:   ret void, !dbg !15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN1CC1Ev.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !16
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !16
// CHECK:STDOUT:   call void @_ZN1CC2Ev(ptr noundef nonnull align 4 dereferenceable(8) %0)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define void @"_COp:thunk:Default.7b1e6a57c714cdb7.Core:C.Cpp"(ptr sret([8 x i8]) %return) #2 !dbg !19 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN1CC1Ev.carbon_thunk.(ptr %return), !dbg !24
// CHECK:STDOUT:   ret void, !dbg !24
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.d288d62be0e5d791:DefaultOrUnformed.Core.93349b0fe912a29b"(ptr sret([8 x i8]) %return) #0 !dbg !25 {
// CHECK:STDOUT:   call void @"_COp:thunk:Default.7b1e6a57c714cdb7.Core:C.Cpp"(ptr %return), !dbg !27
// CHECK:STDOUT:   ret void, !dbg !28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN1CC2Ev(ptr noundef nonnull align 4 dereferenceable(8) %this) unnamed_addr #4 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !16
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   %x_ = getelementptr inbounds nuw %class.C, ptr %this1, i32 0, i32 0
// CHECK:STDOUT:   store i32 8, ptr %x_, align 4, !tbaa !29
// CHECK:STDOUT:   %y_ = getelementptr inbounds nuw %class.C, ptr %this1, i32 0, i32 1
// CHECK:STDOUT:   store i32 9, ptr %y_, align 4, !tbaa !31
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #4 = { mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1, !2, !3, !4}
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!7}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !3 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !4 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !6, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIFile(filename: "implicit_default.carbon", directory: "")
// CHECK:STDOUT: !7 = !{!8, !8, i64 0}
// CHECK:STDOUT: !8 = !{!"int", !9, i64 0}
// CHECK:STDOUT: !9 = !{!"omnipotent char", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !6, line: 6, type: !12, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !12 = !DISubroutineType(types: !13)
// CHECK:STDOUT: !13 = !{null}
// CHECK:STDOUT: !14 = !DILocation(line: 7, column: 3, scope: !11)
// CHECK:STDOUT: !15 = !DILocation(line: 6, column: 1, scope: !11)
// CHECK:STDOUT: !16 = !{!17, !17, i64 0}
// CHECK:STDOUT: !17 = !{!"p1 _ZTS1C", !18, i64 0}
// CHECK:STDOUT: !18 = !{!"any pointer", !9, i64 0}
// CHECK:STDOUT: !19 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:Default.7b1e6a57c714cdb7.Core:C.Cpp", scope: null, file: !20, line: 4, type: !21, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !20 = !DIFile(filename: "./default.h", directory: "")
// CHECK:STDOUT: !21 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !22 = !{!23}
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !DILocation(line: 4, column: 3, scope: !19)
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "Op", linkageName: "_COp.d288d62be0e5d791:DefaultOrUnformed.Core.93349b0fe912a29b", scope: null, file: !26, line: 9, type: !21, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !26 = !DIFile(filename: "min_prelude/parts/default.carbon", directory: "")
// CHECK:STDOUT: !27 = !DILocation(line: 9, column: 28, scope: !25)
// CHECK:STDOUT: !28 = !DILocation(line: 9, column: 21, scope: !25)
// CHECK:STDOUT: !29 = !{!30, !8, i64 0}
// CHECK:STDOUT: !30 = !{!"_ZTS1C", !8, i64 0, !8, i64 4}
// CHECK:STDOUT: !31 = !{!30, !8, i64 4}
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
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc8_65.1.temp = alloca [4 x i8], align 1, !dbg !14
// CHECK:STDOUT:   %_.var.loc9 = alloca [4 x i8], align 1, !dbg !15
// CHECK:STDOUT:   %.loc12_65.1.temp = alloca [4 x i8], align 1, !dbg !16
// CHECK:STDOUT:   %_.var.loc13 = alloca [4 x i8], align 1, !dbg !17
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc8_65.1.temp), !dbg !14
// CHECK:STDOUT:   call void @_ZN26ImplicitlyDefaultedDefaultC1Ev.carbon_thunk.(ptr %.loc8_65.1.temp), !dbg !14
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc9), !dbg !15
// CHECK:STDOUT:   call void @"_COp.d288d62be0e5d791:DefaultOrUnformed.Core.22b4681b019423d7"(ptr %_.var.loc9), !dbg !15
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc12_65.1.temp), !dbg !16
// CHECK:STDOUT:   call void @_ZN26ExplicitlyDefaultedDefaultC1Ev.carbon_thunk.(ptr %.loc12_65.1.temp), !dbg !16
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc13), !dbg !17
// CHECK:STDOUT:   call void @"_COp.d288d62be0e5d791:DefaultOrUnformed.Core.1a27c303748a2360"(ptr %_.var.loc13), !dbg !17
// CHECK:STDOUT:   ret void, !dbg !18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN26ImplicitlyDefaultedDefaultC1Ev.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !19
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !19
// CHECK:STDOUT:   call void @_ZN26ImplicitlyDefaultedDefaultC2Ev(ptr noundef nonnull align 4 dereferenceable(4) %0) #0
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define void @"_COp:thunk:Default.2e5fb550c65543cf.Core:ImplicitlyDefaultedDefault.Cpp"(ptr sret([4 x i8]) %return) #2 !dbg !22 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN26ImplicitlyDefaultedDefaultC1Ev.carbon_thunk.(ptr %return), !dbg !27
// CHECK:STDOUT:   ret void, !dbg !27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN26ExplicitlyDefaultedDefaultC1Ev.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !28
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !28
// CHECK:STDOUT:   call void @_ZN26ExplicitlyDefaultedDefaultC2Ev(ptr noundef nonnull align 4 dereferenceable(4) %0) #0
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define void @"_COp:thunk:Default.cad83ce9b7db87bf.Core:ExplicitlyDefaultedDefault.Cpp"(ptr sret([4 x i8]) %return) #2 !dbg !30 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN26ExplicitlyDefaultedDefaultC1Ev.carbon_thunk.(ptr %return), !dbg !31
// CHECK:STDOUT:   ret void, !dbg !31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.d288d62be0e5d791:DefaultOrUnformed.Core.22b4681b019423d7"(ptr sret([4 x i8]) %return) #0 !dbg !32 {
// CHECK:STDOUT:   call void @"_COp:thunk:Default.2e5fb550c65543cf.Core:ImplicitlyDefaultedDefault.Cpp"(ptr %return), !dbg !34
// CHECK:STDOUT:   ret void, !dbg !35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.d288d62be0e5d791:DefaultOrUnformed.Core.1a27c303748a2360"(ptr sret([4 x i8]) %return) #0 !dbg !36 {
// CHECK:STDOUT:   call void @"_COp:thunk:Default.cad83ce9b7db87bf.Core:ExplicitlyDefaultedDefault.Cpp"(ptr %return), !dbg !37
// CHECK:STDOUT:   ret void, !dbg !38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN26ImplicitlyDefaultedDefaultC2Ev(ptr noundef nonnull align 4 dereferenceable(4) %this) unnamed_addr #4 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !19
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   %x_ = getelementptr inbounds nuw %struct.ImplicitlyDefaultedDefault, ptr %this1, i32 0, i32 0
// CHECK:STDOUT:   store i32 1, ptr %x_, align 4, !tbaa !39
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN26ExplicitlyDefaultedDefaultC2Ev(ptr noundef nonnull align 4 dereferenceable(4) %this) unnamed_addr #5 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !28
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   %x_ = getelementptr inbounds nuw %struct.ExplicitlyDefaultedDefault, ptr %this1, i32 0, i32 0
// CHECK:STDOUT:   store i32 1, ptr %x_, align 4, !tbaa !41
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #4 = { inlinehint mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #5 = { mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1, !2, !3, !4}
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!7}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !3 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !4 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !6, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIFile(filename: "defaulted_default.carbon", directory: "")
// CHECK:STDOUT: !7 = !{!8, !8, i64 0}
// CHECK:STDOUT: !8 = !{!"int", !9, i64 0}
// CHECK:STDOUT: !9 = !{!"omnipotent char", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !6, line: 6, type: !12, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !12 = !DISubroutineType(types: !13)
// CHECK:STDOUT: !13 = !{null}
// CHECK:STDOUT: !14 = !DILocation(line: 8, column: 7, scope: !11)
// CHECK:STDOUT: !15 = !DILocation(line: 9, column: 3, scope: !11)
// CHECK:STDOUT: !16 = !DILocation(line: 12, column: 7, scope: !11)
// CHECK:STDOUT: !17 = !DILocation(line: 13, column: 3, scope: !11)
// CHECK:STDOUT: !18 = !DILocation(line: 6, column: 1, scope: !11)
// CHECK:STDOUT: !19 = !{!20, !20, i64 0}
// CHECK:STDOUT: !20 = !{!"p1 _ZTS26ImplicitlyDefaultedDefault", !21, i64 0}
// CHECK:STDOUT: !21 = !{!"any pointer", !9, i64 0}
// CHECK:STDOUT: !22 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:Default.2e5fb550c65543cf.Core:ImplicitlyDefaultedDefault.Cpp", scope: null, file: !23, line: 2, type: !24, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !23 = !DIFile(filename: "./defaulted_default.h", directory: "")
// CHECK:STDOUT: !24 = !DISubroutineType(types: !25)
// CHECK:STDOUT: !25 = !{!26}
// CHECK:STDOUT: !26 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !27 = !DILocation(line: 2, column: 8, scope: !22)
// CHECK:STDOUT: !28 = !{!29, !29, i64 0}
// CHECK:STDOUT: !29 = !{!"p1 _ZTS26ExplicitlyDefaultedDefault", !21, i64 0}
// CHECK:STDOUT: !30 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:Default.cad83ce9b7db87bf.Core:ExplicitlyDefaultedDefault.Cpp", scope: null, file: !23, line: 7, type: !24, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !31 = !DILocation(line: 7, column: 3, scope: !30)
// CHECK:STDOUT: !32 = distinct !DISubprogram(name: "Op", linkageName: "_COp.d288d62be0e5d791:DefaultOrUnformed.Core.22b4681b019423d7", scope: null, file: !33, line: 9, type: !24, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !33 = !DIFile(filename: "min_prelude/parts/default.carbon", directory: "")
// CHECK:STDOUT: !34 = !DILocation(line: 9, column: 28, scope: !32)
// CHECK:STDOUT: !35 = !DILocation(line: 9, column: 21, scope: !32)
// CHECK:STDOUT: !36 = distinct !DISubprogram(name: "Op", linkageName: "_COp.d288d62be0e5d791:DefaultOrUnformed.Core.1a27c303748a2360", scope: null, file: !33, line: 9, type: !24, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !37 = !DILocation(line: 9, column: 28, scope: !36)
// CHECK:STDOUT: !38 = !DILocation(line: 9, column: 21, scope: !36)
// CHECK:STDOUT: !39 = !{!40, !8, i64 0}
// CHECK:STDOUT: !40 = !{!"_ZTS26ImplicitlyDefaultedDefault", !8, i64 0}
// CHECK:STDOUT: !41 = !{!42, !8, i64 0}
// CHECK:STDOUT: !42 = !{!"_ZTS26ExplicitlyDefaultedDefault", !8, i64 0}
// CHECK:STDOUT: ; ModuleID = 'call_copy.carbon'
// CHECK:STDOUT: source_filename = "call_copy.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCopy.Main(ptr sret([4 x i8]) %return, ptr %c) #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_10.1.temp = alloca [4 x i8], align 1, !dbg !17
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_10.1.temp), !dbg !17
// CHECK:STDOUT:   call void @_ZN4CopyC1ERKS_.carbon_thunk._(ptr %c, ptr %return), !dbg !17
// CHECK:STDOUT:   ret void, !dbg !18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN4CopyC1ERKS_.carbon_thunk._(ptr noundef %0, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !19
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !19
// CHECK:STDOUT:   %1 = load ptr, ptr %return.addr, align 8, !tbaa !19
// CHECK:STDOUT:   %2 = load ptr, ptr %.addr, align 8, !tbaa !19
// CHECK:STDOUT:   call void @_ZN4CopyC1ERKS_(ptr noundef nonnull align 4 dereferenceable(4) %1, ptr noundef nonnull align 4 dereferenceable(4) %2)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define void @"_COp:thunk:Copy.692da64e78c7deec.Core:Copy.Cpp"(ptr sret([4 x i8]) %return, ptr %self) #2 !dbg !22 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN4CopyC1ERKS_.carbon_thunk._(ptr %self, ptr %return), !dbg !26
// CHECK:STDOUT:   ret void, !dbg !26
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
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1, !2, !3, !4}
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!7}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !3 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !4 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !6, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIFile(filename: "call_copy.carbon", directory: "")
// CHECK:STDOUT: !7 = !{!8, !8, i64 0}
// CHECK:STDOUT: !8 = !{!"int", !9, i64 0}
// CHECK:STDOUT: !9 = !{!"omnipotent char", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "Copy", linkageName: "_CCopy.Main", scope: null, file: !6, line: 6, type: !12, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !15)
// CHECK:STDOUT: !12 = !DISubroutineType(types: !13)
// CHECK:STDOUT: !13 = !{!14, !14}
// CHECK:STDOUT: !14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !15 = !{!16}
// CHECK:STDOUT: !16 = !DILocalVariable(arg: 1, scope: !11, type: !14)
// CHECK:STDOUT: !17 = !DILocation(line: 7, column: 10, scope: !11)
// CHECK:STDOUT: !18 = !DILocation(line: 7, column: 3, scope: !11)
// CHECK:STDOUT: !19 = !{!20, !20, i64 0}
// CHECK:STDOUT: !20 = !{!"p1 _ZTS4Copy", !21, i64 0}
// CHECK:STDOUT: !21 = !{!"any pointer", !9, i64 0}
// CHECK:STDOUT: !22 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:Copy.692da64e78c7deec.Core:Copy.Cpp", scope: null, file: !23, line: 5, type: !12, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !24)
// CHECK:STDOUT: !23 = !DIFile(filename: "./copy.h", directory: "")
// CHECK:STDOUT: !24 = !{!25}
// CHECK:STDOUT: !25 = !DILocalVariable(arg: 1, scope: !22, type: !14)
// CHECK:STDOUT: !26 = !DILocation(line: 5, column: 3, scope: !22)
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
// CHECK:STDOUT: @_ZTV7Derived = linkonce_odr dso_local unnamed_addr constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr null, ptr @_ZTI7Derived] }, comdat, align 8
// CHECK:STDOUT: @_ZTT7Derived = linkonce_odr dso_local unnamed_addr constant [1 x ptr] [ptr getelementptr inbounds inrange(-24, 0) ({ [3 x ptr] }, ptr @_ZTV7Derived, i32 0, i32 0, i32 3)], comdat, align 8
// CHECK:STDOUT: @_ZTI7Derived = linkonce_odr dso_local constant { ptr, ptr, i32, i32, ptr, i64 } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv121__vmi_class_type_infoE, i64 2), ptr @_ZTS7Derived, i32 0, i32 1, ptr @_ZTI4Base, i64 -6141 }, comdat, align 8
// CHECK:STDOUT: @_ZTVN10__cxxabiv121__vmi_class_type_infoE = external global [0 x ptr]
// CHECK:STDOUT: @_ZTS7Derived = linkonce_odr dso_local constant [9 x i8] c"7Derived\00", comdat, align 1
// CHECK:STDOUT: @_ZTI4Base = linkonce_odr dso_local constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @_ZTS4Base }, comdat, align 8
// CHECK:STDOUT: @_ZTVN10__cxxabiv117__class_type_infoE = external global [0 x ptr]
// CHECK:STDOUT: @_ZTS4Base = linkonce_odr dso_local constant [6 x i8] c"4Base\00", comdat, align 1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_44.1.temp = alloca [8 x i8], align 1, !dbg !14
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_44.1.temp), !dbg !14
// CHECK:STDOUT:   call void @_ZN7DerivedC1Ev.carbon_thunk.(ptr %.loc7_44.1.temp), !dbg !14
// CHECK:STDOUT:   ret void, !dbg !15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN7DerivedC1Ev.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !16
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !16
// CHECK:STDOUT:   call void @_ZN7DerivedC1Ev(ptr noundef nonnull align 8 dereferenceable(8) %0) #0
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN7DerivedC1Ev(ptr noundef nonnull align 8 dereferenceable(8) %this) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !16
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   store ptr getelementptr inbounds inrange(-24, 0) ({ [3 x ptr] }, ptr @_ZTV7Derived, i32 0, i32 0, i32 3), ptr %this1, align 8, !tbaa !19
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr getelementptr inbounds inrange(-24, 0) ({ [3 x ptr] }, ptr @_ZTV7Derived, i32 0, i32 0, i32 3), { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { inlinehint mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1, !2, !3, !4}
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!7}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !3 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !4 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !6, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIFile(filename: "call_c1_in_thunk.carbon", directory: "")
// CHECK:STDOUT: !7 = !{!8, !8, i64 0}
// CHECK:STDOUT: !8 = !{!"int", !9, i64 0}
// CHECK:STDOUT: !9 = !{!"omnipotent char", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !6, line: 6, type: !12, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !12 = !DISubroutineType(types: !13)
// CHECK:STDOUT: !13 = !{null}
// CHECK:STDOUT: !14 = !DILocation(line: 7, column: 24, scope: !11)
// CHECK:STDOUT: !15 = !DILocation(line: 6, column: 1, scope: !11)
// CHECK:STDOUT: !16 = !{!17, !17, i64 0}
// CHECK:STDOUT: !17 = !{!"p1 _ZTS7Derived", !18, i64 0}
// CHECK:STDOUT: !18 = !{!"any pointer", !9, i64 0}
// CHECK:STDOUT: !19 = !{!20, !20, i64 0}
// CHECK:STDOUT: !20 = !{!"vtable pointer", !10, i64 0}
// CHECK:STDOUT: ; ModuleID = 'call_multi_argument.carbon'
// CHECK:STDOUT: source_filename = "call_multi_argument.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN1CC2Ev = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN1CC2Ei = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN1CC2Eii = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN1CC2Eiiii = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CZero.Main() #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_26.1.temp = alloca [1 x i8], align 1, !dbg !14
// CHECK:STDOUT:   %.loc8_19.2.temp = alloca [1 x i8], align 1, !dbg !15
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_26.1.temp), !dbg !14
// CHECK:STDOUT:   call void @_ZN1CC1Ev.carbon_thunk.(ptr %.loc7_26.1.temp), !dbg !14
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc8_19.2.temp), !dbg !15
// CHECK:STDOUT:   call void @_ZN1CC1Ev.carbon_thunk_tuple.(ptr %.loc8_19.2.temp), !dbg !15
// CHECK:STDOUT:   ret void, !dbg !16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN1CC1Ev.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !17
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !17
// CHECK:STDOUT:   call void @_ZN1CC2Ev(ptr noundef nonnull align 1 dereferenceable(1) %0)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN1CC1Ev.carbon_thunk_tuple.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !17
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !17
// CHECK:STDOUT:   call void @_ZN1CC2Ev(ptr noundef nonnull align 1 dereferenceable(1) %0)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_COne.Main() #0 !dbg !20 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc12_27.1.temp = alloca [1 x i8], align 1, !dbg !21
// CHECK:STDOUT:   %.loc13_18.1.temp = alloca [1 x i8], align 1, !dbg !22
// CHECK:STDOUT:   %.loc14_21.2.temp = alloca [1 x i8], align 1, !dbg !23
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc12_27.1.temp), !dbg !21
// CHECK:STDOUT:   call void @_ZN1CC1Ei.carbon_thunk._(i32 1, ptr %.loc12_27.1.temp), !dbg !21
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc13_18.1.temp), !dbg !22
// CHECK:STDOUT:   call void @_ZN1CC1Ei.carbon_thunk._(i32 2, ptr %.loc13_18.1.temp), !dbg !22
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc14_21.2.temp), !dbg !23
// CHECK:STDOUT:   call void @_ZN1CC1Ei.carbon_thunk_tuple._(i32 3, ptr %.loc14_21.2.temp), !dbg !23
// CHECK:STDOUT:   ret void, !dbg !24
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN1CC1Ei.carbon_thunk._(i32 noundef %0, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca i32, align 4
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store i32 %0, ptr %.addr, align 4, !tbaa !7
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !17
// CHECK:STDOUT:   %1 = load ptr, ptr %return.addr, align 8, !tbaa !17
// CHECK:STDOUT:   %2 = load i32, ptr %.addr, align 4, !tbaa !7
// CHECK:STDOUT:   call void @_ZN1CC2Ei(ptr noundef nonnull align 1 dereferenceable(1) %1, i32 noundef %2)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN1CC1Ei.carbon_thunk_tuple._(i32 noundef %0, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca i32, align 4
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store i32 %0, ptr %.addr, align 4, !tbaa !7
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !17
// CHECK:STDOUT:   %1 = load ptr, ptr %return.addr, align 8, !tbaa !17
// CHECK:STDOUT:   %2 = load i32, ptr %.addr, align 4, !tbaa !7
// CHECK:STDOUT:   call void @_ZN1CC2Ei(ptr noundef nonnull align 1 dereferenceable(1) %1, i32 noundef %2)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTwo.Main() #0 !dbg !25 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc18_30.1.temp = alloca [1 x i8], align 1, !dbg !26
// CHECK:STDOUT:   %.loc19_23.2.temp = alloca [1 x i8], align 1, !dbg !27
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc18_30.1.temp), !dbg !26
// CHECK:STDOUT:   call void @_ZN1CC1Eii.carbon_thunk.__(i32 1, i32 2, ptr %.loc18_30.1.temp), !dbg !26
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc19_23.2.temp), !dbg !27
// CHECK:STDOUT:   call void @_ZN1CC1Eii.carbon_thunk_tuple.__(i32 3, i32 4, ptr %.loc19_23.2.temp), !dbg !27
// CHECK:STDOUT:   ret void, !dbg !28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN1CC1Eii.carbon_thunk.__(i32 noundef %0, i32 noundef %1, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca i32, align 4
// CHECK:STDOUT:   %.addr1 = alloca i32, align 4
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store i32 %0, ptr %.addr, align 4, !tbaa !7
// CHECK:STDOUT:   store i32 %1, ptr %.addr1, align 4, !tbaa !7
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !17
// CHECK:STDOUT:   %2 = load ptr, ptr %return.addr, align 8, !tbaa !17
// CHECK:STDOUT:   %3 = load i32, ptr %.addr, align 4, !tbaa !7
// CHECK:STDOUT:   %4 = load i32, ptr %.addr1, align 4, !tbaa !7
// CHECK:STDOUT:   call void @_ZN1CC2Eii(ptr noundef nonnull align 1 dereferenceable(1) %2, i32 noundef %3, i32 noundef %4)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN1CC1Eii.carbon_thunk_tuple.__(i32 noundef %0, i32 noundef %1, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca i32, align 4
// CHECK:STDOUT:   %.addr1 = alloca i32, align 4
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store i32 %0, ptr %.addr, align 4, !tbaa !7
// CHECK:STDOUT:   store i32 %1, ptr %.addr1, align 4, !tbaa !7
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !17
// CHECK:STDOUT:   %2 = load ptr, ptr %return.addr, align 8, !tbaa !17
// CHECK:STDOUT:   %3 = load i32, ptr %.addr, align 4, !tbaa !7
// CHECK:STDOUT:   %4 = load i32, ptr %.addr1, align 4, !tbaa !7
// CHECK:STDOUT:   call void @_ZN1CC2Eii(ptr noundef nonnull align 1 dereferenceable(1) %2, i32 noundef %3, i32 noundef %4)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CThree.Main() #0 !dbg !29 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc23_33.1.temp = alloca [1 x i8], align 1, !dbg !30
// CHECK:STDOUT:   %.loc24_26.2.temp = alloca [1 x i8], align 1, !dbg !31
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc23_33.1.temp), !dbg !30
// CHECK:STDOUT:   call void @_ZN1CC1Eiiii.carbon_thunk.___(i32 1, i32 2, i32 3, ptr %.loc23_33.1.temp), !dbg !30
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc24_26.2.temp), !dbg !31
// CHECK:STDOUT:   call void @_ZN1CC1Eiiii.carbon_thunk_tuple.___(i32 4, i32 5, i32 6, ptr %.loc24_26.2.temp), !dbg !31
// CHECK:STDOUT:   ret void, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN1CC1Eiiii.carbon_thunk.___(i32 noundef %0, i32 noundef %1, i32 noundef %2, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca i32, align 4
// CHECK:STDOUT:   %.addr1 = alloca i32, align 4
// CHECK:STDOUT:   %.addr2 = alloca i32, align 4
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store i32 %0, ptr %.addr, align 4, !tbaa !7
// CHECK:STDOUT:   store i32 %1, ptr %.addr1, align 4, !tbaa !7
// CHECK:STDOUT:   store i32 %2, ptr %.addr2, align 4, !tbaa !7
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !17
// CHECK:STDOUT:   %3 = load ptr, ptr %return.addr, align 8, !tbaa !17
// CHECK:STDOUT:   %4 = load i32, ptr %.addr, align 4, !tbaa !7
// CHECK:STDOUT:   %5 = load i32, ptr %.addr1, align 4, !tbaa !7
// CHECK:STDOUT:   %6 = load i32, ptr %.addr2, align 4, !tbaa !7
// CHECK:STDOUT:   call void @_ZN1CC2Eiiii(ptr noundef nonnull align 1 dereferenceable(1) %3, i32 noundef %4, i32 noundef %5, i32 noundef %6, i32 noundef 0)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN1CC1Eiiii.carbon_thunk_tuple.___(i32 noundef %0, i32 noundef %1, i32 noundef %2, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca i32, align 4
// CHECK:STDOUT:   %.addr1 = alloca i32, align 4
// CHECK:STDOUT:   %.addr2 = alloca i32, align 4
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store i32 %0, ptr %.addr, align 4, !tbaa !7
// CHECK:STDOUT:   store i32 %1, ptr %.addr1, align 4, !tbaa !7
// CHECK:STDOUT:   store i32 %2, ptr %.addr2, align 4, !tbaa !7
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !17
// CHECK:STDOUT:   %3 = load ptr, ptr %return.addr, align 8, !tbaa !17
// CHECK:STDOUT:   %4 = load i32, ptr %.addr, align 4, !tbaa !7
// CHECK:STDOUT:   %5 = load i32, ptr %.addr1, align 4, !tbaa !7
// CHECK:STDOUT:   %6 = load i32, ptr %.addr2, align 4, !tbaa !7
// CHECK:STDOUT:   call void @_ZN1CC2Eiiii(ptr noundef nonnull align 1 dereferenceable(1) %3, i32 noundef %4, i32 noundef %5, i32 noundef %6, i32 noundef 0)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CFour.Main() #0 !dbg !33 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc28_36.1.temp = alloca [1 x i8], align 1, !dbg !34
// CHECK:STDOUT:   %.loc29_29.2.temp = alloca [1 x i8], align 1, !dbg !35
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc28_36.1.temp), !dbg !34
// CHECK:STDOUT:   call void @_ZN1CC1Eiiii.carbon_thunk.____(i32 1, i32 2, i32 3, i32 4, ptr %.loc28_36.1.temp), !dbg !34
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc29_29.2.temp), !dbg !35
// CHECK:STDOUT:   call void @_ZN1CC1Eiiii.carbon_thunk_tuple.____(i32 5, i32 6, i32 7, i32 8, ptr %.loc29_29.2.temp), !dbg !35
// CHECK:STDOUT:   ret void, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN1CC1Eiiii.carbon_thunk.____(i32 noundef %0, i32 noundef %1, i32 noundef %2, i32 noundef %3, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca i32, align 4
// CHECK:STDOUT:   %.addr1 = alloca i32, align 4
// CHECK:STDOUT:   %.addr2 = alloca i32, align 4
// CHECK:STDOUT:   %.addr3 = alloca i32, align 4
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store i32 %0, ptr %.addr, align 4, !tbaa !7
// CHECK:STDOUT:   store i32 %1, ptr %.addr1, align 4, !tbaa !7
// CHECK:STDOUT:   store i32 %2, ptr %.addr2, align 4, !tbaa !7
// CHECK:STDOUT:   store i32 %3, ptr %.addr3, align 4, !tbaa !7
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !17
// CHECK:STDOUT:   %4 = load ptr, ptr %return.addr, align 8, !tbaa !17
// CHECK:STDOUT:   %5 = load i32, ptr %.addr, align 4, !tbaa !7
// CHECK:STDOUT:   %6 = load i32, ptr %.addr1, align 4, !tbaa !7
// CHECK:STDOUT:   %7 = load i32, ptr %.addr2, align 4, !tbaa !7
// CHECK:STDOUT:   %8 = load i32, ptr %.addr3, align 4, !tbaa !7
// CHECK:STDOUT:   call void @_ZN1CC2Eiiii(ptr noundef nonnull align 1 dereferenceable(1) %4, i32 noundef %5, i32 noundef %6, i32 noundef %7, i32 noundef %8)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN1CC1Eiiii.carbon_thunk_tuple.____(i32 noundef %0, i32 noundef %1, i32 noundef %2, i32 noundef %3, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca i32, align 4
// CHECK:STDOUT:   %.addr1 = alloca i32, align 4
// CHECK:STDOUT:   %.addr2 = alloca i32, align 4
// CHECK:STDOUT:   %.addr3 = alloca i32, align 4
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store i32 %0, ptr %.addr, align 4, !tbaa !7
// CHECK:STDOUT:   store i32 %1, ptr %.addr1, align 4, !tbaa !7
// CHECK:STDOUT:   store i32 %2, ptr %.addr2, align 4, !tbaa !7
// CHECK:STDOUT:   store i32 %3, ptr %.addr3, align 4, !tbaa !7
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !17
// CHECK:STDOUT:   %4 = load ptr, ptr %return.addr, align 8, !tbaa !17
// CHECK:STDOUT:   %5 = load i32, ptr %.addr, align 4, !tbaa !7
// CHECK:STDOUT:   %6 = load i32, ptr %.addr1, align 4, !tbaa !7
// CHECK:STDOUT:   %7 = load i32, ptr %.addr2, align 4, !tbaa !7
// CHECK:STDOUT:   %8 = load i32, ptr %.addr3, align 4, !tbaa !7
// CHECK:STDOUT:   call void @_ZN1CC2Eiiii(ptr noundef nonnull align 1 dereferenceable(1) %4, i32 noundef %5, i32 noundef %6, i32 noundef %7, i32 noundef %8)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN1CC2Ev(ptr noundef nonnull align 1 dereferenceable(1) %this) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !17
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN1CC2Ei(ptr noundef nonnull align 1 dereferenceable(1) %this, i32 noundef %0) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   %.addr = alloca i32, align 4
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !17
// CHECK:STDOUT:   store i32 %0, ptr %.addr, align 4, !tbaa !7
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN1CC2Eii(ptr noundef nonnull align 1 dereferenceable(1) %this, i32 noundef %0, i32 noundef %1) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   %.addr = alloca i32, align 4
// CHECK:STDOUT:   %.addr1 = alloca i32, align 4
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !17
// CHECK:STDOUT:   store i32 %0, ptr %.addr, align 4, !tbaa !7
// CHECK:STDOUT:   store i32 %1, ptr %.addr1, align 4, !tbaa !7
// CHECK:STDOUT:   %this2 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN1CC2Eiiii(ptr noundef nonnull align 1 dereferenceable(1) %this, i32 noundef %0, i32 noundef %1, i32 noundef %2, i32 noundef %3) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   %.addr = alloca i32, align 4
// CHECK:STDOUT:   %.addr1 = alloca i32, align 4
// CHECK:STDOUT:   %.addr2 = alloca i32, align 4
// CHECK:STDOUT:   %.addr3 = alloca i32, align 4
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !17
// CHECK:STDOUT:   store i32 %0, ptr %.addr, align 4, !tbaa !7
// CHECK:STDOUT:   store i32 %1, ptr %.addr1, align 4, !tbaa !7
// CHECK:STDOUT:   store i32 %2, ptr %.addr2, align 4, !tbaa !7
// CHECK:STDOUT:   store i32 %3, ptr %.addr3, align 4, !tbaa !7
// CHECK:STDOUT:   %this4 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_ZN1CC1Ei.carbon_thunk._, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1, !2, !3, !4}
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!7}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !3 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !4 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !6, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIFile(filename: "call_multi_argument.carbon", directory: "")
// CHECK:STDOUT: !7 = !{!8, !8, i64 0}
// CHECK:STDOUT: !8 = !{!"int", !9, i64 0}
// CHECK:STDOUT: !9 = !{!"omnipotent char", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "Zero", linkageName: "_CZero.Main", scope: null, file: !6, line: 6, type: !12, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !12 = !DISubroutineType(types: !13)
// CHECK:STDOUT: !13 = !{null}
// CHECK:STDOUT: !14 = !DILocation(line: 7, column: 18, scope: !11)
// CHECK:STDOUT: !15 = !DILocation(line: 8, column: 18, scope: !11)
// CHECK:STDOUT: !16 = !DILocation(line: 6, column: 1, scope: !11)
// CHECK:STDOUT: !17 = !{!18, !18, i64 0}
// CHECK:STDOUT: !18 = !{!"p1 _ZTS1C", !19, i64 0}
// CHECK:STDOUT: !19 = !{!"any pointer", !9, i64 0}
// CHECK:STDOUT: !20 = distinct !DISubprogram(name: "One", linkageName: "_COne.Main", scope: null, file: !6, line: 11, type: !12, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !21 = !DILocation(line: 12, column: 18, scope: !20)
// CHECK:STDOUT: !22 = !DILocation(line: 13, column: 18, scope: !20)
// CHECK:STDOUT: !23 = !DILocation(line: 14, column: 18, scope: !20)
// CHECK:STDOUT: !24 = !DILocation(line: 11, column: 1, scope: !20)
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "Two", linkageName: "_CTwo.Main", scope: null, file: !6, line: 17, type: !12, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !26 = !DILocation(line: 18, column: 18, scope: !25)
// CHECK:STDOUT: !27 = !DILocation(line: 19, column: 18, scope: !25)
// CHECK:STDOUT: !28 = !DILocation(line: 17, column: 1, scope: !25)
// CHECK:STDOUT: !29 = distinct !DISubprogram(name: "Three", linkageName: "_CThree.Main", scope: null, file: !6, line: 22, type: !12, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !30 = !DILocation(line: 23, column: 18, scope: !29)
// CHECK:STDOUT: !31 = !DILocation(line: 24, column: 18, scope: !29)
// CHECK:STDOUT: !32 = !DILocation(line: 22, column: 1, scope: !29)
// CHECK:STDOUT: !33 = distinct !DISubprogram(name: "Four", linkageName: "_CFour.Main", scope: null, file: !6, line: 27, type: !12, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !34 = !DILocation(line: 28, column: 18, scope: !33)
// CHECK:STDOUT: !35 = !DILocation(line: 29, column: 18, scope: !33)
// CHECK:STDOUT: !36 = !DILocation(line: 27, column: 1, scope: !33)
