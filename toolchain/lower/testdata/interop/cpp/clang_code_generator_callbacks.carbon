// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
// EXTRA-ARGS: --target=x86_64-linux-gnu
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/clang_code_generator_callbacks.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/clang_code_generator_callbacks.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''

// Clang uses an implementation of clang::ASTConsumer to implement code
// generation, it has a variety of callbacks that may be called throughout the
// process of parsing/semantically analyzing the source code. Below are test
// cases based on those callbacks to ensure that C++ code compiled as part of
// Carbon interop imports preserve Clang's behavior.

struct t1 {
  virtual void f1();
};
// HandleVTable - Emit a definition for `t1`'s vtable.
void t1::f1() { }

// CompleteExternalDeclaration - Only be related to the BPF target.

// CompleteTentativeDefinition - Only relevant to C code.

// AssignInheritanceModel - Only relevant to ms inheritance attribute

// HandleTagDeclRequiredDefinition - Only relevant to ms debug info (only
// fails clang/test/DebugInfo/CXX/dllimport-base-class.cpp)

// HandleTagDeclDefinition - updating previous IR when a definition for a type
// is provided.
struct S;
extern S a[10];
S(*b)[10] = &a;
struct S {
  int x;
};
int f() { return a[3].x; }
// without the callback, clang describes the type of 'a' as 10xi8, resulting in
// this:
//   load i32, ptr getelementptr inbounds ([10 x i8], ptr @a, i64 0, i64 3), align 4
// with the callback, the type gets updated and the result is:
//   load i32, ptr getelementptr inbounds ([10 x %struct.S], ptr @a, i64 0, i64 3), align 4

// HandleInlineFunctionDefinition+EmitDeferredDecls - Emit a definition of
// `func`.
struct t2 {
  __attribute__((used)) void func() {}
};

// HandleCXXStaticMemberVarInstantiation - Emit a definition for
// `t3<int>::i`.
template<typename T>
struct t3 {
  static int i;
};
template<typename T>
int t3<T>::i;
void f2() {
  t3<int>::i = 42;
}
''';

// CHECK:STDOUT: ; ModuleID = 'clang_code_generator_callbacks.carbon'
// CHECK:STDOUT: source_filename = "clang_code_generator_callbacks.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %struct.S = type { i32 }
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN2t24funcEv = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN2t3IiE1iE = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: @a = external global [10 x i8], align 1
// CHECK:STDOUT: @b = dso_local global ptr @a, align 8
// CHECK:STDOUT: @_ZN2t3IiE1iE = linkonce_odr dso_local global i32 0, comdat, align 4
// CHECK:STDOUT: @_ZTV2t1 = dso_local unnamed_addr constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr @_ZTI2t1, ptr @_ZN2t12f1Ev] }, align 8
// CHECK:STDOUT: @_ZTI2t1 = dso_local constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @_ZTS2t1 }, align 8
// CHECK:STDOUT: @_ZTVN10__cxxabiv117__class_type_infoE = external global [0 x ptr]
// CHECK:STDOUT: @_ZTS2t1 = dso_local constant [4 x i8] c"2t1\00", align 1
// CHECK:STDOUT: @llvm.compiler.used = appending global [1 x ptr] [ptr @_ZN2t24funcEv], section "llvm.metadata"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define dso_local void @_ZN2t12f1Ev(ptr noundef nonnull align 8 dereferenceable(8) %this) unnamed_addr #0 align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !11
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define dso_local noundef i32 @_Z1fv() #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %0 = load i32, ptr getelementptr inbounds ([10 x %struct.S], ptr @a, i64 0, i64 3), align 4, !tbaa !14
// CHECK:STDOUT:   ret i32 %0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN2t24funcEv(ptr noundef nonnull align 1 dereferenceable(1) %this) #0 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !16
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define dso_local void @_Z2f2v() #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   store i32 42, ptr @_ZN2t3IiE1iE, align 4, !tbaa !7
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
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
// CHECK:STDOUT: !6 = !DIFile(filename: "clang_code_generator_callbacks.carbon", directory: "")
// CHECK:STDOUT: !7 = !{!8, !8, i64 0}
// CHECK:STDOUT: !8 = !{!"int", !9, i64 0}
// CHECK:STDOUT: !9 = !{!"omnipotent char", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !11 = !{!12, !12, i64 0}
// CHECK:STDOUT: !12 = !{!"p1 _ZTS2t1", !13, i64 0}
// CHECK:STDOUT: !13 = !{!"any pointer", !9, i64 0}
// CHECK:STDOUT: !14 = !{!15, !8, i64 0}
// CHECK:STDOUT: !15 = !{!"_ZTS1S", !8, i64 0}
// CHECK:STDOUT: !16 = !{!17, !17, i64 0}
// CHECK:STDOUT: !17 = !{!"p1 _ZTS2t2", !13, i64 0}
