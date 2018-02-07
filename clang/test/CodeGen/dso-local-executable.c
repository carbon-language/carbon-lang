// RUN: %clang_cc1 -triple x86_64-pc-linux -emit-llvm -mrelocation-model static %s -o - | FileCheck --check-prefix=STATIC %s
// STATIC-DAG: @bar = external dso_local global i32
// STATIC-DAG: @weak_bar = extern_weak dso_local global i32
// STATIC-DAG: declare dso_local void @foo()
// STATIC-DAG: @baz = dso_local global i32 42
// STATIC-DAG: define dso_local i32* @zed()

// RUN: %clang_cc1 -triple x86_64-pc-linux -emit-llvm -pic-is-pie -mpie-copy-relocations %s -o - | FileCheck --check-prefix=PIE-COPY %s
// PIE-COPY-DAG: @bar = external dso_local global i32
// PIE-COPY-DAG: @weak_bar = extern_weak global i32
// PIE-COPY-DAG: declare dso_local void @foo()
// PIE-COPY-DAG: @baz = dso_local global i32 42
// PIE-COPY-DAG: define dso_local i32* @zed()

// RUN: %clang_cc1 -triple x86_64-pc-linux -emit-llvm -pic-is-pie %s -o - | FileCheck --check-prefix=PIE %s
// PIE-DAG: @bar = external global i32
// PIE-DAG: @weak_bar = extern_weak global i32
// PIE-DAG: declare dso_local void @foo()
// PIE-DAG: @baz = dso_local global i32 42
// PIE-DAG: define dso_local i32* @zed()

// RUN: %clang_cc1 -triple x86_64-pc-linux -emit-llvm -mrelocation-model static -fno-plt %s -o - | FileCheck --check-prefix=NOPLT %s
// NOPLT-DAG: @bar = external dso_local global i32
// NOPLT-DAG: @weak_bar = extern_weak dso_local global i32
// NOPLT-DAG: declare void @foo()
// NOPLT-DAG: @baz = dso_local global i32 42
// NOPLT-DAG: define dso_local i32* @zed()

// RUN: %clang_cc1 -triple x86_64-pc-linux -emit-llvm -fno-plt -pic-is-pie -mpie-copy-relocations %s -o - | FileCheck --check-prefix=PIE-COPY-NOPLT %s
// PIE-COPY-NOPLT-DAG: @bar = external dso_local global i32
// PIE-COPY-NOPLT-DAG: @weak_bar = extern_weak global i32
// PIE-COPY-NOPLT-DAG: declare void @foo()
// PIE-COPY-NOPLT-DAG: @baz = dso_local global i32 42
// PIE-COPY-NOPLT-DAG: define dso_local i32* @zed()

// RUN: %clang_cc1 -triple x86_64-pc-linux -emit-llvm -pic-is-pie -fno-plt %s -o - | FileCheck --check-prefix=PIE-NO-PLT %s
// RUN: %clang_cc1 -triple powerpc64le-pc-linux -emit-llvm -mrelocation-model static %s -o - | FileCheck --check-prefix=PIE-NO-PLT %s
// PIE-NO-PLT-DAG: @bar = external global i32
// PIE-NO-PLT-DAG: @weak_bar = extern_weak global i32
// PIE-NO-PLT-DAG: declare void @foo()
// PIE-NO-PLT-DAG: @baz = dso_local global i32 42
// PIE-NO-PLT-DAG: define dso_local i32* @zed()

// RUN: %clang_cc1 -triple x86_64-pc-linux -emit-llvm %s -o - | FileCheck --check-prefix=SHARED %s
// SHARED-DAG: @bar = external global i32
// SHARED-DAG: @weak_bar = extern_weak global i32
// SHARED-DAG: declare void @foo()
// SHARED-DAG: @baz = global i32 42
// SHARED-DAG: define i32* @zed()

extern int bar;
__attribute__((weak)) extern int weak_bar;
void foo(void);

int baz = 42;
int *zed() {
  foo();
  return baz ? &weak_bar : &bar;
}
