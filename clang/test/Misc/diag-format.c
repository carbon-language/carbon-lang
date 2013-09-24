// RUN: %clang -fsyntax-only  %s 2>&1 | FileCheck %s -check-prefix=DEFAULT
// RUN: %clang -fsyntax-only -fdiagnostics-format=clang %s 2>&1 | FileCheck %s -check-prefix=DEFAULT
// RUN: %clang -fsyntax-only -fdiagnostics-format=clang -target x86_64-pc-win32 %s 2>&1 | FileCheck %s -check-prefix=DEFAULT
//
// RUN: %clang -fsyntax-only -fdiagnostics-format=msvc  %s 2>&1 | FileCheck %s -check-prefix=MSVC
// RUN: %clang -fsyntax-only -fdiagnostics-format=msvc -target x86_64-pc-win32 %s 2>&1 | FileCheck %s -check-prefix=MSVC
// RUN: %clang -fsyntax-only -fdiagnostics-format=msvc -target x86_64-pc-win32 -fshow-column %s 2>&1 | FileCheck %s -check-prefix=MSVC
//
// RUN: %clang -fsyntax-only -fdiagnostics-format=vi    %s 2>&1 | FileCheck %s -check-prefix=VI
//
// RUN: %clang -fsyntax-only -fdiagnostics-format=msvc -fno-show-column %s 2>&1 | FileCheck %s -check-prefix=MSVC_ORIG
//
// RUN: %clang -fsyntax-only -fno-show-column %s 2>&1 | FileCheck %s -check-prefix=NO_COLUMN
//
// RUN: not %clang -fsyntax-only -Werror -fdiagnostics-format=msvc-fallback %s 2>&1 | FileCheck %s -check-prefix=MSVC-FALLBACK











#ifdef foo
#endif bad // extension!
// DEFAULT: {{.*}}:28:8: warning: extra tokens at end of #endif directive [-Wextra-tokens]
// MSVC: {{.*}}(28,7) : warning: extra tokens at end of #endif directive [-Wextra-tokens]
// VI: {{.*}} +28:8: warning: extra tokens at end of #endif directive [-Wextra-tokens]
// MSVC_ORIG: {{.*}}(28) : warning: extra tokens at end of #endif directive [-Wextra-tokens]
// NO_COLUMN: {{.*}}:28: warning: extra tokens at end of #endif directive [-Wextra-tokens]
// MSVC-FALLBACK: {{.*}}(28,7) : error(clang): extra tokens at end of #endif directive
int x;
