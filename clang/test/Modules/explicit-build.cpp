// RUN: rm -rf %t

// -------------------------------
// Build chained modules A, B, and C
// RUN: %clang_cc1 -x c++ -fmodules -fmodules-cache-path=%t -Rmodule-build -fno-modules-error-recovery \
// RUN:            -fmodule-name=a -emit-module %S/Inputs/explicit-build/module.modulemap -o %t/a.pcm \
// RUN:            2>&1 | FileCheck --check-prefix=CHECK-NO-IMPLICIT-BUILD %s --allow-empty
//
// RUN: %clang_cc1 -x c++ -fmodules -fmodules-cache-path=%t -Rmodule-build -fno-modules-error-recovery \
// RUN:            -fmodule-file=%t/a.pcm \
// RUN:            -fmodule-name=b -emit-module %S/Inputs/explicit-build/module.modulemap -o %t/b.pcm \
// RUN:            2>&1 | FileCheck --check-prefix=CHECK-NO-IMPLICIT-BUILD %s --allow-empty
//
// RUN: %clang_cc1 -x c++ -fmodules -fmodules-cache-path=%t -Rmodule-build -fno-modules-error-recovery \
// RUN:            -fmodule-file=%t/b.pcm \
// RUN:            -fmodule-name=c -emit-module %S/Inputs/explicit-build/module.modulemap -o %t/c.pcm \
// RUN:            2>&1 | FileCheck --check-prefix=CHECK-NO-IMPLICIT-BUILD %s --allow-empty
//
// CHECK-NO-IMPLICIT-BUILD-NOT: building module

// -------------------------------
// Build B with an implicit build of A
// RUN: %clang_cc1 -x c++ -fmodules -fmodules-cache-path=%t -Rmodule-build -fno-modules-error-recovery \
// RUN:            -fmodule-name=b -emit-module %S/Inputs/explicit-build/module.modulemap -o %t/b-not-a.pcm \
// RUN:            2>&1 | FileCheck --check-prefix=CHECK-B-NO-A %s
//
// CHECK-B-NO-A: While building module 'b':
// CHECK-B-NO-A: building module 'a' as

// -------------------------------
// Check that we can use the explicitly-built A, B, and C modules.
// RUN: %clang_cc1 -x c++ -fmodules -fmodules-cache-path=%t -Rmodule-build -fno-modules-error-recovery \
// RUN:            -fmodule-file=%t/a.pcm \
// RUN:            -verify %s -DHAVE_A
//
// RUN: %clang_cc1 -x c++ -fmodules -fmodules-cache-path=%t -Rmodule-build -fno-modules-error-recovery \
// RUN:            -fmodule-file=%t/a.pcm \
// RUN:            -fmodule-map-file=%S/Inputs/explicit-build/module.modulemap \
// RUN:            -verify %s -DHAVE_A
//
// RUN: %clang_cc1 -x c++ -fmodules -fmodules-cache-path=%t -Rmodule-build -fno-modules-error-recovery \
// RUN:            -fmodule-file=%t/b.pcm \
// RUN:            -verify %s -DHAVE_A -DHAVE_B
//
// RUN: %clang_cc1 -x c++ -fmodules -fmodules-cache-path=%t -Rmodule-build -fno-modules-error-recovery \
// RUN:            -fmodule-file=%t/a.pcm \
// RUN:            -fmodule-file=%t/b.pcm \
// RUN:            -verify %s -DHAVE_A -DHAVE_B
//
// RUN: %clang_cc1 -x c++ -fmodules -fmodules-cache-path=%t -Rmodule-build -fno-modules-error-recovery \
// RUN:            -fmodule-file=%t/a.pcm \
// RUN:            -fmodule-file=%t/b.pcm \
// RUN:            -fmodule-file=%t/c.pcm \
// RUN:            -verify %s -DHAVE_A -DHAVE_B -DHAVE_C
//
// RUN: %clang_cc1 -x c++ -fmodules -fmodules-cache-path=%t -Rmodule-build -fno-modules-error-recovery \
// RUN:            -I%S/Inputs/explicit-build \
// RUN:            -fmodule-file=%t/a.pcm \
// RUN:            -fmodule-file=%t/b.pcm \
// RUN:            -fmodule-file=%t/c.pcm \
// RUN:            -verify %s -INCLUDE_ALL -DHAVE_A -DHAVE_B -DHAVE_C

#ifdef INCLUDE_ALL
  #include "a.h"
  #include "b.h"
  #include "c.h"
  static_assert(a == 1, "");
  static_assert(b == 2, "");
  static_assert(c == 3, "");
#else
  const int use_a = a;
  #ifndef HAVE_A
  // expected-error@-2 {{undeclared identifier}}
  #else
  // expected-error@-4 {{must be imported from module 'a'}}
  // expected-note@Inputs/explicit-build/a.h:* {{here}}
  #endif

  const int use_b = b;
  #ifndef HAVE_B
  // expected-error@-2 {{undeclared identifier}}
  #else
  // expected-error@-4 {{must be imported from module 'b'}}
  // expected-note@Inputs/explicit-build/b.h:* {{here}}
  #endif

  const int use_c = c;
  #ifndef HAVE_C
  // expected-error@-2 {{undeclared identifier}}
  #else
  // expected-error@-4 {{must be imported from module 'c'}}
  // expected-note@Inputs/explicit-build/c.h:* {{here}}
  #endif
#endif

// -------------------------------
// Check that we can use a mixture of implicit and explicit modules.
// RUN: not %clang_cc1 -x c++ -fmodules -fmodules-cache-path=%t -Rmodule-build -fno-modules-error-recovery \
// RUN:            -fmodule-file=%t/b-not-a.pcm \
// RUN:            -fmodule-map-file=%S/Inputs/explicit-build/module.modulemap \
// RUN:            %s 2>&1 | FileCheck --check-prefix=CHECK-A-AND-B-NO-A %s

// -------------------------------
// Check that mixing an implicit and explicit form of the 'a' module is rejected.
// RUN: not %clang_cc1 -x c++ -fmodules -fmodules-cache-path=%t -Rmodule-build -fno-modules-error-recovery \
// RUN:            -fmodule-file=%t/a.pcm \
// RUN:            -fmodule-file=%t/b-not-a.pcm \
// RUN:            %s 2>&1 | FileCheck --check-prefix=CHECK-A-AND-B-NO-A %s
//
// RUN: not %clang_cc1 -x c++ -fmodules -fmodules-cache-path=%t -Rmodule-build -fno-modules-error-recovery \
// RUN:            -fmodule-file=%t/a.pcm \
// RUN:            -fmodule-file=%t/b-not-a.pcm \
// RUN:            -fmodule-map-file=%S/Inputs/explicit-build/module.modulemap \
// RUN:            %s 2>&1 | FileCheck --check-prefix=CHECK-A-AND-B-NO-A %s
//
// FIXME: We should load module map files specified on the command line and
// module map files in include paths on demand to allow this, and possibly
// also the previous case.
// CHECK-A-AND-B-NO-A: fatal error: module 'a' {{.*}} is not defined in any loaded module map

// -------------------------------
// Try to use two different flavors of the 'a' module.
// RUN: %clang_cc1 -x c++ -fmodules -fmodules-cache-path=%t -Rmodule-build -fno-modules-error-recovery \
// RUN:            -fmodule-name=a -emit-module %S/Inputs/explicit-build/module.modulemap -o %t/a-alt.pcm \
// RUN:            2>&1 | FileCheck --check-prefix=CHECK-NO-IMPLICIT-BUILD %s --allow-empty
//
// RUN: not %clang_cc1 -x c++ -fmodules -fmodules-cache-path=%t -Rmodule-build -fno-modules-error-recovery \
// RUN:            -fmodule-file=%t/a.pcm \
// RUN:            -fmodule-file=%t/a-alt.pcm \
// RUN:            -fmodule-map-file=%S/Inputs/explicit-build/module.modulemap \
// RUN:            %s 2>&1 | FileCheck --check-prefix=CHECK-MULTIPLE-AS %s
//
// RUN: not %clang_cc1 -x c++ -fmodules -fmodules-cache-path=%t -Rmodule-build -fno-modules-error-recovery \
// RUN:            -fmodule-file=%t/a-alt.pcm \
// RUN:            -fmodule-file=%t/a.pcm \
// RUN:            -fmodule-map-file=%S/Inputs/explicit-build/module.modulemap \
// RUN:            %s 2>&1 | FileCheck --check-prefix=CHECK-MULTIPLE-AS %s
//
// CHECK-MULTIPLE-AS: error: module 'a' has already been loaded; cannot load module file '{{.*a(-alt)?}}.pcm'

// -------------------------------
// Try to import a PCH with -fmodule-file=
// RUN: %clang_cc1 -x c++ -fmodules -fmodules-cache-path=%t -Rmodule-build -fno-modules-error-recovery \
// RUN:            -fmodule-name=a -emit-pch %S/Inputs/explicit-build/a.h -o %t/a.pch \
// RUN:            2>&1 | FileCheck --check-prefix=CHECK-NO-IMPLICIT-BUILD %s --allow-empty
//
// RUN: not %clang_cc1 -x c++ -fmodules -fmodules-cache-path=%t -Rmodule-build -fno-modules-error-recovery \
// RUN:            -fmodule-file=%t/a.pch \
// RUN:            %s 2>&1 | FileCheck --check-prefix=CHECK-A-AS-PCH %s
//
// CHECK-A-AS-PCH: fatal error: AST file '{{.*}}a.pch' was not built as a module
