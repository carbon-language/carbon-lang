// Check that we emit warnings/errors for different combinations of
// exceptions, rtti, and vptr sanitizer flags when targetting the PS4.
// No warnings/errors should be emitted for unknown, except if combining
// the vptr sanitizer with -fno-rtti

// RUN: %clang -### -c -fno-rtti -frtti %s 2>&1 | FileCheck -check-prefix=CHECK-RTTI %s
// RUN: %clang -### -c -frtti -fno-rtti %s 2>&1 | FileCheck -check-prefix=CHECK-NO-RTTI %s

// -fsanitize=vptr
// Make sure we only error/warn once, when trying to enable vptr and
// undefined and have -fno-rtti
// RUN: %clang -### -c -fsanitize=undefined -fsanitize=vptr -fno-rtti %s 2>&1 | FileCheck -check-prefix=CHECK-SAN-ERROR -check-prefix=CHECK-OK %s

// RUN: %clang -### -c -target x86_64-scei-ps4 -fsanitize=vptr %s 2>&1 | FileCheck -check-prefix=CHECK-SAN-WARN %s
// RUN: %clang -### -c -target x86_64-scei-ps4 -fsanitize=vptr -frtti %s 2>&1 | FileCheck -check-prefix=CHECK-OK %s
// RUN: %clang -### -c -target x86_64-scei-ps4 -fsanitize=vptr -fno-rtti %s 2>&1 | FileCheck -check-prefix=CHECK-SAN-ERROR %s
// RUN: %clang -### -c -target x86_64-scei-ps4 -fsanitize=undefined -frtti %s 2>&1 | FileCheck -check-prefix=CHECK-OK %s
// RUN: %clang -### -c -target x86_64-unknown-unknown -fsanitize=vptr %s 2>&1 | FileCheck -check-prefix=CHECK-OK %s
// RUN: %clang -### -c -target x86_64-unknown-unknown -fsanitize=vptr -frtti %s 2>&1 | FileCheck -check-prefix=CHECK-OK %s
// RUN: %clang -### -c -target x86_64-unknown-unknown -fsanitize=vptr -fno-rtti %s 2>&1 | FileCheck -check-prefix=CHECK-SAN-ERROR %s
// RUN: %clang -### -c -target x86_64-unknown-unknown -fsanitize=undefined %s 2>&1 | FileCheck -check-prefix=CHECK-OK %s
// RUN: %clang -### -c -target x86_64-unknown-unknown -fsanitize=undefined -frtti %s 2>&1 | FileCheck -check-prefix=CHECK-OK %s

// Exceptions + no/default rtti
// RUN: %clang -### -c -target x86_64-scei-ps4 -fcxx-exceptions -fno-rtti %s 2>&1 | FileCheck -check-prefix=CHECK-EXC-ERROR-CXX %s
// RUN: %clang -### -c -target x86_64-scei-ps4 -fcxx-exceptions %s 2>&1 | FileCheck -check-prefix=CHECK-EXC-WARN %s
// RUN: %clang -### -c -target x86_64-unknown-unknown -fcxx-exceptions -fno-rtti %s 2>&1 | FileCheck -check-prefix=CHECK-OK %s
// RUN: %clang -### -c -target x86_64-unknown-unknown -fcxx-exceptions %s 2>&1 | FileCheck -check-prefix=CHECK-OK %s
// In C++, -fexceptions implies -fcxx-exceptions
// RUN: %clang -x c++ -### -c -target x86_64-scei-ps4 -fexceptions -fno-rtti %s 2>&1 | FileCheck -check-prefix=CHECK-EXC-ERROR %s
// RUN: %clang -x c++ -### -c -target x86_64-scei-ps4 -fexceptions %s 2>&1 | FileCheck -check-prefix=CHECK-EXC-WARN %s
// RUN: %clang -x c++ -### -c -target x86_64-unknown-unknown -fexceptions -fno-rtti %s 2>&1 | FileCheck -check-prefix=CHECK-OK %s
// RUN: %clang -x c++ -### -c -target x86_64-unknown-unknown -fexceptions %s 2>&1 | FileCheck -check-prefix=CHECK-OK %s

// -frtti + exceptions
// RUN: %clang -### -c -target x86_64-scei-ps4 -fcxx-exceptions -frtti %s 2>&1 | FileCheck -check-prefix=CHECK-OK %s
// RUN: %clang -### -c -target x86_64-unknown-unknown -fcxx-exceptions -frtti %s 2>&1 | FileCheck -check-prefix=CHECK-OK %s

// -f{no-,}rtti/default
// RUN: %clang -### -c -target x86_64-scei-ps4 -frtti %s 2>&1 | FileCheck -check-prefix=CHECK-RTTI %s
// RUN: %clang -### -c -target x86_64-scei-ps4 -fno-rtti %s 2>&1 | FileCheck -check-prefix=CHECK-NO-RTTI %s
// RUN: %clang -### -c -target x86_64-scei-ps4 %s 2>&1 | FileCheck -check-prefix=CHECK-NO-RTTI %s
// RUN: %clang -### -c -target x86_64-unknown-unknown -frtti %s 2>&1 | FileCheck -check-prefix=CHECK-RTTI %s
// RUN: %clang -### -c -target x86_64-unknown-unknown -fno-rtti %s 2>&1 | FileCheck -check-prefix=CHECK-NO-RTTI %s
// RUN: %clang -### -c -target x86_64-unknown-unknown %s 2>&1 | FileCheck -check-prefix=CHECK-RTTI %s

// CHECK-SAN-WARN: implicitly disabling vptr sanitizer because rtti wasn't enabled
// CHECK-SAN-ERROR: invalid argument '-fsanitize=vptr' not allowed with '-fno-rtti'
// CHECK-EXC-WARN: implicitly enabling rtti for exception handling
// CHECK-EXC-ERROR: invalid argument '-fno-rtti' not allowed with '-fexceptions'
// CHECK-EXC-ERROR-CXX: invalid argument '-fno-rtti' not allowed with '-fcxx-exceptions'
// CHECK-RTTI-NOT: "-fno-rtti"
// CHECK-NO-RTTI-NOT: "-frtti"

// CHECK-OK-NOT: {{warning:|error:}}
