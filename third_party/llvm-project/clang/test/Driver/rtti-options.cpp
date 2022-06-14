// Check that we emit warnings/errors for different combinations of
// exceptions, rtti, and vptr sanitizer flags for generic (unknown) x86 linux,
// and for PS4 when its behaviour differs from the generic x86-linux.
// No warnings/errors should be emitted for unknown, except if combining
// the vptr sanitizer with -fno-rtti

// Special cases: -fcxx-exceptions in C code should warn about unused arguments
// We should also not have any rtti-related arguments
// RUN: %clang -x c -### -target x86_64-unknown-unknown -c -fcxx-exceptions %s 2>&1 | FileCheck -check-prefix=CHECK-UNUSED -check-prefix=CHECK-RTTI %s

// Make sure we keep the last -frtti/-fno-rtti argument
// RUN: %clang -### -c -fno-rtti -frtti %s 2>&1 | FileCheck -check-prefix=CHECK-RTTI %s
// RUN: %clang -### -c -frtti -fno-rtti %s 2>&1 | FileCheck -check-prefix=CHECK-NO-RTTI %s

// -fsanitize=vptr
// Make sure we only error/warn once, when trying to enable vptr and
// undefined and have -fno-rtti
// RUN: %clang -### -c -target x86_64-unknown-linux -fsanitize=undefined -fsanitize=vptr -fno-rtti %s 2>&1 | FileCheck -check-prefix=CHECK-SAN-ERROR -check-prefix=CHECK-OK %s
// RUN: %clang -### -c -target x86_64-unknown-linux -fsanitize=vptr %s 2>&1 | FileCheck -check-prefix=CHECK-OK %s
// RUN: %clang -### -c -target x86_64-unknown-linux -fsanitize=vptr -frtti %s 2>&1 | FileCheck -check-prefix=CHECK-OK %s
// RUN: %clang -### -c -target x86_64-unknown-linux -fsanitize=vptr -fno-rtti %s 2>&1 | FileCheck -check-prefix=CHECK-SAN-ERROR %s
// RUN: %clang -### -c -target x86_64-unknown-linux -fsanitize=undefined %s 2>&1 | FileCheck -check-prefix=CHECK-OK %s
// RUN: %clang -### -c -target x86_64-unknown-linux -fsanitize=undefined -frtti %s 2>&1 | FileCheck -check-prefix=CHECK-OK %s
// RUN: %clang -### -c -target x86_64-scei-ps4 -fsanitize=vptr %s 2>&1 | FileCheck -check-prefix=CHECK-SAN-WARN %s
// RUN: %clang -### -c -target x86_64-sie-ps5 -fsanitize=vptr %s 2>&1 | FileCheck -check-prefix=CHECK-SAN-WARN %s

// Exceptions + no/default rtti
// RUN: %clang -### -c -target x86_64-unknown-unknown -fcxx-exceptions -fno-rtti %s 2>&1 | FileCheck -check-prefix=CHECK-OK %s
// RUN: %clang -### -c -target x86_64-unknown-unknown -fcxx-exceptions %s 2>&1 | FileCheck -check-prefix=CHECK-OK %s
// In C++, -fexceptions implies -fcxx-exceptions
// RUN: %clang -x c++ -### -c -target x86_64-unknown-unknown -fexceptions -fno-rtti %s 2>&1 | FileCheck -check-prefix=CHECK-OK %s
// RUN: %clang -x c++ -### -c -target x86_64-unknown-unknown -fexceptions %s 2>&1 | FileCheck -check-prefix=CHECK-OK %s

// -frtti + exceptions
// RUN: %clang -### -c -target x86_64-unknown-unknown -fcxx-exceptions -frtti %s 2>&1 | FileCheck -check-prefix=CHECK-OK %s

// -f{no-,}rtti/default
// RUN: %clang -### -c -target x86_64-scei-ps4 %s 2>&1 | FileCheck -check-prefix=CHECK-NO-RTTI %s
// RUN: %clang -### -c -target x86_64-sie-ps5 %s 2>&1 | FileCheck -check-prefix=CHECK-NO-RTTI %s
// RUN: %clang -### -c -target x86_64-unknown-unknown -frtti %s 2>&1 | FileCheck -check-prefix=CHECK-RTTI %s
// RUN: %clang -### -c -target x86_64-unknown-unknown -fno-rtti %s 2>&1 | FileCheck -check-prefix=CHECK-NO-RTTI %s
// RUN: %clang -### -c -target x86_64-unknown-unknown %s 2>&1 | FileCheck -check-prefix=CHECK-RTTI %s

// CHECK-UNUSED: warning: argument unused during compilation: '-fcxx-exceptions'
// CHECK-SAN-WARN: implicitly disabling vptr sanitizer because rtti wasn't enabled
// CHECK-SAN-ERROR: invalid argument '-fsanitize=vptr' not allowed with '-fno-rtti'
// CHECK-RTTI-NOT: "-fno-rtti"
// CHECK-NO-RTTI: "-fno-rtti"

// CHECK-OK-NOT: {{warning:|error:}}
