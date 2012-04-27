// Check that we error when -faltivec is specified on non-ppc platforms.

// RUN: %clang -ccc-clang-archs powerpc -target powerpc-unk-unk -faltivec -fsyntax-only %s
// RUN: %clang -ccc-clang-archs powerpc64 -target powerpc64-linux-gnu -faltivec -fsyntax-only %s

// RUN: %clang -target i386-pc-win32 -faltivec -fsyntax-only %s 2>&1 | FileCheck %s
// RUN: %clang -target x86_64-unknown-freebsd -faltivec -fsyntax-only %s 2>&1 | FileCheck %s
// RUN: %clang -target armv6-apple-darwin -faltivec -fsyntax-only %s 2>&1 | FileCheck %s
// RUN: %clang -target armv7-apple-darwin -faltivec -fsyntax-only %s 2>&1 | FileCheck %s
// RUN: %clang -ccc-clang-archs mips -target mips-linux-gnu -faltivec -fsyntax-only %s 2>&1 | FileCheck %s
// RUN: %clang -ccc-clang-archs mips64 -target mips64-linux-gnu -faltivec -fsyntax-only %s 2>&1 | FileCheck %s
// RUN: %clang -ccc-clang-archs sparc -target sparc-unknown-solaris -faltivec -fsyntax-only %s 2>&1 | FileCheck %s

// CHECK: invalid argument '-faltivec' only allowed with 'ppc/ppc64'
