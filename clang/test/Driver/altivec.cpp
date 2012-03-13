// Check that we error when -faltivec is specified on a non-ppc platforms.

// RUN: %clang -arch ppc -faltivec -fsyntax-only %s
// RUN: %clang -arch ppc64 -faltivec -fsyntax-only %s

// RUN: not %clang -arch i386 -faltivec -fsyntax-only %s
// RUN: not %clang -arch x86_64 -faltivec -fsyntax-only %s
// RUN: not %clang -arch armv6 -faltivec -fsyntax-only %s
// RUN: not %clang -arch armv7 -faltivec -fsyntax-only %s
// RUN: not %clang -arch mips -faltivec -fsyntax-only %s
// RUN: not %clang -arch mips64 -faltivec -fsyntax-only %s
// RUN: not %clang -arch sparc -faltivec -fsyntax-only %s
