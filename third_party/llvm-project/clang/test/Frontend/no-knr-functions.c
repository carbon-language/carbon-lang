// Ensure that we cannot disable strict prototypes, no matter how hard we try.

// RUN: not %clang_cc1 -fno-no-knr-functions -x c++ %s 2>&1 | FileCheck --check-prefixes=NONO %s
// RUN: not %clang_cc1 -fno-no-knr-functions -x c %s 2>&1 | FileCheck --check-prefixes=NONO %s
// RUN: not %clang_cc1 -fno-no-knr-functions -std=c89 -x c %s 2>&1 | FileCheck --check-prefixes=NONO %s
// RUN: not %clang_cc1 -fknr-functions -x c++ %s 2>&1 | FileCheck --check-prefixes=POS %s
// RUN: not %clang_cc1 -fknr-functions -x c %s 2>&1 | FileCheck --check-prefixes=POS %s
// RUN: not %clang_cc1 -fknr-functions -std=c89 -x c %s 2>&1 | FileCheck --check-prefixes=POS %s

// NONO: error: unknown argument: '-fno-no-knr-functions'
// POS: error: unknown argument: '-fknr-functions'
