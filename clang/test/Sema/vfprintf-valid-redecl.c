// RUN: %clang_cc1 %s -fsyntax-only -pedantic -verify
// expected-no-diagnostics

// PR16344
// Clang has defined 'vfprint' in builtin list. If the following line occurs before any other
// `vfprintf' in this file, and we getPreviousDecl()->getTypeSourceInfo() on it, then we will
// get a null pointer since the one in builtin list doesn't has valid TypeSourceInfo.
int vfprintf(void) { return 0; }

// PR4290
// The following declaration is compatible with vfprintf, so we shouldn't
// warn.
int vfprintf();
