#ifndef HEADER
#define HEADER

#include "blahblah.h"
void erroneous(int);
void erroneous(float);

#else

void foo(void) {
  erroneous(0);
}

#endif

// RUN: c-index-test -write-pch %t.h.pch %s -Xclang -detailed-preprocessing-record
// RUN: c-index-test -test-load-source local %s -include %t.h -Xclang -detailed-preprocessing-record | FileCheck -check-prefix=CHECK-PARSE %s
// RUN: c-index-test -index-file %s -include %t.h -Xclang -detailed-preprocessing-record | FileCheck -check-prefix=CHECK-INDEX %s

// CHECK-PARSE: pch-with-errors.c:10:6: FunctionDecl=foo:10:6 (Definition) Extent=[10:1 - 12:2]
// CHECK-PARSE: pch-with-errors.c:11:3: CallExpr=erroneous:5:6 Extent=[11:3 - 11:15]

// CHECK-INDEX: [indexDeclaration]: kind: function | name: foo
// CHECK-INDEX: [indexEntityReference]: kind: function | name: erroneous

// RUN: %clang -fsyntax-only %s -include %t.h 2>&1 | FileCheck -check-prefix=PCH-ERR %s

// PCH-ERR: error: PCH file contains compiler errors
