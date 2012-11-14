// RUN: %clang_cc1 -fsyntax-only -pedantic-errors %s 2>&1 | FileCheck %s --check-prefix=PRESUMED
// RUN: %clang_cc1 -fsyntax-only -pedantic-errors -fno-diagnostics-use-presumed-location %s 2>&1 | FileCheck %s --check-prefix=SPELLING

#line 100
#define X(y) y
X(int n = error);

// PRESUMED: diag-presumed.c:101:11: error: use of undeclared identifier 'error'
// PRESUMED: diag-presumed.c:100:14: note: expanded from
// SPELLING: diag-presumed.c:6:11: error: use of undeclared identifier 'error'
// SPELLING: diag-presumed.c:5:14: note: expanded from

;
// PRESUMED: diag-presumed.c:108:1: error: extra ';' outside of a functio
// SPELLING: diag-presumed.c:13:1: error: extra ';' outside of a functio

# 1 "thing1.cc" 1
# 1 "thing1.h" 1
# 1 "systemheader.h" 1 3
;
// No diagnostic here: we're in a system header, even if we're using spelling
// locations for the diagnostics..
// PRESUMED-NOT: extra ';'
// SPELLING-NOT: extra ';'

another error;
// PRESUMED: included from {{.*}}diag-presumed.c:112:
// PRESUMED: from thing1.cc:1:
// PRESUMED: from thing1.h:1:
// PRESUMED: systemheader.h:7:1: error: unknown type name 'another'

// SPELLING-NOT: included from
// SPELLING: diag-presumed.c:26:1: error: unknown type name 'another'

# 1 "thing1.h" 2
# 1 "thing1.cc" 2
