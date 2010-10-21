#define OBSCURE(X) X
#define DECORATION

typedef int T;
void OBSCURE(func)(int x) {
  OBSCURE(T) DECORATION value;
}

#include "a.h"

// RUN: c-index-test -cursor-at=%s:1:11 -I%S/Inputs %s | FileCheck -check-prefix=CHECK-1 %s
// CHECK-1: macro definition=OBSCURE
// RUN: c-index-test -cursor-at=%s:2:14 -I%S/Inputs %s | FileCheck -check-prefix=CHECK-2 %s
// CHECK-2: macro definition=DECORATION
// RUN: c-index-test -cursor-at=%s:5:7 -I%S/Inputs %s | FileCheck -check-prefix=CHECK-3 %s
// CHECK-3: macro instantiation=OBSCURE:1:9
// RUN: c-index-test -cursor-at=%s:6:6 -I%S/Inputs %s | FileCheck -check-prefix=CHECK-4 %s
// CHECK-4: macro instantiation=OBSCURE:1:9
// RUN: c-index-test -cursor-at=%s:6:19 -I%S/Inputs %s | FileCheck -check-prefix=CHECK-5 %s
// CHECK-5: macro instantiation=DECORATION:2:9
// RUN: c-index-test -cursor-at=%s:9:10 -I%S/Inputs %s | FileCheck -check-prefix=CHECK-6 %s
// CHECK-6: inclusion directive=a.h
