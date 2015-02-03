// RUN: %clang -M -MG -include nonexistent-preinclude.h %s > %t
// RUN: grep -F nonexistent-preinclude.h %t
// RUN: grep -F nonexistent-ppinclude.h %t

#include "nonexistent-ppinclude.h"
