// RUN: %check_clang_tidy %s portability-restrict-system-includes %t \
// RUN:     -- -config="{CheckOptions: [{key: portability-restrict-system-includes.Includes, value: '-*,stddef.h'}]}" \
// RUN:     -- -isystem %S/Inputs/portability-restrict-system-includes/system

// Test allow-list functionality: disallow all but stddef.h.

#include <stddef.h>
#include <stdint.h>
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: system include stdint.h not allowed
#include <float.h>
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: system include float.h not allowed
