#include "annotate-tokens-include.h"

// RUN: c-index-test -test-annotate-tokens=%s:1:1:2:1 %s | FileCheck %s
// CHECK: Identifier: "include" [1:2 - 1:9] preprocessing directive=
// CHECK: Literal: ""annotate-tokens-include.h"" [1:10 - 1:37] preprocessing directive=

