#define BAR baz
#define WIBBLE(X, Y)
WIBBLE(int, float)
int BAR;
#include "foo.h"

// RUN: c-index-test -test-annotate-tokens=%s:1:1:6:1 -I%S/Inputs %s | FileCheck %s
// CHECK: Punctuation: "#" [1:1 - 1:2] preprocessing directive=
// CHECK: Identifier: "define" [1:2 - 1:8] preprocessing directive=
// CHECK: Identifier: "BAR" [1:9 - 1:12] preprocessing directive=
// CHECK: Identifier: "baz" [1:13 - 1:16] preprocessing directive=
// CHECK: Punctuation: "#" [2:1 - 2:2] preprocessing directive=
// CHECK: Identifier: "define" [2:2 - 2:8] preprocessing directive=
// CHECK: Identifier: "WIBBLE" [2:9 - 2:15] preprocessing directive=
// CHECK: Punctuation: "(" [2:15 - 2:16] preprocessing directive=
// CHECK: Identifier: "X" [2:16 - 2:17] preprocessing directive=
// CHECK: Punctuation: "," [2:17 - 2:18] preprocessing directive=
// CHECK: Identifier: "Y" [2:19 - 2:20] preprocessing directive=
// CHECK: Punctuation: ")" [2:20 - 2:21] preprocessing directive=
// CHECK: Identifier: "WIBBLE" [3:1 - 3:7]
// CHECK: Punctuation: "(" [3:7 - 3:8]
// CHECK: Keyword: "int" [3:8 - 3:11]
// CHECK: Punctuation: "," [3:11 - 3:12]
// CHECK: Keyword: "float" [3:13 - 3:18]
// CHECK: Punctuation: ")" [3:18 - 3:19]
// CHECK: Keyword: "int" [4:1 - 4:4]
// CHECK: Identifier: "BAR" [4:5 - 4:8]
// CHECK: Punctuation: ";" [4:8 - 4:9]
// CHECK: Punctuation: "#" [5:1 - 5:2] preprocessing directive=
// CHECK: Identifier: "include" [5:2 - 5:9] preprocessing directive=
// CHECK: Literal: ""foo.h"" [5:10 - 5:17] preprocessing directive=
