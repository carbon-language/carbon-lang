#define BAR baz
#define WIBBLE(X, Y) X##Y
float WIBBLE(int, float);
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
// CHECK: Identifier: "WIBBLE" [3:7 - 3:13] macro instantiation=
// CHECK: Punctuation: "(" [3:13 - 3:14]
// CHECK: Keyword: "int" [3:14 - 3:17]
// CHECK: Punctuation: "," [3:17 - 3:18]
// CHECK: Keyword: "float" [3:19 - 3:24]
// CHECK: Punctuation: ")" [3:24 - 3:25]
// CHECK: Punctuation: ";" [3:25 - 3:26]
// CHECK: Keyword: "int" [4:1 - 4:4]
// CHECK: Identifier: "BAR" [4:5 - 4:8] macro instantiation=
// CHECK: Punctuation: ";" [4:8 - 4:9]
// CHECK: Punctuation: "#" [5:1 - 5:2] preprocessing directive=
// CHECK: Identifier: "include" [5:2 - 5:9] preprocessing directive=
// CHECK: Literal: ""foo.h"" [5:10 - 5:17] preprocessing directive=
