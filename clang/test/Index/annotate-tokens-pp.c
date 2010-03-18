#define NOTHING(X,Y)
#define STILL_NOTHING NOTHING(honk,warble)
#define BAR baz
#define WIBBLE(X, Y) X##Y
NOTHING(more,junk) float WIBBLE(int, float);
int BAR STILL_NOTHING;
#include "foo.h"
#undef BAR

// RUN: c-index-test -test-annotate-tokens=%s:2:1:9:1 -I%S/Inputs %s | FileCheck %s
// CHECK: Punctuation: "#" [2:1 - 2:2] preprocessing directive=
// CHECK: Identifier: "define" [2:2 - 2:8] preprocessing directive=
// CHECK: Identifier: "STILL_NOTHING" [2:9 - 2:22] macro definition=STILL_NOTHING
// CHECK: Identifier: "NOTHING" [2:23 - 2:30] preprocessing directive=
// CHECK: Punctuation: "(" [2:30 - 2:31] preprocessing directive=
// CHECK: Identifier: "honk" [2:31 - 2:35] preprocessing directive=
// CHECK: Punctuation: "," [2:35 - 2:36] preprocessing directive=
// CHECK: Identifier: "warble" [2:36 - 2:42] preprocessing directive=
// CHECK: Punctuation: ")" [2:42 - 2:43] preprocessing directive=
// CHECK: Punctuation: "#" [3:1 - 3:2] preprocessing directive=
// CHECK: Identifier: "define" [3:2 - 3:8] preprocessing directive=
// CHECK: Identifier: "BAR" [3:9 - 3:12] macro definition=BAR
// CHECK: Identifier: "baz" [3:13 - 3:16] preprocessing directive=
// CHECK: Punctuation: "#" [4:1 - 4:2] preprocessing directive=
// CHECK: Identifier: "define" [4:2 - 4:8] preprocessing directive=
// CHECK: Identifier: "WIBBLE" [4:9 - 4:15] macro definition=WIBBLE
// CHECK: Punctuation: "(" [4:15 - 4:16] preprocessing directive=
// CHECK: Identifier: "X" [4:16 - 4:17] preprocessing directive=
// CHECK: Punctuation: "," [4:17 - 4:18] preprocessing directive=
// CHECK: Identifier: "Y" [4:19 - 4:20] preprocessing directive=
// CHECK: Punctuation: ")" [4:20 - 4:21] preprocessing directive=
// CHECK: Identifier: "X" [4:22 - 4:23] preprocessing directive=
// CHECK: Punctuation: "##" [4:23 - 4:25] preprocessing directive=
// CHECK: Identifier: "Y" [4:25 - 4:26] preprocessing directive=
// CHECK: Identifier: "NOTHING" [5:1 - 5:8] macro instantiation=NOTHING:1:9
// CHECK: Punctuation: "(" [5:8 - 5:9]
// CHECK: Identifier: "more" [5:9 - 5:13]
// CHECK: Punctuation: "," [5:13 - 5:14]
// CHECK: Identifier: "junk" [5:14 - 5:18]
// CHECK: Punctuation: ")" [5:18 - 5:19]
// CHECK: Keyword: "float" [5:20 - 5:25]
// CHECK: Identifier: "WIBBLE" [5:26 - 5:32] macro instantiation=WIBBLE:4:9
// CHECK: Punctuation: "(" [5:32 - 5:33]
// CHECK: Keyword: "int" [5:33 - 5:36]
// CHECK: Punctuation: "," [5:36 - 5:37]
// CHECK: Keyword: "float" [5:38 - 5:43]
// CHECK: Punctuation: ")" [5:43 - 5:44]
// CHECK: Punctuation: ";" [5:44 - 5:45]
// CHECK: Keyword: "int" [6:1 - 6:4]
// CHECK: Identifier: "BAR" [6:5 - 6:8] macro instantiation=BAR:3:9
// CHECK: Identifier: "STILL_NOTHING" [6:9 - 6:22] macro instantiation=STILL_NOTHING:2:9
// CHECK: Punctuation: ";" [6:22 - 6:23]
// CHECK: Punctuation: "#" [7:1 - 7:2] preprocessing directive=
// CHECK: Identifier: "include" [7:2 - 7:9] preprocessing directive=
// CHECK: Literal: ""foo.h"" [7:10 - 7:17] preprocessing directive=
// CHECK: Punctuation: "#" [8:1 - 8:2] preprocessing directive=
// CHECK: Identifier: "undef" [8:2 - 8:7] preprocessing directive=
// CHECK: Identifier: "BAR" [8:8 - 8:11] preprocessing directive=
