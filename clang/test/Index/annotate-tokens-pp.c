#define NOTHING(X,Y)
#define STILL_NOTHING NOTHING(honk,warble)
#define BAR baz
#define WIBBLE(X, Y) X##Y
NOTHING(more,junk) float WIBBLE(int, float);
int BAR STILL_NOTHING;
#include "foo.h"
#undef BAR

#define REVERSE_MACRO(x,y) y + x
#define TWICE_MACRO(y) y + y

void test_macro_args() {
  int z = 1;
  int t = 2;
  int k = REVERSE_MACRO(t,z);
  int j = TWICE_MACRO(k + k);
  int w = j + j;
}

// RUN: c-index-test -test-annotate-tokens=%s:2:1:19:1 -I%S/Inputs %s | FileCheck %s
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
// CHECK: Punctuation: "#" [10:1 - 10:2] preprocessing directive=
// CHECK: Identifier: "define" [10:2 - 10:8] preprocessing directive=
// CHECK: Identifier: "REVERSE_MACRO" [10:9 - 10:22] macro definition=REVERSE_MACRO
// CHECK: Punctuation: "(" [10:22 - 10:23] preprocessing directive=
// CHECK: Identifier: "x" [10:23 - 10:24] preprocessing directive=
// CHECK: Punctuation: "," [10:24 - 10:25] preprocessing directive=
// CHECK: Identifier: "y" [10:25 - 10:26] preprocessing directive=
// CHECK: Punctuation: ")" [10:26 - 10:27] preprocessing directive=
// CHECK: Identifier: "y" [10:28 - 10:29] preprocessing directive=
// CHECK: Punctuation: "+" [10:30 - 10:31] preprocessing directive=
// CHECK: Identifier: "x" [10:32 - 10:33] preprocessing directive=
// CHECK: Punctuation: "#" [11:1 - 11:2] preprocessing directive=
// CHECK: Identifier: "define" [11:2 - 11:8] preprocessing directive=
// CHECK: Identifier: "TWICE_MACRO" [11:9 - 11:20] macro definition=TWICE_MACRO
// CHECK: Punctuation: "(" [11:20 - 11:21] preprocessing directive=
// CHECK: Identifier: "y" [11:21 - 11:22] preprocessing directive=
// CHECK: Punctuation: ")" [11:22 - 11:23] preprocessing directive=
// CHECK: Identifier: "y" [11:24 - 11:25] preprocessing directive=
// CHECK: Punctuation: "+" [11:26 - 11:27] preprocessing directive=
// CHECK: Identifier: "y" [11:28 - 11:29] preprocessing directive=
// CHECK: Keyword: "void" [13:1 - 13:5] FunctionDecl=test_macro_args:13:6 (Definition)
// CHECK: Identifier: "test_macro_args" [13:6 - 13:21] FunctionDecl=test_macro_args:13:6 (Definition)
// CHECK: Punctuation: "(" [13:21 - 13:22] FunctionDecl=test_macro_args:13:6 (Definition)
// CHECK: Punctuation: ")" [13:22 - 13:23] FunctionDecl=test_macro_args:13:6 (Definition)
// CHECK: Punctuation: "{" [13:24 - 13:25] UnexposedStmt=
// CHECK: Keyword: "int" [14:3 - 14:6] VarDecl=z:14:7 (Definition)
// CHECK: Identifier: "z" [14:7 - 14:8] VarDecl=z:14:7 (Definition)
// CHECK: Punctuation: "=" [14:9 - 14:10] VarDecl=z:14:7 (Definition)
// CHECK: Literal: "1" [14:11 - 14:12] UnexposedExpr=
// CHECK: Punctuation: ";" [14:12 - 14:13] UnexposedStmt=
// CHECK: Keyword: "int" [15:3 - 15:6] VarDecl=t:15:7 (Definition)
// CHECK: Identifier: "t" [15:7 - 15:8] VarDecl=t:15:7 (Definition)
// CHECK: Punctuation: "=" [15:9 - 15:10] VarDecl=t:15:7 (Definition)
// CHECK: Literal: "2" [15:11 - 15:12] UnexposedExpr=
// CHECK: Punctuation: ";" [15:12 - 15:13] UnexposedStmt=
// CHECK: Keyword: "int" [16:3 - 16:6] VarDecl=k:16:7 (Definition)
// CHECK: Identifier: "k" [16:7 - 16:8] VarDecl=k:16:7 (Definition)
// CHECK: Punctuation: "=" [16:9 - 16:10] VarDecl=k:16:7 (Definition)
// CHECK: Identifier: "REVERSE_MACRO" [16:11 - 16:24] macro instantiation=REVERSE_MACRO:10:9
// CHECK: Punctuation: "(" [16:24 - 16:25] UnexposedStmt=
// CHECK: Identifier: "t" [16:25 - 16:26] DeclRefExpr=t:15:7
// CHECK: Punctuation: "," [16:26 - 16:27] UnexposedStmt=
// CHECK: Identifier: "z" [16:27 - 16:28] DeclRefExpr=z:14:7
// CHECK: Punctuation: ")" [16:28 - 16:29] UnexposedStmt=
// CHECK: Punctuation: ";" [16:29 - 16:30] UnexposedStmt=
// CHECK: Keyword: "int" [17:3 - 17:6] VarDecl=j:17:7 (Definition)
// CHECK: Identifier: "j" [17:7 - 17:8] VarDecl=j:17:7 (Definition)
// CHECK: Punctuation: "=" [17:9 - 17:10] VarDecl=j:17:7 (Definition)
// CHECK: Identifier: "TWICE_MACRO" [17:11 - 17:22] macro instantiation=TWICE_MACRO:11:9
// CHECK: Punctuation: "(" [17:22 - 17:23] UnexposedStmt=
// CHECK: Identifier: "k" [17:23 - 17:24] DeclRefExpr=k:16:7
// CHECK: Punctuation: "+" [17:25 - 17:26] UnexposedStmt=
// CHECK: Identifier: "k" [17:27 - 17:28] DeclRefExpr=k:16:7
// CHECK: Punctuation: ")" [17:28 - 17:29] UnexposedStmt=
// CHECK: Punctuation: ";" [17:29 - 17:30] UnexposedStmt=
// CHECK: Keyword: "int" [18:3 - 18:6] VarDecl=w:18:7 (Definition)
// CHECK: Identifier: "w" [18:7 - 18:8] VarDecl=w:18:7 (Definition)
// CHECK: Punctuation: "=" [18:9 - 18:10] VarDecl=w:18:7 (Definition)
// CHECK: Identifier: "j" [18:11 - 18:12] DeclRefExpr=j:17:7
// CHECK: Punctuation: "+" [18:13 - 18:14] UnexposedExpr=
// CHECK: Identifier: "j" [18:15 - 18:16] DeclRefExpr=j:17:7
// CHECK: Punctuation: ";" [18:16 - 18:17] UnexposedStmt=
// CHECK: Punctuation: "}" [19:1 - 19:2] UnexposedStmt=

