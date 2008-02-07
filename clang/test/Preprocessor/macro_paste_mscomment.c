// RUN: clang %s -fsyntax-only -fms-extensions
// This horrible stuff should preprocess into (other than whitespace):
//   int foo;
//   int bar;
//   int baz;

int foo;

#define comment /##/  dead tokens live here
comment This is stupidity

int bar;

#define nested(x) int x comment cute little dead tokens...

nested(baz)  rise of the dead tokens

;

