// RUN: clang-cc -P -E -fms-extensions %s | sed '/^#.\+/d' | tr -d '\n' |
// RUN:   grep '^int foo;int bar;int baz;$' | count 1
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

