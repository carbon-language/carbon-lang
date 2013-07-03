// RUN: %clang -target powerpc64-unknown-linux-gnu -maltivec -fsyntax-only %s

// PR16456: Verify that bool, true, false are treated as context-sensitive
// keywords (and therefore available for use as identifiers) when in
// Altivec mode.

typedef enum {
  false_value = 0,
  true_value = 1
} bool;

#define true true_value
#define false false_value

