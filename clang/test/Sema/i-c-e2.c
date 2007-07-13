// RUN: clang %s -fsyntax-only

// FIXME: This fails until conversions are fully explicit in the ast and i-c-e is updated to handle this.
// XFAIL: *

char array[1024/(sizeof (long))];
