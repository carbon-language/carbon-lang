// Just check that clang accepts these.

// RUN: xcc -fsyntax-only -O1 -O2 %s &&
// RUN: xcc -fsyntax-only -O %s
