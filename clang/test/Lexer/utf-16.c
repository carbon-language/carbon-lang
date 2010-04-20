// RUN: not %clang -xc %s.txt -fsyntax-only 2>&1 | grep 'UTF-16 (LE) byte order mark detected'
// rdar://7876588

// This test verifies that clang gives a decent error for UTF-16 source files.