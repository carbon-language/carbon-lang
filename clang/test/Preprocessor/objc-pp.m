// RUN: clang-cc %s -fsyntax-only -verify -pedantic

#import <limits.h>  // no warning on #import in objc mode.

