// RUN: %clang_cc1 -fsyntax-only -verify %s
extern "C" {
@class Protocol;
}

// <rdar://problem/7827709>
extern "C" {
@class I;
}

@interface I
@end
