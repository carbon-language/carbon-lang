// RUN: %clang_cc1 %s -fsyntax-only -verify
// rdar://8632525
extern id objc_msgSend(id self, SEL op, ...);
