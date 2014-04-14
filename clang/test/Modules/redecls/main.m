// RUN: rm -rf %t.mcp
// RUN: %clang_cc1 -fmodules -x objective-c -emit-module -fmodule-name=a %S/module.map -fmodules-cache-path=%t.mcp
// RUN: %clang_cc1 -fmodules -x objective-c -emit-module -fmodule-name=b %S/module.map -fmodules-cache-path=%t.mcp
// RUN: %clang_cc1 -fmodules %s -emit-pch -o %t1.pch -fmodules-cache-path=%t.mcp -I %S
// RUN: %clang_cc1 -fmodules %s -emit-pch -o %t2.pch -include-pch %t1.pch -fmodules-cache-path=%t.mcp -I %S
// RUN: %clang_cc1 -fmodules %s -fsyntax-only -include-pch %t2.pch -I %S -fmodules-cache-path=%t.mcp -verify

#ifndef HEADER1
#define HEADER1

@import a;

#elif !defined(HEADER2)
#define HEADER2

@class AA;
@import b;

#else

// rdar://13712705
@interface SS : AA
@end

#warning parsed this
#endif
// expected-warning@-2{{parsed this}}
