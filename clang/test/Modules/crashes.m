// RUN: rm -rf %t.mcp
// RUN: %clang_cc1 -fmodules-cache-path=%t.mcp -fmodules -F %S/Inputs -fobjc-arc %s -verify

@import Module;

__attribute__((objc_root_class))
@interface Test
// rdar://19904648
@property (assign) id newFile; // expected-error {{property follows Cocoa naming convention for returning 'owned' objects}} \
                               // expected-note {{explicitly declare getter}}
@end

@implementation Test
@end
