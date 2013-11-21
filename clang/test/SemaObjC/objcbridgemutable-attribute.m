// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s
// rdar://15498044

typedef struct __attribute__((objc_bridge_mutable(NSMutableDictionary))) __CFDictionary * CFMutableDictionaryRef; // expected-note {{declared here}}

@interface NSDictionary
@end

@interface NSMutableDictionary : NSDictionary
@end

void Test(NSMutableDictionary *md, NSDictionary *nd, CFMutableDictionaryRef mcf) {

  (void) (CFMutableDictionaryRef)md;
  (void) (CFMutableDictionaryRef)nd; // expected-warning {{'NSDictionary' cannot bridge to 'CFMutableDictionaryRef' (aka 'struct __CFDictionary *')}}
  (void) (NSDictionary *)mcf;  // expected-warning {{'CFMutableDictionaryRef' (aka 'struct __CFDictionary *') bridges to NSMutableDictionary, not 'NSDictionary'}}
  (void) (NSMutableDictionary *)mcf; // ok;
}

