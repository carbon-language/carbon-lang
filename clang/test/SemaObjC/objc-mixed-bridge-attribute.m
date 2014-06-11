// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s
// expected-no-diagnostics
// rdar://17238954

typedef const struct __attribute__((objc_bridge(NSAttributedString))) __CFAttributedString *CFAttributedStringRef;

typedef struct __attribute__((objc_bridge_mutable(NSMutableAttributedString))) __CFAttributedString *CFMutableAttributedStringRef;

@interface NSAttributedString
@end

@interface NSMutableAttributedString
@end

struct __CFAttributedString {
};

void Test1(CFAttributedStringRef attrStr)
{
  id x = (NSAttributedString *) attrStr; // no warning
}

void Test2(NSAttributedString *attrStr) {
  CFAttributedStringRef cfsr = (CFAttributedStringRef) attrStr;
}

