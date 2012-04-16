// RUN: %clang_cc1 -x objective-c++ -fblocks -fms-extensions -rewrite-objc -fobjc-fragile-abi %s -o %t-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -Wno-address-of-temporary -D"SEL=void*" -D"__declspec(X)=" %t-rw.cpp
// RUN: %clang_cc1 -x objective-c++ -fblocks -fms-extensions -rewrite-objc %s -o %t-modern-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -Wno-address-of-temporary -D"SEL=void*" -D"__declspec(X)=" %t-modern-rw.cpp
// radar 7630551

void f(void (^b)(char c));

@interface a
- (void)processStuff;
@end

@implementation a
- (void)processStuff {
    f(^(char x) { });
}
@end

@interface b
- (void)processStuff;
@end

@implementation b
- (void)processStuff {
    f(^(char x) { });
}
@end
