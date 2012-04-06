// RUN: %clang_cc1  -fsyntax-only -verify -Wno-objc-root-class %s

@protocol CYCdef
- (int)name;
@end

@interface JSCdef <CYCdef> {
    int name;
}

@property (assign) int name;
@end

@implementation JSCdef
@synthesize name;
@end

