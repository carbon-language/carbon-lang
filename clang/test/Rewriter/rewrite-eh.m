// RUN: %clang_cc1 -rewrite-objc -o - %s
// rdar://7522880

@interface NSException
@end

@interface Foo
@end

@implementation Foo
- (void)bar {
    @try {
    } @catch (NSException *e) {
    }
    @catch (Foo *f) {
    }
    @catch (...) {
    }
}
@end
