// RUN: clang -cc1 -verify -fsyntax-only %s

@class NSString;

@interface A
-t1 __attribute__((noreturn));
- (NSString *)stringByAppendingFormat:(NSString *)format, ... __attribute__((format(__NSString__, 1, 2)));
-(void) m0 __attribute__((noreturn));
-(void) m1 __attribute__((unused));
@end


@interface INTF
- (int) foo1: (int)arg1 __attribute__((deprecated));

- (int) foo: (int)arg1; 

- (int) foo2: (int)arg1 __attribute__((deprecated)) __attribute__((unavailable));
@end

@implementation INTF
- (int) foo: (int)arg1  __attribute__((deprecated)){ // expected-warning {{method attribute can only be specified}}
        return 10;
}
- (int) foo1: (int)arg1 {
        return 10;
}
- (int) foo2: (int)arg1 __attribute__((deprecated)) {  // expected-warning {{method attribute can only be specified}}
        return 10;
}
@end

