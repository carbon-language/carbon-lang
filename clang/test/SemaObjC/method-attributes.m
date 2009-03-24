// RUN: clang-cc -verify -fsyntax-only %s

@class NSString;

@interface A
-t1 __attribute__((noreturn));
- (NSString *)stringByAppendingFormat:(NSString *)format, ... __attribute__((format(__NSString__, 1, 2)));
-(void) m0 __attribute__((noreturn));
-(void) m1 __attribute__((unused));
@end
