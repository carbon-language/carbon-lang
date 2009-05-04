// RUN: clang-cc  -fsyntax-only -verify %s

@interface foo 
+ (void) cx __attribute__((weak_import));
- (void) x __attribute__((weak_import));
@end

