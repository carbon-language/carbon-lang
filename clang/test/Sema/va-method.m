// RUN: clang -rewrite-test %s | clang

#include <stdarg.h>

@interface NSObject @end
@interface XX : NSObject @end

@implementation XX
- (void)encodeValuesOfObjCTypes:(const char *)types, ... {
   va_list ap;
   va_start(ap, types); // expected-warning {{second parameter of 'va_start' not last named argument}}
   while (*types) ;
   va_end(ap);
}

@end

