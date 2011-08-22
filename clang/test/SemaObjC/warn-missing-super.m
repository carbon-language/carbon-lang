// RUN: %clang_cc1 -fsyntax-only -verify %s

@protocol NSCopying @end

@interface NSObject <NSCopying>
- (void)dealloc;
@end

@implementation NSObject
- (void)dealloc {
  // Root class, shouldn't warn
}
@end

@interface Subclass1 : NSObject
- (void)dealloc;
@end

@implementation Subclass1
- (void)dealloc {
}  // expected-warning{{method possibly missing a [super dealloc] call}}
@end

@interface Subclass2 : NSObject
- (void)dealloc;
@end

@implementation Subclass2
- (void)dealloc {
  [super dealloc];  // Shouldn't warn
}
@end
