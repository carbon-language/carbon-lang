// RUN: clang -checker-cfref -verify %s

#include <CoreFoundation/CFString.h>
#include <Foundation/NSString.h>
#include <Foundation/NSObjCRuntime.h>
#include <Foundation/NSArray.h>

NSComparisonResult f1(NSString* s) {
  NSString *aString = nil;
  return [s compare:aString]; // expected-warning {{Argument to 'NSString' method 'compare:' cannot be nil.}}
}

NSComparisonResult f2(NSString* s) {
  NSString *aString = nil;
  return [s caseInsensitiveCompare:aString]; // expected-warning {{Argument to 'NSString' method 'caseInsensitiveCompare:' cannot be nil.}}
}

NSComparisonResult f3(NSString* s, NSStringCompareOptions op) {
  NSString *aString = nil;
  return [s compare:aString options:op]; // expected-warning {{Argument to 'NSString' method 'compare:options:' cannot be nil.}}
}

NSComparisonResult f4(NSString* s, NSStringCompareOptions op, NSRange R) {
  NSString *aString = nil;
  return [s compare:aString options:op range:R]; // expected-warning {{Argument to 'NSString' method 'compare:options:range:' cannot be nil.}}
}

NSComparisonResult f5(NSString* s, NSStringCompareOptions op, NSRange R) {
  NSString *aString = nil;
  return [s compare:aString options:op range:R locale:nil]; // expected-warning {{Argument to 'NSString' method 'compare:options:range:locale:' cannot be nil.}}
}

NSComparisonResult f6(NSString* s) {
  return [s componentsSeparatedByCharactersInSet:nil]; // expected-warning {{Argument to 'NSString' method 'componentsSeparatedByCharactersInSet:' cannot be nil.}}
}

NSString* f7(NSString* s1, NSString* s2, NSString* s3) {

  NSString* s4 = CFStringCreateWithFormat(kCFAllocatorDefault, NULL,
                                          L"%@ %@ (%@)", 
                                          s1, s2, s3);

  CFRetain(s4);
  return s4; // expected-warning{{leak}}
}

NSMutableArray* f8() {
  
  NSString* s = [[NSString alloc] init];
  NSMutableArray* a = [[NSMutableArray alloc] initWithCapacity:2];
  [a addObject:s];
  [s release]; // no-warning
  return a;
}

void f9() {
  
  NSString* s = [[NSString alloc] init];
  NSString* q = s;
  [s release];
  [q release]; // expected-warning {{used after it is released}}
}

NSString* f10() {
  
  static NSString* s = nil;
  
  if (!s) s = [[NSString alloc] init];
    
  return s; // no-warning
}

@interface C1 : NSObject {}
- (NSString*) getShared;
+ (C1*) sharedInstance;
@end
@implementation C1 : NSObject {}
- (NSString*) getShared {
  static NSString* s = nil;
  if (!s) s = [[NSString alloc] init];    
  return s; // no-warning  
}
+ (C1 *)sharedInstance {
  static C1 *sharedInstance = nil;
  if (!sharedInstance) {
    sharedInstance = [[C1 alloc] init];
  }
  return sharedInstance; // no-warning
}
@end

@interface SharedClass : NSObject
+ (id)sharedInstance;
@end
@implementation SharedClass

- (id)_init {
    if ((self = [super init])) {
        NSLog(@"Bar");
    }
    return self;
}

+ (id)sharedInstance {
    static SharedClass *_sharedInstance = nil;
    if (!_sharedInstance) {
        _sharedInstance = [[SharedClass alloc] _init];
    }
    return _sharedInstance; // no-warning
}
@end
