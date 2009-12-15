// RUN: %clang_cc1 -analyze -analyzer-experimental-internal-checks -checker-cfref -analyzer-store=basic -verify %s
// RUN: %clang_cc1 -analyze -analyzer-experimental-internal-checks -checker-cfref -analyzer-store=region -verify %s

typedef const struct __CFString * CFStringRef;
typedef const struct __CFAllocator * CFAllocatorRef;
typedef const struct __CFURL * CFURLRef;
extern CFURLRef CFURLCreateWithString(CFAllocatorRef allocator, CFStringRef URLString, CFURLRef baseURL);
typedef signed char BOOL;
@protocol NSObject  - (BOOL)isEqual:(id)object; @end
@interface NSObject <NSObject> {} @end
@class NSArray, NSString, NSURL;

@interface NamingTest : NSObject {}
-(NSObject*)photocopy;    // read as "photocopy"
-(NSObject*)photoCopy;    // read as "photo Copy"
-(NSObject*)__blebPRCopy; // read as "bleb PRCopy"
-(NSObject*)__blebPRcopy; // read as "bleb P Rcopy"
-(NSObject*)new_theprefixdoescount; // read as "new theprefixdoescount"
-(NSObject*)newestAwesomeStuff; // read as "newest awesome stuff"

@end

@interface MyClass : NSObject
{
  id myObject;
}
- (NSURL *)myMethod:(NSString *)inString;
- (NSURL *)getMethod:(NSString*)inString;
- (void)addObject:(id)X;
@end

@implementation MyClass

- (NSURL *)myMethod:(NSString *)inString
{
  NSURL *url = (NSURL *)CFURLCreateWithString(0, (CFStringRef)inString, 0); // expected-warning{{leak}}
  return url;
}

- (NSURL *)getMethod:(NSString *)inString
{
  NSURL *url = (NSURL *)CFURLCreateWithString(0, (CFStringRef)inString, 0);
  [self addObject:url];
  return url; // no-warning
}

void testNames(NamingTest* x) {
  [x photocopy]; // no-warning
  [x photoCopy]; // expected-warning{{leak}}
  [x __blebPRCopy]; // expected-warning{{leak}}
  [x __blebPRcopy]; // no-warning
  [x new_theprefixdoescount]; // expected-warning{{leak}}
  [x newestAwesomeStuff]; // no-warning
}


- (void)addObject:(id)X
{
  myObject = X;
}

@end
