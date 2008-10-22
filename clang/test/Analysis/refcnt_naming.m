// RUN: clang -checker-cfref -verify %s

typedef const struct __CFString * CFStringRef;
typedef const struct __CFAllocator * CFAllocatorRef;
typedef const struct __CFURL * CFURLRef;
extern CFURLRef CFURLCreateWithString(CFAllocatorRef allocator, CFStringRef URLString, CFURLRef baseURL);
typedef signed char BOOL;
@protocol NSObject  - (BOOL)isEqual:(id)object; @end
@interface NSObject <NSObject> {} @end
@class NSArray, NSString, NSURL;

@interface MyClass : NSObject
{
}
- (NSURL *)myMethod:(NSString *)inString;
@end

@implementation MyClass

- (NSURL *)myMethod:(NSString *)inString
{
	NSURL *url = (NSURL *)CFURLCreateWithString(0, (CFStringRef)inString, 0);
	return url; // expected-warning{{leak}}
}

@end
