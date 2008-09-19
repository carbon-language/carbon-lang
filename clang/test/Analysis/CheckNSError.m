// RUN: clang -checker-cfref -verify %s

typedef signed char BOOL;
typedef int NSInteger;
typedef struct _NSZone NSZone;
@class NSInvocation, NSMethodSignature, NSCoder, NSString, NSEnumerator;
@protocol NSObject  - (BOOL)isEqual:(id)object; @end
@protocol NSCopying  - (id)copyWithZone:(NSZone *)zone; @end
@protocol NSCoding  - (void)encodeWithCoder:(NSCoder *)aCoder; @end
@interface NSObject <NSObject> {} @end
@class NSDictionary;
@interface NSError : NSObject <NSCopying, NSCoding> {}
+ (id)errorWithDomain:(NSString *)domain code:(NSInteger)code userInfo:(NSDictionary *)dict;
@end
extern NSString * const NSXMLParserErrorDomain ;

@interface A
- (void)myMethodWhichMayFail:(NSError **)error;
- (BOOL)myMethodWhichMayFail2:(NSError **)error;
@end

@implementation A
- (void)myMethodWhichMayFail:(NSError **)error {   // expected-warning: {{Method accepting NSError** argument should have non-void return value to indicate that an error occurred.}}
  *error = [NSError errorWithDomain:@"domain" code:1 userInfo:0]; // expected-warning: {{Potential null dereference.}}
}

- (BOOL)myMethodWhichMayFail2:(NSError **)error {  // no-warning
  if (error) *error = [NSError errorWithDomain:@"domain" code:1 userInfo:0]; // no-warning
  return 0;
}
@end
