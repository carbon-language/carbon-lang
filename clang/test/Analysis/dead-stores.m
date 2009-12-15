// RUN: %clang_cc1 -analyze -analyzer-experimental-internal-checks -warn-dead-stores -verify %s

typedef signed char BOOL;
typedef unsigned int NSUInteger;
typedef struct _NSZone NSZone;
@class NSInvocation, NSMethodSignature, NSCoder, NSString, NSEnumerator;
@protocol NSObject  - (BOOL)isEqual:(id)object; @end
@protocol NSCopying  - (id)copyWithZone:(NSZone *)zone; @end
@protocol NSCoding  - (void)encodeWithCoder:(NSCoder *)aCoder; @end
@interface NSObject <NSObject> {} @end
extern id NSAllocateObject(Class aClass, NSUInteger extraBytes, NSZone *zone);
@interface NSValue : NSObject <NSCopying, NSCoding>  - (void)getValue:(void *)value; @end
typedef float CGFloat;
typedef struct _NSPoint {} NSRange;
@interface NSValue (NSValueRangeExtensions)  + (NSValue *)valueWithRange:(NSRange)range;
- (BOOL)containsObject:(id)anObject;
@end
@class NSURLAuthenticationChallenge;
@interface NSResponder : NSObject <NSCoding> {} @end
@class NSArray, NSDictionary, NSString;
@interface NSObject (NSKeyValueBindingCreation)
+ (void)exposeBinding:(NSString *)binding;
- (NSArray *)exposedBindings;
@end
extern NSString *NSAlignmentBinding;

// This test case was reported as a false positive due to a bug in the
// LiveVariables <-> DeadStores interplay.  We should not flag a warning
// here.  The test case was reported in:
//  http://lists.cs.uiuc.edu/pipermail/cfe-dev/2008-July/002157.html
void DeadStoreTest(NSObject *anObject) {
  NSArray *keys;
  if ((keys = [anObject exposedBindings]) &&   // no-warning
      ([keys containsObject:@"name"] && [keys containsObject:@"icon"])) {}
}

