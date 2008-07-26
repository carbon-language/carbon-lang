// RUN: clang -fsyntax-only -verify %s
typedef signed char BOOL;
typedef unsigned int NSUInteger;
typedef struct _NSZone NSZone;
@class NSInvocation, NSMethodSignature, NSCoder, NSString, NSEnumerator;

@protocol NSObject
- (BOOL)isEqual:(id)object;
@end

@protocol NSCopying
- (id)copyWithZone:(NSZone *)zone;
@end

@protocol NSMutableCopying
- (id)mutableCopyWithZone:(NSZone *)zone;
@end

@protocol NSCoding
- (void)encodeWithCoder:(NSCoder *)aCoder;
@end

@interface NSObject <NSObject> {} @end

typedef float CGFloat;
typedef struct { int a; } NSFastEnumerationState;

@protocol NSFastEnumeration 
- (NSUInteger)countByEnumeratingWithState:(NSFastEnumerationState *)state objects:(id *)stackbuf count:(NSUInteger)len;
@end

typedef unsigned short unichar;

@interface NSString : NSObject <NSCopying, NSMutableCopying, NSCoding>
- (NSUInteger)length;
@end

@interface NSSimpleCString : NSString {} @end

@interface NSConstantString : NSSimpleCString @end

extern void *_NSConstantStringClassReference;

@interface NSDictionary : NSObject <NSCopying, NSMutableCopying, NSCoding, NSFastEnumeration>
- (NSUInteger)count;
@end

@interface NSMutableDictionary : NSDictionary
- (void)removeObjectForKey:(id)aKey;
@end

@class NSArray, NSSet, NSHashTable;

@protocol PBXTrackableTask <NSObject>
- (float) taskPercentComplete;
- taskIdentifier;
@end

@interface PBXTrackableTaskManager : NSObject {
  NSMutableDictionary *_trackableTasks;
}
@end

NSString *XCExecutableDebugTaskIdentifier = @"XCExecutableDebugTaskIdentifier";

@implementation PBXTrackableTaskManager
- (id) init {}
- (void) unregisterTask:(id <PBXTrackableTask>) task {
  @synchronized (self) {
  id taskID = [task taskIdentifier];
  id task = [_trackableTasks objectForKey:taskID]; // expected-warning{{method '-objectForKey:' not found (return type defaults to 'id')}}
  }
}
@end

