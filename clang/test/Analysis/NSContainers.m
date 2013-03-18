// RUN: %clang_cc1 -analyze -analyzer-checker=core,osx.cocoa.NilArg -verify -Wno-objc-root-class %s
typedef unsigned long NSUInteger;
typedef signed char BOOL;
typedef struct _NSZone NSZone;
@class NSInvocation, NSMethodSignature, NSCoder, NSString, NSEnumerator;
@protocol NSObject
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
@protocol NSFastEnumeration
@end
@protocol NSSecureCoding <NSCoding>
@required
+ (BOOL)supportsSecureCoding;
@end
@interface NSObject <NSObject> {}
- (id)init;
+ (id)alloc;
@end

@interface NSArray : NSObject <NSCopying, NSMutableCopying, NSSecureCoding, NSFastEnumeration>

- (NSUInteger)count;
- (id)objectAtIndex:(NSUInteger)index;

@end

@interface NSArray (NSExtendedArray)
- (NSArray *)arrayByAddingObject:(id)anObject;
- (void)setObject:(id)obj atIndexedSubscript:(NSUInteger)idx __attribute__((availability(macosx,introduced=10.8)));
@end

@interface NSMutableArray : NSArray

- (void)addObject:(id)anObject;
- (void)insertObject:(id)anObject atIndex:(NSUInteger)index;
- (void)removeLastObject;
- (void)removeObjectAtIndex:(NSUInteger)index;
- (void)replaceObjectAtIndex:(NSUInteger)index withObject:(id)anObject;

@end

// NSMutableArray API
void testNilArg1() {
  NSMutableArray *marray = [[NSMutableArray alloc] init];
  [marray addObject:0]; // expected-warning {{Argument to 'NSMutableArray' method 'addObject:' cannot be nil}}
}

void testNilArg2() {
  NSMutableArray *marray = [[NSMutableArray alloc] init];
  [marray insertObject:0 atIndex:1]; // expected-warning {{Argument to 'NSMutableArray' method 'insertObject:atIndex:' cannot be nil}}
}

void testNilArg3() {
  NSMutableArray *marray = [[NSMutableArray alloc] init];
  [marray replaceObjectAtIndex:1 withObject:0]; // expected-warning {{Argument to 'NSMutableArray' method 'replaceObjectAtIndex:withObject:' cannot be nil}}
}

void testNilArg4() {
  NSMutableArray *marray = [[NSMutableArray alloc] init];
  [marray setObject:0 atIndexedSubscript:1]; // expected-warning {{Argument to 'NSMutableArray' method 'setObject:atIndexedSubscript:' cannot be nil}}
}

// NSArray API
void testNilArg5() {
  NSArray *array = [[NSArray alloc] init];
  NSArray *copyArray = [array arrayByAddingObject:0]; // expected-warning {{Argument to 'NSArray' method 'arrayByAddingObject:' cannot be nil}}
}

