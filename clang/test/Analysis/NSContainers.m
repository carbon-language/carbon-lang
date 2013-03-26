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

@interface NSDictionary : NSObject <NSCopying, NSMutableCopying, NSSecureCoding, NSFastEnumeration>

- (NSUInteger)count;
- (id)objectForKey:(id)aKey;
- (NSEnumerator *)keyEnumerator;

@end

@interface NSDictionary (NSDictionaryCreation)

+ (id)dictionary;
+ (id)dictionaryWithObject:(id)object forKey:(id <NSCopying>)key;
@end

@interface NSMutableDictionary : NSDictionary

- (void)removeObjectForKey:(id)aKey;
- (void)setObject:(id)anObject forKey:(id <NSCopying>)aKey;

@end

@interface NSMutableDictionary (NSExtendedMutableDictionary)

- (void)addEntriesFromDictionary:(NSDictionary *)otherDictionary;
- (void)removeAllObjects;
- (void)removeObjectsForKeys:(NSArray *)keyArray;
- (void)setDictionary:(NSDictionary *)otherDictionary;
- (void)setObject:(id)obj forKeyedSubscript:(id <NSCopying>)key __attribute__((availability(macosx,introduced=10.8)));

@end

@interface NSString : NSObject <NSCopying, NSMutableCopying, NSSecureCoding>

@end

// NSMutableArray API
void testNilArgNSMutableArray1() {
  NSMutableArray *marray = [[NSMutableArray alloc] init];
  [marray addObject:0]; // expected-warning {{Argument to 'NSMutableArray' method 'addObject:' cannot be nil}}
}

void testNilArgNSMutableArray2() {
  NSMutableArray *marray = [[NSMutableArray alloc] init];
  [marray insertObject:0 atIndex:1]; // expected-warning {{Argument to 'NSMutableArray' method 'insertObject:atIndex:' cannot be nil}}
}

void testNilArgNSMutableArray3() {
  NSMutableArray *marray = [[NSMutableArray alloc] init];
  [marray replaceObjectAtIndex:1 withObject:0]; // expected-warning {{Argument to 'NSMutableArray' method 'replaceObjectAtIndex:withObject:' cannot be nil}}
}

void testNilArgNSMutableArray4() {
  NSMutableArray *marray = [[NSMutableArray alloc] init];
  [marray setObject:0 atIndexedSubscript:1]; // expected-warning {{Argument to 'NSMutableArray' method 'setObject:atIndexedSubscript:' cannot be nil}}
}

void testNilArgNSMutableArray5() {
  NSMutableArray *marray = [[NSMutableArray alloc] init];
  marray[1] = 0; // expected-warning {{Array element cannot be nil}}
}

// NSArray API
void testNilArgNSArray1() {
  NSArray *array = [[NSArray alloc] init];
  NSArray *copyArray = [array arrayByAddingObject:0]; // expected-warning {{Argument to 'NSArray' method 'arrayByAddingObject:' cannot be nil}}
}

// NSMutableDictionary and NSDictionary APIs.
void testNilArgNSMutableDictionary1(NSMutableDictionary *d, NSString* key) {
  [d setObject:0 forKey:key]; // expected-warning {{Argument to 'NSMutableDictionary' method 'setObject:forKey:' cannot be nil}}
}

void testNilArgNSMutableDictionary2(NSMutableDictionary *d, NSObject *obj) {
  [d setObject:obj forKey:0]; // expected-warning {{Argument to 'NSMutableDictionary' method 'setObject:forKey:' cannot be nil}}
}

void testNilArgNSMutableDictionary3(NSMutableDictionary *d) {
  [d removeObjectForKey:0]; // expected-warning {{Argument to 'NSMutableDictionary' method 'removeObjectForKey:' cannot be nil}}
}

void testNilArgNSMutableDictionary5(NSMutableDictionary *d, NSString* key) {
  d[key] = 0; // expected-warning {{Dictionary object cannot be nil}}
}
void testNilArgNSMutableDictionary6(NSMutableDictionary *d, NSString *key) {
  if (key)
    ;
  d[key] = 0; // expected-warning {{Dictionary key cannot be nil}}
  // expected-warning@-1 {{Dictionary object cannot be nil}}
}

NSDictionary *testNilArgNSDictionary1(NSString* key) {
  return [NSDictionary dictionaryWithObject:0 forKey:key]; // expected-warning {{Argument to 'NSDictionary' method 'dictionaryWithObject:forKey:' cannot be nil}}
}
NSDictionary *testNilArgNSDictionary2(NSObject *obj) {
  return [NSDictionary dictionaryWithObject:obj forKey:0]; // expected-warning {{Argument to 'NSDictionary' method 'dictionaryWithObject:forKey:' cannot be nil}}
}

// Test inline defensive checks suppression.
void idc(id x) {
  if (x)
    ;
}
void testIDC(NSMutableDictionary *d, NSString *key) {
  idc(key);
  d[key] = @"abc"; // no-warning
}

@interface Foo
- (int *)getPtr;
- (int)getInt;
@end

void idc2(id x) {
	if (!x)
		return;
}

void testIDC2(Foo *obj) {
	idc2(obj);
	*[obj getPtr] = 1; // no-warning
}

int testIDC3(Foo *obj) {
	idc2(obj);
  return 1/[obj getInt];
}

