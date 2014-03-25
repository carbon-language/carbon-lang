// RUN: %clang_cc1 -analyze -analyzer-checker=core,osx.cocoa.RetainCount,debug.ExprInspection -analyzer-store=region -verify -Wno-objc-root-class %s
// RUN: %clang_cc1 -analyze -analyzer-checker=core,osx.cocoa.RetainCount,debug.ExprInspection -analyzer-store=region -verify -Wno-objc-root-class -fobjc-arc %s

void clang_analyzer_eval(int);

typedef signed char BOOL;
typedef unsigned int NSUInteger;
typedef struct _NSZone NSZone;
@class NSInvocation, NSMethodSignature, NSCoder, NSString, NSEnumerator;
@protocol NSObject  - (BOOL)isEqual:(id)object; @end
@protocol NSCopying  - (id)copyWithZone:(NSZone *)zone; @end
@protocol NSCoding  - (void)encodeWithCoder:(NSCoder *)aCoder; @end
@protocol NSMutableCopying  - (id)mutableCopyWithZone:(NSZone *)zone; @end
@interface NSObject <NSObject> {}
+(id)alloc;
-(id)init;
-(id)autorelease;
-(id)copy;
-(id)retain;
-(oneway void)release;
@end
@interface NSString : NSObject <NSCopying, NSMutableCopying, NSCoding>
- (NSUInteger)length;
-(id)initWithFormat:(NSString *)f,...;
-(BOOL)isEqualToString:(NSString *)s;
+ (id)string;
@end
@interface NSNumber : NSObject {}
+(id)alloc;
-(id)initWithInteger:(int)i;
@end

// rdar://6946338

@interface Test1 : NSObject {
  NSString *text;
}
-(id)myMethod;
@property (nonatomic, assign) NSString *text;
@end


#if !__has_feature(objc_arc)

@implementation Test1

@synthesize text;

-(id)myMethod {
  Test1 *cell = [[[Test1 alloc] init] autorelease];

  NSString *string1 = [[NSString alloc] initWithFormat:@"test %f", 0.0]; // expected-warning {{Potential leak}}
  cell.text = string1;

  return cell;
}

@end


// rdar://8824416

@interface MyNumber : NSObject
{
  NSNumber* _myNumber;
}

- (id)initWithNumber:(NSNumber *)number;

@property (nonatomic, readonly) NSNumber* myNumber;
@property (nonatomic, readonly) NSNumber* newMyNumber;

@end

@implementation MyNumber
@synthesize myNumber=_myNumber;
 
- (id)initWithNumber:(NSNumber *)number
{
  self = [super init];
  
  if ( self )
  {
    _myNumber = [number copy];
  }
  
  return self;
}

- (NSNumber*)newMyNumber
{
  if ( _myNumber )
    return [_myNumber retain];
  
  return [[NSNumber alloc] initWithInteger:1];
}

- (id)valueForUndefinedKey:(NSString*)key
{
  id value = 0;
  
  if ([key isEqualToString:@"MyIvarNumberAsPropertyOverReleased"])
    value = self.myNumber; // _myNumber will be over released, since the value returned from self.myNumber is not retained.
  else if ([key isEqualToString:@"MyIvarNumberAsPropertyOk"])
    value = [self.myNumber retain]; // this line fixes the over release
  else if ([key isEqualToString:@"MyIvarNumberAsNewMyNumber"])
    value = self.newMyNumber; // this one is ok, since value is returned retained
  else 
    value = [[NSNumber alloc] initWithInteger:0];
  
  return [value autorelease]; // expected-warning {{Object autoreleased too many times}}
}

@end

NSNumber* numberFromMyNumberProperty(MyNumber* aMyNumber)
{
  NSNumber* result = aMyNumber.myNumber;
    
  return [result autorelease]; // expected-warning {{Object autoreleased too many times}}
}

#endif


// rdar://6611873

@interface Person : NSObject {
  NSString *_name;
}
@property (retain) NSString * name;
@property (assign) id friend;
@end

@implementation Person
@synthesize name = _name;
@end

#if !__has_feature(objc_arc)
void rdar6611873() {
  Person *p = [[[Person alloc] init] autorelease];
  
  p.name = [[NSString string] retain]; // expected-warning {{leak}}
  p.name = [[NSString alloc] init]; // expected-warning {{leak}}

  p.friend = [[Person alloc] init]; // expected-warning {{leak}}
}
#endif

@interface SubPerson : Person
-(NSString *)foo;
@end

@implementation SubPerson
-(NSString *)foo {
  return super.name;
}
@end


#if !__has_feature(objc_arc)
// <rdar://problem/9241180> Static analyzer doesn't detect uninitialized variable issues for property accesses
@interface RDar9241180
@property (readwrite,assign) id x;
-(id)testAnalyzer1:(int) y;
-(void)testAnalyzer2;
@end

@implementation RDar9241180
@synthesize x;
-(id)testAnalyzer1:(int)y {
    RDar9241180 *o;
    if (y && o.x) // expected-warning {{Property access on an uninitialized object pointer}}
      return o;
    return o; // expected-warning {{Undefined or garbage value returned to caller}}
}
-(void)testAnalyzer2 {
  id y;
  self.x = y;  // expected-warning {{Argument for property setter is an uninitialized value}}
}
@end
#endif


//------
// Property accessor synthesis
//------

extern void doSomethingWithPerson(Person *p);
extern void doSomethingWithName(NSString *name);

void testConsistencyRetain(Person *p) {
  clang_analyzer_eval(p.name == p.name); // expected-warning{{TRUE}}

  id origName = p.name;
  clang_analyzer_eval(p.name == origName); // expected-warning{{TRUE}}
  doSomethingWithPerson(p);
  clang_analyzer_eval(p.name == origName); // expected-warning{{UNKNOWN}}
}

void testConsistencyAssign(Person *p) {
  clang_analyzer_eval(p.friend == p.friend); // expected-warning{{TRUE}}

  id origFriend = p.friend;
  clang_analyzer_eval(p.friend == origFriend); // expected-warning{{TRUE}}
  doSomethingWithPerson(p);
  clang_analyzer_eval(p.friend == origFriend); // expected-warning{{UNKNOWN}}
}

#if !__has_feature(objc_arc)
void testOverrelease(Person *p, int coin) {
  switch (coin) {
  case 0:
    [p.name release]; // expected-warning{{not owned}}
    break;
  case 1:
    [p.friend release]; // expected-warning{{not owned}}
    break;
  case 2: {
    id friend = p.friend;
    doSomethingWithPerson(p);
    [friend release]; // expected-warning{{not owned}}
  }
  }
}

// <rdar://problem/16333368>
@implementation Person (Rdar16333368)

- (void)testDeliberateRelease:(Person *)other {
  doSomethingWithName(self.name);
  [_name release]; // no-warning
  self->_name = 0;

  doSomethingWithName(other->_name);
  [other.name release]; // expected-warning{{not owned}}
}

- (void)deliberateReleaseFalseNegative {
  // This is arguably a false negative because the result of p.friend shouldn't
  // be released, even though we are manipulating the ivar in between the two
  // actions.
  id name = self.name;
  _name = 0;
  [name release];
}

- (void)testRetainAndRelease {
  [self.name retain];
  [self.name release];
  [self.name release]; // expected-warning{{not owned}}
}

- (void)testRetainAndReleaseIVar {
  [self.name retain];
  [_name release];
  [_name release]; // expected-warning{{not owned}}
}

@end
#endif

@interface IntWrapper
@property int value;
@end

@implementation IntWrapper
@synthesize value;
@end

void testConsistencyInt(IntWrapper *w) {
  clang_analyzer_eval(w.value == w.value); // expected-warning{{TRUE}}

  int origValue = w.value;
  if (origValue != 42)
    return;

  clang_analyzer_eval(w.value == 42); // expected-warning{{TRUE}}
}

void testConsistencyInt2(IntWrapper *w) {
  if (w.value != 42)
    return;

  clang_analyzer_eval(w.value == 42); // expected-warning{{TRUE}}
}


@interface IntWrapperAuto
@property int value;
@end

@implementation IntWrapperAuto
@end

void testConsistencyIntAuto(IntWrapperAuto *w) {
  clang_analyzer_eval(w.value == w.value); // expected-warning{{TRUE}}

  int origValue = w.value;
  if (origValue != 42)
    return;

  clang_analyzer_eval(w.value == 42); // expected-warning{{TRUE}}
}

void testConsistencyIntAuto2(IntWrapperAuto *w) {
  if (w.value != 42)
    return;

  clang_analyzer_eval(w.value == 42); // expected-warning{{TRUE}}
}


typedef struct {
  int value;
} IntWrapperStruct;

@interface StructWrapper
@property IntWrapperStruct inner;
@end

@implementation StructWrapper
@synthesize inner;
@end

void testConsistencyStruct(StructWrapper *w) {
  clang_analyzer_eval(w.inner.value == w.inner.value); // expected-warning{{TRUE}}

  int origValue = w.inner.value;
  if (origValue != 42)
    return;

  clang_analyzer_eval(w.inner.value == 42); // expected-warning{{TRUE}}
}


@interface OpaqueIntWrapper
@property int value;
@end

// For now, don't assume a property is implemented using an ivar unless we can
// actually see that it is.
void testOpaqueConsistency(OpaqueIntWrapper *w) {
  clang_analyzer_eval(w.value == w.value); // expected-warning{{UNKNOWN}}
}

