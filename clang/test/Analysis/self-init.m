// RUN: %clang_cc1 -analyze -analyzer-check-objc-mem %s -verify

@class NSZone, NSCoder;
@protocol NSObject
@end 
@protocol NSCopying  - (id)copyWithZone:(NSZone *)zone;
@end 
@protocol NSMutableCopying  - (id)mutableCopyWithZone:(NSZone *)zone;
@end 
@protocol NSCoding  - (void)encodeWithCoder:(NSCoder *)aCoder;
@end
@interface NSObject <NSObject> {}
+ (id)allocWithZone:(NSZone *)zone;
+ (id)alloc;
- (void)dealloc;
-(id)class;
-(id)init;
-(id)release;
@end
@interface NSProxy <NSObject> {}
@end

//#import "Foundation/NSObject.h"
typedef unsigned NSUInteger;
typedef int NSInteger;

@class NSInvocation, NSMethodSignature, NSCoder, NSString, NSEnumerator;
@interface NSString : NSObject <NSCopying, NSMutableCopying, NSCoding>
- (NSUInteger)length;
+ (id)stringWithUTF8String:(const char *)nullTerminatedCString;
@end extern NSString * const NSBundleDidLoadNotification;
@interface NSAssertionHandler : NSObject {}
+ (NSAssertionHandler *)currentHandler;
- (void)handleFailureInMethod:(SEL)selector object:(id)object file:(NSString *)fileName lineNumber:(NSInteger)line description:(NSString *)format,...;
@end
extern NSString * const NSConnectionReplyMode;

@interface NSBundle : NSObject
+(id)loadNibNamed:(NSString*)s owner:(id)o;
@end

void log(void *obj);
extern void *somePtr;

@interface MyObj : NSObject {
	id myivar;
	int myint;
}
-(id)_init;
-(id)initWithSomething:(int)x;
-(void)doSomething;
@end

@interface MyProxyObj : NSProxy {}
-(id)init;
@end

@implementation MyObj

-(id)init {
  do { if (!((somePtr != 0))) { [[NSAssertionHandler currentHandler] handleFailureInMethod:_cmd object:self file:[NSString stringWithUTF8String:"init.m"] lineNumber:21 description:(@"Invalid parameter not satisfying: %s"), ("x != 0"), (0), (0), (0), (0)]; } } while(0);
  return [self initWithSomething:0];
}

-(id)init2 {
  self = [self initWithSomething:0];
  return self;
}

-(id)init3 {
	log([self class]);
	return [self initWithSomething:0];
}

-(id)init4 {
	self = [super init];
	if (self) {
		log(&self);
	}
	return self;
}

- (id)initWithSomething:(int)x {    
	if ((self = [super init]))
		myint = x;
	return self;
}

-(id)_init {
	myivar = 0;
	return self;
}

-(id)init5 {
  [NSBundle loadNibNamed:@"Window" owner:self];
  return [self initWithSomething:0];
}

-(id)init6 {
  [NSBundle loadNibNamed:@"Window" owner:myivar]; // expected-warning {{Using an ivar}}
  return [self initWithSomething:0];
}

-(id)init7 {
  if (0 != (self = [self _init]))
    myivar = 0;
  return self;
}

-(id)init8 {
    if ((self = [super init])) {
		log(&self);
		myivar = 0;
    }
    return self;
}

-(id)init9 {
	[self doSomething];
    return self; // expected-warning {{Returning 'self'}}
}

-(id)init10 {
	myivar = 0; // expected-warning {{Using an ivar}}
    return self;
}

-(id)init11 {
	return self; // expected-warning {{Returning 'self'}}
}

-(id)init12 {
	[super init];
	return self; // expected-warning {{Returning 'self'}}
}

-(id)init13 {
	if ((self == [super init])) {
	  myivar = 0; // expected-warning {{Using an ivar}}
	}
	return self; // expected-warning {{Returning 'self'}}
}

-(void)doSomething {}

@end

@implementation MyProxyObj

- (id)init { return self; }

@end
