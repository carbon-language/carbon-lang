// RUN: %clang_cc1 -analyze -analyzer-checker=core,osx.cocoa.RetainCount -analyzer-ipa=dynamic-bifurcate -verify %s

typedef signed char BOOL;
typedef struct objc_class *Class;
typedef struct objc_object {
    Class isa;
} *id;
@protocol NSObject  - (BOOL)isEqual:(id)object; @end
@interface NSObject <NSObject> {}
+(id)alloc;
+(id)new;
- (oneway void)release;
-(id)init;
-(id)autorelease;
-(id)copy;
- (Class)class;
-(id)retain;
@end

@interface SelfStaysLive : NSObject
- (id)init;
@end

@implementation SelfStaysLive
- (id)init {
  return [super init];
}
@end

void selfStaysLive() {
    SelfStaysLive *foo = [[SelfStaysLive alloc] init]; 
    [foo release];
}

// Test that retain release checker warns on leaks and use-after-frees when 
// self init is not enabled.  
// radar://12115830
@interface ParentOfCell : NSObject
- (id)initWithInt: (int)inInt;
@end
@interface Cell : ParentOfCell{
  int x;
}
- (id)initWithInt: (int)inInt;
+ (void)testOverRelease;
+ (void)testLeak;
@property int x;
@end
@implementation Cell
@synthesize x;
- (id) initWithInt: (int)inInt {
  [super initWithInt: inInt];
  self.x = inInt; // no-warning 
  return self; // Self Init checker would produce a warning here.
}
+ (void) testOverRelease {
  Cell *sharedCell3 = [[Cell alloc] initWithInt: 3];
  [sharedCell3 release];
  [sharedCell3 release]; // expected-warning {{Reference-counted object is used after it is released}}
}
+ (void) testLeak {
  Cell *sharedCell4 = [[Cell alloc] initWithInt: 3]; // expected-warning {{leak}}
}
@end

