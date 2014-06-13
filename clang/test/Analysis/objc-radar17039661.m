// RUN: %clang_cc1 -analyze -analyzer-checker=core,osx.cocoa.RetainCount -verify -fblocks %s

@class NSString;
typedef long NSInteger;
typedef unsigned char BOOL;
@interface NSObject {}
+(id)alloc;
-(id)init;
-(id)autorelease;
-(id)copy;
-(id)retain;
@end
@interface NSNumber : NSObject
+ (NSNumber *)numberWithInteger:(NSInteger)value __attribute__((availability(ios,introduced=2.0)));
@end

NSInteger *inoutIntegerValueGlobal;
NSInteger *inoutIntegerValueGlobal2;
NSString *traitNameGlobal;
static BOOL cond;

static inline void reallyPerformAction(void (^integerHandler)(NSInteger *inoutIntegerValue, NSString *traitName)) {
  integerHandler(inoutIntegerValueGlobal, traitNameGlobal);
  integerHandler(inoutIntegerValueGlobal2,traitNameGlobal);
}

static inline BOOL performAction(NSNumber *(^action)(NSNumber *traitValue)) {
  __attribute__((__blocks__(byref))) BOOL didFindTrait = 0;
  reallyPerformAction(^(NSInteger *inoutIntegerValue,NSString *traitName) {

    if (cond) {

      NSNumber *traitValue = @(*inoutIntegerValue);

      NSNumber *newTraitValue = action(traitValue);

      if (traitValue != newTraitValue) {
        *inoutIntegerValue = newTraitValue ? *inoutIntegerValue : *inoutIntegerValue;
      }
      didFindTrait = 1;
    }

  });
  return didFindTrait;
}

void runTest() {
  __attribute__((__blocks__(byref))) NSNumber *builtinResult = ((NSNumber *)0);
  BOOL wasBuiltinTrait = performAction(^(NSNumber *traitValue) {
    builtinResult = [traitValue retain]; // expected-warning {{Potential leak of an object}}

    return traitValue;
  });
  if (wasBuiltinTrait) {
    [builtinResult autorelease];
    return;
  } else {
    return;
  }
}
