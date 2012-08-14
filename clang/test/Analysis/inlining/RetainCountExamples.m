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