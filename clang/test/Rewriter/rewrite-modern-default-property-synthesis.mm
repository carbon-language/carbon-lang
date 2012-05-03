// RUN: %clang_cc1 -x objective-c++ -fms-extensions -fobjc-default-synthesize-properties -rewrite-objc %s -o %t-rw.cpp 
// RUN: %clang_cc1 -fsyntax-only  -DSEL="void *" -Did="struct objc_object *" -Wno-attributes -Wno-address-of-temporary -D"__declspec(X)=" %t-rw.cpp
// rdar://11374235

extern "C" void *sel_registerName(const char *);

@interface NSObject 
- (void) release;
- (id) retain;
@end
@class NSString;

@interface SynthItAll : NSObject
@property int howMany;
@property (retain) NSString* what; 
@end

@implementation SynthItAll
@end


@interface SynthSetter : NSObject
@property (nonatomic) int howMany; 
@property (nonatomic, retain) NSString* what; 
@end

@implementation SynthSetter

- (int) howMany {
    return _howMany;
}
// - (void) setHowMany: (int) value

- (NSString*) what {
    return _what;
}
// - (void) setWhat: (NSString*) value    
@end


@interface SynthGetter : NSObject
@property (nonatomic) int howMany;
@property (nonatomic, retain) NSString* what;
@end

@implementation SynthGetter
// - (int) howMany
- (void) setHowMany: (int) value {
    _howMany = value;
}

// - (NSString*) what
- (void) setWhat: (NSString*) value {
    if (_what != value) {
        [_what release];
        _what = [value retain];
    }
}
@end

