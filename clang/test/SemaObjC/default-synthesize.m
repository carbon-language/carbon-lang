// RUN: %clang_cc1 -fsyntax-only -fobjc-nonfragile-abi2 -verify %s

@interface NSString @end

@interface NSObject @end

@interface SynthItAll
@property int howMany;
@property (retain) NSString* what;
@end

@implementation SynthItAll
//@synthesize howMany, what;
@end


@interface SynthSetter : NSObject
@property (nonatomic) int howMany;  // REM: nonatomic to avoid warnings about only implementing one of the pair
@property (nonatomic, retain) NSString* what;
@end

@implementation SynthSetter
//@synthesize howMany, what;

- (int) howMany {
    return self.howMany;
}
// - (void) setHowMany: (int) value

- (NSString*) what {
    return self.what;
}
// - (void) setWhat: (NSString*) value    
@end


@interface SynthGetter : NSObject
@property (nonatomic) int howMany;  // REM: nonatomic to avoid warnings about only implementing one of the pair
@property (nonatomic, retain) NSString* what;
@end

@implementation SynthGetter
//@synthesize howMany, what;

// - (int) howMany
- (void) setHowMany: (int) value {
    self.howMany = value;
}

// - (NSString*) what
- (void) setWhat: (NSString*) value {
    if (self.what != value) {
    }
}
@end


@interface SynthNone : NSObject
@property int howMany;
@property (retain) NSString* what;
@end

@implementation SynthNone
//@synthesize howMany, what;  // REM: Redundant anyway

- (int) howMany {
    return self.howMany;
}
- (void) setHowMany: (int) value {
    self.howMany = value;
}

- (NSString*) what {
    return self.what;
}
- (void) setWhat: (NSString*) value {
    if (self.what != value) {
    }
}
@end

