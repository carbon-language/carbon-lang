// RUN: %clang_cc1 -fsyntax-only -fobjc-default-synthesize-properties -verify -Wno-objc-root-class %s

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
//@synthesize howMany, what;
@end


@interface SynthSetter : NSObject
@property (nonatomic) int howMany;  // REM: nonatomic to avoid warnings about only implementing one of the pair
@property (nonatomic, retain) NSString* what;
@end

@implementation SynthSetter
//@synthesize howMany, what;

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
@property (nonatomic) int howMany;  // REM: nonatomic to avoid warnings about only implementing one of the pair
@property (nonatomic, retain) NSString* what;
@end

@implementation SynthGetter
//@synthesize howMany, what;

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


@interface SynthNone : NSObject
@property int howMany;
@property (retain) NSString* what;
@end

@implementation SynthNone
//@synthesize howMany, what;  // REM: Redundant anyway

- (int) howMany {
    return howMany; // expected-error {{use of undeclared identifier 'howMany'}}
}
- (void) setHowMany: (int) value {
    howMany = value; // expected-error {{use of undeclared identifier 'howMany'}}
}

- (NSString*) what {
    return what; // expected-error {{use of undeclared identifier 'what'}}
}
- (void) setWhat: (NSString*) value {
    if (what != value) { // expected-error {{use of undeclared identifier 'what'}}
        [what release]; // expected-error {{use of undeclared identifier 'what'}}
        what = [value retain]; // expected-error {{use of undeclared identifier 'what'}}
    }
}
@end

// rdar://8349319
// No default synthesis if implementation has getter (readonly) and setter(readwrite) methods.
@interface DSATextSearchResult 
@property(assign,readonly) float relevance;
@property(assign,readonly) char isTitleMatch;
@end

@interface DSANodeSearchResult : DSATextSearchResult {}
@end


@implementation DSATextSearchResult 
-(char)isTitleMatch {
    return (char)0;
}

-(float)relevance {
    return 0.0;
}
@end

@implementation DSANodeSearchResult
-(id)initWithNode:(id )node relevance:(float)relevance isTitleMatch:(char)isTitleMatch {
        relevance = 0.0;        
        isTitleMatch = 'a';
	return self;
}
@end

