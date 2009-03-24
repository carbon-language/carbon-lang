// RUN: clang-cc -fsyntax-only -verify %s

@interface A
- (void) setMoo: (int) x;	//  expected-note {{previous definition is here}}
- (int) setMoo1: (int) x;	//  expected-note {{previous definition is here}}
- (int) setOk : (int) x : (double) d;
@end

@implementation A 
-(void) setMoo: (float) x {}	//  expected-warning {{conflicting types for 'setMoo:'}}
- (char) setMoo1: (int) x {}	//  expected-warning {{conflicting types for 'setMoo1:'}}
- (int) setOk : (int) x : (double) d {}
@end



@interface C
+ (void) cMoo: (int) x;	//  expected-note {{previous definition is here}}
@end

@implementation C 
+(float) cMoo: (float) x {}	//  expected-warning {{conflicting types for 'cMoo:'}}
@end


@interface A(CAT)
- (void) setCat: (int) x;	//  expected-note {{previous definition is here}}
+ (void) cCat: (int) x;	//  expected-note {{previous definition is here}}
@end

@implementation A(CAT) 
-(float) setCat: (float) x {}	//  expected-warning {{conflicting types for 'setCat:'}}
+ (int) cCat: (int) x {}	//  expected-warning {{conflicting types for 'cCat:'}}
@end

