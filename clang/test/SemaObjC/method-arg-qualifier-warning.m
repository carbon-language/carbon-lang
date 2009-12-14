// RUN: clang -cc1  -fsyntax-only -verify %s

typedef signed char BOOL;

@interface NSString
- (BOOL)isEqualToString:(NSString *)aString;
@end

static const NSString * Identifier1 =   @"Identifier1";
static NSString const * Identifier2 =   @"Identifier2";
static NSString * const Identifier3 =   @"Identifier3";

int main () {
        
    [@"Identifier1" isEqualToString:Identifier1]; // expected-warning {{sending 'NSString const *' discards qualifiers, expected 'NSString *'}}
    [@"Identifier2" isEqualToString:Identifier2]; // expected-warning {{sending 'NSString const *' discards qualifiers, expected 'NSString *'}}
    [@"Identifier3" isEqualToString:Identifier3];
    return 0;
}

