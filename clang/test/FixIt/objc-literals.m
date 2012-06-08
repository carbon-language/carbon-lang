// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: cp %s %t
// RUN: not %clang_cc1 -fsyntax-only -fixit -x objective-c %t
// RUN: %clang_cc1 -fsyntax-only -pedantic -Werror -x objective-c %t
// RUN: FileCheck -input-file=%t %s

typedef unsigned char BOOL;

@interface NSObject
- (BOOL)isEqual:(id)other;
@end

@interface NSNumber : NSObject
+ (NSNumber *)numberWithChar:(char)value;
+ (NSNumber *)numberWithUnsignedChar:(unsigned char)value;
+ (NSNumber *)numberWithShort:(short)value;
+ (NSNumber *)numberWithUnsignedShort:(unsigned short)value;
+ (NSNumber *)numberWithInt:(int)value;
+ (NSNumber *)numberWithUnsignedInt:(unsigned int)value;
+ (NSNumber *)numberWithLong:(long)value;
+ (NSNumber *)numberWithUnsignedLong:(unsigned long)value;
+ (NSNumber *)numberWithLongLong:(long long)value;
+ (NSNumber *)numberWithUnsignedLongLong:(unsigned long long)value;
+ (NSNumber *)numberWithFloat:(float)value;
+ (NSNumber *)numberWithDouble:(double)value;
+ (NSNumber *)numberWithBool:(BOOL)value;
@end

@interface NSArray : NSObject
+ (id)arrayWithObjects:(const id [])objects count:(unsigned long)cnt;
@end

@interface NSDictionary : NSObject
+ (id)dictionaryWithObjects:(const id [])objects forKeys:(const id [])keys count:(unsigned long)cnt;
@end

@interface NSString : NSObject
@end

void fixes() {
  id arr = @[
    17, // expected-error{{numeric literal must be prefixed by '@' in a collection}}
    'a', // expected-error{{character literal must be prefixed by '@'}}
    "blah" // expected-error{{string literal must be prefixed by '@'}}
  ];
}

void testComparisons(id obj) {
  if (obj == @"abc") return; // expected-error{{direct comparison of a string literal is not allowed; use -isEqual: instead}}
  if (obj != @"def") return; // expected-error{{direct comparison of a string literal is not allowed; use -isEqual: instead}}
  if (@"ghi" == obj) return; // expected-error{{direct comparison of a string literal is not allowed; use -isEqual: instead}}

  // CHECK: void testComparisons(id obj) {
  // Make sure these three substitutions aren't matching the CHECK lines.
  // CHECK-NEXT: if ([obj isEqual: @"abc"]) return;
  // CHECK-NEXT: if (![obj isEqual: @"def"]) return;
  // CHECK-NEXT: if ([@"ghi" isEqual: obj]) return;

  if (@[] == obj) return; // expected-error{{direct comparison of an array literal is not allowed; use -isEqual: instead}}
  if (@{} == obj) return; // expected-error{{direct comparison of a dictionary literal is not allowed; use -isEqual: instead}}
  if (@12 == obj) return; // expected-error{{direct comparison of a numeric literal is not allowed; use -isEqual: instead}}
  if (@1.0 == obj) return; // expected-error{{direct comparison of a numeric literal is not allowed; use -isEqual: instead}}
  if (@__objc_yes == obj) return; // expected-error{{direct comparison of a numeric literal is not allowed; use -isEqual: instead}}
  if (@(1+1) == obj) return; // expected-error{{direct comparison of a boxed expression is not allowed; use -isEqual: instead}}
}

