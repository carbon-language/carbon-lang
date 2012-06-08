// RUN: %clang_cc1 -fsyntax-only -verify %s

typedef unsigned char BOOL;

@interface BaseObject
+ (instancetype)new;
@end

@interface NSObject : BaseObject
- (BOOL)isEqual:(id)other;
@end

@interface NSNumber : NSObject
+ (NSNumber *)numberWithInt:(int)value;
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

void testComparisonsWithFixits(id obj) {
  if (obj == @"") return; // expected-error{{direct comparison of a string literal is not allowed; use -isEqual: instead}}
  if (obj != @"") return; // expected-error{{direct comparison of a string literal is not allowed; use -isEqual: instead}}
  if (@"" == obj) return; // expected-error{{direct comparison of a string literal is not allowed; use -isEqual: instead}}
  if (@"" == @"") return; // expected-error{{direct comparison of a string literal is not allowed; use -isEqual: instead}}

  if (@[] == obj) return; // expected-error{{direct comparison of an array literal is not allowed; use -isEqual: instead}}
  if (@{} == obj) return; // expected-error{{direct comparison of a dictionary literal is not allowed; use -isEqual: instead}}
  if (@12 == obj) return; // expected-error{{direct comparison of a numeric literal is not allowed; use -isEqual: instead}}
  if (@1.0 == obj) return; // expected-error{{direct comparison of a numeric literal is not allowed; use -isEqual: instead}}
  if (@__objc_yes == obj) return; // expected-error{{direct comparison of a numeric literal is not allowed; use -isEqual: instead}}
  if (@(1+1) == obj) return; // expected-error{{direct comparison of a boxed expression is not allowed; use -isEqual: instead}}
}


@interface BadEqualReturnString : NSString
- (void)isEqual:(id)other;
@end

@interface BadEqualArgString : NSString
- (BOOL)isEqual:(int)other;
@end


void testComparisonsWithoutFixits() {
  // All of these verifications use regex form to ensure we /don't/ append
  // "use -isEqual: instead" to any of these.

  if ([BaseObject new] == @"") return; // expected-error-re{{direct comparison of a string literal is not allowed$}}

  if ([BadEqualReturnString new] == @"") return; // expected-error-re{{direct comparison of a string literal is not allowed$}}
  if ([BadEqualArgString new] == @"") return; // expected-error-re{{direct comparison of a string literal is not allowed$}}

  if (@"" < @"") return; // expected-error-re{{direct comparison of a string literal is not allowed$}}
  if (@"" > @"") return; // expected-error-re{{direct comparison of a string literal is not allowed$}}
  if (@"" <= @"") return; // expected-error-re{{direct comparison of a string literal is not allowed$}}
  if (@"" >= @"") return; // expected-error-re{{direct comparison of a string literal is not allowed$}}
}

