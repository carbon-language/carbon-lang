// RUN: %clang_cc1 -fsyntax-only -verify %s
typedef _Bool BOOL;

@interface NSNumber @end

@interface NSNumber (NSNumberCreation)
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
+ (int)numberWithBool:(BOOL)value; // expected-note{{method returns unexpected type 'int' (should be an object type)}}
@end

@interface NSString
+ (char)stringWithUTF8String:(const char *)value; // expected-note{{method returns unexpected type 'char' (should be an object type)}}
@end

@interface NSArray
@end

@interface NSArray (NSArrayCreation)
+ (id)arrayWithObjects:(const int [])objects // expected-note{{first parameter has unexpected type 'const int *' (should be 'const id *')}}
                 count:(unsigned long)cnt;
@end

@interface NSDictionary
+ (id)dictionaryWithObjects:(const id [])objects
                    forKeys:(const int [])keys // expected-note{{second parameter has unexpected type 'const int *' (should be 'const id *')}}
                      count:(unsigned long)cnt;
@end

void test_sig() {
  (void)@__objc_yes; // expected-error{{literal construction method 'numberWithBool:' has incompatible signature}}
  id array = @[ @17 ]; // expected-error{{literal construction method 'arrayWithObjects:count:' has incompatible signature}}
  id dict = @{ @"hello" : @17 }; // expected-error{{literal construction method 'dictionaryWithObjects:forKeys:count:' has incompatible signature}}
  id str = @("hello"); // expected-error{{literal construction method 'stringWithUTF8String:' has incompatible signature}}
}
