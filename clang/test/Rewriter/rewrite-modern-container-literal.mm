// RUN: %clang_cc1 -x objective-c++ -fms-extensions -rewrite-objc %s -o %t-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -Wno-address-of-temporary -D"id=void*" -D"SEL=void*" -D"__declspec(X)=" %t-rw.cpp
// rdar://10803676

void *sel_registerName(const char *);

@interface NSNumber
+ (NSNumber *)numberWithChar:(char)value;
+ (NSNumber *)numberWithInt:(int)value;
@end

@protocol NSCopying @end
typedef unsigned long NSUInteger;

@interface NSDictionary
+ (id)dictionaryWithObjects:(const id [])objects forKeys:(const id <NSCopying> [])keys count:(NSUInteger)cnt;
@end

@interface NSArray 
+ (id)arrayWithObjects:(const id [])objects count:(NSUInteger)cnt;
@end

@interface NSString<NSCopying>
@end

id NSUserName();

@interface NSDate
+ (id)date;
@end

int main() {
NSArray *array = @[ @"Hello", NSUserName(), [NSDate date], [NSNumber numberWithInt:42]];

NSDictionary *dictionary = @{ @"name" : NSUserName(), @"date" : [NSDate date], @"process" : @"processInfo"};
}

