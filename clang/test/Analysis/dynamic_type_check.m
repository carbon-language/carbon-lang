// RUN: %clang_cc1 -analyze -analyzer-checker=core,alpha.core.DynamicTypeChecker -verify %s


#define nil 0
typedef unsigned long NSUInteger;
typedef int BOOL;

@protocol NSObject
+ (id)alloc;
- (id)init;
@end

@protocol NSCopying
@end

__attribute__((objc_root_class))
@interface NSObject <NSObject>
@end

@interface NSString : NSObject <NSCopying>
@end

@interface NSMutableString : NSString
@end

@interface NSNumber : NSObject <NSCopying>
@end

void testTypeCheck(NSString* str) {
  id obj = str;
  NSNumber *num = obj; // expected-warning {{}}
  (void)num;
}
