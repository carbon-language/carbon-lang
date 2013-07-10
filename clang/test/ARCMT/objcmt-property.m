// RUN: rm -rf %t
// RUN: %clang_cc1 -objcmt-migrate-property -mt-migrate-directory %t %s -x objective-c -fobjc-runtime-has-weak -fobjc-arc -fobjc-default-synthesize-properties -triple x86_64-apple-darwin11
// RUN: c-arcmt-test -mt-migrate-directory %t | arcmt-test -verify-transformed-files %s.result
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsyntax-only -x objective-c -fobjc-runtime-has-weak -fobjc-arc -fobjc-default-synthesize-properties %s.result

@class NSString;
@protocol NSCopying @end

@interface NSObject <NSCopying>
@end

@interface NSDictionary : NSObject
@end

@interface I : NSObject {
  int ivarVal;
}
- (void) setWeakProp : (NSString *__weak)Val;
- (NSString *__weak) WeakProp;

- (NSString *) StrongProp;
- (void) setStrongProp : (NSString *)Val;

- (NSString *) UnavailProp  __attribute__((unavailable));
- (void) setUnavailProp  : (NSString *)Val;

- (NSString *) UnavailProp1  __attribute__((unavailable));
- (void) setUnavailProp1  : (NSString *)Val  __attribute__((unavailable));

- (NSString *) UnavailProp2;
- (void) setUnavailProp2  : (NSString *)Val  __attribute__((unavailable));

- (NSDictionary*) undoAction;
- (void) setUndoAction: (NSDictionary*)Arg;
@end

@implementation I
@end

@class NSArray;

@interface MyClass2  {
@private
    NSArray *_names1;
    NSArray *_names2;
    NSArray *_names3;
    NSArray *_names4;
}
- (void)setNames1:(NSArray *)names;
- (void)setNames4:(__strong NSArray *)names;
- (void)setNames3:(__strong NSArray *)names;
- (void)setNames2:(NSArray *)names;
- (NSArray *) names2;
- (NSArray *)names3;
- (__strong NSArray *)names4;
- (NSArray *) names1;
@end
