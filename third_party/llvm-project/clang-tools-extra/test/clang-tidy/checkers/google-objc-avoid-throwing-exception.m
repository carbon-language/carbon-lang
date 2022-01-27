// RUN: %check_clang_tidy %s google-objc-avoid-throwing-exception %t
@class NSString;

@interface NSException

+ (void)raise:(NSString *)name format:(NSString *)format;
+ (void)raise:(NSString *)name format:(NSString *)format arguments:(NSString *)args; // using NSString type since va_list cannot be recognized here

@end

@interface NotException

+ (void)raise:(NSString *)name format:(NSString *)format;

@end

@implementation Foo
- (void)f {
    NSString *foo = @"foo";
    @throw foo;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: pass in NSError ** instead of throwing exception to indicate Objective-C errors [google-objc-avoid-throwing-exception]
}

- (void)f2 {
    [NSException raise:@"TestException" format:@"Test"];
    // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: pass in NSError ** instead of throwing exception to indicate Objective-C errors [google-objc-avoid-throwing-exception]
    [NSException raise:@"TestException" format:@"Test %@" arguments:@"bar"];
    // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: pass in NSError ** instead of throwing exception to indicate Objective-C errors [google-objc-avoid-throwing-exception]
    [NotException raise:@"NotException" format:@"Test"];
}
@end

