// RUN: %check_clang_tidy %s objc-property-declaration %t
@class CIColor;
@class NSArray;
@class NSData;
@class NSString;
@class UIViewController;

typedef void *CGColorRef;

@interface Foo
@property(assign, nonatomic) int NotCamelCase;
// CHECK-MESSAGES: :[[@LINE-1]]:34: warning: property name 'NotCamelCase' not using lowerCamelCase style or not prefixed in a category, according to the Apple Coding Guidelines [objc-property-declaration]
// CHECK-FIXES: @property(assign, nonatomic) int notCamelCase;
@property(assign, nonatomic) int camelCase;
@property(strong, nonatomic) NSString *URLString;
@property(strong, nonatomic) NSString *bundleID;
@property(strong, nonatomic) NSData *RGBABytes;
@property(strong, nonatomic) UIViewController *notificationsVC;
@property(strong, nonatomic) NSString *URL_string;
// CHECK-MESSAGES: :[[@LINE-1]]:40: warning: property name 'URL_string' not using lowerCamelCase style or not prefixed in a category, according to the Apple Coding Guidelines [objc-property-declaration]
@property(strong, nonatomic) NSString *supportURLsCamelCase;
@property(strong, nonatomic) NSString *supportURLCamelCase;
@property(strong, nonatomic) NSString *VCsPluralToAdd;
@property(assign, nonatomic) int centerX;
@property(assign, nonatomic) int enable2GBackgroundFetch;
@property(assign, nonatomic) int shouldUseCFPreferences;
@property(assign, nonatomic) int enableGLAcceleration;
@property(assign, nonatomic) int ID;
@property(assign, nonatomic) int hasADog;
@property(nonatomic, readonly) CGColorRef CGColor;
@property(nonatomic, readonly) CIColor *CIColor;
@property(nonatomic, copy) NSArray *IDs;
@end

@interface Foo (Bar)
@property(assign, nonatomic) int abc_NotCamelCase;
// CHECK-MESSAGES: :[[@LINE-1]]:34: warning: property name 'abc_NotCamelCase' not using lowerCamelCase style or not prefixed in a category, according to the Apple Coding Guidelines [objc-property-declaration]
@property(assign, nonatomic) int abCD_camelCase;
// CHECK-MESSAGES: :[[@LINE-1]]:34: warning: property name 'abCD_camelCase' not using lowerCamelCase style or not prefixed in a category, according to the Apple Coding Guidelines [objc-property-declaration]
// CHECK-FIXES: @property(assign, nonatomic) int abcd_camelCase;
@property(assign, nonatomic) int abCD_NotCamelCase;
// CHECK-MESSAGES: :[[@LINE-1]]:34: warning: property name 'abCD_NotCamelCase' not using lowerCamelCase style or not prefixed in a category, according to the Apple Coding Guidelines [objc-property-declaration]
// CHECK-FIXES: @property(assign, nonatomic) int abcd_notCamelCase;
@property(assign, nonatomic) int wrongFormat_;
// CHECK-MESSAGES: :[[@LINE-1]]:34: warning: property name 'wrongFormat_' not using lowerCamelCase style or not prefixed in a category, according to the Apple Coding Guidelines [objc-property-declaration]
@property(strong, nonatomic) NSString *URLStr;
@property(assign, nonatomic) int abc_camelCase;
@property(strong, nonatomic) NSString *abc_URL;
@property(strong, nonatomic) NSString *opac2_sourceComponent;
@end

@interface Foo ()
@property(assign, nonatomic) int abc_inClassExtension;
// CHECK-MESSAGES: :[[@LINE-1]]:34: warning: property name 'abc_inClassExtension' not using lowerCamelCase style or not prefixed in a category, according to the Apple Coding Guidelines [objc-property-declaration]
@end