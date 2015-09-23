// RUN: %clang_cc1 -analyze -fblocks -analyzer-store=region -analyzer-checker=alpha.osx.cocoa.NonLocalizedStringChecker -analyzer-checker=alpha.osx.cocoa.EmptyLocalizationContextChecker -verify  %s

// The larger set of tests in located in localization.m. These are tests
// specific for non-aggressive reporting.

// These declarations were reduced using Delta-Debugging from Foundation.h
// on Mac OS X.

#define nil ((id)0)
#define NSLocalizedString(key, comment)                                        \
  [[NSBundle mainBundle] localizedStringForKey:(key) value:@"" table:nil]
#define NSLocalizedStringFromTable(key, tbl, comment)                          \
  [[NSBundle mainBundle] localizedStringForKey:(key) value:@"" table:(tbl)]
#define NSLocalizedStringFromTableInBundle(key, tbl, bundle, comment)          \
  [bundle localizedStringForKey:(key) value:@"" table:(tbl)]
#define NSLocalizedStringWithDefaultValue(key, tbl, bundle, val, comment)      \
  [bundle localizedStringForKey:(key) value:(val) table:(tbl)]
@interface NSObject
+ (id)alloc;
- (id)init;
@end
@interface NSString : NSObject
@end
@interface NSBundle : NSObject
+ (NSBundle *)mainBundle;
- (NSString *)localizedStringForKey:(NSString *)key
                              value:(NSString *)value
                              table:(NSString *)tableName;
@end
@interface UILabel : NSObject
@property(nullable, nonatomic, copy) NSString *text;
@end
@interface TestObject : NSObject
@property(strong) NSString *text;
@end

@interface LocalizationTestSuite : NSObject
int random();
@end

// Test cases begin here
@implementation LocalizationTestSuite

// An object passed in as an parameter's string member
// should not be considered unlocalized
- (void)testObjectAsArgument:(TestObject *)argumentObject {
  UILabel *testLabel = [[UILabel alloc] init];

  [testLabel setText:[argumentObject text]]; // no-warning
  [testLabel setText:argumentObject.text];   // no-warning
}

- (void)testLocalizationErrorDetectedOnPathway {
  UILabel *testLabel = [[UILabel alloc] init];
  NSString *bar = NSLocalizedString(@"Hello", @"Comment");

  if (random()) {
    bar = @"Unlocalized string";
  }

  [testLabel setText:bar]; // expected-warning {{String should be localized}}
}

- (void)testOneCharacterStringsDoNotGiveAWarning {
  UILabel *testLabel = [[UILabel alloc] init];
  NSString *bar = NSLocalizedString(@"Hello", @"Comment");

  if (random()) {
    bar = @"-";
  }

  [testLabel setText:bar]; // no-warning
}

- (void)testOneCharacterUTFStringsDoNotGiveAWarning {
  UILabel *testLabel = [[UILabel alloc] init];
  NSString *bar = NSLocalizedString(@"Hello", @"Comment");

  if (random()) {
    bar = @"\u2014";
  }

  [testLabel setText:bar]; // no-warning
}

@end
