// RUN: rm -rf %t
// RUN: %clang_cc1 -objcmt-migrate-property -mt-migrate-directory %t %s -x objective-c -fobjc-runtime-has-weak -fobjc-arc -fobjc-default-synthesize-properties -triple x86_64-apple-darwin11
// RUN: c-arcmt-test -mt-migrate-directory %t | arcmt-test -verify-transformed-files %s.result
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsyntax-only -x objective-c -fobjc-runtime-has-weak -fobjc-arc -fobjc-default-synthesize-properties %s.result

typedef long NSInteger;
typedef unsigned long NSUInteger;

#define NS_ENUM(_type, _name) enum _name : _type _name; enum _name : _type
#define NS_OPTIONS(_type, _name) enum _name : _type _name; enum _name : _type

enum {
  blah,
  blarg
};
typedef NSInteger wibble;

enum {
    UIViewAutoresizingNone                 = 0,
    UIViewAutoresizingFlexibleLeftMargin,
    UIViewAutoresizingFlexibleWidth,
    UIViewAutoresizingFlexibleRightMargin,
    UIViewAutoresizingFlexibleTopMargin,
    UIViewAutoresizingFlexibleHeight,
    UIViewAutoresizingFlexibleBottomMargin
};
typedef NSUInteger UITableViewCellStyle;

typedef enum {
    UIViewAnimationTransitionNone,
    UIViewAnimationTransitionFlipFromLeft,
    UIViewAnimationTransitionFlipFromRight,
    UIViewAnimationTransitionCurlUp,
    UIViewAnimationTransitionCurlDown,
} UIViewAnimationTransition;

typedef enum {
    UIViewOne   = 0,
    UIViewTwo   = 1 << 0,
    UIViewThree = 1 << 1,
    UIViewFour  = 1 << 2,
    UIViewFive  = 1 << 3,
    UIViewSix   = 1 << 4,
    UIViewSeven = 1 << 5
} UITableView;

enum {
  UIOne = 0,
  UITwo = 0x1,
  UIthree = 0x8,
  UIFour = 0x100
};
typedef NSInteger UI;

typedef enum {
  UIP2One = 0,
  UIP2Two = 0x1,
  UIP2three = 0x8,
  UIP2Four = 0x100
} UIPOWER2;

enum {
  UNOne,
  UNTwo
};

// Should use NS_ENUM even though it is all power of 2.
enum {
  UIKOne = 1,
  UIKTwo = 2,
};
typedef NSInteger UIK;

typedef enum  {
    NSTickMarkBelow = 0,
    NSTickMarkAbove = 1,
    NSTickMarkLeft = NSTickMarkAbove,
    NSTickMarkRight = NSTickMarkBelow
} NSTickMarkPosition;

enum {
    UIViewNone         = 0x0,
    UIViewMargin       = 0x1,
    UIViewWidth        = 0x2,
    UIViewRightMargin  = 0x3,
    UIViewBottomMargin = 0xbadbeef
};
typedef NSInteger UITableStyle;

enum {
    UIView0         = 0,
    UIView1 = 0XBADBEEF
};
typedef NSInteger UIStyle;
