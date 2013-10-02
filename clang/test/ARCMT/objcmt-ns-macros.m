// RUN: rm -rf %t
// RUN: %clang_cc1 -objcmt-migrate-ns-macros -mt-migrate-directory %t %s -x objective-c -fobjc-runtime-has-weak -fobjc-arc -triple x86_64-apple-darwin11
// RUN: c-arcmt-test -mt-migrate-directory %t | arcmt-test -verify-transformed-files %s.result
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsyntax-only -x objective-c -fobjc-runtime-has-weak -fobjc-arc %s.result

typedef long NSInteger;
typedef unsigned long NSUInteger;

#define NS_ENUM(_type, _name) enum _name : _type _name; enum _name : _type
#define NS_OPTIONS(_type, _name) enum _name : _type _name; enum _name : _type
#define DEPRECATED  __attribute__((deprecated))

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

enum {
    NSTIFFFileType,
    NSBMPFileType,
    NSGIFFileType,
    NSJPEGFileType,
    NSPNGFileType,
    NSJPEG2000FileType
};
typedef NSUInteger NSBitmapImageFileType;

enum {
    NSWarningAlertStyle = 0,
    NSInformationalAlertStyle = 1,
    NSCriticalAlertStyle = 2
};
typedef NSUInteger NSAlertStyle;

enum {
    D_NSTIFFFileType,
    D_NSBMPFileType,
    D_NSGIFFileType,
    D_NSJPEGFileType,
    D_NSPNGFileType,
    D_NSJPEG2000FileType
};
typedef NSUInteger D_NSBitmapImageFileType DEPRECATED;

typedef enum  {
    D_NSTickMarkBelow = 0,
    D_NSTickMarkAbove = 1
} D_NSTickMarkPosition DEPRECATED;


#define NS_ENUM_AVAILABLE(X,Y)

enum {
    NSFStrongMemory NS_ENUM_AVAILABLE(10_5, 6_0) = (0UL << 0),       
    NSFOpaqueMemory NS_ENUM_AVAILABLE(10_5, 6_0) = (2UL << 0),
    NSFMallocMemory NS_ENUM_AVAILABLE(10_5, 6_0) = (3UL << 0),       
    NSFMachVirtualMemory NS_ENUM_AVAILABLE(10_5, 6_0) = (4UL << 0),
    NSFWeakMemory NS_ENUM_AVAILABLE(10_8, 6_0) = (5UL << 0),         

    NSFObjectPersonality NS_ENUM_AVAILABLE(10_5, 6_0) = (0UL << 8),         
    NSFOpaquePersonality NS_ENUM_AVAILABLE(10_5, 6_0) = (1UL << 8),         
    NSFObjectPointerPersonality NS_ENUM_AVAILABLE(10_5, 6_0) = (2UL << 8),  
    NSFCStringPersonality NS_ENUM_AVAILABLE(10_5, 6_0) = (3UL << 8),        
    NSFStructPersonality NS_ENUM_AVAILABLE(10_5, 6_0) = (4UL << 8),         
    NSFIntegerPersonality NS_ENUM_AVAILABLE(10_5, 6_0) = (5UL << 8),        
    NSFCopyIn NS_ENUM_AVAILABLE(10_5, 6_0) = (1UL << 16),      
};

typedef NSUInteger NSFOptions;

typedef enum {
  UIP0One = 0,
  UIP0Two = 1,
  UIP0Three = 2,
  UIP0Four = 10,
  UIP0Last = 0x100
} UIP;

typedef enum {
  UIPZero = 0x0,
  UIPOne = 0x1,
  UIPTwo = 0x2,
  UIP10 = 0x10,
  UIPHundred = 0x100
} UIP_3;

typedef enum {
  UIP4Zero = 0x0,
  UIP4One = 0x1,
  UIP4Two = 0x2,
  UIP410 = 0x10,
  UIP4Hundred = 100
} UIP4_3;

typedef enum {
  UIP5Zero = 0x0,
  UIP5Two = 0x2,
  UIP510 = 0x3,
  UIP5Hundred = 0x4
} UIP5_3;

typedef enum {
  UIP6Zero = 0x0,
  UIP6One = 0x1,
  UIP6Two = 0x2,
  UIP610 = 10,
  UIP6Hundred = 0x100
} UIP6_3;

typedef enum {
  UIP7Zero = 0x0,
  UIP7One = 1,
  UIP7Two = 0x2,
  UIP710 = 10,
  UIP7Hundred = 100
} UIP7_3;


typedef enum {
  Random = 0,
  Random1 = 2,
  Random2 = 4,
  Random3 = 0x12345,
  Random4 = 0x3444444,
  Random5 = 0xbadbeef,
  Random6
} UIP8_3;
