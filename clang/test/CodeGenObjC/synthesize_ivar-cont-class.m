// RUN: clang-cc -fobjc-nonfragile-abi -emit-llvm -o %t %s
// RUN: grep '@"OBJC_IVAR_$_XCOrganizerDeviceNodeInfo.viewController"' %t

@interface XCOrganizerNodeInfo
@property (readonly, retain) id viewController;
@end

@interface XCOrganizerDeviceNodeInfo : XCOrganizerNodeInfo
@end

@interface XCOrganizerDeviceNodeInfo()
@property (retain) id viewController;
@end

@implementation XCOrganizerDeviceNodeInfo
@synthesize viewController;
@end

