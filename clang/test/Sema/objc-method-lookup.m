// RUN: clang -fsyntax-only -verify %s
typedef signed char BOOL;

@protocol NSObject
- (BOOL) isEqual:(id) object;
@end

@interface NSObject < NSObject > {} @end

@class NSString, NSPort;

@interface NSPortNameServer:NSObject
+ (NSPortNameServer *) systemDefaultPortNameServer;
@end

@interface NSMachBootstrapServer:NSPortNameServer + (id) sharedInstance; @end

enum {
  NSWindowsNTOperatingSystem = 1, NSWindows95OperatingSystem, NSSolarisOperatingSystem, NSHPUXOperatingSystem, NSMACHOperatingSystem, NSSunOSOperatingSystem, NSOSF1OperatingSystem
};

@interface NSRunLoop:NSObject {} @end

@interface NSRunLoop(NSRunLoopConveniences)
- (void) run;
@end

extern NSString *const NSWillBecomeMultiThreadedNotification;

@interface SenTestTool:NSObject {}
@end

@implementation SenTestTool
+ (void) initialize {}
+(SenTestTool *) sharedInstance {}
-(int) run {}
+(int) run {
  return[[self sharedInstance] run];
}
@end
