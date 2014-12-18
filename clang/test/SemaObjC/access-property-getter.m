// RUN: %clang_cc1 -verify %s

@protocol NSObject
- (oneway void)release;
@end

@protocol XCOutputStreams <NSObject>
@end


@interface XCWorkQueueCommandInvocation 
{
    id <XCOutputStreams> _outputStream;
}
@end

@interface XCWorkQueueCommandSubprocessInvocation : XCWorkQueueCommandInvocation
@end

@interface XCWorkQueueCommandLocalSubprocessInvocation : XCWorkQueueCommandSubprocessInvocation
@end

@interface XCWorkQueueCommandDistributedSubprocessInvocation : XCWorkQueueCommandSubprocessInvocation
@end

@interface XCWorkQueueCommandCacheFetchInvocation : XCWorkQueueCommandSubprocessInvocation

@end

@implementation XCWorkQueueCommandCacheFetchInvocation
- (id)harvestPredictivelyProcessedOutputFiles
{
     _outputStream.release;	// expected-warning {{property access result unused - getters should not be used for side effects}}
     return 0;
}
@end

// rdar://19137815
#pragma clang diagnostic ignored "-Wunused-getter-return-value"

@interface NSObject @end

@interface I : NSObject
@property (copy) id window;
@end

@implementation I
- (void) Meth {
  [self window];
  self.window;
}
@end

