// RUN: %clang_cc1 %s -rewrite-objc -o -
// rdar://5716356
// FIXME: Should be able to pipe into clang, but code is not
// yet correct for other reasons: rdar://5716940

@class NSNotification;
@class NSMutableArray;

void foo(NSMutableArray *notificationArray, id X) {
  for (NSNotification *notification in notificationArray)
    [X postNotification:notification];
}

