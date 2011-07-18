// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fobjc-nonfragile-abi -fsyntax-only -fobjc-arc -x objective-c %s.result
// RUN: arcmt-test --args -triple x86_64-apple-darwin10 -fobjc-nonfragile-abi -fsyntax-only -x objective-c %s > %t
// RUN: diff %t %s.result

#include "Common.h"

@interface NSData : NSObject
- (const void *)bytes;
@end

typedef struct _NSRange {
    NSUInteger location;
    NSUInteger length;
} NSRange;

@interface NSData (NSExtendedData)
- (void)getBytes:(void *)buffer length:(NSUInteger)length;
- (void)getBytes:(void *)buffer range:(NSRange)range;
@end

@interface NSData (NSDeprecated)
- (void)getBytes:(void *)buffer;
@end

void test(NSData* parmdata) {
  NSData *data, *data2 = parmdata;
  void *p = [data bytes];
  p = [data bytes];

  [data2 getBytes:&p length:sizeof(p)];
  p = [parmdata bytes];
}
