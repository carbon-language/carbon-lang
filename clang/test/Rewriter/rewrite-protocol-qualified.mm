// RUN: %clang_cc1 -x objective-c++ -fblocks -fms-extensions -rewrite-objc %s -o %t-rw.cpp
// RUN: FileCheck -check-prefix LP --input-file=%t-rw.cpp %s
// radar 7589414

@protocol NSPortDelegate;
@interface NSConnection @end

@interface NSMessagePort
- (void) clone;
@end

@implementation NSMessagePort
- (void) clone {
     NSConnection <NSPortDelegate> *conn = 0;
     id <NSPortDelegate> *idc = 0;
}
@end

// CHECK-LP: NSConnection /*<NSPortDelegate>*/ *conn = 0; 

// CHECK-LP: id /*<NSPortDelegate>*/ *idc = 0; 
