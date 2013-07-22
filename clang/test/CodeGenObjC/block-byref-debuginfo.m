// RUN: %clang_cc1 -fblocks -fobjc-arc -fobjc-runtime-has-weak -g -triple x86_64-apple-darwin -O0 -emit-llvm %s -o - | FileCheck %s

// Test that the foo is aligned at an 8 byte boundary in the DWARF
// expression (256) that locates it inside of the byref descriptor:
// CHECK: metadata !"foo", i32 0, i64 {{[0-9]+}}, i64 64, i64 256, i32 0, metadata

typedef unsigned char uint8_t;
@protocol NSObject
@end
@interface NSObject <NSObject> {
}
@end
typedef void (^dispatch_block_t)(void);
typedef long dispatch_once_t;
static __inline__ __attribute__((__always_inline__)) __attribute__((__nonnull__)) __attribute__((__nothrow__))
void
_dispatch_once(dispatch_once_t *predicate, dispatch_block_t block)
{
};
typedef struct Buffer *BufferRef;
typedef struct Foo_s {
    uint8_t *data;
} Foo;
@protocol DelegateProtocol <NSObject>
@end
@interface FileReader : NSObject <DelegateProtocol>
{
 dispatch_once_t offset;
}
@end
@implementation FileReader
- (BufferRef) bar:(int *)index
{
  __attribute__((__blocks__(byref))) Foo foo;
  _dispatch_once(&offset, ^{});
  return 0;
}
@end
