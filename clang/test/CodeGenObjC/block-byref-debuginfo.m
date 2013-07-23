// RUN: %clang_cc1 -fblocks -fobjc-arc -fobjc-runtime-has-weak -g -triple x86_64-apple-darwin -O0 -emit-llvm %s -o - | FileCheck %s

// Test that the foo is aligned at an 8 byte boundary in the DWARF
// expression (256) that locates it inside of the byref descriptor:
// CHECK: metadata !"foo", i32 0, i64 {{[0-9]+}}, i64 64, i64 256, i32 0, metadata

@interface NSObject {
}
@end
typedef struct Buffer *BufferRef;
typedef struct Foo_s {
    unsigned char *data;
} Foo;
@interface FileReader : NSObject {
}
@end
@implementation FileReader
- (BufferRef) bar:(int *)index
{
  __attribute__((__blocks__(byref))) Foo foo;
  return 0;
}
@end
