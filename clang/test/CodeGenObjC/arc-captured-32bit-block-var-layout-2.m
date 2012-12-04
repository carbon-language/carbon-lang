// RUN: %clang_cc1 -fblocks -fobjc-arc -fobjc-runtime-has-weak -triple i386-apple-darwin -O0 -emit-llvm %s -o %t-64.s
// RUN: FileCheck --input-file=%t-64.s %s
// rdar://12773256

@class NSString;
extern void NSLog(NSString *format, ...);
extern int printf(const char *, ...);

int main() {
  NSString *strong;
  unsigned long long eightByte = 0x8001800181818181ull;
  // Test1
// CHECK: @"\01L_OBJC_CLASS_NAME_{{.*}}" = internal global [3 x i8] c"\220\00"
  void (^block1)() = ^{ printf("%#llx", eightByte); NSLog(@"%@", strong); };
// %block = alloca <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0*, i64, %0* }>, align 8
// block variable layout: BL_NON_OBJECT_WORD:3, BL_STRONG:1, BL_OPERATOR:0

  // Test2
  int i = 1;
// CHECK: @"\01L_OBJC_CLASS_NAME_{{.*}}" = internal global [3 x i8] c"#0\00"
  void (^block2)() = ^{ printf("%#llx, %d", eightByte, i); NSLog(@"%@", strong); };
// %block = alloca <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, i32, i64, i32, %0* }>, align 8
// block variable layout: BL_NON_OBJECT_WORD:4, BL_STRONG:1, BL_OPERATOR:0

  //  Test3
  char ch = 'a';
// CHECK: @"\01L_OBJC_CLASS_NAME_{{.*}}" = internal global [3 x i8] c"\220\00"
  void (^block3)() = ^{ printf("%c %#llx", ch, eightByte); NSLog(@"%@", strong); };
// %block = alloca <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0*, i64, %0*, i8 }>, align 8
// block variable layout: BL_NON_OBJECT_WORD:3, BL_STRONG:1, BL_OPERATOR:0  

  // Test4
  unsigned long fourByte = 0x8001ul;
// CHECK: @"\01L_OBJC_CLASS_NAME_{{.*}}" = internal global [3 x i8] c" 0\00"
  void (^block4)() = ^{ printf("%c %#lx", ch, fourByte); NSLog(@"%@", strong); };
// block variable layout: BL_NON_OBJECT_WORD:1, BL_STRONG:1, BL_OPERATOR:0
// %block = alloca <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, i32, %0*, i8 }>, align 4

  // Test5
// CHECK: @"\01L_OBJC_CLASS_NAME_{{.*}}" = internal global [3 x i8] c"\220\00"
  void (^block5)() = ^{ NSLog(@"%@", strong); printf("%c %#llx", ch, eightByte); };
// %block = alloca <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0*, i64, %0*, i8 }>, align 8
// block variable layout: BL_NON_OBJECT_WORD:3, BL_STRONG:1, BL_OPERATOR:0

  // Test6
// CHECK: @"\01L_OBJC_CLASS_NAME_{{.*}}" = internal global [1 x i8] zeroinitializer
  void (^block6)() = ^{ printf("%#llx", eightByte); };
// %block = alloca <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, [4 x i8], i64 }>, align 8
// block variable layout: BL_OPERATOR:0
}

/**
struct __block_literal_generic { // 32bytes (64bit) and 20 bytes (32bit).
0  void *__isa;
4  int __flags;
8  int __reserved;
12  void (*__invoke)(void *);
16  struct __block_descriptor *__descriptor;
};
*/
