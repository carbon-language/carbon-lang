// RUN: %clang_cc1 -fblocks -fobjc-arc -fobjc-runtime-has-weak -triple x86_64-apple-darwin -O0 -emit-llvm %s -o - | FileCheck %s

// rdar://12759433
@class NSString;

void Test12759433() {
 __block __unsafe_unretained NSString *uuByref = (__bridge NSString *)(void*)0x102030405060708;
 void (^block)() = ^{ uuByref = 0; };
 block();
}
// CHECK: %struct.__block_byref_uuByref = type { i8*, %struct.__block_byref_uuByref*, i32, i32, [[ZERO:%.*]]* }
int main() {
  __block __weak id wid;
  __block long XXX;
  __block id ID;
  __block struct S {
     int iS;
     double iD;
     void *pv;
     __unsafe_unretained id unsunr;
  } import;
  void (^c)() = ^{
// Inline flag for BYREF variable layout (1107296256): BLOCK_BYREF_HAS_COPY_DISPOSE BLOCK_BYREF_LAYOUT_WEAK
// CHECK: store i32 1107296256, i32* [[T0:%.*]]
  wid = 0;

// Inline flag for BYREF variable layout (536870912): BLOCK_BYREF_LAYOUT_NON_OBJECT
// CHECK: store i32 536870912, i32* [[T1:%.*]]
    XXX = 12345;

// Inline flag for BYREF variable layout (838860800): BLOCK_BYREF_HAS_COPY_DISPOSE BLOCK_BYREF_LAYOUT_STRONG
// CHECK: store i32 838860800, i32* [[T2:%.*]]
    ID = 0;

// Inline flag for BYREF variable layout (268435456): BLOCK_BYREF_LAYOUT_EXTENDED
// BYREF variable layout: BL_NON_OBJECT_WORD:3, BL_UNRETAINED:1, BL_OPERATOR:0
// CHECK: store i32 268435456, i32* [[T3:%.*]]
    import.iD = 3.14;

  };
  c();
}
