// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -g -S -emit-llvm %s -o - | FileCheck %s
// CHECK: {{.*}} [ DW_TAG_structure_type ] [Circle] [line 11,
@interface NSObject {
  struct objc_object *isa;
}
@end

@interface Shape : NSObject

@end
@interface Circle : Shape

@end
@implementation Circle

@end
