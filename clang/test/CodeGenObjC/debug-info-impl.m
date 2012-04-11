// RUN: %clang -fverbose-asm -g -S -emit-llvm %s -o - | FileCheck %s
// CHECK: metadata !{i32 786451, metadata !6, metadata !"Circle", metadata !6, i32 11, i64 64, i64 64, i32 0, i32 512, null, metadata !7, i32 16, i32 0} ; [ DW_TAG_structure_type ]
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
