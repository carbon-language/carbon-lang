// RUN: %clang_cc1 -g -emit-llvm -triple x86_64-apple-darwin -o - %s | FileCheck %s
// Check that we emit the correct method names for properties from a protocol.
// rdar://problem/13798000
@protocol NSObject
- (id)init;
@end
@interface NSObject <NSObject> {}
@end

@class Selection;

@protocol HasASelection <NSObject>
@property (nonatomic, retain) Selection* selection;
// CHECK: [ DW_TAG_subprogram ] [line [[@LINE-1]]] [local] [def] [-[MyClass selection]]
// CHECK: [ DW_TAG_subprogram ] [line [[@LINE-2]]] [local] [def] [-[MyClass setSelection:]]
// CHECK: [ DW_TAG_subprogram ] [line [[@LINE-3]]] [local] [def] [-[OtherClass selection]]
// CHECK: [ DW_TAG_subprogram ] [line [[@LINE-4]]] [local] [def] [-[OtherClass setSelection:]]
@end

@interface MyClass : NSObject <HasASelection> {
  Selection *_selection;
}
@end

@implementation MyClass
@synthesize selection = _selection;
@end

@interface OtherClass : NSObject <HasASelection> {
  Selection *_selection;
}
@end
@implementation OtherClass
@synthesize selection = _selection;
@end
