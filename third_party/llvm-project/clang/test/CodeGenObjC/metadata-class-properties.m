// RUN: %clang_cc1 -triple x86_64-apple-macosx10.11  -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-apple-macosx10.10  -emit-llvm -o - %s | FileCheck -check-prefix=CHECK-NULL %s
// RUN: %clang_cc1 -triple x86_64-apple-macosx10.11  -emit-llvm -o - -fobjc-runtime=macosx-fragile-10.5 %s | FileCheck -check-prefix=CHECK-FRAGILE %s

// CHECK: @"_OBJC_$_CLASS_PROP_LIST_Proto" = internal global {{.*}} section "__DATA, __objc_const", align 8
// CHECK: @"_OBJC_PROTOCOL_$_Proto" = {{.*}} global %struct._protocol_t { {{.*}} i32 96, i32 {{.*}} @"_OBJC_$_CLASS_PROP_LIST_Proto" {{.*}} }
// CHECK: @"_OBJC_$_CLASS_PROP_LIST_Foo_$_Category" = internal global {{.*}} section "__DATA, __objc_const", align 8
// CHECK: @"_OBJC_$_CATEGORY_Foo_$_Category" = internal global %struct._category_t { {{.*}} @"_OBJC_$_CLASS_PROP_LIST_Foo_$_Category" {{.*}}, i32 64 }, section "__DATA, __objc_const", align 8

// CHECK: @"_OBJC_$_CLASS_PROP_LIST_C" = internal global {{.*}} section "__DATA, __objc_const", align 8
// CHECK: @"_OBJC_METACLASS_RO_$_C" = internal global %struct._class_ro_t { {{.*}} @"_OBJC_$_CLASS_PROP_LIST_C" {{.*}} }, section "__DATA, __objc_const", align 8

// CHECK: !{i32 1, !"Objective-C Class Properties", i32 64}

// CHECK-NULL-NOT: @"_OBJC_$_CLASS_PROP_LIST_Proto"
// CHECK-NULL: @"_OBJC_PROTOCOL_$_Proto" = {{.*}} global %struct._protocol_t { {{.*}} %struct._prop_list_t* null, i32 96, i32 {{.*}} %struct._prop_list_t* null }
// CHECK-NULL-NOT: @"_OBJC_$_CLASS_PROP_LIST_Foo_$_Category" = internal global {{.*}} section "__DATA, __objc_const", align 8
// CHECK-NULL: @"_OBJC_$_CATEGORY_Foo_$_Category" = internal global %struct._category_t { {{.*}} %struct._prop_list_t* null, %struct._prop_list_t* null, {{.*}} }, section "__DATA, __objc_const", align 8

// CHECK-NULL-NOT: @"_OBJC_$_CLASS_PROP_LIST_C" = internal global {{.*}} section "__DATA, __objc_const", align 8
// CHECK-NULL: @"_OBJC_METACLASS_RO_$_C" = internal global %struct._class_ro_t { {{.*}} %struct._prop_list_t* null }, section "__DATA, __objc_const", align 8

// CHECK-NULL: !{i32 1, !"Objective-C Class Properties", i32 64}

// CHECK-FRAGILE: @"OBJC_$_CLASS_PROP_PROTO_LIST_Proto" = private global {{.*}} section "__OBJC,__property,regular,no_dead_strip", align 8
// CHECK-FRAGILE: @_OBJC_PROTOCOLEXT_Proto = internal global %struct._objc_protocol_extension { i32 48, {{.*}} @"OBJC_$_CLASS_PROP_PROTO_LIST_Proto" {{.*}} }, align 8
// CHECK-FRAGILE: @"_OBJC_$_CLASS_PROP_LIST_Foo_Category" = private global {{.*}} section "__OBJC,__property,regular,no_dead_strip", align 8
// CHECK-FRAGILE: @OBJC_CATEGORY_Foo_Category = private global %struct._objc_category { {{.*}}, i32 64, {{.*}} @"_OBJC_$_CLASS_PROP_LIST_Foo_Category" {{.*}} }, section "__OBJC,__category,regular,no_dead_strip", align 8

// CHECK-FRAGILE: @"_OBJC_$_CLASS_PROP_LIST_C" = private global {{.*}} section "__OBJC,__property,regular,no_dead_strip", align 8
// CHECK-FRAGILE: @OBJC_CLASSEXT_C = private global %struct._objc_class_extension { {{.*}} @"_OBJC_$_CLASS_PROP_LIST_C" {{.*}} }, section "__OBJC,__class_ext,regular,no_dead_strip", align 8

// CHECK-FRAGILE: !{i32 1, !"Objective-C Class Properties", i32 64}

@interface Foo @end

@protocol Proto
@property (class, readonly) int proto_property;
@end

@interface Foo (Category) <Proto> @end

@implementation Foo (Category)
+(int)proto_property { return 0; }
@end

__attribute__((objc_root_class))
@interface C
@property(class, readonly) int p;
@end
@implementation C
+(int)p { return 1; }
@end
