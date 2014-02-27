// 32-bit

// RUN: %clang_cc1 -triple i386-apple-darwin9 -fobjc-runtime=macosx-fragile-10.5 -emit-llvm -o - %s | FileCheck %s
// CHECK: @"\01L_OBJC_METACLASS_A" = private global {{.*}}, section "__OBJC,__meta_class,regular,no_dead_strip", align 4
// CHECK: @"\01L_OBJC_CLASS_A" = private global {{.*}}, section "__OBJC,__class,regular,no_dead_strip", align 4
// CHECK: @"\01L_OBJC_CATEGORY_A_Cat" = private global {{.*}}, section "__OBJC,__category,regular,no_dead_strip", align 4
// CHECK: @"\01L_OBJC_PROTOCOL_P" = private global {{.*}}, section "__OBJC,__protocol,regular,no_dead_strip", align 4
// CHECK: @"\01L_OBJC_CLASS_PROTOCOLS_C" = private global {{.*}}, section "__OBJC,__cat_cls_meth,regular,no_dead_strip", align 4
// CHECK: @"\01L_OBJC_METACLASS_C" = private global {{.*}}, section "__OBJC,__meta_class,regular,no_dead_strip", align 4
// CHECK: @"\01L_OBJC_CLASS_C" = private global {{.*}}, section "__OBJC,__class,regular,no_dead_strip", align 4
// CHECK: @"\01L_OBJC_MODULES" = private global {{.*}}, section "__OBJC,__module_info,regular,no_dead_strip", align 4

// 64-bit

// RUNX: %clang_cc1 -triple i386-apple-darwin9 -emit-llvm -o %t %s &&
// RUNX: grep '@"OBJC_CLASS_$_A" = global' %t &&
// RUNX: grep '@"OBJC_CLASS_$_C" = global' %t &&
// RUNX: grep '@"OBJC_METACLASS_$_A" = global' %t &&
// RUNX: grep '@"OBJC_METACLASS_$_C" = global' %t &&
// RUNX: grep '@"\\01L_OBJC_CLASSLIST_REFERENCES_$_0" = private global .*, section "__DATA, __objc_classrefs, regular, no_dead_strip", align 8' %t &&
// RUNX: grep '@"\\01L_OBJC_LABEL_CATEGORY_$" = private global .*, section "__DATA, __objc_catlist, regular, no_dead_strip", align 8' %t &&
// RUNX: grep '@"\\01L_OBJC_LABEL_CLASS_$" = private global .*, section "__DATA, __objc_classlist, regular, no_dead_strip", align 8' %t &&
// RUNX: grep '@"\\01l_OBJC_$_CATEGORY_A_$_Cat" = private global .*, section "__DATA, __objc_const", align 8' %t &&
// RUNX: grep '@"\\01l_OBJC_CLASS_PROTOCOLS_$_C" = private global .*, section "__DATA, __objc_const", align 8' %t &&
// RUNX: grep '@"\\01l_OBJC_CLASS_RO_$_A" = private global .*, section "__DATA, __objc_const", align 8' %t &&
// RUNX: grep '@"\\01l_OBJC_CLASS_RO_$_C" = private global .*, section "__DATA, __objc_const", align 8' %t &&
// RUNX: grep '@"\\01l_OBJC_LABEL_PROTOCOL_$_P" = weak hidden global .*, section "__DATA, __objc_protolist, coalesced, no_dead_strip", align 8' %t &&
// RUNX: grep '@"\\01l_OBJC_METACLASS_RO_$_A" = private global .*, section "__DATA, __objc_const", align 8' %t &&
// RUNX: grep '@"\\01l_OBJC_METACLASS_RO_$_C" = private global .*, section "__DATA, __objc_const", align 8' %t &&
// RUNX: grep '@"\\01l_OBJC_PROTOCOL_$_P" = weak hidden global .*, section "__DATA,__datacoal_nt,coalesced", align 8' %t &&


@interface A @end
@implementation A
@end
@implementation A (Cat)
@end
@protocol P
@end
@interface C <P>
@end
@implementation C
@end
