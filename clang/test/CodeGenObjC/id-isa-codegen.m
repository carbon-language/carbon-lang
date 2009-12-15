// RUN: %clang_cc1 -emit-llvm -o %t %s

typedef struct objc_class *Class;

typedef struct objc_object {
    Class isa;
} *id;

@interface I
+ (Class) class;
- (void)meth : (id)object : (id)src_object;
+ (unsigned char) isSubclassOfClass:(Class)aClass ;
@end

@implementation I
+ (Class) class {return 0;}
+ (unsigned char) isSubclassOfClass:(Class)aClass {return 0;}
- (void)meth : (id)object  : (id)src_object {
    [object->isa isSubclassOfClass:[I class]];

    [(*object).isa isSubclassOfClass:[I class]];

    object->isa = src_object->isa;
    (*src_object).isa = (*object).isa;
}
@end

