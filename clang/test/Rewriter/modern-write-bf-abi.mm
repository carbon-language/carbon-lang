// RUN: %clang_cc1 -x objective-c++ -Wno-return-type -fms-extensions -rewrite-objc %s -o %t-modern-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -Wno-address-of-temporary -D"Class=void*" -D"id=void*" -D"SEL=void*" -D"__declspec(X)=" %t-modern-rw.cpp
// rdar://13138459

// -Did="void*" -DSEL="void *" -DClass="void*"
@interface NSMutableArray {
  id isa;
}
@end

typedef unsigned char BOOL;
typedef unsigned long NSUInteger;

__attribute__((visibility("hidden")))
@interface __NSArrayM : NSMutableArray {
    NSUInteger _used;
    NSUInteger _doHardRetain:1;
    NSUInteger _doWeakAccess:1;
#if __LP64__
    NSUInteger _size:62;
#else
    NSUInteger _size:30;
#endif
    NSUInteger _hasObjects:1;
    NSUInteger _hasStrongReferences:1;
#if __LP64__
    NSUInteger _offset:62;
#else
    NSUInteger _offset:30;
#endif
    unsigned long _mutations;
    id *_list;
}
@end


id __CFAllocateObject2();
BOOL objc_collectingEnabled();

@implementation __NSArrayM
+ (id)__new:(const id [])objects :(NSUInteger)count :(BOOL)hasObjects :(BOOL)hasStrong :(BOOL)transferRetain {
    __NSArrayM *newArray = (__NSArrayM *)__CFAllocateObject2();
    newArray->_size = count;
    newArray->_mutations = 1;
    newArray->_doHardRetain = (hasObjects && hasStrong);
    newArray->_doWeakAccess = (objc_collectingEnabled() && !hasStrong);
    newArray->_hasObjects = hasObjects;
    newArray->_hasStrongReferences = hasStrong;
    newArray->_list = 0;
    return *newArray->_list;
}
@end
