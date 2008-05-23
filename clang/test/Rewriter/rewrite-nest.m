// RUN: clang -rewrite-objc %s -o =-

#include <objc/objc.h>

@interface NSMapTable @end
@interface NSEnumerator @end

typedef unsigned int NSUInteger;

@interface NSConcreteMapTable : NSMapTable {
@public
    NSUInteger capacity;
}
@end

@interface NSConcreteMapTableValueEnumerator : NSEnumerator {
    NSConcreteMapTable *mapTable;
}
@end

@implementation NSConcreteMapTableValueEnumerator

- nextObject {
    while (mapTable->capacity) {
    }
    return 0;
}
@end

