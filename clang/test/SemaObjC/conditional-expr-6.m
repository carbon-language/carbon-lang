// RUN: clang-cc -fsyntax-only -verify %s

@interface NSObject @end

@interface NSInterm : NSObject
@end

@interface NSArray : NSInterm 
@end

@interface NSSet : NSObject
@end


NSObject* test (int argc) {
    NSArray *array = ((void*)0);
    NSSet *set = ((void*)0);
    return (argc) ? set : array ;
}


NSObject* test1 (int argc) {
    NSArray *array = ((void*)0);
    NSSet *set = ((void*)0);
    return (argc) ? array : set;
}
