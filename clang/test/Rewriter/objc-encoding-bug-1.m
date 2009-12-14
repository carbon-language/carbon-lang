// RUN: clang -cc1 -rewrite-objc %s -o -

typedef struct NSMethodFrameArgInfo {
    struct NSMethodFrameArgInfo *subInfo;
    struct NSMethodFrameArgInfo *an;
} NSMethodFrameArgInfo;

@interface NSMethodSignature 
- (NSMethodFrameArgInfo *)_argInfo;
@end

@implementation NSMethodSignature

- (NSMethodFrameArgInfo *)_argInfo{
    return 0;
}

@end

