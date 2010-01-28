// RUN: %clang_cc1 -x objective-c++ -fblocks -fms-extensions -rewrite-objc %s -o %t-rw.cpp
// RUN: FileCheck -check-prefix LP --input-file=%t-rw.cpp %s
// radar 7583971


@interface NSURLResponse {
@public
  NSURLResponse *InnerResponse;
}
@end

@interface NSCachedURLResponseInternal 
{
    @public
    NSURLResponse *response;
}
@end

@interface NSCachedURLResponse
{
    @private
    NSCachedURLResponseInternal *_internal;
}
- (void) Meth;
@end

@implementation NSCachedURLResponse
- (void) Meth {
    _internal->response->InnerResponse = 0;
  }
@end

// CHECK-LP: ((struct NSURLResponse_IMPL *)((struct NSCachedURLResponseInternal_IMPL *)((struct NSCachedURLResponse_IMPL *)self)->_internal)->response)->InnerResponse
