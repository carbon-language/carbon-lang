// RUN: clang -triple x86_64-unknown-unknown -emit-llvm -o %t %s

@interface BASE  {
@private
    void* _reserved;
}
@end

@class PVR;

@interface PVRHandldler 
{
          PVR *_imageBrowser;
}
@end

@implementation PVRHandldler @end


@interface PVR   : BASE
@end

@implementation PVR
@end
