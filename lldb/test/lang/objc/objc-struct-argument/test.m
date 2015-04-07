#import <Foundation/Foundation.h>

struct things_to_sum {
    int a;
    int b;
    int c;
};

@interface ThingSummer : NSObject {
};
-(int)sumThings:(struct things_to_sum)tts;
@end

@implementation ThingSummer
-(int)sumThings:(struct things_to_sum)tts
{
  return tts.a + tts.b + tts.c;
}
@end

int main()
{
  @autoreleasepool
  {
    ThingSummer *summer = [ThingSummer alloc];
    struct things_to_sum tts = { 2, 3, 4 };
    int ret = [summer sumThings:tts];

    NSRect rect = {{0, 0}, {10, 20}};    

    // Set breakpoint here.
    return rect.origin.x;
  }
}
