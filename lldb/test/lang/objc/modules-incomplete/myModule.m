#include "myModule.h"
#include "stdio.h"

@implementation MyClass {
};
-(void)publicMethod {
  printf("Hello public!\n");
}
-(int)privateMethod {
  printf("Hello private!\n");
  return 5;
}
@end

