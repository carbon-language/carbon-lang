// RUN: %clang_cc1 -rewrite-objc %s -o -

@implementation Intf
{
  id ivar;
  id ivar1[12];

  id **ivar3;

  id (*ivar4) (id, id);
}
@end
