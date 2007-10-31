// RUN: clang -rewrite-test %s

@implementation Intf
{
  id ivar;
  id ivar1[12];

  id **ivar3;

  id (*ivar4) (id, id);
}
@end
