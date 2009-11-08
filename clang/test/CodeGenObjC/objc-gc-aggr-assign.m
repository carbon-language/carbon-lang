// RUN: clang-cc -fnext-runtime -fobjc-gc -emit-llvm -o %t %s
// RUN: grep objc_memmove_collectable %t | grep call | count 3

static int count;

typedef struct S {
   int ii;
} SS;

struct type_s {
   SS may_recurse;
   id id_val;
};

@interface NamedObject
{
  struct type_s type_s_ivar;
}
- (void) setSome : (struct type_s) arg;
- (struct type_s) getSome;
@property(assign) struct type_s aggre_prop;
@end

@implementation NamedObject 
- (void) setSome : (struct type_s) arg
  {
     type_s_ivar = arg;
  }
- (struct type_s) getSome 
  {
    return type_s_ivar;
  }
@synthesize aggre_prop = type_s_ivar;
@end

struct type_s some = {{1234}, (id)0};

struct type_s get(void)
{
  return some;
}

void f(const struct type_s *in, struct type_s *out) {
  *out = *in;
}

