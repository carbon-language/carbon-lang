// RUN: %clang_cc1 -analyze -analyzer-checker=core -w -verify %s

@interface MyObject
- (void)takePointer:(void *)ptr __attribute__((nonnull(1)));
- (void)takePointerArg:(void *)__attribute__((nonnull)) ptr;

@end

void testNonNullMethod(int *p, MyObject *obj) {
  if (p)
    return;
  [obj takePointer:p]; // expected-warning{{nonnull}}
}


@interface Subclass : MyObject
// [[nonnull]] is an inherited attribute.
- (void)takePointer:(void *)ptr;
@end

void testSubclass(int *p, Subclass *obj) {
  if (p)
    return;
  [obj takePointer:p]; // expected-warning{{nonnull}}
}

void testSubclassArg(int *p, Subclass *obj) {
  if (p)
    return;
  [obj takePointerArg:p]; // expected-warning{{nonnull}}
}


union rdar16153464_const_cp_t {
  const struct rdar16153464_cczp *zp;
  const struct rdar16153464_cczp_prime *prime;
} __attribute__((transparent_union));

struct rdar16153464_header {
  union rdar16153464_const_cp_t cp;
  unsigned char pad[16 - sizeof(union rdar16153464_const_cp_t *)];
} __attribute__((aligned(16)));


struct rdar16153464_full_ctx {
  struct rdar16153464_header hdr;
} __attribute__((aligned(16)));


struct rdar16153464_pub_ctx {
  struct rdar16153464_header hdr;
} __attribute__((aligned(16)));


union rdar16153464_full_ctx_t {
  struct rdar16153464_full_ctx *_full;
  struct rdar16153464_header *hdr;
  struct rdar16153464_body *body;
  struct rdar16153464_public *pub;
} __attribute__((transparent_union));

union rdar16153464_pub_ctx_t {
  struct rdar16153464_pub_ctx *_pub;
  struct rdar16153464_full_ctx *_full;
  struct rdar16153464_header *hdr;
  struct rdar16153464_body *body;
  struct rdar16153464_public *pub;
  union rdar16153464_full_ctx_t innert;
} __attribute__((transparent_union));

int rdar16153464(union rdar16153464_full_ctx_t inner)
{
  extern void rdar16153464_check(union rdar16153464_pub_ctx_t outer) __attribute((nonnull(1)));
  rdar16153464_check((union rdar16153464_pub_ctx_t){ .innert = inner }); // no-warning
  rdar16153464_check(inner); // no-warning
  rdar16153464_check(0); // expected-warning{{nonnull}}
}
