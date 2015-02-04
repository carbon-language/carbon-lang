#include <isl/ctx.h>
#include <isl/printer.h>

#if defined(__cplusplus)
extern "C" {
#endif

#define ISL_xCAT(A,B) A ## B
#define ISL_CAT(A,B) ISL_xCAT(A,B)
#define ISL_KEY ISL_CAT(isl_,ISL_KEY_BASE)
#define ISL_VAL ISL_CAT(isl_,ISL_VAL_BASE)
#define ISL_xFN(TYPE,NAME) TYPE ## _ ## NAME
#define ISL_FN(TYPE,NAME) ISL_xFN(TYPE,NAME)
#define ISL_xHMAP(KEY,VAL_BASE) KEY ## _to_ ## VAL_BASE
#define ISL_yHMAP(KEY,VAL_BASE) ISL_xHMAP(KEY,VAL_BASE)
#define ISL_HMAP ISL_yHMAP(ISL_KEY,ISL_VAL_BASE)
#define ISL_HMAP_BASE ISL_yHMAP(ISL_KEY_BASE,ISL_VAL_BASE)

struct ISL_HMAP;
typedef struct ISL_HMAP	ISL_HMAP;

__isl_give ISL_HMAP *ISL_FN(ISL_HMAP,alloc)(isl_ctx *ctx, int min_size);
__isl_give ISL_HMAP *ISL_FN(ISL_HMAP,copy)(__isl_keep ISL_HMAP *hmap);
__isl_null ISL_HMAP *ISL_FN(ISL_HMAP,free)(__isl_take ISL_HMAP *hmap);

isl_ctx *ISL_FN(ISL_HMAP,get_ctx)(__isl_keep ISL_HMAP *hmap);

int ISL_FN(ISL_HMAP,has)(__isl_keep ISL_HMAP *hmap, __isl_keep ISL_KEY *key);
__isl_give ISL_VAL *ISL_FN(ISL_HMAP,get)(__isl_keep ISL_HMAP *hmap,
	__isl_take ISL_KEY *key);
__isl_give ISL_HMAP *ISL_FN(ISL_HMAP,set)(__isl_take ISL_HMAP *hmap,
	__isl_take ISL_KEY *key, __isl_take ISL_VAL *val);
__isl_give ISL_HMAP *ISL_FN(ISL_HMAP,drop)(__isl_take ISL_HMAP *hmap,
	__isl_take ISL_KEY *key);

int ISL_FN(ISL_HMAP,foreach)(__isl_keep ISL_HMAP *hmap,
	int (*fn)(__isl_take ISL_KEY *key, __isl_take ISL_VAL *val, void *user),
	void *user);

__isl_give isl_printer *ISL_FN(isl_printer_print,ISL_HMAP_BASE)(
	__isl_take isl_printer *p, __isl_keep ISL_HMAP *hmap);
void ISL_FN(ISL_HMAP,dump)(__isl_keep ISL_HMAP *hmap);

#undef ISL_xCAT
#undef ISL_CAT
#undef ISL_KEY
#undef ISL_VAL
#undef ISL_xFN
#undef ISL_FN
#undef ISL_xHMAP
#undef ISL_yHMAP
#undef ISL_HMAP

#if defined(__cplusplus)
}
#endif
