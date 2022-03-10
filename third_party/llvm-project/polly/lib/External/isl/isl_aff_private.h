#ifndef ISL_AFF_PRIVATE_H
#define ISL_AFF_PRIVATE_H

#include <isl/aff.h>
#include <isl/vec.h>
#include <isl/mat.h>
#include <isl/local_space.h>
#include <isl_int.h>
#include <isl_reordering.h>
#include <isl/stream.h>

/* ls represents the domain space.
 *
 * If the first two elements of "v" (the denominator and the constant term)
 * are zero, then the isl_aff represents NaN.
 */
struct isl_aff {
	int ref;

	isl_local_space	*ls;
	isl_vec		*v;
};

#undef EL
#define EL isl_aff

#include <isl_list_templ.h>

struct isl_pw_aff_piece {
	struct isl_set *set;
	struct isl_aff *aff;
};

struct isl_pw_aff {
	int ref;

	isl_space *dim;

	int n;

	size_t size;
	struct isl_pw_aff_piece p[1];
};

#undef PW
#define PW isl_pw_aff

#include <isl_pw_templ.h>

#undef EL
#define EL isl_pw_aff

#include <isl_list_templ.h>

struct isl_pw_multi_aff_piece {
	isl_set *set;
	isl_multi_aff *maff;
};

struct isl_pw_multi_aff {
	int ref;

	isl_space *dim;

	int n;

	size_t size;
	struct isl_pw_multi_aff_piece p[1];
};

#undef PW
#define PW isl_pw_multi_aff

#include <isl_pw_templ.h>

__isl_give isl_aff *isl_aff_alloc_vec(__isl_take isl_local_space *ls,
	__isl_take isl_vec *v);
__isl_give isl_aff *isl_aff_alloc(__isl_take isl_local_space *ls);

isl_size isl_aff_domain_dim(__isl_keep isl_aff *aff, enum isl_dim_type type);
isl_size isl_aff_domain_offset(__isl_keep isl_aff *aff, enum isl_dim_type type);

__isl_give isl_aff *isl_aff_reset_space_and_domain(__isl_take isl_aff *aff,
	__isl_take isl_space *space, __isl_take isl_space *domain);
__isl_give isl_aff *isl_aff_reset_domain_space(__isl_take isl_aff *aff,
	__isl_take isl_space *space);
__isl_give isl_aff *isl_aff_realign_domain(__isl_take isl_aff *aff,
	__isl_take isl_reordering *r);

__isl_give isl_aff *isl_aff_set_constant(__isl_take isl_aff *aff, isl_int v);
__isl_give isl_aff *isl_aff_set_coefficient(__isl_take isl_aff *aff,
	enum isl_dim_type type, int pos, isl_int v);
__isl_give isl_aff *isl_aff_add_constant(__isl_take isl_aff *aff, isl_int v);

__isl_give isl_aff *isl_aff_domain_factor_domain(__isl_take isl_aff *aff);

int isl_aff_plain_cmp(__isl_keep isl_aff *aff1, __isl_keep isl_aff *aff2);

__isl_give isl_aff *isl_aff_remove_unused_divs(__isl_take isl_aff *aff);
__isl_give isl_aff *isl_aff_normalize(__isl_take isl_aff *aff);

__isl_give isl_aff *isl_aff_expand_divs( __isl_take isl_aff *aff,
	__isl_take isl_mat *div, int *exp);

__isl_give isl_aff *isl_stream_read_aff(__isl_keep isl_stream *s);

__isl_give isl_pw_aff *isl_pw_aff_alloc_size(__isl_take isl_space *space,
	int n);
__isl_give isl_pw_aff *isl_pw_aff_reset_space(__isl_take isl_pw_aff *pwaff,
	__isl_take isl_space *space);
__isl_give isl_pw_aff *isl_pw_aff_reset_domain_space(
	__isl_take isl_pw_aff *pwaff, __isl_take isl_space *space);
__isl_give isl_pw_aff *isl_pw_aff_add_disjoint(
	__isl_take isl_pw_aff *pwaff1, __isl_take isl_pw_aff *pwaff2);

__isl_give isl_pw_aff *isl_pw_aff_domain_factor_domain(
	__isl_take isl_pw_aff *pa);

__isl_give isl_pw_aff *isl_pw_aff_union_opt(__isl_take isl_pw_aff *pwaff1,
	__isl_take isl_pw_aff *pwaff2, int max);

__isl_give isl_pw_aff *isl_pw_aff_set_rational(__isl_take isl_pw_aff *pwaff);
__isl_give isl_pw_aff_list *isl_pw_aff_list_set_rational(
	__isl_take isl_pw_aff_list *list);

__isl_give isl_aff *isl_aff_scale_down(__isl_take isl_aff *aff, isl_int f);
__isl_give isl_pw_aff *isl_pw_aff_scale(__isl_take isl_pw_aff *pwaff,
	isl_int f);
__isl_give isl_pw_aff *isl_pw_aff_scale_down(__isl_take isl_pw_aff *pwaff,
	isl_int f);

__isl_give isl_pw_aff *isl_stream_read_pw_aff(__isl_keep isl_stream *s);

isl_bool isl_aff_matching_params(__isl_keep isl_aff *aff,
	__isl_keep isl_space *space);
isl_stat isl_aff_check_match_domain_space(__isl_keep isl_aff *aff,
	__isl_keep isl_space *space);

#undef BASE
#define BASE aff

#include <isl_multi_templ.h>

__isl_give isl_multi_aff *isl_multi_aff_dup(__isl_keep isl_multi_aff *multi);

__isl_give isl_multi_aff *isl_multi_aff_align_divs(
	__isl_take isl_multi_aff *maff);

__isl_give isl_multi_aff *isl_multi_aff_from_basic_set_equalities(
	__isl_take isl_basic_set *bset);

__isl_give isl_multi_aff *isl_multi_aff_from_aff_mat(
	__isl_take isl_space *space, __isl_take isl_mat *mat);

#undef EL
#define EL isl_pw_multi_aff

#include <isl_list_templ.h>

__isl_give isl_pw_multi_aff *isl_pw_multi_aff_move_dims(
	__isl_take isl_pw_multi_aff *pma,
	enum isl_dim_type dst_type, unsigned dst_pos,
	enum isl_dim_type src_type, unsigned src_pos, unsigned n);

__isl_give isl_pw_multi_aff *isl_pw_multi_aff_reset_domain_space(
	__isl_take isl_pw_multi_aff *pwmaff, __isl_take isl_space *space);
__isl_give isl_pw_multi_aff *isl_pw_multi_aff_reset_space(
	__isl_take isl_pw_multi_aff *pwmaff, __isl_take isl_space *space);
__isl_give isl_pw_multi_aff *isl_pw_multi_aff_add_disjoint(
	__isl_take isl_pw_multi_aff *pma1, __isl_take isl_pw_multi_aff *pma2);

__isl_give isl_pw_multi_aff *isl_pw_multi_aff_project_out(
	__isl_take isl_pw_multi_aff *pma,
	enum isl_dim_type type, unsigned first, unsigned n);

isl_stat isl_seq_preimage(isl_int *dst, isl_int *src,
	__isl_keep isl_multi_aff *ma, int n_before, int n_after,
	int n_div_ma, int n_div_bmap,
	isl_int f, isl_int c1, isl_int c2, isl_int g, int has_denom);

__isl_give isl_aff *isl_aff_substitute_equalities(__isl_take isl_aff *aff,
	__isl_take isl_basic_set *eq);
__isl_give isl_pw_multi_aff *isl_pw_multi_aff_substitute(
	__isl_take isl_pw_multi_aff *pma, unsigned pos,
	__isl_keep isl_pw_aff *subs);

__isl_give isl_pw_multi_aff *isl_stream_read_pw_multi_aff(
	__isl_keep isl_stream *s);

__isl_give isl_union_pw_aff *isl_stream_read_union_pw_aff(
	__isl_keep isl_stream *s);

isl_stat isl_pw_aff_check_named_params(__isl_keep isl_pw_aff *pa);
isl_stat isl_multi_aff_check_named_params(__isl_keep isl_multi_aff *ma);
isl_stat isl_pw_multi_aff_check_named_params(__isl_keep isl_pw_multi_aff *pma);

isl_bool isl_pw_aff_matching_params(__isl_keep isl_pw_aff *pa,
	__isl_keep isl_space *space);
isl_stat isl_pw_aff_check_match_domain_space(__isl_keep isl_pw_aff *pa,
	__isl_keep isl_space *space);

__isl_give isl_basic_set *isl_aff_pos_basic_set(__isl_take isl_aff *aff);

#undef BASE
#define BASE pw_aff
#undef DOMBASE
#define DOMBASE set
#define EXPLICIT_DOMAIN

#include <isl_multi_templ.h>

#undef EXPLICIT_DOMAIN

__isl_give isl_map *isl_map_intersect_multi_pw_aff_explicit_domain(
	__isl_take isl_map *map, __isl_keep isl_multi_pw_aff *mpa);

#undef EL
#define EL isl_union_pw_aff

#include <isl_list_templ.h>

#undef BASE
#define BASE union_pw_aff
#undef DOMBASE
#define DOMBASE union_set
#define EXPLICIT_DOMAIN

#include <isl_multi_templ.h>

#undef EXPLICIT_DOMAIN

#undef EL
#define EL isl_union_pw_multi_aff

#include <isl_list_templ.h>

#endif
