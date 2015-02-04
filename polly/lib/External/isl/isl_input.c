/*
 * Copyright 2008-2009 Katholieke Universiteit Leuven
 * Copyright 2010      INRIA Saclay
 * Copyright 2012-2013 Ecole Normale Superieure
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, K.U.Leuven, Departement
 * Computerwetenschappen, Celestijnenlaan 200A, B-3001 Leuven, Belgium
 * and INRIA Saclay - Ile-de-France, Parc Club Orsay Universite,
 * ZAC des vignes, 4 rue Jacques Monod, 91893 Orsay, France 
 * and Ecole Normale Superieure, 45 rue d’Ulm, 75230 Paris, France
 */

#include <ctype.h>
#include <stdio.h>
#include <string.h>
#include <isl_ctx_private.h>
#include <isl_map_private.h>
#include <isl/set.h>
#include <isl_seq.h>
#include <isl_stream_private.h>
#include <isl/obj.h>
#include "isl_polynomial_private.h"
#include <isl/union_map.h>
#include <isl_mat_private.h>
#include <isl_aff_private.h>
#include <isl_vec_private.h>
#include <isl/list.h>
#include <isl_val_private.h>

struct variable {
	char    	    	*name;
	int	     		 pos;
	struct variable		*next;
};

struct vars {
	struct isl_ctx	*ctx;
	int		 n;
	struct variable	*v;
};

static struct vars *vars_new(struct isl_ctx *ctx)
{
	struct vars *v;
	v = isl_alloc_type(ctx, struct vars);
	if (!v)
		return NULL;
	v->ctx = ctx;
	v->n = 0;
	v->v = NULL;
	return v;
}

static void variable_free(struct variable *var)
{
	while (var) {
		struct variable *next = var->next;
		free(var->name);
		free(var);
		var = next;
	}
}

static void vars_free(struct vars *v)
{
	if (!v)
		return;
	variable_free(v->v);
	free(v);
}

static void vars_drop(struct vars *v, int n)
{
	struct variable *var;

	if (!v || !v->v)
		return;

	v->n -= n;

	var = v->v;
	while (--n >= 0) {
		struct variable *next = var->next;
		free(var->name);
		free(var);
		var = next;
	}
	v->v = var;
}

static struct variable *variable_new(struct vars *v, const char *name, int len,
				int pos)
{
	struct variable *var;
	var = isl_calloc_type(v->ctx, struct variable);
	if (!var)
		goto error;
	var->name = strdup(name);
	var->name[len] = '\0';
	var->pos = pos;
	var->next = v->v;
	return var;
error:
	variable_free(v->v);
	return NULL;
}

static int vars_pos(struct vars *v, const char *s, int len)
{
	int pos;
	struct variable *q;

	if (len == -1)
		len = strlen(s);
	for (q = v->v; q; q = q->next) {
		if (strncmp(q->name, s, len) == 0 && q->name[len] == '\0')
			break;
	}
	if (q)
		pos = q->pos;
	else {
		pos = v->n;
		v->v = variable_new(v, s, len, v->n);
		if (!v->v)
			return -1;
		v->n++;
	}
	return pos;
}

static int vars_add_anon(struct vars *v)
{
	v->v = variable_new(v, "", 0, v->n);

	if (!v->v)
		return -1;
	v->n++;

	return 0;
}

/* Obtain next token, with some preprocessing.
 * In particular, evaluate expressions of the form x^y,
 * with x and y values.
 */
static struct isl_token *next_token(struct isl_stream *s)
{
	struct isl_token *tok, *tok2;

	tok = isl_stream_next_token(s);
	if (!tok || tok->type != ISL_TOKEN_VALUE)
		return tok;
	if (!isl_stream_eat_if_available(s, '^'))
		return tok;
	tok2 = isl_stream_next_token(s);
	if (!tok2 || tok2->type != ISL_TOKEN_VALUE) {
		isl_stream_error(s, tok2, "expecting constant value");
		goto error;
	}

	isl_int_pow_ui(tok->u.v, tok->u.v, isl_int_get_ui(tok2->u.v));

	isl_token_free(tok2);
	return tok;
error:
	isl_token_free(tok);
	isl_token_free(tok2);
	return NULL;
}

/* Read an isl_val from "s".
 *
 * The following token sequences are recognized
 *
 *	"infty"		->	infty
 *	"-" "infty"	->	-infty
 *	"NaN"		->	NaN
 *	n "/" d		->	n/d
 *	v		->	v
 *
 * where n, d and v are integer constants.
 */
__isl_give isl_val *isl_stream_read_val(struct isl_stream *s)
{
	struct isl_token *tok = NULL;
	struct isl_token *tok2 = NULL;
	isl_val *val;

	tok = next_token(s);
	if (!tok) {
		isl_stream_error(s, NULL, "unexpected EOF");
		goto error;
	}
	if (tok->type == ISL_TOKEN_INFTY) {
		isl_token_free(tok);
		return isl_val_infty(s->ctx);
	}
	if (tok->type == '-' &&
	    isl_stream_eat_if_available(s, ISL_TOKEN_INFTY)) {
		isl_token_free(tok);
		return isl_val_neginfty(s->ctx);
	}
	if (tok->type == ISL_TOKEN_NAN) {
		isl_token_free(tok);
		return isl_val_nan(s->ctx);
	}
	if (tok->type != ISL_TOKEN_VALUE) {
		isl_stream_error(s, tok, "expecting value");
		goto error;
	}

	if (isl_stream_eat_if_available(s, '/')) {
		tok2 = next_token(s);
		if (!tok2) {
			isl_stream_error(s, NULL, "unexpected EOF");
			goto error;
		}
		if (tok2->type != ISL_TOKEN_VALUE) {
			isl_stream_error(s, tok2, "expecting value");
			goto error;
		}
		val = isl_val_rat_from_isl_int(s->ctx, tok->u.v, tok2->u.v);
		val = isl_val_normalize(val);
	} else {
		val = isl_val_int_from_isl_int(s->ctx, tok->u.v);
	}

	isl_token_free(tok);
	isl_token_free(tok2);
	return val;
error:
	isl_token_free(tok);
	isl_token_free(tok2);
	return NULL;
}

/* Read an isl_val from "str".
 */
struct isl_val *isl_val_read_from_str(struct isl_ctx *ctx,
	const char *str)
{
	isl_val *val;
	struct isl_stream *s = isl_stream_new_str(ctx, str);
	if (!s)
		return NULL;
	val = isl_stream_read_val(s);
	isl_stream_free(s);
	return val;
}

static int accept_cst_factor(struct isl_stream *s, isl_int *f)
{
	struct isl_token *tok;

	tok = next_token(s);
	if (!tok || tok->type != ISL_TOKEN_VALUE) {
		isl_stream_error(s, tok, "expecting constant value");
		goto error;
	}

	isl_int_mul(*f, *f, tok->u.v);

	isl_token_free(tok);

	if (isl_stream_eat_if_available(s, '*'))
		return accept_cst_factor(s, f);

	return 0;
error:
	isl_token_free(tok);
	return -1;
}

/* Given an affine expression aff, return an affine expression
 * for aff % d, with d the next token on the stream, which is
 * assumed to be a constant.
 *
 * We introduce an integer division q = [aff/d] and the result
 * is set to aff - d q.
 */
static __isl_give isl_pw_aff *affine_mod(struct isl_stream *s,
	struct vars *v, __isl_take isl_pw_aff *aff)
{
	struct isl_token *tok;
	isl_pw_aff *q;

	tok = next_token(s);
	if (!tok || tok->type != ISL_TOKEN_VALUE) {
		isl_stream_error(s, tok, "expecting constant value");
		goto error;
	}

	q = isl_pw_aff_copy(aff);
	q = isl_pw_aff_scale_down(q, tok->u.v);
	q = isl_pw_aff_floor(q);
	q = isl_pw_aff_scale(q, tok->u.v);

	aff = isl_pw_aff_sub(aff, q);

	isl_token_free(tok);
	return aff;
error:
	isl_pw_aff_free(aff);
	isl_token_free(tok);
	return NULL;
}

static __isl_give isl_pw_aff *accept_affine(struct isl_stream *s,
	__isl_take isl_space *space, struct vars *v);
static __isl_give isl_pw_aff_list *accept_affine_list(struct isl_stream *s,
	__isl_take isl_space *dim, struct vars *v);

static __isl_give isl_pw_aff *accept_minmax(struct isl_stream *s,
	__isl_take isl_space *dim, struct vars *v)
{
	struct isl_token *tok;
	isl_pw_aff_list *list = NULL;
	int min;

	tok = isl_stream_next_token(s);
	if (!tok)
		goto error;
	min = tok->type == ISL_TOKEN_MIN;
	isl_token_free(tok);

	if (isl_stream_eat(s, '('))
		goto error;

	list = accept_affine_list(s, isl_space_copy(dim), v);
	if (!list)
		goto error;

	if (isl_stream_eat(s, ')'))
		goto error;

	isl_space_free(dim);
	return min ? isl_pw_aff_list_min(list) : isl_pw_aff_list_max(list);
error:
	isl_space_free(dim);
	isl_pw_aff_list_free(list);
	return NULL;
}

/* Is "tok" the start of an integer division?
 */
static int is_start_of_div(struct isl_token *tok)
{
	if (!tok)
		return 0;
	if (tok->type == '[')
		return 1;
	if (tok->type == ISL_TOKEN_FLOOR)
		return 1;
	if (tok->type == ISL_TOKEN_CEIL)
		return 1;
	if (tok->type == ISL_TOKEN_FLOORD)
		return 1;
	if (tok->type == ISL_TOKEN_CEILD)
		return 1;
	return 0;
}

/* Read an integer division from "s" and return it as an isl_pw_aff.
 *
 * The integer division can be of the form
 *
 *	[<affine expression>]
 *	floor(<affine expression>)
 *	ceil(<affine expression>)
 *	floord(<affine expression>,<denominator>)
 *	ceild(<affine expression>,<denominator>)
 */
static __isl_give isl_pw_aff *accept_div(struct isl_stream *s,
	__isl_take isl_space *dim, struct vars *v)
{
	struct isl_token *tok;
	int f = 0;
	int c = 0;
	int extra = 0;
	isl_pw_aff *pwaff = NULL;

	if (isl_stream_eat_if_available(s, ISL_TOKEN_FLOORD))
		extra = f = 1;
	else if (isl_stream_eat_if_available(s, ISL_TOKEN_CEILD))
		extra = c = 1;
	else if (isl_stream_eat_if_available(s, ISL_TOKEN_FLOOR))
		f = 1;
	else if (isl_stream_eat_if_available(s, ISL_TOKEN_CEIL))
		c = 1;
	if (f || c) {
		if (isl_stream_eat(s, '('))
			goto error;
	} else {
		if (isl_stream_eat(s, '['))
			goto error;
	}

	pwaff = accept_affine(s, isl_space_copy(dim), v);

	if (extra) {
		if (isl_stream_eat(s, ','))
			goto error;

		tok = next_token(s);
		if (!tok)
			goto error;
		if (tok->type != ISL_TOKEN_VALUE) {
			isl_stream_error(s, tok, "expected denominator");
			isl_stream_push_token(s, tok);
			goto error;
		}
		isl_pw_aff_scale_down(pwaff,  tok->u.v);
		isl_token_free(tok);
	}

	if (c)
		pwaff = isl_pw_aff_ceil(pwaff);
	else
		pwaff = isl_pw_aff_floor(pwaff);

	if (f || c) {
		if (isl_stream_eat(s, ')'))
			goto error;
	} else {
		if (isl_stream_eat(s, ']'))
			goto error;
	}

	isl_space_free(dim);
	return pwaff;
error:
	isl_space_free(dim);
	isl_pw_aff_free(pwaff);
	return NULL;
}

static __isl_give isl_pw_aff *accept_affine_factor(struct isl_stream *s,
	__isl_take isl_space *dim, struct vars *v)
{
	struct isl_token *tok = NULL;
	isl_pw_aff *res = NULL;

	tok = next_token(s);
	if (!tok) {
		isl_stream_error(s, NULL, "unexpected EOF");
		goto error;
	}

	if (tok->type == ISL_TOKEN_AFF) {
		res = isl_pw_aff_copy(tok->u.pwaff);
		isl_token_free(tok);
	} else if (tok->type == ISL_TOKEN_IDENT) {
		int n = v->n;
		int pos = vars_pos(v, tok->u.s, -1);
		isl_aff *aff;

		if (pos < 0)
			goto error;
		if (pos >= n) {
			vars_drop(v, v->n - n);
			isl_stream_error(s, tok, "unknown identifier");
			goto error;
		}

		aff = isl_aff_zero_on_domain(isl_local_space_from_space(isl_space_copy(dim)));
		if (!aff)
			goto error;
		isl_int_set_si(aff->v->el[2 + pos], 1);
		res = isl_pw_aff_from_aff(aff);
		isl_token_free(tok);
	} else if (tok->type == ISL_TOKEN_VALUE) {
		if (isl_stream_eat_if_available(s, '*')) {
			res = accept_affine_factor(s, isl_space_copy(dim), v);
			res = isl_pw_aff_scale(res, tok->u.v);
		} else {
			isl_local_space *ls;
			isl_aff *aff;
			ls = isl_local_space_from_space(isl_space_copy(dim));
			aff = isl_aff_zero_on_domain(ls);
			aff = isl_aff_add_constant(aff, tok->u.v);
			res = isl_pw_aff_from_aff(aff);
		}
		isl_token_free(tok);
	} else if (tok->type == '(') {
		isl_token_free(tok);
		tok = NULL;
		res = accept_affine(s, isl_space_copy(dim), v);
		if (!res)
			goto error;
		if (isl_stream_eat(s, ')'))
			goto error;
	} else if (is_start_of_div(tok)) {
		isl_stream_push_token(s, tok);
		tok = NULL;
		res = accept_div(s, isl_space_copy(dim), v);
	} else if (tok->type == ISL_TOKEN_MIN || tok->type == ISL_TOKEN_MAX) {
		isl_stream_push_token(s, tok);
		tok = NULL;
		res = accept_minmax(s, isl_space_copy(dim), v);
	} else {
		isl_stream_error(s, tok, "expecting factor");
		goto error;
	}
	if (isl_stream_eat_if_available(s, '%') ||
	    isl_stream_eat_if_available(s, ISL_TOKEN_MOD)) {
		isl_space_free(dim);
		return affine_mod(s, v, res);
	}
	if (isl_stream_eat_if_available(s, '*')) {
		isl_int f;
		isl_int_init(f);
		isl_int_set_si(f, 1);
		if (accept_cst_factor(s, &f) < 0) {
			isl_int_clear(f);
			goto error2;
		}
		res = isl_pw_aff_scale(res, f);
		isl_int_clear(f);
	}
	if (isl_stream_eat_if_available(s, '/')) {
		isl_int f;
		isl_int_init(f);
		isl_int_set_si(f, 1);
		if (accept_cst_factor(s, &f) < 0) {
			isl_int_clear(f);
			goto error2;
		}
		res = isl_pw_aff_scale_down(res, f);
		isl_int_clear(f);
	}

	isl_space_free(dim);
	return res;
error:
	isl_token_free(tok);
error2:
	isl_pw_aff_free(res);
	isl_space_free(dim);
	return NULL;
}

static __isl_give isl_pw_aff *add_cst(__isl_take isl_pw_aff *pwaff, isl_int v)
{
	isl_aff *aff;
	isl_space *space;

	space = isl_pw_aff_get_domain_space(pwaff);
	aff = isl_aff_zero_on_domain(isl_local_space_from_space(space));
	aff = isl_aff_add_constant(aff, v);

	return isl_pw_aff_add(pwaff, isl_pw_aff_from_aff(aff));
}

/* Return a piecewise affine expression defined on the specified domain
 * that represents NaN.
 */
static __isl_give isl_pw_aff *nan_on_domain(__isl_keep isl_space *space)
{
	isl_local_space *ls;

	ls = isl_local_space_from_space(isl_space_copy(space));
	return isl_pw_aff_nan_on_domain(ls);
}

static __isl_give isl_pw_aff *accept_affine(struct isl_stream *s,
	__isl_take isl_space *space, struct vars *v)
{
	struct isl_token *tok = NULL;
	isl_local_space *ls;
	isl_pw_aff *res;
	int sign = 1;

	ls = isl_local_space_from_space(isl_space_copy(space));
	res = isl_pw_aff_from_aff(isl_aff_zero_on_domain(ls));
	if (!res)
		goto error;

	for (;;) {
		tok = next_token(s);
		if (!tok) {
			isl_stream_error(s, NULL, "unexpected EOF");
			goto error;
		}
		if (tok->type == '-') {
			sign = -sign;
			isl_token_free(tok);
			continue;
		}
		if (tok->type == '(' || is_start_of_div(tok) ||
		    tok->type == ISL_TOKEN_MIN || tok->type == ISL_TOKEN_MAX ||
		    tok->type == ISL_TOKEN_IDENT ||
		    tok->type == ISL_TOKEN_AFF) {
			isl_pw_aff *term;
			isl_stream_push_token(s, tok);
			tok = NULL;
			term = accept_affine_factor(s,
						    isl_space_copy(space), v);
			if (sign < 0)
				res = isl_pw_aff_sub(res, term);
			else
				res = isl_pw_aff_add(res, term);
			if (!res)
				goto error;
			sign = 1;
		} else if (tok->type == ISL_TOKEN_VALUE) {
			if (sign < 0)
				isl_int_neg(tok->u.v, tok->u.v);
			if (isl_stream_eat_if_available(s, '*') ||
			    isl_stream_next_token_is(s, ISL_TOKEN_IDENT)) {
				isl_pw_aff *term;
				term = accept_affine_factor(s,
						    isl_space_copy(space), v);
				term = isl_pw_aff_scale(term, tok->u.v);
				res = isl_pw_aff_add(res, term);
				if (!res)
					goto error;
			} else {
				res = add_cst(res, tok->u.v);
			}
			sign = 1;
		} else if (tok->type == ISL_TOKEN_NAN) {
			res = isl_pw_aff_add(res, nan_on_domain(space));
		} else {
			isl_stream_error(s, tok, "unexpected isl_token");
			isl_stream_push_token(s, tok);
			isl_pw_aff_free(res);
			isl_space_free(space);
			return NULL;
		}
		isl_token_free(tok);

		tok = next_token(s);
		if (tok && tok->type == '-') {
			sign = -sign;
			isl_token_free(tok);
		} else if (tok && tok->type == '+') {
			/* nothing */
			isl_token_free(tok);
		} else if (tok && tok->type == ISL_TOKEN_VALUE &&
			   isl_int_is_neg(tok->u.v)) {
			isl_stream_push_token(s, tok);
		} else {
			if (tok)
				isl_stream_push_token(s, tok);
			break;
		}
	}

	isl_space_free(space);
	return res;
error:
	isl_space_free(space);
	isl_token_free(tok);
	isl_pw_aff_free(res);
	return NULL;
}

static int is_comparator(struct isl_token *tok)
{
	if (!tok)
		return 0;

	switch (tok->type) {
	case ISL_TOKEN_LT:
	case ISL_TOKEN_GT:
	case ISL_TOKEN_LE:
	case ISL_TOKEN_GE:
	case ISL_TOKEN_NE:
	case '=':
		return 1;
	default:
		return 0;
	}
}

static __isl_give isl_map *read_formula(struct isl_stream *s,
	struct vars *v, __isl_take isl_map *map, int rational);
static __isl_give isl_pw_aff *accept_extended_affine(struct isl_stream *s,
	__isl_take isl_space *dim, struct vars *v, int rational);

/* Accept a ternary operator, given the first argument.
 */
static __isl_give isl_pw_aff *accept_ternary(struct isl_stream *s,
	__isl_take isl_map *cond, struct vars *v, int rational)
{
	isl_space *dim;
	isl_pw_aff *pwaff1 = NULL, *pwaff2 = NULL, *pa_cond;

	if (!cond)
		return NULL;

	if (isl_stream_eat(s, '?'))
		goto error;

	dim = isl_space_wrap(isl_map_get_space(cond));
	pwaff1 = accept_extended_affine(s, dim, v, rational);
	if (!pwaff1)
		goto error;

	if (isl_stream_eat(s, ':'))
		goto error;

	dim = isl_pw_aff_get_domain_space(pwaff1);
	pwaff2 = accept_extended_affine(s, dim, v, rational);
	if (!pwaff1)
		goto error;

	pa_cond = isl_set_indicator_function(isl_map_wrap(cond));
	return isl_pw_aff_cond(pa_cond, pwaff1, pwaff2);
error:
	isl_map_free(cond);
	isl_pw_aff_free(pwaff1);
	isl_pw_aff_free(pwaff2);
	return NULL;
}

/* Set *line and *col to those of the next token, if any.
 */
static void set_current_line_col(struct isl_stream *s, int *line, int *col)
{
	struct isl_token *tok;

	tok = isl_stream_next_token(s);
	if (!tok)
		return;

	*line = tok->line;
	*col = tok->col;
	isl_stream_push_token(s, tok);
}

/* Push a token encapsulating "pa" onto "s", with the given
 * line and column.
 */
static int push_aff(struct isl_stream *s, int line, int col,
	__isl_take isl_pw_aff *pa)
{
	struct isl_token *tok;

	tok = isl_token_new(s->ctx, line, col, 0);
	if (!tok)
		goto error;
	tok->type = ISL_TOKEN_AFF;
	tok->u.pwaff = pa;
	isl_stream_push_token(s, tok);

	return 0;
error:
	isl_pw_aff_free(pa);
	return -1;
}

/* Accept an affine expression that may involve ternary operators.
 * We first read an affine expression.
 * If it is not followed by a comparison operator, we simply return it.
 * Otherwise, we assume the affine expression is part of the first
 * argument of a ternary operator and try to parse that.
 */
static __isl_give isl_pw_aff *accept_extended_affine(struct isl_stream *s,
	__isl_take isl_space *dim, struct vars *v, int rational)
{
	isl_space *space;
	isl_map *cond;
	isl_pw_aff *pwaff;
	struct isl_token *tok;
	int line = -1, col = -1;
	int is_comp;

	set_current_line_col(s, &line, &col);

	pwaff = accept_affine(s, dim, v);
	if (rational)
		pwaff = isl_pw_aff_set_rational(pwaff);
	if (!pwaff)
		return NULL;

	tok = isl_stream_next_token(s);
	if (!tok)
		return isl_pw_aff_free(pwaff);

	is_comp = is_comparator(tok);
	isl_stream_push_token(s, tok);
	if (!is_comp)
		return pwaff;

	space = isl_pw_aff_get_domain_space(pwaff);
	cond = isl_map_universe(isl_space_unwrap(space));

	if (push_aff(s, line, col, pwaff) < 0)
		cond = isl_map_free(cond);
	if (!cond)
		return NULL;

	cond = read_formula(s, v, cond, rational);

	return accept_ternary(s, cond, v, rational);
}

static __isl_give isl_map *read_var_def(struct isl_stream *s,
	__isl_take isl_map *map, enum isl_dim_type type, struct vars *v,
	int rational)
{
	isl_pw_aff *def;
	int pos;
	isl_map *def_map;

	if (type == isl_dim_param)
		pos = isl_map_dim(map, isl_dim_param);
	else {
		pos = isl_map_dim(map, isl_dim_in);
		if (type == isl_dim_out)
			pos += isl_map_dim(map, isl_dim_out);
		type = isl_dim_in;
	}
	--pos;

	def = accept_extended_affine(s, isl_space_wrap(isl_map_get_space(map)),
					v, rational);
	def_map = isl_map_from_pw_aff(def);
	def_map = isl_map_equate(def_map, type, pos, isl_dim_out, 0);
	def_map = isl_set_unwrap(isl_map_domain(def_map));

	map = isl_map_intersect(map, def_map);

	return map;
}

static __isl_give isl_pw_aff_list *accept_affine_list(struct isl_stream *s,
	__isl_take isl_space *dim, struct vars *v)
{
	isl_pw_aff *pwaff;
	isl_pw_aff_list *list;
	struct isl_token *tok = NULL;

	pwaff = accept_affine(s, isl_space_copy(dim), v);
	list = isl_pw_aff_list_from_pw_aff(pwaff);
	if (!list)
		goto error;

	for (;;) {
		tok = isl_stream_next_token(s);
		if (!tok) {
			isl_stream_error(s, NULL, "unexpected EOF");
			goto error;
		}
		if (tok->type != ',') {
			isl_stream_push_token(s, tok);
			break;
		}
		isl_token_free(tok);

		pwaff = accept_affine(s, isl_space_copy(dim), v);
		list = isl_pw_aff_list_concat(list,
				isl_pw_aff_list_from_pw_aff(pwaff));
		if (!list)
			goto error;
	}

	isl_space_free(dim);
	return list;
error:
	isl_space_free(dim);
	isl_pw_aff_list_free(list);
	return NULL;
}

static __isl_give isl_map *read_defined_var_list(struct isl_stream *s,
	struct vars *v, __isl_take isl_map *map, int rational)
{
	struct isl_token *tok;

	while ((tok = isl_stream_next_token(s)) != NULL) {
		int p;
		int n = v->n;

		if (tok->type != ISL_TOKEN_IDENT)
			break;

		p = vars_pos(v, tok->u.s, -1);
		if (p < 0)
			goto error;
		if (p < n) {
			isl_stream_error(s, tok, "expecting unique identifier");
			goto error;
		}

		map = isl_map_add_dims(map, isl_dim_out, 1);

		isl_token_free(tok);
		tok = isl_stream_next_token(s);
		if (tok && tok->type == '=') {
			isl_token_free(tok);
			map = read_var_def(s, map, isl_dim_out, v, rational);
			tok = isl_stream_next_token(s);
		}

		if (!tok || tok->type != ',')
			break;

		isl_token_free(tok);
	}
	if (tok)
		isl_stream_push_token(s, tok);

	return map;
error:
	isl_token_free(tok);
	isl_map_free(map);
	return NULL;
}

static int next_is_tuple(struct isl_stream *s)
{
	struct isl_token *tok;
	int is_tuple;

	tok = isl_stream_next_token(s);
	if (!tok)
		return 0;
	if (tok->type == '[') {
		isl_stream_push_token(s, tok);
		return 1;
	}
	if (tok->type != ISL_TOKEN_IDENT && !tok->is_keyword) {
		isl_stream_push_token(s, tok);
		return 0;
	}

	is_tuple = isl_stream_next_token_is(s, '[');

	isl_stream_push_token(s, tok);

	return is_tuple;
}

/* Allocate an initial tuple with zero dimensions and an anonymous,
 * unstructured space.
 * A tuple is represented as an isl_multi_pw_aff.
 * The range space is the space of the tuple.
 * The domain space is an anonymous space
 * with a dimension for each variable in the set of variables in "v".
 * If a given dimension is not defined in terms of earlier dimensions in
 * the input, then the corresponding isl_pw_aff is set equal to one time
 * the variable corresponding to the dimension being defined.
 */
static __isl_give isl_multi_pw_aff *tuple_alloc(struct vars *v)
{
	return isl_multi_pw_aff_alloc(isl_space_alloc(v->ctx, 0, v->n, 0));
}

/* Is "pa" an expression in term of earlier dimensions?
 * The alternative is that the dimension is defined to be equal to itself,
 * meaning that it has a universe domain and an expression that depends
 * on itself.  "i" is the position of the expression in a sequence
 * of "n" expressions.  The final dimensions of "pa" correspond to
 * these "n" expressions.
 */
static int pw_aff_is_expr(__isl_keep isl_pw_aff *pa, int i, int n)
{
	isl_aff *aff;

	if (!pa)
		return -1;
	if (pa->n != 1)
		return 1;
	if (!isl_set_plain_is_universe(pa->p[0].set))
		return 1;

	aff = pa->p[0].aff;
	if (isl_int_is_zero(aff->v->el[aff->v->size - n + i]))
		return 1;
	return 0;
}

/* Does the tuple contain any dimensions that are defined
 * in terms of earlier dimensions?
 */
static int tuple_has_expr(__isl_keep isl_multi_pw_aff *tuple)
{
	int i, n;
	int has_expr = 0;
	isl_pw_aff *pa;

	if (!tuple)
		return -1;
	n = isl_multi_pw_aff_dim(tuple, isl_dim_out);
	for (i = 0; i < n; ++i) {
		pa = isl_multi_pw_aff_get_pw_aff(tuple, i);
		has_expr = pw_aff_is_expr(pa, i, n);
		isl_pw_aff_free(pa);
		if (has_expr < 0 || has_expr)
			break;
	}

	return has_expr;
}

/* Add a dimension to the given tuple.
 * The dimension is initially undefined, so it is encoded
 * as one times itself.
 */
static __isl_give isl_multi_pw_aff *tuple_add_dim(
	__isl_take isl_multi_pw_aff *tuple, struct vars *v)
{
	isl_space *space;
	isl_aff *aff;
	isl_pw_aff *pa;

	tuple = isl_multi_pw_aff_add_dims(tuple, isl_dim_in, 1);
	space = isl_multi_pw_aff_get_domain_space(tuple);
	aff = isl_aff_zero_on_domain(isl_local_space_from_space(space));
	aff = isl_aff_add_coefficient_si(aff, isl_dim_in, v->n, 1);
	pa = isl_pw_aff_from_aff(aff);
	tuple = isl_multi_pw_aff_flat_range_product(tuple,
					    isl_multi_pw_aff_from_pw_aff(pa));

	return tuple;
}

/* Set the name of dimension "pos" in "tuple" to "name".
 * During printing, we add primes if the same name appears more than once
 * to distinguish the occurrences.  Here, we remove those primes from "name"
 * before setting the name of the dimension.
 */
static __isl_give isl_multi_pw_aff *tuple_set_dim_name(
	__isl_take isl_multi_pw_aff *tuple, int pos, char *name)
{
	char *prime;

	if (!name)
		return tuple;

	prime = strchr(name, '\'');
	if (prime)
		*prime = '\0';
	tuple = isl_multi_pw_aff_set_dim_name(tuple, isl_dim_set, pos, name);
	if (prime)
		*prime = '\'';

	return tuple;
}

/* Accept a piecewise affine expression.
 *
 * At the outer level, the piecewise affine expression may be of the form
 *
 *	aff1 : condition1; aff2 : conditions2; ...
 *
 * or simply
 *
 *	aff
 *
 * each of the affine expressions may in turn include ternary operators.
 *
 * There may be parentheses around some subexpression of "aff1"
 * around "aff1" itself, around "aff1 : condition1" and/or
 * around the entire piecewise affine expression.
 * We therefore remove the opening parenthesis (if any) from the stream
 * in case the closing parenthesis follows the colon, but if the closing
 * parenthesis is the first thing in the stream after the parsed affine
 * expression, we push the parsed expression onto the stream and parse
 * again in case the parentheses enclose some subexpression of "aff1".
 */
static __isl_give isl_pw_aff *accept_piecewise_affine(struct isl_stream *s,
	__isl_take isl_space *space, struct vars *v, int rational)
{
	isl_pw_aff *res;
	isl_space *res_space;

	res_space = isl_space_from_domain(isl_space_copy(space));
	res_space = isl_space_add_dims(res_space, isl_dim_out, 1);
	res = isl_pw_aff_empty(res_space);
	do {
		isl_pw_aff *pa;
		int seen_paren;
		int line = -1, col = -1;

		set_current_line_col(s, &line, &col);
		seen_paren = isl_stream_eat_if_available(s, '(');
		if (seen_paren)
			pa = accept_piecewise_affine(s, isl_space_copy(space),
							v, rational);
		else
			pa = accept_extended_affine(s, isl_space_copy(space),
							v, rational);
		if (seen_paren && isl_stream_eat_if_available(s, ')')) {
			seen_paren = 0;
			if (push_aff(s, line, col, pa) < 0)
				goto error;
			pa = accept_extended_affine(s, isl_space_copy(space),
							v, rational);
		}
		if (isl_stream_eat_if_available(s, ':')) {
			isl_space *dom_space;
			isl_set *dom;

			dom_space = isl_pw_aff_get_domain_space(pa);
			dom = isl_set_universe(dom_space);
			dom = read_formula(s, v, dom, rational);
			pa = isl_pw_aff_intersect_domain(pa, dom);
		}

		res = isl_pw_aff_union_add(res, pa);

		if (seen_paren && isl_stream_eat(s, ')'))
			goto error;
	} while (isl_stream_eat_if_available(s, ';'));

	isl_space_free(space);

	return res;
error:
	isl_space_free(space);
	return isl_pw_aff_free(res);
}

/* Read an affine expression from "s" and replace the definition
 * of dimension "pos" in "tuple" by this expression.
 *
 * accept_extended_affine requires a wrapped space as input.
 * The domain space of "tuple", on the other hand is an anonymous space,
 * so we have to adjust the space of the isl_pw_aff before adding it
 * to "tuple".
 */
static __isl_give isl_multi_pw_aff *read_tuple_var_def(struct isl_stream *s,
	__isl_take isl_multi_pw_aff *tuple, int pos, struct vars *v,
	int rational)
{
	isl_space *space;
	isl_pw_aff *def;

	space = isl_space_wrap(isl_space_alloc(s->ctx, 0, v->n, 0));

	def = accept_piecewise_affine(s, space, v, rational);

	space = isl_space_set_alloc(s->ctx, 0, v->n);
	def = isl_pw_aff_reset_domain_space(def, space);
	tuple = isl_multi_pw_aff_set_pw_aff(tuple, pos, def);

	return tuple;
}

/* Read a list of variables and/or affine expressions and return the list
 * as an isl_multi_pw_aff.
 * The elements in the list are separated by either "," or "][".
 * If "comma" is set then only "," is allowed.
 */
static __isl_give isl_multi_pw_aff *read_tuple_var_list(struct isl_stream *s,
	struct vars *v, int rational, int comma)
{
	int i = 0;
	struct isl_token *tok;
	isl_multi_pw_aff *res;

	res = tuple_alloc(v);

	if (isl_stream_next_token_is(s, ']'))
		return res;

	while ((tok = next_token(s)) != NULL) {
		int new_name = 0;

		res = tuple_add_dim(res, v);

		if (tok->type == ISL_TOKEN_IDENT) {
			int n = v->n;
			int p = vars_pos(v, tok->u.s, -1);
			if (p < 0)
				goto error;
			new_name = p >= n;
		}

		if (tok->type == '*') {
			if (vars_add_anon(v) < 0)
				goto error;
			isl_token_free(tok);
		} else if (new_name) {
			res = tuple_set_dim_name(res, i, v->v->name);
			isl_token_free(tok);
			if (isl_stream_eat_if_available(s, '='))
				res = read_tuple_var_def(s, res, i, v,
							rational);
		} else {
			isl_stream_push_token(s, tok);
			tok = NULL;
			if (vars_add_anon(v) < 0)
				goto error;
			res = read_tuple_var_def(s, res, i, v, rational);
		}

		tok = isl_stream_next_token(s);
		if (!comma && tok && tok->type == ']' &&
		    isl_stream_next_token_is(s, '[')) {
			isl_token_free(tok);
			tok = isl_stream_next_token(s);
		} else if (!tok || tok->type != ',')
			break;

		isl_token_free(tok);
		i++;
	}
	if (tok)
		isl_stream_push_token(s, tok);

	return res;
error:
	isl_token_free(tok);
	return isl_multi_pw_aff_free(res);
}

/* Read a tuple and represent it as an isl_multi_pw_aff.  See tuple_alloc.
 */
static __isl_give isl_multi_pw_aff *read_tuple(struct isl_stream *s,
	struct vars *v, int rational, int comma)
{
	struct isl_token *tok;
	char *name = NULL;
	isl_multi_pw_aff *res = NULL;

	tok = isl_stream_next_token(s);
	if (!tok)
		goto error;
	if (tok->type == ISL_TOKEN_IDENT || tok->is_keyword) {
		name = strdup(tok->u.s);
		isl_token_free(tok);
		if (!name)
			goto error;
	} else
		isl_stream_push_token(s, tok);
	if (isl_stream_eat(s, '['))
		goto error;
	if (next_is_tuple(s)) {
		isl_multi_pw_aff *out;
		int n;
		res = read_tuple(s, v, rational, comma);
		if (isl_stream_eat(s, ISL_TOKEN_TO))
			goto error;
		out = read_tuple(s, v, rational, comma);
		n = isl_multi_pw_aff_dim(out, isl_dim_out);
		res = isl_multi_pw_aff_add_dims(res, isl_dim_in, n);
		res = isl_multi_pw_aff_range_product(res, out);
	} else
		res = read_tuple_var_list(s, v, rational, comma);
	if (isl_stream_eat(s, ']'))
		goto error;

	if (name) {
		res = isl_multi_pw_aff_set_tuple_name(res, isl_dim_out, name);
		free(name);
	}

	return res;
error:
	free(name);
	return isl_multi_pw_aff_free(res);
}

/* Add the tuple represented by the isl_multi_pw_aff "tuple" to "map".
 * We first create the appropriate space in "map" based on the range
 * space of this isl_multi_pw_aff.  Then, we add equalities based
 * on the affine expressions.  These live in an anonymous space,
 * however, so we first need to reset the space to that of "map".
 */
static __isl_give isl_map *map_from_tuple(__isl_take isl_multi_pw_aff *tuple,
	__isl_take isl_map *map, enum isl_dim_type type, struct vars *v,
	int rational)
{
	int i, n;
	isl_ctx *ctx;
	isl_space *space = NULL;

	if (!map || !tuple)
		goto error;
	ctx = isl_multi_pw_aff_get_ctx(tuple);
	n = isl_multi_pw_aff_dim(tuple, isl_dim_out);
	space = isl_space_range(isl_multi_pw_aff_get_space(tuple));
	if (!space)
		goto error;

	if (type == isl_dim_param) {
		if (isl_space_has_tuple_name(space, isl_dim_set) ||
		    isl_space_is_wrapping(space)) {
			isl_die(ctx, isl_error_invalid,
				"parameter tuples cannot be named or nested",
				goto error);
		}
		map = isl_map_add_dims(map, type, n);
		for (i = 0; i < n; ++i) {
			isl_id *id;
			if (!isl_space_has_dim_name(space, isl_dim_set, i))
				isl_die(ctx, isl_error_invalid,
					"parameters must be named",
					goto error);
			id = isl_space_get_dim_id(space, isl_dim_set, i);
			map = isl_map_set_dim_id(map, isl_dim_param, i, id);
		}
	} else if (type == isl_dim_in) {
		isl_set *set;

		set = isl_set_universe(isl_space_copy(space));
		if (rational)
			set = isl_set_set_rational(set);
		set = isl_set_intersect_params(set, isl_map_params(map));
		map = isl_map_from_domain(set);
	} else {
		isl_set *set;

		set = isl_set_universe(isl_space_copy(space));
		if (rational)
			set = isl_set_set_rational(set);
		map = isl_map_from_domain_and_range(isl_map_domain(map), set);
	}

	for (i = 0; i < n; ++i) {
		isl_pw_aff *pa;
		isl_space *space;
		isl_aff *aff;
		isl_set *set;
		isl_map *map_i;

		pa = isl_multi_pw_aff_get_pw_aff(tuple, i);
		space = isl_pw_aff_get_domain_space(pa);
		aff = isl_aff_zero_on_domain(isl_local_space_from_space(space));
		aff = isl_aff_add_coefficient_si(aff,
						isl_dim_in, v->n - n + i, -1);
		pa = isl_pw_aff_add(pa, isl_pw_aff_from_aff(aff));
		if (rational)
			pa = isl_pw_aff_set_rational(pa);
		set = isl_pw_aff_zero_set(pa);
		map_i = isl_map_from_range(set);
		map_i = isl_map_reset_space(map_i, isl_map_get_space(map));
		map = isl_map_intersect(map, map_i);
	}

	isl_space_free(space);
	isl_multi_pw_aff_free(tuple);
	return map;
error:
	isl_space_free(space);
	isl_multi_pw_aff_free(tuple);
	isl_map_free(map);
	return NULL;
}

/* Read a tuple from "s" and add it to "map".
 * The tuple is initially represented as an isl_multi_pw_aff and
 * then added to "map".
 */
static __isl_give isl_map *read_map_tuple(struct isl_stream *s,
	__isl_take isl_map *map, enum isl_dim_type type, struct vars *v,
	int rational, int comma)
{
	isl_multi_pw_aff *tuple;

	tuple = read_tuple(s, v, rational, comma);
	if (!tuple)
		return isl_map_free(map);

	return map_from_tuple(tuple, map, type, v, rational);
}

static __isl_give isl_set *construct_constraints(
	__isl_take isl_set *set, int type,
	__isl_keep isl_pw_aff_list *left, __isl_keep isl_pw_aff_list *right,
	int rational)
{
	isl_set *cond;

	left = isl_pw_aff_list_copy(left);
	right = isl_pw_aff_list_copy(right);
	if (rational) {
		left = isl_pw_aff_list_set_rational(left);
		right = isl_pw_aff_list_set_rational(right);
	}
	if (type == ISL_TOKEN_LE)
		cond = isl_pw_aff_list_le_set(left, right);
	else if (type == ISL_TOKEN_GE)
		cond = isl_pw_aff_list_ge_set(left, right);
	else if (type == ISL_TOKEN_LT)
		cond = isl_pw_aff_list_lt_set(left, right);
	else if (type == ISL_TOKEN_GT)
		cond = isl_pw_aff_list_gt_set(left, right);
	else if (type == ISL_TOKEN_NE)
		cond = isl_pw_aff_list_ne_set(left, right);
	else
		cond = isl_pw_aff_list_eq_set(left, right);

	return isl_set_intersect(set, cond);
}

static __isl_give isl_map *add_constraint(struct isl_stream *s,
	struct vars *v, __isl_take isl_map *map, int rational)
{
	struct isl_token *tok = NULL;
	isl_pw_aff_list *list1 = NULL, *list2 = NULL;
	isl_set *set;

	set = isl_map_wrap(map);
	list1 = accept_affine_list(s, isl_set_get_space(set), v);
	if (!list1)
		goto error;
	tok = isl_stream_next_token(s);
	if (!is_comparator(tok)) {
		isl_stream_error(s, tok, "missing operator");
		if (tok)
			isl_stream_push_token(s, tok);
		tok = NULL;
		goto error;
	}
	for (;;) {
		list2 = accept_affine_list(s, isl_set_get_space(set), v);
		if (!list2)
			goto error;

		set = construct_constraints(set, tok->type, list1, list2,
						rational);
		isl_token_free(tok);
		isl_pw_aff_list_free(list1);
		list1 = list2;

		tok = isl_stream_next_token(s);
		if (!is_comparator(tok)) {
			if (tok)
				isl_stream_push_token(s, tok);
			break;
		}
	}
	isl_pw_aff_list_free(list1);

	return isl_set_unwrap(set);
error:
	if (tok)
		isl_token_free(tok);
	isl_pw_aff_list_free(list1);
	isl_pw_aff_list_free(list2);
	isl_set_free(set);
	return NULL;
}

static __isl_give isl_map *read_exists(struct isl_stream *s,
	struct vars *v, __isl_take isl_map *map, int rational)
{
	int n = v->n;
	int seen_paren = isl_stream_eat_if_available(s, '(');

	map = isl_map_from_domain(isl_map_wrap(map));
	map = read_defined_var_list(s, v, map, rational);

	if (isl_stream_eat(s, ':'))
		goto error;

	map = read_formula(s, v, map, rational);
	map = isl_set_unwrap(isl_map_domain(map));

	vars_drop(v, v->n - n);
	if (seen_paren && isl_stream_eat(s, ')'))
		goto error;

	return map;
error:
	isl_map_free(map);
	return NULL;
}

/* Parse an expression between parentheses and push the result
 * back on the stream.
 *
 * The parsed expression may be either an affine expression
 * or a condition.  The first type is pushed onto the stream
 * as an isl_pw_aff, while the second is pushed as an isl_map.
 *
 * If the initial token indicates the start of a condition,
 * we parse it as such.
 * Otherwise, we first parse an affine expression and push
 * that onto the stream.  If the affine expression covers the
 * entire expression between parentheses, we return.
 * Otherwise, we assume that the affine expression is the
 * start of a condition and continue parsing.
 */
static int resolve_paren_expr(struct isl_stream *s,
	struct vars *v, __isl_take isl_map *map, int rational)
{
	struct isl_token *tok, *tok2;
	int line, col;
	isl_pw_aff *pwaff;

	tok = isl_stream_next_token(s);
	if (!tok || tok->type != '(')
		goto error;

	if (isl_stream_next_token_is(s, '('))
		if (resolve_paren_expr(s, v, isl_map_copy(map), rational))
			goto error;

	if (isl_stream_next_token_is(s, ISL_TOKEN_EXISTS) ||
	    isl_stream_next_token_is(s, ISL_TOKEN_NOT) ||
	    isl_stream_next_token_is(s, ISL_TOKEN_TRUE) ||
	    isl_stream_next_token_is(s, ISL_TOKEN_FALSE) ||
	    isl_stream_next_token_is(s, ISL_TOKEN_MAP)) {
		map = read_formula(s, v, map, rational);
		if (isl_stream_eat(s, ')'))
			goto error;
		tok->type = ISL_TOKEN_MAP;
		tok->u.map = map;
		isl_stream_push_token(s, tok);
		return 0;
	}

	tok2 = isl_stream_next_token(s);
	if (!tok2)
		goto error;
	line = tok2->line;
	col = tok2->col;
	isl_stream_push_token(s, tok2);

	pwaff = accept_affine(s, isl_space_wrap(isl_map_get_space(map)), v);
	if (!pwaff)
		goto error;

	tok2 = isl_token_new(s->ctx, line, col, 0);
	if (!tok2)
		goto error2;
	tok2->type = ISL_TOKEN_AFF;
	tok2->u.pwaff = pwaff;

	if (isl_stream_eat_if_available(s, ')')) {
		isl_stream_push_token(s, tok2);
		isl_token_free(tok);
		isl_map_free(map);
		return 0;
	}

	isl_stream_push_token(s, tok2);

	map = read_formula(s, v, map, rational);
	if (isl_stream_eat(s, ')'))
		goto error;

	tok->type = ISL_TOKEN_MAP;
	tok->u.map = map;
	isl_stream_push_token(s, tok);

	return 0;
error2:
	isl_pw_aff_free(pwaff);
error:
	isl_token_free(tok);
	isl_map_free(map);
	return -1;
}

static __isl_give isl_map *read_conjunct(struct isl_stream *s,
	struct vars *v, __isl_take isl_map *map, int rational)
{
	if (isl_stream_next_token_is(s, '('))
		if (resolve_paren_expr(s, v, isl_map_copy(map), rational))
			goto error;

	if (isl_stream_next_token_is(s, ISL_TOKEN_MAP)) {
		struct isl_token *tok;
		tok = isl_stream_next_token(s);
		if (!tok)
			goto error;
		isl_map_free(map);
		map = isl_map_copy(tok->u.map);
		isl_token_free(tok);
		return map;
	}

	if (isl_stream_eat_if_available(s, ISL_TOKEN_EXISTS))
		return read_exists(s, v, map, rational);

	if (isl_stream_eat_if_available(s, ISL_TOKEN_TRUE))
		return map;

	if (isl_stream_eat_if_available(s, ISL_TOKEN_FALSE)) {
		isl_space *dim = isl_map_get_space(map);
		isl_map_free(map);
		return isl_map_empty(dim);
	}
		
	return add_constraint(s, v, map, rational);
error:
	isl_map_free(map);
	return NULL;
}

static __isl_give isl_map *read_conjuncts(struct isl_stream *s,
	struct vars *v, __isl_take isl_map *map, int rational)
{
	isl_map *res;
	int negate;

	negate = isl_stream_eat_if_available(s, ISL_TOKEN_NOT);
	res = read_conjunct(s, v, isl_map_copy(map), rational);
	if (negate)
		res = isl_map_subtract(isl_map_copy(map), res);

	while (res && isl_stream_eat_if_available(s, ISL_TOKEN_AND)) {
		isl_map *res_i;

		negate = isl_stream_eat_if_available(s, ISL_TOKEN_NOT);
		res_i = read_conjunct(s, v, isl_map_copy(map), rational);
		if (negate)
			res = isl_map_subtract(res, res_i);
		else
			res = isl_map_intersect(res, res_i);
	}

	isl_map_free(map);
	return res;
}

static struct isl_map *read_disjuncts(struct isl_stream *s,
	struct vars *v, __isl_take isl_map *map, int rational)
{
	isl_map *res;

	if (isl_stream_next_token_is(s, '}')) {
		isl_space *dim = isl_map_get_space(map);
		isl_map_free(map);
		return isl_map_universe(dim);
	}

	res = read_conjuncts(s, v, isl_map_copy(map), rational);
	while (isl_stream_eat_if_available(s, ISL_TOKEN_OR)) {
		isl_map *res_i;

		res_i = read_conjuncts(s, v, isl_map_copy(map), rational);
		res = isl_map_union(res, res_i);
	}

	isl_map_free(map);
	return res;
}

/* Read a first order formula from "s", add the corresponding
 * constraints to "map" and return the result.
 *
 * In particular, read a formula of the form
 *
 *	a
 *
 * or
 *
 *	a implies b
 *
 * where a and b are disjunctions.
 *
 * In the first case, map is replaced by
 *
 *	map \cap { [..] : a }
 *
 * In the second case, it is replaced by
 *
 *	(map \setminus { [..] : a}) \cup (map \cap { [..] : b })
 */
static __isl_give isl_map *read_formula(struct isl_stream *s,
	struct vars *v, __isl_take isl_map *map, int rational)
{
	isl_map *res;

	res = read_disjuncts(s, v, isl_map_copy(map), rational);

	if (isl_stream_eat_if_available(s, ISL_TOKEN_IMPLIES)) {
		isl_map *res2;

		res = isl_map_subtract(isl_map_copy(map), res);
		res2 = read_disjuncts(s, v, map, rational);
		res = isl_map_union(res, res2);
	} else
		isl_map_free(map);

	return res;
}

static int polylib_pos_to_isl_pos(__isl_keep isl_basic_map *bmap, int pos)
{
	if (pos < isl_basic_map_dim(bmap, isl_dim_out))
		return 1 + isl_basic_map_dim(bmap, isl_dim_param) +
			   isl_basic_map_dim(bmap, isl_dim_in) + pos;
	pos -= isl_basic_map_dim(bmap, isl_dim_out);

	if (pos < isl_basic_map_dim(bmap, isl_dim_in))
		return 1 + isl_basic_map_dim(bmap, isl_dim_param) + pos;
	pos -= isl_basic_map_dim(bmap, isl_dim_in);

	if (pos < isl_basic_map_dim(bmap, isl_dim_div))
		return 1 + isl_basic_map_dim(bmap, isl_dim_param) +
			   isl_basic_map_dim(bmap, isl_dim_in) +
			   isl_basic_map_dim(bmap, isl_dim_out) + pos;
	pos -= isl_basic_map_dim(bmap, isl_dim_div);

	if (pos < isl_basic_map_dim(bmap, isl_dim_param))
		return 1 + pos;

	return 0;
}

static __isl_give isl_basic_map *basic_map_read_polylib_constraint(
	struct isl_stream *s, __isl_take isl_basic_map *bmap)
{
	int j;
	struct isl_token *tok;
	int type;
	int k;
	isl_int *c;
	unsigned nparam;
	unsigned dim;

	if (!bmap)
		return NULL;

	nparam = isl_basic_map_dim(bmap, isl_dim_param);
	dim = isl_basic_map_dim(bmap, isl_dim_out);

	tok = isl_stream_next_token(s);
	if (!tok || tok->type != ISL_TOKEN_VALUE) {
		isl_stream_error(s, tok, "expecting coefficient");
		if (tok)
			isl_stream_push_token(s, tok);
		goto error;
	}
	if (!tok->on_new_line) {
		isl_stream_error(s, tok, "coefficient should appear on new line");
		isl_stream_push_token(s, tok);
		goto error;
	}

	type = isl_int_get_si(tok->u.v);
	isl_token_free(tok);

	isl_assert(s->ctx, type == 0 || type == 1, goto error);
	if (type == 0) {
		k = isl_basic_map_alloc_equality(bmap);
		c = bmap->eq[k];
	} else {
		k = isl_basic_map_alloc_inequality(bmap);
		c = bmap->ineq[k];
	}
	if (k < 0)
		goto error;

	for (j = 0; j < 1 + isl_basic_map_total_dim(bmap); ++j) {
		int pos;
		tok = isl_stream_next_token(s);
		if (!tok || tok->type != ISL_TOKEN_VALUE) {
			isl_stream_error(s, tok, "expecting coefficient");
			if (tok)
				isl_stream_push_token(s, tok);
			goto error;
		}
		if (tok->on_new_line) {
			isl_stream_error(s, tok,
				"coefficient should not appear on new line");
			isl_stream_push_token(s, tok);
			goto error;
		}
		pos = polylib_pos_to_isl_pos(bmap, j);
		isl_int_set(c[pos], tok->u.v);
		isl_token_free(tok);
	}

	return bmap;
error:
	isl_basic_map_free(bmap);
	return NULL;
}

static __isl_give isl_basic_map *basic_map_read_polylib(struct isl_stream *s)
{
	int i;
	struct isl_token *tok;
	struct isl_token *tok2;
	int n_row, n_col;
	int on_new_line;
	unsigned in = 0, out, local = 0;
	struct isl_basic_map *bmap = NULL;
	int nparam = 0;

	tok = isl_stream_next_token(s);
	if (!tok) {
		isl_stream_error(s, NULL, "unexpected EOF");
		return NULL;
	}
	tok2 = isl_stream_next_token(s);
	if (!tok2) {
		isl_token_free(tok);
		isl_stream_error(s, NULL, "unexpected EOF");
		return NULL;
	}
	if (tok->type != ISL_TOKEN_VALUE || tok2->type != ISL_TOKEN_VALUE) {
		isl_stream_push_token(s, tok2);
		isl_stream_push_token(s, tok);
		isl_stream_error(s, NULL,
				 "expecting constraint matrix dimensions");
		return NULL;
	}
	n_row = isl_int_get_si(tok->u.v);
	n_col = isl_int_get_si(tok2->u.v);
	on_new_line = tok2->on_new_line;
	isl_token_free(tok2);
	isl_token_free(tok);
	isl_assert(s->ctx, !on_new_line, return NULL);
	isl_assert(s->ctx, n_row >= 0, return NULL);
	isl_assert(s->ctx, n_col >= 2 + nparam, return NULL);
	tok = isl_stream_next_token_on_same_line(s);
	if (tok) {
		if (tok->type != ISL_TOKEN_VALUE) {
			isl_stream_error(s, tok,
				    "expecting number of output dimensions");
			isl_stream_push_token(s, tok);
			goto error;
		}
		out = isl_int_get_si(tok->u.v);
		isl_token_free(tok);

		tok = isl_stream_next_token_on_same_line(s);
		if (!tok || tok->type != ISL_TOKEN_VALUE) {
			isl_stream_error(s, tok,
				    "expecting number of input dimensions");
			if (tok)
				isl_stream_push_token(s, tok);
			goto error;
		}
		in = isl_int_get_si(tok->u.v);
		isl_token_free(tok);

		tok = isl_stream_next_token_on_same_line(s);
		if (!tok || tok->type != ISL_TOKEN_VALUE) {
			isl_stream_error(s, tok,
				    "expecting number of existentials");
			if (tok)
				isl_stream_push_token(s, tok);
			goto error;
		}
		local = isl_int_get_si(tok->u.v);
		isl_token_free(tok);

		tok = isl_stream_next_token_on_same_line(s);
		if (!tok || tok->type != ISL_TOKEN_VALUE) {
			isl_stream_error(s, tok,
				    "expecting number of parameters");
			if (tok)
				isl_stream_push_token(s, tok);
			goto error;
		}
		nparam = isl_int_get_si(tok->u.v);
		isl_token_free(tok);
		if (n_col != 1 + out + in + local + nparam + 1) {
			isl_stream_error(s, NULL,
				    "dimensions don't match");
			goto error;
		}
	} else
		out = n_col - 2 - nparam;
	bmap = isl_basic_map_alloc(s->ctx, nparam, in, out, local, n_row, n_row);
	if (!bmap)
		return NULL;

	for (i = 0; i < local; ++i) {
		int k = isl_basic_map_alloc_div(bmap);
		if (k < 0)
			goto error;
		isl_seq_clr(bmap->div[k], 1 + 1 + nparam + in + out + local);
	}

	for (i = 0; i < n_row; ++i)
		bmap = basic_map_read_polylib_constraint(s, bmap);

	tok = isl_stream_next_token_on_same_line(s);
	if (tok) {
		isl_stream_error(s, tok, "unexpected extra token on line");
		isl_stream_push_token(s, tok);
		goto error;
	}

	bmap = isl_basic_map_simplify(bmap);
	bmap = isl_basic_map_finalize(bmap);
	return bmap;
error:
	isl_basic_map_free(bmap);
	return NULL;
}

static struct isl_map *map_read_polylib(struct isl_stream *s)
{
	struct isl_token *tok;
	struct isl_token *tok2;
	int i, n;
	struct isl_map *map;

	tok = isl_stream_next_token(s);
	if (!tok) {
		isl_stream_error(s, NULL, "unexpected EOF");
		return NULL;
	}
	tok2 = isl_stream_next_token_on_same_line(s);
	if (tok2 && tok2->type == ISL_TOKEN_VALUE) {
		isl_stream_push_token(s, tok2);
		isl_stream_push_token(s, tok);
		return isl_map_from_basic_map(basic_map_read_polylib(s));
	}
	if (tok2) {
		isl_stream_error(s, tok2, "unexpected token");
		isl_stream_push_token(s, tok2);
		isl_stream_push_token(s, tok);
		return NULL;
	}
	n = isl_int_get_si(tok->u.v);
	isl_token_free(tok);

	isl_assert(s->ctx, n >= 1, return NULL);

	map = isl_map_from_basic_map(basic_map_read_polylib(s));

	for (i = 1; map && i < n; ++i)
		map = isl_map_union(map,
			isl_map_from_basic_map(basic_map_read_polylib(s)));

	return map;
}

static int optional_power(struct isl_stream *s)
{
	int pow;
	struct isl_token *tok;

	tok = isl_stream_next_token(s);
	if (!tok)
		return 1;
	if (tok->type != '^') {
		isl_stream_push_token(s, tok);
		return 1;
	}
	isl_token_free(tok);
	tok = isl_stream_next_token(s);
	if (!tok || tok->type != ISL_TOKEN_VALUE) {
		isl_stream_error(s, tok, "expecting exponent");
		if (tok)
			isl_stream_push_token(s, tok);
		return 1;
	}
	pow = isl_int_get_si(tok->u.v);
	isl_token_free(tok);
	return pow;
}

static __isl_give isl_pw_qpolynomial *read_term(struct isl_stream *s,
	__isl_keep isl_map *map, struct vars *v);

static __isl_give isl_pw_qpolynomial *read_factor(struct isl_stream *s,
	__isl_keep isl_map *map, struct vars *v)
{
	isl_pw_qpolynomial *pwqp;
	struct isl_token *tok;

	tok = next_token(s);
	if (!tok) {
		isl_stream_error(s, NULL, "unexpected EOF");
		return NULL;
	}
	if (tok->type == '(') {
		int pow;

		isl_token_free(tok);
		pwqp = read_term(s, map, v);
		if (!pwqp)
			return NULL;
		if (isl_stream_eat(s, ')'))
			goto error;
		pow = optional_power(s);
		pwqp = isl_pw_qpolynomial_pow(pwqp, pow);
	} else if (tok->type == ISL_TOKEN_VALUE) {
		struct isl_token *tok2;
		isl_qpolynomial *qp;

		tok2 = isl_stream_next_token(s);
		if (tok2 && tok2->type == '/') {
			isl_token_free(tok2);
			tok2 = next_token(s);
			if (!tok2 || tok2->type != ISL_TOKEN_VALUE) {
				isl_stream_error(s, tok2, "expected denominator");
				isl_token_free(tok);
				isl_token_free(tok2);
				return NULL;
			}
			qp = isl_qpolynomial_rat_cst_on_domain(isl_map_get_space(map),
						    tok->u.v, tok2->u.v);
			isl_token_free(tok2);
		} else {
			isl_stream_push_token(s, tok2);
			qp = isl_qpolynomial_cst_on_domain(isl_map_get_space(map),
						tok->u.v);
		}
		isl_token_free(tok);
		pwqp = isl_pw_qpolynomial_from_qpolynomial(qp);
	} else if (tok->type == ISL_TOKEN_INFTY) {
		isl_qpolynomial *qp;
		isl_token_free(tok);
		qp = isl_qpolynomial_infty_on_domain(isl_map_get_space(map));
		pwqp = isl_pw_qpolynomial_from_qpolynomial(qp);
	} else if (tok->type == ISL_TOKEN_NAN) {
		isl_qpolynomial *qp;
		isl_token_free(tok);
		qp = isl_qpolynomial_nan_on_domain(isl_map_get_space(map));
		pwqp = isl_pw_qpolynomial_from_qpolynomial(qp);
	} else if (tok->type == ISL_TOKEN_IDENT) {
		int n = v->n;
		int pos = vars_pos(v, tok->u.s, -1);
		int pow;
		isl_qpolynomial *qp;
		if (pos < 0) {
			isl_token_free(tok);
			return NULL;
		}
		if (pos >= n) {
			vars_drop(v, v->n - n);
			isl_stream_error(s, tok, "unknown identifier");
			isl_token_free(tok);
			return NULL;
		}
		isl_token_free(tok);
		pow = optional_power(s);
		qp = isl_qpolynomial_var_pow_on_domain(isl_map_get_space(map), pos, pow);
		pwqp = isl_pw_qpolynomial_from_qpolynomial(qp);
	} else if (is_start_of_div(tok)) {
		isl_pw_aff *pwaff;
		int pow;

		isl_stream_push_token(s, tok);
		pwaff = accept_div(s, isl_map_get_space(map), v);
		pow = optional_power(s);
		pwqp = isl_pw_qpolynomial_from_pw_aff(pwaff);
		pwqp = isl_pw_qpolynomial_pow(pwqp, pow);
	} else if (tok->type == '-') {
		isl_token_free(tok);
		pwqp = read_factor(s, map, v);
		pwqp = isl_pw_qpolynomial_neg(pwqp);
	} else {
		isl_stream_error(s, tok, "unexpected isl_token");
		isl_stream_push_token(s, tok);
		return NULL;
	}

	if (isl_stream_eat_if_available(s, '*') ||
	    isl_stream_next_token_is(s, ISL_TOKEN_IDENT)) {
		isl_pw_qpolynomial *pwqp2;

		pwqp2 = read_factor(s, map, v);
		pwqp = isl_pw_qpolynomial_mul(pwqp, pwqp2);
	}

	return pwqp;
error:
	isl_pw_qpolynomial_free(pwqp);
	return NULL;
}

static __isl_give isl_pw_qpolynomial *read_term(struct isl_stream *s,
	__isl_keep isl_map *map, struct vars *v)
{
	struct isl_token *tok;
	isl_pw_qpolynomial *pwqp;

	pwqp = read_factor(s, map, v);

	for (;;) {
		tok = next_token(s);
		if (!tok)
			return pwqp;

		if (tok->type == '+') {
			isl_pw_qpolynomial *pwqp2;

			isl_token_free(tok);
			pwqp2 = read_factor(s, map, v);
			pwqp = isl_pw_qpolynomial_add(pwqp, pwqp2);
		} else if (tok->type == '-') {
			isl_pw_qpolynomial *pwqp2;

			isl_token_free(tok);
			pwqp2 = read_factor(s, map, v);
			pwqp = isl_pw_qpolynomial_sub(pwqp, pwqp2);
		} else if (tok->type == ISL_TOKEN_VALUE &&
			    isl_int_is_neg(tok->u.v)) {
			isl_pw_qpolynomial *pwqp2;

			isl_stream_push_token(s, tok);
			pwqp2 = read_factor(s, map, v);
			pwqp = isl_pw_qpolynomial_add(pwqp, pwqp2);
		} else {
			isl_stream_push_token(s, tok);
			break;
		}
	}

	return pwqp;
}

static __isl_give isl_map *read_optional_formula(struct isl_stream *s,
	__isl_take isl_map *map, struct vars *v, int rational)
{
	struct isl_token *tok;

	tok = isl_stream_next_token(s);
	if (!tok) {
		isl_stream_error(s, NULL, "unexpected EOF");
		goto error;
	}
	if (tok->type == ':' ||
	    (tok->type == ISL_TOKEN_OR && !strcmp(tok->u.s, "|"))) {
		isl_token_free(tok);
		map = read_formula(s, v, map, rational);
	} else
		isl_stream_push_token(s, tok);

	return map;
error:
	isl_map_free(map);
	return NULL;
}

static struct isl_obj obj_read_poly(struct isl_stream *s,
	__isl_take isl_map *map, struct vars *v, int n)
{
	struct isl_obj obj = { isl_obj_pw_qpolynomial, NULL };
	isl_pw_qpolynomial *pwqp;
	struct isl_set *set;

	pwqp = read_term(s, map, v);
	map = read_optional_formula(s, map, v, 0);
	set = isl_map_range(map);

	pwqp = isl_pw_qpolynomial_intersect_domain(pwqp, set);

	vars_drop(v, v->n - n);

	obj.v = pwqp;
	return obj;
}

static struct isl_obj obj_read_poly_or_fold(struct isl_stream *s,
	__isl_take isl_set *set, struct vars *v, int n)
{
	struct isl_obj obj = { isl_obj_pw_qpolynomial_fold, NULL };
	isl_pw_qpolynomial *pwqp;
	isl_pw_qpolynomial_fold *pwf = NULL;

	if (!isl_stream_eat_if_available(s, ISL_TOKEN_MAX))
		return obj_read_poly(s, set, v, n);

	if (isl_stream_eat(s, '('))
		goto error;

	pwqp = read_term(s, set, v);
	pwf = isl_pw_qpolynomial_fold_from_pw_qpolynomial(isl_fold_max, pwqp);

	while (isl_stream_eat_if_available(s, ',')) {
		isl_pw_qpolynomial_fold *pwf_i;
		pwqp = read_term(s, set, v);
		pwf_i = isl_pw_qpolynomial_fold_from_pw_qpolynomial(isl_fold_max,
									pwqp);
		pwf = isl_pw_qpolynomial_fold_fold(pwf, pwf_i);
	}

	if (isl_stream_eat(s, ')'))
		goto error;

	set = read_optional_formula(s, set, v, 0);
	pwf = isl_pw_qpolynomial_fold_intersect_domain(pwf, set);

	vars_drop(v, v->n - n);

	obj.v = pwf;
	return obj;
error:
	isl_set_free(set);
	isl_pw_qpolynomial_fold_free(pwf);
	obj.type = isl_obj_none;
	return obj;
}

static int is_rational(struct isl_stream *s)
{
	struct isl_token *tok;

	tok = isl_stream_next_token(s);
	if (!tok)
		return 0;
	if (tok->type == ISL_TOKEN_RAT && isl_stream_next_token_is(s, ':')) {
		isl_token_free(tok);
		isl_stream_eat(s, ':');
		return 1;
	}

	isl_stream_push_token(s, tok);

	return 0;
}

static struct isl_obj obj_read_body(struct isl_stream *s,
	__isl_take isl_map *map, struct vars *v)
{
	struct isl_token *tok;
	struct isl_obj obj = { isl_obj_set, NULL };
	int n = v->n;
	int rational;

	rational = is_rational(s);
	if (rational)
		map = isl_map_set_rational(map);

	if (isl_stream_next_token_is(s, ':')) {
		obj.type = isl_obj_set;
		obj.v = read_optional_formula(s, map, v, rational);
		return obj;
	}

	if (!next_is_tuple(s))
		return obj_read_poly_or_fold(s, map, v, n);

	map = read_map_tuple(s, map, isl_dim_in, v, rational, 0);
	if (!map)
		goto error;
	tok = isl_stream_next_token(s);
	if (!tok)
		goto error;
	if (tok->type == ISL_TOKEN_TO) {
		obj.type = isl_obj_map;
		isl_token_free(tok);
		if (!next_is_tuple(s)) {
			isl_set *set = isl_map_domain(map);
			return obj_read_poly_or_fold(s, set, v, n);
		}
		map = read_map_tuple(s, map, isl_dim_out, v, rational, 0);
		if (!map)
			goto error;
	} else {
		map = isl_map_domain(map);
		isl_stream_push_token(s, tok);
	}

	map = read_optional_formula(s, map, v, rational);

	vars_drop(v, v->n - n);

	obj.v = map;
	return obj;
error:
	isl_map_free(map);
	obj.type = isl_obj_none;
	return obj;
}

static struct isl_obj to_union(isl_ctx *ctx, struct isl_obj obj)
{
	if (obj.type == isl_obj_map) {
		obj.v = isl_union_map_from_map(obj.v);
		obj.type = isl_obj_union_map;
	} else if (obj.type == isl_obj_set) {
		obj.v = isl_union_set_from_set(obj.v);
		obj.type = isl_obj_union_set;
	} else if (obj.type == isl_obj_pw_qpolynomial) {
		obj.v = isl_union_pw_qpolynomial_from_pw_qpolynomial(obj.v);
		obj.type = isl_obj_union_pw_qpolynomial;
	} else if (obj.type == isl_obj_pw_qpolynomial_fold) {
		obj.v = isl_union_pw_qpolynomial_fold_from_pw_qpolynomial_fold(obj.v);
		obj.type = isl_obj_union_pw_qpolynomial_fold;
	} else
		isl_assert(ctx, 0, goto error);
	return obj;
error:
	obj.type->free(obj.v);
	obj.type = isl_obj_none;
	return obj;
}

static struct isl_obj obj_add(struct isl_ctx *ctx,
	struct isl_obj obj1, struct isl_obj obj2)
{
	if (obj1.type == isl_obj_set && obj2.type == isl_obj_union_set)
		obj1 = to_union(ctx, obj1);
	if (obj1.type == isl_obj_union_set && obj2.type == isl_obj_set)
		obj2 = to_union(ctx, obj2);
	if (obj1.type == isl_obj_map && obj2.type == isl_obj_union_map)
		obj1 = to_union(ctx, obj1);
	if (obj1.type == isl_obj_union_map && obj2.type == isl_obj_map)
		obj2 = to_union(ctx, obj2);
	if (obj1.type == isl_obj_pw_qpolynomial &&
	    obj2.type == isl_obj_union_pw_qpolynomial)
		obj1 = to_union(ctx, obj1);
	if (obj1.type == isl_obj_union_pw_qpolynomial &&
	    obj2.type == isl_obj_pw_qpolynomial)
		obj2 = to_union(ctx, obj2);
	if (obj1.type == isl_obj_pw_qpolynomial_fold &&
	    obj2.type == isl_obj_union_pw_qpolynomial_fold)
		obj1 = to_union(ctx, obj1);
	if (obj1.type == isl_obj_union_pw_qpolynomial_fold &&
	    obj2.type == isl_obj_pw_qpolynomial_fold)
		obj2 = to_union(ctx, obj2);
	isl_assert(ctx, obj1.type == obj2.type, goto error);
	if (obj1.type == isl_obj_map && !isl_map_has_equal_space(obj1.v, obj2.v)) {
		obj1 = to_union(ctx, obj1);
		obj2 = to_union(ctx, obj2);
	}
	if (obj1.type == isl_obj_set && !isl_set_has_equal_space(obj1.v, obj2.v)) {
		obj1 = to_union(ctx, obj1);
		obj2 = to_union(ctx, obj2);
	}
	if (obj1.type == isl_obj_pw_qpolynomial &&
	    !isl_pw_qpolynomial_has_equal_space(obj1.v, obj2.v)) {
		obj1 = to_union(ctx, obj1);
		obj2 = to_union(ctx, obj2);
	}
	if (obj1.type == isl_obj_pw_qpolynomial_fold &&
	    !isl_pw_qpolynomial_fold_has_equal_space(obj1.v, obj2.v)) {
		obj1 = to_union(ctx, obj1);
		obj2 = to_union(ctx, obj2);
	}
	obj1.v = obj1.type->add(obj1.v, obj2.v);
	return obj1;
error:
	obj1.type->free(obj1.v);
	obj2.type->free(obj2.v);
	obj1.type = isl_obj_none;
	obj1.v = NULL;
	return obj1;
}

static struct isl_obj obj_read(struct isl_stream *s)
{
	isl_map *map = NULL;
	struct isl_token *tok;
	struct vars *v = NULL;
	struct isl_obj obj = { isl_obj_set, NULL };

	tok = next_token(s);
	if (!tok) {
		isl_stream_error(s, NULL, "unexpected EOF");
		goto error;
	}
	if (tok->type == ISL_TOKEN_VALUE) {
		struct isl_token *tok2;
		struct isl_map *map;

		tok2 = isl_stream_next_token(s);
		if (!tok2 || tok2->type != ISL_TOKEN_VALUE ||
		    isl_int_is_neg(tok2->u.v)) {
			if (tok2)
				isl_stream_push_token(s, tok2);
			obj.type = isl_obj_val;
			obj.v = isl_val_int_from_isl_int(s->ctx, tok->u.v);
			isl_token_free(tok);
			return obj;
		}
		isl_stream_push_token(s, tok2);
		isl_stream_push_token(s, tok);
		map = map_read_polylib(s);
		if (!map)
			goto error;
		if (isl_map_may_be_set(map))
			obj.v = isl_map_range(map);
		else {
			obj.type = isl_obj_map;
			obj.v = map;
		}
		return obj;
	}
	v = vars_new(s->ctx);
	if (!v) {
		isl_stream_push_token(s, tok);
		goto error;
	}
	map = isl_map_universe(isl_space_params_alloc(s->ctx, 0));
	if (tok->type == '[') {
		isl_stream_push_token(s, tok);
		map = read_map_tuple(s, map, isl_dim_param, v, 0, 0);
		if (!map)
			goto error;
		tok = isl_stream_next_token(s);
		if (!tok || tok->type != ISL_TOKEN_TO) {
			isl_stream_error(s, tok, "expecting '->'");
			if (tok)
				isl_stream_push_token(s, tok);
			goto error;
		}
		isl_token_free(tok);
		tok = isl_stream_next_token(s);
	}
	if (!tok || tok->type != '{') {
		isl_stream_error(s, tok, "expecting '{'");
		if (tok)
			isl_stream_push_token(s, tok);
		goto error;
	}
	isl_token_free(tok);

	tok = isl_stream_next_token(s);
	if (!tok)
		;
	else if (tok->type == ISL_TOKEN_IDENT && !strcmp(tok->u.s, "Sym")) {
		isl_token_free(tok);
		if (isl_stream_eat(s, '='))
			goto error;
		map = read_map_tuple(s, map, isl_dim_param, v, 0, 1);
		if (!map)
			goto error;
	} else if (tok->type == '}') {
		obj.type = isl_obj_union_set;
		obj.v = isl_union_set_empty(isl_map_get_space(map));
		isl_token_free(tok);
		goto done;
	} else
		isl_stream_push_token(s, tok);

	for (;;) {
		struct isl_obj o;
		tok = NULL;
		o = obj_read_body(s, isl_map_copy(map), v);
		if (o.type == isl_obj_none || !o.v)
			goto error;
		if (!obj.v)
			obj = o;
		else {
			obj = obj_add(s->ctx, obj, o);
			if (obj.type == isl_obj_none || !obj.v)
				goto error;
		}
		tok = isl_stream_next_token(s);
		if (!tok || tok->type != ';')
			break;
		isl_token_free(tok);
		if (isl_stream_next_token_is(s, '}')) {
			tok = isl_stream_next_token(s);
			break;
		}
	}

	if (tok && tok->type == '}') {
		isl_token_free(tok);
	} else {
		isl_stream_error(s, tok, "unexpected isl_token");
		if (tok)
			isl_token_free(tok);
		goto error;
	}
done:
	vars_free(v);
	isl_map_free(map);

	return obj;
error:
	isl_map_free(map);
	obj.type->free(obj.v);
	if (v)
		vars_free(v);
	obj.v = NULL;
	return obj;
}

struct isl_obj isl_stream_read_obj(struct isl_stream *s)
{
	return obj_read(s);
}

__isl_give isl_map *isl_stream_read_map(struct isl_stream *s)
{
	struct isl_obj obj;

	obj = obj_read(s);
	if (obj.v)
		isl_assert(s->ctx, obj.type == isl_obj_map ||
				   obj.type == isl_obj_set, goto error);
	
	if (obj.type == isl_obj_set)
		obj.v = isl_map_from_range(obj.v);

	return obj.v;
error:
	obj.type->free(obj.v);
	return NULL;
}

__isl_give isl_set *isl_stream_read_set(struct isl_stream *s)
{
	struct isl_obj obj;

	obj = obj_read(s);
	if (obj.v) {
		if (obj.type == isl_obj_map && isl_map_may_be_set(obj.v)) {
			obj.v = isl_map_range(obj.v);
			obj.type = isl_obj_set;
		}
		isl_assert(s->ctx, obj.type == isl_obj_set, goto error);
	}

	return obj.v;
error:
	obj.type->free(obj.v);
	return NULL;
}

__isl_give isl_union_map *isl_stream_read_union_map(struct isl_stream *s)
{
	struct isl_obj obj;

	obj = obj_read(s);
	if (obj.type == isl_obj_map) {
		obj.type = isl_obj_union_map;
		obj.v = isl_union_map_from_map(obj.v);
	}
	if (obj.type == isl_obj_set) {
		obj.type = isl_obj_union_set;
		obj.v = isl_union_set_from_set(obj.v);
	}
	if (obj.v && obj.type == isl_obj_union_set &&
	    isl_union_set_is_empty(obj.v))
		obj.type = isl_obj_union_map;
	if (obj.v && obj.type != isl_obj_union_map)
		isl_die(s->ctx, isl_error_invalid, "invalid input", goto error);

	return obj.v;
error:
	obj.type->free(obj.v);
	return NULL;
}

__isl_give isl_union_set *isl_stream_read_union_set(struct isl_stream *s)
{
	struct isl_obj obj;

	obj = obj_read(s);
	if (obj.type == isl_obj_set) {
		obj.type = isl_obj_union_set;
		obj.v = isl_union_set_from_set(obj.v);
	}
	if (obj.v)
		isl_assert(s->ctx, obj.type == isl_obj_union_set, goto error);

	return obj.v;
error:
	obj.type->free(obj.v);
	return NULL;
}

static __isl_give isl_basic_map *basic_map_read(struct isl_stream *s)
{
	struct isl_obj obj;
	struct isl_map *map;
	struct isl_basic_map *bmap;

	obj = obj_read(s);
	if (obj.v && (obj.type != isl_obj_map && obj.type != isl_obj_set))
		isl_die(s->ctx, isl_error_invalid, "not a (basic) set or map",
			goto error);
	map = obj.v;
	if (!map)
		return NULL;

	if (map->n > 1)
		isl_die(s->ctx, isl_error_invalid,
			"set or map description involves "
			"more than one disjunct", goto error);

	if (map->n == 0)
		bmap = isl_basic_map_empty_like_map(map);
	else
		bmap = isl_basic_map_copy(map->p[0]);

	isl_map_free(map);

	return bmap;
error:
	obj.type->free(obj.v);
	return NULL;
}

static __isl_give isl_basic_set *basic_set_read(struct isl_stream *s)
{
	isl_basic_map *bmap;
	bmap = basic_map_read(s);
	if (!bmap)
		return NULL;
	if (!isl_basic_map_may_be_set(bmap))
		isl_die(s->ctx, isl_error_invalid,
			"input is not a set", goto error);
	return isl_basic_map_range(bmap);
error:
	isl_basic_map_free(bmap);
	return NULL;
}

__isl_give isl_basic_map *isl_basic_map_read_from_file(isl_ctx *ctx,
	FILE *input)
{
	struct isl_basic_map *bmap;
	struct isl_stream *s = isl_stream_new_file(ctx, input);
	if (!s)
		return NULL;
	bmap = basic_map_read(s);
	isl_stream_free(s);
	return bmap;
}

__isl_give isl_basic_set *isl_basic_set_read_from_file(isl_ctx *ctx,
	FILE *input)
{
	isl_basic_set *bset;
	struct isl_stream *s = isl_stream_new_file(ctx, input);
	if (!s)
		return NULL;
	bset = basic_set_read(s);
	isl_stream_free(s);
	return bset;
}

struct isl_basic_map *isl_basic_map_read_from_str(struct isl_ctx *ctx,
	const char *str)
{
	struct isl_basic_map *bmap;
	struct isl_stream *s = isl_stream_new_str(ctx, str);
	if (!s)
		return NULL;
	bmap = basic_map_read(s);
	isl_stream_free(s);
	return bmap;
}

struct isl_basic_set *isl_basic_set_read_from_str(struct isl_ctx *ctx,
	const char *str)
{
	isl_basic_set *bset;
	struct isl_stream *s = isl_stream_new_str(ctx, str);
	if (!s)
		return NULL;
	bset = basic_set_read(s);
	isl_stream_free(s);
	return bset;
}

__isl_give isl_map *isl_map_read_from_file(struct isl_ctx *ctx,
	FILE *input)
{
	struct isl_map *map;
	struct isl_stream *s = isl_stream_new_file(ctx, input);
	if (!s)
		return NULL;
	map = isl_stream_read_map(s);
	isl_stream_free(s);
	return map;
}

__isl_give isl_map *isl_map_read_from_str(struct isl_ctx *ctx,
	const char *str)
{
	struct isl_map *map;
	struct isl_stream *s = isl_stream_new_str(ctx, str);
	if (!s)
		return NULL;
	map = isl_stream_read_map(s);
	isl_stream_free(s);
	return map;
}

__isl_give isl_set *isl_set_read_from_file(struct isl_ctx *ctx,
	FILE *input)
{
	isl_set *set;
	struct isl_stream *s = isl_stream_new_file(ctx, input);
	if (!s)
		return NULL;
	set = isl_stream_read_set(s);
	isl_stream_free(s);
	return set;
}

struct isl_set *isl_set_read_from_str(struct isl_ctx *ctx,
	const char *str)
{
	isl_set *set;
	struct isl_stream *s = isl_stream_new_str(ctx, str);
	if (!s)
		return NULL;
	set = isl_stream_read_set(s);
	isl_stream_free(s);
	return set;
}

__isl_give isl_union_map *isl_union_map_read_from_file(isl_ctx *ctx,
	FILE *input)
{
	isl_union_map *umap;
	struct isl_stream *s = isl_stream_new_file(ctx, input);
	if (!s)
		return NULL;
	umap = isl_stream_read_union_map(s);
	isl_stream_free(s);
	return umap;
}

__isl_give isl_union_map *isl_union_map_read_from_str(struct isl_ctx *ctx,
		const char *str)
{
	isl_union_map *umap;
	struct isl_stream *s = isl_stream_new_str(ctx, str);
	if (!s)
		return NULL;
	umap = isl_stream_read_union_map(s);
	isl_stream_free(s);
	return umap;
}

__isl_give isl_union_set *isl_union_set_read_from_file(isl_ctx *ctx,
	FILE *input)
{
	isl_union_set *uset;
	struct isl_stream *s = isl_stream_new_file(ctx, input);
	if (!s)
		return NULL;
	uset = isl_stream_read_union_set(s);
	isl_stream_free(s);
	return uset;
}

__isl_give isl_union_set *isl_union_set_read_from_str(struct isl_ctx *ctx,
		const char *str)
{
	isl_union_set *uset;
	struct isl_stream *s = isl_stream_new_str(ctx, str);
	if (!s)
		return NULL;
	uset = isl_stream_read_union_set(s);
	isl_stream_free(s);
	return uset;
}

static __isl_give isl_vec *isl_vec_read_polylib(struct isl_stream *s)
{
	struct isl_vec *vec = NULL;
	struct isl_token *tok;
	unsigned size;
	int j;

	tok = isl_stream_next_token(s);
	if (!tok || tok->type != ISL_TOKEN_VALUE) {
		isl_stream_error(s, tok, "expecting vector length");
		goto error;
	}

	size = isl_int_get_si(tok->u.v);
	isl_token_free(tok);

	vec = isl_vec_alloc(s->ctx, size);

	for (j = 0; j < size; ++j) {
		tok = isl_stream_next_token(s);
		if (!tok || tok->type != ISL_TOKEN_VALUE) {
			isl_stream_error(s, tok, "expecting constant value");
			goto error;
		}
		isl_int_set(vec->el[j], tok->u.v);
		isl_token_free(tok);
	}

	return vec;
error:
	isl_token_free(tok);
	isl_vec_free(vec);
	return NULL;
}

static __isl_give isl_vec *vec_read(struct isl_stream *s)
{
	return isl_vec_read_polylib(s);
}

__isl_give isl_vec *isl_vec_read_from_file(isl_ctx *ctx, FILE *input)
{
	isl_vec *v;
	struct isl_stream *s = isl_stream_new_file(ctx, input);
	if (!s)
		return NULL;
	v = vec_read(s);
	isl_stream_free(s);
	return v;
}

__isl_give isl_pw_qpolynomial *isl_stream_read_pw_qpolynomial(
	struct isl_stream *s)
{
	struct isl_obj obj;

	obj = obj_read(s);
	if (obj.v)
		isl_assert(s->ctx, obj.type == isl_obj_pw_qpolynomial,
			   goto error);

	return obj.v;
error:
	obj.type->free(obj.v);
	return NULL;
}

__isl_give isl_pw_qpolynomial *isl_pw_qpolynomial_read_from_str(isl_ctx *ctx,
		const char *str)
{
	isl_pw_qpolynomial *pwqp;
	struct isl_stream *s = isl_stream_new_str(ctx, str);
	if (!s)
		return NULL;
	pwqp = isl_stream_read_pw_qpolynomial(s);
	isl_stream_free(s);
	return pwqp;
}

__isl_give isl_pw_qpolynomial *isl_pw_qpolynomial_read_from_file(isl_ctx *ctx,
		FILE *input)
{
	isl_pw_qpolynomial *pwqp;
	struct isl_stream *s = isl_stream_new_file(ctx, input);
	if (!s)
		return NULL;
	pwqp = isl_stream_read_pw_qpolynomial(s);
	isl_stream_free(s);
	return pwqp;
}

/* Is the next token an identifer not in "v"?
 */
static int next_is_fresh_ident(struct isl_stream *s, struct vars *v)
{
	int n = v->n;
	int fresh;
	struct isl_token *tok;

	tok = isl_stream_next_token(s);
	if (!tok)
		return 0;
	fresh = tok->type == ISL_TOKEN_IDENT && vars_pos(v, tok->u.s, -1) >= n;
	isl_stream_push_token(s, tok);

	vars_drop(v, v->n - n);

	return fresh;
}

/* First read the domain of the affine expression, which may be
 * a parameter space or a set.
 * The tricky part is that we don't know if the domain is a set or not,
 * so when we are trying to read the domain, we may actually be reading
 * the affine expression itself (defined on a parameter domains)
 * If the tuple we are reading is named, we assume it's the domain.
 * Also, if inside the tuple, the first thing we find is a nested tuple
 * or a new identifier, we again assume it's the domain.
 * Otherwise, we assume we are reading an affine expression.
 */
static __isl_give isl_set *read_aff_domain(struct isl_stream *s,
	__isl_take isl_set *dom, struct vars *v)
{
	struct isl_token *tok;

	tok = isl_stream_next_token(s);
	if (tok && (tok->type == ISL_TOKEN_IDENT || tok->is_keyword)) {
		isl_stream_push_token(s, tok);
		return read_map_tuple(s, dom, isl_dim_set, v, 1, 0);
	}
	if (!tok || tok->type != '[') {
		isl_stream_error(s, tok, "expecting '['");
		goto error;
	}
	if (next_is_tuple(s) || next_is_fresh_ident(s, v)) {
		isl_stream_push_token(s, tok);
		dom = read_map_tuple(s, dom, isl_dim_set, v, 1, 0);
	} else
		isl_stream_push_token(s, tok);

	return dom;
error:
	if (tok)
		isl_stream_push_token(s, tok);
	isl_set_free(dom);
	return NULL;
}

/* Read an affine expression from "s".
 */
__isl_give isl_aff *isl_stream_read_aff(struct isl_stream *s)
{
	isl_aff *aff;
	isl_multi_aff *ma;

	ma = isl_stream_read_multi_aff(s);
	if (!ma)
		return NULL;
	if (isl_multi_aff_dim(ma, isl_dim_out) != 1)
		isl_die(s->ctx, isl_error_invalid,
			"expecting single affine expression",
			goto error);

	aff = isl_multi_aff_get_aff(ma, 0);
	isl_multi_aff_free(ma);
	return aff;
error:
	isl_multi_aff_free(ma);
	return NULL;
}

/* Read a piecewise affine expression from "s" with domain (space) "dom".
 */
static __isl_give isl_pw_aff *read_pw_aff_with_dom(struct isl_stream *s,
	__isl_take isl_set *dom, struct vars *v)
{
	isl_pw_aff *pwaff = NULL;

	if (!isl_set_is_params(dom) && isl_stream_eat(s, ISL_TOKEN_TO))
		goto error;

	if (isl_stream_eat(s, '['))
		goto error;

	pwaff = accept_affine(s, isl_set_get_space(dom), v);

	if (isl_stream_eat(s, ']'))
		goto error;

	dom = read_optional_formula(s, dom, v, 0);
	pwaff = isl_pw_aff_intersect_domain(pwaff, dom);

	return pwaff;
error:
	isl_set_free(dom);
	isl_pw_aff_free(pwaff);
	return NULL;
}

__isl_give isl_pw_aff *isl_stream_read_pw_aff(struct isl_stream *s)
{
	struct vars *v;
	isl_set *dom = NULL;
	isl_set *aff_dom;
	isl_pw_aff *pa = NULL;
	int n;

	v = vars_new(s->ctx);
	if (!v)
		return NULL;

	dom = isl_set_universe(isl_space_params_alloc(s->ctx, 0));
	if (next_is_tuple(s)) {
		dom = read_map_tuple(s, dom, isl_dim_param, v, 1, 0);
		if (isl_stream_eat(s, ISL_TOKEN_TO))
			goto error;
	}
	if (isl_stream_eat(s, '{'))
		goto error;

	n = v->n;
	aff_dom = read_aff_domain(s, isl_set_copy(dom), v);
	pa = read_pw_aff_with_dom(s, aff_dom, v);
	vars_drop(v, v->n - n);

	while (isl_stream_eat_if_available(s, ';')) {
		isl_pw_aff *pa_i;

		n = v->n;
		aff_dom = read_aff_domain(s, isl_set_copy(dom), v);
		pa_i = read_pw_aff_with_dom(s, aff_dom, v);
		vars_drop(v, v->n - n);

		pa = isl_pw_aff_union_add(pa, pa_i);
	}

	if (isl_stream_eat(s, '}'))
		goto error;

	vars_free(v);
	isl_set_free(dom);
	return pa;
error:
	vars_free(v);
	isl_set_free(dom);
	isl_pw_aff_free(pa);
	return NULL;
}

__isl_give isl_aff *isl_aff_read_from_str(isl_ctx *ctx, const char *str)
{
	isl_aff *aff;
	struct isl_stream *s = isl_stream_new_str(ctx, str);
	if (!s)
		return NULL;
	aff = isl_stream_read_aff(s);
	isl_stream_free(s);
	return aff;
}

__isl_give isl_pw_aff *isl_pw_aff_read_from_str(isl_ctx *ctx, const char *str)
{
	isl_pw_aff *pa;
	struct isl_stream *s = isl_stream_new_str(ctx, str);
	if (!s)
		return NULL;
	pa = isl_stream_read_pw_aff(s);
	isl_stream_free(s);
	return pa;
}

/* Read an isl_pw_multi_aff from "s".
 * We currently read a generic object and if it turns out to be a set or
 * a map, we convert that to an isl_pw_multi_aff.
 * It would be more efficient if we were to construct the isl_pw_multi_aff
 * directly.
 */
__isl_give isl_pw_multi_aff *isl_stream_read_pw_multi_aff(struct isl_stream *s)
{
	struct isl_obj obj;

	obj = obj_read(s);
	if (!obj.v)
		return NULL;

	if (obj.type == isl_obj_map)
		return isl_pw_multi_aff_from_map(obj.v);
	if (obj.type == isl_obj_set)
		return isl_pw_multi_aff_from_set(obj.v);

	obj.type->free(obj.v);
	isl_die(s->ctx, isl_error_invalid, "unexpected object type",
		return NULL);
}

__isl_give isl_pw_multi_aff *isl_pw_multi_aff_read_from_str(isl_ctx *ctx,
	const char *str)
{
	isl_pw_multi_aff *pma;
	struct isl_stream *s = isl_stream_new_str(ctx, str);
	if (!s)
		return NULL;
	pma = isl_stream_read_pw_multi_aff(s);
	isl_stream_free(s);
	return pma;
}

/* Read an isl_union_pw_multi_aff from "s".
 * We currently read a generic object and if it turns out to be a set or
 * a map, we convert that to an isl_union_pw_multi_aff.
 * It would be more efficient if we were to construct
 * the isl_union_pw_multi_aff directly.
 */
__isl_give isl_union_pw_multi_aff *isl_stream_read_union_pw_multi_aff(
	struct isl_stream *s)
{
	struct isl_obj obj;

	obj = obj_read(s);
	if (!obj.v)
		return NULL;

	if (obj.type == isl_obj_map || obj.type == isl_obj_set)
		obj = to_union(s->ctx, obj);
	if (obj.type == isl_obj_union_map)
		return isl_union_pw_multi_aff_from_union_map(obj.v);
	if (obj.type == isl_obj_union_set)
		return isl_union_pw_multi_aff_from_union_set(obj.v);

	obj.type->free(obj.v);
	isl_die(s->ctx, isl_error_invalid, "unexpected object type",
		return NULL);
}

/* Read an isl_union_pw_multi_aff from "str".
 */
__isl_give isl_union_pw_multi_aff *isl_union_pw_multi_aff_read_from_str(
	isl_ctx *ctx, const char *str)
{
	isl_union_pw_multi_aff *upma;
	struct isl_stream *s = isl_stream_new_str(ctx, str);
	if (!s)
		return NULL;
	upma = isl_stream_read_union_pw_multi_aff(s);
	isl_stream_free(s);
	return upma;
}

/* Assuming "pa" represents a single affine expression defined on a universe
 * domain, extract this affine expression.
 */
static __isl_give isl_aff *aff_from_pw_aff(__isl_take isl_pw_aff *pa)
{
	isl_aff *aff;

	if (!pa)
		return NULL;
	if (pa->n != 1)
		isl_die(isl_pw_aff_get_ctx(pa), isl_error_invalid,
			"expecting single affine expression",
			goto error);
	if (!isl_set_plain_is_universe(pa->p[0].set))
		isl_die(isl_pw_aff_get_ctx(pa), isl_error_invalid,
			"expecting universe domain",
			goto error);

	aff = isl_aff_copy(pa->p[0].aff);
	isl_pw_aff_free(pa);
	return aff;
error:
	isl_pw_aff_free(pa);
	return NULL;
}

/* Read a multi-affine expression from "s".
 * If the multi-affine expression has a domain, then the tuple
 * representing this domain cannot involve any affine expressions.
 * The tuple representing the actual expressions needs to consist
 * of only affine expressions.  Moreover, these expressions can
 * only depend on parameters and input dimensions and not on other
 * output dimensions.
 */
__isl_give isl_multi_aff *isl_stream_read_multi_aff(struct isl_stream *s)
{
	struct vars *v;
	isl_set *dom = NULL;
	isl_multi_pw_aff *tuple = NULL;
	int dim, i, n;
	isl_space *space, *dom_space;
	isl_multi_aff *ma = NULL;

	v = vars_new(s->ctx);
	if (!v)
		return NULL;

	dom = isl_set_universe(isl_space_params_alloc(s->ctx, 0));
	if (next_is_tuple(s)) {
		dom = read_map_tuple(s, dom, isl_dim_param, v, 1, 0);
		if (isl_stream_eat(s, ISL_TOKEN_TO))
			goto error;
	}
	if (!isl_set_plain_is_universe(dom))
		isl_die(s->ctx, isl_error_invalid,
			"expecting universe parameter domain", goto error);
	if (isl_stream_eat(s, '{'))
		goto error;

	tuple = read_tuple(s, v, 0, 0);
	if (!tuple)
		goto error;
	if (isl_stream_eat_if_available(s, ISL_TOKEN_TO)) {
		isl_set *set;
		isl_space *space;
		int has_expr;

		has_expr = tuple_has_expr(tuple);
		if (has_expr < 0)
			goto error;
		if (has_expr)
			isl_die(s->ctx, isl_error_invalid,
				"expecting universe domain", goto error);
		space = isl_space_range(isl_multi_pw_aff_get_space(tuple));
		set = isl_set_universe(space);
		dom = isl_set_intersect_params(set, dom);
		isl_multi_pw_aff_free(tuple);
		tuple = read_tuple(s, v, 0, 0);
		if (!tuple)
			goto error;
	}

	if (isl_stream_eat(s, '}'))
		goto error;

	n = isl_multi_pw_aff_dim(tuple, isl_dim_out);
	dim = isl_set_dim(dom, isl_dim_all);
	dom_space = isl_set_get_space(dom);
	space = isl_space_range(isl_multi_pw_aff_get_space(tuple));
	space = isl_space_align_params(space, isl_space_copy(dom_space));
	if (!isl_space_is_params(dom_space))
		space = isl_space_map_from_domain_and_range(
				isl_space_copy(dom_space), space);
	isl_space_free(dom_space);
	ma = isl_multi_aff_alloc(space);

	for (i = 0; i < n; ++i) {
		isl_pw_aff *pa;
		isl_aff *aff;
		pa = isl_multi_pw_aff_get_pw_aff(tuple, i);
		aff = aff_from_pw_aff(pa);
		if (!aff)
			goto error;
		if (isl_aff_involves_dims(aff, isl_dim_in, dim, i + 1)) {
			isl_aff_free(aff);
			isl_die(s->ctx, isl_error_invalid,
				"not an affine expression", goto error);
		}
		aff = isl_aff_drop_dims(aff, isl_dim_in, dim, n);
		space = isl_multi_aff_get_domain_space(ma);
		aff = isl_aff_reset_domain_space(aff, space);
		ma = isl_multi_aff_set_aff(ma, i, aff);
	}

	isl_multi_pw_aff_free(tuple);
	vars_free(v);
	isl_set_free(dom);
	return ma;
error:
	isl_multi_pw_aff_free(tuple);
	vars_free(v);
	isl_set_free(dom);
	isl_multi_aff_free(ma);
	return NULL;
}

__isl_give isl_multi_aff *isl_multi_aff_read_from_str(isl_ctx *ctx,
	const char *str)
{
	isl_multi_aff *maff;
	struct isl_stream *s = isl_stream_new_str(ctx, str);
	if (!s)
		return NULL;
	maff = isl_stream_read_multi_aff(s);
	isl_stream_free(s);
	return maff;
}

/* Read an isl_multi_pw_aff from "s".
 *
 * The input format is similar to that of map, except that any conditions
 * on the domains should be specified inside the tuple since each
 * piecewise affine expression may have a different domain.
 *
 * Since we do not know in advance if the isl_multi_pw_aff lives
 * in a set or a map space, we first read the first tuple and check
 * if it is followed by a "->".  If so, we convert the tuple into
 * the domain of the isl_multi_pw_aff and read in the next tuple.
 * This tuple (or the first tuple if it was not followed by a "->")
 * is then converted into the isl_multi_pw_aff.
 *
 * Note that the function read_tuple accepts tuples where some output or
 * set dimensions are defined in terms of other output or set dimensions
 * since this function is also used to read maps.  As a special case,
 * read_tuple also accept dimensions that are defined in terms of themselves
 * (i.e., that are not defined).
 * These cases are not allowed when reading am isl_multi_pw_aff so we check
 * that the definition of the output/set dimensions does not involve any
 * output/set dimensions.
 * We then drop the output dimensions from the domain of the result
 * of read_tuple (which is of the form [input, output] -> [output],
 * with anonymous domain) and reset the space.
 */
__isl_give isl_multi_pw_aff *isl_stream_read_multi_pw_aff(struct isl_stream *s)
{
	struct vars *v;
	isl_set *dom = NULL;
	isl_multi_pw_aff *tuple = NULL;
	int dim, i, n;
	isl_space *space, *dom_space;
	isl_multi_pw_aff *mpa = NULL;

	v = vars_new(s->ctx);
	if (!v)
		return NULL;

	dom = isl_set_universe(isl_space_params_alloc(s->ctx, 0));
	if (next_is_tuple(s)) {
		dom = read_map_tuple(s, dom, isl_dim_param, v, 1, 0);
		if (isl_stream_eat(s, ISL_TOKEN_TO))
			goto error;
	}
	if (isl_stream_eat(s, '{'))
		goto error;

	tuple = read_tuple(s, v, 0, 0);
	if (!tuple)
		goto error;
	if (isl_stream_eat_if_available(s, ISL_TOKEN_TO)) {
		isl_map *map = map_from_tuple(tuple, dom, isl_dim_in, v, 0);
		dom = isl_map_domain(map);
		tuple = read_tuple(s, v, 0, 0);
		if (!tuple)
			goto error;
	}

	if (isl_stream_eat(s, '}'))
		goto error;

	n = isl_multi_pw_aff_dim(tuple, isl_dim_out);
	dim = isl_set_dim(dom, isl_dim_all);
	dom_space = isl_set_get_space(dom);
	space = isl_space_range(isl_multi_pw_aff_get_space(tuple));
	space = isl_space_align_params(space, isl_space_copy(dom_space));
	if (!isl_space_is_params(dom_space))
		space = isl_space_map_from_domain_and_range(
				isl_space_copy(dom_space), space);
	isl_space_free(dom_space);
	mpa = isl_multi_pw_aff_alloc(space);

	for (i = 0; i < n; ++i) {
		isl_pw_aff *pa;
		pa = isl_multi_pw_aff_get_pw_aff(tuple, i);
		if (!pa)
			goto error;
		if (isl_pw_aff_involves_dims(pa, isl_dim_in, dim, i + 1)) {
			isl_pw_aff_free(pa);
			isl_die(s->ctx, isl_error_invalid,
				"not an affine expression", goto error);
		}
		pa = isl_pw_aff_drop_dims(pa, isl_dim_in, dim, n);
		space = isl_multi_pw_aff_get_domain_space(mpa);
		pa = isl_pw_aff_reset_domain_space(pa, space);
		mpa = isl_multi_pw_aff_set_pw_aff(mpa, i, pa);
	}

	isl_multi_pw_aff_free(tuple);
	vars_free(v);
	mpa = isl_multi_pw_aff_intersect_domain(mpa, dom);
	return mpa;
error:
	isl_multi_pw_aff_free(tuple);
	vars_free(v);
	isl_set_free(dom);
	isl_multi_pw_aff_free(mpa);
	return NULL;
}

/* Read an isl_multi_pw_aff from "str".
 */
__isl_give isl_multi_pw_aff *isl_multi_pw_aff_read_from_str(isl_ctx *ctx,
	const char *str)
{
	isl_multi_pw_aff *mpa;
	struct isl_stream *s = isl_stream_new_str(ctx, str);
	if (!s)
		return NULL;
	mpa = isl_stream_read_multi_pw_aff(s);
	isl_stream_free(s);
	return mpa;
}

__isl_give isl_union_pw_qpolynomial *isl_stream_read_union_pw_qpolynomial(
	struct isl_stream *s)
{
	struct isl_obj obj;

	obj = obj_read(s);
	if (obj.type == isl_obj_pw_qpolynomial) {
		obj.type = isl_obj_union_pw_qpolynomial;
		obj.v = isl_union_pw_qpolynomial_from_pw_qpolynomial(obj.v);
	}
	if (obj.v)
		isl_assert(s->ctx, obj.type == isl_obj_union_pw_qpolynomial,
			   goto error);

	return obj.v;
error:
	obj.type->free(obj.v);
	return NULL;
}

__isl_give isl_union_pw_qpolynomial *isl_union_pw_qpolynomial_read_from_str(
	isl_ctx *ctx, const char *str)
{
	isl_union_pw_qpolynomial *upwqp;
	struct isl_stream *s = isl_stream_new_str(ctx, str);
	if (!s)
		return NULL;
	upwqp = isl_stream_read_union_pw_qpolynomial(s);
	isl_stream_free(s);
	return upwqp;
}
