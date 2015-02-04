/*
 * Copyright 2008-2009 Katholieke Universiteit Leuven
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, K.U.Leuven, Departement
 * Computerwetenschappen, Celestijnenlaan 200A, B-3001 Leuven, Belgium
 */

#ifndef ISL_STREAM_H
#define ISL_STREAM_H

#include <stdio.h>
#include <isl/hash.h>
#include <isl/aff_type.h>
#include <isl/obj.h>
#include <isl/val.h>

#if defined(__cplusplus)
extern "C" {
#endif

enum isl_token_type { ISL_TOKEN_ERROR = -1,
			ISL_TOKEN_UNKNOWN = 256, ISL_TOKEN_VALUE,
			ISL_TOKEN_IDENT, ISL_TOKEN_GE,
			ISL_TOKEN_LE, ISL_TOKEN_GT, ISL_TOKEN_LT,
			ISL_TOKEN_NE, ISL_TOKEN_EQ_EQ,
			ISL_TOKEN_LEX_GE, ISL_TOKEN_LEX_LE,
			ISL_TOKEN_LEX_GT, ISL_TOKEN_LEX_LT,
			ISL_TOKEN_TO, ISL_TOKEN_AND,
			ISL_TOKEN_OR, ISL_TOKEN_EXISTS, ISL_TOKEN_NOT,
			ISL_TOKEN_DEF, ISL_TOKEN_INFTY, ISL_TOKEN_NAN,
			ISL_TOKEN_MIN, ISL_TOKEN_MAX, ISL_TOKEN_RAT,
			ISL_TOKEN_TRUE, ISL_TOKEN_FALSE,
			ISL_TOKEN_CEILD, ISL_TOKEN_FLOORD, ISL_TOKEN_MOD,
			ISL_TOKEN_STRING,
			ISL_TOKEN_MAP, ISL_TOKEN_AFF,
			ISL_TOKEN_CEIL, ISL_TOKEN_FLOOR,
			ISL_TOKEN_IMPLIES,
			ISL_TOKEN_LAST };

struct isl_token;

__isl_give isl_val *isl_token_get_val(isl_ctx *ctx, struct isl_token *tok);
__isl_give char *isl_token_get_str(isl_ctx *ctx, struct isl_token *tok);
int isl_token_get_type(struct isl_token *tok);
void isl_token_free(struct isl_token *tok);

struct isl_stream {
	struct isl_ctx	*ctx;
	FILE        	*file;
	const char  	*str;
	int	    	line;
	int	    	col;
	int	    	eof;

	char	    	*buffer;
	size_t	    	size;
	size_t	    	len;
	int	    	c;
	int		un[5];
	int		n_un;

	struct isl_token	*tokens[5];
	int	    	n_token;

	struct isl_hash_table	*keywords;
	enum isl_token_type	 next_type;
};

struct isl_stream* isl_stream_new_file(struct isl_ctx *ctx, FILE *file);
struct isl_stream* isl_stream_new_str(struct isl_ctx *ctx, const char *str);
void isl_stream_free(struct isl_stream *s);

void isl_stream_error(struct isl_stream *s, struct isl_token *tok, char *msg);

struct isl_token *isl_stream_next_token(struct isl_stream *s);
struct isl_token *isl_stream_next_token_on_same_line(struct isl_stream *s);
int isl_stream_next_token_is(struct isl_stream *s, int type);
void isl_stream_push_token(struct isl_stream *s, struct isl_token *tok);
void isl_stream_flush_tokens(struct isl_stream *s);
int isl_stream_eat_if_available(struct isl_stream *s, int type);
char *isl_stream_read_ident_if_available(struct isl_stream *s);
int isl_stream_eat(struct isl_stream *s, int type);
int isl_stream_is_empty(struct isl_stream *s);
int isl_stream_skip_line(struct isl_stream *s);

enum isl_token_type isl_stream_register_keyword(struct isl_stream *s,
	const char *name);

struct isl_obj isl_stream_read_obj(struct isl_stream *s);
__isl_give isl_multi_aff *isl_stream_read_multi_aff(struct isl_stream *s);
__isl_give isl_map *isl_stream_read_map(struct isl_stream *s);
__isl_give isl_set *isl_stream_read_set(struct isl_stream *s);
__isl_give isl_pw_qpolynomial *isl_stream_read_pw_qpolynomial(
	struct isl_stream *s);
__isl_give isl_union_map *isl_stream_read_union_map(struct isl_stream *s);

#if defined(__cplusplus)
}
#endif

#endif
