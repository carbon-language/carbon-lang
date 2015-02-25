/*
 * Copyright 2013-2014 Ecole Normale Superieure
 * Copyright 2014      INRIA Rocquencourt
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * Ecole Normale Superieure, 45 rue d'Ulm, 75230 Paris, France
 * and Inria Paris - Rocquencourt, Domaine de Voluceau - Rocquencourt,
 * B.P. 105 - 78153 Le Chesnay, France
 */

#include <isl/schedule_node.h>
#include <isl_schedule_band.h>
#include <isl_schedule_private.h>

isl_ctx *isl_schedule_band_get_ctx(__isl_keep isl_schedule_band *band)
{
	return band ? isl_multi_union_pw_aff_get_ctx(band->mupa) : NULL;
}

/* Return a new uninitialized isl_schedule_band.
 */
static __isl_give isl_schedule_band *isl_schedule_band_alloc(isl_ctx *ctx)
{
	isl_schedule_band *band;

	band = isl_calloc_type(ctx, isl_schedule_band);
	if (!band)
		return NULL;

	band->ref = 1;

	return band;
}

/* Return a new isl_schedule_band with partial schedule "mupa".
 * First replace "mupa" by its greatest integer part to ensure
 * that the schedule is always integral.
 * The band is not marked permutable and the dimensions are not
 * marked coincident.
 */
__isl_give isl_schedule_band *isl_schedule_band_from_multi_union_pw_aff(
	__isl_take isl_multi_union_pw_aff *mupa)
{
	isl_ctx *ctx;
	isl_schedule_band *band;

	mupa = isl_multi_union_pw_aff_floor(mupa);
	if (!mupa)
		return NULL;
	ctx = isl_multi_union_pw_aff_get_ctx(mupa);
	band = isl_schedule_band_alloc(ctx);
	if (!band)
		goto error;

	band->n = isl_multi_union_pw_aff_dim(mupa, isl_dim_set);
	band->coincident = isl_calloc_array(ctx, int, band->n);
	band->mupa = mupa;

	if (band->n && !band->coincident)
		return isl_schedule_band_free(band);

	return band;
error:
	isl_multi_union_pw_aff_free(mupa);
	return NULL;
}

/* Create a duplicate of the given isl_schedule_band.
 */
__isl_give isl_schedule_band *isl_schedule_band_dup(
	__isl_keep isl_schedule_band *band)
{
	int i;
	isl_ctx *ctx;
	isl_schedule_band *dup;

	if (!band)
		return NULL;

	ctx = isl_schedule_band_get_ctx(band);
	dup = isl_schedule_band_alloc(ctx);
	if (!dup)
		return NULL;

	dup->n = band->n;
	dup->coincident = isl_alloc_array(ctx, int, band->n);
	if (band->n && !dup->coincident)
		return isl_schedule_band_free(dup);

	for (i = 0; i < band->n; ++i)
		dup->coincident[i] = band->coincident[i];
	dup->permutable = band->permutable;

	dup->mupa = isl_multi_union_pw_aff_copy(band->mupa);
	if (!dup->mupa)
		return isl_schedule_band_free(dup);

	return dup;
}

/* Return an isl_schedule_band that is equal to "band" and that has only
 * a single reference.
 */
__isl_give isl_schedule_band *isl_schedule_band_cow(
	__isl_take isl_schedule_band *band)
{
	if (!band)
		return NULL;

	if (band->ref == 1)
		return band;
	band->ref--;
	return isl_schedule_band_dup(band);
}

/* Return a new reference to "band".
 */
__isl_give isl_schedule_band *isl_schedule_band_copy(
	__isl_keep isl_schedule_band *band)
{
	if (!band)
		return NULL;

	band->ref++;
	return band;
}

/* Free a reference to "band" and return NULL.
 */
__isl_null isl_schedule_band *isl_schedule_band_free(
	__isl_take isl_schedule_band *band)
{
	if (!band)
		return NULL;

	if (--band->ref > 0)
		return NULL;

	isl_multi_union_pw_aff_free(band->mupa);
	free(band->coincident);
	free(band);

	return NULL;
}

/* Are "band1" and "band2" obviously equal?
 */
int isl_schedule_band_plain_is_equal(__isl_keep isl_schedule_band *band1,
	__isl_keep isl_schedule_band *band2)
{
	int i;

	if (!band1 || !band2)
		return -1;
	if (band1 == band2)
		return 1;

	if (band1->n != band2->n)
		return 0;
	for (i = 0; i < band1->n; ++i)
		if (band1->coincident[i] != band2->coincident[i])
			return 0;
	if (band1->permutable != band2->permutable)
		return 0;

	return isl_multi_union_pw_aff_plain_is_equal(band1->mupa, band2->mupa);
}

/* Return the number of scheduling dimensions in the band.
 */
int isl_schedule_band_n_member(__isl_keep isl_schedule_band *band)
{
	return band ? band->n : 0;
}

/* Is the given scheduling dimension coincident within the band and
 * with respect to the coincidence constraints?
 */
int isl_schedule_band_member_get_coincident(__isl_keep isl_schedule_band *band,
	int pos)
{
	if (!band)
		return -1;

	if (pos < 0 || pos >= band->n)
		isl_die(isl_schedule_band_get_ctx(band), isl_error_invalid,
			"invalid member position", return -1);

	return band->coincident[pos];
}

/* Mark the given scheduling dimension as being coincident or not
 * according to "coincident".
 */
__isl_give isl_schedule_band *isl_schedule_band_member_set_coincident(
	__isl_take isl_schedule_band *band, int pos, int coincident)
{
	if (!band)
		return NULL;
	if (isl_schedule_band_member_get_coincident(band, pos) == coincident)
		return band;
	band = isl_schedule_band_cow(band);
	if (!band)
		return NULL;

	if (pos < 0 || pos >= band->n)
		isl_die(isl_schedule_band_get_ctx(band), isl_error_invalid,
			"invalid member position",
			isl_schedule_band_free(band));

	band->coincident[pos] = coincident;

	return band;
}

/* Is the schedule band mark permutable?
 */
int isl_schedule_band_get_permutable(__isl_keep isl_schedule_band *band)
{
	if (!band)
		return -1;
	return band->permutable;
}

/* Mark the schedule band permutable or not according to "permutable"?
 */
__isl_give isl_schedule_band *isl_schedule_band_set_permutable(
	__isl_take isl_schedule_band *band, int permutable)
{
	if (!band)
		return NULL;
	if (band->permutable == permutable)
		return band;
	band = isl_schedule_band_cow(band);
	if (!band)
		return NULL;

	band->permutable = permutable;

	return band;
}

/* Return the schedule space of the band.
 */
__isl_give isl_space *isl_schedule_band_get_space(
	__isl_keep isl_schedule_band *band)
{
	if (!band)
		return NULL;
	return isl_multi_union_pw_aff_get_space(band->mupa);
}

/* Return the schedule of the band in isolation.
 */
__isl_give isl_multi_union_pw_aff *isl_schedule_band_get_partial_schedule(
	__isl_keep isl_schedule_band *band)
{
	return band ? isl_multi_union_pw_aff_copy(band->mupa) : NULL;
}

/* Multiply the partial schedule of "band" with the factors in "mv".
 * Replace the result by its greatest integer part to ensure
 * that the schedule is always integral.
 */
__isl_give isl_schedule_band *isl_schedule_band_scale(
	__isl_take isl_schedule_band *band, __isl_take isl_multi_val *mv)
{
	band = isl_schedule_band_cow(band);
	if (!band || !mv)
		goto error;
	band->mupa = isl_multi_union_pw_aff_scale_multi_val(band->mupa, mv);
	band->mupa = isl_multi_union_pw_aff_floor(band->mupa);
	if (!band->mupa)
		return isl_schedule_band_free(band);
	return band;
error:
	isl_schedule_band_free(band);
	isl_multi_val_free(mv);
	return NULL;
}

/* Divide the partial schedule of "band" by the factors in "mv".
 * Replace the result by its greatest integer part to ensure
 * that the schedule is always integral.
 */
__isl_give isl_schedule_band *isl_schedule_band_scale_down(
	__isl_take isl_schedule_band *band, __isl_take isl_multi_val *mv)
{
	band = isl_schedule_band_cow(band);
	if (!band || !mv)
		goto error;
	band->mupa = isl_multi_union_pw_aff_scale_down_multi_val(band->mupa,
								mv);
	band->mupa = isl_multi_union_pw_aff_floor(band->mupa);
	if (!band->mupa)
		return isl_schedule_band_free(band);
	return band;
error:
	isl_schedule_band_free(band);
	isl_multi_val_free(mv);
	return NULL;
}

/* Given the schedule of a band, construct the corresponding
 * schedule for the tile loops based on the given tile sizes
 * and return the result.
 *
 * If the scale tile loops options is set, then the tile loops
 * are scaled by the tile sizes.
 *
 * That is replace each schedule dimension "i" by either
 * "floor(i/s)" or "s * floor(i/s)".
 */
static isl_multi_union_pw_aff *isl_multi_union_pw_aff_tile(
	__isl_take isl_multi_union_pw_aff *sched,
	__isl_take isl_multi_val *sizes)
{
	isl_ctx *ctx;
	int i, n;
	isl_val *v;
	int scale;

	ctx = isl_multi_val_get_ctx(sizes);
	scale = isl_options_get_tile_scale_tile_loops(ctx);

	n = isl_multi_union_pw_aff_dim(sched, isl_dim_set);
	for (i = 0; i < n; ++i) {
		isl_union_pw_aff *upa;

		upa = isl_multi_union_pw_aff_get_union_pw_aff(sched, i);
		v = isl_multi_val_get_val(sizes, i);

		upa = isl_union_pw_aff_scale_down_val(upa, isl_val_copy(v));
		upa = isl_union_pw_aff_floor(upa);
		if (scale)
			upa = isl_union_pw_aff_scale_val(upa, isl_val_copy(v));
		isl_val_free(v);

		sched = isl_multi_union_pw_aff_set_union_pw_aff(sched, i, upa);
	}

	isl_multi_val_free(sizes);
	return sched;
}

/* Replace "band" by a band corresponding to the tile loops of a tiling
 * with the given tile sizes.
 */
__isl_give isl_schedule_band *isl_schedule_band_tile(
	__isl_take isl_schedule_band *band, __isl_take isl_multi_val *sizes)
{
	band = isl_schedule_band_cow(band);
	if (!band || !sizes)
		goto error;
	band->mupa = isl_multi_union_pw_aff_tile(band->mupa, sizes);
	if (!band->mupa)
		return isl_schedule_band_free(band);
	return band;
error:
	isl_schedule_band_free(band);
	isl_multi_val_free(sizes);
	return NULL;
}

/* Replace "band" by a band corresponding to the point loops of a tiling
 * with the given tile sizes.
 * "tile" is the corresponding tile loop band.
 *
 * If the shift point loops option is set, then the point loops
 * are shifted to start at zero.  That is, each schedule dimension "i"
 * is replaced by "i - s * floor(i/s)".
 * The expression "floor(i/s)" (or "s * floor(i/s)") is extracted from
 * the tile band.
 *
 * Otherwise, the band is left untouched.
 */
__isl_give isl_schedule_band *isl_schedule_band_point(
	__isl_take isl_schedule_band *band, __isl_keep isl_schedule_band *tile,
	__isl_take isl_multi_val *sizes)
{
	isl_ctx *ctx;
	isl_multi_union_pw_aff *scaled;

	if (!band || !sizes)
		goto error;

	ctx = isl_schedule_band_get_ctx(band);
	if (!isl_options_get_tile_shift_point_loops(ctx)) {
		isl_multi_val_free(sizes);
		return band;
	}
	band = isl_schedule_band_cow(band);
	if (!band)
		goto error;

	scaled = isl_schedule_band_get_partial_schedule(tile);
	if (!isl_options_get_tile_scale_tile_loops(ctx))
		scaled = isl_multi_union_pw_aff_scale_multi_val(scaled, sizes);
	else
		isl_multi_val_free(sizes);
	band->mupa = isl_multi_union_pw_aff_sub(band->mupa, scaled);
	if (!band->mupa)
		return isl_schedule_band_free(band);
	return band;
error:
	isl_schedule_band_free(band);
	isl_multi_val_free(sizes);
	return NULL;
}

/* Drop the "n" dimensions starting at "pos" from "band".
 *
 * We apply the transformation even if "n" is zero to ensure consistent
 * behavior with respect to changes in the schedule space.
 */
__isl_give isl_schedule_band *isl_schedule_band_drop(
	__isl_take isl_schedule_band *band, int pos, int n)
{
	int i;

	if (pos < 0 || n < 0 || pos + n > band->n)
		isl_die(isl_schedule_band_get_ctx(band), isl_error_internal,
			"range out of bounds",
			return isl_schedule_band_free(band));

	band = isl_schedule_band_cow(band);
	if (!band)
		return NULL;

	band->mupa = isl_multi_union_pw_aff_drop_dims(band->mupa,
							isl_dim_set, pos, n);
	if (!band->mupa)
		return isl_schedule_band_free(band);

	for (i = pos + n; i < band->n; ++i)
		band->coincident[i - n] = band->coincident[i];

	band->n -= n;

	return band;
}

/* Reset the user pointer on all identifiers of parameters and tuples
 * in "band".
 */
__isl_give isl_schedule_band *isl_schedule_band_reset_user(
	__isl_take isl_schedule_band *band)
{
	band = isl_schedule_band_cow(band);
	if (!band)
		return NULL;

	band->mupa = isl_multi_union_pw_aff_reset_user(band->mupa);
	if (!band->mupa)
		return isl_schedule_band_free(band);

	return band;
}

/* Align the parameters of "band" to those of "space".
 */
__isl_give isl_schedule_band *isl_schedule_band_align_params(
	__isl_take isl_schedule_band *band, __isl_take isl_space *space)
{
	band = isl_schedule_band_cow(band);
	if (!band || !space)
		goto error;

	band->mupa = isl_multi_union_pw_aff_align_params(band->mupa, space);
	if (!band->mupa)
		return isl_schedule_band_free(band);

	return band;
error:
	isl_space_free(space);
	isl_schedule_band_free(band);
	return NULL;
}

/* Compute the pullback of "band" by the function represented by "upma".
 * In other words, plug in "upma" in the iteration domains of "band".
 */
__isl_give isl_schedule_band *isl_schedule_band_pullback_union_pw_multi_aff(
	__isl_take isl_schedule_band *band,
	__isl_take isl_union_pw_multi_aff *upma)
{
	band = isl_schedule_band_cow(band);
	if (!band || !upma)
		goto error;

	band->mupa =
		isl_multi_union_pw_aff_pullback_union_pw_multi_aff(band->mupa,
									upma);
	if (!band->mupa)
		return isl_schedule_band_free(band);

	return band;
error:
	isl_union_pw_multi_aff_free(upma);
	isl_schedule_band_free(band);
	return NULL;
}

/* Compute the gist of "band" with respect to "context".
 * In particular, compute the gist of the associated partial schedule.
 */
__isl_give isl_schedule_band *isl_schedule_band_gist(
	__isl_take isl_schedule_band *band, __isl_take isl_union_set *context)
{
	if (!band || !context)
		goto error;
	if (band->n == 0) {
		isl_union_set_free(context);
		return band;
	}
	band = isl_schedule_band_cow(band);
	if (!band)
		goto error;
	band->mupa = isl_multi_union_pw_aff_gist(band->mupa, context);
	if (!band->mupa)
		return isl_schedule_band_free(band);
	return band;
error:
	isl_union_set_free(context);
	isl_schedule_band_free(band);
	return NULL;
}
