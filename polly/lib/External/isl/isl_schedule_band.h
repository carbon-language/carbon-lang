#ifndef ISL_SCHEDULE_BAND_H
#define ISL_SCHEDULE_BAND_H

#include <isl/aff.h>
#include <isl/union_map.h>

/* Information about a band within a schedule.
 *
 * n is the number of scheduling dimensions within the band.
 * coincident is an array of length n, indicating whether a scheduling dimension
 *	satisfies the coincidence constraints in the sense that
 *	the corresponding dependence distances are zero.
 * permutable is set if the band is permutable.
 * mupa is the partial schedule corresponding to this band.  The dimension
 *	of mupa is equal to n.
 */
struct isl_schedule_band {
	int ref;

	int n;
	int *coincident;
	int permutable;

	isl_multi_union_pw_aff *mupa;
};
typedef struct isl_schedule_band isl_schedule_band;

__isl_give isl_schedule_band *isl_schedule_band_from_multi_union_pw_aff(
	__isl_take isl_multi_union_pw_aff *mupa);
__isl_give isl_schedule_band *isl_schedule_band_copy(
	__isl_keep isl_schedule_band *band);
__isl_null isl_schedule_band *isl_schedule_band_free(
	__isl_take isl_schedule_band *band);

isl_ctx *isl_schedule_band_get_ctx(__isl_keep isl_schedule_band *band);

__isl_give isl_space *isl_schedule_band_get_space(
	__isl_keep isl_schedule_band *band);
__isl_give isl_multi_union_pw_aff *isl_schedule_band_get_partial_schedule(
	__isl_keep isl_schedule_band *band);

int isl_schedule_band_n_member(__isl_keep isl_schedule_band *band);
int isl_schedule_band_member_get_coincident(
	__isl_keep isl_schedule_band *band, int pos);
__isl_give isl_schedule_band *isl_schedule_band_member_set_coincident(
	__isl_take isl_schedule_band *band, int pos, int coincident);
int isl_schedule_band_get_permutable(__isl_keep isl_schedule_band *band);
__isl_give isl_schedule_band *isl_schedule_band_set_permutable(
	__isl_take isl_schedule_band *band, int permutable);

__isl_give isl_schedule_band *isl_schedule_band_scale(
	__isl_take isl_schedule_band *band, __isl_take isl_multi_val *mv);
__isl_give isl_schedule_band *isl_schedule_band_scale_down(
	__isl_take isl_schedule_band *band, __isl_take isl_multi_val *mv);
__isl_give isl_schedule_band *isl_schedule_band_tile(
	__isl_take isl_schedule_band *band, __isl_take isl_multi_val *sizes);
__isl_give isl_schedule_band *isl_schedule_band_point(
	__isl_take isl_schedule_band *band, __isl_keep isl_schedule_band *tile,
	__isl_take isl_multi_val *sizes);
__isl_give isl_schedule_band *isl_schedule_band_drop(
	__isl_take isl_schedule_band *band, int pos, int n);

#endif
