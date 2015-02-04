#ifndef ISL_SCHEDULE_H
#define ISL_SCHEDULE_H

#include <isl/union_set_type.h>
#include <isl/union_map_type.h>
#include <isl/band.h>
#include <isl/list.h>

#if defined(__cplusplus)
extern "C" {
#endif

struct isl_schedule_constraints;
typedef struct isl_schedule_constraints isl_schedule_constraints;
struct isl_schedule;
typedef struct isl_schedule isl_schedule;

int isl_options_set_schedule_max_coefficient(isl_ctx *ctx, int val);
int isl_options_get_schedule_max_coefficient(isl_ctx *ctx);

int isl_options_set_schedule_max_constant_term(isl_ctx *ctx, int val);
int isl_options_get_schedule_max_constant_term(isl_ctx *ctx);

int isl_options_set_schedule_maximize_band_depth(isl_ctx *ctx, int val);
int isl_options_get_schedule_maximize_band_depth(isl_ctx *ctx);

int isl_options_set_schedule_outer_coincidence(isl_ctx *ctx, int val);
int isl_options_get_schedule_outer_coincidence(isl_ctx *ctx);

int isl_options_set_schedule_split_scaled(isl_ctx *ctx, int val);
int isl_options_get_schedule_split_scaled(isl_ctx *ctx);

int isl_options_set_schedule_separate_components(isl_ctx *ctx, int val);
int isl_options_get_schedule_separate_components(isl_ctx *ctx);

#define		ISL_SCHEDULE_FUSE_MAX			0
#define		ISL_SCHEDULE_FUSE_MIN			1
int isl_options_set_schedule_fuse(isl_ctx *ctx, int val);
int isl_options_get_schedule_fuse(isl_ctx *ctx);

__isl_give isl_schedule_constraints *isl_schedule_constraints_copy(
	__isl_keep isl_schedule_constraints *sc);
__isl_give isl_schedule_constraints *isl_schedule_constraints_on_domain(
	__isl_take isl_union_set *domain);
__isl_give isl_schedule_constraints *isl_schedule_constraints_set_validity(
	__isl_take isl_schedule_constraints *sc,
	__isl_take isl_union_map *validity);
__isl_give isl_schedule_constraints *isl_schedule_constraints_set_coincidence(
	__isl_take isl_schedule_constraints *sc,
	__isl_take isl_union_map *coincidence);
__isl_give isl_schedule_constraints *isl_schedule_constraints_set_proximity(
	__isl_take isl_schedule_constraints *sc,
	__isl_take isl_union_map *proximity);
__isl_give isl_schedule_constraints *
isl_schedule_constraints_set_conditional_validity(
	__isl_take isl_schedule_constraints *sc,
	__isl_take isl_union_map *condition,
	__isl_take isl_union_map *validity);
__isl_null isl_schedule_constraints *isl_schedule_constraints_free(
	__isl_take isl_schedule_constraints *sc);

isl_ctx *isl_schedule_constraints_get_ctx(
	__isl_keep isl_schedule_constraints *sc);

void isl_schedule_constraints_dump(__isl_keep isl_schedule_constraints *sc);

__isl_give isl_schedule *isl_schedule_constraints_compute_schedule(
	__isl_take isl_schedule_constraints *sc);

__isl_give isl_schedule *isl_union_set_compute_schedule(
	__isl_take isl_union_set *domain,
	__isl_take isl_union_map *validity,
	__isl_take isl_union_map *proximity);
__isl_null isl_schedule *isl_schedule_free(__isl_take isl_schedule *sched);
__isl_give isl_union_map *isl_schedule_get_map(__isl_keep isl_schedule *sched);

isl_ctx *isl_schedule_get_ctx(__isl_keep isl_schedule *sched);

__isl_give isl_band_list *isl_schedule_get_band_forest(
	__isl_keep isl_schedule *schedule);

__isl_give isl_printer *isl_printer_print_schedule(__isl_take isl_printer *p,
	__isl_keep isl_schedule *schedule);
void isl_schedule_dump(__isl_keep isl_schedule *schedule);

int isl_schedule_foreach_band(__isl_keep isl_schedule *sched,
	int (*fn)(__isl_keep isl_band *band, void *user), void *user);

#if defined(__cplusplus)
}
#endif

#endif
