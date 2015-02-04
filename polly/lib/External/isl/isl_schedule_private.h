#ifndef ISL_SCHEDLUE_PRIVATE_H
#define ISL_SCHEDLUE_PRIVATE_H

#include <isl/aff.h>
#include <isl/schedule.h>

/* The schedule for an individual domain, plus information about the bands
 * and scheduling dimensions.
 * In particular, we keep track of the number of bands and for each
 * band, the starting position of the next band.  The first band starts at
 * position 0.
 * For each scheduling dimension, we keep track of whether it satisfies
 * the coincidence constraints (within its band).
 */
struct isl_schedule_node {
	isl_multi_aff *sched;
	int	 n_band;
	int	*band_end;
	int	*band_id;
	int	*coincident;
};

/* Information about the computed schedule.
 * n is the number of nodes/domains/statements.
 * n_band is the maximal number of bands.
 * n_total_row is the number of coordinates of the schedule.
 * dim contains a description of the parameters.
 * band_forest points to a band forest representation of the schedule
 * and may be NULL if the forest hasn't been created yet.
 */
struct isl_schedule {
	int ref;

	int n;
	int n_band;
	int n_total_row;
	isl_space *dim;

	isl_band_list *band_forest;

	struct isl_schedule_node node[1];
};

#endif
