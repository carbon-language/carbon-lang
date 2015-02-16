#ifndef ISL_SCHEDULE_TYPE_H
#define ISL_SCHEDULE_TYPE_H

#if defined(__cplusplus)
extern "C" {
#endif

enum isl_schedule_node_type {
	isl_schedule_node_error = -1,
	isl_schedule_node_band,
	isl_schedule_node_domain,
	isl_schedule_node_filter,
	isl_schedule_node_leaf,
	isl_schedule_node_sequence,
	isl_schedule_node_set
};

struct isl_schedule_node;
typedef struct isl_schedule_node isl_schedule_node;

struct isl_schedule;
typedef struct isl_schedule isl_schedule;

#if defined(__cplusplus)
}
#endif

#endif
