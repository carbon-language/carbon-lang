#ifndef __OMPT_INTERNAL_H__
#define __OMPT_INTERNAL_H__

#include "ompt.h"
#include "ompt-event-specific.h"

#define OMPT_VERSION 1

#define _OMP_EXTERN extern "C"

#define OMPT_INVOKER(x) \
  ((x == fork_context_gnu) ? ompt_invoker_program : ompt_invoker_runtime)


#define ompt_callback(e) e ## _callback

/* track and track_callback share a bit so that one can test whether either is
 * set by anding a bit.
 */
typedef enum {
    ompt_status_disabled       = 0x0,
    ompt_status_ready          = 0x1,
    ompt_status_track          = 0x2,
    ompt_status_track_callback = 0x6,
} ompt_status_t;


typedef struct ompt_callbacks_s {
#define ompt_event_macro(event, callback, eventid) callback ompt_callback(event);

    FOREACH_OMPT_EVENT(ompt_event_macro)

#undef ompt_event_macro
} ompt_callbacks_t;



typedef struct {
    ompt_frame_t        frame;
    void*               function;
    ompt_task_id_t      task_id;
} ompt_task_info_t;


typedef struct {
    ompt_parallel_id_t  parallel_id;
    void                *microtask;
} ompt_team_info_t;


typedef struct ompt_lw_taskteam_s {
    ompt_team_info_t    ompt_team_info;
    ompt_task_info_t    ompt_task_info;
    struct ompt_lw_taskteam_s *parent;
} ompt_lw_taskteam_t;


typedef struct ompt_parallel_info_s {
    ompt_task_id_t parent_task_id;    /* id of parent task            */
    ompt_parallel_id_t parallel_id;   /* id of parallel region        */
    ompt_frame_t *parent_task_frame;  /* frame data of parent task    */
    void *parallel_function;          /* pointer to outlined function */
} ompt_parallel_info_t;


typedef struct {
    ompt_state_t        state;
    ompt_wait_id_t      wait_id;
    void                *idle_frame;
} ompt_thread_info_t;


extern ompt_status_t ompt_status;
extern ompt_callbacks_t ompt_callbacks;

#ifdef __cplusplus
extern "C" {
#endif

void ompt_init(void);
void ompt_fini(void);

#ifdef __cplusplus
};
#endif

#endif
