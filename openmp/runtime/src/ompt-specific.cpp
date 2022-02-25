/*
 * ompt-specific.cpp -- OMPT internal functions
 */

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//******************************************************************************
// include files
//******************************************************************************

#include "kmp.h"
#include "ompt-specific.h"

#if KMP_OS_UNIX
#include <dlfcn.h>
#endif

#if KMP_OS_WINDOWS
#define THREAD_LOCAL __declspec(thread)
#else
#define THREAD_LOCAL __thread
#endif

#define OMPT_WEAK_ATTRIBUTE KMP_WEAK_ATTRIBUTE_INTERNAL

//******************************************************************************
// macros
//******************************************************************************

#define LWT_FROM_TEAM(team) (team)->t.ompt_serialized_team_info

#define OMPT_THREAD_ID_BITS 16

//******************************************************************************
// private operations
//******************************************************************************

//----------------------------------------------------------
// traverse the team and task hierarchy
// note: __ompt_get_teaminfo and __ompt_get_task_info_object
//       traverse the hierarchy similarly and need to be
//       kept consistent
//----------------------------------------------------------

ompt_team_info_t *__ompt_get_teaminfo(int depth, int *size) {
  kmp_info_t *thr = ompt_get_thread();

  if (thr) {
    kmp_team *team = thr->th.th_team;
    if (team == NULL)
      return NULL;

    ompt_lw_taskteam_t *next_lwt = LWT_FROM_TEAM(team), *lwt = NULL;

    while (depth > 0) {
      // next lightweight team (if any)
      if (lwt)
        lwt = lwt->parent;

      // next heavyweight team (if any) after
      // lightweight teams are exhausted
      if (!lwt && team) {
        if (next_lwt) {
          lwt = next_lwt;
          next_lwt = NULL;
        } else {
          team = team->t.t_parent;
          if (team) {
            next_lwt = LWT_FROM_TEAM(team);
          }
        }
      }

      depth--;
    }

    if (lwt) {
      // lightweight teams have one task
      if (size)
        *size = 1;

      // return team info for lightweight team
      return &lwt->ompt_team_info;
    } else if (team) {
      // extract size from heavyweight team
      if (size)
        *size = team->t.t_nproc;

      // return team info for heavyweight team
      return &team->t.ompt_team_info;
    }
  }

  return NULL;
}

ompt_task_info_t *__ompt_get_task_info_object(int depth) {
  ompt_task_info_t *info = NULL;
  kmp_info_t *thr = ompt_get_thread();

  if (thr) {
    kmp_taskdata_t *taskdata = thr->th.th_current_task;
    ompt_lw_taskteam_t *lwt = NULL,
                       *next_lwt = LWT_FROM_TEAM(taskdata->td_team);

    while (depth > 0) {
      // next lightweight team (if any)
      if (lwt)
        lwt = lwt->parent;

      // next heavyweight team (if any) after
      // lightweight teams are exhausted
      if (!lwt && taskdata) {
        if (next_lwt) {
          lwt = next_lwt;
          next_lwt = NULL;
        } else {
          taskdata = taskdata->td_parent;
          if (taskdata) {
            next_lwt = LWT_FROM_TEAM(taskdata->td_team);
          }
        }
      }
      depth--;
    }

    if (lwt) {
      info = &lwt->ompt_task_info;
    } else if (taskdata) {
      info = &taskdata->ompt_task_info;
    }
  }

  return info;
}

ompt_task_info_t *__ompt_get_scheduling_taskinfo(int depth) {
  ompt_task_info_t *info = NULL;
  kmp_info_t *thr = ompt_get_thread();

  if (thr) {
    kmp_taskdata_t *taskdata = thr->th.th_current_task;

    ompt_lw_taskteam_t *lwt = NULL,
                       *next_lwt = LWT_FROM_TEAM(taskdata->td_team);

    while (depth > 0) {
      // next lightweight team (if any)
      if (lwt)
        lwt = lwt->parent;

      // next heavyweight team (if any) after
      // lightweight teams are exhausted
      if (!lwt && taskdata) {
        // first try scheduling parent (for explicit task scheduling)
        if (taskdata->ompt_task_info.scheduling_parent) {
          taskdata = taskdata->ompt_task_info.scheduling_parent;
        } else if (next_lwt) {
          lwt = next_lwt;
          next_lwt = NULL;
        } else {
          // then go for implicit tasks
          taskdata = taskdata->td_parent;
          if (taskdata) {
            next_lwt = LWT_FROM_TEAM(taskdata->td_team);
          }
        }
      }
      depth--;
    }

    if (lwt) {
      info = &lwt->ompt_task_info;
    } else if (taskdata) {
      info = &taskdata->ompt_task_info;
    }
  }

  return info;
}

//******************************************************************************
// interface operations
//******************************************************************************

//----------------------------------------------------------
// thread support
//----------------------------------------------------------

ompt_data_t *__ompt_get_thread_data_internal() {
  if (__kmp_get_gtid() >= 0) {
    kmp_info_t *thread = ompt_get_thread();
    if (thread == NULL)
      return NULL;
    return &(thread->th.ompt_thread_info.thread_data);
  }
  return NULL;
}

//----------------------------------------------------------
// state support
//----------------------------------------------------------

void __ompt_thread_assign_wait_id(void *variable) {
  kmp_info_t *ti = ompt_get_thread();

  if (ti)
    ti->th.ompt_thread_info.wait_id = (ompt_wait_id_t)(uintptr_t)variable;
}

int __ompt_get_state_internal(ompt_wait_id_t *omp_wait_id) {
  kmp_info_t *ti = ompt_get_thread();

  if (ti) {
    if (omp_wait_id)
      *omp_wait_id = ti->th.ompt_thread_info.wait_id;
    return ti->th.ompt_thread_info.state;
  }
  return ompt_state_undefined;
}

//----------------------------------------------------------
// parallel region support
//----------------------------------------------------------

int __ompt_get_parallel_info_internal(int ancestor_level,
                                      ompt_data_t **parallel_data,
                                      int *team_size) {
  if (__kmp_get_gtid() >= 0) {
    ompt_team_info_t *info;
    if (team_size) {
      info = __ompt_get_teaminfo(ancestor_level, team_size);
    } else {
      info = __ompt_get_teaminfo(ancestor_level, NULL);
    }
    if (parallel_data) {
      *parallel_data = info ? &(info->parallel_data) : NULL;
    }
    return info ? 2 : 0;
  } else {
    return 0;
  }
}

//----------------------------------------------------------
// lightweight task team support
//----------------------------------------------------------

void __ompt_lw_taskteam_init(ompt_lw_taskteam_t *lwt, kmp_info_t *thr, int gtid,
                             ompt_data_t *ompt_pid, void *codeptr) {
  // initialize parallel_data with input, return address to parallel_data on
  // exit
  lwt->ompt_team_info.parallel_data = *ompt_pid;
  lwt->ompt_team_info.master_return_address = codeptr;
  lwt->ompt_task_info.task_data.value = 0;
  lwt->ompt_task_info.frame.enter_frame = ompt_data_none;
  lwt->ompt_task_info.frame.exit_frame = ompt_data_none;
  lwt->ompt_task_info.scheduling_parent = NULL;
  lwt->heap = 0;
  lwt->parent = 0;
}

void __ompt_lw_taskteam_link(ompt_lw_taskteam_t *lwt, kmp_info_t *thr,
                             int on_heap, bool always) {
  ompt_lw_taskteam_t *link_lwt = lwt;
  if (always ||
      thr->th.th_team->t.t_serialized >
          1) { // we already have a team, so link the new team and swap values
    if (on_heap) { // the lw_taskteam cannot stay on stack, allocate it on heap
      link_lwt =
          (ompt_lw_taskteam_t *)__kmp_allocate(sizeof(ompt_lw_taskteam_t));
    }
    link_lwt->heap = on_heap;

    // would be swap in the (on_stack) case.
    ompt_team_info_t tmp_team = lwt->ompt_team_info;
    link_lwt->ompt_team_info = *OMPT_CUR_TEAM_INFO(thr);
    *OMPT_CUR_TEAM_INFO(thr) = tmp_team;

    // link the taskteam into the list of taskteams:
    ompt_lw_taskteam_t *my_parent =
        thr->th.th_team->t.ompt_serialized_team_info;
    link_lwt->parent = my_parent;
    thr->th.th_team->t.ompt_serialized_team_info = link_lwt;
#if OMPD_SUPPORT
    if (ompd_state & OMPD_ENABLE_BP) {
      ompd_bp_parallel_begin();
    }
#endif

    ompt_task_info_t tmp_task = lwt->ompt_task_info;
    link_lwt->ompt_task_info = *OMPT_CUR_TASK_INFO(thr);
    *OMPT_CUR_TASK_INFO(thr) = tmp_task;
  } else {
    // this is the first serialized team, so we just store the values in the
    // team and drop the taskteam-object
    *OMPT_CUR_TEAM_INFO(thr) = lwt->ompt_team_info;
#if OMPD_SUPPORT
    if (ompd_state & OMPD_ENABLE_BP) {
      ompd_bp_parallel_begin();
    }
#endif
    *OMPT_CUR_TASK_INFO(thr) = lwt->ompt_task_info;
  }
}

void __ompt_lw_taskteam_unlink(kmp_info_t *thr) {
  ompt_lw_taskteam_t *lwtask = thr->th.th_team->t.ompt_serialized_team_info;
  if (lwtask) {
    ompt_task_info_t tmp_task = lwtask->ompt_task_info;
    lwtask->ompt_task_info = *OMPT_CUR_TASK_INFO(thr);
    *OMPT_CUR_TASK_INFO(thr) = tmp_task;
#if OMPD_SUPPORT
    if (ompd_state & OMPD_ENABLE_BP) {
      ompd_bp_parallel_end();
    }
#endif
    thr->th.th_team->t.ompt_serialized_team_info = lwtask->parent;

    ompt_team_info_t tmp_team = lwtask->ompt_team_info;
    lwtask->ompt_team_info = *OMPT_CUR_TEAM_INFO(thr);
    *OMPT_CUR_TEAM_INFO(thr) = tmp_team;

    if (lwtask->heap) {
      __kmp_free(lwtask);
      lwtask = NULL;
    }
  }
  //    return lwtask;
}

//----------------------------------------------------------
// task support
//----------------------------------------------------------

int __ompt_get_task_info_internal(int ancestor_level, int *type,
                                  ompt_data_t **task_data,
                                  ompt_frame_t **task_frame,
                                  ompt_data_t **parallel_data,
                                  int *thread_num) {
  if (__kmp_get_gtid() < 0)
    return 0;

  if (ancestor_level < 0)
    return 0;

  // copied from __ompt_get_scheduling_taskinfo
  ompt_task_info_t *info = NULL;
  ompt_team_info_t *team_info = NULL;
  kmp_info_t *thr = ompt_get_thread();
  int level = ancestor_level;

  if (thr) {
    kmp_taskdata_t *taskdata = thr->th.th_current_task;
    if (taskdata == NULL)
      return 0;
    kmp_team *team = thr->th.th_team, *prev_team = NULL;
    if (team == NULL)
      return 0;
    ompt_lw_taskteam_t *lwt = NULL,
                       *next_lwt = LWT_FROM_TEAM(taskdata->td_team),
                       *prev_lwt = NULL;

    while (ancestor_level > 0) {
      // needed for thread_num
      prev_team = team;
      prev_lwt = lwt;
      // next lightweight team (if any)
      if (lwt)
        lwt = lwt->parent;

      // next heavyweight team (if any) after
      // lightweight teams are exhausted
      if (!lwt && taskdata) {
        // first try scheduling parent (for explicit task scheduling)
        if (taskdata->ompt_task_info.scheduling_parent) {
          taskdata = taskdata->ompt_task_info.scheduling_parent;
        } else if (next_lwt) {
          lwt = next_lwt;
          next_lwt = NULL;
        } else {
          // then go for implicit tasks
          taskdata = taskdata->td_parent;
          if (team == NULL)
            return 0;
          team = team->t.t_parent;
          if (taskdata) {
            next_lwt = LWT_FROM_TEAM(taskdata->td_team);
          }
        }
      }
      ancestor_level--;
    }

    if (lwt) {
      info = &lwt->ompt_task_info;
      team_info = &lwt->ompt_team_info;
      if (type) {
        *type = ompt_task_implicit;
      }
    } else if (taskdata) {
      info = &taskdata->ompt_task_info;
      team_info = &team->t.ompt_team_info;
      if (type) {
        if (taskdata->td_parent) {
          *type = (taskdata->td_flags.tasktype ? ompt_task_explicit
                                               : ompt_task_implicit) |
                  TASK_TYPE_DETAILS_FORMAT(taskdata);
        } else {
          *type = ompt_task_initial;
        }
      }
    }
    if (task_data) {
      *task_data = info ? &info->task_data : NULL;
    }
    if (task_frame) {
      // OpenMP spec asks for the scheduling task to be returned.
      *task_frame = info ? &info->frame : NULL;
    }
    if (parallel_data) {
      *parallel_data = team_info ? &(team_info->parallel_data) : NULL;
    }
    if (thread_num) {
      if (level == 0)
        *thread_num = __kmp_get_tid();
      else if (prev_lwt)
        *thread_num = 0;
      else
        *thread_num = prev_team->t.t_master_tid;
      //        *thread_num = team->t.t_master_tid;
    }
    return info ? 2 : 0;
  }
  return 0;
}

int __ompt_get_task_memory_internal(void **addr, size_t *size, int blocknum) {
  if (blocknum != 0)
    return 0; // support only a single block

  kmp_info_t *thr = ompt_get_thread();
  if (!thr)
    return 0;

  kmp_taskdata_t *taskdata = thr->th.th_current_task;
  kmp_task_t *task = KMP_TASKDATA_TO_TASK(taskdata);

  if (taskdata->td_flags.tasktype != TASK_EXPLICIT)
    return 0; // support only explicit task

  void *ret_addr;
  int64_t ret_size = taskdata->td_size_alloc - sizeof(kmp_taskdata_t);

  // kmp_task_t->data1 is an optional member
  if (taskdata->td_flags.destructors_thunk)
    ret_addr = &task->data1 + 1;
  else
    ret_addr = &task->part_id + 1;

  ret_size -= (char *)(ret_addr) - (char *)(task);
  if (ret_size < 0)
    return 0;

  *addr = ret_addr;
  *size = (size_t)ret_size;
  return 1;
}

//----------------------------------------------------------
// team support
//----------------------------------------------------------

void __ompt_team_assign_id(kmp_team_t *team, ompt_data_t ompt_pid) {
  team->t.ompt_team_info.parallel_data = ompt_pid;
}

//----------------------------------------------------------
// misc
//----------------------------------------------------------

static uint64_t __ompt_get_unique_id_internal() {
  static uint64_t thread = 1;
  static THREAD_LOCAL uint64_t ID = 0;
  if (ID == 0) {
    uint64_t new_thread = KMP_TEST_THEN_INC64((kmp_int64 *)&thread);
    ID = new_thread << (sizeof(uint64_t) * 8 - OMPT_THREAD_ID_BITS);
  }
  return ++ID;
}

ompt_sync_region_t __ompt_get_barrier_kind(enum barrier_type bt,
                                           kmp_info_t *thr) {
  if (bt == bs_forkjoin_barrier)
    return ompt_sync_region_barrier_implicit;

  if (bt != bs_plain_barrier)
    return ompt_sync_region_barrier_implementation;

  if (!thr->th.th_ident)
    return ompt_sync_region_barrier;

  kmp_int32 flags = thr->th.th_ident->flags;

  if ((flags & KMP_IDENT_BARRIER_EXPL) != 0)
    return ompt_sync_region_barrier_explicit;

  if ((flags & KMP_IDENT_BARRIER_IMPL) != 0)
    return ompt_sync_region_barrier_implicit;

  return ompt_sync_region_barrier_implementation;
}
