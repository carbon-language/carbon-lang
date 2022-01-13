//===---- parallel.cu - GPU OpenMP parallel implementation ------- CUDA -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Parallel implementation in the GPU. Here is the pattern:
//
//    while (not finished) {
//
//    if (master) {
//      sequential code, decide which par loop to do, or if finished
//     __kmpc_kernel_prepare_parallel() // exec by master only
//    }
//    syncthreads // A
//    __kmpc_kernel_parallel() // exec by all
//    if (this thread is included in the parallel) {
//      switch () for all parallel loops
//      __kmpc_kernel_end_parallel() // exec only by threads in parallel
//    }
//
//
//    The reason we don't exec end_parallel for the threads not included
//    in the parallel loop is that for each barrier in the parallel
//    region, these non-included threads will cycle through the
//    syncthread A. Thus they must preserve their current threadId that
//    is larger than thread in team.
//
//    To make a long story short...
//
//===----------------------------------------------------------------------===//
#pragma omp declare target

#include "common/omptarget.h"
#include "target_impl.h"

////////////////////////////////////////////////////////////////////////////////
// support for parallel that goes parallel (1 static level only)
////////////////////////////////////////////////////////////////////////////////

INLINE static uint16_t determineNumberOfThreads(uint16_t NumThreadsClause,
                                                uint16_t NThreadsICV,
                                                uint16_t ThreadLimit) {
  uint16_t ThreadsRequested = NThreadsICV;
  if (NumThreadsClause != 0) {
    ThreadsRequested = NumThreadsClause;
  }

  uint16_t ThreadsAvailable = GetNumberOfWorkersInTeam();
  if (ThreadLimit != 0 && ThreadLimit < ThreadsAvailable) {
    ThreadsAvailable = ThreadLimit;
  }

  uint16_t NumThreads = ThreadsAvailable;
  if (ThreadsRequested != 0 && ThreadsRequested < NumThreads) {
    NumThreads = ThreadsRequested;
  }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  // On Volta and newer architectures we require that all lanes in
  // a warp participate in the parallel region.  Round down to a
  // multiple of WARPSIZE since it is legal to do so in OpenMP.
  if (NumThreads < WARPSIZE) {
    NumThreads = 1;
  } else {
    NumThreads = (NumThreads & ~((uint16_t)WARPSIZE - 1));
  }
#endif

  return NumThreads;
}

// This routine is always called by the team master..
EXTERN void __kmpc_kernel_prepare_parallel(void *WorkFn,
                                           kmp_int32 NumThreadsClause) {
  PRINT0(LD_IO, "call to __kmpc_kernel_prepare_parallel\n");

  omptarget_nvptx_workFn = WorkFn;

  // This routine is only called by the team master.  The team master is
  // the first thread of the last warp.  It always has the logical thread
  // id of 0 (since it is a shadow for the first worker thread).
  const int threadId = 0;
  omptarget_nvptx_TaskDescr *currTaskDescr =
      omptarget_nvptx_threadPrivateContext->GetTopLevelTaskDescr(threadId);
  ASSERT0(LT_FUSSY, currTaskDescr, "expected a top task descr");
  ASSERT0(LT_FUSSY, !currTaskDescr->InParallelRegion(),
          "cannot be called in a parallel region.");
  if (currTaskDescr->InParallelRegion()) {
    PRINT0(LD_PAR, "already in parallel: go seq\n");
    return;
  }

  uint16_t NumThreads =
      determineNumberOfThreads(NumThreadsClause, nThreads, threadLimit);

  if (NumThreadsClause != 0) {
    // Reset request to avoid propagating to successive #parallel
    NumThreadsClause = 0;
  }

  ASSERT(LT_FUSSY, NumThreads > 0, "bad thread request of %d threads",
         (int)NumThreads);
  ASSERT0(LT_FUSSY,
          __kmpc_get_hardware_thread_id_in_block() == GetMasterThreadID(),
          "only team master can create parallel");

  // Set number of threads on work descriptor.
  omptarget_nvptx_WorkDescr &workDescr = getMyWorkDescriptor();
  workDescr.WorkTaskDescr()->CopyToWorkDescr(currTaskDescr);
  threadsInTeam = NumThreads;
}

// All workers call this function.  Deactivate those not needed.
// Fn - the outlined work function to execute.
// returns True if this thread is active, else False.
//
// Only the worker threads call this routine.
EXTERN bool __kmpc_kernel_parallel(void **WorkFn) {
  PRINT0(LD_IO | LD_PAR, "call to __kmpc_kernel_parallel\n");

  // Work function and arguments for L1 parallel region.
  *WorkFn = omptarget_nvptx_workFn;

  // If this is the termination signal from the master, quit early.
  if (!*WorkFn) {
    PRINT0(LD_IO | LD_PAR, "call to __kmpc_kernel_parallel finished\n");
    return false;
  }

  // Only the worker threads call this routine and the master warp
  // never arrives here.  Therefore, use the nvptx thread id.
  int threadId = __kmpc_get_hardware_thread_id_in_block();
  omptarget_nvptx_WorkDescr &workDescr = getMyWorkDescriptor();
  // Set to true for workers participating in the parallel region.
  bool isActive = false;
  // Initialize state for active threads.
  if (threadId < threadsInTeam) {
    // init work descriptor from workdesccr
    omptarget_nvptx_TaskDescr *newTaskDescr =
        omptarget_nvptx_threadPrivateContext->Level1TaskDescr(threadId);
    ASSERT0(LT_FUSSY, newTaskDescr, "expected a task descr");
    newTaskDescr->CopyFromWorkDescr(workDescr.WorkTaskDescr());
    // install new top descriptor
    omptarget_nvptx_threadPrivateContext->SetTopLevelTaskDescr(threadId,
                                                               newTaskDescr);
    // init private from int value
    PRINT(LD_PAR,
          "thread will execute parallel region with id %d in a team of "
          "%d threads\n",
          (int)newTaskDescr->ThreadId(), (int)nThreads);

    isActive = true;
  }

  return isActive;
}

EXTERN void __kmpc_kernel_end_parallel() {
  // pop stack
  PRINT0(LD_IO | LD_PAR, "call to __kmpc_kernel_end_parallel\n");
  ASSERT0(LT_FUSSY, isRuntimeInitialized(), "Expected initialized runtime.");

  // Only the worker threads call this routine and the master warp
  // never arrives here.  Therefore, use the nvptx thread id.
  int threadId = __kmpc_get_hardware_thread_id_in_block();
  omptarget_nvptx_TaskDescr *currTaskDescr = getMyTopTaskDescriptor(threadId);
  omptarget_nvptx_threadPrivateContext->SetTopLevelTaskDescr(
      threadId, currTaskDescr->GetPrevTaskDescr());
}

////////////////////////////////////////////////////////////////////////////////
// support for parallel that goes sequential
////////////////////////////////////////////////////////////////////////////////

static void serializedParallel(kmp_Ident *loc, uint32_t global_tid) {
  PRINT0(LD_IO, "call to serializedParallel\n");

  IncParallelLevel(/*ActiveParallel=*/false, __kmpc_impl_activemask());

  if (isRuntimeUninitialized()) {
    ASSERT0(LT_FUSSY, __kmpc_is_spmd_exec_mode(),
            "Expected SPMD mode with uninitialized runtime.");
    return;
  }

  // assume this is only called for nested parallel
  int threadId = GetLogicalThreadIdInBlock();

  // unlike actual parallel, threads in the same team do not share
  // the workTaskDescr in this case and num threads is fixed to 1

  // get current task
  omptarget_nvptx_TaskDescr *currTaskDescr = getMyTopTaskDescriptor(threadId);
  currTaskDescr->SaveLoopData();

  // allocate new task descriptor and copy value from current one, set prev to
  // it
  omptarget_nvptx_TaskDescr *newTaskDescr =
      (omptarget_nvptx_TaskDescr *)SafeMalloc(sizeof(omptarget_nvptx_TaskDescr),
                                              "new seq parallel task");
  newTaskDescr->CopyParent(currTaskDescr);

  // tweak values for serialized parallel case:
  // - each thread becomes ID 0 in its serialized parallel, and
  // - there is only one thread per team
  newTaskDescr->ThreadId() = 0;

  // set new task descriptor as top
  omptarget_nvptx_threadPrivateContext->SetTopLevelTaskDescr(threadId,
                                                             newTaskDescr);
}

static void endSerializedParallel(kmp_Ident *loc,
                                           uint32_t global_tid) {
  PRINT0(LD_IO, "call to endSerializedParallel\n");

  DecParallelLevel(/*ActiveParallel=*/false, __kmpc_impl_activemask());

  if (isRuntimeUninitialized()) {
    ASSERT0(LT_FUSSY, __kmpc_is_spmd_exec_mode(),
            "Expected SPMD mode with uninitialized runtime.");
    return;
  }

  // pop stack
  int threadId = GetLogicalThreadIdInBlock();
  omptarget_nvptx_TaskDescr *currTaskDescr = getMyTopTaskDescriptor(threadId);
  // set new top
  omptarget_nvptx_threadPrivateContext->SetTopLevelTaskDescr(
      threadId, currTaskDescr->GetPrevTaskDescr());
  // free
  SafeFree(currTaskDescr, "new seq parallel task");
  currTaskDescr = getMyTopTaskDescriptor(threadId);
  currTaskDescr->RestoreLoopData();
}

NOINLINE EXTERN uint8_t __kmpc_parallel_level() {
  return parallelLevel[GetWarpId()] & (OMP_ACTIVE_PARALLEL_LEVEL - 1);
}

// This kmpc call returns the thread id across all teams. It's value is
// cached by the compiler and used when calling the runtime. On nvptx
// it's cheap to recalculate this value so we never use the result
// of this call.
EXTERN int32_t __kmpc_global_thread_num(kmp_Ident *loc) {
  return GetOmpThreadId();
}

////////////////////////////////////////////////////////////////////////////////
// push params
////////////////////////////////////////////////////////////////////////////////

// Do nothing. The host guarantees we started the requested number of
// teams and we only need inspection of gridDim.

EXTERN void __kmpc_push_num_teams(kmp_Ident *loc, int32_t tid,
                                  int32_t num_teams, int32_t thread_limit) {
  PRINT(LD_IO, "call kmpc_push_num_teams %d\n", (int)num_teams);
  ASSERT0(LT_FUSSY, 0, "should never have anything with new teams on device");
}

EXTERN void __kmpc_push_proc_bind(kmp_Ident *loc, uint32_t tid, int proc_bind) {
  PRINT(LD_IO, "call kmpc_push_proc_bind %d\n", (int)proc_bind);
}

////////////////////////////////////////////////////////////////////////////////
// parallel interface
////////////////////////////////////////////////////////////////////////////////

NOINLINE EXTERN void __kmpc_parallel_51(kmp_Ident *ident, kmp_int32 global_tid,
                                        kmp_int32 if_expr,
                                        kmp_int32 num_threads, int proc_bind,
                                        void *fn, void *wrapper_fn, void **args,
                                        size_t nargs) {
  // Handle the serialized case first, same for SPMD/non-SPMD except that in
  // SPMD mode we already incremented the parallel level counter, account for
  // that.
  bool InParallelRegion =
      (__kmpc_parallel_level() > __kmpc_is_spmd_exec_mode());
  if (!if_expr || InParallelRegion) {
    serializedParallel(ident, global_tid);
    __kmp_invoke_microtask(global_tid, 0, fn, args, nargs);
    endSerializedParallel(ident, global_tid);
    return;
  }

  if (__kmpc_is_spmd_exec_mode()) {
    __kmp_invoke_microtask(global_tid, 0, fn, args, nargs);
    return;
  }

  __kmpc_kernel_prepare_parallel((void *)wrapper_fn, num_threads);

  if (nargs) {
    void **GlobalArgs;
    __kmpc_begin_sharing_variables(&GlobalArgs, nargs);
    // TODO: faster memcpy?
#pragma unroll
    for (int I = 0; I < nargs; I++)
      GlobalArgs[I] = args[I];
  }

  // TODO: what if that's a parallel region with a single thread? this is
  // considered not active in the existing implementation.
  bool IsActiveParallelRegion = threadsInTeam != 1;
  int NumWarps =
      threadsInTeam / WARPSIZE + ((threadsInTeam % WARPSIZE) ? 1 : 0);
  // Increment parallel level for non-SPMD warps.
  for (int I = 0; I < NumWarps; ++I)
    parallelLevel[I] +=
        (1 + (IsActiveParallelRegion ? OMP_ACTIVE_PARALLEL_LEVEL : 0));

  // Master signals work to activate workers.
  __kmpc_barrier_simple_spmd(ident, 0);

  // OpenMP [2.5, Parallel Construct, p.49]
  // There is an implied barrier at the end of a parallel region. After the
  // end of a parallel region, only the master thread of the team resumes
  // execution of the enclosing task region.
  //
  // The master waits at this barrier until all workers are done.
  __kmpc_barrier_simple_spmd(ident, 0);

  // Decrement parallel level for non-SPMD warps.
  for (int I = 0; I < NumWarps; ++I)
    parallelLevel[I] -=
        (1 + (IsActiveParallelRegion ? OMP_ACTIVE_PARALLEL_LEVEL : 0));
  // TODO: Is synchronization needed since out of parallel execution?

  if (nargs)
    __kmpc_end_sharing_variables();

  // TODO: proc_bind is a noop?
  // if (proc_bind != proc_bind_default)
  //  __kmpc_push_proc_bind(ident, global_tid, proc_bind);
}

#pragma omp end declare target
