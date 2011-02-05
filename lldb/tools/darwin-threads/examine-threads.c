#include <mach/mach.h>
#include <stdio.h>
#include <stdlib.h>

void
dump_thread_basic_info (uint32_t index, uint32_t tid, struct thread_basic_info *i)
{
    const char * thread_run_state = NULL;

    switch (i->run_state)
    {
    case TH_STATE_RUNNING:          thread_run_state = "running"; break;    // 1 thread is running normally
    case TH_STATE_STOPPED:          thread_run_state = "stopped"; break;    // 2 thread is stopped
    case TH_STATE_WAITING:          thread_run_state = "waiting"; break;    // 3 thread is waiting normally
    case TH_STATE_UNINTERRUPTIBLE:  thread_run_state = "uninter"; break;    // 4 thread is in an uninterruptible wait
    case TH_STATE_HALTED:           thread_run_state = "halted "; break;     // 5 thread is halted at a
    default:                        thread_run_state = "???"; break;
    }

//    printf("[%3u] tid: 0x%4.4x, pc: 0x%16.16llx, sp: 0x%16.16llx, user: %d.%06.6d, system: %d.%06.6d, cpu: %2d, policy: %2d, run_state: %2d (%s), flags: %2d, suspend_count: %2d (current %2d), sleep_time: %d",
    printf("[%3u] tid: 0x%4.4x user: %d.%06d, system: %d.%06d, cpu: %2d, policy: %2d, run_state: %2d (%s), flags: %2d, suspend_count: %2d, sleep_time: %d\n",
        index,
        tid,
        i->user_time.seconds,      i->user_time.microseconds,
        i->system_time.seconds,    i->system_time.microseconds,
        i->cpu_usage,
        i->policy,
        i->run_state,
        thread_run_state,
        i->flags,
        i->suspend_count,
        i->sleep_time);
    //DumpRegisterState(0);
}


int main (int argc, char ** argv)
{
  kern_return_t kret;
  task_t itask;
  thread_array_t thread_list;
  unsigned int nthreads;
  pid_t pid;

  if (argc < 2)
    {
      printf ("Usage: %s <PID>.\n", argv[0]);
      return -1;
    }

  pid = atoi (argv[1]);
  printf ("Examining process: %d.\n", pid);

  kret = task_for_pid (mach_task_self (), pid, &itask);
  if (kret != KERN_SUCCESS)
    {
      printf ("Could not get task for pid %d.\n", pid);
      return -1;
    }

  struct task_basic_info info_for_task;
  unsigned int task_info_count = TASK_BASIC_INFO_COUNT;

  kret =
    task_info (itask, TASK_BASIC_INFO, (task_info_t) &info_for_task, &task_info_count);
  if (kret != KERN_SUCCESS)
    {
      printf ("Could not get task info for task: 0x%4.4x.\n", itask);
    }
  
  printf ("Task suspend: %d.\n", info_for_task.suspend_count);

  kret = task_threads (itask, &thread_list, &nthreads);
  if (kret != KERN_SUCCESS)
    {
      printf ("Could not get task threads for task 0x%4.4x.\n", itask);
      return -1;
    }

  int i;
  for (i = 0; i < nthreads; i++)
    {
      struct thread_basic_info info;
      unsigned int thread_info_count = THREAD_BASIC_INFO_COUNT;
      kern_return_t kret;
      
      kret = thread_info (thread_list[i], THREAD_BASIC_INFO,
                        (thread_info_t) & info, &thread_info_count);

      if (kret != KERN_SUCCESS)
        {
          printf ("Error getting thread basic info for thread 0x%4.4x.\n", thread_list[i]);
        }
      else
        {
            dump_thread_basic_info (i + 1, thread_list[i], &info);
        }
        if (argc > 2) printf("thread_resume (tid = 0x%4.4x) => %i\n", thread_list[i], thread_resume (thread_list[i]));      
    }
  
  if (argc > 2) printf("task_resume (task = 0x%4.4x) => %i\n", itask, task_resume (itask));
  return 1;

}
