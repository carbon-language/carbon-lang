
#include <clc/clc.h>

void barrier_local(void);
void barrier_global(void);

void barrier(cl_mem_fence_flags flags) {
  if (flags & CLK_LOCAL_MEM_FENCE) {
    barrier_local();
  }

  if (flags & CLK_GLOBAL_MEM_FENCE) {
    barrier_global();
  }
}
