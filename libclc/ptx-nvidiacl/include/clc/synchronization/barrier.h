_CLC_INLINE void barrier(cl_mem_fence_flags flags) {
  if (flags & CLK_LOCAL_MEM_FENCE) {
    __builtin_ptx_bar_sync(0);
  }
}

