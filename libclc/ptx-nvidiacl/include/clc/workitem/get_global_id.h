_CLC_INLINE size_t get_global_id(uint dim) {
  switch (dim) {
  case 0:  return __builtin_ptx_read_ctaid_x()*__builtin_ptx_read_ntid_x()+__builtin_ptx_read_tid_x();
  case 1:  return __builtin_ptx_read_ctaid_y()*__builtin_ptx_read_ntid_y()+__builtin_ptx_read_tid_y();
  case 2:  return __builtin_ptx_read_ctaid_z()*__builtin_ptx_read_ntid_z()+__builtin_ptx_read_tid_z();
  default: return 0;
  }
}
