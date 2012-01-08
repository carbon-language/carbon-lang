_CLC_INLINE size_t get_global_size(uint dim) {
  switch (dim) {
  case 0:  return __builtin_ptx_read_nctaid_x()*__builtin_ptx_read_ntid_x();
  case 1:  return __builtin_ptx_read_nctaid_y()*__builtin_ptx_read_ntid_y();
  case 2:  return __builtin_ptx_read_nctaid_z()*__builtin_ptx_read_ntid_z();
  default: return 0;
  }
}
