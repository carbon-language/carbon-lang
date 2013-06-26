#include <clc/clc.h>

_CLC_DEF size_t get_global_id(uint dim) {
  switch (dim) {
  case 0:  return __builtin_r600_read_tgid_x()*__builtin_r600_read_ngroups_x()+__builtin_r600_read_tidig_x();
  case 1:  return __builtin_r600_read_tgid_y()*__builtin_r600_read_ngroups_y()+__builtin_r600_read_tidig_y();
  case 2:  return __builtin_r600_read_tgid_z()*__builtin_r600_read_ngroups_z()+__builtin_r600_read_tidig_z();
  default: return 0;
  }
}
