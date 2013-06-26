#include <clc/clc.h>

_CLC_DEF size_t get_global_size(uint dim) {
  switch (dim) {
  case 0: return __builtin_r600_read_global_size_x();
  case 1: return __builtin_r600_read_global_size_y();
  case 2: return __builtin_r600_read_global_size_z();
  default: return 1;
  }
}
