#ifndef ISL_MAP_TO_BASIC_SET_H
#define ISL_MAP_TO_BASIC_SET_H

#include <isl/set_type.h>
#include <isl/map_type.h>

#define ISL_KEY_BASE	map
#define ISL_VAL_BASE	basic_set
#include <isl/hmap.h>
#undef ISL_KEY_BASE
#undef ISL_VAL_BASE

#endif
