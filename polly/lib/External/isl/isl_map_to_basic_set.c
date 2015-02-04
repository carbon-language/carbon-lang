#include <isl/map_to_basic_set.h>
#include <isl/map.h>
#include <isl/set.h>

#define KEY_BASE	map
#define KEY_EQUAL	isl_map_plain_is_equal
#define VAL_BASE	basic_set
#define VAL_EQUAL	isl_basic_set_plain_is_equal

#include <isl_hmap_templ.c>
