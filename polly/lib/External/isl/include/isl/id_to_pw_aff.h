#ifndef ISL_ID_TO_PW_AFF_H
#define ISL_ID_TO_PW_AFF_H

#include <isl/id.h>
#include <isl/aff_type.h>

#define ISL_KEY_BASE	id
#define ISL_VAL_BASE	pw_aff
#include <isl/hmap.h>
#undef ISL_KEY_BASE
#undef ISL_VAL_BASE

#endif
