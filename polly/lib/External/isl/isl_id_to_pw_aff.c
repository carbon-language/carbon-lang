#include <isl/id_to_pw_aff.h>
#include <isl/aff.h>

#define isl_id_is_equal(id1,id2)	id1 == id2

#define KEY_BASE	id
#define KEY_EQUAL	isl_id_is_equal
#define VAL_BASE	pw_aff
#define VAL_EQUAL	isl_pw_aff_plain_is_equal

#include <isl_hmap_templ.c>
