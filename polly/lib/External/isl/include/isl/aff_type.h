#ifndef ISL_AFF_TYPE_H
#define ISL_AFF_TYPE_H

#include <isl/list.h>

#if defined(__cplusplus)
extern "C" {
#endif

struct isl_aff;
typedef struct isl_aff isl_aff;

ISL_DECLARE_LIST(aff)

struct isl_pw_aff;
typedef struct isl_pw_aff isl_pw_aff;

ISL_DECLARE_LIST(pw_aff)

struct isl_multi_aff;
typedef struct isl_multi_aff isl_multi_aff;

struct isl_pw_multi_aff;
typedef struct isl_pw_multi_aff isl_pw_multi_aff;

struct isl_union_pw_multi_aff;
typedef struct isl_union_pw_multi_aff isl_union_pw_multi_aff;

struct isl_multi_pw_aff;
typedef struct isl_multi_pw_aff isl_multi_pw_aff;

#if defined(__cplusplus)
}
#endif

#endif
