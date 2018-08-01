/* Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef CFI_ISO_FORTRAN_BINDING_H_
#define CFI_ISO_FORTRAN_BINDING_H_

#include <stddef.h>

/* Standard interface to Fortran from C and C++.
 * These interfaces are named in section 18.5 of the Fortran 2018
 * standard, with most of the actual details being left to the
 * implementation.
 */

#ifdef __cplusplus
// C++ does not support flexible array members, so they have to be
// declared with single elements.
#define CFI_ISO_FORTRAN_BINDING_FLEXIBLE_ARRAY 1
namespace Fortran {
namespace ISO {
inline namespace Fortran_2018 {
#else
#define CFI_ISO_FORTRAN_BINDING_FLEXIBLE_ARRAY
#endif

/* 18.5.4 */
#define CFI_VERSION 20180515

#define CFI_MAX_RANK 15
typedef unsigned char CFI_rank_t;

// This type is probably larger than a default Fortran INTEGER
// and should be used for all array indexing and loop bound calculations.
typedef ptrdiff_t CFI_index_t;

#define CFI_DESC_T(rank) \
  struct { \
    CFI_cdesc_t cdesc; /* must be first */ \
    CFI_dim_t dim[rank]; \
  };

typedef unsigned short CFI_attribute_t;
#define CFI_attribute_pointer 1
#define CFI_attribute_allocatable 2
#define CFI_attribute_other 0 /* neither pointer nor allocatable */

typedef signed char CFI_type_t;
/* These codes are required to be macros (i.e., #ifdef will work).
 * They are not required to be distinct, but neither are they required
 * to have had their synonyms combined.
 * Extension: 128-bit integers are anticipated
 */
#define CFI_type_signed_char 1
#define CFI_type_short 2
#define CFI_type_int 3
#define CFI_type_long 4
#define CFI_type_long_long 5
#define CFI_type_size_t 6
#define CFI_type_int8_t 7
#define CFI_type_int16_t 8
#define CFI_type_int32_t 9
#define CFI_type_int64_t 10
#define CFI_type_int128_t 11
#define CFI_type_int_least8_t 12
#define CFI_type_int_least16_t 13
#define CFI_type_int_least32_t 14
#define CFI_type_int_least64_t 15
#define CFI_type_int_least128_t 16
#define CFI_type_int_fast8_t 17
#define CFI_type_int_fast16_t 18
#define CFI_type_int_fast32_t 19
#define CFI_type_int_fast64_t 20
#define CFI_type_int_fast128_t 21
#define CFI_type_intmax_t 22
#define CFI_type_intptr_t 23
#define CFI_type_ptrdiff_t 24
#define CFI_type_float 25
#define CFI_type_double 26
#define CFI_type_long_double 27
#define CFI_type_float_Complex 28
#define CFI_type_double_Complex 29
#define CFI_type_long_double_Complex 30
#define CFI_type_Bool 31
#define CFI_type_char 32
#define CFI_type_cptr 33
#define CFI_type_struct 34
#define CFI_type_other (-1)  // must be negative

/* Error code macros */
#define CFI_SUCCESS 0  /* must be zero */
#define CFI_ERROR_BASE_ADDR_NULL 1
#define CFI_ERROR_BASE_ADDR_NOT_NULL 2
#define CFI_INVALID_ELEM_LEN 3
#define CFI_INVALID_RANK 4
#define CFI_INVALID_TYPE 5
#define CFI_INVALID_ATTRIBUTE 6
#define CFI_INVALID_EXTENT 7
#define CFI_INVALID_DESCRIPTOR 8
#define CFI_ERROR_MEM_ALLOCATION 9
#define CFI_ERROR_OUT_OF_BOUNDS 10

/* 18.5.2 per-dimension information */
typedef struct CFI_dim_t {
  CFI_index_t lower_bound;
  CFI_index_t extent; /* == -1 for assumed size */
  CFI_index_t sm; /* memory stride in bytes */
} CFI_dim_t;

/* 18.5.3 generic data descriptor */
typedef struct CFI_cdesc_t {
  /* These three members must appear first, in exactly this order. */
  void *base_addr;
  size_t elem_len; /* element size in bytes */
  int version; /* == CFI_VERSION */
  CFI_rank_t rank; /* [0 .. CFI_MAX_RANK] */
  CFI_type_t type;
  CFI_attribute_t attribute;
  unsigned char f18Addendum;
  CFI_dim_t dim[CFI_ISO_FORTRAN_BINDING_FLEXIBLE_ARRAY]; /* must appear last */
} CFI_cdesc_t;

/* 18.5.5 procedural interfaces*/
#ifdef __cplusplus
extern "C" {
#endif
void *CFI_address(const CFI_cdesc_t *, const CFI_index_t subscripts[]);
int CFI_allocate(CFI_cdesc_t *, const CFI_index_t lower_bounds[],
    const CFI_index_t upper_bounds[], size_t elem_len);
int CFI_deallocate(CFI_cdesc_t *);
int CFI_establish(CFI_cdesc_t *, void *base_addr, CFI_attribute_t, CFI_type_t,
    size_t elem_len, CFI_rank_t, const CFI_index_t extents[]);
int CFI_is_contiguous(const CFI_cdesc_t *);
int CFI_section(CFI_cdesc_t *, const CFI_cdesc_t *source,
    const CFI_index_t lower_bounds[], const CFI_index_t upper_bounds[],
    const CFI_index_t strides[]);
int CFI_select_part(CFI_cdesc_t *, const CFI_cdesc_t *source,
    size_t displacement, size_t elem_len);
int CFI_setpointer(
    CFI_cdesc_t *, const CFI_cdesc_t *source, const CFI_index_t lower_bounds[]);
#ifdef __cplusplus
}  // extern "C"
}  // inline namespace Fortran_2018
}  // namespace ISO
}  // namespace Fortran
#endif

#undef CFI_ISO_FORTRAN_BINDING_FLEXIBLE_ARRAY

#endif /* CFI_ISO_FORTRAN_BINDING_H_ */
