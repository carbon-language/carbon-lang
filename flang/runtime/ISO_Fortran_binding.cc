// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Implements the required interoperability API from ISO_Fortran_binding.h
// as specified in section 18.5.5 of Fortran 2018.

#include "descriptor.h"
#include <cstdlib>

namespace Fortran::ISO {
extern "C" {

void *CFI_address(const CFI_cdesc_t *descriptor,
                  const CFI_index_t subscripts[]) {
  auto p = reinterpret_cast<char *>(descriptor->base_addr);
  std::size_t rank{descriptor->rank};
  const CFI_dim_t *dim{descriptor->dim};
  for (std::size_t j{0}; j < rank; ++j, ++dim) {
    p += (subscripts[j] - dim->lower_bound) * dim->sm;
  }
  return reinterpret_cast<void *>(p);
}

int CFI_allocate(CFI_cdesc_t *descriptor, const CFI_index_t lower_bounds[],
                 const CFI_index_t upper_bounds[], std::size_t elem_len) {
  if (descriptor->version != CFI_VERSION) {
    return CFI_INVALID_DESCRIPTOR;
  }
  if ((descriptor->attribute &
      ~(CFI_attribute_pointer | CFI_attribute_allocatable)) != 0) {
    // Non-interoperable object
    return CFI_INVALID_DESCRIPTOR;
  }
  if (descriptor->base_addr != nullptr) {
    return CFI_ERROR_BASE_ADDR_NOT_NULL;
  }
  if (descriptor->rank > CFI_MAX_RANK) {
    return CFI_INVALID_RANK;
  }
  // TODO: CFI_INVALID_TYPE?
  if (descriptor->type != CFI_type_cptr) {
    elem_len = descriptor->elem_len;
    if (elem_len <= 0) {
      return CFI_INVALID_ELEM_LEN;
    }
  }
  std::size_t rank{descriptor->rank};
  CFI_dim_t *dim{descriptor->dim};
  std::size_t byteSize{elem_len};
  for (std::size_t j{0}; j < rank; ++j, ++dim) {
    CFI_index_t lb{lower_bounds[j]};
    CFI_index_t ub{upper_bounds[j]};
    CFI_index_t extent{ub >= lb ? ub - lb + 1 : 0};
    dim->lower_bound = lb;
    dim->extent = extent;
    dim->sm = byteSize;
    byteSize *= extent;
  }
  void *p{std::malloc(byteSize)};
  if (p == nullptr) {
    return CFI_ERROR_MEM_ALLOCATION;
  }
  descriptor->base_addr = p;
  return CFI_SUCCESS;
}

int CFI_deallocate(CFI_cdesc_t *descriptor) {
  if (descriptor->version != CFI_VERSION) {
    return CFI_INVALID_DESCRIPTOR;
  }
  if ((descriptor->attribute &
      ~(CFI_attribute_pointer | CFI_attribute_allocatable)) != 0) {
    // Non-interoperable object
    return CFI_INVALID_DESCRIPTOR;
  }
  if (descriptor->base_addr == nullptr) {
    return CFI_ERROR_BASE_ADDR_NULL;
  }
  std::free(descriptor->base_addr);
  descriptor->base_addr = nullptr;
  return CFI_SUCCESS;
}

int CFI_establish(CFI_cdesc_t *descriptor, void *base_addr,
                  CFI_attribute_t attribute, CFI_type_t type,
                  std::size_t elem_len, CFI_rank_t rank,
                  const CFI_index_t extents[]) {
  if ((attribute & ~(CFI_attribute_pointer | CFI_attribute_allocatable)) != 0) {
    return CFI_INVALID_ATTRIBUTE;
  }
  if ((attribute & CFI_attribute_allocatable) != 0 &&
      base_addr != nullptr) {
    return CFI_ERROR_BASE_ADDR_NOT_NULL;
  }
  if (rank > CFI_MAX_RANK) {
    return CFI_INVALID_RANK;
  }
  if (rank > 0 && base_addr != nullptr && extents == nullptr) {
    return CFI_INVALID_EXTENT;
  }
  if (type != CFI_type_struct && type != CFI_type_other &&
      type != CFI_type_cptr) {
    // TODO: force value of elem_len
  }
  descriptor->base_addr = base_addr;
  descriptor->elem_len = elem_len;
  descriptor->version = CFI_VERSION;
  descriptor->rank = rank;
  descriptor->attribute = attribute;
  std::size_t byteSize{elem_len};
  for (std::size_t j{0}; j < rank; ++j) {
    descriptor->dim[j].lower_bound = 1;
    descriptor->dim[j].extent = extents[j];
    descriptor->dim[j].sm = byteSize;
    byteSize *= extents[j];
  }
  return CFI_SUCCESS;
}

int CFI_is_contiguous(const CFI_cdesc_t *descriptor) {
  return 0;  // TODO
}

int CFI_section(CFI_cdesc_t *result, const CFI_cdesc_t *source,
                const CFI_index_t lower_bounds[],
                const CFI_index_t upper_bounds[],
                const CFI_index_t strides[]) {
  return CFI_INVALID_DESCRIPTOR;  // TODO
}

int CFI_select_part(CFI_cdesc_t *result, const CFI_cdesc_t *source,
                    std::size_t displacement, std::size_t elem_len) {
  return CFI_INVALID_DESCRIPTOR;  // TODO
}

int CFI_setpointer(CFI_cdesc_t *result, const CFI_cdesc_t *source,
                   const CFI_index_t lower_bounds[]) {
  return CFI_INVALID_DESCRIPTOR;  // TODO
}
}  // extern "C"
}  // namespace Fortran::ISO
