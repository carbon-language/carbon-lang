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

#include "testing.h"
#include "../../include/flang/ISO_Fortran_binding.h"
#include "../../runtime/descriptor.h"
#include <iostream>
#include <type_traits>

using namespace Fortran::runtime;
using namespace Fortran::ISO;

int check_CFI_establish(CFI_cdesc_t *dv, void *base_addr,
    CFI_attribute_t attribute, CFI_type_t type, std::size_t elem_len,
    CFI_rank_t rank, const CFI_index_t extents[]) {
  // CFI_establish reqs from F2018 section 18.5.5
  int ret_code =
      CFI_establish(dv, base_addr, attribute, type, elem_len, rank, extents);
  Descriptor *res = reinterpret_cast<Descriptor *>(dv);
  if (ret_code == CFI_SUCCESS) {
    res->Check();
    MATCH(res->IsPointer(), (attribute == CFI_attribute_pointer));
    MATCH(res->IsAllocatable(), (attribute == CFI_attribute_allocatable));
    MATCH(res->IsContiguous(), true);
    MATCH(res->rank(), rank);
    std::size_t head{0};
    MATCH(reinterpret_cast<std::intptr_t>(res->Element<char>(head)),
        reinterpret_cast<std::intptr_t>(base_addr));
    if (base_addr != nullptr) {
      for (int i{0}; i < rank; ++i) {
        MATCH(res->GetDimension(i).Extent(), extents[i]);
      }
    }
    if (attribute == CFI_attribute_allocatable) {
      MATCH(res->IsAllocated(), false);
    }
    if (attribute == CFI_attribute_pointer) {
      if (base_addr != nullptr) {
        // TODO should also be checked for Fortran character types and
        // CFI_type_other
        for (int i{0}; i < rank; ++i) {
          MATCH(res->GetDimension(i).LowerBound(), 0);
        }
      }
    }
    if (type == CFI_type_struct) {
      MATCH(elem_len, res->ElementBytes());
    }
  }
  return ret_code;
}

void add_noise_to_cdesc(CFI_cdesc_t *dv, CFI_rank_t rank) {
  static int trap;
  dv->rank = 16;
  dv->base_addr = reinterpret_cast<void *>(&trap);
  dv->elem_len = 320;
  dv->type = CFI_type_struct;
  dv->attribute = CFI_attribute_pointer;
  for (int i{0}; i < rank; i++) {
    dv->dim[i].extent = -42;
    dv->dim[i].lower_bound = -42;
    dv->dim[i].sm = -42;
  }
}

int main() {
  const int rank{3};
  static const CFI_index_t extents[]{2, 3, 66};
  CFI_CDESC_T(rank) dv_3darray_storage;
  static CFI_CDESC_T(0) dv_scalar_storage;
  CFI_cdesc_t *dv_3darray{reinterpret_cast<CFI_cdesc_t *>(&dv_3darray_storage)};
  CFI_cdesc_t *dv_scalar{reinterpret_cast<CFI_cdesc_t *>(&dv_scalar_storage)};
  CFI_attribute_t attr_cases[]{
      CFI_attribute_pointer, CFI_attribute_allocatable, CFI_attribute_other};
  CFI_attribute_t nonalloctable_cases[]{
      CFI_attribute_pointer, CFI_attribute_other};
  CFI_type_t type_cases[]{CFI_type_int, CFI_type_struct, CFI_type_double};
  int ret_code = CFI_INVALID_DESCRIPTOR;
  // Test CFI_CDESC_T macro defined in section 18.5.4 of F2018 standard
  // CFI_CDESC_T must give storage that is:
  // unqualified
  MATCH(std::is_const<decltype(dv_3darray_storage)>::value, false);
  MATCH(std::is_volatile<decltype(dv_3darray_storage)>::value, false);
  // suitable in size
  MATCH(sizeof(decltype(dv_3darray_storage)),
      Descriptor::SizeInBytes(rank, false));
  MATCH(sizeof(decltype(dv_scalar_storage)), Descriptor::SizeInBytes(0, false));
  // suitable in alignment
  MATCH(reinterpret_cast<std::uintptr_t>(&dv_3darray_storage) &
          (alignof(CFI_cdesc_t) - 1),
      0);
  MATCH(reinterpret_cast<std::uintptr_t>(&dv_scalar_storage) &
          (alignof(CFI_cdesc_t) - 1),
      0);

  // Testing CFI_establish definied in section 18.5.5
  for (CFI_attribute_t attribute : attr_cases) {
    for (CFI_type_t type : type_cases) {
      add_noise_to_cdesc(dv_3darray, rank);
      ret_code = check_CFI_establish(
          dv_3darray, nullptr, attribute, type, 42, rank, extents);
      MATCH(ret_code, CFI_SUCCESS);
      add_noise_to_cdesc(dv_scalar, 0);
      ret_code = check_CFI_establish(
          dv_scalar, nullptr, attribute, type, 42, 0, extents);
      MATCH(ret_code, CFI_SUCCESS);
    }
  }
  // If base_addr is null, extents shall be ignored even if rank !=0
  add_noise_to_cdesc(dv_3darray, rank);  // => dv_3darray->dim[2].extent = -42
  ret_code = check_CFI_establish(
      dv_3darray, nullptr, CFI_attribute_other, CFI_type_int, 4, rank, extents);
  MATCH(false,
      dv_3darray->dim[2].extent ==
          66);  // extents was read even if could have been null

  static char base;
  void *base_addr = reinterpret_cast<void *>(&base);
  // arg base_addr shall be null if the attribute arg is allocatable, else OK
  add_noise_to_cdesc(dv_3darray, rank);
  ret_code = check_CFI_establish(dv_3darray, base_addr,
      CFI_attribute_allocatable, CFI_type_int, 4, rank, extents);
  MATCH(true, ret_code == CFI_ERROR_BASE_ADDR_NOT_NULL);
  add_noise_to_cdesc(dv_scalar, 0);
  ret_code = check_CFI_establish(dv_3darray, base_addr,
      CFI_attribute_allocatable, CFI_type_int, 4, rank, extents);
  MATCH(true, ret_code == CFI_ERROR_BASE_ADDR_NOT_NULL);

  for (CFI_attribute_t attribute : nonalloctable_cases) {
    for (CFI_type_t type : type_cases) {
      add_noise_to_cdesc(dv_3darray, rank);
      ret_code = check_CFI_establish(
          dv_3darray, base_addr, attribute, type, 42, rank, extents);
      MATCH(ret_code, CFI_SUCCESS);
      add_noise_to_cdesc(dv_scalar, 0);
      ret_code = check_CFI_establish(
          dv_scalar, base_addr, attribute, type, 42, 0, nullptr);
      MATCH(ret_code, CFI_SUCCESS);
    }
  }
  // TODO: check invalid rank and type

  typedef struct {
    double x;
    double _Complex y;
  } t;
  // TODO: test CFI_address
  // TODO: test CFI_allocate
  // TODO: test CFI_deallocate
  // TODO: test CFI_establish
  // TODO: test CFI_is_contiguous
  // TODO: test CFI_section
  // TODO: test CFI_select_part
  // TODO: test CFI_setpointer
  return testing::Complete();
}
