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
#include <type_traits>
#ifdef VERBOSE
#include <iostream>
#endif

using namespace Fortran::runtime;
using namespace Fortran::ISO;

// CFI_CDESC_T test helpers
template<int rank> class Test_CFI_CDESC_T {
public:
  Test_CFI_CDESC_T() : dvStorage_{0} {};
  ~Test_CFI_CDESC_T(){};
  void Check() {
    // Test CFI_CDESC_T macro defined in section 18.5.4 of F2018 standard
    // CFI_CDESC_T must give storage that is:
    using type = decltype(dvStorage_);
    // unqualified
    MATCH(false, std::is_const<type>::value);
    MATCH(false, std::is_volatile<type>::value);
    // suitable in size
    if (rank > 0) {
      MATCH(sizeof(dvStorage_), Descriptor::SizeInBytes(rank_, false));
    } else {  // C++ implem overalocates for rank=0 by 24bytes.
      MATCH(true, sizeof(dvStorage_) >= Descriptor::SizeInBytes(rank_, false));
    }
    // suitable in alignment
    MATCH(0,
        reinterpret_cast<std::uintptr_t>(&dvStorage_) &
            (alignof(CFI_cdesc_t) - 1));
  }

private:
  static constexpr int rank_ = rank;
  CFI_CDESC_T(rank) dvStorage_;
};

template<int rank> static void TestForAllSmallerRanks() {
  static_assert(rank > 0, "rank<0!");
  Test_CFI_CDESC_T<rank> obj;
  obj.Check();
  TestForAllSmallerRanks<rank - 1>();
}
template<> void TestForAllSmallerRanks<0>() {
  Test_CFI_CDESC_T<0> obj;
  obj.Check();
}

// CFI_establish test helper
static void AddNoiseToCdesc(CFI_cdesc_t *dv, CFI_rank_t rank) {
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

#ifdef VERBOSE
static void DumpTestWorld(void *bAddr, CFI_attribute_t attr, CFI_type_t ty,
    std::size_t eLen, CFI_rank_t rank, CFI_index_t *eAddr) {
  std::cout << " base_addr: " << std::hex
            << reinterpret_cast<std::intptr_t>(bAddr)
            << " attribute: " << static_cast<int>(attr) << std::dec
            << " type: " << static_cast<int>(ty) << " elem_len: " << eLen
            << " rank: " << static_cast<int>(rank) << " extent: " << std::hex
            << reinterpret_cast<std::intptr_t>(eAddr) << std::endl
            << std::dec;
}
#endif

static void check_CFI_establish(CFI_cdesc_t *dv, void *base_addr,
    CFI_attribute_t attribute, CFI_type_t type, std::size_t elem_len,
    CFI_rank_t rank, const CFI_index_t extents[]) {
#ifdef VERBOSE
  DumpTestWorld(base_addr, attribute, type, elem_len, rank, extent);
#endif
  // CFI_establish reqs from F2018 section 18.5.5
  int retCode =
      CFI_establish(dv, base_addr, attribute, type, elem_len, rank, extents);
  Descriptor *res = reinterpret_cast<Descriptor *>(dv);
  if (retCode == CFI_SUCCESS) {
    res->Check();
    MATCH((attribute == CFI_attribute_pointer), res->IsPointer());
    MATCH((attribute == CFI_attribute_allocatable), res->IsAllocatable());
    MATCH(rank, res->rank());
    MATCH(reinterpret_cast<std::intptr_t>(dv->base_addr),
        reinterpret_cast<std::intptr_t>(base_addr));
    if (base_addr != nullptr) {
      MATCH(true, res->IsContiguous());
      for (int i{0}; i < rank; ++i) {
        MATCH(extents[i], res->GetDimension(i).Extent());
      }
    }
    if (attribute == CFI_attribute_allocatable) {
      MATCH(res->IsAllocated(), false);
    }
    if (attribute == CFI_attribute_pointer) {
      if (base_addr != nullptr) {
        for (int i{0}; i < rank; ++i) {
          MATCH(0, res->GetDimension(i).LowerBound());
        }
      }
    }
    if (type == CFI_type_struct || type == CFI_type_char ||
        type == CFI_type_other) {
      MATCH(elem_len, res->ElementBytes());
    }
  }
  // Checking failure/success according to combination of args forbidden by the
  // standard:
  int numErr{0};
  int expectedRetCode{CFI_SUCCESS};
  if (base_addr != nullptr && attribute == CFI_attribute_allocatable) {
    ++numErr;
    expectedRetCode = CFI_ERROR_BASE_ADDR_NOT_NULL;
  }
  if (rank < 0 || rank > CFI_MAX_RANK) {
    ++numErr;
    expectedRetCode = CFI_INVALID_RANK;
  }
  if (type < 0 || type > CFI_type_struct) {
    ++numErr;
    expectedRetCode = CFI_INVALID_TYPE;
  }

  if ((type == CFI_type_struct || type == CFI_type_char ||
          type == CFI_type_other) &&
      elem_len <= 0) {
    ++numErr;
    expectedRetCode = CFI_INVALID_ELEM_LEN;
  }
  if (rank > 0 && base_addr != nullptr && extents == nullptr) {
    ++numErr;
    expectedRetCode = CFI_INVALID_EXTENT;
  }
  if (numErr > 1) {
    MATCH(true, retCode != CFI_SUCCESS);
  } else {
    MATCH(retCode, expectedRetCode);
  }
}

int main() {
  // Testing CFI_CDESC_T macro for rank CFI_MAX_Rank to 0
  TestForAllSmallerRanks<CFI_MAX_RANK>();

  // Testing CFI_establish definied in section 18.5.5
  CFI_index_t extents[CFI_MAX_RANK];
  for (int i{0}; i < CFI_MAX_RANK; ++i) {
    extents[i] = i + 66;
  }
  CFI_CDESC_T(CFI_MAX_RANK) dv_storage;
  CFI_cdesc_t *dv{reinterpret_cast<CFI_cdesc_t *>(&dv_storage)};
  static char base;
  void *dummyAddr = reinterpret_cast<void *>(&base);
  // Define test space
  CFI_attribute_t attrCases[]{
      CFI_attribute_pointer, CFI_attribute_allocatable, CFI_attribute_other};
  CFI_type_t typeCases[]{CFI_type_int, CFI_type_struct, CFI_type_double,
      CFI_type_char, CFI_type_other, CFI_type_struct + 1};
  CFI_index_t *extentCases[]{extents, static_cast<CFI_index_t *>(nullptr)};
  void *baseAddrCases[]{dummyAddr, static_cast<void *>(nullptr)};
  CFI_rank_t rankCases[]{0, 1, CFI_MAX_RANK, CFI_MAX_RANK + 1};
  std::size_t lenCases[]{0, 42};

  for (CFI_attribute_t attribute : attrCases) {
    for (void *base_addr : baseAddrCases) {
      for (CFI_index_t *extent : extentCases) {
        for (CFI_rank_t rank : rankCases) {
          for (CFI_type_t type : typeCases) {
            for (size_t elem_len : lenCases) {
              AddNoiseToCdesc(dv, CFI_MAX_RANK);
              check_CFI_establish(
                  dv, base_addr, attribute, type, elem_len, rank, extent);
            }
          }
        }
      }
    }
  }
  // If base_addr is null, extents shall be ignored even if rank !=0
  const int rank3d{3};
  static CFI_CDESC_T(rank3d) dv3darrayStorage;
  CFI_cdesc_t *dv_3darray{reinterpret_cast<CFI_cdesc_t *>(&dv3darrayStorage)};
  AddNoiseToCdesc(dv_3darray, rank3d);  // => dv_3darray->dim[2].extent = -42
  check_CFI_establish(dv_3darray, nullptr, CFI_attribute_other, CFI_type_int, 4,
      rank3d, extents);
  MATCH(false,
      dv_3darray->dim[2].extent == 2 + 66);  // extents was read

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
