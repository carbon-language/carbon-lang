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

#include "descriptor.h"
#include "../lib/common/idioms.h"
#include "../lib/evaluate/integer.h"
#include <algorithm>
#include <bitset>
#include <cinttypes>
#include <cstdlib>

namespace Fortran::runtime {

template<int BITS> inline std::int64_t LoadInt64(const char *p) {
  using Int = const evaluate::value::Integer<BITS>;
  Int *ip{reinterpret_cast<Int *>(p)};
  return ip->ToInt64();
}

static inline std::int64_t GetInt64(const char *p, std::size_t bytes) {
  switch (bytes) {
  case 1: return LoadInt64<8>(p);
  case 2: return LoadInt64<16>(p);
  case 4: return LoadInt64<32>(p);
  case 8: return LoadInt64<64>(p);
  default: CRASH_NO_CASE;
  }
}

// F2018 16.9.163
Descriptor *RESHAPE(const Descriptor &source, const Descriptor &shape,
    const Descriptor *pad, const Descriptor *order) {
  // Compute and check the rank of the result.
  CHECK(shape.rank() == 1);
  CHECK(shape.type().IsInteger());
  SubscriptValue resultRank{shape.GetDimension(0).Extent()};
  CHECK(resultRank >= 0 && resultRank <= static_cast<SubscriptValue>(maxRank));

  // Extract and check the shape of the result; compute its element count.
  SubscriptValue resultExtent[maxRank];
  std::size_t shapeElementBytes{shape.ElementBytes()};
  std::size_t resultElements{1};
  SubscriptValue shapeSubscript{shape.GetDimension(0).LowerBound()};
  for (SubscriptValue j{0}; j < resultRank; ++j, ++shapeSubscript) {
    resultExtent[j] =
        GetInt64(shape.Element<char>(&shapeSubscript), shapeElementBytes);
    CHECK(resultExtent[j] >= 0);
    resultElements *= resultExtent[j];
  }

  // Check that there are sufficient elements in the SOURCE=, or that
  // the optional PAD= argument is present and nonempty.
  std::size_t sourceElements{source.Elements()};
  std::size_t padElements{pad ? pad->Elements() : 0};
  if (resultElements < sourceElements) {
    CHECK(padElements > 0);
    CHECK(pad->ElementBytes() == source.ElementBytes());
  }

  // Extract and check the optional ORDER= argument, which must be a
  // permutation of [1..resultRank].
  int dimOrder[maxRank];
  if (order != nullptr) {
    CHECK(order->rank() == 1);
    CHECK(order->type().IsInteger());
    CHECK(order->GetDimension(0).Extent() == resultRank);
    std::bitset<maxRank> values;
    SubscriptValue orderSubscript{order->GetDimension(0).LowerBound()};
    for (SubscriptValue j{0}; j < resultRank; ++j, ++orderSubscript) {
      auto k{GetInt64(order->Element<char>(orderSubscript), shapeElementBytes)};
      CHECK(k >= 1 && k <= resultRank && !values.test(k - 1));
      values.set(k - 1);
      dimOrder[k - 1] = j;
    }
  } else {
    for (int j{0}; j < resultRank; ++j) {
      dimOrder[j] = j;
    }
  }

  // Allocate the result's data storage.
  std::size_t elementBytes{source.ElementBytes()};
  std::size_t resultBytes{resultElements * elementBytes};
  void *data{std::malloc(resultBytes)};
  CHECK(resultBytes == 0 || data != nullptr);

  // Create and populate the result's descriptor.
  const DescriptorAddendum *sourceAddendum{source.Addendum()};
  const DerivedType *sourceDerivedType{
      sourceAddendum ? sourceAddendum->derivedType() : nullptr};
  Descriptor *result{nullptr};
  if (sourceDerivedType != nullptr) {
    result =
        Descriptor::Create(*sourceDerivedType, data, resultRank, resultExtent);
  } else {
    result = Descriptor::Create(
        source.type(), elementBytes, data, resultRank, resultExtent);
  }
  DescriptorAddendum *resultAddendum{result->Addendum()};
  CHECK(resultAddendum != nullptr);
  resultAddendum->flags() |= DescriptorAddendum::DoNotFinalize;
  resultAddendum->flags() |= DescriptorAddendum::AllContiguous;
  if (sourceDerivedType != nullptr) {
    std::size_t lenParameters{sourceDerivedType->lenParameters()};
    for (std::size_t j{0}; j < lenParameters; ++j) {
      resultAddendum->SetLenParameterValue(
          j, sourceAddendum->LenParameterValue(j));
    }
  }

  // Populate the result's elements.
  SubscriptValue resultSubscript[maxRank];
  result->GetLowerBounds(resultSubscript);
  SubscriptValue sourceSubscript[maxRank];
  source.GetLowerBounds(sourceSubscript);
  std::size_t resultElement{0};
  std::size_t elementsFromSource{std::min(resultElements, sourceElements)};
  for (; resultElement < elementsFromSource; ++resultElement) {
    std::memcpy(result->Element<void>(resultSubscript),
        source.Element<const void>(sourceSubscript), elementBytes);
    source.IncrementSubscripts(sourceSubscript);
    result->IncrementSubscripts(resultSubscript, dimOrder);
  }
  if (resultElement < resultElements) {
    // Remaining elements come from the optional PAD= argument.
    SubscriptValue padSubscript[maxRank];
    pad->GetLowerBounds(padSubscript);
    for (; resultElement < resultElements; ++resultElement) {
      std::memcpy(result->Element<void>(resultSubscript),
          pad->Element<const void>(padSubscript), elementBytes);
      pad->IncrementSubscripts(padSubscript);
      result->IncrementSubscripts(resultSubscript, dimOrder);
    }
  }

  return result;
}

}  // namespace Fortran::runtime
