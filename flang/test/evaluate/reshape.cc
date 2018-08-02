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
#include "../../runtime/descriptor.h"
#include "../../runtime/transformational.h"
#include <cinttypes>

using namespace Fortran::common;
using namespace Fortran::runtime;

int main() {
  std::size_t dataElements{24};
  std::int32_t *data{new std::int32_t[dataElements]};
  for (std::size_t j{0}; j < dataElements; ++j) {
    data[j] = j;
  }

  static const SubscriptValue sourceExtent[]{2, 3, 4};
  Descriptor *source{Descriptor::Create(TypeCategory::Integer, sizeof data[0],
      reinterpret_cast<void *>(data), 3, sourceExtent,
      CFI_attribute_allocatable)};
  source->Check();
  MATCH(3, source->rank());
  MATCH(2, source->GetDimension(0).Extent());
  MATCH(3, source->GetDimension(1).Extent());
  MATCH(4, source->GetDimension(2).Extent());

  static const std::int16_t shapeData[]{8, 4};
  static const SubscriptValue shapeExtent{2};
  Descriptor *shape{Descriptor::Create(TypeCategory::Integer,
      static_cast<int>(sizeof shapeData[0]),
      const_cast<void *>(reinterpret_cast<const void *>(shapeData)), 1,
      &shapeExtent)};
  shape->Check();
  MATCH(1, shape->rank());
  MATCH(2, shape->GetDimension(0).Extent());

  StaticDescriptor<3> padDescriptor;
  static const std::int32_t padData[]{24, 25, 26, 27, 28, 29, 30, 31};
  static const SubscriptValue padExtent[]{2, 2, 3};
  padDescriptor.descriptor().Establish(TypeCategory::Integer,
      static_cast<int>(sizeof padData[0]),
      const_cast<void *>(reinterpret_cast<const void *>(padData)), 3,
      padExtent);
  padDescriptor.Check();

  Descriptor *result{RESHAPE(*source, *shape, &padDescriptor.descriptor())};

  TEST(result != nullptr);
  result->Check();
  MATCH(sizeof(std::int32_t), result->ElementBytes());
  MATCH(2, result->rank());
  TEST(result->type().IsInteger());
  for (std::int32_t j{0}; j < 32; ++j) {
    MATCH(j, *result->Element<std::int32_t>(j * sizeof(std::int32_t)));
  }
  for (std::int32_t j{0}; j < 32; ++j) {
    SubscriptValue ss[2]{1 + (j % 8), 1 + (j / 8)};
    MATCH(j, *result->Element<std::int32_t>(ss));
  }

  // TODO: test ORDER=

  // Plug leaks; should run cleanly beneath valgrind
  free(result->raw().base_addr);
  result->Destroy();
  shape->Destroy();
  source->Destroy();
  delete[] data;

  return testing::Complete();
}
