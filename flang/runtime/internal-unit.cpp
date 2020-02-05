//===-- runtime/internal-unit.cpp -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "internal-unit.h"
#include "descriptor.h"
#include "io-error.h"
#include <algorithm>
#include <type_traits>

namespace Fortran::runtime::io {

template<bool isInput>
InternalDescriptorUnit<isInput>::InternalDescriptorUnit(
    Scalar scalar, std::size_t length) {
  recordLength = length;
  endfileRecordNumber = 2;
  void *pointer{reinterpret_cast<void *>(const_cast<char *>(scalar))};
  descriptor().Establish(TypeCode{CFI_type_char}, length, pointer, 0, nullptr,
      CFI_attribute_pointer);
}

template<bool isInput>
InternalDescriptorUnit<isInput>::InternalDescriptorUnit(
    const Descriptor &that, const Terminator &terminator) {
  RUNTIME_CHECK(terminator, that.type().IsCharacter());
  Descriptor &d{descriptor()};
  RUNTIME_CHECK(
      terminator, that.SizeInBytes() <= d.SizeInBytes(maxRank, true, 0));
  new (&d) Descriptor{that};
  d.Check();
  recordLength = d.ElementBytes();
  endfileRecordNumber = d.Elements() + 1;
  d.GetLowerBounds(at_);
}

template<bool isInput> void InternalDescriptorUnit<isInput>::EndIoStatement() {
  if constexpr (!isInput) {
    // blank fill
    while (currentRecordNumber < endfileRecordNumber.value_or(0)) {
      char *record{descriptor().template Element<char>(at_)};
      std::fill_n(record + furthestPositionInRecord,
          recordLength.value_or(0) - furthestPositionInRecord, ' ');
      furthestPositionInRecord = 0;
      ++currentRecordNumber;
      descriptor().IncrementSubscripts(at_);
    }
  }
}

template<bool isInput>
bool InternalDescriptorUnit<isInput>::Emit(
    const char *data, std::size_t bytes, IoErrorHandler &handler) {
  if constexpr (isInput) {
    handler.Crash(
        "InternalDescriptorUnit<true>::Emit() called for an input statement");
    return false;
  }
  if (currentRecordNumber >= endfileRecordNumber.value_or(0)) {
    handler.SignalEnd();
    return false;
  }
  char *record{descriptor().template Element<char>(at_)};
  auto furthestAfter{std::max(furthestPositionInRecord,
      positionInRecord + static_cast<std::int64_t>(bytes))};
  bool ok{true};
  if (furthestAfter > static_cast<std::int64_t>(recordLength.value_or(0))) {
    handler.SignalEor();
    furthestAfter = recordLength.value_or(0);
    bytes = std::max(std::int64_t{0}, furthestAfter - positionInRecord);
    ok = false;
  }
  std::memcpy(record + positionInRecord, data, bytes);
  positionInRecord += bytes;
  furthestPositionInRecord = furthestAfter;
  return ok;
}

template<bool isInput>
bool InternalDescriptorUnit<isInput>::AdvanceRecord(IoErrorHandler &handler) {
  if (currentRecordNumber >= endfileRecordNumber.value_or(0)) {
    handler.SignalEnd();
    return false;
  }
  if (!HandleAbsolutePosition(recordLength.value_or(0), handler)) {
    return false;
  }
  ++currentRecordNumber;
  descriptor().IncrementSubscripts(at_);
  positionInRecord = 0;
  furthestPositionInRecord = 0;
  return true;
}

template<bool isInput>
bool InternalDescriptorUnit<isInput>::HandleAbsolutePosition(
    std::int64_t n, IoErrorHandler &handler) {
  n = std::max<std::int64_t>(0, n);
  bool ok{true};
  if (n > static_cast<std::int64_t>(recordLength.value_or(n))) {
    handler.SignalEor();
    n = *recordLength;
    ok = false;
  }
  if (n > furthestPositionInRecord && ok) {
    if constexpr (!isInput) {
      char *record{descriptor().template Element<char>(at_)};
      std::fill_n(
          record + furthestPositionInRecord, n - furthestPositionInRecord, ' ');
    }
    furthestPositionInRecord = n;
  }
  positionInRecord = n;
  return ok;
}

template<bool isInput>
bool InternalDescriptorUnit<isInput>::HandleRelativePosition(
    std::int64_t n, IoErrorHandler &handler) {
  return HandleAbsolutePosition(positionInRecord + n, handler);
}

template class InternalDescriptorUnit<false>;
template class InternalDescriptorUnit<true>;
}
