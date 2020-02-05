//===-- runtime/internal-unit.h ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Fortran internal I/O "units"

#ifndef FORTRAN_RUNTIME_IO_INTERNAL_UNIT_H_
#define FORTRAN_RUNTIME_IO_INTERNAL_UNIT_H_

#include "connection.h"
#include "descriptor.h"
#include <cinttypes>
#include <type_traits>

namespace Fortran::runtime::io {

class IoErrorHandler;

// Points to (but does not own) a CHARACTER scalar or array for internal I/O.
// Does not buffer.
template<bool isInput> class InternalDescriptorUnit : public ConnectionState {
public:
  using Scalar = std::conditional_t<isInput, const char *, char *>;
  InternalDescriptorUnit(Scalar, std::size_t);
  InternalDescriptorUnit(const Descriptor &, const Terminator &);
  void EndIoStatement();

  bool Emit(const char *, std::size_t bytes, IoErrorHandler &);
  bool AdvanceRecord(IoErrorHandler &);
  bool HandleAbsolutePosition(std::int64_t, IoErrorHandler &);
  bool HandleRelativePosition(std::int64_t, IoErrorHandler &);

private:
  Descriptor &descriptor() { return staticDescriptor_.descriptor(); }
  StaticDescriptor<maxRank, true /*addendum*/> staticDescriptor_;
  SubscriptValue at_[maxRank];
};

extern template class InternalDescriptorUnit<false>;
extern template class InternalDescriptorUnit<true>;
}
#endif  // FORTRAN_RUNTIME_IO_INTERNAL_UNIT_H_
