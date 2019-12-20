//===-- lib/evaluate/intrinsics.h -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//----------------------------------------------------------------------------//

#ifndef FORTRAN_EVALUATE_INTRINSICS_H_
#define FORTRAN_EVALUATE_INTRINSICS_H_

#include "call.h"
#include "characteristics.h"
#include "type.h"
#include "../common/default-kinds.h"
#include "../parser/char-block.h"
#include "../parser/message.h"
#include <optional>
#include <ostream>
#include <string>

namespace Fortran::evaluate {

class FoldingContext;

struct CallCharacteristics {
  std::string name;
  bool isSubroutineCall{false};
};

struct SpecificCall {
  SpecificCall(SpecificIntrinsic &&si, ActualArguments &&as)
    : specificIntrinsic{std::move(si)}, arguments{std::move(as)} {}
  SpecificIntrinsic specificIntrinsic;
  ActualArguments arguments;
};

struct UnrestrictedSpecificIntrinsicFunctionInterface
  : public characteristics::Procedure {
  UnrestrictedSpecificIntrinsicFunctionInterface(
      characteristics::Procedure &&p, std::string n)
    : characteristics::Procedure{std::move(p)}, genericName{n} {}
  std::string genericName;
  // N.B. If there are multiple arguments, they all have the same type.
  // All argument and result types are intrinsic types with default kinds.
};

class IntrinsicProcTable {
private:
  class Implementation;

public:
  ~IntrinsicProcTable();
  static IntrinsicProcTable Configure(
      const common::IntrinsicTypeDefaultKinds &);

  // Check whether a name should be allowed to appear on an INTRINSIC
  // statement.
  bool IsIntrinsic(const std::string &) const;

  // Probe the intrinsics for a match against a specific call.
  // On success, the actual arguments are transferred to the result
  // in dummy argument order; on failure, the actual arguments remain
  // untouched.
  std::optional<SpecificCall> Probe(
      const CallCharacteristics &, ActualArguments &, FoldingContext &) const;

  // Probe the intrinsics with the name of a potential unrestricted specific
  // intrinsic.
  std::optional<UnrestrictedSpecificIntrinsicFunctionInterface>
  IsUnrestrictedSpecificIntrinsicFunction(const std::string &) const;

  std::ostream &Dump(std::ostream &) const;

private:
  Implementation *impl_{nullptr};  // owning pointer
};
}
#endif  // FORTRAN_EVALUATE_INTRINSICS_H_
