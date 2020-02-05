//===-- runtime/connection.cpp ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "connection.h"
#include "environment.h"

namespace Fortran::runtime::io {

std::size_t ConnectionState::RemainingSpaceInRecord() const {
  return recordLength.value_or(
             executionEnvironment.listDirectedOutputLineLengthLimit) -
      positionInRecord;
}
}
