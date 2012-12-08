//===- lld/Driver/LinkerInvocation.h - Linker Invocation ------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
///
/// Drives the actual link.
///
//===----------------------------------------------------------------------===//

#ifndef LLD_DRIVER_LINKER_INVOCATION_H
#define LLD_DRIVER_LINKER_INVOCATION_H

#include "lld/Driver/LinkerOptions.h"

namespace lld {
class LinkerInvocation {
public:
  LinkerInvocation(const LinkerOptions &lo) : _options(lo) {}

  /// \brief Perform the link.
  void operator()();

private:
  const LinkerOptions &_options;
};
}

#endif
