//===- lib/Core/InputGraph.cpp --------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lld/Core/InputGraph.h"
#include "lld/Core/Resolver.h"
#include <memory>

using namespace lld;

std::error_code FileNode::parse(const LinkingContext &, raw_ostream &) {
  if (_file)
    if (std::error_code ec = _file->parse())
      return ec;
  return std::error_code();
}
