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

void InputGraph::addInputElement(std::unique_ptr<InputElement> ie) {
  _inputArgs.push_back(std::move(ie));
}

void InputGraph::addInputElementFront(std::unique_ptr<InputElement> ie) {
  _inputArgs.insert(_inputArgs.begin(), std::move(ie));
}

std::error_code FileNode::parse(const LinkingContext &, raw_ostream &) {
  if (_file)
    if (std::error_code ec = _file->parse())
      return ec;
  return std::error_code();
}
