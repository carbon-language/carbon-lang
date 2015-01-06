//===- lld/Driver/WrapperInputGraph.h - dummy InputGraph node -------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_DRIVER_WRAPPER_INPUT_GRAPH_H
#define LLD_DRIVER_WRAPPER_INPUT_GRAPH_H

#include "lld/Core/InputGraph.h"
#include "lld/ReaderWriter/CoreLinkingContext.h"
#include "lld/ReaderWriter/Reader.h"
#include "llvm/Support/Errc.h"
#include <map>
#include <memory>

namespace lld {

class WrapperNode : public FileNode {
public:
  WrapperNode(std::unique_ptr<File> file) : FileNode(file->path()) {
    _files.push_back(std::move(file));
  }
};

}

#endif
