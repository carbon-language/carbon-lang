//===- LTO.h ----------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides a way to combine bitcode files into one COFF
// file by compiling them using LLVM.
//
// If LTO is in use, your input files are not in regular COFF files
// but instead LLVM bitcode files. In that case, the linker has to
// convert bitcode files into the native format so that we can create
// a COFF file that contains native code. This file provides that
// functionality.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_COFF_LTO_H
#define LLD_COFF_LTO_H

#include "lld/Common/LLVM.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <vector>

namespace llvm {
namespace lto {
class LTO;
}
}

namespace lld {
namespace coff {

class BitcodeFile;
class InputFile;
class COFFLinkerContext;

class BitcodeCompiler {
public:
  BitcodeCompiler();
  ~BitcodeCompiler();

  void add(BitcodeFile &f);
  std::vector<InputFile *> compile(COFFLinkerContext &ctx);

private:
  std::unique_ptr<llvm::lto::LTO> ltoObj;
  std::vector<SmallString<0>> buf;
  std::vector<std::unique_ptr<MemoryBuffer>> files;
  std::unique_ptr<llvm::raw_fd_ostream> indexFile;
  llvm::DenseSet<StringRef> thinIndices;
};
}
}

#endif
