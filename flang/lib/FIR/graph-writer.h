// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

#ifndef FORTRAN_FIR_GRAPH_WRITER_H_
#define FORTRAN_FIR_GRAPH_WRITER_H_

#include "program.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include <list>
#include <map>
#include <set>
#include <sstream>
#include <string>

namespace Fortran::FIR {

struct GraphWriter {
  static void setOutput(llvm::raw_ostream *output) { defaultOutput_ = output; }
  static void setOutput(const std::string &filename) {
    std::error_code ec;
    setOutput(new llvm::raw_fd_ostream(filename, ec, llvm::sys::fs::F_None));
    CHECK(!ec);
  }
  static void print(Program &program) {
    GraphWriter writer{getOutput()};
    writer.dump(program);
  }
  static void print(Procedure &procedure) {
    GraphWriter writer{getOutput()};
    writer.dump(procedure);
  }
  static void print(Region &region) {
    GraphWriter writer{getOutput()};
    writer.dump(region);
  }

private:
  GraphWriter(llvm::raw_ostream &output) : output_{output} {}
  ~GraphWriter() {
    if (defaultOutput_) {
      delete *defaultOutput_;
      defaultOutput_ = std::nullopt;
    }
  }
  void dump(Program &program);
  void dump(Procedure &procedure, bool box = false);
  void dump(Region &region, bool box = false);
  void dumpHeader();
  void dumpFooter();
  unsigned counter() { return count_++; }
  void dump(
      BasicBlock &block, std::optional<const char *> color = std::nullopt);
  void dumpInternalEdges(BasicBlock &block, std::set<BasicBlock *> &nodeSet,
      std::list<std::pair<BasicBlock *, BasicBlock *>> &emitAfter);
  std::string block_id(BasicBlock &block) {
    unsigned num;
    if (blockIds_.count(&block)) {
      num = blockIds_[&block];
    } else {
      blockIds_[&block] = num = blockNum_++;
    }
    std::ostringstream buffer;
    buffer << "BB_" << num;
    return buffer.str();
  }
  static llvm::raw_ostream &getOutput() {
    return defaultOutput_ ? *defaultOutput_.value() : llvm::outs();
  }

  unsigned count_{0u};
  llvm::raw_ostream &output_;
  unsigned blockNum_{0u};
  bool isEntry_{false};
  bool isExit_{false};
  std::map<BasicBlock *, unsigned> blockIds_;
  static std::optional<llvm::raw_ostream *> defaultOutput_;
};

}

#endif
