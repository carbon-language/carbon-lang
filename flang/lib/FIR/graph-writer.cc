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

#include "graph-writer.h"

namespace Fortran::FIR {

std::optional<llvm::raw_ostream *> GraphWriter::defaultOutput_{std::nullopt};

void GraphWriter::dumpHeader() { output_ << "digraph G {\n"; }
void GraphWriter::dumpFooter() { output_ << "}\n"; }

void GraphWriter::dump(Program &program) {
  dumpHeader();
  for (auto iter{program.procedureMap_.begin()},
       iend{program.procedureMap_.end()};
       iter != iend; ++iter) {
    dump(*iter->getValue(), true);
  }
  dumpFooter();
}

void GraphWriter::dump(Procedure &procedure, bool box) {
  if (box) {
    output_ << "subgraph cluster" << counter()
            << " {\n  node[style=filled];\n  color=red;\n";
  }
  for (auto iter{procedure.regionList_.begin()},
       iend{procedure.regionList_.end()};
       iter != iend; ++iter) {
    dump(*iter, true);
  }
  if (box) {
    output_ << "  label = \"procedure";
    if (procedure.getName().empty()) {
      output_ << '#' << counter();
    } else {
      output_ << ": " << procedure.getName().str();
    }
    output_ << "\"\n}\n";
  }
}

void GraphWriter::dump(Region &region, bool box) {
  if (box) {
    output_ << "  subgraph cluster" << counter()
            << " {\n    node[style=filled];\n";
  }
  for (auto iter{region.begin()}, iend{region.end()}; iter != iend; ++iter) {
    dump(*iter, true);
  }
  std::set<BasicBlock *> myNodes;
  auto blocks{region.getBlocks()};
  auto iend{blocks.end()};
  auto iexit{iend};
  --iexit;
  auto ientry{blocks.begin()};
  for (auto iter{ientry}; iter != iend; ++iter) {
    isEntry_ = iter == ientry && region.IsOutermost();
    isExit_ = iter == iexit && region.IsOutermost();
    dump(**iter);
    myNodes.insert(*iter);
  }
  std::list<std::pair<BasicBlock *, BasicBlock *>> emitAfter;
  for (auto iter{blocks.begin()}, iend{blocks.end()}; iter != iend; ++iter) {
    dumpInternalEdges(**iter, myNodes, emitAfter);
  }
  if (box) {
    output_ << "    style=dashed;\n    color=blue;\n    label = \"region#"
            << counter() << "\\nvariables: {...}\\n\"\n  }\n";
  }
  for (auto pair : emitAfter) {
    output_ << "  " << block_id(*pair.first) << " -> " << block_id(*pair.second)
            << ";\n";
  }
}

void GraphWriter::dump(BasicBlock &block, std::optional<const char *> color) {
  output_ << "    " << block_id(block) << " [label = \"";
  if (isEntry_) {
    output_ << "<<ENTRY>>\\n";
  }
  output_ << block_id(block) << '(' << reinterpret_cast<std::intptr_t>(&block)
          << ")\\n";
  for (auto &action : block.getSublist(static_cast<Statement *>(nullptr))) {
    output_ << action.dump() << "\\n";
  }
  if (isExit_) {
    output_ << "<<EXIT>>";
  }
  output_ << "\",shape=rectangle";
  if (color) {
    output_ << ",color=" << *color;
  }
  output_ << "];\n";
}

void GraphWriter::dumpInternalEdges(BasicBlock &block,
    std::set<BasicBlock *> &nodeSet,
    std::list<std::pair<BasicBlock *, BasicBlock *>> &emitAfter) {
  for (auto succ : succ_list(block)) {
    if (nodeSet.count(succ)) {
      output_ << "    " << block_id(block) << " -> " << block_id(*succ)
              << ";\n";
    } else {
      emitAfter.push_back({&block, succ});
    }
  }
}

}
