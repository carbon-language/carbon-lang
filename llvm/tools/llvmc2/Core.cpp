//===--- Core.cpp - The LLVM Compiler Driver --------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open
// Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Core driver abstractions.
//
//===----------------------------------------------------------------------===//

#include "Core.h"
#include "Utility.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>

using namespace llvm;
using namespace llvmcc;

extern cl::list<std::string> InputFilenames;
extern cl::opt<std::string> OutputFilename;
extern cl::opt<bool> VerboseMode;

namespace {
  void print_string (const std::string& str) {
    std::cerr << str << ' ';
  }
}

int llvmcc::Action::Execute() {
  if (VerboseMode) {
    std::cerr << Command_ << " ";
    std::for_each(Args_.begin(), Args_.end(), print_string);
    std::cerr << '\n';
  }
  return ExecuteProgram(Command_, Args_);
}

int llvmcc::CompilationGraph::Build (const sys::Path& tempDir) const {
  sys::Path In(InputFilenames.at(0)), Out;

  // Find out which language corresponds to the suffix of the first input file
  LanguageMap::const_iterator Lang = ExtsToLangs.find(In.getSuffix());
  if (Lang == ExtsToLangs.end())
    throw std::runtime_error("Unknown suffix!");

  // Find the toolchain corresponding to this language
  ToolChainMap::const_iterator ToolsIt = ToolChains.find(Lang->second);
  if (ToolsIt == ToolChains.end())
    throw std::runtime_error("Unknown language!");
  ToolChain Tools = ToolsIt->second;

  PathVector JoinList;

  for (cl::list<std::string>::const_iterator B = InputFilenames.begin(),
        E = InputFilenames.end(); B != E; ++B) {
    In = sys::Path(*B);

    // Pass input file through the toolchain
    for (ToolChain::const_iterator B = Tools.begin(), E = Tools.end();
         B != E; ++B) {

      const Tool* CurTool = B->getPtr();

      // Is this the last step in the chain?
      if (llvm::next(B) == E || CurTool->IsLast()) {
        JoinList.push_back(In);
        break;
      }
      else {
        Out = tempDir;
        Out.appendComponent(In.getBasename());
        Out.appendSuffix(CurTool->OutputSuffix());
        Out.makeUnique(true, NULL);
        Out.eraseFromDisk();
      }

      if (CurTool->GenerateAction(In, Out).Execute() != 0)
        throw std::runtime_error("Tool returned error code!");

      In = Out; Out.clear();
    }
  }

  // Pass .o files to linker
  const Tool* JoinNode = (--Tools.end())->getPtr();

  // If the final output name is empty, set it to "a.out"
  if (!OutputFilename.empty()) {
    Out = sys::Path(OutputFilename);
  }
  else {
    Out = sys::Path("a");
    Out.appendSuffix(JoinNode->OutputSuffix());
  }

  if (JoinNode->GenerateAction(JoinList, Out).Execute() != 0)
    throw std::runtime_error("Tool returned error code!");

  return 0;
}

void llvmcc::Tool::UnpackValues (const std::string& from,
                                 std::vector<std::string>& to) const {
  SplitString(from, to, ",");
}

