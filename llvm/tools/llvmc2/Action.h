//===--- Tools.h - The LLVM Compiler Driver ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open
// Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Action - encapsulates a single shell command.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVMC2_ACTION_H
#define LLVM_TOOLS_LLVMC2_ACTION_H

#include <string>
#include <vector>

namespace llvmcc {

  class Action {
    std::string Command_;
    std::vector<std::string> Args_;
  public:
    Action (std::string const& C,
            std::vector<std::string> const& A)
      : Command_(C), Args_(A)
    {}

    int Execute() const;
  };

}

#endif // LLVM_TOOLS_LLVMC2_ACTION_H
