//===--- Core.h - The LLVM Compiler Driver ----------------------*- C++ -*-===//
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

#ifndef LLVM_TOOLS_LLVMCC_CORE_H
#define LLVM_TOOLS_LLVMCC_CORE_H

#include "Utility.h"

#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/System/Path.h"

#include <stdexcept>
#include <string>
#include <vector>

// Core functionality

namespace llvmcc {

  typedef std::vector<llvm::sys::Path> PathVector;
  typedef llvm::StringMap<std::string> LanguageMap;

  class Action {
    std::string Command_;
    std::vector<std::string> Args_;
  public:
    Action (std::string const& C,
            std::vector<std::string> const& A)
      : Command_(C), Args_(A)
    {}

    int Execute();
  };

  class Tool : public llvm::RefCountedBaseVPTR<Tool> {
  public:
    virtual Action GenerateAction (PathVector const& inFiles,
                                  llvm::sys::Path const& outFile) const = 0;

    virtual Action GenerateAction (llvm::sys::Path const& inFile,
                                  llvm::sys::Path const& outFile) const = 0;

    virtual std::string Name() const = 0;
    virtual std::string InputLanguage() const = 0;
    virtual std::string OutputLanguage() const = 0;
    virtual std::string OutputSuffix() const = 0;

    virtual bool IsLast() const = 0;
    virtual bool IsJoin() const = 0;

    // Helper function that is called by the auto-generated code
    // Splits strings of the form ",-foo,-bar,-baz"
    // TOFIX: find a better name
    void UnpackValues (std::string const& from,
                       std::vector<std::string>& to) const;

    virtual ~Tool()
    {}
  };

  typedef std::vector<llvm::IntrusiveRefCntPtr<Tool> > ToolChain;
  typedef llvm::StringMap<ToolChain> ToolChainMap;

  struct CompilationGraph {
    ToolChainMap ToolChains;
    LanguageMap ExtsToLangs;

    int Build(llvm::sys::Path const& tempDir) const;
  };
}

#endif // LLVM_TOOLS_LLVMCC_CORE_H
