//===--- Tools.h - The LLVM Compiler Driver ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open
// Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Tool abstract base class - an interface to tool descriptions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVMC2_TOOL_H
#define LLVM_TOOLS_LLVMC2_TOOL_H

#include "Action.h"

#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/System/Path.h"

#include <string>
#include <vector>

namespace llvmcc {

  typedef std::vector<llvm::sys::Path> PathVector;

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
    static void UnpackValues (std::string const& from,
                              std::vector<std::string>& to);

    virtual ~Tool()
    {}
  };

}

#endif //LLVM_TOOLS_LLVMC2_TOOL_H
