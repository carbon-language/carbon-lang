//===--- Tool.h - The LLVM Compiler Driver ----------------------*- C++ -*-===//
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

#include "llvm/CompilerDriver/Action.h"

#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/System/Path.h"

#include <string>
#include <vector>

namespace llvmc {

  class LanguageMap;
  typedef std::vector<llvm::sys::Path> PathVector;
  typedef llvm::StringSet<> InputLanguagesSet;

  /// Tool - A class
  class Tool : public llvm::RefCountedBaseVPTR<Tool> {
  public:

    virtual ~Tool() {}

    virtual Action GenerateAction (const PathVector& inFiles,
                                   const llvm::sys::Path& outFile,
                                   const InputLanguagesSet& InLangs,
                                   const LanguageMap& LangMap) const = 0;

    virtual Action GenerateAction (const llvm::sys::Path& inFile,
                                   const llvm::sys::Path& outFile,
                                   const InputLanguagesSet& InLangs,
                                   const LanguageMap& LangMap) const = 0;

    virtual const char*  Name() const = 0;
    virtual const char** InputLanguages() const = 0;
    virtual const char*  OutputLanguage() const = 0;
    virtual const char*  OutputSuffix() const = 0;

    virtual bool IsLast() const = 0;
    virtual bool IsJoin() const = 0;
  };

  /// JoinTool - A Tool that has an associated input file list.
  class JoinTool : public Tool {
  public:
    void AddToJoinList(const llvm::sys::Path& P) { JoinList_.push_back(P); }
    void ClearJoinList() { JoinList_.clear(); }
    bool JoinListEmpty() const { return JoinList_.empty(); }

    Action GenerateAction(const llvm::sys::Path& outFile,
                          const InputLanguagesSet& InLangs,
                          const LanguageMap& LangMap) const {
      return GenerateAction(JoinList_, outFile, InLangs, LangMap);
    }
    // We shouldn't shadow base class's version of GenerateAction.
    using Tool::GenerateAction;

  private:
    PathVector JoinList_;
  };

}

#endif //LLVM_TOOLS_LLVMC2_TOOL_H
