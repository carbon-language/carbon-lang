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

#ifndef LLVM_INCLUDE_COMPILER_DRIVER_TOOL_H
#define LLVM_INCLUDE_COMPILER_DRIVER_TOOL_H

#include "llvm/CompilerDriver/Action.h"

#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/System/Path.h"

#include <string>
#include <vector>
#include <utility>

namespace llvmc {

  class LanguageMap;
  typedef std::vector<std::pair<unsigned, std::string> > ArgsVector;
  typedef std::vector<llvm::sys::Path> PathVector;
  typedef std::vector<std::string> StrVector;
  typedef llvm::StringSet<> InputLanguagesSet;

  /// Tool - Represents a single tool.
  class Tool : public llvm::RefCountedBaseVPTR<Tool> {
  public:

    virtual ~Tool() {}

    /// GenerateAction - Generate an Action given particular command-line
    /// options. Returns non-zero value on error.
    virtual int GenerateAction (Action& Out,
                                const PathVector& inFiles,
                                const bool HasChildren,
                                const llvm::sys::Path& TempDir,
                                const InputLanguagesSet& InLangs,
                                const LanguageMap& LangMap) const = 0;

    /// GenerateAction - Generate an Action given particular command-line
    /// options. Returns non-zero value on error.
    virtual int GenerateAction (Action& Out,
                                const llvm::sys::Path& inFile,
                                const bool HasChildren,
                                const llvm::sys::Path& TempDir,
                                const InputLanguagesSet& InLangs,
                                const LanguageMap& LangMap) const = 0;

    virtual const char*  Name() const = 0;
    virtual const char** InputLanguages() const = 0;
    virtual const char** OutputLanguages() const = 0;

    virtual bool IsJoin() const = 0;
    virtual bool WorksOnEmpty() const = 0;

  protected:
    /// OutFileName - Generate the output file name.
    llvm::sys::Path OutFilename(const llvm::sys::Path& In,
                                const llvm::sys::Path& TempDir,
                                bool StopCompilation,
                                const char* OutputSuffix) const;

    StrVector SortArgs(ArgsVector& Args) const;
  };

  /// JoinTool - A Tool that has an associated input file list.
  class JoinTool : public Tool {
  public:
    void AddToJoinList(const llvm::sys::Path& P) { JoinList_.push_back(P); }
    void ClearJoinList() { JoinList_.clear(); }
    bool JoinListEmpty() const { return JoinList_.empty(); }

    int GenerateAction(Action& Out,
                       const bool HasChildren,
                       const llvm::sys::Path& TempDir,
                       const InputLanguagesSet& InLangs,
                       const LanguageMap& LangMap) const {
      return GenerateAction(Out, JoinList_, HasChildren, TempDir, InLangs,
                            LangMap);
    }
    // We shouldn't shadow base class's version of GenerateAction.
    using Tool::GenerateAction;

  private:
    PathVector JoinList_;
  };

}

#endif // LLVM_INCLUDE_COMPILER_DRIVER_TOOL_H
