//===---  BugType.h - Bug Information Desciption ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines BugType, a class representing a bug type.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_BUGTYPE
#define LLVM_CLANG_ANALYSIS_BUGTYPE

#include "clang/GR/BugReporter/BugReporter.h"
#include "llvm/ADT/FoldingSet.h"
#include <string>

namespace clang {

namespace GR {

class ExplodedNode;
class GRExprEngine;

class BugType {
private:
  const std::string Name;
  const std::string Category;
  llvm::FoldingSet<BugReportEquivClass> EQClasses;
  friend class BugReporter;
  bool SuppressonSink;
public:
  BugType(llvm::StringRef name, llvm::StringRef cat)
    : Name(name), Category(cat), SuppressonSink(false) {}
  virtual ~BugType();

  // FIXME: Should these be made strings as well?
  llvm::StringRef getName() const { return Name; }
  llvm::StringRef getCategory() const { return Category; }
  
  /// isSuppressOnSink - Returns true if bug reports associated with this bug
  ///  type should be suppressed if the end node of the report is post-dominated
  ///  by a sink node.
  bool isSuppressOnSink() const { return SuppressonSink; }
  void setSuppressOnSink(bool x) { SuppressonSink = x; }

  virtual void FlushReports(BugReporter& BR);

  typedef llvm::FoldingSet<BugReportEquivClass>::iterator iterator;
  iterator begin() { return EQClasses.begin(); }
  iterator end() { return EQClasses.end(); }

  typedef llvm::FoldingSet<BugReportEquivClass>::const_iterator const_iterator;
  const_iterator begin() const { return EQClasses.begin(); }
  const_iterator end() const { return EQClasses.end(); }
};

class BuiltinBug : public BugType {
  const std::string desc;
public:
  BuiltinBug(const char *name, const char *description)
    : BugType(name, "Logic error"), desc(description) {}
  
  BuiltinBug(const char *name)
    : BugType(name, "Logic error"), desc(name) {}
  
  llvm::StringRef getDescription() const { return desc; }
};

} // end GR namespace

} // end clang namespace
#endif
