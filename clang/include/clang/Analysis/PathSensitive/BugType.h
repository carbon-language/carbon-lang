//===---  BugType.h - Bug Information Desciption ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines BugType, a class representing a bug tpye.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_BUGTYPE
#define LLVM_CLANG_ANALYSIS_BUGTYPE

#include <llvm/ADT/FoldingSet.h>
#include <string>

namespace clang {

class BugReportEquivClass;
class BugReporter;
class BuiltinBugReport;
class BugReporterContext;
class GRExprEngine;

class BugType {
private:
  const std::string Name;
  const std::string Category;
  llvm::FoldingSet<BugReportEquivClass> EQClasses;
  friend class BugReporter;
  bool SuppressonSink;
public:
  BugType(const char *name, const char* cat)
    : Name(name), Category(cat), SuppressonSink(false) {}
  virtual ~BugType();

  // FIXME: Should these be made strings as well?
  const std::string& getName() const { return Name; }
  const std::string& getCategory() const { return Category; }
  
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
  GRExprEngine &Eng;
protected:
  const std::string desc;
public:
  BuiltinBug(GRExprEngine *eng, const char* n, const char* d)
    : BugType(n, "Logic errors"), Eng(*eng), desc(d) {}

  BuiltinBug(GRExprEngine *eng, const char* n)
    : BugType(n, "Logic errors"), Eng(*eng), desc(n) {}

  const std::string &getDescription() const { return desc; }

  virtual void FlushReportsImpl(BugReporter& BR, GRExprEngine& Eng) {}

  void FlushReports(BugReporter& BR) { FlushReportsImpl(BR, Eng); }

  virtual void registerInitialVisitors(BugReporterContext& BRC,
                                       const ExplodedNode* N,
                                       BuiltinBugReport *R) {}

  template <typename ITER> void Emit(BugReporter& BR, ITER I, ITER E);
};
} // end clang namespace
#endif
