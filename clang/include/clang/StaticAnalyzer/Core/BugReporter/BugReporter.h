//===---  BugReporter.h - Generate PathDiagnostics --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines BugReporter, a utility class for generating
//  PathDiagnostics for analyses based on ProgramState.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_GR_BUGREPORTER
#define LLVM_CLANG_GR_BUGREPORTER

#include "clang/Basic/SourceLocation.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporterVisitor.h"
#include "clang/StaticAnalyzer/Core/BugReporter/PathDiagnostic.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramState.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/ImmutableList.h"
#include "llvm/ADT/ImmutableSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/DenseSet.h"
#include <list>

namespace clang {

class ASTContext;
class DiagnosticsEngine;
class Stmt;
class ParentMap;

namespace ento {

class PathDiagnostic;
class ExplodedNode;
class ExplodedGraph;
class BugReport;
class BugReporter;
class BugReporterContext;
class ExprEngine;
class BugType;

//===----------------------------------------------------------------------===//
// Interface for individual bug reports.
//===----------------------------------------------------------------------===//

/// This class provides an interface through which checkers can create
/// individual bug reports.
class BugReport {
public:  
  class NodeResolver {
    virtual void anchor();
  public:
    virtual ~NodeResolver() {}
    virtual const ExplodedNode*
            getOriginalNode(const ExplodedNode *N) = 0;
  };

  typedef const SourceRange *ranges_iterator;
  typedef SmallVector<BugReporterVisitor *, 8> VisitorList;
  typedef VisitorList::iterator visitor_iterator;
  typedef SmallVector<StringRef, 2> ExtraTextList;

protected:
  friend class BugReporter;
  friend class BugReportEquivClass;

  BugType& BT;
  std::string ShortDescription;
  std::string Description;
  PathDiagnosticLocation Location;
  PathDiagnosticLocation UniqueingLocation;
  const ExplodedNode *ErrorNode;
  SmallVector<SourceRange, 4> Ranges;
  ExtraTextList ExtraText;
  
  typedef llvm::DenseSet<SymbolRef> Symbols;
  typedef llvm::DenseSet<const MemRegion *> Regions;

  /// A set of symbols that are registered with this report as being
  /// "interesting", and thus used to help decide which diagnostics
  /// to include when constructing the final path diagnostic.
  Symbols interestingSymbols;

  /// A set of regions that are registered with this report as being
  /// "interesting", and thus used to help decide which diagnostics
  /// to include when constructing the final path diagnostic.
  Regions interestingRegions;

  /// A set of custom visitors which generate "event" diagnostics at
  /// interesting points in the path.
  VisitorList Callbacks;

  /// Used for ensuring the visitors are only added once.
  llvm::FoldingSet<BugReporterVisitor> CallbacksSet;

  /// Used for clients to tell if the report's configuration has changed
  /// since the last time they checked.
  unsigned ConfigurationChangeToken;

public:
  BugReport(BugType& bt, StringRef desc, const ExplodedNode *errornode)
    : BT(bt), Description(desc), ErrorNode(errornode),
      ConfigurationChangeToken(0) {}

  BugReport(BugType& bt, StringRef shortDesc, StringRef desc,
            const ExplodedNode *errornode)
    : BT(bt), ShortDescription(shortDesc), Description(desc),
      ErrorNode(errornode), ConfigurationChangeToken(0) {}

  BugReport(BugType& bt, StringRef desc, PathDiagnosticLocation l)
    : BT(bt), Description(desc), Location(l), ErrorNode(0),
      ConfigurationChangeToken(0) {}

  /// \brief Create a BugReport with a custom uniqueing location.
  ///
  /// The reports that have the same report location, description, bug type, and
  /// ranges are uniqued - only one of the equivalent reports will be presented
  /// to the user. This method allows to rest the location which should be used
  /// for uniquing reports. For example, memory leaks checker, could set this to
  /// the allocation site, rather then the location where the bug is reported.
  BugReport(BugType& bt, StringRef desc, const ExplodedNode *errornode,
            PathDiagnosticLocation LocationToUnique)
    : BT(bt), Description(desc), UniqueingLocation(LocationToUnique),
      ErrorNode(errornode), ConfigurationChangeToken(0) {}

  virtual ~BugReport();

  const BugType& getBugType() const { return BT; }
  BugType& getBugType() { return BT; }

  const ExplodedNode *getErrorNode() const { return ErrorNode; }

  const StringRef getDescription() const { return Description; }

  const StringRef getShortDescription() const {
    return ShortDescription.empty() ? Description : ShortDescription;
  }

  void markInteresting(SymbolRef sym);
  void markInteresting(const MemRegion *R);
  void markInteresting(SVal V);
  
  bool isInteresting(SymbolRef sym) const;
  bool isInteresting(const MemRegion *R) const;
  bool isInteresting(SVal V) const;

  unsigned getConfigurationChangeToken() const {
    return ConfigurationChangeToken;
  }
  
  /// \brief This allows for addition of meta data to the diagnostic.
  ///
  /// Currently, only the HTMLDiagnosticClient knows how to display it. 
  void addExtraText(StringRef S) {
    ExtraText.push_back(S);
  }

  virtual const ExtraTextList &getExtraText() {
    return ExtraText;
  }

  /// \brief Return the "definitive" location of the reported bug.
  ///
  ///  While a bug can span an entire path, usually there is a specific
  ///  location that can be used to identify where the key issue occurred.
  ///  This location is used by clients rendering diagnostics.
  virtual PathDiagnosticLocation getLocation(const SourceManager &SM) const;

  const Stmt *getStmt() const;

  /// \brief Add a range to a bug report.
  ///
  /// Ranges are used to highlight regions of interest in the source code.
  /// They should be at the same source code line as the BugReport location.
  /// By default, the source range of the statement corresponding to the error
  /// node will be used; add a single invalid range to specify absence of
  /// ranges.
  void addRange(SourceRange R) {
    assert((R.isValid() || Ranges.empty()) && "Invalid range can only be used "
                           "to specify that the report does not have a range.");
    Ranges.push_back(R);
  }

  /// \brief Get the SourceRanges associated with the report.
  virtual std::pair<ranges_iterator, ranges_iterator> getRanges();

  /// \brief Add custom or predefined bug report visitors to this report.
  ///
  /// The visitors should be used when the default trace is not sufficient.
  /// For example, they allow constructing a more elaborate trace.
  /// \sa registerConditionVisitor(), registerTrackNullOrUndefValue(),
  /// registerFindLastStore(), registerNilReceiverVisitor(), and
  /// registerVarDeclsLastStore().
  void addVisitor(BugReporterVisitor *visitor);

	/// Iterators through the custom diagnostic visitors.
  visitor_iterator visitor_begin() { return Callbacks.begin(); }
  visitor_iterator visitor_end() { return Callbacks.end(); }

  /// Profile to identify equivalent bug reports for error report coalescing.
  /// Reports are uniqued to ensure that we do not emit multiple diagnostics
  /// for each bug.
  virtual void Profile(llvm::FoldingSetNodeID& hash) const;
};

//===----------------------------------------------------------------------===//
// BugTypes (collections of related reports).
//===----------------------------------------------------------------------===//

class BugReportEquivClass : public llvm::FoldingSetNode {
  /// List of *owned* BugReport objects.
  std::list<BugReport*> Reports;

  friend class BugReporter;
  void AddReport(BugReport* R) { Reports.push_back(R); }
public:
  BugReportEquivClass(BugReport* R) { Reports.push_back(R); }
  ~BugReportEquivClass();

  void Profile(llvm::FoldingSetNodeID& ID) const {
    assert(!Reports.empty());
    (*Reports.begin())->Profile(ID);
  }

  class iterator {
    std::list<BugReport*>::iterator impl;
  public:
    iterator(std::list<BugReport*>::iterator i) : impl(i) {}
    iterator &operator++() { ++impl; return *this; }
    bool operator==(const iterator &I) const { return I.impl == impl; }
    bool operator!=(const iterator &I) const { return I.impl != impl; }
    BugReport* operator*() const { return *impl; }
    BugReport* operator->() const { return *impl; }
  };

  class const_iterator {
    std::list<BugReport*>::const_iterator impl;
  public:
    const_iterator(std::list<BugReport*>::const_iterator i) : impl(i) {}
    const_iterator &operator++() { ++impl; return *this; }
    bool operator==(const const_iterator &I) const { return I.impl == impl; }
    bool operator!=(const const_iterator &I) const { return I.impl != impl; }
    const BugReport* operator*() const { return *impl; }
    const BugReport* operator->() const { return *impl; }
  };

  iterator begin() { return iterator(Reports.begin()); }
  iterator end() { return iterator(Reports.end()); }

  const_iterator begin() const { return const_iterator(Reports.begin()); }
  const_iterator end() const { return const_iterator(Reports.end()); }
};

//===----------------------------------------------------------------------===//
// BugReporter and friends.
//===----------------------------------------------------------------------===//

class BugReporterData {
public:
  virtual ~BugReporterData();
  virtual DiagnosticsEngine& getDiagnostic() = 0;
  virtual PathDiagnosticConsumer* getPathDiagnosticConsumer() = 0;
  virtual ASTContext &getASTContext() = 0;
  virtual SourceManager& getSourceManager() = 0;
};

/// BugReporter is a utility class for generating PathDiagnostics for analysis.
/// It collects the BugReports and BugTypes and knows how to generate
/// and flush the corresponding diagnostics.
class BugReporter {
public:
  enum Kind { BaseBRKind, GRBugReporterKind };

private:
  typedef llvm::ImmutableSet<BugType*> BugTypesTy;
  BugTypesTy::Factory F;
  BugTypesTy BugTypes;

  const Kind kind;
  BugReporterData& D;

  /// Generate and flush the diagnostics for the given bug report.
  void FlushReport(BugReportEquivClass& EQ);

  /// The set of bug reports tracked by the BugReporter.
  llvm::FoldingSet<BugReportEquivClass> EQClasses;
  /// A vector of BugReports for tracking the allocated pointers and cleanup.
  std::vector<BugReportEquivClass *> EQClassesVector;

protected:
  BugReporter(BugReporterData& d, Kind k) : BugTypes(F.getEmptySet()), kind(k),
                                            D(d) {}

public:
  BugReporter(BugReporterData& d) : BugTypes(F.getEmptySet()), kind(BaseBRKind),
                                    D(d) {}
  virtual ~BugReporter();

  /// \brief Generate and flush diagnostics for all bug reports.
  void FlushReports();

  Kind getKind() const { return kind; }

  DiagnosticsEngine& getDiagnostic() {
    return D.getDiagnostic();
  }

  PathDiagnosticConsumer* getPathDiagnosticConsumer() {
    return D.getPathDiagnosticConsumer();
  }

  /// \brief Iterator over the set of BugTypes tracked by the BugReporter.
  typedef BugTypesTy::iterator iterator;
  iterator begin() { return BugTypes.begin(); }
  iterator end() { return BugTypes.end(); }

  /// \brief Iterator over the set of BugReports tracked by the BugReporter.
  typedef llvm::FoldingSet<BugReportEquivClass>::iterator EQClasses_iterator;
  EQClasses_iterator EQClasses_begin() { return EQClasses.begin(); }
  EQClasses_iterator EQClasses_end() { return EQClasses.end(); }

  ASTContext &getContext() { return D.getASTContext(); }

  SourceManager& getSourceManager() { return D.getSourceManager(); }

  virtual void GeneratePathDiagnostic(PathDiagnostic& pathDiagnostic,
        SmallVectorImpl<BugReport *> &bugReports) {}

  void Register(BugType *BT);

  /// \brief Add the given report to the set of reports tracked by BugReporter.
  ///
  /// The reports are usually generated by the checkers. Further, they are
  /// folded based on the profile value, which is done to coalesce similar
  /// reports.
  void EmitReport(BugReport *R);

  void EmitBasicReport(StringRef BugName, StringRef BugStr,
                       PathDiagnosticLocation Loc,
                       SourceRange* RangeBeg, unsigned NumRanges);

  void EmitBasicReport(StringRef BugName, StringRef BugCategory,
                       StringRef BugStr, PathDiagnosticLocation Loc,
                       SourceRange* RangeBeg, unsigned NumRanges);


  void EmitBasicReport(StringRef BugName, StringRef BugStr,
                       PathDiagnosticLocation Loc) {
    EmitBasicReport(BugName, BugStr, Loc, 0, 0);
  }

  void EmitBasicReport(StringRef BugName, StringRef BugCategory,
                       StringRef BugStr, PathDiagnosticLocation Loc) {
    EmitBasicReport(BugName, BugCategory, BugStr, Loc, 0, 0);
  }

  void EmitBasicReport(StringRef BugName, StringRef BugStr,
                       PathDiagnosticLocation Loc, SourceRange R) {
    EmitBasicReport(BugName, BugStr, Loc, &R, 1);
  }

  void EmitBasicReport(StringRef BugName, StringRef Category,
                       StringRef BugStr, PathDiagnosticLocation Loc,
                       SourceRange R) {
    EmitBasicReport(BugName, Category, BugStr, Loc, &R, 1);
  }

  static bool classof(const BugReporter* R) { return true; }

private:
  llvm::StringMap<BugType *> StrBugTypes;

  /// \brief Returns a BugType that is associated with the given name and
  /// category.
  BugType *getBugTypeForName(StringRef name, StringRef category);
};

// FIXME: Get rid of GRBugReporter.  It's the wrong abstraction.
class GRBugReporter : public BugReporter {
  ExprEngine& Eng;
public:
  GRBugReporter(BugReporterData& d, ExprEngine& eng)
    : BugReporter(d, GRBugReporterKind), Eng(eng) {}

  virtual ~GRBugReporter();

  /// getEngine - Return the analysis engine used to analyze a given
  ///  function or method.
  ExprEngine &getEngine() { return Eng; }

  /// getGraph - Get the exploded graph created by the analysis engine
  ///  for the analyzed method or function.
  ExplodedGraph &getGraph();

  /// getStateManager - Return the state manager used by the analysis
  ///  engine.
  ProgramStateManager &getStateManager();

  virtual void GeneratePathDiagnostic(PathDiagnostic &pathDiagnostic,
                     SmallVectorImpl<BugReport*> &bugReports);

  /// classof - Used by isa<>, cast<>, and dyn_cast<>.
  static bool classof(const BugReporter* R) {
    return R->getKind() == GRBugReporterKind;
  }
};

class BugReporterContext {
  virtual void anchor();
  GRBugReporter &BR;
public:
  BugReporterContext(GRBugReporter& br) : BR(br) {}

  virtual ~BugReporterContext() {}

  GRBugReporter& getBugReporter() { return BR; }

  ExplodedGraph &getGraph() { return BR.getGraph(); }

  ProgramStateManager& getStateManager() {
    return BR.getStateManager();
  }

  SValBuilder& getSValBuilder() {
    return getStateManager().getSValBuilder();
  }

  ASTContext &getASTContext() {
    return BR.getContext();
  }

  SourceManager& getSourceManager() {
    return BR.getSourceManager();
  }

  virtual BugReport::NodeResolver& getNodeResolver() = 0;
};

} // end GR namespace

} // end clang namespace

#endif
