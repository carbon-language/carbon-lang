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
//  PathDiagnostics for analyses based on GRState.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_BUGREPORTER
#define LLVM_CLANG_ANALYSIS_BUGREPORTER

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Analysis/PathSensitive/GRState.h"
#include "clang/Analysis/PathSensitive/ExplodedGraph.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/ImmutableSet.h"
#include <list>

namespace clang {
  
class PathDiagnostic;
class PathDiagnosticPiece;
class PathDiagnosticClient;
class ASTContext;
class Diagnostic;
class BugReporter;
class BugReporterContext;
class GRExprEngine;
class GRState;
class Stmt;
class BugType;
class ParentMap;
  
//===----------------------------------------------------------------------===//
// Interface for individual bug reports.
//===----------------------------------------------------------------------===//

class BugReporterVisitor {
public:
  virtual ~BugReporterVisitor();
  virtual PathDiagnosticPiece* VisitNode(const ExplodedNode* N,
                                         const ExplodedNode* PrevN,
                                         BugReporterContext& BRC) = 0;
  
  virtual bool isOwnedByReporterContext() { return true; }
};
  
// FIXME: Combine this with RangedBugReport and remove RangedBugReport.
class BugReport : public BugReporterVisitor {
protected:
  BugType& BT;
  std::string ShortDescription;
  std::string Description;
  const ExplodedNode *EndNode;
  SourceRange R;
  
protected:
  friend class BugReporter;
  friend class BugReportEquivClass;

  virtual void Profile(llvm::FoldingSetNodeID& hash) const {
    hash.AddInteger(getLocation().getRawEncoding());
  }
  
public:
  class NodeResolver {
  public:
    virtual ~NodeResolver() {}
    virtual const ExplodedNode*
            getOriginalNode(const ExplodedNode* N) = 0;
  };
  
  BugReport(BugType& bt, const char* desc, const ExplodedNode *n)
    : BT(bt), Description(desc), EndNode(n) {}
  
  BugReport(BugType& bt, const char* shortDesc, const char* desc,
            const ExplodedNode *n)
  : BT(bt), ShortDescription(shortDesc), Description(desc), EndNode(n) {}

  virtual ~BugReport();
  
  virtual bool isOwnedByReporterContext() { return false; }

  const BugType& getBugType() const { return BT; }
  BugType& getBugType() { return BT; }
  
  // FIXME: Perhaps this should be moved into a subclass?
  const ExplodedNode* getEndNode() const { return EndNode; }
  
  // FIXME: Do we need this?  Maybe getLocation() should return a ProgramPoint
  // object.
  // FIXME: If we do need it, we can probably just make it private to
  // BugReporter.
  const Stmt* getStmt(BugReporter& BR) const;
  
  const std::string& getDescription() const { return Description; }

  const std::string& getShortDescription() const {
    return ShortDescription.empty() ? Description : ShortDescription;
  }
  
  // FIXME: Is this needed?
  virtual std::pair<const char**,const char**> getExtraDescriptiveText() {
    return std::make_pair((const char**)0,(const char**)0);
  }
  
  // FIXME: Perhaps move this into a subclass.
  virtual PathDiagnosticPiece* getEndPath(BugReporterContext& BRC,
                                          const ExplodedNode* N);
  
  /// getLocation - Return the "definitive" location of the reported bug.
  ///  While a bug can span an entire path, usually there is a specific
  ///  location that can be used to identify where the key issue occured.
  ///  This location is used by clients rendering diagnostics.
  virtual SourceLocation getLocation() const;
  
  /// getRanges - Returns the source ranges associated with this bug.
  virtual void getRanges(BugReporter& BR,const SourceRange*& beg,
                         const SourceRange*& end);

  virtual PathDiagnosticPiece* VisitNode(const ExplodedNode* N,
                                         const ExplodedNode* PrevN,
                                         BugReporterContext& BR);
  
  virtual void registerInitialVisitors(BugReporterContext& BRC,
                                       const ExplodedNode* N) {}
};

//===----------------------------------------------------------------------===//
// BugTypes (collections of related reports).
//===----------------------------------------------------------------------===//
  
class BugReportEquivClass : public llvm::FoldingSetNode {
  // List of *owned* BugReport objects.
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
    iterator& operator++() { ++impl; return *this; }
    bool operator==(const iterator& I) const { return I.impl == impl; }
    bool operator!=(const iterator& I) const { return I.impl != impl; }
    BugReport* operator*() const { return *impl; }
    BugReport* operator->() const { return *impl; }
  };
  
  class const_iterator {
    std::list<BugReport*>::const_iterator impl;
  public:
    const_iterator(std::list<BugReport*>::const_iterator i) : impl(i) {}
    const_iterator& operator++() { ++impl; return *this; }
    bool operator==(const const_iterator& I) const { return I.impl == impl; }
    bool operator!=(const const_iterator& I) const { return I.impl != impl; }
    const BugReport* operator*() const { return *impl; }
    const BugReport* operator->() const { return *impl; }
  };
    
  iterator begin() { return iterator(Reports.begin()); }
  iterator end() { return iterator(Reports.end()); }
  
  const_iterator begin() const { return const_iterator(Reports.begin()); }
  const_iterator end() const { return const_iterator(Reports.end()); }
};
  
class BugType {
private:
  const std::string Name;
  const std::string Category;
  llvm::FoldingSet<BugReportEquivClass> EQClasses;
  friend class BugReporter;
public:
  BugType(const char *name, const char* cat) : Name(name), Category(cat) {}
  virtual ~BugType();
  
  // FIXME: Should these be made strings as well?
  const std::string& getName() const { return Name; }
  const std::string& getCategory() const { return Category; }

  virtual void FlushReports(BugReporter& BR);  

  typedef llvm::FoldingSet<BugReportEquivClass>::iterator iterator;
  iterator begin() { return EQClasses.begin(); }
  iterator end() { return EQClasses.end(); }
  
  typedef llvm::FoldingSet<BugReportEquivClass>::const_iterator const_iterator;
  const_iterator begin() const { return EQClasses.begin(); }
  const_iterator end() const { return EQClasses.end(); }
};
  
//===----------------------------------------------------------------------===//
// Specialized subclasses of BugReport.
//===----------------------------------------------------------------------===//
  
// FIXME: Collapse this with the default BugReport class.
class RangedBugReport : public BugReport {
  std::vector<SourceRange> Ranges;
public:
  RangedBugReport(BugType& D, const char* description, ExplodedNode *n)
    : BugReport(D, description, n) {}
  
  RangedBugReport(BugType& D, const char *shortDescription,
                  const char *description, ExplodedNode *n)
  : BugReport(D, shortDescription, description, n) {}
  
  ~RangedBugReport();

  // FIXME: Move this out of line.
  void addRange(SourceRange R) { Ranges.push_back(R); }
  
  // FIXME: Move this out of line.
  void getRanges(BugReporter& BR,const SourceRange*& beg,
                 const SourceRange*& end) {
    
    if (Ranges.empty()) {
      beg = NULL;
      end = NULL;
    }
    else {
      beg = &Ranges[0];
      end = beg + Ranges.size();
    }
  }
};
  
//===----------------------------------------------------------------------===//
// BugReporter and friends.
//===----------------------------------------------------------------------===//

class BugReporterData {
public:
  virtual ~BugReporterData();
  virtual Diagnostic& getDiagnostic() = 0;  
  virtual PathDiagnosticClient* getPathDiagnosticClient() = 0;  
  virtual ASTContext& getContext() = 0;
  virtual SourceManager& getSourceManager() = 0;
  virtual CFG* getCFG() = 0;
  virtual ParentMap& getParentMap() = 0;
  virtual LiveVariables* getLiveVariables() = 0;
};
  
class BugReporter {
public:
  enum Kind { BaseBRKind, GRBugReporterKind };

private:
  typedef llvm::ImmutableSet<BugType*> BugTypesTy;
  BugTypesTy::Factory F;
  BugTypesTy BugTypes;

  const Kind kind;  
  BugReporterData& D;
  
  void FlushReport(BugReportEquivClass& EQ);

protected:
  BugReporter(BugReporterData& d, Kind k) : BugTypes(F.GetEmptySet()), kind(k), D(d) {}

public:
  BugReporter(BugReporterData& d) : BugTypes(F.GetEmptySet()), kind(BaseBRKind), D(d) {}
  virtual ~BugReporter();
  
  void FlushReports();
  
  Kind getKind() const { return kind; }
  
  Diagnostic& getDiagnostic() {
    return D.getDiagnostic();
  }
  
  PathDiagnosticClient* getPathDiagnosticClient() {
    return D.getPathDiagnosticClient();
  }
  
  typedef BugTypesTy::iterator iterator;
  iterator begin() { return BugTypes.begin(); }
  iterator end() { return BugTypes.end(); }
  
  ASTContext& getContext() { return D.getContext(); }
  
  SourceManager& getSourceManager() { return D.getSourceManager(); }
  
  CFG* getCFG() { return D.getCFG(); }
  
  ParentMap& getParentMap() { return D.getParentMap(); }
  
  LiveVariables* getLiveVariables() { return D.getLiveVariables(); }
  
  virtual void GeneratePathDiagnostic(PathDiagnostic& PD,
                                      BugReportEquivClass& EQ) {}

  void Register(BugType *BT);
  
  void EmitReport(BugReport *R);
  
  void EmitBasicReport(const char* BugName, const char* BugStr,
                       SourceLocation Loc,
                       SourceRange* RangeBeg, unsigned NumRanges);

  void EmitBasicReport(const char* BugName, const char* BugCategory,
                       const char* BugStr, SourceLocation Loc,
                       SourceRange* RangeBeg, unsigned NumRanges);
  
  
  void EmitBasicReport(const char* BugName, const char* BugStr,
                       SourceLocation Loc) {
    EmitBasicReport(BugName, BugStr, Loc, 0, 0);
  }
  
  void EmitBasicReport(const char* BugName, const char* BugCategory,
                       const char* BugStr, SourceLocation Loc) {
    EmitBasicReport(BugName, BugCategory, BugStr, Loc, 0, 0);
  }
  
  void EmitBasicReport(const char* BugName, const char* BugStr,
                       SourceLocation Loc, SourceRange R) {
    EmitBasicReport(BugName, BugStr, Loc, &R, 1);
  }
  
  void EmitBasicReport(const char* BugName, const char* Category,
                       const char* BugStr, SourceLocation Loc, SourceRange R) {
    EmitBasicReport(BugName, Category, BugStr, Loc, &R, 1);
  }
  
  static bool classof(const BugReporter* R) { return true; }
};

// FIXME: Get rid of GRBugReporter.  It's the wrong abstraction.
class GRBugReporter : public BugReporter {
  GRExprEngine& Eng;
  llvm::SmallSet<SymbolRef, 10> NotableSymbols;
public:  
  GRBugReporter(BugReporterData& d, GRExprEngine& eng)
    : BugReporter(d, GRBugReporterKind), Eng(eng) {}
  
  virtual ~GRBugReporter();
  
  /// getEngine - Return the analysis engine used to analyze a given
  ///  function or method.
  GRExprEngine& getEngine() { return Eng; }

  /// getGraph - Get the exploded graph created by the analysis engine
  ///  for the analyzed method or function.
  ExplodedGraph<GRState>& getGraph();
  
  /// getStateManager - Return the state manager used by the analysis
  ///  engine.
  GRStateManager& getStateManager();
  
  virtual void GeneratePathDiagnostic(PathDiagnostic& PD,
                                      BugReportEquivClass& R);

  void addNotableSymbol(SymbolRef Sym) {
    NotableSymbols.insert(Sym);
  }
  
  bool isNotable(SymbolRef Sym) const {
    return (bool) NotableSymbols.count(Sym);
  }
    
  /// classof - Used by isa<>, cast<>, and dyn_cast<>.
  static bool classof(const BugReporter* R) {
    return R->getKind() == GRBugReporterKind;
  }
};
  
class BugReporterContext {
  GRBugReporter &BR;
  std::vector<BugReporterVisitor*> Callbacks;
public:
  BugReporterContext(GRBugReporter& br) : BR(br) {}
  virtual ~BugReporterContext();
  
  void addVisitor(BugReporterVisitor* visitor) {
    if (visitor) Callbacks.push_back(visitor);
  }
  
  typedef std::vector<BugReporterVisitor*>::iterator visitor_iterator;
  visitor_iterator visitor_begin() { return Callbacks.begin(); }
  visitor_iterator visitor_end() { return Callbacks.end(); }  
  
  GRBugReporter& getBugReporter() { return BR; }  
  
  ExplodedGraph<GRState>& getGraph() { return BR.getGraph(); }
  
  void addNotableSymbol(SymbolRef Sym) {
    // FIXME: For now forward to GRBugReporter.
    BR.addNotableSymbol(Sym);
  }
  
  bool isNotable(SymbolRef Sym) const {
    // FIXME: For now forward to GRBugReporter.
    return BR.isNotable(Sym);
  }
  
  GRStateManager& getStateManager() {
    return BR.getStateManager();
  }
  
  ValueManager& getValueManager() {
    return getStateManager().getValueManager();
  }
  
  ASTContext& getASTContext() {
    return BR.getContext();
  }
  
  SourceManager& getSourceManager() {
    return BR.getSourceManager();
  }
  
  const Decl& getCodeDecl() {
    return getStateManager().getCodeDecl();
  }
  
  const CFG& getCFG() {
    return *BR.getCFG();
  }
  
  virtual BugReport::NodeResolver& getNodeResolver() = 0;  
};

class DiagBugReport : public RangedBugReport {
  std::list<std::string> Strs;
  FullSourceLoc L;
public:
  DiagBugReport(BugType& D, const char* desc, FullSourceLoc l) :
  RangedBugReport(D, desc, 0), L(l) {}
  
  virtual ~DiagBugReport() {}
  
  // FIXME: Move out-of-line (virtual function).
  SourceLocation getLocation() const { return L; }
  
  void addString(const std::string& s) { Strs.push_back(s); }  
  
  typedef std::list<std::string>::const_iterator str_iterator;
  str_iterator str_begin() const { return Strs.begin(); }
  str_iterator str_end() const { return Strs.end(); }
};
  
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
  
namespace bugreporter {
  
const Stmt *GetDerefExpr(const ExplodedNode *N);
const Stmt *GetReceiverExpr(const ExplodedNode *N);
const Stmt *GetDenomExpr(const ExplodedNode *N);
const Stmt *GetCalleeExpr(const ExplodedNode *N);
const Stmt *GetRetValExpr(const ExplodedNode *N);

void registerTrackNullOrUndefValue(BugReporterContext& BRC, const Stmt *S,
                                   const ExplodedNode* N);

} // end namespace clang::bugreporter
  
//===----------------------------------------------------------------------===//

} // end clang namespace

#endif
