// BugReporter.h - Generate PathDiagnostics  ----------*- C++ -*--//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines BugReporter, a utility class for generating
//  PathDiagnostics for analyses based on ValueState.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_BUGREPORTER
#define LLVM_CLANG_ANALYSIS_BUGREPORTER

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Analysis/PathSensitive/ValueState.h"
#include "clang/Analysis/PathSensitive/ExplodedGraph.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include <vector>
#include <list>


namespace clang {
  
class PathDiagnostic;
class PathDiagnosticPiece;
class PathDiagnosticClient;
class ASTContext;
class Diagnostic;
class BugReporter;
class GRExprEngine;
class ValueState;
class Stmt;
class BugReport;
class ParentMap;
  
class BugType {
public:
  BugType() {}
  virtual ~BugType();
  
  virtual const char* getName() const = 0;
  virtual const char* getDescription() const { return getName(); }
  
  virtual std::pair<const char**,const char**> getExtraDescriptiveText() {
    return std::pair<const char**, const char**>(0, 0);
  }
      
  virtual void EmitWarnings(BugReporter& BR) {}
  virtual void GetErrorNodes(std::vector<ExplodedNode<ValueState>*>& Nodes) {}
  
  virtual bool isCached(BugReport& R) = 0;
};
  
class BugTypeCacheLocation : public BugType {
  llvm::SmallPtrSet<void*,10> CachedErrors;
public:
  BugTypeCacheLocation() {}
  virtual ~BugTypeCacheLocation() {}  
  virtual bool isCached(BugReport& R);
  bool isCached(ProgramPoint P);
};
  
  
class BugReport {
  BugType& Desc;
  ExplodedNode<ValueState> *EndNode;
  SourceRange R;  
public:
  BugReport(BugType& D, ExplodedNode<ValueState> *n) : Desc(D), EndNode(n) {}
  virtual ~BugReport();
  
  const BugType& getBugType() const { return Desc; }
  BugType& getBugType() { return Desc; }
  
  ExplodedNode<ValueState>* getEndNode() const { return EndNode; }
  
  Stmt* getStmt(BugReporter& BR) const;
    
  const char* getName() const { return getBugType().getName(); }

  virtual const char* getDescription() const {
    return getBugType().getDescription();
  }
  
  virtual std::pair<const char**,const char**> getExtraDescriptiveText() {
    return getBugType().getExtraDescriptiveText();
  }
  
  virtual PathDiagnosticPiece* getEndPath(BugReporter& BR,
                                          ExplodedNode<ValueState>* N);
  
  virtual FullSourceLoc getLocation(SourceManager& Mgr);
  
  virtual void getRanges(BugReporter& BR,const SourceRange*& beg,
                         const SourceRange*& end);
  
  virtual PathDiagnosticPiece* VisitNode(ExplodedNode<ValueState>* N,
                                         ExplodedNode<ValueState>* PrevN,
                                         ExplodedGraph<ValueState>& G,
                                         BugReporter& BR);
};
  
class RangedBugReport : public BugReport {
  std::vector<SourceRange> Ranges;
public:
  RangedBugReport(BugType& D, ExplodedNode<ValueState> *n)
    : BugReport(D, n) {}
  
  virtual ~RangedBugReport();
  
  void addRange(SourceRange R) { Ranges.push_back(R); }
  
  virtual void getRanges(BugReporter& BR,const SourceRange*& beg,           
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
  
class BugReporterData {
public:
  virtual ~BugReporterData();
  virtual Diagnostic& getDiagnostic() = 0;  
  virtual PathDiagnosticClient* getPathDiagnosticClient() = 0;  
  virtual ASTContext& getContext() = 0;
  virtual SourceManager& getSourceManager() = 0;
  virtual CFG& getCFG() = 0;
  virtual ParentMap& getParentMap() = 0;
  virtual LiveVariables& getLiveVariables() = 0;
};
  
class BugReporter {
public:
  enum Kind { BaseBRKind, GRBugReporterKind };

protected:
  Kind kind;  
  BugReporterData& D;
  
  BugReporter(BugReporterData& d, Kind k) : kind(k), D(d) {}
  
public:
  BugReporter(BugReporterData& d) : kind(BaseBRKind), D(d) {}
  virtual ~BugReporter();
  
  Kind getKind() const { return kind; }
  
  Diagnostic& getDiagnostic() {
    return D.getDiagnostic();
  }
  
  PathDiagnosticClient* getPathDiagnosticClient() {
    return D.getPathDiagnosticClient();
  }
  
  ASTContext& getContext() {
    return D.getContext();
  }
  
  SourceManager& getSourceManager() {
    return D.getSourceManager();
  }
  
  CFG& getCFG() {
    return D.getCFG();
  }
  
  ParentMap& getParentMap() {
    return D.getParentMap();  
  }
  
  LiveVariables& getLiveVariables() {
    return D.getLiveVariables();
  }
  
  virtual void GeneratePathDiagnostic(PathDiagnostic& PD, BugReport& R) {}

  void EmitWarning(BugReport& R);
  
  static bool classof(const BugReporter* R) { return true; }
};
  
class GRBugReporter : public BugReporter {
  GRExprEngine& Eng;
  llvm::SmallSet<SymbolID, 10> NotableSymbols;
public:
  
  GRBugReporter(BugReporterData& d, GRExprEngine& eng)
    : BugReporter(d, GRBugReporterKind), Eng(eng) {}
  
  virtual ~GRBugReporter();
  
  GRExprEngine& getEngine() {
    return Eng;
  }

  ExplodedGraph<ValueState>& getGraph();
  
  ValueStateManager& getStateManager();
  
  virtual void GeneratePathDiagnostic(PathDiagnostic& PD, BugReport& R);

  void addNotableSymbol(SymbolID Sym) {
    NotableSymbols.insert(Sym);
  }
  
  bool isNotable(SymbolID Sym) const {
    return (bool) NotableSymbols.count(Sym);
  }
  
  static bool classof(const BugReporter* R) {
    return R->getKind() == GRBugReporterKind;
  }
};
  

class DiagBugReport : public RangedBugReport {
  std::list<std::string> Strs;
  FullSourceLoc L;
  const char* description;
public:
  DiagBugReport(const char* desc, BugType& D, FullSourceLoc l) :
  RangedBugReport(D, NULL), L(l), description(desc) {}
  
  virtual ~DiagBugReport() {}
  virtual FullSourceLoc getLocation(SourceManager&) { return L; }
  
  virtual const char* getDescription() const {
    return description;
  }
  
  void addString(const std::string& s) { Strs.push_back(s); }  
  
  typedef std::list<std::string>::const_iterator str_iterator;
  str_iterator str_begin() const { return Strs.begin(); }
  str_iterator str_end() const { return Strs.end(); }
};

class DiagCollector : public DiagnosticClient {
  std::list<DiagBugReport> Reports;
  BugType& D;
public:
  DiagCollector(BugType& d) : D(d) {}
  
  virtual ~DiagCollector() {}
  
  virtual void HandleDiagnostic(Diagnostic &Diags, 
                                Diagnostic::Level DiagLevel,
                                FullSourceLoc Pos,
                                diag::kind ID,
                                const std::string *Strs,
                                unsigned NumStrs,
                                const SourceRange *Ranges, 
                                unsigned NumRanges) {
    
    // FIXME: Use a map from diag::kind to BugType, instead of having just
    //  one BugType.
    
    Reports.push_back(DiagBugReport(Diags.getDescription(ID), D, Pos));
    DiagBugReport& R = Reports.back();
    
    for ( ; NumRanges ; --NumRanges, ++Ranges)
      R.addRange(*Ranges);
    
    for ( ; NumStrs ; --NumStrs, ++Strs)
      R.addString(*Strs);    
  }
  
  // Iterators.
  
  typedef std::list<DiagBugReport>::iterator iterator;
  iterator begin() { return Reports.begin(); }
  iterator end() { return Reports.end(); }
};
  
} // end clang namespace

#endif
