//===--- PathDiagnostic.h - Path-Specific Diagnostic Handling ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the PathDiagnostic-related interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_PATH_DIAGNOSTIC_H
#define LLVM_CLANG_PATH_DIAGNOSTIC_H

#include "clang/Basic/Diagnostic.h"
#include "llvm/ADT/OwningPtr.h"

#include <vector>
#include <list>
#include <string>

namespace clang {

class PathDiagnosticPiece {
  FullSourceLoc Pos;
  std::vector<std::string> strs;
  std::vector<SourceRange> ranges;
public:
  
  PathDiagnosticPiece(FullSourceLoc pos) : Pos(pos) {}
  
  void addString(const std::string& s) {
    strs.push_back(s);
  }
  
  const std::string* strs_begin() const {
    return strs.empty() ? NULL : &strs[0];
  }

  const std::string* strs_end() const {
    return strs_begin() + strs.size();
  }
  
  void addRange(SourceRange R) {
    ranges.push_back(R);
  }
  
  void addRange(SourceLocation B, SourceLocation E) {
    ranges.push_back(SourceRange(B,E));
  }
  
  const SourceRange* ranges_begin() const {
    return ranges.empty() ? NULL : &ranges[0];
  }
  
  const SourceRange* ranges_end() const { 
    return ranges_begin() + ranges.size();
  }
    
  const SourceManager& getSourceManager() const {
    return Pos.getManager();
  }
    
  FullSourceLoc getLocation() const { return Pos; }
};
  
class PathDiagnostic {
  std::list<PathDiagnosticPiece*> path;
  Diagnostic::Level DiagLevel;
  diag::kind ID;
  unsigned Size;

public:
  
  PathDiagnostic(Diagnostic::Level lvl, diag::kind i)
    : DiagLevel(lvl), ID(i), Size(0) {}
  
  ~PathDiagnostic();

  void push_front(PathDiagnosticPiece* piece) {
    path.push_front(piece);
    ++Size;
  }
  
  void push_back(PathDiagnosticPiece* piece) {
    path.push_back(piece);
    ++Size;
  }
  
  class iterator {
  public:  
    typedef std::list<PathDiagnosticPiece*>::iterator ImplTy;
    
  private:
    ImplTy I;
    
  public:
    iterator(const ImplTy& i) : I(i) {}
    
    bool operator==(const iterator& X) const { return I == X.I; }
    bool operator!=(const iterator& X) const { return I != X.I; }
    
    PathDiagnosticPiece& operator*() const { return **I; }
    PathDiagnosticPiece* operator->() const { return *I; }
    
    iterator& operator++() { ++I; return *this; }
    iterator& operator--() { --I; return *this; }
  };
  
  class const_iterator {
  public:  
    typedef std::list<PathDiagnosticPiece*>::const_iterator ImplTy;
    
  private:
    ImplTy I;

  public:
    const_iterator(const ImplTy& i) : I(i) {}
    
    bool operator==(const const_iterator& X) const { return I == X.I; }
    bool operator!=(const const_iterator& X) const { return I != X.I; }
    
    const PathDiagnosticPiece& operator*() const { return **I; }
    const PathDiagnosticPiece* operator->() const { return *I; }
    
    const_iterator& operator++() { ++I; return *this; }
    const_iterator& operator--() { --I; return *this; }
  };
  
  iterator begin() { return path.begin(); }
  iterator end() { return path.end(); }

  const_iterator begin() const { return path.begin(); }
  const_iterator end() const { return path.end(); }
  
  unsigned size() const { return Size; }
  bool empty() const { return Size == 0; }
  
  Diagnostic::Level getLevel() const { return DiagLevel; }
  diag::kind getDiagKind() const { return ID; }
};
  
class PathDiagnosticClient : public DiagnosticClient  {
public:
  PathDiagnosticClient() {}
  virtual ~PathDiagnosticClient() {}
    
  virtual void HandleDiagnostic(Diagnostic &Diags, 
                                Diagnostic::Level DiagLevel,
                                FullSourceLoc Pos,
                                diag::kind ID,
                                const std::string *Strs,
                                unsigned NumStrs,
                                const SourceRange *Ranges, 
                                unsigned NumRanges);
    
  virtual void HandlePathDiagnostic(Diagnostic& Diag,
                                    const PathDiagnostic& D) = 0;
};

} //end clang namespace
#endif
