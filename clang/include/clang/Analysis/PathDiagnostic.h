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
#include <algorithm>

namespace clang {

class PathDiagnosticPiece {
public:
  enum DisplayHint { Above, Below };

private:
  FullSourceLoc Pos;
  std::string str;
  DisplayHint Hint;
  std::vector<SourceRange> ranges;
  
public:
  
  PathDiagnosticPiece(FullSourceLoc pos, const std::string& s,
                      DisplayHint hint = Above)
    : Pos(pos), str(s), Hint(hint) {}
  
  PathDiagnosticPiece(FullSourceLoc pos, const char* s,
                      DisplayHint hint = Above)
    : Pos(pos), str(s), Hint(hint) {}
  
  const std::string& getString() const { return str; }
   
  DisplayHint getDisplayHint() const { return Hint; }
  
  void addRange(SourceRange R) {
    ranges.push_back(R);
  }
  
  void addRange(SourceLocation B, SourceLocation E) {
    ranges.push_back(SourceRange(B,E));
  }
  
  typedef const SourceRange* range_iterator;
  
  range_iterator ranges_begin() const {
    return ranges.empty() ? NULL : &ranges[0];
  }
  
  range_iterator ranges_end() const { 
    return ranges_begin() + ranges.size();
  }
    
  const SourceManager& getSourceManager() const {
    return Pos.getManager();
  }
    
  FullSourceLoc getLocation() const { return Pos; }
};
  
class PathDiagnostic {
  std::list<PathDiagnosticPiece*> path;
  unsigned Size;
  std::string Desc;
  std::vector<std::string> OtherDesc;

public:  
  PathDiagnostic() : Size(0) {}

  PathDiagnostic(const char* desc) : Size(0), Desc(desc) {}
  
  PathDiagnostic(const std::string& desc) : Size(0), Desc(desc) {}
  
  ~PathDiagnostic();

  const std::string& getDescription() const { return Desc; }
  
  typedef std::vector<std::string>::const_iterator meta_iterator;
  meta_iterator meta_begin() const { return OtherDesc.begin(); }
  meta_iterator meta_end() const { return OtherDesc.end(); }
  void addMeta(const std::string& s) { OtherDesc.push_back(s); }
  void addMeta(const char* s) { OtherDesc.push_back(s); }
  
  void push_front(PathDiagnosticPiece* piece) {
    path.push_front(piece);
    ++Size;
  }
  
  void push_back(PathDiagnosticPiece* piece) {
    path.push_back(piece);
    ++Size;
  }
  
  PathDiagnosticPiece* back() {
    return path.back();
  }
  
  const PathDiagnosticPiece* back() const {
    return path.back();
  }
  
  unsigned size() const { return Size; }
  bool empty() const { return Size == 0; }
  
  class iterator {
  public:  
    typedef std::list<PathDiagnosticPiece*>::iterator ImplTy;
    
    typedef PathDiagnosticPiece              value_type;
    typedef value_type&                      reference;
    typedef value_type*                      pointer;
    typedef ptrdiff_t                        difference_type;
    typedef std::bidirectional_iterator_tag  iterator_category;
    
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
    
    typedef const PathDiagnosticPiece        value_type;
    typedef value_type&                      reference;
    typedef value_type*                      pointer;
    typedef ptrdiff_t                        difference_type;
    typedef std::bidirectional_iterator_tag  iterator_category;
    
  private:
    ImplTy I;

  public:
    const_iterator(const ImplTy& i) : I(i) {}
    
    bool operator==(const const_iterator& X) const { return I == X.I; }
    bool operator!=(const const_iterator& X) const { return I != X.I; }
    
    reference operator*() const { return **I; }
    pointer operator->() const { return *I; }
    
    const_iterator& operator++() { ++I; return *this; }
    const_iterator& operator--() { --I; return *this; }
  };

  typedef std::reverse_iterator<iterator>       reverse_iterator;
  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;  


  // forward iterator creation methods.

  iterator begin() { return path.begin(); }
  iterator end() { return path.end(); }
  
  const_iterator begin() const { return path.begin(); }
  const_iterator end() const { return path.end(); }
  
  // reverse iterator creation methods.
  reverse_iterator rbegin()            { return reverse_iterator(end()); }
  const_reverse_iterator rbegin() const{ return const_reverse_iterator(end()); }
  reverse_iterator rend()              { return reverse_iterator(begin()); }
  const_reverse_iterator rend() const { return const_reverse_iterator(begin());}
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
    
  virtual void HandlePathDiagnostic(const PathDiagnostic* D) = 0;
};

} //end clang namespace
#endif
