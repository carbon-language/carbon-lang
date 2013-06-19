//===--- MicrosoftVBTables.h - Virtual Base Table Emission ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class generates data about MSVC virtual base tables.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/BaseSubobject.h"
#include "clang/Basic/LLVM.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/IR/GlobalVariable.h"
#include <vector>

namespace clang {

class ASTRecordLayout;

namespace CodeGen {

class CodeGenModule;

struct VBTableInfo {
  VBTableInfo(const CXXRecordDecl *ReusingBase, BaseSubobject VBPtrSubobject,
              llvm::GlobalVariable *GV)
    : ReusingBase(ReusingBase), VBPtrSubobject(VBPtrSubobject), GV(GV) { }

  /// The vbtable will hold all of the virtual bases of ReusingBase.  This may
  /// or may not be the same class as VBPtrSubobject.Base.  A derived class will
  /// reuse the vbptr of the first non-virtual base subobject that has one.
  const CXXRecordDecl *ReusingBase;

  /// The vbptr is stored inside this subobject.
  BaseSubobject VBPtrSubobject;

  /// The GlobalVariable for this vbtable.
  llvm::GlobalVariable *GV;

  /// \brief Emits a definition for GV by setting it's initializer.
  void EmitVBTableDefinition(CodeGenModule &CGM, const CXXRecordDecl *RD,
                             llvm::GlobalVariable::LinkageTypes Linkage) const;
};

// These are embedded in a DenseMap and the elements are large, so we don't want
// SmallVector.
typedef std::vector<VBTableInfo> VBTableVector;

struct VBTablePath;

typedef llvm::SmallVector<VBTablePath *, 6> VBTablePathVector;

/// Produces MSVC-compatible vbtable data.  The symbols produced by this builder
/// match those produced by MSVC 2012, which is different from MSVC 2010.
///
/// Unlike Itanium, which uses only one vtable per class, MSVC uses a different
/// symbol for every "address point" installed in base subobjects.  As a result,
/// we have to compute unique symbols for every table.  Since there can be
/// multiple non-virtual base subobjects of the same class, combining the most
/// derived class with the base containing the vtable is insufficient.  The most
/// trivial algorithm would be to mangle in the entire path from base to most
/// derived, but that would be too easy and would create unnecessarily large
/// symbols.  ;)
///
/// MSVC 2012 appears to minimize the vbtable names using the following
/// algorithm.  First, walk the class hierarchy in the usual order, depth first,
/// left to right, to find all of the subobjects which contain a vbptr field.
/// Visiting each class node yields a list of inheritance paths to vbptrs.  Each
/// record with a vbptr creates an initially empty path.
///
/// To combine paths from child nodes, the paths are compared to check for
/// ambiguity.  Paths are "ambiguous" if multiple paths have the same set of
/// components in the same order.  Each group of ambiguous paths is extended by
/// appending the class of the base from which it came.  If the current class
/// node produced an ambiguous path, its path is extended with the current class.
/// After extending paths, MSVC again checks for ambiguity, and extends any
/// ambiguous path which wasn't already extended.  Because each node yields an
/// unambiguous set of paths, MSVC doesn't need to extend any path more than once
/// to produce an unambiguous set of paths.
///
/// The VBTableBuilder class attempts to implement this algorithm by repeatedly
/// bucketing paths together by sorting them.
///
/// TODO: Presumably vftables use the same algorithm.
///
/// TODO: Implement the MSVC 2010 name mangling scheme to avoid emitting
/// duplicate vbtables with different symbols.
class VBTableBuilder {
public:
  VBTableBuilder(CodeGenModule &CGM, const CXXRecordDecl *MostDerived);

  void enumerateVBTables(VBTableVector &VBTables);

private:
  bool hasVBPtr(const CXXRecordDecl *RD);

  llvm::GlobalVariable *getAddrOfVBTable(const CXXRecordDecl *ReusingBase,
                                      ArrayRef<const CXXRecordDecl *> BasePath);

  /// Enumerates paths to bases with vbptrs.  The paths elements are compressed
  /// to contain only the classes necessary to form an unambiguous path.
  void findUnambiguousPaths(const CXXRecordDecl *ReusingBase,
                            BaseSubobject CurSubobject,
                            VBTablePathVector &Paths);

  void extendPath(VBTablePath *Info, bool SecondPass);

  bool rebucketPaths(VBTablePathVector &Paths, size_t PathsStart,
                     bool SecondPass = false);

  CodeGenModule &CGM;

  const CXXRecordDecl *MostDerived;

  /// Caches the layout of the most derived class.
  const ASTRecordLayout &DerivedLayout;

  /// Set of vbases to avoid re-visiting the same vbases.
  llvm::SmallPtrSet<const CXXRecordDecl*, 4> VBasesSeen;
};

} // namespace CodeGen

} // namespace clang
