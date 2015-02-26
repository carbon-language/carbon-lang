//===- DebugLoc.h - Debug Location Information ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a number of light weight data structures used
// to describe and track debug location information.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_DEBUGLOC_H
#define LLVM_IR_DEBUGLOC_H

#include "llvm/IR/TrackingMDRef.h"
#include "llvm/Support/DataTypes.h"

namespace llvm {

  class LLVMContext;
  class raw_ostream;
  class MDNode;

  /// DebugLoc - Debug location id.  This is carried by Instruction, SDNode,
  /// and MachineInstr to compactly encode file/line/scope information for an
  /// operation.
  class DebugLoc {
    TrackingMDNodeRef Loc;

  public:
    DebugLoc() {}
    DebugLoc(DebugLoc &&X) : Loc(std::move(X.Loc)) {}
    DebugLoc(const DebugLoc &X) : Loc(X.Loc) {}
    DebugLoc &operator=(DebugLoc &&X) {
      Loc = std::move(X.Loc);
      return *this;
    }
    DebugLoc &operator=(const DebugLoc &X) {
      Loc = X.Loc;
      return *this;
    }

    /// \brief Check whether this has a trivial destructor.
    bool hasTrivialDestructor() const { return Loc.hasTrivialDestructor(); }

    /// get - Get a new DebugLoc that corresponds to the specified line/col
    /// scope/inline location.
    static DebugLoc get(unsigned Line, unsigned Col, MDNode *Scope,
                        MDNode *InlinedAt = nullptr);

    /// getFromDILocation - Translate the DILocation quad into a DebugLoc.
    static DebugLoc getFromDILocation(MDNode *N);

    /// getFromDILexicalBlock - Translate the DILexicalBlock into a DebugLoc.
    static DebugLoc getFromDILexicalBlock(MDNode *N);

    /// isUnknown - Return true if this is an unknown location.
    bool isUnknown() const { return !Loc; }

    unsigned getLine() const;
    unsigned getCol() const;

    /// getScope - This returns the scope pointer for this DebugLoc, or null if
    /// invalid.
    MDNode *getScope() const;
    MDNode *getScope(const LLVMContext &) const { return getScope(); }

    /// getInlinedAt - This returns the InlinedAt pointer for this DebugLoc, or
    /// null if invalid or not present.
    MDNode *getInlinedAt() const;
    MDNode *getInlinedAt(const LLVMContext &) const { return getInlinedAt(); }

    /// getScopeAndInlinedAt - Return both the Scope and the InlinedAt values.
    void getScopeAndInlinedAt(MDNode *&Scope, MDNode *&IA) const;
    void getScopeAndInlinedAt(MDNode *&Scope, MDNode *&IA,
                              const LLVMContext &) const {
      return getScopeAndInlinedAt(Scope, IA);
    }

    /// getScopeNode - Get MDNode for DebugLoc's scope, or null if invalid.
    MDNode *getScopeNode() const;
    MDNode *getScopeNode(const LLVMContext &) const { return getScopeNode(); }

    // getFnDebugLoc - Walk up the scope chain of given debug loc and find line
    // number info for the function.
    DebugLoc getFnDebugLoc() const;
    DebugLoc getFnDebugLoc(const LLVMContext &) const {
      return getFnDebugLoc();
    }

    /// getAsMDNode - This method converts the compressed DebugLoc node into a
    /// DILocation compatible MDNode.
    MDNode *getAsMDNode() const;
    MDNode *getAsMDNode(LLVMContext &) const { return getAsMDNode(); }

    bool operator==(const DebugLoc &DL) const { return Loc == DL.Loc; }
    bool operator!=(const DebugLoc &DL) const { return !(*this == DL); }

    void dump() const;
    void dump(const LLVMContext &) const { dump(); }
    /// \brief prints source location /path/to/file.exe:line:col @[inlined at]
    void print(raw_ostream &OS) const;
  };

} // end namespace llvm

#endif /* LLVM_SUPPORT_DEBUGLOC_H */
