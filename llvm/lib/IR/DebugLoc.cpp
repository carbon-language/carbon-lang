//===-- DebugLoc.cpp - Implement DebugLoc class ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/DebugLoc.h"
#include "LLVMContextImpl.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/IR/DebugInfo.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
// DebugLoc Implementation
//===----------------------------------------------------------------------===//
DebugLoc::DebugLoc(MDLocation *L) : Loc(L) {}
DebugLoc::DebugLoc(MDNode *L) : Loc(L) {}

MDLocation *DebugLoc::get() const {
  return cast_or_null<MDLocation>(Loc.get());
}

unsigned DebugLoc::getLine() const {
  assert(get() && "Expected valid DebugLoc");
  return get()->getLine();
}

unsigned DebugLoc::getCol() const {
  assert(get() && "Expected valid DebugLoc");
  return get()->getColumn();
}

MDNode *DebugLoc::getScope() const {
  assert(get() && "Expected valid DebugLoc");
  return get()->getScope();
}

MDLocation *DebugLoc::getInlinedAt() const {
  assert(get() && "Expected valid DebugLoc");
  return get()->getInlinedAt();
}

MDNode *DebugLoc::getInlinedAtScope() const {
  return cast<MDLocation>(Loc)->getInlinedAtScope();
}

DebugLoc DebugLoc::getFnDebugLoc() const {
  // FIXME: Add a method on \a MDLocation that does this work.
  const MDNode *Scope = getInlinedAtScope();
  if (DISubprogram SP = getDISubprogram(Scope))
    return DebugLoc::get(SP.getScopeLineNumber(), 0, SP);

  return DebugLoc();
}

DebugLoc DebugLoc::get(unsigned Line, unsigned Col,
                       MDNode *Scope, MDNode *InlinedAt) {
  // If no scope is available, this is an unknown location.
  if (!Scope)
    return DebugLoc();

  return MDLocation::get(Scope->getContext(), Line, Col, Scope, InlinedAt);
}

void DebugLoc::dump() const {
#ifndef NDEBUG
  if (!Loc)
    return;

  dbgs() << getLine();
  if (getCol() != 0)
    dbgs() << ',' << getCol();
  if (DebugLoc InlinedAtDL = DebugLoc(getInlinedAt())) {
    dbgs() << " @ ";
    InlinedAtDL.dump();
  } else
    dbgs() << "\n";
#endif
}

void DebugLoc::print(raw_ostream &OS) const {
  if (!Loc)
    return;

  // Print source line info.
  DIScope Scope = cast<MDScope>(getScope());
  OS << Scope.getFilename();
  OS << ':' << getLine();
  if (getCol() != 0)
    OS << ':' << getCol();

  if (DebugLoc InlinedAtDL = getInlinedAt()) {
    OS << " @[ ";
    InlinedAtDL.print(OS);
    OS << " ]";
  }
}
