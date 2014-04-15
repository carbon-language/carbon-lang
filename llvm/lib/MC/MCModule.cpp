//===- lib/MC/MCModule.cpp - MCModule implementation ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "llvm/MC/MCModule.h"
#include "llvm/MC/MCAtom.h"
#include "llvm/MC/MCFunction.h"
#include <algorithm>

using namespace llvm;

static bool AtomComp(const MCAtom *L, uint64_t Addr) {
  return L->getEndAddr() < Addr;
}

static bool AtomCompInv(uint64_t Addr, const MCAtom *R) {
  return Addr < R->getEndAddr();
}

void MCModule::map(MCAtom *NewAtom) {
  uint64_t Begin = NewAtom->Begin;

  assert(Begin <= NewAtom->End && "Creating MCAtom with endpoints reversed?");

  // Check for atoms already covering this range.
  AtomListTy::iterator I = std::lower_bound(atom_begin(), atom_end(),
                                            Begin, AtomComp);
  assert((I == atom_end() || (*I)->getBeginAddr() > NewAtom->End)
         && "Offset range already occupied!");

  // Insert the new atom to the list.
  Atoms.insert(I, NewAtom);
}

MCTextAtom *MCModule::createTextAtom(uint64_t Begin, uint64_t End) {
  MCTextAtom *NewAtom = new MCTextAtom(this, Begin, End);
  map(NewAtom);
  return NewAtom;
}

MCDataAtom *MCModule::createDataAtom(uint64_t Begin, uint64_t End) {
  MCDataAtom *NewAtom = new MCDataAtom(this, Begin, End);
  map(NewAtom);
  return NewAtom;
}

// remap - Update the interval mapping for an atom.
void MCModule::remap(MCAtom *Atom, uint64_t NewBegin, uint64_t NewEnd) {
  // Find and erase the old mapping.
  AtomListTy::iterator I = std::lower_bound(atom_begin(), atom_end(),
                                            Atom->Begin, AtomComp);
  assert(I != atom_end() && "Atom offset not found in module!");
  assert(*I == Atom && "Previous atom mapping was invalid!");
  Atoms.erase(I);

  // FIXME: special case NewBegin == Atom->Begin

  // Insert the new mapping.
  AtomListTy::iterator NewI = std::lower_bound(atom_begin(), atom_end(),
                                               NewBegin, AtomComp);
  assert((NewI == atom_end() || (*NewI)->getBeginAddr() > Atom->End)
         && "Offset range already occupied!");
  Atoms.insert(NewI, Atom);

  // Update the atom internal bounds.
  Atom->Begin = NewBegin;
  Atom->End = NewEnd;
}

const MCAtom *MCModule::findAtomContaining(uint64_t Addr) const {
  AtomListTy::const_iterator I = std::lower_bound(atom_begin(), atom_end(),
                                                  Addr, AtomComp);
  if (I != atom_end() && (*I)->getBeginAddr() <= Addr)
    return *I;
  return nullptr;
}

MCAtom *MCModule::findAtomContaining(uint64_t Addr) {
  return const_cast<MCAtom*>(
    const_cast<const MCModule *>(this)->findAtomContaining(Addr));
}

const MCAtom *MCModule::findFirstAtomAfter(uint64_t Addr) const {
  AtomListTy::const_iterator I = std::upper_bound(atom_begin(), atom_end(),
                                                  Addr, AtomCompInv);
  if (I != atom_end())
    return *I;
  return nullptr;
}

MCAtom *MCModule::findFirstAtomAfter(uint64_t Addr) {
  return const_cast<MCAtom*>(
    const_cast<const MCModule *>(this)->findFirstAtomAfter(Addr));
}

MCFunction *MCModule::createFunction(StringRef Name) {
  std::unique_ptr<MCFunction> MCF(new MCFunction(Name, this));
  Functions.push_back(std::move(MCF));
  return Functions.back().get();
}

static bool CompBBToAtom(MCBasicBlock *BB, const MCTextAtom *Atom) {
  return BB->getInsts() < Atom;
}

void MCModule::splitBasicBlocksForAtom(const MCTextAtom *TA,
                                       const MCTextAtom *NewTA) {
  BBsByAtomTy::iterator
    I = std::lower_bound(BBsByAtom.begin(), BBsByAtom.end(),
                         TA, CompBBToAtom);
  for (; I != BBsByAtom.end() && (*I)->getInsts() == TA; ++I) {
    MCBasicBlock *BB = *I;
    MCBasicBlock *NewBB = &BB->getParent()->createBlock(*NewTA);
    BB->splitBasicBlock(NewBB);
  }
}

void MCModule::trackBBForAtom(const MCTextAtom *Atom, MCBasicBlock *BB) {
  assert(Atom == BB->getInsts() && "Text atom doesn't back the basic block!");
  BBsByAtomTy::iterator I = std::lower_bound(BBsByAtom.begin(),
                                             BBsByAtom.end(),
                                             Atom, CompBBToAtom);
  for (; I != BBsByAtom.end() && (*I)->getInsts() == Atom; ++I)
    if (*I == BB)
      return;
  BBsByAtom.insert(I, BB);
}

MCModule::MCModule() : Entrypoint(0) { }

MCModule::~MCModule() {
  for (AtomListTy::iterator AI = atom_begin(),
                            AE = atom_end();
                            AI != AE; ++AI)
    delete *AI;
}
