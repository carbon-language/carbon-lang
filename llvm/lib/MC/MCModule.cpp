//===- lib/MC/MCModule.cpp - MCModule implementation ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCModule.h"
#include "llvm/MC/MCAtom.h"
#include "llvm/MC/MCFunction.h"
#include <algorithm>

using namespace llvm;

static bool AtomComp(const MCAtom *LHS, const MCAtom *RHS) {
  return LHS->getEndAddr() < RHS->getEndAddr();
}

MCModule::const_atom_iterator MCModule::atom_lower_bound(uint64_t Addr) const {
  // This is a dummy atom, because VS 2008 doesn't like asymmetric comparators.
  MCDataAtom AddrAtom(0, Addr, Addr);
  return std::lower_bound(atom_begin(), atom_end(), &AddrAtom, AtomComp);
}

MCModule::atom_iterator MCModule::atom_lower_bound(uint64_t Addr) {
  MCDataAtom AddrAtom(0, Addr, Addr);
  return std::lower_bound(atom_begin(), atom_end(), &AddrAtom, AtomComp);
}

void MCModule::map(MCAtom *NewAtom) {
  uint64_t Begin = NewAtom->Begin;

  assert(Begin <= NewAtom->End && "Creating MCAtom with endpoints reversed?");

  // Check for atoms already covering this range.
  AtomListTy::iterator I = atom_lower_bound(Begin);
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
  AtomListTy::iterator I = atom_lower_bound(Atom->Begin);
  assert(I != atom_end() && "Atom offset not found in module!");
  assert(*I == Atom && "Previous atom mapping was invalid!");
  Atoms.erase(I);

  // Insert the new mapping.
  AtomListTy::iterator NewI = atom_lower_bound(NewBegin);
  Atoms.insert(NewI, Atom);

  // Update the atom internal bounds.
  Atom->Begin = NewBegin;
  Atom->End = NewEnd;
}

const MCAtom *MCModule::findAtomContaining(uint64_t Addr) const {
  AtomListTy::const_iterator I = atom_lower_bound(Addr);
  if (I != atom_end() && (*I)->getBeginAddr() <= Addr)
    return *I;
  return 0;
}

MCAtom *MCModule::findAtomContaining(uint64_t Addr) {
  AtomListTy::iterator I = atom_lower_bound(Addr);
  if (I != atom_end() && (*I)->getBeginAddr() <= Addr)
    return *I;
  return 0;
}

MCFunction *MCModule::createFunction(const StringRef &Name) {
  Functions.push_back(new MCFunction(Name));
  return Functions.back();
}

MCModule::~MCModule() {
  for (AtomListTy::iterator AI = atom_begin(),
                            AE = atom_end();
                            AI != AE; ++AI)
    delete *AI;
  for (FunctionListTy::iterator FI = func_begin(),
                                FE = func_end();
                                FI != FE; ++FI)
    delete *FI;
}
