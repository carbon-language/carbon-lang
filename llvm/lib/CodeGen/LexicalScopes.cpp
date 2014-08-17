//===- LexicalScopes.cpp - Collecting lexical scope info ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements LexicalScopes analysis.
//
// This pass collects lexical scope information and maps machine instructions
// to respective lexical scopes.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/LexicalScopes.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormattedStream.h"
using namespace llvm;

#define DEBUG_TYPE "lexicalscopes"

/// reset - Reset the instance so that it's prepared for another function.
void LexicalScopes::reset() {
  MF = nullptr;
  CurrentFnLexicalScope = nullptr;
  LexicalScopeMap.clear();
  AbstractScopeMap.clear();
  InlinedLexicalScopeMap.clear();
  AbstractScopesList.clear();
}

/// initialize - Scan machine function and constuct lexical scope nest.
void LexicalScopes::initialize(const MachineFunction &Fn) {
  reset();
  MF = &Fn;
  SmallVector<InsnRange, 4> MIRanges;
  DenseMap<const MachineInstr *, LexicalScope *> MI2ScopeMap;
  extractLexicalScopes(MIRanges, MI2ScopeMap);
  if (CurrentFnLexicalScope) {
    constructScopeNest(CurrentFnLexicalScope);
    assignInstructionRanges(MIRanges, MI2ScopeMap);
  }
}

/// extractLexicalScopes - Extract instruction ranges for each lexical scopes
/// for the given machine function.
void LexicalScopes::extractLexicalScopes(
    SmallVectorImpl<InsnRange> &MIRanges,
    DenseMap<const MachineInstr *, LexicalScope *> &MI2ScopeMap) {

  // Scan each instruction and create scopes. First build working set of scopes.
  for (const auto &MBB : *MF) {
    const MachineInstr *RangeBeginMI = nullptr;
    const MachineInstr *PrevMI = nullptr;
    DebugLoc PrevDL;
    for (const auto &MInsn : MBB) {
      // Check if instruction has valid location information.
      const DebugLoc MIDL = MInsn.getDebugLoc();
      if (MIDL.isUnknown()) {
        PrevMI = &MInsn;
        continue;
      }

      // If scope has not changed then skip this instruction.
      if (MIDL == PrevDL) {
        PrevMI = &MInsn;
        continue;
      }

      // Ignore DBG_VALUE. It does not contribute to any instruction in output.
      if (MInsn.isDebugValue())
        continue;

      if (RangeBeginMI) {
        // If we have already seen a beginning of an instruction range and
        // current instruction scope does not match scope of first instruction
        // in this range then create a new instruction range.
        InsnRange R(RangeBeginMI, PrevMI);
        MI2ScopeMap[RangeBeginMI] = getOrCreateLexicalScope(PrevDL);
        MIRanges.push_back(R);
      }

      // This is a beginning of a new instruction range.
      RangeBeginMI = &MInsn;

      // Reset previous markers.
      PrevMI = &MInsn;
      PrevDL = MIDL;
    }

    // Create last instruction range.
    if (RangeBeginMI && PrevMI && !PrevDL.isUnknown()) {
      InsnRange R(RangeBeginMI, PrevMI);
      MIRanges.push_back(R);
      MI2ScopeMap[RangeBeginMI] = getOrCreateLexicalScope(PrevDL);
    }
  }
}

LexicalScope *LexicalScopes::findInlinedScope(DebugLoc DL) {
  MDNode *Scope = nullptr;
  MDNode *IA = nullptr;
  DL.getScopeAndInlinedAt(Scope, IA, MF->getFunction()->getContext());
  auto I = InlinedLexicalScopeMap.find(std::make_pair(Scope, IA));
  return I != InlinedLexicalScopeMap.end() ? &I->second : nullptr;
}

/// findLexicalScope - Find lexical scope, either regular or inlined, for the
/// given DebugLoc. Return NULL if not found.
LexicalScope *LexicalScopes::findLexicalScope(DebugLoc DL) {
  MDNode *Scope = nullptr;
  MDNode *IA = nullptr;
  DL.getScopeAndInlinedAt(Scope, IA, MF->getFunction()->getContext());
  if (!Scope)
    return nullptr;

  // The scope that we were created with could have an extra file - which
  // isn't what we care about in this case.
  DIDescriptor D = DIDescriptor(Scope);
  if (D.isLexicalBlockFile())
    Scope = DILexicalBlockFile(Scope).getScope();

  if (IA) {
    auto I = InlinedLexicalScopeMap.find(std::make_pair(Scope, IA));
    return I != InlinedLexicalScopeMap.end() ? &I->second : nullptr;
  }
  return findLexicalScope(Scope);
}

/// getOrCreateLexicalScope - Find lexical scope for the given DebugLoc. If
/// not available then create new lexical scope.
LexicalScope *LexicalScopes::getOrCreateLexicalScope(DebugLoc DL) {
  MDNode *Scope = nullptr;
  MDNode *InlinedAt = nullptr;
  DL.getScopeAndInlinedAt(Scope, InlinedAt, MF->getFunction()->getContext());

  if (InlinedAt) {
    // Create an abstract scope for inlined function.
    getOrCreateAbstractScope(Scope);
    // Create an inlined scope for inlined function.
    return getOrCreateInlinedScope(Scope, InlinedAt);
  }

  return getOrCreateRegularScope(Scope);
}

/// getOrCreateRegularScope - Find or create a regular lexical scope.
LexicalScope *LexicalScopes::getOrCreateRegularScope(MDNode *Scope) {
  DIDescriptor D = DIDescriptor(Scope);
  if (D.isLexicalBlockFile()) {
    Scope = DILexicalBlockFile(Scope).getScope();
    D = DIDescriptor(Scope);
  }

  auto I = LexicalScopeMap.find(Scope);
  if (I != LexicalScopeMap.end())
    return &I->second;

  LexicalScope *Parent = nullptr;
  if (D.isLexicalBlock())
    Parent = getOrCreateLexicalScope(DebugLoc::getFromDILexicalBlock(Scope));
  // FIXME: Use forward_as_tuple instead of make_tuple, once MSVC2012
  // compatibility is no longer required.
  I = LexicalScopeMap.emplace(std::piecewise_construct, std::make_tuple(Scope),
                              std::make_tuple(Parent, DIDescriptor(Scope),
                                              nullptr, false)).first;

  if (!Parent && DIDescriptor(Scope).isSubprogram() &&
      DISubprogram(Scope).describes(MF->getFunction()))
    CurrentFnLexicalScope = &I->second;

  return &I->second;
}

/// getOrCreateInlinedScope - Find or create an inlined lexical scope.
LexicalScope *LexicalScopes::getOrCreateInlinedScope(MDNode *ScopeNode,
                                                     MDNode *InlinedAt) {
  std::pair<const MDNode*, const MDNode*> P(ScopeNode, InlinedAt);
  auto I = InlinedLexicalScopeMap.find(P);
  if (I != InlinedLexicalScopeMap.end())
    return &I->second;

  LexicalScope *Parent;
  DILexicalBlock Scope(ScopeNode);
  if (Scope.isSubprogram())
    Parent = getOrCreateLexicalScope(DebugLoc::getFromDILocation(InlinedAt));
  else
    Parent = getOrCreateInlinedScope(Scope.getContext(), InlinedAt);

  // FIXME: Use forward_as_tuple instead of make_tuple, once MSVC2012
  // compatibility is no longer required.
  I = InlinedLexicalScopeMap.emplace(std::piecewise_construct,
                                     std::make_tuple(P),
                                     std::make_tuple(Parent, Scope, InlinedAt,
                                                     false)).first;
  return &I->second;
}

/// getOrCreateAbstractScope - Find or create an abstract lexical scope.
LexicalScope *LexicalScopes::getOrCreateAbstractScope(const MDNode *N) {
  assert(N && "Invalid Scope encoding!");

  DIDescriptor Scope(N);
  if (Scope.isLexicalBlockFile())
    Scope = DILexicalBlockFile(Scope).getScope();
  auto I = AbstractScopeMap.find(Scope);
  if (I != AbstractScopeMap.end())
    return &I->second;

  LexicalScope *Parent = nullptr;
  if (Scope.isLexicalBlock()) {
    DILexicalBlock DB(Scope);
    DIDescriptor ParentDesc = DB.getContext();
    Parent = getOrCreateAbstractScope(ParentDesc);
  }
  I = AbstractScopeMap.emplace(std::piecewise_construct,
                               std::forward_as_tuple(Scope),
                               std::forward_as_tuple(Parent, Scope,
                                                     nullptr, true)).first;
  if (Scope.isSubprogram())
    AbstractScopesList.push_back(&I->second);
  return &I->second;
}

/// constructScopeNest
void LexicalScopes::constructScopeNest(LexicalScope *Scope) {
  assert(Scope && "Unable to calculate scope dominance graph!");
  SmallVector<LexicalScope *, 4> WorkStack;
  WorkStack.push_back(Scope);
  unsigned Counter = 0;
  while (!WorkStack.empty()) {
    LexicalScope *WS = WorkStack.back();
    const SmallVectorImpl<LexicalScope *> &Children = WS->getChildren();
    bool visitedChildren = false;
    for (SmallVectorImpl<LexicalScope *>::const_iterator SI = Children.begin(),
                                                         SE = Children.end();
         SI != SE; ++SI) {
      LexicalScope *ChildScope = *SI;
      if (!ChildScope->getDFSOut()) {
        WorkStack.push_back(ChildScope);
        visitedChildren = true;
        ChildScope->setDFSIn(++Counter);
        break;
      }
    }
    if (!visitedChildren) {
      WorkStack.pop_back();
      WS->setDFSOut(++Counter);
    }
  }
}

/// assignInstructionRanges - Find ranges of instructions covered by each
/// lexical scope.
void LexicalScopes::assignInstructionRanges(
    SmallVectorImpl<InsnRange> &MIRanges,
    DenseMap<const MachineInstr *, LexicalScope *> &MI2ScopeMap) {

  LexicalScope *PrevLexicalScope = nullptr;
  for (SmallVectorImpl<InsnRange>::const_iterator RI = MIRanges.begin(),
                                                  RE = MIRanges.end();
       RI != RE; ++RI) {
    const InsnRange &R = *RI;
    LexicalScope *S = MI2ScopeMap.lookup(R.first);
    assert(S && "Lost LexicalScope for a machine instruction!");
    if (PrevLexicalScope && !PrevLexicalScope->dominates(S))
      PrevLexicalScope->closeInsnRange(S);
    S->openInsnRange(R.first);
    S->extendInsnRange(R.second);
    PrevLexicalScope = S;
  }

  if (PrevLexicalScope)
    PrevLexicalScope->closeInsnRange();
}

/// getMachineBasicBlocks - Populate given set using machine basic blocks which
/// have machine instructions that belong to lexical scope identified by
/// DebugLoc.
void LexicalScopes::getMachineBasicBlocks(
    DebugLoc DL, SmallPtrSetImpl<const MachineBasicBlock *> &MBBs) {
  MBBs.clear();
  LexicalScope *Scope = getOrCreateLexicalScope(DL);
  if (!Scope)
    return;

  if (Scope == CurrentFnLexicalScope) {
    for (const auto &MBB : *MF)
      MBBs.insert(&MBB);
    return;
  }

  SmallVectorImpl<InsnRange> &InsnRanges = Scope->getRanges();
  for (SmallVectorImpl<InsnRange>::iterator I = InsnRanges.begin(),
                                            E = InsnRanges.end();
       I != E; ++I) {
    InsnRange &R = *I;
    MBBs.insert(R.first->getParent());
  }
}

/// dominates - Return true if DebugLoc's lexical scope dominates at least one
/// machine instruction's lexical scope in a given machine basic block.
bool LexicalScopes::dominates(DebugLoc DL, MachineBasicBlock *MBB) {
  LexicalScope *Scope = getOrCreateLexicalScope(DL);
  if (!Scope)
    return false;

  // Current function scope covers all basic blocks in the function.
  if (Scope == CurrentFnLexicalScope && MBB->getParent() == MF)
    return true;

  bool Result = false;
  for (MachineBasicBlock::iterator I = MBB->begin(), E = MBB->end(); I != E;
       ++I) {
    DebugLoc IDL = I->getDebugLoc();
    if (IDL.isUnknown())
      continue;
    if (LexicalScope *IScope = getOrCreateLexicalScope(IDL))
      if (Scope->dominates(IScope))
        return true;
  }
  return Result;
}

/// dump - Print data structures.
void LexicalScope::dump(unsigned Indent) const {
#ifndef NDEBUG
  raw_ostream &err = dbgs();
  err.indent(Indent);
  err << "DFSIn: " << DFSIn << " DFSOut: " << DFSOut << "\n";
  const MDNode *N = Desc;
  err.indent(Indent);
  N->dump();
  if (AbstractScope)
    err << std::string(Indent, ' ') << "Abstract Scope\n";

  if (!Children.empty())
    err << std::string(Indent + 2, ' ') << "Children ...\n";
  for (unsigned i = 0, e = Children.size(); i != e; ++i)
    if (Children[i] != this)
      Children[i]->dump(Indent + 2);
#endif
}
