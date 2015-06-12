//===-- User.cpp - Implement the User class -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/User.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Operator.h"

namespace llvm {
class BasicBlock;

//===----------------------------------------------------------------------===//
//                                 User Class
//===----------------------------------------------------------------------===//

void User::anchor() {}

void User::replaceUsesOfWith(Value *From, Value *To) {
  if (From == To) return;   // Duh what?

  assert((!isa<Constant>(this) || isa<GlobalValue>(this)) &&
         "Cannot call User::replaceUsesOfWith on a constant!");

  for (unsigned i = 0, E = getNumOperands(); i != E; ++i)
    if (getOperand(i) == From) {  // Is This operand is pointing to oldval?
      // The side effects of this setOperand call include linking to
      // "To", adding "this" to the uses list of To, and
      // most importantly, removing "this" from the use list of "From".
      setOperand(i, To); // Fix it now...
    }
}

//===----------------------------------------------------------------------===//
//                         User allocHungoffUses Implementation
//===----------------------------------------------------------------------===//

void User::allocHungoffUses(unsigned N, bool IsPhi) {
  assert(HasHungOffUses && "alloc must have hung off uses");
  // Allocate the array of Uses, followed by a pointer (with bottom bit set) to
  // the User.
  size_t size = N * sizeof(Use) + sizeof(Use::UserRef);
  if (IsPhi)
    size += N * sizeof(BasicBlock *);
  Use *Begin = static_cast<Use*>(::operator new(size));
  Use *End = Begin + N;
  (void) new(End) Use::UserRef(const_cast<User*>(this), 1);
  setOperandList(Use::initTags(Begin, End));
}

void User::growHungoffUses(unsigned NewNumUses, bool IsPhi) {
  assert(HasHungOffUses && "realloc must have hung off uses");

  unsigned OldNumUses = getNumOperands();

  // We don't support shrinking the number of uses.  We wouldn't have enough
  // space to copy the old uses in to the new space.
  assert(NewNumUses > OldNumUses && "realloc must grow num uses");

  Use *OldOps = getOperandList();
  allocHungoffUses(NewNumUses, IsPhi);
  Use *NewOps = getOperandList();

  // Now copy from the old operands list to the new one.
  std::copy(OldOps, OldOps + OldNumUses, NewOps);

  // If this is a Phi, then we need to copy the BB pointers too.
  if (IsPhi) {
    auto *OldPtr =
        reinterpret_cast<char *>(OldOps + OldNumUses) + sizeof(Use::UserRef);
    auto *NewPtr =
        reinterpret_cast<char *>(NewOps + NewNumUses) + sizeof(Use::UserRef);
    std::copy(OldPtr, OldPtr + (OldNumUses * sizeof(BasicBlock *)), NewPtr);
  }
  Use::zap(OldOps, OldOps + OldNumUses, true);
}

//===----------------------------------------------------------------------===//
//                         User operator new Implementations
//===----------------------------------------------------------------------===//

void *User::operator new(size_t Size, unsigned Us) {
  assert(Us < (1u << NumUserOperandsBits) && "Too many operands");
  void *Storage = ::operator new(Size + sizeof(Use) * Us);
  Use *Start = static_cast<Use*>(Storage);
  Use *End = Start + Us;
  User *Obj = reinterpret_cast<User*>(End);
  Obj->setOperandList(Start);
  Obj->HasHungOffUses = false;
  Obj->NumUserOperands = Us;
  Use::initTags(Start, End);
  return Obj;
}

void *User::operator new(size_t Size) {
  void *Storage = ::operator new(Size);
  User *Obj = reinterpret_cast<User*>(Storage);
  Obj->setOperandList(nullptr);
  Obj->HasHungOffUses = true;
  Obj->NumUserOperands = 0;
  return Obj;
}

//===----------------------------------------------------------------------===//
//                         User operator delete Implementation
//===----------------------------------------------------------------------===//

void User::operator delete(void *Usr) {
  User *Start = static_cast<User*>(Usr);
  Use *Storage = static_cast<Use*>(Usr) - Start->NumUserOperands;
  // If there were hung-off uses, they will have been freed already and
  // NumOperands reset to 0, so here we just free the User itself.
  ::operator delete(Storage);
}

//===----------------------------------------------------------------------===//
//                             Operator Class
//===----------------------------------------------------------------------===//

Operator::~Operator() {
  llvm_unreachable("should never destroy an Operator");
}

} // End llvm namespace
