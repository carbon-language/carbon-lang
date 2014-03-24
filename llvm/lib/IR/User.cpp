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

//===----------------------------------------------------------------------===//
//                                 User Class
//===----------------------------------------------------------------------===//

void User::anchor() {}

// replaceUsesOfWith - Replaces all references to the "From" definition with
// references to the "To" definition.
//
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

Use *User::allocHungoffUses(unsigned N) const {
  // Allocate the array of Uses, followed by a pointer (with bottom bit set) to
  // the User.
  size_t size = N * sizeof(Use) + sizeof(Use::UserRef);
  Use *Begin = static_cast<Use*>(::operator new(size));
  Use *End = Begin + N;
  (void) new(End) Use::UserRef(const_cast<User*>(this), 1);
  return Use::initTags(Begin, End);
}

//===----------------------------------------------------------------------===//
//                         User operator new Implementations
//===----------------------------------------------------------------------===//

void *User::operator new(size_t s, unsigned Us) {
  void *Storage = ::operator new(s + sizeof(Use) * Us);
  Use *Start = static_cast<Use*>(Storage);
  Use *End = Start + Us;
  User *Obj = reinterpret_cast<User*>(End);
  Obj->OperandList = Start;
  Obj->NumOperands = Us;
  Use::initTags(Start, End);
  return Obj;
}

//===----------------------------------------------------------------------===//
//                         User operator delete Implementation
//===----------------------------------------------------------------------===//

void User::operator delete(void *Usr) {
  User *Start = static_cast<User*>(Usr);
  Use *Storage = static_cast<Use*>(Usr) - Start->NumOperands;
  // If there were hung-off uses, they will have been freed already and
  // NumOperands reset to 0, so here we just free the User itself.
  ::operator delete(Storage);
}

//===----------------------------------------------------------------------===//
//                             Operator Class
//===----------------------------------------------------------------------===//

#if defined(_MSC_VER)
// In Release modes, Visual Studio complains that the Operator destructor
// never returns, which is true by design. 
// This does *not* depend on llvm_unreachable being dependent on NDEBUG:
// even if llvm_unreachable is replaced by __assume(false), VC still warns in
// Release modes but not in Debug modes. The real reason is optimization flags.
// With /Od in Debug modes the warning is not issued whereas with /O1 it is.
// I could not find any documentation to this effect, it is reproducable:
// Try compiling http://msdn.microsoft.com/en-us/library/khwfyc5d(v=vs.90).aspx
// with /O1 and then with /Od.
// Anyhow, solution is same as lib/Support/Process.cpp:~self_process().

#pragma warning(push)
#pragma warning(disable:4722)
#endif

Operator::~Operator() {
  llvm_unreachable("should never destroy an Operator");
}

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

} // End llvm namespace
