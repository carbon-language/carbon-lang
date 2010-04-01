//===-- llvm/Support/CallSite.h - Abstract Call & Invoke instrs -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the CallSite class, which is a handy wrapper for code that
// wants to treat Call and Invoke instructions in a generic way. When in non-
// mutation context (e.g. an analysis) ImmutableCallSite should be used.
// Finally, when some degree of customization is necessary between these two
// extremes, CallSiteBase<> can be supplied with fine-tuned parameters.
//
// NOTE: These classes are supposed to have "value semantics". So they should be
// passed by value, not by reference; they should not be "new"ed or "delete"d.
// They are efficiently copyable, assignable and constructable, with cost
// equivalent to copying a pointer (notice that they have only a single data
// member). The internal representation carries a flag which indicates which of
// the two variants is enclosed. This allows for cheaper checks when various
// accessors of CallSite are employed.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_CALLSITE_H
#define LLVM_SUPPORT_CALLSITE_H

#include "llvm/Attributes.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/BasicBlock.h"
#include "llvm/CallingConv.h"
#include "llvm/Instruction.h"

namespace llvm {

class CallInst;
class InvokeInst;

template <typename FunTy = const Function,
          typename ValTy = const Value,
          typename UserTy = const User,
          typename InstrTy = const Instruction,
          typename CallTy = const CallInst,
          typename InvokeTy = const InvokeInst,
          typename IterTy = User::const_op_iterator>
class CallSiteBase {
protected:
  PointerIntPair<InstrTy*, 1, bool> I;
public:
  CallSiteBase() : I(0, false) {}
  CallSiteBase(CallTy *CI) : I(reinterpret_cast<InstrTy*>(CI), true) {}
  CallSiteBase(InvokeTy *II) : I(reinterpret_cast<InstrTy*>(II), false) {}
  CallSiteBase(ValTy *II) { *this = get(II); }
  CallSiteBase(InstrTy *II) {
    assert(II && "Null instruction given?");
    *this = get(II);
    assert(I.getPointer());
  }

  /// CallSiteBase::get - This static method is sort of like a constructor.  It
  /// will create an appropriate call site for a Call or Invoke instruction, but
  /// it can also create a null initialized CallSiteBase object for something
  /// which is NOT a call site.
  ///
  static CallSiteBase get(ValTy *V) {
    if (InstrTy *II = dyn_cast<InstrTy>(V)) {
      if (II->getOpcode() == Instruction::Call)
        return CallSiteBase(reinterpret_cast<CallTy*>(II));
      else if (II->getOpcode() == Instruction::Invoke)
        return CallSiteBase(reinterpret_cast<InvokeTy*>(II));
    }
    return CallSiteBase();
  }

  /// isCall - true if a CallInst is enclosed.
  /// Note that !isCall() does not mean it is an InvokeInst enclosed,
  /// it also could signify a NULL Instruction pointer.
  bool isCall() const { return I.getInt(); }

  /// isInvoke - true if a InvokeInst is enclosed.
  ///
  bool isInvoke() const { return getInstruction() && !I.getInt(); }

  InstrTy *getInstruction() const { return I.getPointer(); }
  InstrTy *operator->() const { return I.getPointer(); }
  operator bool() const { return I.getPointer(); }

  /// getCalledValue - Return the pointer to function that is being called...
  ///
  ValTy *getCalledValue() const {
    assert(getInstruction() && "Not a call or invoke instruction!");
    return *getCallee();
  }

  /// getCalledFunction - Return the function being called if this is a direct
  /// call, otherwise return null (if it's an indirect call).
  ///
  FunTy *getCalledFunction() const {
    return dyn_cast<FunTy>(getCalledValue());
  }

  /// setCalledFunction - Set the callee to the specified value...
  ///
  void setCalledFunction(Value *V) {
    assert(getInstruction() && "Not a call or invoke instruction!");
    *getCallee() = V;
  }

  /// isCallee - Determine whether the passed iterator points to the
  /// callee operand's Use.
  ///
  bool isCallee(value_use_iterator<UserTy> UI) const {
    return getCallee() == &UI.getUse();
  }

  ValTy *getArgument(unsigned ArgNo) const {
    assert(arg_begin() + ArgNo < arg_end() && "Argument # out of range!");
    return *(arg_begin()+ArgNo);
  }

  void setArgument(unsigned ArgNo, Value* newVal) {
    assert(getInstruction() && "Not a call or invoke instruction!");
    assert(arg_begin() + ArgNo < arg_end() && "Argument # out of range!");
    getInstruction()->setOperand(getArgumentOffset() + ArgNo, newVal);
  }

  /// Given a value use iterator, returns the argument that corresponds to it.
  /// Iterator must actually correspond to an argument.
  unsigned getArgumentNo(value_use_iterator<UserTy> I) const {
    assert(getInstruction() && "Not a call or invoke instruction!");
    assert(arg_begin() <= &I.getUse() && &I.getUse() < arg_end()
           && "Argument # out of range!");
    return &I.getUse() - arg_begin();
  }

  /// arg_iterator - The type of iterator to use when looping over actual
  /// arguments at this call site...
  typedef IterTy arg_iterator;

  /// arg_begin/arg_end - Return iterators corresponding to the actual argument
  /// list for a call site.
  IterTy arg_begin() const {
    assert(getInstruction() && "Not a call or invoke instruction!");
    // Skip non-arguments
    return (*this)->op_begin() + getArgumentOffset();
  }

  IterTy arg_end() const { return (*this)->op_end() - getArgumentEndOffset(); }
  bool arg_empty() const { return arg_end() == arg_begin(); }
  unsigned arg_size() const { return unsigned(arg_end() - arg_begin()); }
  
private:
  /// Returns the operand number of the first argument
  unsigned getArgumentOffset() const {
    if (isCall())
      return 1; // Skip Function (ATM)
    else
      return 0; // Args are at the front
  }

  unsigned getArgumentEndOffset() const {
    if (isCall())
      return 0; // Unchanged (ATM)
    else
      return 3; // Skip BB, BB, Function
  }

  IterTy getCallee() const {
      // FIXME: this is slow, since we do not have the fast versions
      // of the op_*() functions here. See CallSite::getCallee.
      //
    if (isCall())
      return getInstruction()->op_begin(); // Unchanged (ATM)
    else
      return getInstruction()->op_end() - 3; // Skip BB, BB, Function
  }
};

/// ImmutableCallSite - establish a view to a call site for examination
class ImmutableCallSite : public CallSiteBase<> {
  typedef CallSiteBase<> _Base;
public:
  ImmutableCallSite(const Value* V) : _Base(V) {}
  ImmutableCallSite(const CallInst *CI) : _Base(CI) {}
  ImmutableCallSite(const InvokeInst *II) : _Base(II) {}
  ImmutableCallSite(const Instruction *II) : _Base(II) {}
};

class CallSite : public CallSiteBase<Function, Value, User, Instruction,
                                     CallInst, InvokeInst, User::op_iterator> {
  typedef CallSiteBase<Function, Value, User, Instruction,
                       CallInst, InvokeInst, User::op_iterator> _Base;
public:
  CallSite() {}
  CallSite(_Base B) : _Base(B) {}
  CallSite(CallInst *CI) : _Base(CI) {}
  CallSite(InvokeInst *II) : _Base(II) {}
  CallSite(Instruction *II) : _Base(II) {}

  bool operator==(const CallSite &CS) const { return I == CS.I; }
  bool operator!=(const CallSite &CS) const { return I != CS.I; }

  /// CallSite::get - This static method is sort of like a constructor.  It will
  /// create an appropriate call site for a Call or Invoke instruction, but it
  /// can also create a null initialized CallSite object for something which is
  /// NOT a call site.
  ///
  static CallSite get(Value *V) {
    return _Base::get(V);
  }

  /// getCallingConv/setCallingConv - get or set the calling convention of the
  /// call.
  CallingConv::ID getCallingConv() const;
  void setCallingConv(CallingConv::ID CC);

  /// getAttributes/setAttributes - get or set the parameter attributes of
  /// the call.
  const AttrListPtr &getAttributes() const;
  void setAttributes(const AttrListPtr &PAL);

  /// paramHasAttr - whether the call or the callee has the given attribute.
  bool paramHasAttr(uint16_t i, Attributes attr) const;

  /// @brief Extract the alignment for a call or parameter (0=unknown).
  uint16_t getParamAlignment(uint16_t i) const;

  /// @brief Return true if the call should not be inlined.
  bool isNoInline() const;
  void setIsNoInline(bool Value = true);
  
  /// @brief Determine if the call does not access memory.
  bool doesNotAccessMemory() const;
  void setDoesNotAccessMemory(bool doesNotAccessMemory = true);

  /// @brief Determine if the call does not access or only reads memory.
  bool onlyReadsMemory() const;
  void setOnlyReadsMemory(bool onlyReadsMemory = true);

  /// @brief Determine if the call cannot return.
  bool doesNotReturn() const;
  void setDoesNotReturn(bool doesNotReturn = true);

  /// @brief Determine if the call cannot unwind.
  bool doesNotThrow() const;
  void setDoesNotThrow(bool doesNotThrow = true);

  /// getType - Return the type of the instruction that generated this call site
  ///
  const Type *getType() const { return (*this)->getType(); }

  /// getCaller - Return the caller function for this call site
  ///
  Function *getCaller() const { return (*this)->getParent()->getParent(); }

  /// hasArgument - Returns true if this CallSite passes the given Value* as an
  /// argument to the called function.
  bool hasArgument(const Value *Arg) const;

  bool operator<(const CallSite &CS) const {
    return getInstruction() < CS.getInstruction();
  }

private:
  User::op_iterator getCallee() const;
};

} // End llvm namespace

#endif
