//===- TargetCallingConv.td - Target Calling Conventions ---*- tablegen -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines the target-independent interfaces with which targets
// describe their calling conventions.
//
//===----------------------------------------------------------------------===//

class CCAction;
class CallingConv;

/// CCPredicateAction - Instances of this class check some predicate, then
/// delegate to another action if the predicate is true.
class CCPredicateAction<CCAction A> : CCAction {
  CCAction SubAction = A;
}

/// CCMatchType - If the current argument is one of the specified types, apply
/// Action A.
class CCMatchType<list<ValueType> VTs, CCAction A> : CCPredicateAction<A> {
}

/// CCMatchIf - If the predicate matches, apply A.
class CCMatchIf<string predicate, CCAction A> : CCPredicateAction<A> {
  string Predicate = predicate;
}

/// CCMatchIfCC - Match of the current calling convention is 'CC'.
class CCMatchIfCC<string CC, CCAction A> : CCPredicateAction<A> {
  string CallingConv = CC;
}

/// CCAssignToReg - This action matches if there is a register in the specified
/// list that is still available.  If so, it assigns the value to the first
/// available register and succeeds.
class CCAssignToReg<list<Register> regList> : CCAction {
  list<Register> RegList = regList;
}

/// CCAssignToStack - This action always matches: it assigns the value to a
/// stack slot of the specified size and alignment on the stack.
class CCAssignToStack<int size, int align> : CCAction {
  int Size = size;
  int Align = align;
}


/// CCPromoteToType - If applied, this promotes the specified current value to
/// the specified type.
class CCPromoteToType<ValueType destTy> : CCAction {
  ValueType DestTy = destTy;
}

/// CCDelegateTo - This action invokes the specified sub-calling-convention.  It
/// is successful if the specified CC matches.
class CCDelegateTo<CallingConv cc> : CCAction {
  CallingConv CC = cc;
}

/// CallingConv - An instance of this is used to define each calling convention
/// that the target supports.
class CallingConv<list<CCAction> actions> {
  list<CCAction> Actions = actions;
}

