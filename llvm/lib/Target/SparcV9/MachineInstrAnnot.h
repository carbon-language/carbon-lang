//===-- llvm/CodeGen/MachineInstrAnnot.h ------------------------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
//  Annotations used to pass information between code generation phases.
// 
//===----------------------------------------------------------------------===//

#ifndef MACHINE_INSTR_ANNOT_h
#define MACHINE_INSTR_ANNOT_h

#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/Target/TargetRegInfo.h"

namespace llvm {

class Value;
class TmpInstruction;
class CallInst;

class CallArgInfo {
  // Flag values for different argument passing methods
  static const unsigned char IntArgReg = 0x1;
  static const unsigned char FPArgReg  = 0x2;
  static const unsigned char StackSlot = 0x4;
  
  Value*        argVal;         // this argument
  int           argCopyReg;     // register used for second copy of arg. when
                                // multiple  copies must be passed in registers
  unsigned char passingMethod;  // flags recording passing methods
  
public:
  // Constructors
  CallArgInfo(Value* _argVal)
    : argVal(_argVal), argCopyReg(TargetRegInfo::getInvalidRegNum()),
      passingMethod(0x0) {}
  
  CallArgInfo(const CallArgInfo& obj)
    : argVal(obj.argVal), argCopyReg(obj.argCopyReg),
      passingMethod(obj.passingMethod) {}
  
  // Accessor methods
  Value*        getArgVal()       { return argVal; }
  int           getArgCopy()      { return argCopyReg; }
  bool          usesIntArgReg()   { return (bool) (passingMethod & IntArgReg);} 
  bool          usesFPArgReg()    { return (bool) (passingMethod & FPArgReg); } 
  bool          usesStackSlot()   { return (bool) (passingMethod & StackSlot);} 
  
  // Modifier methods
  void          replaceArgVal(Value* newVal) { argVal = newVal; }
  void          setUseIntArgReg() { passingMethod |= IntArgReg; }
  void          setUseFPArgReg()  { passingMethod |= FPArgReg; }
  void          setUseStackSlot() { passingMethod |= StackSlot; }
  void          setArgCopy(int copyReg) { argCopyReg = copyReg; }
};


class CallArgsDescriptor {

  std::vector<CallArgInfo> argInfoVec;  // Descriptor for each argument
  CallInst* callInstr;                  // The call instruction == result value
  Value* funcPtr;                       // Pointer for indirect calls 
  TmpInstruction* retAddrReg;           // Tmp value for return address reg.
  bool isVarArgs;                       // Is this a varargs call?
  bool noPrototype;                     // Is this a call with no prototype?
  
public:
  CallArgsDescriptor(CallInst* _callInstr, TmpInstruction* _retAddrReg,
                     bool _isVarArgs, bool _noPrototype);
  
  // Accessor methods to retrieve information about the call
  // Note that operands are numbered 1..#CallArgs
  unsigned int    getNumArgs() const          { return argInfoVec.size(); }
  CallArgInfo&    getArgInfo(unsigned int op) { assert(op < argInfoVec.size());
                                                return argInfoVec[op]; }
  CallInst*       getCallInst() const         { return callInstr; }
  CallInst*       getReturnValue() const;
  Value*          getIndirectFuncPtr() const  { return funcPtr; }
  TmpInstruction* getReturnAddrReg() const    { return retAddrReg; }
  bool            isVarArgsFunc() const       { return isVarArgs; }
  bool            hasNoPrototype() const      { return noPrototype; }

  // Mechanism to get the descriptor for a CALL MachineInstr.
  // 
  static CallArgsDescriptor *get(const MachineInstr* MI);
};

} // End llvm namespace

#endif
