//===-- llvm/CodeGen/MachineInstrAnnot.h ------------------------*- C++ -*-===//
//
//  Annotations used to pass information between code generation phases.
// 
//===----------------------------------------------------------------------===//

#ifndef MACHINE_INSTR_ANNOT_h
#define MACHINE_INSTR_ANNOT_h

#include "llvm/Annotation.h"
#include "llvm/CodeGen/MachineInstr.h"

class Value;
class TmpInstruction;
class CallInst;

class CallArgInfo {
  // Flag values for different argument passing methods
  static const unsigned char IntArgReg = 0x1;
  static const unsigned char FPArgReg  = 0x2;
  static const unsigned char StackSlot = 0x4;
  
  const Value* argVal;                  // this argument
  const Value* argValCopy;              // second copy of arg. when multiple 
                                        // copies must be passed in registers
  unsigned char passingMethod;          // flags recording passing methods
  
public:
  // Constructors
  CallArgInfo(const Value* _argVal)
    : argVal(_argVal), argValCopy(NULL), passingMethod(0x0) {}
  
  CallArgInfo(const CallArgInfo& obj)
    : argVal(obj.argVal), argValCopy(obj.argValCopy),
      passingMethod(obj.passingMethod) {}
  
  // Accessor methods
  const Value*  getArgVal()       { return argVal; }
  const Value*  getArgCopy()      { return argValCopy; }
  bool          usesIntArgReg()   { return (bool) (passingMethod & IntArgReg);} 
  bool          usesFPArgReg()    { return (bool) (passingMethod & FPArgReg); } 
  bool          usesStackSlot()   { return (bool) (passingMethod & StackSlot);} 
  
  // Modifier methods
  void          replaceArgVal(const Value* newVal) { argVal = newVal; }
  void          setUseIntArgReg() { passingMethod |= IntArgReg; }
  void          setUseFPArgReg()  { passingMethod |= FPArgReg; }
  void          setUseStackSlot() { passingMethod |= StackSlot; }
  void          setArgCopy(const Value* tmp) { argValCopy = tmp; }
};


class CallArgsDescriptor: public Annotation { // Annotation for a MachineInstr
  static AnnotationID AID;              // AnnotationID for this class
  std::vector<CallArgInfo> argInfoVec;  // Descriptor for each argument
  const CallInst* callInstr;            // The call instruction == result value
  const Value* funcPtr;                 // Pointer for indirect calls 
  TmpInstruction* retAddrReg;           // Tmp value for return address reg.
  bool isVarArgs;                       // Is this a varargs call?
  bool noPrototype;                     // Is this a call with no prototype?
  
public:
  CallArgsDescriptor(const CallInst* _callInstr, TmpInstruction* _retAddrReg,
                     bool _isVarArgs, bool _noPrototype);
  
  // Accessor methods to retrieve information about the call
  // Note that operands are numbered 1..#CallArgs
  unsigned int    getNumArgs() const          { return argInfoVec.size(); }
  CallArgInfo&    getArgInfo(unsigned int op) { assert(op < argInfoVec.size());
                                                return argInfoVec[op]; }
  const CallInst* getReturnValue() const      { return callInstr; }
  const Value*    getIndirectFuncPtr() const  { return funcPtr; }
  TmpInstruction* getReturnAddrReg() const    { return retAddrReg; }
  bool            isVarArgsFunc() const       { return isVarArgs; }
  bool            hasNoPrototype() const      { return noPrototype; }
  
  // Annotation mechanism to annotate a MachineInstr with the descriptor.
  // This is not demand-driven because annotations can only be created
  // at restricted points during code generation.
  static inline CallArgsDescriptor *get(const MachineInstr* MI) {
    return (CallArgsDescriptor *) MI->getAnnotation(AID);
  }
};


#endif
