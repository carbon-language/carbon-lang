//===-- SparcRegInfo.h - Define TargetRegInfo for Sparc ---------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This class implements the virtual class TargetRegInfo for Sparc.
//
//----------------------------------------------------------------------------

#ifndef SPARC_REGINFO_H
#define SPARC_REGINFO_H

namespace llvm {

class SparcRegInfo : public TargetRegInfo {

private:

  // Number of registers used for passing int args (usually 6: %o0 - %o5)
  //
  unsigned const NumOfIntArgRegs;

  // Number of registers used for passing float args (usually 32: %f0 - %f31)
  //
  unsigned const NumOfFloatArgRegs;

  // The following methods are used to color special live ranges (e.g.
  // function args and return values etc.) with specific hardware registers
  // as required. See SparcRegInfo.cpp for the implementation.
  //
  void suggestReg4RetAddr(MachineInstr *RetMI, 
			  LiveRangeInfo &LRI) const;

  void suggestReg4CallAddr(MachineInstr *CallMI, LiveRangeInfo &LRI) const;
  
  // Helper used by the all the getRegType() functions.
  int getRegTypeForClassAndType(unsigned regClassID, const Type* type) const;

public:
  // Type of registers available in Sparc. There can be several reg types
  // in the same class. For instace, the float reg class has Single/Double
  // types
  //
  enum RegTypes {
    IntRegType,
    FPSingleRegType,
    FPDoubleRegType,
    IntCCRegType,
    FloatCCRegType,
    SpecialRegType
  };

  // The actual register classes in the Sparc
  //
  // **** WARNING: If this enum order is changed, also modify 
  // getRegisterClassOfValue method below since it assumes this particular 
  // order for efficiency.
  // 
  enum RegClassIDs { 
    IntRegClassID,                      // Integer
    FloatRegClassID,                    // Float (both single/double)
    IntCCRegClassID,                    // Int Condition Code
    FloatCCRegClassID,                  // Float Condition code
    SpecialRegClassID                   // Special (unallocated) registers
  };

  SparcRegInfo(const SparcTargetMachine &tgt);

  // To find the register class used for a specified Type
  //
  unsigned getRegClassIDOfType(const Type *type,
                               bool isCCReg = false) const;

  // To find the register class to which a specified register belongs
  //
  unsigned getRegClassIDOfRegType(int regType) const;
  
  // getZeroRegNum - returns the register that contains always zero this is the
  // unified register number
  //
  virtual int getZeroRegNum() const;

  // getCallAddressReg - returns the reg used for pushing the address when a
  // function is called. This can be used for other purposes between calls
  //
  unsigned getCallAddressReg() const;

  // Returns the register containing the return address.
  // It should be made sure that this  register contains the return 
  // value when a return instruction is reached.
  //
  unsigned getReturnAddressReg() const;

  // Number of registers used for passing int args (usually 6: %o0 - %o5)
  // and float args (usually 32: %f0 - %f31)
  //
  unsigned const getNumOfIntArgRegs() const   { return NumOfIntArgRegs; }
  unsigned const getNumOfFloatArgRegs() const { return NumOfFloatArgRegs; }
  
  // Compute which register can be used for an argument, if any
  // 
  int regNumForIntArg(bool inCallee, bool isVarArgsCall,
                      unsigned argNo, unsigned& regClassId) const;

  int regNumForFPArg(unsigned RegType, bool inCallee, bool isVarArgsCall,
                     unsigned argNo, unsigned& regClassId) const;
  
  // The following methods are used to color special live ranges (e.g.
  // function args and return values etc.) with specific hardware registers
  // as required. See SparcRegInfo.cpp for the implementation for Sparc.
  //
  void suggestRegs4MethodArgs(const Function *Meth, 
			      LiveRangeInfo& LRI) const;

  void suggestRegs4CallArgs(MachineInstr *CallMI, 
			    LiveRangeInfo& LRI) const; 

  void suggestReg4RetValue(MachineInstr *RetMI, 
                           LiveRangeInfo& LRI) const;
  
  void colorMethodArgs(const Function *Meth,  LiveRangeInfo& LRI,
                       std::vector<MachineInstr*>& InstrnsBefore,
                       std::vector<MachineInstr*>& InstrnsAfter) const;

  // method used for printing a register for debugging purposes
  //
  void printReg(const LiveRange *LR) const;
  
  // returns the # of bytes of stack space allocated for each register
  // type. For Sparc, currently we allocate 8 bytes on stack for all 
  // register types. We can optimize this later if necessary to save stack
  // space (However, should make sure that stack alignment is correct)
  //
  inline int getSpilledRegSize(int RegType) const {
    return 8;
  }


  // To obtain the return value and the indirect call address (if any)
  // contained in a CALL machine instruction
  //
  const Value * getCallInstRetVal(const MachineInstr *CallMI) const;
  const Value * getCallInstIndirectAddrVal(const MachineInstr *CallMI) const;

  // The following methods are used to generate "copy" machine instructions
  // for an architecture.
  //
  // The function regTypeNeedsScratchReg() can be used to check whether a
  // scratch register is needed to copy a register of type `regType' to
  // or from memory.  If so, such a scratch register can be provided by
  // the caller (e.g., if it knows which regsiters are free); otherwise
  // an arbitrary one will be chosen and spilled by the copy instructions.
  //
  bool regTypeNeedsScratchReg(int RegType,
                              int& scratchRegClassId) const;

  void cpReg2RegMI(std::vector<MachineInstr*>& mvec,
                   unsigned SrcReg, unsigned DestReg,
                   int RegType) const;

  void cpReg2MemMI(std::vector<MachineInstr*>& mvec,
                   unsigned SrcReg, unsigned DestPtrReg,
                   int Offset, int RegType, int scratchReg = -1) const;

  void cpMem2RegMI(std::vector<MachineInstr*>& mvec,
                   unsigned SrcPtrReg, int Offset, unsigned DestReg,
                   int RegType, int scratchReg = -1) const;

  void cpValue2Value(Value *Src, Value *Dest,
                     std::vector<MachineInstr*>& mvec) const;

  // Get the register type for a register identified different ways.
  // Note that getRegTypeForLR(LR) != getRegTypeForDataType(LR->getType())!
  // The reg class of a LR depends both on the Value types in it and whether
  // they are CC registers or not (for example).
  int getRegTypeForDataType(const Type* type) const;
  int getRegTypeForLR(const LiveRange *LR) const;
  int getRegType(int unifiedRegNum) const;

  virtual unsigned getFramePointer() const;
  virtual unsigned getStackPointer() const;
};

} // End llvm namespace

#endif
