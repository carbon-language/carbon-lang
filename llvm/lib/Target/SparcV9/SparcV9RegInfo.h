//===-- SparcV9RegInfo.h - SparcV9 Target Register Info ---------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file is used to describe the register file of the SparcV9 target to
// its register allocator.
//
//===----------------------------------------------------------------------===//

#ifndef SPARCV9REGINFO_H
#define SPARCV9REGINFO_H

#include "Support/hash_map"
#include <string>
#include <cassert>

namespace llvm {

class TargetMachine;
class IGNode;
class Type;
class Value;
class LiveRangeInfo;
class Function;
class LiveRange;
class AddedInstrns;
class MachineInstr;
class BasicBlock;
class SparcV9TargetMachine;


///----------------------------------------------------------------------------
///   Interface to description of machine register class (e.g., int reg class
///   float reg class etc)
///
class TargetRegClassInfo {
protected:
  const unsigned RegClassID;        // integer ID of a reg class
  const unsigned NumOfAvailRegs;    // # of avail for coloring -without SP etc.
  const unsigned NumOfAllRegs;      // # of all registers -including SP,g0 etc.
  
public:
  inline unsigned getRegClassID()     const { return RegClassID; }
  inline unsigned getNumOfAvailRegs() const { return NumOfAvailRegs; }
  inline unsigned getNumOfAllRegs()   const { return NumOfAllRegs; }

  // This method marks the registers used for a given register number.
  // This defaults to marking a single register but may mark multiple
  // registers when a single number denotes paired registers.
  // 
  void markColorsUsed(unsigned RegInClass,
                              int UserRegType,
                              int RegTypeWanted,
                              std::vector<bool> &IsColorUsedArr) const {
    assert(RegInClass < NumOfAllRegs && RegInClass < IsColorUsedArr.size());
    assert(UserRegType == RegTypeWanted &&
       "Default method is probably incorrect for class with multiple types.");
    IsColorUsedArr[RegInClass] = true;
  }

  // This method finds unused registers of the specified register type,
  // using the given "used" flag array IsColorUsedArr.  It defaults to
  // checking a single entry in the array directly, but that can be overridden
  // for paired registers and other such silliness.
  // It returns -1 if no unused color is found.
  // 
  int findUnusedColor(int RegTypeWanted,
                          const std::vector<bool> &IsColorUsedArr) const {
    // find first unused color in the IsColorUsedArr directly
    unsigned NC = this->getNumOfAvailRegs();
    assert(IsColorUsedArr.size() >= NC && "Invalid colors-used array");
    for (unsigned c = 0; c < NC; c++)
      if (!IsColorUsedArr[c])
        return c;
    return -1;
  }

  // This method should find a color which is not used by neighbors
  // (i.e., a false position in IsColorUsedArr) and 
  void colorIGNode(IGNode *Node,
                           const std::vector<bool> &IsColorUsedArr) const;

  // Check whether a specific register is volatile, i.e., whether it is not
  // preserved across calls
  bool isRegVolatile(int Reg) const;

  // Check whether a specific register is modified as a side-effect of the
  // call instruction itself,
  bool modifiedByCall(int Reg) const {return false; }

  virtual const char* const getRegName(unsigned reg) const;

  TargetRegClassInfo(unsigned ID, unsigned NVR, unsigned NAR)
    : RegClassID(ID), NumOfAvailRegs(NVR), NumOfAllRegs(NAR) {}
};

/// SparcV9RegInfo - Interface to register info of SparcV9 target machine
///
class SparcV9RegInfo {
  SparcV9RegInfo(const SparcV9RegInfo &);  // DO NOT IMPLEMENT
  void operator=(const SparcV9RegInfo &); // DO NOT IMPLEMENT
protected:
  // A vector of all machine register classes
  //
  std::vector<const TargetRegClassInfo *> MachineRegClassArr;    
  
public:
  const TargetMachine &target;

  // A register can be initialized to an invalid number. That number can
  // be obtained using this method.
  //
  static int getInvalidRegNum() { return -1; }


  // According the definition of a MachineOperand class, a Value in a
  // machine instruction can go into either a normal register or a 
  // condition code register. If isCCReg is true below, the ID of the condition
  // code register class will be returned. Otherwise, the normal register
  // class (eg. int, float) must be returned.

  // To find the register class used for a specified Type
  //
  unsigned getRegClassIDOfType  (const Type *type,
					 bool isCCReg = false) const;

  // To find the register class to which a specified register belongs
  //
  unsigned getRegClassIDOfRegType(int regType) const;

  unsigned getRegClassIDOfReg(int unifiedRegNum) const {
    unsigned classId = 0;
    (void) getClassRegNum(unifiedRegNum, classId);
    return classId;
  }

  unsigned int getNumOfRegClasses() const { 
    return MachineRegClassArr.size(); 
  }  

  const TargetRegClassInfo *getMachineRegClass(unsigned i) const { 
    return MachineRegClassArr[i]; 
  }

  // getZeroRegNum - returns the register that is hardwired to always contain
  // zero, if any (-1 if none). This is the unified register number.
  //
  unsigned getZeroRegNum() const;

  // The following methods are used to color special live ranges (e.g.
  // method args and return values etc.) with specific hardware registers
  // as required. See SparcRegInfo.cpp for the implementation for Sparc.
  //
  void suggestRegs4MethodArgs(const Function *Func, 
                                      LiveRangeInfo& LRI) const;

  void suggestRegs4CallArgs(MachineInstr *CallI, 
                                    LiveRangeInfo& LRI) const;

  void suggestReg4RetValue(MachineInstr *RetI, 
				   LiveRangeInfo& LRI) const;

  void colorMethodArgs(const Function *Func,
                           LiveRangeInfo &LRI,
                           std::vector<MachineInstr*>& InstrnsBefore,
                           std::vector<MachineInstr*>& InstrnsAfter) const;


  // Check whether a specific register is volatile, i.e., whether it is not
  // preserved across calls
  inline bool isRegVolatile(int RegClassID, int Reg) const {
    return MachineRegClassArr[RegClassID]->isRegVolatile(Reg);
  }

  // Check whether a specific register is modified as a side-effect of the
  // call instruction itself,
  inline bool modifiedByCall(int RegClassID, int Reg) const {
    return MachineRegClassArr[RegClassID]->modifiedByCall(Reg);
  }
  
  // getCallAddressReg - Returns the reg used for pushing the address
  // when a method is called. This can be used for other purposes
  // between calls
  //
  unsigned getCallAddressReg() const;

  // Each register class has a separate space for register IDs. To convert
  // a regId in a register class to a common Id, or vice versa,
  // we use the folloing two methods.
  //
  // This method converts from class reg. number to unified register number.
  int getUnifiedRegNum(unsigned regClassID, int reg) const {
    if (reg == getInvalidRegNum()) { return getInvalidRegNum(); }
    assert(regClassID < getNumOfRegClasses() && "Invalid register class");
    int totalRegs = 0;
    for (unsigned rcid = 0; rcid < regClassID; ++rcid)
      totalRegs += MachineRegClassArr[rcid]->getNumOfAllRegs();
    return reg + totalRegs;
  }

  // This method converts the unified number to the number in its class,
  // and returns the class ID in regClassID.
  int getClassRegNum(int uRegNum, unsigned& regClassID) const {
    if (uRegNum == getInvalidRegNum()) { return getInvalidRegNum(); }
    
    int totalRegs = 0, rcid = 0, NC = getNumOfRegClasses();  
    while (rcid < NC &&
           uRegNum>= totalRegs+(int)MachineRegClassArr[rcid]->getNumOfAllRegs())
    {
      totalRegs += MachineRegClassArr[rcid]->getNumOfAllRegs();
      rcid++;
    }
    if (rcid == NC) {
      assert(0 && "getClassRegNum(): Invalid register number");
      return getInvalidRegNum();
    }
    regClassID = rcid;
    return uRegNum - totalRegs;
  }
  
  // Returns the assembly-language name of the specified machine register.
  // 
  const char * const getUnifiedRegName(int UnifiedRegNum) const {
    unsigned regClassID = getNumOfRegClasses(); // initialize to invalid value
    int regNumInClass = getClassRegNum(UnifiedRegNum, regClassID);
    return MachineRegClassArr[regClassID]->getRegName(regNumInClass);
  }

  // This method gives the the number of bytes of stack space allocated 
  // to a register when it is spilled to the stack, according to its
  // register type.
  //
  // For SparcV9, currently we allocate 8 bytes on stack for all 
  // register types. We can optimize this later if necessary to save stack
  // space (However, should make sure that stack alignment is correct)
  //
  int getSpilledRegSize(int RegType) const {
    return 8;
  }

private:
  // Number of registers used for passing int args (usually 6: %o0 - %o5)
  //
  unsigned const NumOfIntArgRegs;

  // Number of registers used for passing float args (usually 32: %f0 - %f31)
  //
  unsigned const NumOfFloatArgRegs;

  // The following methods are used to color special live ranges (e.g.
  // function args and return values etc.) with specific hardware registers
  // as required. See SparcV9RegInfo.cpp for the implementation.
  //
  void suggestReg4RetAddr(MachineInstr *RetMI, 
			  LiveRangeInfo &LRI) const;

  void suggestReg4CallAddr(MachineInstr *CallMI, LiveRangeInfo &LRI) const;
  
  // Helper used by the all the getRegType() functions.
  int getRegTypeForClassAndType(unsigned regClassID, const Type* type) const;

public:
  // Type of registers available in SparcV9. There can be several reg types
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

  // The actual register classes in the SparcV9
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

  SparcV9RegInfo(const SparcV9TargetMachine &tgt);

  ~SparcV9RegInfo() {
    for (unsigned i = 0, e = MachineRegClassArr.size(); i != e; ++i)
      delete MachineRegClassArr[i];
  }

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
  

  // method used for printing a register for debugging purposes
  //
  void printReg(const LiveRange *LR) const;

  // To obtain the return value and the indirect call address (if any)
  // contained in a CALL machine instruction
  //
  const Value * getCallInstRetVal(const MachineInstr *CallMI) const;
  const Value * getCallInstIndirectAddrVal(const MachineInstr *CallMI) const;

  // The following methods are used to generate "copy" machine instructions
  // for an architecture. Currently they are used in TargetRegClass 
  // interface. However, they can be moved to TargetInstrInfo interface if
  // necessary.
  //
  // The function regTypeNeedsScratchReg() can be used to check whether a
  // scratch register is needed to copy a register of type `regType' to
  // or from memory.  If so, such a scratch register can be provided by
  // the caller (e.g., if it knows which regsiters are free); otherwise
  // an arbitrary one will be chosen and spilled by the copy instructions.
  // If a scratch reg is needed, the reg. type that must be used
  // for scratch registers is returned in scratchRegType.
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

  unsigned getFramePointer() const;
  unsigned getStackPointer() const;
};

} // End llvm namespace

#endif // SPARCV9REGINFO_H
