//===-- llvm/Target/TargetRegInfo.h - Target Register Info -------*- C++ -*-==//
//
// This file is used to describe the register system of a target to the
// register allocator.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETREGINFO_H
#define LLVM_TARGET_TARGETREGINFO_H

#include "Support/hash_map"
#include <string>
#include <cassert>

class TargetMachine;
class IGNode;
class Type;
class Value;
class LiveRangeInfo;
class Function;
class LiveRange;
class AddedInstrns;
class MachineInstr;
class PhyRegAlloc;
class BasicBlock;

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
  virtual void markColorsUsed(unsigned RegInClass,
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
  virtual int findUnusedColor(int RegTypeWanted,
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
  virtual void colorIGNode(IGNode *Node,
                           const std::vector<bool> &IsColorUsedArr) const = 0;

  virtual bool isRegVolatile(int Reg) const = 0;

  // If any specific register needs extra information
  virtual bool modifiedByCall(int Reg) const {return false; }

  virtual const char* const getRegName(unsigned reg) const = 0;

  TargetRegClassInfo(unsigned ID, unsigned NVR, unsigned NAR)
    : RegClassID(ID), NumOfAvailRegs(NVR), NumOfAllRegs(NAR) {}
};



//---------------------------------------------------------------------------
/// TargetRegInfo - Interface to register info of target machine
///
class TargetRegInfo {
  TargetRegInfo(const TargetRegInfo &);  // DO NOT IMPLEMENT
  void operator=(const TargetRegInfo &); // DO NOT IMPLEMENT
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

  TargetRegInfo(const TargetMachine& tgt) : target(tgt) { }
  virtual ~TargetRegInfo() {
    for (unsigned i = 0, e = MachineRegClassArr.size(); i != e; ++i)
      delete MachineRegClassArr[i];
  }

  // According the definition of a MachineOperand class, a Value in a
  // machine instruction can go into either a normal register or a 
  // condition code register. If isCCReg is true below, the ID of the condition
  // code register class will be returned. Otherwise, the normal register
  // class (eg. int, float) must be returned.
  virtual unsigned getRegClassIDOfType  (const Type *type,
					 bool isCCReg = false) const = 0;
  virtual unsigned getRegClassIDOfRegType(int regType) const = 0;

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

  // returns the register that is hardwired to zero if any (-1 if none)
  //
  virtual int getZeroRegNum() const = 0;

  // Number of registers used for passing int args (usually 6: %o0 - %o5)
  // and float args (usually 32: %f0 - %f31)
  //
  virtual unsigned const getNumOfIntArgRegs() const   = 0;
  virtual unsigned const getNumOfFloatArgRegs() const = 0;

  // The following methods are used to color special live ranges (e.g.
  // method args and return values etc.) with specific hardware registers
  // as required. See SparcRegInfo.cpp for the implementation for Sparc.
  //
  virtual void suggestRegs4MethodArgs(const Function *Func, 
			 LiveRangeInfo &LRI) const = 0;

  virtual void suggestRegs4CallArgs(MachineInstr *CallI, 
                                    LiveRangeInfo &LRI) const = 0;

  virtual void suggestReg4RetValue(MachineInstr *RetI, 
				   LiveRangeInfo &LRI) const = 0;

  virtual void colorMethodArgs(const Function *Func,  LiveRangeInfo &LRI,
                               AddedInstrns *FirstAI) const = 0;

  // Method for inserting caller saving code. The caller must save all the
  // volatile registers across a call based on the calling conventions of
  // an architecture. This must insert code for saving and restoring 
  // such registers on
  //
  virtual void insertCallerSavingCode(std::vector<MachineInstr*>& instrnsBefore,
                                      std::vector<MachineInstr*>& instrnsAfter,
                                      MachineInstr *CallMI,
				      const BasicBlock *BB, 
				      PhyRegAlloc &PRA) const = 0;

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
  virtual bool regTypeNeedsScratchReg(int RegType,
                                      int& scratchRegType) const = 0;
  
  virtual void cpReg2RegMI(std::vector<MachineInstr*>& mvec,
                           unsigned SrcReg, unsigned DestReg,
                           int RegType) const = 0;
  
  virtual void cpReg2MemMI(std::vector<MachineInstr*>& mvec,
                           unsigned SrcReg, unsigned DestPtrReg, int Offset,
                           int RegType, int scratchReg = -1) const=0;

  virtual void cpMem2RegMI(std::vector<MachineInstr*>& mvec,
                           unsigned SrcPtrReg, int Offset, unsigned DestReg,
                           int RegType, int scratchReg = -1) const=0;
  
  virtual void cpValue2Value(Value *Src, Value *Dest,
                             std::vector<MachineInstr*>& mvec) const = 0;

  virtual bool isRegVolatile(int RegClassID, int Reg) const = 0;
  
  // Returns the reg used for pushing the address when a method is called.
  // This can be used for other purposes between calls
  //
  virtual unsigned getCallAddressReg() const = 0;

  // Returns the register containing the return address.
  //It should be made sure that this 
  // register contains the return value when a return instruction is reached.
  //
  virtual unsigned getReturnAddressReg() const = 0; 
  

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

  // Get the register type for a register identified different ways.
  // Note that getRegTypeForLR(LR) != getRegTypeForDataType(LR->getType())!
  // The reg class of a LR depends both on the Value types in it and whether
  // they are CC registers or not (for example).
  virtual int getRegTypeForDataType(const Type* type) const = 0;
  virtual int getRegTypeForLR(const LiveRange *LR) const = 0;
  virtual int getRegType(int unifiedRegNum) const = 0;
  
  // The following methods are used to get the frame/stack pointers
  // 
  virtual unsigned getFramePointer() const = 0;
  virtual unsigned getStackPointer() const = 0;

  // This method gives the the number of bytes of stack spaceallocated 
  // to a register when it is spilled to the stack.
  //
  virtual int getSpilledRegSize(int RegType) const = 0;
};

#endif
