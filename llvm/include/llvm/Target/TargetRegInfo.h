//===-- llvm/Target/TargetRegInfo.h - Target Register Info -------*- C++ -*-==//
//
// This file is used to describe the register system of a target to the
// register allocator.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETREGINFO_H
#define LLVM_TARGET_TARGETREGINFO_H

#include "Support/NonCopyable.h"
#include "Support/hash_map"
#include <string>

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

  // This method should find a color which is not used by neighbors
  // (i.e., a false position in IsColorUsedArr) and 
  virtual void colorIGNode(IGNode *Node,
                           std::vector<bool> &IsColorUsedArr) const = 0;
  virtual bool isRegVolatile(int Reg) const = 0;

  TargetRegClassInfo(unsigned ID, unsigned NVR, unsigned NAR)
    : RegClassID(ID), NumOfAvailRegs(NVR), NumOfAllRegs(NAR) {}
};



//---------------------------------------------------------------------------
/// TargetRegInfo - Interface to register info of target machine
///
class TargetRegInfo : public NonCopyableV {
protected:
  // A vector of all machine register classes
  //
  std::vector<const TargetRegClassInfo *> MachineRegClassArr;    
  
public:
  const TargetMachine &target;

  TargetRegInfo(const TargetMachine& tgt) : target(tgt) { }
  ~TargetRegInfo() {
    for (unsigned i = 0, e = MachineRegClassArr.size(); i != e; ++i)
      delete MachineRegClassArr[i];
  }

  // According the definition of a MachineOperand class, a Value in a
  // machine instruction can go into either a normal register or a 
  // condition code register. If isCCReg is true below, the ID of the condition
  // code register class will be returned. Otherwise, the normal register
  // class (eg. int, float) must be returned.
  virtual unsigned getRegClassIDOfType  (const Type *type,
					 bool isCCReg = false) const =0;
  virtual unsigned getRegClassIDOfReg   (int unifiedRegNum)    const =0;
  virtual unsigned getRegClassIDOfRegType(int regType)         const =0;
  
  inline unsigned int getNumOfRegClasses() const { 
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

  virtual void colorCallArgs(MachineInstr *CalI, 
			     LiveRangeInfo& LRI, AddedInstrns *CallAI, 
			     PhyRegAlloc &PRA, const BasicBlock *BB) const = 0;

  virtual void colorRetValue(MachineInstr *RetI, LiveRangeInfo &LRI,
			     AddedInstrns *RetAI) const = 0;



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
  

  // Each register class has a seperate space for register IDs. To convert
  // a regId in a register class to a common Id, or vice versa,
  // we use the folloing methods.
  //
  virtual int getUnifiedRegNum(unsigned regClassID, int reg) const = 0;
  virtual int getClassRegNum(int unifiedRegNum, unsigned& regClassID) const =0;
  
  // Returns the assembly-language name of the specified machine register.
  virtual const char * const getUnifiedRegName(int UnifiedRegNum) const = 0;

  virtual int getRegType(const Type* type) const = 0;
  virtual int getRegType(const LiveRange *LR) const = 0;
  virtual int getRegType(int unifiedRegNum) const = 0;
  
  // The following methods are used to get the frame/stack pointers
  // 
  virtual unsigned getFramePointer() const = 0;
  virtual unsigned getStackPointer() const = 0;

  // A register can be initialized to an invalid number. That number can
  // be obtained using this method.
  //
  virtual int getInvalidRegNum() const = 0;


  // Method for inserting caller saving code. The caller must save all the
  // volatile registers across a call based on the calling conventions of
  // an architecture. This must insert code for saving and restoring 
  // such registers on
  //
  virtual void insertCallerSavingCode(std::vector<MachineInstr*>& instrnsBefore,
                                      std::vector<MachineInstr*>& instrnsAfter,
                                      MachineInstr *MInst, 
				      const BasicBlock *BB, 
				      PhyRegAlloc &PRA) const = 0;

  // This method gives the the number of bytes of stack spaceallocated 
  // to a register when it is spilled to the stack.
  //
  virtual int getSpilledRegSize(int RegType) const = 0;
};

#endif
