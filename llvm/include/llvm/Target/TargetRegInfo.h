//===-- llvm/Target/RegInfo.h - Target Register Information ------*- C++ -*-==//
//
// This file is used to describe the register system of a target to the
// register allocator.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_MACHINEREGINFO_H
#define LLVM_TARGET_MACHINEREGINFO_H

#include "llvm/Support/NonCopyable.h"
#include <hash_map>
#include <string>

class IGNode;
class Value;
class LiveRangeInfo;
class Method;
class Instruction;
class LiveRange;
class AddedInstrns;
class MachineInstr;


//-----------------------------------------------------------------------------
// class MachineRegClassInfo
// 
// Purpose:
//   Interface to description of machine register class (e.g., int reg class
//   float reg class etc)
// 
//--------------------------------------------------------------------------


class MachineRegClassInfo {

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
  virtual void colorIGNode(IGNode * Node, bool IsColorUsedArr[] ) const = 0;


  MachineRegClassInfo(const unsigned ID, const unsigned NVR, 
		      const unsigned NAR): RegClassID(ID), NumOfAvailRegs(NVR),
					     NumOfAllRegs(NAR)
  { }                         // empty constructor

};



//---------------------------------------------------------------------------
// class MachineRegInfo
// 
// Purpose:
//   Interface to register info of target machine
// 
//--------------------------------------------------------------------------



typedef hash_map<const MachineInstr *, AddedInstrns *> AddedInstrMapType;

// A vector of all machine register classes
typedef vector<const MachineRegClassInfo *> MachineRegClassArrayType;


class MachineRegInfo : public NonCopyableV {

protected:

  MachineRegClassArrayType MachineRegClassArr;    

  
public:


  // According the definition of a MachineOperand class, a Value in a
  // machine instruction can go into either a normal register or a 
  // condition code register. If isCCReg is true below, the ID of the condition
  // code regiter class will be returned. Otherwise, the normal register
  // class (eg. int, float) must be returned.
  virtual unsigned getRegClassIDOfValue (const Value *const Val,
					 bool isCCReg = false) const =0;


  inline unsigned int getNumOfRegClasses() const { 
    return MachineRegClassArr.size(); 
  }  

  const MachineRegClassInfo *const getMachineRegClass(unsigned i) const { 
    return MachineRegClassArr[i]; 
  }

  // returns the register that is hardwired to zero if any (-1 if none)
  virtual inline int      getZeroRegNum()     const = 0;

  //virtual unsigned getRegClassIDOfValue (const Value *const Val) const = 0;
  // this method must give the exact register class of a machine operand
  // e.g, Int, Float, Int CC, Float CC 
  //virtual unsigned getRCIDOfMachineOp (const MachineOperand &MO) const = 0;


  virtual void colorArgs(const Method *const Meth, 
			 LiveRangeInfo & LRI) const = 0;

  virtual void colorCallArgs(vector<const Instruction *> & CallInstrList, 
			     LiveRangeInfo& LRI, 
			     AddedInstrMapType& AddedInstrMap ) const = 0 ;

  virtual void colorRetArg(vector<const Instruction *> & 
			   RetInstrList, LiveRangeInfo& LRI,
			   AddedInstrMapType &AddedInstrMap) const =0;

  // returns the reg used for pushing the address when a method is called.
  // This can be used for other purposes between calls
  virtual unsigned getCallAddressReg() const = 0;

  // and when we return from a method. It should be made sure that this 
  // register contains the return value when a return instruction is reached.
  virtual unsigned getReturnAddressReg() const = 0; 
  
  virtual int getUnifiedRegNum(int RegClassID, int reg) const = 0;

  virtual const string getUnifiedRegName(int UnifiedRegNum) const = 0;

  //virtual void printReg(const LiveRange *const LR) const =0;

  MachineRegInfo() { }

};




#endif



#if 0

//---------------------------------------------------------------------------
// class MachineRegInfo
// 
// Purpose:
//   Interface to register info of target machine
// 
//--------------------------------------------------------------------------



typedef hash_map<const MachineInstr *, AddedInstrns *> AddedInstrMapType;

// A vector of all machine register classes
typedef vector<const MachineRegClassInfo *> MachineRegClassArrayType;


class MachineRegInfo : public NonCopyableV {

protected:

  MachineRegClassArrayType MachineRegClassArr;


public:

 MachineRegInfo() {}
  
  // According the definition of a MachineOperand class, a Value in a
  // machine instruction can go into either a normal register or a 
  // condition code register. If isCCReg is true below, the ID of the condition
  // code regiter class will be returned. Otherwise, the normal register
  // class (eg. int, float) must be returned.
  virtual unsigned getRegClassIDOfValue (const Value *const Val,
					 bool isCCReg = false) const =0;


  // returns the register that is hardwired to zero if any (-1 if none)
  virtual inline int      getZeroRegNum()     const = 0;
  
  inline unsigned int getNumOfRegClasses() const { 
    return MachineRegClassArr.size(); 
  }  

  const MachineRegClassInfo *const getMachineRegClass(unsigned i) const { 
    return MachineRegClassArr[i]; 
  }



  //virtual unsigned getRegClassIDOfValue (const Value *const Val) const = 0;
  // this method must give the exact register class of a machine operand
  // e.g, Int, Float, Int CC, Float CC 
  //virtual unsigned getRCIDOfMachineOp (const MachineOperand &MO) const = 0;


  virtual void colorArgs(const Method *const Meth, 
			 LiveRangeInfo & LRI) const = 0;

  virtual void colorCallArgs(vector<const Instruction *> & CallInstrList, 
			     LiveRangeInfo& LRI, 
			     AddedInstrMapType& AddedInstrMap ) const = 0 ;

  virtual int getUnifiedRegNum(int RegClassID, int reg) const = 0;

  virtual const string getUnifiedRegName(int UnifiedRegNum) const = 0;

  //virtual void printReg(const LiveRange *const LR) const =0;
};


#endif






