//===-- llvm/Target/RegInfo.h - Target Register Information ------*- C++ -*-==//
//
// This file is used to describe the register system of a target to the register
// allocator.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_REGINFO_H
#define LLVM_TARGET_REGINFO_H

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

class IGNode;
class MachineRegClassInfo {
protected:  
  const unsigned RegClassID;        // integer ID of a reg class
  const unsigned NumOfAvailRegs;    // # of avail for coloring -without SP etc.
  const unsigned NumOfAllRegs;      // # of all registers -including SP,g0 etc.

public:
  
  inline unsigned getRegClassID() const { return RegClassID; }
  inline unsigned getNumOfAvailRegs() const { return NumOfAvailRegs; }
  inline unsigned getNumOfAllRegs() const { return NumOfAllRegs; }



  // This method should find a color which is not used by neighbors
  // (i.e., a false position in IsColorUsedArr) and 
  virtual void colorIGNode(IGNode * Node, bool IsColorUsedArr[] ) const = 0;


  MachineRegClassInfo(const unsigned ID, const unsigned NVR, 
		      const unsigned NAR): RegClassID(ID), NumOfAvailRegs(NVR),
                                           NumOfAllRegs(NAR) { }
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
  inline unsigned int getNumOfRegClasses() const { 
    return MachineRegClassArr.size(); 
  }  

  const MachineRegClassInfo *const getMachineRegClass(unsigned i) const { 
    return MachineRegClassArr[i]; 
  }


  virtual unsigned getRegClassIDOfValue (const Value *const Val) const = 0;

  virtual void colorArgs(const Method *const Meth, 
			 LiveRangeInfo & LRI) const = 0;

  virtual void colorCallArgs(vector<const Instruction *> & CallInstrList, 
			     LiveRangeInfo& LRI, 
			     AddedInstrMapType& AddedInstrMap ) const = 0;

  virtual int getUnifiedRegNum(int RegClassID, int reg) const = 0;

  virtual const string getUnifiedRegName(int reg) const = 0;

  //virtual void printReg(const LiveRange *const LR) const =0;

  MachineRegInfo() { }

};

#endif
