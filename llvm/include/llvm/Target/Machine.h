//===-- llvm/Target/Machine.h - General Target Information -------*- C++ -*-==//
//
// This file describes the general parts of a Target machine.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_MACHINE_H
#define LLVM_TARGET_MACHINE_H

#include "llvm/Target/Data.h"
#include "llvm/Support/NonCopyable.h"
#include "llvm/Support/DataTypes.h"
#include <string>
#include <hash_map>
#include <hash_set>
#include <algorithm>

class StructType;
struct MachineInstrDescriptor;
class TargetMachine;
class MachineInstrInfo;

//---------------------------------------------------------------------------
// Data types used to define information about a single machine instruction
//---------------------------------------------------------------------------

typedef int MachineOpCode;
typedef int OpCodeMask;

static const unsigned MAX_OPCODE_SIZE = 16;

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

class LiveRangeInfo;
class Method;
class Instruction;
class LiveRange;
class AddedInstrns;
class MachineInstr;
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





//---------------------------------------------------------------------------
// class TargetMachine
// 
// Purpose:
//   Primary interface to machine description for the target machine.
// 
//---------------------------------------------------------------------------

class TargetMachine : public NonCopyableV {
public:
  const string     TargetName;
  const TargetData DataLayout;		// Calculates type size & alignment
  int              optSizeForSubWordData;
  int	           minMemOpWordSize;
  int	           maxAtomicMemOpWordSize;
  
  // Register information.  This needs to be reorganized into a single class.
  int		zeroRegNum;	// register that gives 0 if any (-1 if none)
  
public:
  TargetMachine(const string &targetname,
		unsigned char PtrSize = 8, unsigned char PtrAl = 8,
		unsigned char DoubleAl = 8, unsigned char FloatAl = 4,
		unsigned char LongAl = 8, unsigned char IntAl = 4,
		unsigned char ShortAl = 2, unsigned char ByteAl = 1)
    : TargetName(targetname), DataLayout(targetname, PtrSize, PtrAl,
					 DoubleAl, FloatAl, LongAl, IntAl, 
					 ShortAl, ByteAl) { }
  virtual ~TargetMachine() {}
  
  virtual const MachineInstrInfo& getInstrInfo() const = 0;
  
  virtual unsigned int	findOptimalStorageSize	(const Type* ty) const;
  
  // This really should be in the register info class
  virtual bool		regsMayBeAliased	(unsigned int regNum1,
						 unsigned int regNum2) const {
    return (regNum1 == regNum2);
  }
  
  // compileMethod - This does everything neccesary to compile a method into the
  // built in representation.  This allows the target to have complete control
  // over how it does compilation.  This does not emit assembly or output
  // machine code however, this is done later.
  //
  virtual bool compileMethod(Method *M) = 0;

  // emitAssembly - Output assembly language code (a .s file) for the specified
  // method. The specified method must have been compiled before this may be
  // used.
  //
  virtual void emitAssembly(Method *M, ostream &OutStr) {  /* todo */ }
};

#endif
