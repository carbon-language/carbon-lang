/* Title:   PhyRegAlloc.h
   Author:  Ruchira Sasanka
   Date:    Aug 20, 01
   Purpose: This is the main entry point for register allocation.

   Notes:

 * RegisterClasses: Each RegClass accepts a 
   MachineRegClass which contains machine specific info about that register
   class. The code in the RegClass is machine independent and they use
   access functions in the MachineRegClass object passed into it to get
   machine specific info.

 * Machine dependent work: All parts of the register coloring algorithm
   except coloring of an individual node are machine independent.

   Register allocation must be done  as:

     static const MachineRegInfo MRI = MachineRegInfo();  // machine reg info 

     MethodLiveVarInfo LVI(*MethodI );                    // compute LV info
     LVI.analyze();

     PhyRegAlloc PRA(*MethodI, &MRI, &LVI);               // allocate regs
     PRA.allocateRegisters();

   Assumptions: 
     All values in a live range will be of the same physical reg class.

*/ 

#ifndef PHY_REG_ALLOC_H
#define PHY_REG_ALLOC_H

#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/RegClass.h"
#include "llvm/CodeGen/LiveRangeInfo.h"
#include "llvm/Analysis/LiveVar/MethodLiveVarInfo.h"

#include <deque>


//----------------------------------------------------------------------------
// Class AddedInstrns:
// When register allocator inserts new instructions in to the existing 
// instruction stream, it does NOT directly modify the instruction stream.
// Rather, it creates an object of AddedInstrns and stick it in the 
// AddedInstrMap for an existing instruction. This class contains two vectors
// to store such instructions added before and after an existing instruction.
//----------------------------------------------------------------------------

class AddedInstrns
{
 public:
  deque<MachineInstr *> InstrnsBefore;  // Added insts BEFORE an existing inst
  deque<MachineInstr *> InstrnsAfter;   // Added insts AFTER an existing inst

  AddedInstrns() : InstrnsBefore(), InstrnsAfter() { }
};

typedef hash_map<const MachineInstr *, AddedInstrns *> AddedInstrMapType;



//----------------------------------------------------------------------------
// Class RegStackOffsets:
// This class is responsible for managing stack frame of the method for
// register allocation.
//
//----------------------------------------------------------------------------

class RegStackOffsets {

 private:
  int curSpilledVarOff;                 // cur pos of spilled LRs
  int curNewTmpPosOffset;               // cur pos of tmp values on stack
  bool isTmpRegionUsable;               // can we call getNewTmpPosOffFromFP

  const int SizeOfStackItem;            // size of an item on stack
  const int StackSpillStartFromFP;      // start position of spill region
  int StartOfTmpRegion;                 // start of the tmp var region

 public:

  // constructor  

  RegStackOffsets(int SEnSize=8, int StartSpill=176 ) : 
    SizeOfStackItem(SEnSize),  StackSpillStartFromFP(StartSpill) {

    curSpilledVarOff = StartSpill;   
    isTmpRegionUsable = false;
  };


  int getNewSpillOffFromFP() { 
    int tmp =  curSpilledVarOff;     
    curSpilledVarOff += SizeOfStackItem;
    return tmp;   // **TODO: Is sending un-incremented value correct?
  };


  // The following method must be called only after allocating space
  // for spilled LRs and calling setEndOfSpillRegion()
  int getNewTmpPosOffFromFP() { 
    assert( isTmpRegionUsable && "Spill region still open");
    int tmp = curNewTmpPosOffset;
    curNewTmpPosOffset += SizeOfStackItem;
    return tmp; //**TODO: Is sending un-incremented val correct?
  };


  // This method is called when we have allocated space for all spilled
  // LRs. The tmp region can be used only after a call to this method.

  void setEndOfSpillRegion() {
    assert(( ! isTmpRegionUsable) && "setEndOfSpillRegion called again");
    isTmpRegionUsable = true;
    StartOfTmpRegion = curSpilledVarOff; 
  }
  

  // called when temporary values allocated on stack are no longer needed
  void resetTmpPos() { 
    curNewTmpPosOffset = StartOfTmpRegion;
  }
 

};



//----------------------------------------------------------------------------
// class PhyRegAlloc:
// Main class the register allocator. Call allocateRegisters() to allocate
// registers for a Method.
//----------------------------------------------------------------------------


class PhyRegAlloc
{

  vector<RegClass *> RegClassList  ;    // vector of register classes
  const Method *const Meth;             // name of the method we work on
  const TargetMachine &TM;              // target machine
  MethodLiveVarInfo *const LVI;         // LV information for this method 
                                        // (already computed for BBs) 
  LiveRangeInfo LRI;                    // LR info  (will be computed)
  const MachineRegInfo &MRI;            // Machine Register information
  const unsigned NumOfRegClasses;       // recorded here for efficiency

  //vector<const Instruction *> CallInstrList;  // a list of all call instrs
  //vector<const Instruction *> RetInstrList;   // a list of all return instrs

  
  AddedInstrMapType AddedInstrMap;      // to store instrns added in this phase

  RegStackOffsets StackOffsets;

  //vector<const MachineInstr *> PhiInstList;   // a list of all phi instrs

  //------- private methods ---------------------------------------------------

  void addInterference(const Value *const Def, const LiveVarSet *const LVSet, 
		       const bool isCallInst);

  void addInterferencesForArgs();
  void createIGNodeListsAndIGs();
  void buildInterferenceGraphs();
  //void insertCallerSavingCode(const MachineInstr *MInst, 
  //			      const BasicBlock *BB );

  void setCallInterferences(const MachineInstr *MInst, 
			    const LiveVarSet *const LVSetAft );

  void move2DelayedInstr(const MachineInstr *OrigMI, 
			 const MachineInstr *DelayedMI );

  void markUnusableSugColors();
  void allocateStackSpace4SpilledLRs();

  RegStackOffsets & getStackOffsets() {
    return  StackOffsets;
  }


  inline void constructLiveRanges() 
    { LRI.constructLiveRanges(); }      

  void colorIncomingArgs();
  void colorCallRetArgs();
  void updateMachineCode();

  void printLabel(const Value *const Val);
  void printMachineCode();

  friend class UltraSparcRegInfo;


  int getUsableRegAtMI(RegClass *RC,  const int RegType, const MachineInstr *MInst,
		       const LiveVarSet *LVSetBef, MachineInstr *MIBef, 
		       MachineInstr *MIAft );

  int getUnusedRegAtMI(RegClass *RC,  const MachineInstr *MInst, 
		       const LiveVarSet *LVSetBef);

  void setRegsUsedByThisInst(RegClass *RC, const MachineInstr *MInst );
  int getRegNotUsedByThisInst(RegClass *RC, const MachineInstr *MInst);



  void PhyRegAlloc::insertPhiEleminateInstrns();

 public:

  PhyRegAlloc(const Method *const M, const TargetMachine& TM, 
	      MethodLiveVarInfo *const Lvi);

  void allocateRegisters();             // main method called for allocatin

};



/*


What to do:

  * Insert IntCCReg checking code to insertCallerSaving
  * add methods like cpCCReg2Mem & cpMem2CCReg (these will accept an array
  and push back or push_front the instr according to PUSH_BACK, PUSH_FRONT
  flags

*/
  
  









#endif

