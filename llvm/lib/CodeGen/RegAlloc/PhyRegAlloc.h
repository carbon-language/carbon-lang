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

class AddedInstrns
{
 public:
  deque<MachineInstr *> InstrnsBefore;
  deque<MachineInstr *> InstrnsAfter;

  AddedInstrns() : InstrnsBefore(), InstrnsAfter() { }
};

typedef hash_map<const MachineInstr *, AddedInstrns *> AddedInstrMapType;



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



  //------- private methods ---------------------------------------------------

  void addInterference(const Value *const Def, const LiveVarSet *const LVSet, 
		       const bool isCallInst);

  void addInterferencesForArgs();
  void createIGNodeListsAndIGs();
  void buildInterferenceGraphs();
  void insertCallerSavingCode(const MachineInstr *MInst, 
			      const BasicBlock *BB );


  inline void constructLiveRanges() 
    { LRI.constructLiveRanges(); }      

  void colorIncomingArgs();
  void colorCallRetArgs();
  void updateMachineCode();

  void printLabel(const Value *const Val);
  void printMachineCode();
  
 public:
  PhyRegAlloc(const Method *const M, const TargetMachine& TM, 
	      MethodLiveVarInfo *const Lvi);

  void allocateRegisters();             // main method called for allocatin

};








#endif

