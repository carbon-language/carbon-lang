/* Title:   PhyRegAlloc.h   -*- C++ -*-
   Author:  Ruchira Sasanka
   Date:    Aug 20, 01
   Purpose: This is the main entry point for register allocation.

   Notes:
   =====

 * RegisterClasses: Each RegClass accepts a 
   TargetRegClass which contains machine specific info about that register
   class. The code in the RegClass is machine independent and they use
   access functions in the TargetRegClass object passed into it to get
   machine specific info.

 * Machine dependent work: All parts of the register coloring algorithm
   except coloring of an individual node are machine independent.
*/ 

#ifndef PHY_REG_ALLOC_H
#define PHY_REG_ALLOC_H

#include "LiveRangeInfo.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include <map>

class MachineFunction;
class TargetRegInfo;
class FunctionLiveVarInfo;
class MachineInstr;
class LoopInfo;
class RegClass;

//----------------------------------------------------------------------------
// Class AddedInstrns:
// When register allocator inserts new instructions in to the existing 
// instruction stream, it does NOT directly modify the instruction stream.
// Rather, it creates an object of AddedInstrns and stick it in the 
// AddedInstrMap for an existing instruction. This class contains two vectors
// to store such instructions added before and after an existing instruction.
//----------------------------------------------------------------------------

struct AddedInstrns {
  std::vector<MachineInstr*> InstrnsBefore;//Insts added BEFORE an existing inst
  std::vector<MachineInstr*> InstrnsAfter; //Insts added AFTER an existing inst
};

//----------------------------------------------------------------------------
// class PhyRegAlloc:
// Main class the register allocator. Call allocateRegisters() to allocate
// registers for a Function.
//----------------------------------------------------------------------------

class PhyRegAlloc {
  std::vector<RegClass *> RegClassList; // vector of register classes
  const TargetMachine &TM;              // target machine
  const Function *Fn;                   // name of the function we work on
  MachineFunction &MF;                  // descriptor for method's native code
  FunctionLiveVarInfo *const LVI;       // LV information for this method 
                                        // (already computed for BBs) 
  LiveRangeInfo LRI;                    // LR info  (will be computed)
  const TargetRegInfo &MRI;             // Machine Register information
  const unsigned NumOfRegClasses;       // recorded here for efficiency

  // Map to indicate whether operands of each MachineInstr have been updated
  // according to their assigned colors.  This is primarily for debugging and
  // could be removed in the long run.
  std::map<const MachineInstr *, bool> OperandsColoredMap;
  
  // AddedInstrMap - Used to store instrns added in this phase
  std::map<const MachineInstr *, AddedInstrns> AddedInstrMap;

  // ScratchRegsUsed - Contains scratch register uses for a particular MI.
  typedef std::multimap<const MachineInstr*, int> ScratchRegsUsedTy;
  ScratchRegsUsedTy ScratchRegsUsed;

  AddedInstrns AddedInstrAtEntry;       // to store instrns added at entry
  LoopInfo *LoopDepthCalc;              // to calculate loop depths 

  PhyRegAlloc(const PhyRegAlloc&);     // DO NOT IMPLEMENT
  void operator=(const PhyRegAlloc&);  // DO NOT IMPLEMENT
public:
  PhyRegAlloc(Function *F, const TargetMachine& TM, FunctionLiveVarInfo *Lvi,
              LoopInfo *LoopDepthCalc);
  ~PhyRegAlloc();

  // main method called for allocating registers
  //
  void allocateRegisters();           

  // access to register classes by class ID
  // 
  const RegClass* getRegClassByID(unsigned id) const {
    return RegClassList[id];
  }
  RegClass* getRegClassByID(unsigned id) {
    return RegClassList[id];
  }

private:
  void addInterference(const Value *Def, const ValueSet *LVSet, 
		       bool isCallInst);

  void addInterferencesForArgs();
  void createIGNodeListsAndIGs();
  void buildInterferenceGraphs();

  void setCallInterferences(const MachineInstr *MI, 
			    const ValueSet *LVSetAft);

  void move2DelayedInstr(const MachineInstr *OrigMI, 
			 const MachineInstr *DelayedMI);

  void markUnusableSugColors();
  void allocateStackSpace4SpilledLRs();

  void insertCode4SpilledLR(const LiveRange *LR, 
                            MachineBasicBlock::iterator& MII,
                            MachineBasicBlock &MBB, unsigned OpNum);

  // Method for inserting caller saving code. The caller must save all the
  // volatile registers live across a call.
  void insertCallerSavingCode(std::vector<MachineInstr*>& instrnsBefore,
                              std::vector<MachineInstr*>& instrnsAfter,
                              MachineInstr *CallMI,
                              const BasicBlock *BB);

  inline void constructLiveRanges() { LRI.constructLiveRanges(); }      

  void colorIncomingArgs();
  void colorCallRetArgs();
  void updateMachineCode();
  void updateInstruction(MachineBasicBlock::iterator& MII,
                         MachineBasicBlock &MBB);

  void printLabel(const Value *Val);
  void printMachineCode();


  int getUsableUniRegAtMI(int RegType, const ValueSet *LVSetBef,
			  MachineInstr *MI,
                          std::vector<MachineInstr*>& MIBef,
                          std::vector<MachineInstr*>& MIAft);
  
  // Callback method used to find unused registers. 
  // LVSetBef is the live variable set to search for an unused register.
  // If it is not specified, the LV set before the current MI is used.
  // This is sufficient as long as no new copy instructions are generated
  // to copy the free register to memory.
  // 
  int getUnusedUniRegAtMI(RegClass *RC, int RegType,
                          const MachineInstr *MI,
                          const ValueSet *LVSetBef = 0);
  
  void setRelRegsUsedByThisInst(RegClass *RC, int RegType,
                                const MachineInstr *MI);

  int getUniRegNotUsedByThisInst(RegClass *RC, int RegType,
                                 const MachineInstr *MI);

  void addInterf4PseudoInstr(const MachineInstr *MI);
};


#endif

