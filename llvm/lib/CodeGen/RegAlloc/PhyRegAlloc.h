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

#include "llvm/CodeGen/LiveRangeInfo.h"
#include "Support/NonCopyable.h"
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

class PhyRegAlloc : public NonCopyable {
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

  AddedInstrns AddedInstrAtEntry;       // to store instrns added at entry
  LoopInfo *LoopDepthCalc;              // to calculate loop depths 
  std::vector<unsigned> ResColList;     // A set of reserved regs if desired.
                                        // currently not used

public:
  PhyRegAlloc(Function *F, const TargetMachine& TM, FunctionLiveVarInfo *Lvi,
              LoopInfo *LoopDepthCalc);
  ~PhyRegAlloc();

  // main method called for allocating registers
  //
  void allocateRegisters();           


  // access to register classes by class ID
  // 
  const RegClass*  getRegClassByID(unsigned int id) const {
                                                    return RegClassList[id];
  }
        RegClass*  getRegClassByID(unsigned int id)       {
                                                    return RegClassList[id]; }
  
  
private:
  void addInterference(const Value *Def, const ValueSet *LVSet, 
		       bool isCallInst);

  void addInterferencesForArgs();
  void createIGNodeListsAndIGs();
  void buildInterferenceGraphs();

  void setCallInterferences(const MachineInstr *MInst, 
			    const ValueSet *LVSetAft );

  void move2DelayedInstr(const MachineInstr *OrigMI, 
			 const MachineInstr *DelayedMI );

  void markUnusableSugColors();
  void allocateStackSpace4SpilledLRs();

  void insertCode4SpilledLR     (const LiveRange *LR, 
                                 MachineInstr *MInst,
                                 const BasicBlock *BB,
                                 const unsigned OpNum);

  inline void constructLiveRanges() { LRI.constructLiveRanges(); }      

  void colorIncomingArgs();
  void colorCallRetArgs();
  void updateMachineCode();
  void updateInstruction(MachineInstr* MInst, BasicBlock* BB);

  void printLabel(const Value *const Val);
  void printMachineCode();


  friend class UltraSparcRegInfo;  // FIXME: remove this

  int getUsableUniRegAtMI(int RegType, 
			  const ValueSet *LVSetBef,
			  MachineInstr *MInst,
                          std::vector<MachineInstr*>& MIBef,
                          std::vector<MachineInstr*>& MIAft);
  
  int getUnusedUniRegAtMI(RegClass *RC,  const MachineInstr *MInst, 
                          const ValueSet *LVSetBef);

  void setRelRegsUsedByThisInst(RegClass *RC, const MachineInstr *MInst );
  int getUniRegNotUsedByThisInst(RegClass *RC, const MachineInstr *MInst);

  void addInterf4PseudoInstr(const MachineInstr *MInst);
};


#endif

