//===-- SparcInternals.h - Header file for Sparc backend ---------*- C++ -*--=//
//
// This file defines stuff that is to be private to the Sparc backend, but is 
// shared among different portions of the backend.
//
//===----------------------------------------------------------------------===//

#ifndef SPARC_INTERNALS_H
#define SPARC_INTERNALS_H


#include "SparcRegClassInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/MachineInstrInfo.h"

#include "llvm/Target/MachineSchedInfo.h"
#include "llvm/CodeGen/RegClass.h"
#include "llvm/Type.h"

#include <sys/types.h>

class UltraSparc;

// OpCodeMask definitions for the Sparc V9
// 
const OpCodeMask	Immed		= 0x00002000; // immed or reg operand?
const OpCodeMask	Annul		= 0x20000000; // annul delay instr?
const OpCodeMask	PredictTaken	= 0x00080000; // predict branch taken?


enum SparcInstrSchedClass {
  SPARC_NONE,		/* Instructions with no scheduling restrictions */
  SPARC_IEUN,		/* Integer class that can use IEU0 or IEU1 */
  SPARC_IEU0,		/* Integer class IEU0 */
  SPARC_IEU1,		/* Integer class IEU1 */
  SPARC_FPM,		/* FP Multiply or Divide instructions */
  SPARC_FPA,		/* All other FP instructions */	
  SPARC_CTI,		/* Control-transfer instructions */
  SPARC_LD,		/* Load instructions */
  SPARC_ST,		/* Store instructions */
  SPARC_SINGLE,		/* Instructions that must issue by themselves */
  
  SPARC_INV,		/* This should stay at the end for the next value */
  SPARC_NUM_SCHED_CLASSES = SPARC_INV
};


//---------------------------------------------------------------------------
// enum SparcMachineOpCode. 
// const MachineInstrDescriptor SparcMachineInstrDesc[]
// 
// Purpose:
//   Description of UltraSparc machine instructions.
// 
//---------------------------------------------------------------------------

enum SparcMachineOpCode {
#define I(ENUM, OPCODESTRING, NUMOPERANDS, RESULTPOS, MAXIMM, IMMSE, \
          NUMDELAYSLOTS, LATENCY, SCHEDCLASS, INSTFLAGS)             \
   ENUM,
#include "SparcInstr.def"

  // End-of-array marker
  INVALID_OPCODE,
  NUM_REAL_OPCODES = RETURN+1,		// number of valid opcodes
  NUM_TOTAL_OPCODES = INVALID_OPCODE
};


// Array of machine instruction descriptions...
extern const MachineInstrDescriptor SparcMachineInstrDesc[];


//---------------------------------------------------------------------------
// class UltraSparcInstrInfo 
// 
// Purpose:
//   Information about individual instructions.
//   Most information is stored in the SparcMachineInstrDesc array above.
//   Other information is computed on demand, and most such functions
//   default to member functions in base class MachineInstrInfo. 
//---------------------------------------------------------------------------

class UltraSparcInstrInfo : public MachineInstrInfo {
public:
  /*ctor*/	UltraSparcInstrInfo();
  
  virtual bool		hasResultInterlock	(MachineOpCode opCode) const
  {
    // All UltraSPARC instructions have interlocks (note that delay slots
    // are not considered here).
    // However, instructions that use the result of an FCMP produce a
    // 9-cycle stall if they are issued less than 3 cycles after the FCMP.
    // Force the compiler to insert a software interlock (i.e., gap of
    // 2 other groups, including NOPs if necessary).
    return (opCode == FCMPS || opCode == FCMPD || opCode == FCMPQ);
  }

  //-------------------------------------------------------------------------
  // Code generation support for creating individual machine instructions
  //-------------------------------------------------------------------------
  
  // Create an instruction sequence to put the constant `val' into
  // the virtual register `dest'.  The generated instructions are
  // returned in `minstrVec'.  Any temporary registers (TmpInstruction)
  // created are returned in `tempVec'.
  // 
  virtual void  CreateCodeToLoadConst(Value* val,
                                      Instruction* dest,
                                      vector<MachineInstr*>& minstrVec,
                                      vector<TmpInstruction*>& tempVec) const;
};



class LiveRange;
class UltraSparc;



class UltraSparcRegInfo : public MachineRegInfo
{

 private:

  // The actual register classes in the Sparc

  enum RegClassIDs { 
    IntRegClassID, 
    FloatRegClassID, 
    IntCCRegClassID,
    FloatCCRegClassID 
  };


  // Type of registers available in Sparc. There can be several reg types
  // in the same class. For instace, the float reg class has Single/Double
  // types
  enum RegTypes {
    IntRegType,
    FPSingleRegType,
    FPDoubleRegType,
    IntCCRegType,
    FloatCCRegType
  };


  // WARNING: If the above enum order must be changed, also modify 
  // getRegisterClassOfValue method below since it assumes this particular 
  // order for efficiency.


  // reverse pointer to get info about the ultra sparc machine
  const UltraSparc *const UltraSparcInfo;

  // Both int and float rguments can be passed in 6 int regs - 
  // %o0 to %o5 (cannot be changed)
  unsigned const NumOfIntArgRegs;
  unsigned const NumOfFloatArgRegs;
  int const InvalidRegNum;

  //void setCallArgColor(LiveRange *const LR, const unsigned RegNo) const;

  void setCallOrRetArgCol(LiveRange *const LR, const unsigned RegNo,
			 const MachineInstr *MI,AddedInstrMapType &AIMap)const;

  MachineInstr * getCopy2RegMI(const Value *SrcVal, const unsigned Reg,
			       unsigned RegClassID) const ;


  void suggestReg4RetAddr(const MachineInstr * RetMI, 
			  LiveRangeInfo& LRI) const;

  void suggestReg4CallAddr(const MachineInstr * CallMI, LiveRangeInfo& LRI,
			   vector<RegClass *> RCList) const;


  Value *getValue4ReturnAddr( const MachineInstr * MInst ) const ;

  int getRegType(const LiveRange *const LR) const {

    unsigned Typ;

    switch(  (LR->getRegClass())->getID() ) {

    case IntRegClassID: return IntRegType; 

    case FloatRegClassID: 
                          Typ =  LR->getTypeID();
			  if( Typ == Type::FloatTyID ) 
			    return FPSingleRegType;
                          else if( Typ == Type::DoubleTyID )
			    return FPDoubleRegType;
                          else assert(0 && "Unknown type in FloatRegClass");

    case IntCCRegClassID: return IntCCRegType; 
      
    case FloatCCRegClassID: return FloatCCRegType ; 

    default: assert( 0 && "Unknown reg class ID");

    }

  }

  int getRegType(const Value *const Val) const {

    unsigned Typ;

    switch( getRegClassIDOfValue(Val)  ) {

    case IntRegClassID: return IntRegType; 

    case FloatRegClassID: 
                          Typ =  (Val->getType())->getPrimitiveID();
			  if( Typ == Type::FloatTyID ) 
			    return FPSingleRegType;
                          else if( Typ == Type::DoubleTyID )
			    return FPDoubleRegType;
                          else assert(0 && "Unknown type in FloatRegClass");

    case IntCCRegClassID: return IntCCRegType; 
      
    case FloatCCRegClassID: return FloatCCRegType ; 

    default: assert( 0 && "Unknown reg class ID");

    }

  }



  // ***TODO: See this method is necessary

  MachineInstr * cpValue2RegMI(Value * Val,  const unsigned DestReg,
			       const int RegType) const;

  const Value *getCallInstRetAddr(const MachineInstr *CallMI) const;
  const unsigned getCallInstNumArgs(const MachineInstr *CallMI) const;


 public:


  UltraSparcRegInfo(const UltraSparc *const USI ) : UltraSparcInfo(USI), 
						    NumOfIntArgRegs(6), 
						    NumOfFloatArgRegs(32),
						    InvalidRegNum(1000)
  {    
    MachineRegClassArr.push_back( new SparcIntRegClass(IntRegClassID) );
    MachineRegClassArr.push_back( new SparcFloatRegClass(FloatRegClassID) );
    MachineRegClassArr.push_back( new SparcIntCCRegClass(IntCCRegClassID) );
    MachineRegClassArr.push_back( new SparcFloatCCRegClass(FloatCCRegClassID));

    assert( SparcFloatRegOrder::StartOfNonVolatileRegs == 32 && 
	    "32 Float regs are used for float arg passing");
  }

  // ***** TODO  Delete
  ~UltraSparcRegInfo(void) { }              // empty destructor 


  inline const UltraSparc & getUltraSparcInfo() const { 
    return *UltraSparcInfo;
  }



  inline unsigned getRegClassIDOfValue (const Value *const Val,
					bool isCCReg = false) const {

    Type::PrimitiveID ty = (Val->getType())->getPrimitiveID();

    unsigned res;
    
    if( (ty && ty <= Type::LongTyID) || (ty == Type::LabelTyID) ||
	(ty == Type::MethodTyID) ||  (ty == Type::PointerTyID) )
      res =  IntRegClassID;             // sparc int reg (ty=0: void)
    else if( ty <= Type::DoubleTyID)
      res = FloatRegClassID;           // sparc float reg class
    else { 
      cerr << "TypeID: " << ty << endl;
      assert(0 && "Cannot resolve register class for type");
    }

    if(isCCReg)
      return res + 2;      // corresponidng condition code regiser 
    else 
      return res;
  }

  // returns the register tha contains always zero
  // this is the unified register number
  inline int getZeroRegNum() const { return SparcIntRegOrder::g0; }

  // returns the reg used for pushing the address when a method is called.
  // This can be used for other purposes between calls
  unsigned getCallAddressReg() const  { return SparcIntRegOrder::o7; }

  
  // and when we return from a method. It should be made sure that this 
  // register contains the return value when a return instruction is reached.
  unsigned getReturnAddressReg()  const { return SparcIntRegOrder::i7; }

  void suggestRegs4MethodArgs(const Method *const Meth, 
			      LiveRangeInfo& LRI) const;

  void suggestRegs4CallArgs(const MachineInstr *const CallMI, 
			    LiveRangeInfo& LRI, vector<RegClass *> RCL) const; 

  void suggestReg4RetValue(const MachineInstr *const RetMI, 
			    LiveRangeInfo& LRI ) const;


  void colorMethodArgs(const Method *const Meth,  LiveRangeInfo& LRI,
		       AddedInstrns *const FirstAI) const;

  void colorCallArgs(const MachineInstr *const CallMI, LiveRangeInfo& LRI,
		     AddedInstrns *const CallAI) const;

  void colorRetValue(const MachineInstr *const RetI,   LiveRangeInfo& LRI,
		     AddedInstrns *const RetAI) const;


  // bool handleSpecialMInstr(const MachineInstr * MInst, 
  //			   LiveRangeInfo& LRI, vector<RegClass *> RCL) const;


    static void printReg(const LiveRange *const LR)  ;

  // this method provides a unique number for each register 
  inline int getUnifiedRegNum(int RegClassID, int reg) const {

    if( RegClassID == IntRegClassID && reg < 32 ) 
      return reg;
    else if ( RegClassID == FloatRegClassID && reg < 64)
      return reg + 32;                  // we have 32 int regs
    else if( RegClassID == FloatCCRegClassID && reg < 4)
      return reg + 32 + 64;             // 32 int, 64 float
    else if( RegClassID == IntCCRegClassID ) 
      return 4+ 32 + 64;                // only int cc reg
    else if (reg==InvalidRegNum)                
      return InvalidRegNum;
    else  
      assert(0 && "Invalid register class or reg number");

  }

  // given the unified register number, this gives the name
  inline const string getUnifiedRegName(int reg) const {
    if( reg < 32 ) 
      return SparcIntRegOrder::getRegName(reg);
    else if ( reg < (64 + 32) )
      return SparcFloatRegOrder::getRegName( reg  - 32);                  
    else if( reg < (64+32+4) )
      return SparcFloatCCRegOrder::getRegName( reg -32 - 64);
    else if ( reg == 64+32+4)
      return "xcc";                     // only integer cc reg

    else if (reg== InvalidRegNum)       //****** TODO: Remove
      return "<*NoReg*>";
    else 
      assert(0 && "Invalid register number");
  }


  const Value * getCallInstRetVal(const MachineInstr *CallMI) const;

  MachineInstr * cpReg2RegMI(const unsigned SrcReg, const unsigned DestReg,
			     const int RegType) const;

  MachineInstr * cpReg2MemMI(const unsigned SrcReg,  const unsigned DestPtrReg,
			     const int Offset, const int RegType) const;

  MachineInstr * cpMem2RegMI(const unsigned SrcPtrReg, const int Offset,
			     const unsigned DestReg, const int RegType) const;

  inline bool isRegVolatile(const int RegClassID, const int Reg) const {
    return  (MachineRegClassArr[RegClassID])->isRegVolatile(Reg);
  }


  inline unsigned getFramePointer() const {
    return SparcIntRegOrder::i6;
  }

  inline unsigned getStackPointer() const {
    return SparcIntRegOrder::o6;
  }

  inline int getInvalidRegNum() const {
    return InvalidRegNum;
  }

};



/*---------------------------------------------------------------------------
Scheduling guidelines for SPARC IIi:

I-Cache alignment rules (pg 326)
-- Align a branch target instruction so that it's entire group is within
   the same cache line (may be 1-4 instructions).
** Don't let a branch that is predicted taken be the last instruction
   on an I-cache line: delay slot will need an entire line to be fetched
-- Make a FP instruction or a branch be the 4th instruction in a group.
   For branches, there are tradeoffs in reordering to make this happen
   (see pg. 327).
** Don't put a branch in a group that crosses a 32-byte boundary!
   An artificial branch is inserted after every 32 bytes, and having
   another branch will force the group to be broken into 2 groups. 

iTLB rules:
-- Don't let a loop span two memory pages, if possible

Branch prediction performance:
-- Don't make the branch in a delay slot the target of a branch
-- Try not to have 2 predicted branches within a group of 4 instructions
   (because each such group has a single branch target field).
-- Try to align branches in slots 0, 2, 4 or 6 of a cache line (to avoid
   the wrong prediction bits being used in some cases).

D-Cache timing constraints:
-- Signed int loads of less than 64 bits have 3 cycle latency, not 2
-- All other loads that hit in D-Cache have 2 cycle latency
-- All loads are returned IN ORDER, so a D-Cache miss will delay a later hit
-- Mis-aligned loads or stores cause a trap.  In particular, replace
   mis-aligned FP double precision l/s with 2 single-precision l/s.
-- Simulations of integer codes show increase in avg. group size of
   33% when code (including esp. non-faulting loads) is moved across
   one branch, and 50% across 2 branches.

E-Cache timing constraints:
-- Scheduling for E-cache (D-Cache misses) is effective (due to load buffering)

Store buffer timing constraints:
-- Stores can be executed in same cycle as instruction producing the value
-- Stores are buffered and have lower priority for E-cache until
   highwater mark is reached in the store buffer (5 stores)

Pipeline constraints:
-- Shifts can only use IEU0.
-- CC setting instructions can only use IEU1.
-- Several other instructions must only use IEU1:
   EDGE(?), ARRAY(?), CALL, JMPL, BPr, PST, and FCMP.
-- Two instructions cannot store to the same register file in a single cycle
   (single write port per file).

Issue and grouping constraints:
-- FP and branch instructions must use slot 4.
-- Shift instructions cannot be grouped with other IEU0-specific instructions.
-- CC setting instructions cannot be grouped with other IEU1-specific instrs.
-- Several instructions must be issued in a single-instruction group:
	MOVcc or MOVr, MULs/x and DIVs/x, SAVE/RESTORE, many others
-- A CALL or JMPL breaks a group, ie, is not combined with subsequent instrs.
-- 
-- 

Branch delay slot scheduling rules:
-- A CTI couple (two back-to-back CTI instructions in the dynamic stream)
   has a 9-instruction penalty: the entire pipeline is flushed when the
   second instruction reaches stage 9 (W-Writeback).
-- Avoid putting multicycle instructions, and instructions that may cause
   load misses, in the delay slot of an annulling branch.
-- Avoid putting WR, SAVE..., RESTORE and RETURN instructions in the
   delay slot of an annulling branch.

 *--------------------------------------------------------------------------- */

//---------------------------------------------------------------------------
// List of CPUResources for UltraSPARC IIi.
//---------------------------------------------------------------------------

const CPUResource  AllIssueSlots(   "All Instr Slots", 4);
const CPUResource  IntIssueSlots(   "Int Instr Slots", 3);
const CPUResource  First3IssueSlots("Instr Slots 0-3", 3);
const CPUResource  LSIssueSlots(    "Load-Store Instr Slot", 1);
const CPUResource  CTIIssueSlots(   "Ctrl Transfer Instr Slot", 1);
const CPUResource  FPAIssueSlots(   "Int Instr Slot 1", 1);
const CPUResource  FPMIssueSlots(   "Int Instr Slot 1", 1);

// IEUN instructions can use either Alu and should use IAluN.
// IEU0 instructions must use Alu 1 and should use both IAluN and IAlu0. 
// IEU1 instructions must use Alu 2 and should use both IAluN and IAlu1. 
const CPUResource  IAluN("Int ALU 1or2", 2);
const CPUResource  IAlu0("Int ALU 1",    1);
const CPUResource  IAlu1("Int ALU 2",    1);

const CPUResource  LSAluC1("Load/Store Unit Addr Cycle", 1);
const CPUResource  LSAluC2("Load/Store Unit Issue Cycle", 1);
const CPUResource  LdReturn("Load Return Unit", 1);

const CPUResource  FPMAluC1("FP Mul/Div Alu Cycle 1", 1);
const CPUResource  FPMAluC2("FP Mul/Div Alu Cycle 2", 1);
const CPUResource  FPMAluC3("FP Mul/Div Alu Cycle 3", 1);

const CPUResource  FPAAluC1("FP Other Alu Cycle 1", 1);
const CPUResource  FPAAluC2("FP Other Alu Cycle 2", 1);
const CPUResource  FPAAluC3("FP Other Alu Cycle 3", 1);

const CPUResource  IRegReadPorts("Int Reg ReadPorts", INT_MAX);	 // CHECK
const CPUResource  IRegWritePorts("Int Reg WritePorts", 2);	 // CHECK
const CPUResource  FPRegReadPorts("FP Reg Read Ports", INT_MAX); // CHECK
const CPUResource  FPRegWritePorts("FP Reg Write Ports", 1);	 // CHECK

const CPUResource  CTIDelayCycle( "CTI  delay cycle", 1);
const CPUResource  FCMPDelayCycle("FCMP delay cycle", 1);


//---------------------------------------------------------------------------
// const InstrClassRUsage SparcRUsageDesc[]
// 
// Purpose:
//   Resource usage information for instruction in each scheduling class.
//   The InstrRUsage Objects for individual classes are specified first.
//   Note that fetch and decode are decoupled from the execution pipelines
//   via an instr buffer, so they are not included in the cycles below.
//---------------------------------------------------------------------------

const InstrClassRUsage NoneClassRUsage = {
  SPARC_NONE,
  /*totCycles*/ 7,
  
  /* maxIssueNum */ 4,
  /* isSingleIssue */ false,
  /* breaksGroup */ false,
  /* numBubbles */ 0,
  
  /*numSlots*/ 4,
  /* feasibleSlots[] */ { 0, 1, 2, 3 },
  
  /*numEntries*/ 0,
  /* V[] */ {
    /*Cycle G */
    /*Ccle E */
    /*Cycle C */
    /*Cycle N1*/
    /*Cycle N1*/
    /*Cycle N1*/
    /*Cycle W */
  }
};

const InstrClassRUsage IEUNClassRUsage = {
  SPARC_IEUN,
  /*totCycles*/ 7,
  
  /* maxIssueNum */ 3,
  /* isSingleIssue */ false,
  /* breaksGroup */ false,
  /* numBubbles */ 0,
  
  /*numSlots*/ 3,
  /* feasibleSlots[] */ { 0, 1, 2 },
  
  /*numEntries*/ 4,
  /* V[] */ {
    /*Cycle G */ { AllIssueSlots.rid, 0, 1 },
                 { IntIssueSlots.rid, 0, 1 },
    /*Cycle E */ { IAluN.rid, 1, 1 },
    /*Cycle C */
    /*Cycle N1*/
    /*Cycle N1*/
    /*Cycle N1*/
    /*Cycle W */ { IRegWritePorts.rid, 6, 1  }
  }
};

const InstrClassRUsage IEU0ClassRUsage = {
  SPARC_IEU0,
  /*totCycles*/ 7,
  
  /* maxIssueNum */ 1,
  /* isSingleIssue */ false,
  /* breaksGroup */ false,
  /* numBubbles */ 0,
  
  /*numSlots*/ 3,
  /* feasibleSlots[] */ { 0, 1, 2 },
  
  /*numEntries*/ 5,
  /* V[] */ {
    /*Cycle G */ { AllIssueSlots.rid, 0, 1 },
                 { IntIssueSlots.rid, 0, 1 },
    /*Cycle E */ { IAluN.rid, 1, 1 },
                 { IAlu0.rid, 1, 1 },
    /*Cycle C */
    /*Cycle N1*/
    /*Cycle N1*/
    /*Cycle N1*/
    /*Cycle W */ { IRegWritePorts.rid, 6, 1 }
  }
};

const InstrClassRUsage IEU1ClassRUsage = {
  SPARC_IEU1,
  /*totCycles*/ 7,
  
  /* maxIssueNum */ 1,
  /* isSingleIssue */ false,
  /* breaksGroup */ false,
  /* numBubbles */ 0,
  
  /*numSlots*/ 3,
  /* feasibleSlots[] */ { 0, 1, 2 },
  
  /*numEntries*/ 5,
  /* V[] */ {
    /*Cycle G */ { AllIssueSlots.rid, 0, 1 },
               { IntIssueSlots.rid, 0, 1 },
    /*Cycle E */ { IAluN.rid, 1, 1 },
               { IAlu1.rid, 1, 1 },
    /*Cycle C */
    /*Cycle N1*/
    /*Cycle N1*/
    /*Cycle N1*/
    /*Cycle W */ { IRegWritePorts.rid, 6, 1 }
  }
};

const InstrClassRUsage FPMClassRUsage = {
  SPARC_FPM,
  /*totCycles*/ 7,
  
  /* maxIssueNum */ 1,
  /* isSingleIssue */ false,
  /* breaksGroup */ false,
  /* numBubbles */ 0,
  
  /*numSlots*/ 4,
  /* feasibleSlots[] */ { 0, 1, 2, 3 },
  
  /*numEntries*/ 7,
  /* V[] */ {
    /*Cycle G */ { AllIssueSlots.rid,   0, 1 },
                 { FPMIssueSlots.rid,   0, 1 },
    /*Cycle E */ { FPRegReadPorts.rid,  1, 1 },
    /*Cycle C */ { FPMAluC1.rid,        2, 1 },
    /*Cycle N1*/ { FPMAluC2.rid,        3, 1 },
    /*Cycle N1*/ { FPMAluC3.rid,        4, 1 },
    /*Cycle N1*/
    /*Cycle W */ { FPRegWritePorts.rid, 6, 1 }
  }
};

const InstrClassRUsage FPAClassRUsage = {
  SPARC_FPA,
  /*totCycles*/ 7,
  
  /* maxIssueNum */ 1,
  /* isSingleIssue */ false,
  /* breaksGroup */ false,
  /* numBubbles */ 0,
  
  /*numSlots*/ 4,
  /* feasibleSlots[] */ { 0, 1, 2, 3 },
  
  /*numEntries*/ 7,
  /* V[] */ {
    /*Cycle G */ { AllIssueSlots.rid,   0, 1 },
                 { FPAIssueSlots.rid,   0, 1 },
    /*Cycle E */ { FPRegReadPorts.rid,  1, 1 },
    /*Cycle C */ { FPAAluC1.rid,        2, 1 },
    /*Cycle N1*/ { FPAAluC2.rid,        3, 1 },
    /*Cycle N1*/ { FPAAluC3.rid,        4, 1 },
    /*Cycle N1*/
    /*Cycle W */ { FPRegWritePorts.rid, 6, 1 }
  }
};

const InstrClassRUsage LDClassRUsage = {
  SPARC_LD,
  /*totCycles*/ 7,
  
  /* maxIssueNum */ 1,
  /* isSingleIssue */ false,
  /* breaksGroup */ false,
  /* numBubbles */ 0,
  
  /*numSlots*/ 3,
  /* feasibleSlots[] */ { 0, 1, 2, },
  
  /*numEntries*/ 6,
  /* V[] */ {
    /*Cycle G */ { AllIssueSlots.rid,    0, 1 },
                 { First3IssueSlots.rid, 0, 1 },
                 { LSIssueSlots.rid,     0, 1 },
    /*Cycle E */ { LSAluC1.rid,          1, 1 },
    /*Cycle C */ { LSAluC2.rid,          2, 1 },
                 { LdReturn.rid,         2, 1 },
    /*Cycle N1*/
    /*Cycle N1*/
    /*Cycle N1*/
    /*Cycle W */ { IRegWritePorts.rid,   6, 1 }
  }
};

const InstrClassRUsage STClassRUsage = {
  SPARC_ST,
  /*totCycles*/ 7,
  
  /* maxIssueNum */ 1,
  /* isSingleIssue */ false,
  /* breaksGroup */ false,
  /* numBubbles */ 0,
  
  /*numSlots*/ 3,
  /* feasibleSlots[] */ { 0, 1, 2 },
  
  /*numEntries*/ 4,
  /* V[] */ {
    /*Cycle G */ { AllIssueSlots.rid,    0, 1 },
                 { First3IssueSlots.rid, 0, 1 },
                 { LSIssueSlots.rid,     0, 1 },
    /*Cycle E */ { LSAluC1.rid,          1, 1 },
    /*Cycle C */ { LSAluC2.rid,          2, 1 }
    /*Cycle N1*/
    /*Cycle N1*/
    /*Cycle N1*/
    /*Cycle W */
  }
};

const InstrClassRUsage CTIClassRUsage = {
  SPARC_CTI,
  /*totCycles*/ 7,
  
  /* maxIssueNum */ 1,
  /* isSingleIssue */ false,
  /* breaksGroup */ false,
  /* numBubbles */ 0,
  
  /*numSlots*/ 4,
  /* feasibleSlots[] */ { 0, 1, 2, 3 },
  
  /*numEntries*/ 4,
  /* V[] */ {
    /*Cycle G */ { AllIssueSlots.rid,    0, 1 },
		 { CTIIssueSlots.rid,    0, 1 },
    /*Cycle E */ { IAlu0.rid,            1, 1 },
    /*Cycles E-C */ { CTIDelayCycle.rid, 1, 2 }
    /*Cycle C */             
    /*Cycle N1*/
    /*Cycle N1*/
    /*Cycle N1*/
    /*Cycle W */
  }
};

const InstrClassRUsage SingleClassRUsage = {
  SPARC_SINGLE,
  /*totCycles*/ 7,
  
  /* maxIssueNum */ 1,
  /* isSingleIssue */ true,
  /* breaksGroup */ false,
  /* numBubbles */ 0,
  
  /*numSlots*/ 1,
  /* feasibleSlots[] */ { 0 },
  
  /*numEntries*/ 5,
  /* V[] */ {
    /*Cycle G */ { AllIssueSlots.rid,    0, 1 },
                 { AllIssueSlots.rid,    0, 1 },
                 { AllIssueSlots.rid,    0, 1 },
                 { AllIssueSlots.rid,    0, 1 },
    /*Cycle E */ { IAlu0.rid,            1, 1 }
    /*Cycle C */
    /*Cycle N1*/
    /*Cycle N1*/
    /*Cycle N1*/
    /*Cycle W */
  }
};


const InstrClassRUsage SparcRUsageDesc[] = {
  NoneClassRUsage,
  IEUNClassRUsage,
  IEU0ClassRUsage,
  IEU1ClassRUsage,
  FPMClassRUsage,
  FPAClassRUsage,
  CTIClassRUsage,
  LDClassRUsage,
  STClassRUsage,
  SingleClassRUsage
};


//---------------------------------------------------------------------------
// const InstrIssueDelta  SparcInstrIssueDeltas[]
// 
// Purpose:
//   Changes to issue restrictions information in InstrClassRUsage for
//   instructions that differ from other instructions in their class.
//---------------------------------------------------------------------------

const InstrIssueDelta  SparcInstrIssueDeltas[] = {

  // opCode,  isSingleIssue,  breaksGroup,  numBubbles

				// Special cases for single-issue only
				// Other single issue cases are below.
//{ LDDA,	true,	true,	0 },
//{ STDA,	true,	true,	0 },
//{ LDDF,	true,	true,	0 },
//{ LDDFA,	true,	true,	0 },
  { ADDC,	true,	true,	0 },
  { ADDCcc,	true,	true,	0 },
  { SUBC,	true,	true,	0 },
  { SUBCcc,	true,	true,	0 },
//{ SAVE,	true,	true,	0 },
//{ RESTORE,	true,	true,	0 },
//{ LDSTUB,	true,	true,	0 },
//{ SWAP,	true,	true,	0 },
//{ SWAPA,	true,	true,	0 },
//{ CAS,	true,	true,	0 },
//{ CASA,	true,	true,	0 },
//{ CASX,	true,	true,	0 },
//{ CASXA,	true,	true,	0 },
//{ LDFSR,	true,	true,	0 },
//{ LDFSRA,	true,	true,	0 },
//{ LDXFSR,	true,	true,	0 },
//{ LDXFSRA,	true,	true,	0 },
//{ STFSR,	true,	true,	0 },
//{ STFSRA,	true,	true,	0 },
//{ STXFSR,	true,	true,	0 },
//{ STXFSRA,	true,	true,	0 },
//{ SAVED,	true,	true,	0 },
//{ RESTORED,	true,	true,	0 },
//{ FLUSH,	true,	true,	9 },
//{ FLUSHW,	true,	true,	9 },
//{ ALIGNADDR,	true,	true,	0 },
  { RETURN,	true,	true,	0 },
//{ DONE,	true,	true,	0 },
//{ RETRY,	true,	true,	0 },
//{ WR,		true,	true,	0 },
//{ WRPR,	true,	true,	4 },
//{ RD,		true,	true,	0 },
//{ RDPR,	true,	true,	0 },
//{ TCC,	true,	true,	0 },
//{ SHUTDOWN,	true,	true,	0 },
  
				// Special cases for breaking group *before*
				// CURRENTLY NOT SUPPORTED!
  { CALL,	false,	false,	0 },
  { JMPL,	false,	false,	0 },
  
				// Special cases for breaking the group *after*
  { MULX,	true,	true,	(4+34)/2 },
  { FDIVS,	false,	true,	0 },
  { FDIVD,	false,	true,	0 },
  { FDIVQ,	false,	true,	0 },
  { FSQRTS,	false,	true,	0 },
  { FSQRTD,	false,	true,	0 },
  { FSQRTQ,	false,	true,	0 },
//{ FCMP{LE,GT,NE,EQ}, false, true, 0 },
  
				// Instructions that introduce bubbles
//{ MULScc,	true,	true,	2 },
//{ SMULcc,	true,	true,	(4+18)/2 },
//{ UMULcc,	true,	true,	(4+19)/2 },
  { SDIVX,	true,	true,	68 },
  { UDIVX,	true,	true,	68 },
//{ SDIVcc,	true,	true,	36 },
//{ UDIVcc,	true,	true,	37 },
//{ WR,		false,	false,	4 },
//{ WRPR,	false,	false,	4 },
};


//---------------------------------------------------------------------------
// const InstrRUsageDelta SparcInstrUsageDeltas[]
// 
// Purpose:
//   Changes to resource usage information in InstrClassRUsage for
//   instructions that differ from other instructions in their class.
//---------------------------------------------------------------------------

const InstrRUsageDelta SparcInstrUsageDeltas[] = {

  // MachineOpCode, Resource, Start cycle, Num cycles

  // 
  // JMPL counts as a load/store instruction for issue!
  //
  { JMPL,     LSIssueSlots.rid,  0,  1 },
  
  // 
  // Many instructions cannot issue for the next 2 cycles after an FCMP
  // We model that with a fake resource FCMPDelayCycle.
  // 
  { FCMPS,    FCMPDelayCycle.rid, 1, 3 },
  { FCMPD,    FCMPDelayCycle.rid, 1, 3 },
  { FCMPQ,    FCMPDelayCycle.rid, 1, 3 },
  
  { MULX,     FCMPDelayCycle.rid, 1, 1 },
  { SDIVX,    FCMPDelayCycle.rid, 1, 1 },
  { UDIVX,    FCMPDelayCycle.rid, 1, 1 },
//{ SMULcc,   FCMPDelayCycle.rid, 1, 1 },
//{ UMULcc,   FCMPDelayCycle.rid, 1, 1 },
//{ SDIVcc,   FCMPDelayCycle.rid, 1, 1 },
//{ UDIVcc,   FCMPDelayCycle.rid, 1, 1 },
  { STD,      FCMPDelayCycle.rid, 1, 1 },
  { FMOVRSZ,  FCMPDelayCycle.rid, 1, 1 },
  { FMOVRSLEZ,FCMPDelayCycle.rid, 1, 1 },
  { FMOVRSLZ, FCMPDelayCycle.rid, 1, 1 },
  { FMOVRSNZ, FCMPDelayCycle.rid, 1, 1 },
  { FMOVRSGZ, FCMPDelayCycle.rid, 1, 1 },
  { FMOVRSGEZ,FCMPDelayCycle.rid, 1, 1 },
  
  // 
  // Some instructions are stalled in the GROUP stage if a CTI is in
  // the E or C stage
  // 
  { LDD,      CTIDelayCycle.rid,  1, 1 },
//{ LDDA,     CTIDelayCycle.rid,  1, 1 },
//{ LDDSTUB,  CTIDelayCycle.rid,  1, 1 },
//{ LDDSTUBA, CTIDelayCycle.rid,  1, 1 },
//{ SWAP,     CTIDelayCycle.rid,  1, 1 },
//{ SWAPA,    CTIDelayCycle.rid,  1, 1 },
//{ CAS,      CTIDelayCycle.rid,  1, 1 },
//{ CASA,     CTIDelayCycle.rid,  1, 1 },
//{ CASX,     CTIDelayCycle.rid,  1, 1 },
//{ CASXA,    CTIDelayCycle.rid,  1, 1 },
  
  //
  // Signed int loads of less than dword size return data in cycle N1 (not C)
  // and put all loads in consecutive cycles into delayed load return mode.
  //
  { LDSB,    LdReturn.rid,  2, -1 },
  { LDSB,    LdReturn.rid,  3,  1 },
  
  { LDSH,    LdReturn.rid,  2, -1 },
  { LDSH,    LdReturn.rid,  3,  1 },
  
  { LDSW,    LdReturn.rid,  2, -1 },
  { LDSW,    LdReturn.rid,  3,  1 },


#undef EXPLICIT_BUBBLES_NEEDED
#ifdef EXPLICIT_BUBBLES_NEEDED
  // 
  // MULScc inserts one bubble.
  // This means it breaks the current group (captured in UltraSparcSchedInfo)
  // *and occupies all issue slots for the next cycle
  // 
//{ MULScc,  AllIssueSlots.rid, 2, 2-1 },
//{ MULScc,  AllIssueSlots.rid, 2, 2-1 },
//{ MULScc,  AllIssueSlots.rid, 2, 2-1 },
//{ MULScc,  AllIssueSlots.rid,  2, 2-1 },
  
  // 
  // SMULcc inserts between 4 and 18 bubbles, depending on #leading 0s in rs1.
  // We just model this with a simple average.
  // 
//{ SMULcc,  AllIssueSlots.rid, 2, ((4+18)/2)-1 },
//{ SMULcc,  AllIssueSlots.rid, 2, ((4+18)/2)-1 },
//{ SMULcc,  AllIssueSlots.rid, 2, ((4+18)/2)-1 },
//{ SMULcc,  AllIssueSlots.rid,  2, ((4+18)/2)-1 },
  
  // SMULcc inserts between 4 and 19 bubbles, depending on #leading 0s in rs1.
//{ UMULcc,  AllIssueSlots.rid, 2, ((4+19)/2)-1 },
//{ UMULcc,  AllIssueSlots.rid, 2, ((4+19)/2)-1 },
//{ UMULcc,  AllIssueSlots.rid, 2, ((4+19)/2)-1 },
//{ UMULcc,  AllIssueSlots.rid,  2, ((4+19)/2)-1 },
  
  // 
  // MULX inserts between 4 and 34 bubbles, depending on #leading 0s in rs1.
  // 
  { MULX,    AllIssueSlots.rid, 2, ((4+34)/2)-1 },
  { MULX,    AllIssueSlots.rid, 2, ((4+34)/2)-1 },
  { MULX,    AllIssueSlots.rid, 2, ((4+34)/2)-1 },
  { MULX,    AllIssueSlots.rid,  2, ((4+34)/2)-1 },
  
  // 
  // SDIVcc inserts 36 bubbles.
  // 
//{ SDIVcc,  AllIssueSlots.rid, 2, 36-1 },
//{ SDIVcc,  AllIssueSlots.rid, 2, 36-1 },
//{ SDIVcc,  AllIssueSlots.rid, 2, 36-1 },
//{ SDIVcc,  AllIssueSlots.rid,  2, 36-1 },
  
  // UDIVcc inserts 37 bubbles.
//{ UDIVcc,  AllIssueSlots.rid, 2, 37-1 },
//{ UDIVcc,  AllIssueSlots.rid, 2, 37-1 },
//{ UDIVcc,  AllIssueSlots.rid, 2, 37-1 },
//{ UDIVcc,  AllIssueSlots.rid,  2, 37-1 },
  
  // 
  // SDIVX inserts 68 bubbles.
  // 
  { SDIVX,   AllIssueSlots.rid, 2, 68-1 },
  { SDIVX,   AllIssueSlots.rid, 2, 68-1 },
  { SDIVX,   AllIssueSlots.rid, 2, 68-1 },
  { SDIVX,   AllIssueSlots.rid,  2, 68-1 },
  
  // 
  // UDIVX inserts 68 bubbles.
  // 
  { UDIVX,   AllIssueSlots.rid, 2, 68-1 },
  { UDIVX,   AllIssueSlots.rid, 2, 68-1 },
  { UDIVX,   AllIssueSlots.rid, 2, 68-1 },
  { UDIVX,   AllIssueSlots.rid,  2, 68-1 },
  
  // 
  // WR inserts 4 bubbles.
  // 
//{ WR,     AllIssueSlots.rid, 2, 68-1 },
//{ WR,     AllIssueSlots.rid, 2, 68-1 },
//{ WR,     AllIssueSlots.rid, 2, 68-1 },
//{ WR,     AllIssueSlots.rid,  2, 68-1 },
  
  // 
  // WRPR inserts 4 bubbles.
  // 
//{ WRPR,   AllIssueSlots.rid, 2, 68-1 },
//{ WRPR,   AllIssueSlots.rid, 2, 68-1 },
//{ WRPR,   AllIssueSlots.rid, 2, 68-1 },
//{ WRPR,   AllIssueSlots.rid,  2, 68-1 },
  
  // 
  // DONE inserts 9 bubbles.
  // 
//{ DONE,   AllIssueSlots.rid, 2, 9-1 },
//{ DONE,   AllIssueSlots.rid, 2, 9-1 },
//{ DONE,   AllIssueSlots.rid, 2, 9-1 },
//{ DONE,   AllIssueSlots.rid, 2, 9-1 },
  
  // 
  // RETRY inserts 9 bubbles.
  // 
//{ RETRY,   AllIssueSlots.rid, 2, 9-1 },
//{ RETRY,   AllIssueSlots.rid, 2, 9-1 },
//{ RETRY,   AllIssueSlots.rid, 2, 9-1 },
//{ RETRY,   AllIssueSlots.rid,  2, 9-1 },

#endif  /*EXPLICIT_BUBBLES_NEEDED */
};



// Additional delays to be captured in code:
// 1. RDPR from several state registers (page 349)
// 2. RD   from *any* register (page 349)
// 3. Writes to TICK, PSTATE, TL registers and FLUSH{W} instr (page 349)
// 4. Integer store can be in same group as instr producing value to store.
// 5. BICC and BPICC can be in the same group as instr producing CC (pg 350)
// 6. FMOVr cannot be in the same or next group as an IEU instr (pg 351).
// 7. The second instr. of a CTI group inserts 9 bubbles (pg 351)
// 8. WR{PR}, SVAE, SAVED, RESTORE, RESTORED, RETURN, RETRY, and DONE that
//    follow an annulling branch cannot be issued in the same group or in
//    the 3 groups following the branch.
// 9. A predicted annulled load does not stall dependent instructions.
//    Other annulled delay slot instructions *do* stall dependents, so
//    nothing special needs to be done for them during scheduling.
//10. Do not put a load use that may be annulled in the same group as the
//    branch.  The group will stall until the load returns.
//11. Single-prec. FP loads lock 2 registers, for dependency checking.
//
// 
// Additional delays we cannot or will not capture:
// 1. If DCTI is last word of cache line, it is delayed until next line can be
//    fetched.  Also, other DCTI alignment-related delays (pg 352)
// 2. Load-after-store is delayed by 7 extra cycles if load hits in D-Cache.
//    Also, several other store-load and load-store conflicts (pg 358)
// 3. MEMBAR, LD{X}FSR, LDD{A} and a bunch of other load stalls (pg 358)
// 4. There can be at most 8 outstanding buffered store instructions
//     (including some others like MEMBAR, LDSTUB, CAS{AX}, and FLUSH)



//---------------------------------------------------------------------------
// class UltraSparcSchedInfo
// 
// Purpose:
//   Interface to instruction scheduling information for UltraSPARC.
//   The parameter values above are based on UltraSPARC IIi.
//---------------------------------------------------------------------------


class UltraSparcSchedInfo: public MachineSchedInfo {
public:
  /*ctor*/	   UltraSparcSchedInfo	(const MachineInstrInfo* mii);
  /*dtor*/ virtual ~UltraSparcSchedInfo	() {}
protected:
  virtual void	initializeResources	();
};


//---------------------------------------------------------------------------
// class UltraSparcMachine 
// 
// Purpose:
//   Primary interface to machine description for the UltraSPARC.
//   Primarily just initializes machine-dependent parameters in
//   class TargetMachine, and creates machine-dependent subclasses
//   for classes such as InstrInfo, SchedInfo and RegInfo. 
//---------------------------------------------------------------------------

class UltraSparc : public TargetMachine {
private:
  UltraSparcInstrInfo instrInfo;
  UltraSparcSchedInfo schedInfo;
  UltraSparcRegInfo   regInfo;
public:
  UltraSparc();
  virtual ~UltraSparc() {}
  
  virtual const MachineInstrInfo &getInstrInfo() const { return instrInfo; }
  virtual const MachineSchedInfo &getSchedInfo() const { return schedInfo; }
  virtual const MachineRegInfo   &getRegInfo()   const { return regInfo; }
  
  // compileMethod - For the sparc, we do instruction selection, followed by
  // delay slot scheduling, then register allocation.
  //
  virtual bool compileMethod(Method *M);

  //
  // emitAssembly - Output assembly language code (a .s file) for the specified
  // module. The specified module must have been compiled before this may be
  // used.
  //
  virtual void emitAssembly(const Module *M, ostream &OutStr) const;
};


#endif
