//===-- SparcInternals.h ----------------------------------------*- C++ -*-===//
// 
// This file defines stuff that is to be private to the Sparc backend, but is
// shared among different portions of the backend.
//
//===----------------------------------------------------------------------===//

#ifndef SPARC_INTERNALS_H
#define SPARC_INTERNALS_H

#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetSchedInfo.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/Target/TargetCacheInfo.h"
#include "llvm/Target/TargetRegInfo.h"
#include "llvm/Target/TargetOptInfo.h"
#include "llvm/Type.h"
#include "SparcRegClassInfo.h"
#include "Config/sys/types.h"

class LiveRange;
class UltraSparc;
class Pass;

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
// const TargetInstrDescriptor SparcMachineInstrDesc[]
// 
// Purpose:
//   Description of UltraSparc machine instructions.
// 
//---------------------------------------------------------------------------

namespace V9 {
  enum SparcMachineOpCode {
#define I(ENUM, OPCODESTRING, NUMOPERANDS, RESULTPOS, MAXIMM, IMMSE, \
          NUMDELAYSLOTS, LATENCY, SCHEDCLASS, INSTFLAGS)             \
   ENUM,
#include "SparcInstr.def"

    // End-of-array marker
    INVALID_OPCODE,
    NUM_REAL_OPCODES = PHI,		// number of valid opcodes
    NUM_TOTAL_OPCODES = INVALID_OPCODE
  };
}


// Array of machine instruction descriptions...
extern const TargetInstrDescriptor SparcMachineInstrDesc[];


//---------------------------------------------------------------------------
// class UltraSparcInstrInfo 
// 
// Purpose:
//   Information about individual instructions.
//   Most information is stored in the SparcMachineInstrDesc array above.
//   Other information is computed on demand, and most such functions
//   default to member functions in base class TargetInstrInfo. 
//---------------------------------------------------------------------------

struct UltraSparcInstrInfo : public TargetInstrInfo {
  UltraSparcInstrInfo();

  //
  // All immediate constants are in position 1 except the
  // store instructions and SETxx.
  // 
  virtual int getImmedConstantPos(MachineOpCode opCode) const {
    bool ignore;
    if (this->maxImmedConstant(opCode, ignore) != 0) {
      // 1st store opcode
      assert(! this->isStore((MachineOpCode) V9::STBr - 1));
      // last store opcode
      assert(! this->isStore((MachineOpCode) V9::STXFSRi + 1));

      if (opCode == V9::SETSW || opCode == V9::SETUW ||
          opCode == V9::SETX  || opCode == V9::SETHI)
        return 0;
      if (opCode >= V9::STBr && opCode <= V9::STXFSRi)
        return 2;
      return 1;
    }
    else
      return -1;
  }

  /// createNOPinstr - returns the target's implementation of NOP, which is
  /// usually a pseudo-instruction, implemented by a degenerate version of
  /// another instruction, e.g. X86: xchg ax, ax; SparcV9: sethi 0, g0
  ///
  MachineInstr* createNOPinstr() const {
    return BuildMI(V9::SETHI, 2).addZImm(0).addReg(SparcIntRegClass::g0);
  }

  /// isNOPinstr - not having a special NOP opcode, we need to know if a given
  /// instruction is interpreted as an `official' NOP instr, i.e., there may be
  /// more than one way to `do nothing' but only one canonical way to slack off.
  ///
  bool isNOPinstr(const MachineInstr &MI) const {
    // Make sure the instruction is EXACTLY `sethi g0, 0'
    if (MI.getOpcode() == V9::SETHI && MI.getNumOperands() == 2) {
      const MachineOperand &op0 = MI.getOperand(0), &op1 = MI.getOperand(1);
      if (op0.isImmediate() && op0.getImmedValue() == 0 &&
          op1.isMachineRegister() &&
          op1.getMachineRegNum() == SparcIntRegClass::g0)
      {
        return true;
      }
    }
    return false;
  }
  
  virtual bool hasResultInterlock(MachineOpCode opCode) const
  {
    // All UltraSPARC instructions have interlocks (note that delay slots
    // are not considered here).
    // However, instructions that use the result of an FCMP produce a
    // 9-cycle stall if they are issued less than 3 cycles after the FCMP.
    // Force the compiler to insert a software interlock (i.e., gap of
    // 2 other groups, including NOPs if necessary).
    return (opCode == V9::FCMPS || opCode == V9::FCMPD || opCode == V9::FCMPQ);
  }

  //-------------------------------------------------------------------------
  // Queries about representation of LLVM quantities (e.g., constants)
  //-------------------------------------------------------------------------

  virtual bool ConstantMayNotFitInImmedField(const Constant* CV,
                                             const Instruction* I) const;

  //-------------------------------------------------------------------------
  // Code generation support for creating individual machine instructions
  //-------------------------------------------------------------------------

  // Get certain common op codes for the current target.  This and all the
  // Create* methods below should be moved to a machine code generation class
  // 
  virtual MachineOpCode getNOPOpCode() const { return V9::NOP; }

  // Get the value of an integral constant in the form that must
  // be put into the machine register.  The specified constant is interpreted
  // as (i.e., converted if necessary to) the specified destination type.  The
  // result is always returned as an uint64_t, since the representation of
  // int64_t and uint64_t are identical.  The argument can be any known const.
  // 
  // isValidConstant is set to true if a valid constant was found.
  // 
  virtual uint64_t ConvertConstantToIntType(const TargetMachine &target,
                                            const Value *V,
                                            const Type *destType,
                                            bool  &isValidConstant) const;

  // Create an instruction sequence to put the constant `val' into
  // the virtual register `dest'.  `val' may be a Constant or a
  // GlobalValue, viz., the constant address of a global variable or function.
  // The generated instructions are returned in `mvec'.
  // Any temp. registers (TmpInstruction) created are recorded in mcfi.
  // Any stack space required is allocated via mcff.
  // 
  virtual void  CreateCodeToLoadConst(const TargetMachine& target,
                                      Function* F,
                                      Value* val,
                                      Instruction* dest,
                                      std::vector<MachineInstr*>& mvec,
                                      MachineCodeForInstruction& mcfi) const;

  // Create an instruction sequence to copy an integer value `val'
  // to a floating point value `dest' by copying to memory and back.
  // val must be an integral type.  dest must be a Float or Double.
  // The generated instructions are returned in `mvec'.
  // Any temp. registers (TmpInstruction) created are recorded in mcfi.
  // Any stack space required is allocated via mcff.
  // 
  virtual void  CreateCodeToCopyIntToFloat(const TargetMachine& target,
                                       Function* F,
                                       Value* val,
                                       Instruction* dest,
                                       std::vector<MachineInstr*>& mvec,
                                       MachineCodeForInstruction& mcfi) const;

  // Similarly, create an instruction sequence to copy an FP value
  // `val' to an integer value `dest' by copying to memory and back.
  // The generated instructions are returned in `mvec'.
  // Any temp. registers (TmpInstruction) created are recorded in mcfi.
  // Any stack space required is allocated via mcff.
  // 
  virtual void  CreateCodeToCopyFloatToInt(const TargetMachine& target,
                                       Function* F,
                                       Value* val,
                                       Instruction* dest,
                                       std::vector<MachineInstr*>& mvec,
                                       MachineCodeForInstruction& mcfi) const;
  
  // Create instruction(s) to copy src to dest, for arbitrary types
  // The generated instructions are returned in `mvec'.
  // Any temp. registers (TmpInstruction) created are recorded in mcfi.
  // Any stack space required is allocated via mcff.
  // 
  virtual void CreateCopyInstructionsByType(const TargetMachine& target,
                                       Function* F,
                                       Value* src,
                                       Instruction* dest,
                                       std::vector<MachineInstr*>& mvec,
                                       MachineCodeForInstruction& mcfi) const;

  // Create instruction sequence to produce a sign-extended register value
  // from an arbitrary sized value (sized in bits, not bytes).
  // The generated instructions are appended to `mvec'.
  // Any temp. registers (TmpInstruction) created are recorded in mcfi.
  // Any stack space required is allocated via mcff.
  // 
  virtual void CreateSignExtensionInstructions(const TargetMachine& target,
                                       Function* F,
                                       Value* srcVal,
                                       Value* destVal,
                                       unsigned int numLowBits,
                                       std::vector<MachineInstr*>& mvec,
                                       MachineCodeForInstruction& mcfi) const;

  // Create instruction sequence to produce a zero-extended register value
  // from an arbitrary sized value (sized in bits, not bytes).
  // The generated instructions are appended to `mvec'.
  // Any temp. registers (TmpInstruction) created are recorded in mcfi.
  // Any stack space required is allocated via mcff.
  // 
  virtual void CreateZeroExtensionInstructions(const TargetMachine& target,
                                       Function* F,
                                       Value* srcVal,
                                       Value* destVal,
                                       unsigned int numLowBits,
                                       std::vector<MachineInstr*>& mvec,
                                       MachineCodeForInstruction& mcfi) const;
};


//----------------------------------------------------------------------------
// class UltraSparcRegInfo
//
// This class implements the virtual class TargetRegInfo for Sparc.
//
//----------------------------------------------------------------------------

class UltraSparcRegInfo : public TargetRegInfo {

private:

  // Number of registers used for passing int args (usually 6: %o0 - %o5)
  //
  unsigned const NumOfIntArgRegs;

  // Number of registers used for passing float args (usually 32: %f0 - %f31)
  //
  unsigned const NumOfFloatArgRegs;

  // ========================  Private Methods =============================

  // The following methods are used to color special live ranges (e.g.
  // function args and return values etc.) with specific hardware registers
  // as required. See SparcRegInfo.cpp for the implementation.
  //
  void suggestReg4RetAddr(MachineInstr *RetMI, 
			  LiveRangeInfo &LRI) const;

  void suggestReg4CallAddr(MachineInstr *CallMI, LiveRangeInfo &LRI) const;
  
  // Helper used by the all the getRegType() functions.
  int getRegTypeForClassAndType(unsigned regClassID, const Type* type) const;

public:
  // Type of registers available in Sparc. There can be several reg types
  // in the same class. For instace, the float reg class has Single/Double
  // types
  //
  enum RegTypes {
    IntRegType,
    FPSingleRegType,
    FPDoubleRegType,
    IntCCRegType,
    FloatCCRegType,
    SpecialRegType
  };

  // The actual register classes in the Sparc
  //
  // **** WARNING: If this enum order is changed, also modify 
  // getRegisterClassOfValue method below since it assumes this particular 
  // order for efficiency.
  // 
  enum RegClassIDs { 
    IntRegClassID,                      // Integer
    FloatRegClassID,                    // Float (both single/double)
    IntCCRegClassID,                    // Int Condition Code
    FloatCCRegClassID,                  // Float Condition code
    SpecialRegClassID                   // Special (unallocated) registers
  };

  UltraSparcRegInfo(const UltraSparc &tgt);

  // To find the register class used for a specified Type
  //
  unsigned getRegClassIDOfType(const Type *type,
                               bool isCCReg = false) const;

  // To find the register class to which a specified register belongs
  //
  unsigned getRegClassIDOfRegType(int regType) const;
  
  // getZeroRegNum - returns the register that contains always zero this is the
  // unified register number
  //
  virtual int getZeroRegNum() const;

  // getCallAddressReg - returns the reg used for pushing the address when a
  // function is called. This can be used for other purposes between calls
  //
  unsigned getCallAddressReg() const;

  // Returns the register containing the return address.
  // It should be made sure that this  register contains the return 
  // value when a return instruction is reached.
  //
  unsigned getReturnAddressReg() const;

  // Number of registers used for passing int args (usually 6: %o0 - %o5)
  // and float args (usually 32: %f0 - %f31)
  //
  unsigned const getNumOfIntArgRegs() const   { return NumOfIntArgRegs; }
  unsigned const getNumOfFloatArgRegs() const { return NumOfFloatArgRegs; }
  
  // Compute which register can be used for an argument, if any
  // 
  int regNumForIntArg(bool inCallee, bool isVarArgsCall,
                      unsigned argNo, unsigned& regClassId) const;

  int regNumForFPArg(unsigned RegType, bool inCallee, bool isVarArgsCall,
                     unsigned argNo, unsigned& regClassId) const;
  
  // The following methods are used to color special live ranges (e.g.
  // function args and return values etc.) with specific hardware registers
  // as required. See SparcRegInfo.cpp for the implementation for Sparc.
  //
  void suggestRegs4MethodArgs(const Function *Meth, 
			      LiveRangeInfo& LRI) const;

  void suggestRegs4CallArgs(MachineInstr *CallMI, 
			    LiveRangeInfo& LRI) const; 

  void suggestReg4RetValue(MachineInstr *RetMI, 
                           LiveRangeInfo& LRI) const;
  
  void colorMethodArgs(const Function *Meth,  LiveRangeInfo& LRI,
                       std::vector<MachineInstr*>& InstrnsBefore,
                       std::vector<MachineInstr*>& InstrnsAfter) const;

  // method used for printing a register for debugging purposes
  //
  void printReg(const LiveRange *LR) const;
  
  // returns the # of bytes of stack space allocated for each register
  // type. For Sparc, currently we allocate 8 bytes on stack for all 
  // register types. We can optimize this later if necessary to save stack
  // space (However, should make sure that stack alignment is correct)
  //
  inline int getSpilledRegSize(int RegType) const {
    return 8;
  }


  // To obtain the return value and the indirect call address (if any)
  // contained in a CALL machine instruction
  //
  const Value * getCallInstRetVal(const MachineInstr *CallMI) const;
  const Value * getCallInstIndirectAddrVal(const MachineInstr *CallMI) const;

  // The following methods are used to generate "copy" machine instructions
  // for an architecture.
  //
  // The function regTypeNeedsScratchReg() can be used to check whether a
  // scratch register is needed to copy a register of type `regType' to
  // or from memory.  If so, such a scratch register can be provided by
  // the caller (e.g., if it knows which regsiters are free); otherwise
  // an arbitrary one will be chosen and spilled by the copy instructions.
  //
  bool regTypeNeedsScratchReg(int RegType,
                              int& scratchRegClassId) const;

  void cpReg2RegMI(std::vector<MachineInstr*>& mvec,
                   unsigned SrcReg, unsigned DestReg,
                   int RegType) const;

  void cpReg2MemMI(std::vector<MachineInstr*>& mvec,
                   unsigned SrcReg, unsigned DestPtrReg,
                   int Offset, int RegType, int scratchReg = -1) const;

  void cpMem2RegMI(std::vector<MachineInstr*>& mvec,
                   unsigned SrcPtrReg, int Offset, unsigned DestReg,
                   int RegType, int scratchReg = -1) const;

  void cpValue2Value(Value *Src, Value *Dest,
                     std::vector<MachineInstr*>& mvec) const;

  // Get the register type for a register identified different ways.
  // Note that getRegTypeForLR(LR) != getRegTypeForDataType(LR->getType())!
  // The reg class of a LR depends both on the Value types in it and whether
  // they are CC registers or not (for example).
  int getRegTypeForDataType(const Type* type) const;
  int getRegTypeForLR(const LiveRange *LR) const;
  int getRegType(int unifiedRegNum) const;

  virtual unsigned getFramePointer() const;
  virtual unsigned getStackPointer() const;
};




//---------------------------------------------------------------------------
// class UltraSparcSchedInfo
// 
// Purpose:
//   Interface to instruction scheduling information for UltraSPARC.
//   The parameter values above are based on UltraSPARC IIi.
//---------------------------------------------------------------------------


class UltraSparcSchedInfo: public TargetSchedInfo {
public:
  UltraSparcSchedInfo(const TargetMachine &tgt);
protected:
  virtual void initializeResources();
};


//---------------------------------------------------------------------------
// class UltraSparcFrameInfo 
// 
// Purpose:
//   Interface to stack frame layout info for the UltraSPARC.
//   Starting offsets for each area of the stack frame are aligned at
//   a multiple of getStackFrameSizeAlignment().
//---------------------------------------------------------------------------

class UltraSparcFrameInfo: public TargetFrameInfo {
  const TargetMachine &target;
public:
  UltraSparcFrameInfo(const TargetMachine &TM)
    : TargetFrameInfo(StackGrowsDown, StackFrameSizeAlignment, 0), target(TM) {}
  
public:
  // These methods provide constant parameters of the frame layout.
  // 
  int  getStackFrameSizeAlignment() const { return StackFrameSizeAlignment;}
  int  getMinStackFrameSize()       const { return MinStackFrameSize; }
  int  getNumFixedOutgoingArgs()    const { return NumFixedOutgoingArgs; }
  int  getSizeOfEachArgOnStack()    const { return SizeOfEachArgOnStack; }
  bool argsOnStackHaveFixedSize()   const { return true; }

  // This method adjusts a stack offset to meet alignment rules of target.
  // The fixed OFFSET (0x7ff) must be subtracted and the result aligned.
  virtual int  adjustAlignment                  (int unalignedOffset,
                                                 bool growUp,
                                                 unsigned int align) const {
    return unalignedOffset + (growUp? +1:-1)*((unalignedOffset-OFFSET) % align);
  }

  // These methods compute offsets using the frame contents for a
  // particular function.  The frame contents are obtained from the
  // MachineCodeInfoForMethod object for the given function.
  // 
  int getFirstIncomingArgOffset  (MachineFunction& mcInfo,
                                  bool& growUp) const
  {
    growUp = true;                         // arguments area grows upwards
    return FirstIncomingArgOffsetFromFP;
  }
  int getFirstOutgoingArgOffset  (MachineFunction& mcInfo,
                                  bool& growUp) const
  {
    growUp = true;                         // arguments area grows upwards
    return FirstOutgoingArgOffsetFromSP;
  }
  int getFirstOptionalOutgoingArgOffset(MachineFunction& mcInfo,
                                        bool& growUp)const
  {
    growUp = true;                         // arguments area grows upwards
    return FirstOptionalOutgoingArgOffsetFromSP;
  }
  
  int getFirstAutomaticVarOffset (MachineFunction& mcInfo,
                                  bool& growUp) const;
  int getRegSpillAreaOffset      (MachineFunction& mcInfo,
                                  bool& growUp) const;
  int getTmpAreaOffset           (MachineFunction& mcInfo,
                                  bool& growUp) const;
  int getDynamicAreaOffset       (MachineFunction& mcInfo,
                                  bool& growUp) const;

  //
  // These methods specify the base register used for each stack area
  // (generally FP or SP)
  // 
  virtual int getIncomingArgBaseRegNum()               const {
    return (int) target.getRegInfo().getFramePointer();
  }
  virtual int getOutgoingArgBaseRegNum()               const {
    return (int) target.getRegInfo().getStackPointer();
  }
  virtual int getOptionalOutgoingArgBaseRegNum()       const {
    return (int) target.getRegInfo().getStackPointer();
  }
  virtual int getAutomaticVarBaseRegNum()              const {
    return (int) target.getRegInfo().getFramePointer();
  }
  virtual int getRegSpillAreaBaseRegNum()              const {
    return (int) target.getRegInfo().getFramePointer();
  }
  virtual int getDynamicAreaBaseRegNum()               const {
    return (int) target.getRegInfo().getStackPointer();
  }

  virtual int getIncomingArgOffset(MachineFunction& mcInfo,
				   unsigned argNum) const {
    assert(argsOnStackHaveFixedSize()); 
  
    unsigned relativeOffset = argNum * getSizeOfEachArgOnStack();
    bool growUp;                          // do args grow up or down
    int firstArg = getFirstIncomingArgOffset(mcInfo, growUp);
    return growUp ? firstArg + relativeOffset : firstArg - relativeOffset; 
  }

  virtual int getOutgoingArgOffset(MachineFunction& mcInfo,
				   unsigned argNum) const {
    assert(argsOnStackHaveFixedSize()); 
    //assert(((int) argNum - this->getNumFixedOutgoingArgs())
    //     <= (int) mcInfo.getInfo()->getMaxOptionalNumArgs());
    
    unsigned relativeOffset = argNum * getSizeOfEachArgOnStack();
    bool growUp;                          // do args grow up or down
    int firstArg = getFirstOutgoingArgOffset(mcInfo, growUp);
    return growUp ? firstArg + relativeOffset : firstArg - relativeOffset; 
  }
  
private:
  /*----------------------------------------------------------------------
    This diagram shows the stack frame layout used by llc on Sparc V9.
    Note that only the location of automatic variables, spill area,
    temporary storage, and dynamically allocated stack area are chosen
    by us.  The rest conform to the Sparc V9 ABI.
    All stack addresses are offset by OFFSET = 0x7ff (2047).

    Alignment assumptions and other invariants:
    (1) %sp+OFFSET and %fp+OFFSET are always aligned on 16-byte boundary
    (2) Variables in automatic, spill, temporary, or dynamic regions
        are aligned according to their size as in all memory accesses.
    (3) Everything below the dynamically allocated stack area is only used
        during a call to another function, so it is never needed when
        the current function is active.  This is why space can be allocated
        dynamically by incrementing %sp any time within the function.
    
    STACK FRAME LAYOUT:

       ...
       %fp+OFFSET+176      Optional extra incoming arguments# 1..N
       %fp+OFFSET+168      Incoming argument #6
       ...                 ...
       %fp+OFFSET+128      Incoming argument #1
       ...                 ...
    ---%fp+OFFSET-0--------Bottom of caller's stack frame--------------------
       %fp+OFFSET-8        Automatic variables <-- ****TOP OF STACK FRAME****
                           Spill area
                           Temporary storage
       ...

       %sp+OFFSET+176+8N   Bottom of dynamically allocated stack area
       %sp+OFFSET+168+8N   Optional extra outgoing argument# N
       ...                 ...
       %sp+OFFSET+176      Optional extra outgoing argument# 1
       %sp+OFFSET+168      Outgoing argument #6
       ...                 ...
       %sp+OFFSET+128      Outgoing argument #1
       %sp+OFFSET+120      Save area for %i7
       ...                 ...
       %sp+OFFSET+0        Save area for %l0 <-- ****BOTTOM OF STACK FRAME****

   *----------------------------------------------------------------------*/

  // All stack addresses must be offset by 0x7ff (2047) on Sparc V9.
  static const int OFFSET                                  = (int) 0x7ff;
  static const int StackFrameSizeAlignment                 =  16;
  static const int MinStackFrameSize                       = 176;
  static const int NumFixedOutgoingArgs                    =   6;
  static const int SizeOfEachArgOnStack                    =   8;
  static const int FirstIncomingArgOffsetFromFP            = 128 + OFFSET;
  static const int FirstOptionalIncomingArgOffsetFromFP    = 176 + OFFSET;
  static const int StaticAreaOffsetFromFP                  =   0 + OFFSET;
  static const int FirstOutgoingArgOffsetFromSP            = 128 + OFFSET;
  static const int FirstOptionalOutgoingArgOffsetFromSP    = 176 + OFFSET;
};


//---------------------------------------------------------------------------
// class UltraSparcCacheInfo 
// 
// Purpose:
//   Interface to cache parameters for the UltraSPARC.
//   Just use defaults for now.
//---------------------------------------------------------------------------

struct UltraSparcCacheInfo: public TargetCacheInfo {
  UltraSparcCacheInfo(const TargetMachine &T) : TargetCacheInfo(T) {} 
};


//---------------------------------------------------------------------------
// class UltraSparcOptInfo 
// 
// Purpose:
//   Interface to machine-level optimization routines for the UltraSPARC.
//---------------------------------------------------------------------------

struct UltraSparcOptInfo: public TargetOptInfo {
  UltraSparcOptInfo(const TargetMachine &T) : TargetOptInfo(T) {} 

  virtual bool IsUselessCopy    (const MachineInstr* MI) const;
};

/// createAddRegNumToValuesPass - this pass adds unsigned register numbers to
/// instructions, since that's not done by the Sparc InstSelector, but that's
/// how the target-independent register allocator in the JIT likes to see
/// instructions. This pass enables the usage of the JIT register allocator(s).
Pass *createAddRegNumToValuesPass();

/// createStackSlotsPass - External interface to stack-slots pass that enters 2
/// empty slots at the top of each function stack
Pass *createStackSlotsPass(const TargetMachine &TM);

// Interface to pre-selection pass that specializes LLVM code for a target
// machine.
Pass *createPreSelectionPass(TargetMachine &Target);

// External interface to peephole optimization pass operating on machine code.
FunctionPass *createPeepholeOptsPass(TargetMachine &Target);


//---------------------------------------------------------------------------
// class UltraSparc 
// 
// Purpose:
//   Primary interface to machine description for the UltraSPARC.
//   Primarily just initializes machine-dependent parameters in
//   class TargetMachine, and creates machine-dependent subclasses
//   for classes such as InstrInfo, SchedInfo and RegInfo. 
//---------------------------------------------------------------------------

class UltraSparc : public TargetMachine {
  UltraSparcInstrInfo instrInfo;
  UltraSparcSchedInfo schedInfo;
  UltraSparcRegInfo   regInfo;
  UltraSparcFrameInfo frameInfo;
  UltraSparcCacheInfo cacheInfo;
  UltraSparcOptInfo   optInfo;
public:
  UltraSparc();

  virtual const TargetInstrInfo  &getInstrInfo() const { return instrInfo; }
  virtual const TargetSchedInfo  &getSchedInfo() const { return schedInfo; }
  virtual const TargetRegInfo    &getRegInfo()   const { return regInfo; }
  virtual const TargetFrameInfo  &getFrameInfo() const { return frameInfo; }
  virtual const TargetCacheInfo  &getCacheInfo() const { return cacheInfo; }
  virtual const TargetOptInfo    &getOptInfo()   const { return optInfo; }

  virtual bool addPassesToEmitAssembly(PassManager &PM, std::ostream &Out);
  virtual bool addPassesToJITCompile(FunctionPassManager &PM);
  virtual bool addPassesToEmitMachineCode(FunctionPassManager &PM,
                                          MachineCodeEmitter &MCE);

  // getPrologEpilogInsertionPass - Inserts prolog/epilog code.
  FunctionPass* getPrologEpilogInsertionPass();

  // getFunctionAsmPrinterPass - Writes out machine code for a single function
  Pass* getFunctionAsmPrinterPass(std::ostream &Out);

  // getModuleAsmPrinterPass - Writes generated machine code to assembly file.
  Pass* getModuleAsmPrinterPass(std::ostream &Out);

  // getEmitBytecodeToAsmPass - Emits final LLVM bytecode to assembly file.
  Pass* getEmitBytecodeToAsmPass(std::ostream &Out);
};

Pass *getFunctionInfo(std::ostream &out);

#endif
