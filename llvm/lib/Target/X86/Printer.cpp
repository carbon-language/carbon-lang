//===-- X86/Printer.cpp - Convert X86 code to human readable rep. ---------===//
//
// This file contains a printer that converts from our internal representation
// of LLVM code to a nice human readable form that is suitable for debuggging.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "llvm/Function.h"
#include "llvm/Constant.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "Support/Statistic.h"
#include "Support/hash_map"
#include "llvm/Type.h"
#include "llvm/Constants.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/DerivedTypes.h"
#include "llvm/SlotCalculator.h"
#include "Support/StringExtras.h"
#include "llvm/Module.h"

namespace {
  std::set<const Value *> MangledGlobals;
  struct Printer : public MachineFunctionPass {
    std::ostream &O;
    typedef std::map<const Value *, unsigned> ValueMapTy;
    ValueMapTy NumberForBB;
    Printer(std::ostream &o) : O(o) {}
    const TargetData *TD;
    std::string CurrentFnName;
    virtual const char *getPassName() const {
      return "X86 Assembly Printer";
    }

    void printMachineInstruction(const MachineInstr *MI, std::ostream &O,
				 const TargetMachine &TM) const;
    void printOp(std::ostream &O, const MachineOperand &MO,
		 const MRegisterInfo &RI, bool elideOffsetKeyword = false) const;
    void printMemReference(std::ostream &O, const MachineInstr *MI,
			   unsigned Op,
			   const MRegisterInfo &RI) const;
    void printConstantPool(MachineConstantPool *MCP);
    bool runOnMachineFunction(MachineFunction &F);    
    std::string ConstantExprToString(const ConstantExpr* CE);
    std::string valToExprString(const Value* V);
    bool doInitialization(Module &M);
    bool doFinalization(Module &M);
    void PrintZeroBytesToPad(int numBytes);
    void printConstantValueOnly(const Constant* CV, int numPadBytesAfter = 0);
    void printSingleConstantValue(const Constant* CV);
  };
} // end of anonymous namespace

/// createX86CodePrinterPass - Print out the specified machine code function to
/// the specified stream.  This function should work regardless of whether or
/// not the function is in SSA form or not.
///
Pass *createX86CodePrinterPass(std::ostream &O) {
  return new Printer(O);
}

// We don't want identifier names with ., space, or - in them, 
// so we replace them with underscores.
static std::string makeNameProper(std::string x) {
  std::string tmp;
  for (std::string::iterator sI = x.begin(), sEnd = x.end(); sI != sEnd; sI++)
    switch (*sI) {
    case '.': tmp += "d_"; break;
    case ' ': tmp += "s_"; break;
    case '-': tmp += "D_"; break;
    default:  tmp += *sI;
    }
  return tmp;
}

static std::string getValueName(const Value *V) {
  if (V->hasName()) { // Print out the label if it exists...
    // Name mangling occurs as follows:
    // - If V is not a global, mangling always occurs.
    // - Otherwise, mangling occurs when any of the following are true:
    //   1) V has internal linkage
    //   2) V's name would collide if it is not mangled.
    //
    if(const GlobalValue* gv = dyn_cast<GlobalValue>(V)) {
      if(!gv->hasInternalLinkage() && !MangledGlobals.count(gv)) {
        // No internal linkage, name will not collide -> no mangling.
        return makeNameProper(gv->getName());
      }
    }
    // Non-global, or global with internal linkage / colliding name -> mangle.
    return "l" + utostr(V->getType()->getUniqueID()) + "_" +
      makeNameProper(V->getName());      
  }
  static int Count = 0;
  Count++;
  return "ltmp_" + itostr(Count) + "_" + utostr(V->getType()->getUniqueID());
}

// valToExprString - Helper function for ConstantExprToString().
// Appends result to argument string S.
// 
std::string Printer::valToExprString(const Value* V) {
  std::string S;
  bool failed = false;
  if (const Constant* CV = dyn_cast<Constant>(V)) { // symbolic or known
    if (const ConstantBool *CB = dyn_cast<ConstantBool>(CV))
      S += std::string(CB == ConstantBool::True ? "1" : "0");
    else if (const ConstantSInt *CI = dyn_cast<ConstantSInt>(CV))
      S += itostr(CI->getValue());
    else if (const ConstantUInt *CI = dyn_cast<ConstantUInt>(CV))
      S += utostr(CI->getValue());
    else if (const ConstantFP *CFP = dyn_cast<ConstantFP>(CV))
      S += ftostr(CFP->getValue());
    else if (isa<ConstantPointerNull>(CV))
      S += "0";
    else if (const ConstantPointerRef *CPR = dyn_cast<ConstantPointerRef>(CV))
      S += valToExprString(CPR->getValue());
    else if (const ConstantExpr *CE = dyn_cast<ConstantExpr>(CV))
      S += ConstantExprToString(CE);
    else
      failed = true;
  } else if (const GlobalValue* GV = dyn_cast<GlobalValue>(V)) {
    S += getValueName(GV);
  }
  else
    failed = true;

  if (failed) {
    assert(0 && "Cannot convert value to string");
    S += "<illegal-value>";
  }
  return S;
}

// ConstantExprToString() - Convert a ConstantExpr to an asm expression
// and return this as a string.
std::string Printer::ConstantExprToString(const ConstantExpr* CE) {
  std::string S;
  switch(CE->getOpcode()) {
  case Instruction::GetElementPtr:
    { // generate a symbolic expression for the byte address
      const Value* ptrVal = CE->getOperand(0);
      std::vector<Value*> idxVec(CE->op_begin()+1, CE->op_end());
      S += "(" + valToExprString(ptrVal) + ") + ("
	+ utostr(TD->getIndexedOffset(ptrVal->getType(),idxVec)) + ")";
      break;
    }

  case Instruction::Cast:
    // Support only non-converting casts for now, i.e., a no-op.
    // This assertion is not a complete check.
    assert(TD->getTypeSize(CE->getType()) ==
	   TD->getTypeSize(CE->getOperand(0)->getType()));
    S += "(" + valToExprString(CE->getOperand(0)) + ")";
    break;

  case Instruction::Add:
    S += "(" + valToExprString(CE->getOperand(0)) + ") + ("
      + valToExprString(CE->getOperand(1)) + ")";
    break;

  default:
    assert(0 && "Unsupported operator in ConstantExprToString()");
    break;
  }

  return S;
}

// Print a single constant value.
void
Printer::printSingleConstantValue(const Constant* CV)
{
  assert(CV->getType() != Type::VoidTy &&
         CV->getType() != Type::TypeTy &&
         CV->getType() != Type::LabelTy &&
         "Unexpected type for Constant");
  
  assert((!isa<ConstantArray>(CV) && ! isa<ConstantStruct>(CV))
         && "Aggregate types should be handled outside this function");

  const Type *type = CV->getType();
  O << "\t";
  switch(type->getPrimitiveID())
    {
    case Type::BoolTyID: case Type::UByteTyID: case Type::SByteTyID:
      O << ".byte";
      break;
    case Type::UShortTyID: case Type::ShortTyID:
      O << ".word";
      break;
    case Type::UIntTyID: case Type::IntTyID: case Type::PointerTyID:
      O << ".long";
      break;
    case Type::ULongTyID: case Type::LongTyID:
      O << ".quad";
      break;
    case Type::FloatTyID:
      O << ".long";
      break;
    case Type::DoubleTyID:
      O << ".quad";
      break;
    case Type::ArrayTyID:
      if ((cast<ArrayType>(type)->getElementType() == Type::UByteTy) ||
	  (cast<ArrayType>(type)->getElementType() == Type::SByteTy))
	O << ".string";
      else
	assert (0 && "Can't handle printing this type of array");
      break;
    default:
      assert (0 && "Can't handle printing this type of thing");
      break;
    }
  O << "\t";
  
  if (type->isPrimitiveType())
    {
      if (type->isFloatingPoint()) {
	// FP Constants are printed as integer constants to avoid losing
	// precision...
	double Val = cast<ConstantFP>(CV)->getValue();
	if (type == Type::FloatTy) {
	  float FVal = (float)Val;
	  char *ProxyPtr = (char*)&FVal;        // Abide by C TBAA rules
	  O << *(unsigned int*)ProxyPtr;            
	} else if (type == Type::DoubleTy) {
	  char *ProxyPtr = (char*)&Val;         // Abide by C TBAA rules
	  O << *(uint64_t*)ProxyPtr;            
	} else {
	  assert(0 && "Unknown floating point type!");
	}
        
	O << "\t# " << type->getDescription() << " value: " << Val << "\n";
      } else {
	WriteAsOperand(O, CV, false, false) << "\n";
      }
    }
  else if (const ConstantPointerRef* CPR = dyn_cast<ConstantPointerRef>(CV))
    {
      // This is a constant address for a global variable or method.
      // Use the name of the variable or method as the address value.
      O << getValueName(CPR->getValue()) << "\n";
    }
  else if (isa<ConstantPointerNull>(CV))
    {
      // Null pointer value
      O << "0\n";
    }
  else if (const ConstantExpr* CE = dyn_cast<ConstantExpr>(CV))
    {
      // Constant expression built from operators, constants, and
      // symbolic addrs
      O << ConstantExprToString(CE) << "\n";
    }
  else
    {
      assert(0 && "Unknown elementary type for constant");
    }
}

// Can we treat the specified array as a string?  Only if it is an array of
// ubytes or non-negative sbytes.
//
static bool isStringCompatible(const ConstantArray *CVA) {
  const Type *ETy = cast<ArrayType>(CVA->getType())->getElementType();
  if (ETy == Type::UByteTy) return true;
  if (ETy != Type::SByteTy) return false;

  for (unsigned i = 0; i < CVA->getNumOperands(); ++i)
    if (cast<ConstantSInt>(CVA->getOperand(i))->getValue() < 0)
      return false;

  return true;
}

// toOctal - Convert the low order bits of X into an octal letter
static inline char toOctal(int X) {
  return (X&7)+'0';
}

// getAsCString - Return the specified array as a C compatible string, only if
// the predicate isStringCompatible is true.
//
static std::string getAsCString(const ConstantArray *CVA) {
  assert(isStringCompatible(CVA) && "Array is not string compatible!");

  std::string Result;
  const Type *ETy = cast<ArrayType>(CVA->getType())->getElementType();
  Result = "\"";
  for (unsigned i = 0; i < CVA->getNumOperands(); ++i) {
    unsigned char C = (ETy == Type::SByteTy) ?
      (unsigned char)cast<ConstantSInt>(CVA->getOperand(i))->getValue() :
      (unsigned char)cast<ConstantUInt>(CVA->getOperand(i))->getValue();

    if (C == '"') {
      Result += "\\\"";
    } else if (C == '\\') {
      Result += "\\\\";
    } else if (isprint(C)) {
      Result += C;
    } else {
      switch(C) {
      case '\a': Result += "\\a"; break;
      case '\b': Result += "\\b"; break;
      case '\f': Result += "\\f"; break;
      case '\n': Result += "\\n"; break;
      case '\r': Result += "\\r"; break;
      case '\t': Result += "\\t"; break;
      case '\v': Result += "\\v"; break;
      default:
        Result += '\\';
        Result += toOctal(C >> 6);
        Result += toOctal(C >> 3);
        Result += toOctal(C >> 0);
        break;
      }
    }
  }
  Result += "\"";
  return Result;
}

// Print a constant value or values (it may be an aggregate).
// Uses printSingleConstantValue() to print each individual value.
void
Printer::printConstantValueOnly(const Constant* CV,
				int numPadBytesAfter /* = 0 */)
{
  const ConstantArray *CVA = dyn_cast<ConstantArray>(CV);

  if (CVA && isStringCompatible(CVA))
    { // print the string alone and return
      O << "\t" << ".string" << "\t" << getAsCString(CVA) << "\n";
    }
  else if (CVA)
    { // Not a string.  Print the values in successive locations
      const std::vector<Use> &constValues = CVA->getValues();
      for (unsigned i=0; i < constValues.size(); i++)
        printConstantValueOnly(cast<Constant>(constValues[i].get()));
    }
  else if (const ConstantStruct *CVS = dyn_cast<ConstantStruct>(CV))
    { // Print the fields in successive locations. Pad to align if needed!
      const StructLayout *cvsLayout =
        TD->getStructLayout(CVS->getType());
      const std::vector<Use>& constValues = CVS->getValues();
      unsigned sizeSoFar = 0;
      for (unsigned i=0, N = constValues.size(); i < N; i++)
        {
          const Constant* field = cast<Constant>(constValues[i].get());

          // Check if padding is needed and insert one or more 0s.
          unsigned fieldSize = TD->getTypeSize(field->getType());
          int padSize = ((i == N-1? cvsLayout->StructSize
			  : cvsLayout->MemberOffsets[i+1])
                         - cvsLayout->MemberOffsets[i]) - fieldSize;
          sizeSoFar += (fieldSize + padSize);

          // Now print the actual field value
          printConstantValueOnly(field, padSize);
        }
      assert(sizeSoFar == cvsLayout->StructSize &&
             "Layout of constant struct may be incorrect!");
    }
  else
    printSingleConstantValue(CV);

  if (numPadBytesAfter) {
    unsigned numBytes = numPadBytesAfter;
    for ( ; numBytes >= 8; numBytes -= 8)
      printSingleConstantValue(Constant::getNullValue(Type::ULongTy));
    if (numBytes >= 4)
      {
	printSingleConstantValue(Constant::getNullValue(Type::UIntTy));
	numBytes -= 4;
      }
    while (numBytes--)
      printSingleConstantValue(Constant::getNullValue(Type::UByteTy));
  }
}

// printConstantPool - Print out any constants which have been spilled to
// memory...
void Printer::printConstantPool(MachineConstantPool *MCP){
  const std::vector<Constant*> &CP = MCP->getConstants();
  if (CP.empty()) return;

  for (unsigned i = 0, e = CP.size(); i != e; ++i) {
    O << "\t.section .rodata\n";
    O << "\t.align " << (unsigned)TD->getTypeAlignment(CP[i]->getType())
      << "\n";
    O << ".CPI" << CurrentFnName << "_" << i << ":\t\t\t\t\t#"
      << *CP[i] << "\n";
    printConstantValueOnly (CP[i]);
  }
}

/// runOnMachineFunction - This uses the X86InstructionInfo::print method
/// to print assembly for each instruction.
bool Printer::runOnMachineFunction(MachineFunction &MF) {
  static unsigned BBNumber = 0;
  const TargetMachine &TM = MF.getTarget();
  const TargetInstrInfo &TII = TM.getInstrInfo();
  TD = &TM.getTargetData();

  // What's my mangled name?
  CurrentFnName = getValueName(MF.getFunction());

  // Print out constants referenced by the function
  printConstantPool(MF.getConstantPool());

  // Print out labels for the function.
  O << "\t.text\n";
  O << "\t.align 16\n";
  O << "\t.globl\t" << CurrentFnName << "\n";
  O << "\t.type\t" << CurrentFnName << ", @function\n";
  O << CurrentFnName << ":\n";

  // Number each basic block so that we can consistently refer to them
  // in PC-relative references.
  NumberForBB.clear();
  for (MachineFunction::const_iterator I = MF.begin(), E = MF.end();
       I != E; ++I) {
    NumberForBB[I->getBasicBlock()] = BBNumber++;
  }

  // Print out code for the function.
  for (MachineFunction::const_iterator I = MF.begin(), E = MF.end();
       I != E; ++I) {
    // Print a label for the basic block.
    O << ".BB" << NumberForBB[I->getBasicBlock()] << ":\t# "
      << I->getBasicBlock()->getName() << "\n";
    for (MachineBasicBlock::const_iterator II = I->begin(), E = I->end();
	 II != E; ++II) {
      // Print the assembly for the instruction.
      O << "\t";
      printMachineInstruction(*II, O, TM);
    }
  }

  // We didn't modify anything.
  return false;
}

static bool isScale(const MachineOperand &MO) {
  return MO.isImmediate() &&
    (MO.getImmedValue() == 1 || MO.getImmedValue() == 2 ||
     MO.getImmedValue() == 4 || MO.getImmedValue() == 8);
}

static bool isMem(const MachineInstr *MI, unsigned Op) {
  if (MI->getOperand(Op).isFrameIndex()) return true;
  if (MI->getOperand(Op).isConstantPoolIndex()) return true;
  return Op+4 <= MI->getNumOperands() &&
    MI->getOperand(Op  ).isRegister() &&isScale(MI->getOperand(Op+1)) &&
    MI->getOperand(Op+2).isRegister() &&MI->getOperand(Op+3).isImmediate();
}

void Printer::printOp(std::ostream &O, const MachineOperand &MO,
		      const MRegisterInfo &RI,
		      bool elideOffsetKeyword /* = false */) const {
  switch (MO.getType()) {
  case MachineOperand::MO_VirtualRegister:
    if (Value *V = MO.getVRegValueOrNull()) {
      O << "<" << V->getName() << ">";
      return;
    }
    // FALLTHROUGH
  case MachineOperand::MO_MachineRegister:
    if (MO.getReg() < MRegisterInfo::FirstVirtualRegister)
      O << RI.get(MO.getReg()).Name;
    else
      O << "%reg" << MO.getReg();
    return;

  case MachineOperand::MO_SignExtendedImmed:
  case MachineOperand::MO_UnextendedImmed:
    O << (int)MO.getImmedValue();
    return;
  case MachineOperand::MO_PCRelativeDisp:
    {
      ValueMapTy::const_iterator i = NumberForBB.find(MO.getVRegValue());
      assert (i != NumberForBB.end()
	      && "Could not find a BB I previously put in the NumberForBB map!");
      O << ".BB" << i->second << " # PC rel: " << MO.getVRegValue()->getName();
    }
    return;
  case MachineOperand::MO_GlobalAddress:
    if (!elideOffsetKeyword) O << "OFFSET "; O << getValueName(MO.getGlobal());
    return;
  case MachineOperand::MO_ExternalSymbol:
    O << MO.getSymbolName();
    return;
  default:
    O << "<unknown operand type>"; return;    
  }
}

static const std::string sizePtr(const TargetInstrDescriptor &Desc) {
  switch (Desc.TSFlags & X86II::ArgMask) {
  default: assert(0 && "Unknown arg size!");
  case X86II::Arg8:   return "BYTE PTR"; 
  case X86II::Arg16:  return "WORD PTR"; 
  case X86II::Arg32:  return "DWORD PTR"; 
  case X86II::Arg64:  return "QWORD PTR"; 
  case X86II::ArgF32:  return "DWORD PTR"; 
  case X86II::ArgF64:  return "QWORD PTR"; 
  case X86II::ArgF80:  return "XWORD PTR"; 
  }
}

void Printer::printMemReference(std::ostream &O, const MachineInstr *MI,
				unsigned Op,
				const MRegisterInfo &RI) const {
  assert(isMem(MI, Op) && "Invalid memory reference!");

  if (MI->getOperand(Op).isFrameIndex()) {
    O << "[frame slot #" << MI->getOperand(Op).getFrameIndex();
    if (MI->getOperand(Op+3).getImmedValue())
      O << " + " << MI->getOperand(Op+3).getImmedValue();
    O << "]";
    return;
  } else if (MI->getOperand(Op).isConstantPoolIndex()) {
    O << "[.CPI" << CurrentFnName << "_"
      << MI->getOperand(Op).getConstantPoolIndex();
    if (MI->getOperand(Op+3).getImmedValue())
      O << " + " << MI->getOperand(Op+3).getImmedValue();
    O << "]";
    return;
  }

  const MachineOperand &BaseReg  = MI->getOperand(Op);
  int ScaleVal                   = MI->getOperand(Op+1).getImmedValue();
  const MachineOperand &IndexReg = MI->getOperand(Op+2);
  int DispVal                    = MI->getOperand(Op+3).getImmedValue();

  O << "[";
  bool NeedPlus = false;
  if (BaseReg.getReg()) {
    printOp(O, BaseReg, RI);
    NeedPlus = true;
  }

  if (IndexReg.getReg()) {
    if (NeedPlus) O << " + ";
    if (ScaleVal != 1)
      O << ScaleVal << "*";
    printOp(O, IndexReg, RI);
    NeedPlus = true;
  }

  if (DispVal) {
    if (NeedPlus)
      if (DispVal > 0)
	O << " + ";
      else {
	O << " - ";
	DispVal = -DispVal;
      }
    O << DispVal;
  }
  O << "]";
}

/// printMachineInstruction -- Print out an x86 instruction in intel syntax
///
void Printer::printMachineInstruction(const MachineInstr *MI, std::ostream &O,
				      const TargetMachine &TM) const {
  unsigned Opcode = MI->getOpcode();
  const TargetInstrInfo &TII = TM.getInstrInfo();
  const TargetInstrDescriptor &Desc = TII.get(Opcode);
  const MRegisterInfo &RI = *TM.getRegisterInfo();

  switch (Desc.TSFlags & X86II::FormMask) {
  case X86II::Pseudo:
    // Print pseudo-instructions as comments; either they should have been
    // turned into real instructions by now, or they don't need to be
    // seen by the assembler (e.g., IMPLICIT_USEs.)
    O << "# ";
    if (Opcode == X86::PHI) {
      printOp(O, MI->getOperand(0), RI);
      O << " = phi ";
      for (unsigned i = 1, e = MI->getNumOperands(); i != e; i+=2) {
	if (i != 1) O << ", ";
	O << "[";
	printOp(O, MI->getOperand(i), RI);
	O << ", ";
	printOp(O, MI->getOperand(i+1), RI);
	O << "]";
      }
    } else {
      unsigned i = 0;
      if (MI->getNumOperands() && (MI->getOperand(0).opIsDefOnly() || 
                                   MI->getOperand(0).opIsDefAndUse())) {
	printOp(O, MI->getOperand(0), RI);
	O << " = ";
	++i;
      }
      O << TII.getName(MI->getOpcode());

      for (unsigned e = MI->getNumOperands(); i != e; ++i) {
	O << " ";
	if (MI->getOperand(i).opIsDefOnly() || 
            MI->getOperand(i).opIsDefAndUse()) O << "*";
	printOp(O, MI->getOperand(i), RI);
	if (MI->getOperand(i).opIsDefOnly() || 
            MI->getOperand(i).opIsDefAndUse()) O << "*";
      }
    }
    O << "\n";
    return;

  case X86II::RawFrm:
    // The accepted forms of Raw instructions are:
    //   1. nop     - No operand required
    //   2. jmp foo - PC relative displacement operand
    //   3. call bar - GlobalAddress Operand or External Symbol Operand
    //
    assert(MI->getNumOperands() == 0 ||
           (MI->getNumOperands() == 1 &&
	    (MI->getOperand(0).isPCRelativeDisp() ||
	     MI->getOperand(0).isGlobalAddress() ||
	     MI->getOperand(0).isExternalSymbol())) &&
           "Illegal raw instruction!");
    O << TII.getName(MI->getOpcode()) << " ";

    if (MI->getNumOperands() == 1) {
      printOp(O, MI->getOperand(0), RI, true); // Don't print "OFFSET"...
    }
    O << "\n";
    return;

  case X86II::AddRegFrm: {
    // There are currently two forms of acceptable AddRegFrm instructions.
    // Either the instruction JUST takes a single register (like inc, dec, etc),
    // or it takes a register and an immediate of the same size as the register
    // (move immediate f.e.).  Note that this immediate value might be stored as
    // an LLVM value, to represent, for example, loading the address of a global
    // into a register.  The initial register might be duplicated if this is a
    // M_2_ADDR_REG instruction
    //
    assert(MI->getOperand(0).isRegister() &&
           (MI->getNumOperands() == 1 || 
            (MI->getNumOperands() == 2 &&
             (MI->getOperand(1).getVRegValueOrNull() ||
              MI->getOperand(1).isImmediate() ||
	      MI->getOperand(1).isRegister() ||
	      MI->getOperand(1).isGlobalAddress() ||
	      MI->getOperand(1).isExternalSymbol()))) &&
           "Illegal form for AddRegFrm instruction!");

    unsigned Reg = MI->getOperand(0).getReg();
    
    O << TII.getName(MI->getOpCode()) << " ";
    printOp(O, MI->getOperand(0), RI);
    if (MI->getNumOperands() == 2 &&
	(!MI->getOperand(1).isRegister() ||
	 MI->getOperand(1).getVRegValueOrNull() ||
	 MI->getOperand(1).isGlobalAddress() ||
	 MI->getOperand(1).isExternalSymbol())) {
      O << ", ";
      printOp(O, MI->getOperand(1), RI);
    }
    O << "\n";
    return;
  }
  case X86II::MRMDestReg: {
    // There are two acceptable forms of MRMDestReg instructions, those with 2,
    // 3 and 4 operands:
    //
    // 2 Operands: this is for things like mov that do not read a second input
    //
    // 3 Operands: in this form, the first two registers (the destination, and
    // the first operand) should be the same, post register allocation.  The 3rd
    // operand is an additional input.  This should be for things like add
    // instructions.
    //
    // 4 Operands: This form is for instructions which are 3 operands forms, but
    // have a constant argument as well.
    //
    bool isTwoAddr = TII.isTwoAddrInstr(Opcode);
    assert(MI->getOperand(0).isRegister() &&
           (MI->getNumOperands() == 2 ||
	    (isTwoAddr && MI->getOperand(1).isRegister() &&
	     MI->getOperand(0).getReg() == MI->getOperand(1).getReg() &&
	     (MI->getNumOperands() == 3 ||
	      (MI->getNumOperands() == 4 && MI->getOperand(3).isImmediate()))))
           && "Bad format for MRMDestReg!");

    O << TII.getName(MI->getOpCode()) << " ";
    printOp(O, MI->getOperand(0), RI);
    O << ", ";
    printOp(O, MI->getOperand(1+isTwoAddr), RI);
    if (MI->getNumOperands() == 4) {
      O << ", ";
      printOp(O, MI->getOperand(3), RI);
    }
    O << "\n";
    return;
  }

  case X86II::MRMDestMem: {
    // These instructions are the same as MRMDestReg, but instead of having a
    // register reference for the mod/rm field, it's a memory reference.
    //
    assert(isMem(MI, 0) && MI->getNumOperands() == 4+1 &&
           MI->getOperand(4).isRegister() && "Bad format for MRMDestMem!");

    O << TII.getName(MI->getOpCode()) << " " << sizePtr(Desc) << " ";
    printMemReference(O, MI, 0, RI);
    O << ", ";
    printOp(O, MI->getOperand(4), RI);
    O << "\n";
    return;
  }

  case X86II::MRMSrcReg: {
    // There is a two forms that are acceptable for MRMSrcReg instructions,
    // those with 3 and 2 operands:
    //
    // 3 Operands: in this form, the last register (the second input) is the
    // ModR/M input.  The first two operands should be the same, post register
    // allocation.  This is for things like: add r32, r/m32
    //
    // 2 Operands: this is for things like mov that do not read a second input
    //
    assert(MI->getOperand(0).isRegister() &&
           MI->getOperand(1).isRegister() &&
           (MI->getNumOperands() == 2 || 
            (MI->getNumOperands() == 3 && MI->getOperand(2).isRegister()))
           && "Bad format for MRMSrcReg!");
    if (MI->getNumOperands() == 3 &&
        MI->getOperand(0).getReg() != MI->getOperand(1).getReg())
      O << "**";

    O << TII.getName(MI->getOpCode()) << " ";
    printOp(O, MI->getOperand(0), RI);
    O << ", ";
    printOp(O, MI->getOperand(MI->getNumOperands()-1), RI);
    O << "\n";
    return;
  }

  case X86II::MRMSrcMem: {
    // These instructions are the same as MRMSrcReg, but instead of having a
    // register reference for the mod/rm field, it's a memory reference.
    //
    assert(MI->getOperand(0).isRegister() &&
           (MI->getNumOperands() == 1+4 && isMem(MI, 1)) || 
           (MI->getNumOperands() == 2+4 && MI->getOperand(1).isRegister() && 
            isMem(MI, 2))
           && "Bad format for MRMDestReg!");
    if (MI->getNumOperands() == 2+4 &&
        MI->getOperand(0).getReg() != MI->getOperand(1).getReg())
      O << "**";

    O << TII.getName(MI->getOpCode()) << " ";
    printOp(O, MI->getOperand(0), RI);
    O << ", " << sizePtr(Desc) << " ";
    printMemReference(O, MI, MI->getNumOperands()-4, RI);
    O << "\n";
    return;
  }

  case X86II::MRMS0r: case X86II::MRMS1r:
  case X86II::MRMS2r: case X86II::MRMS3r:
  case X86II::MRMS4r: case X86II::MRMS5r:
  case X86II::MRMS6r: case X86II::MRMS7r: {
    // In this form, the following are valid formats:
    //  1. sete r
    //  2. cmp reg, immediate
    //  2. shl rdest, rinput  <implicit CL or 1>
    //  3. sbb rdest, rinput, immediate   [rdest = rinput]
    //    
    assert(MI->getNumOperands() > 0 && MI->getNumOperands() < 4 &&
           MI->getOperand(0).isRegister() && "Bad MRMSxR format!");
    assert((MI->getNumOperands() != 2 ||
            MI->getOperand(1).isRegister() || MI->getOperand(1).isImmediate())&&
           "Bad MRMSxR format!");
    assert((MI->getNumOperands() < 3 ||
	    (MI->getOperand(1).isRegister() && MI->getOperand(2).isImmediate())) &&
           "Bad MRMSxR format!");

    if (MI->getNumOperands() > 1 && MI->getOperand(1).isRegister() && 
        MI->getOperand(0).getReg() != MI->getOperand(1).getReg())
      O << "**";

    O << TII.getName(MI->getOpCode()) << " ";
    printOp(O, MI->getOperand(0), RI);
    if (MI->getOperand(MI->getNumOperands()-1).isImmediate()) {
      O << ", ";
      printOp(O, MI->getOperand(MI->getNumOperands()-1), RI);
    }
    if (Desc.TSFlags & X86II::PrintImplUses) {
      for (const unsigned *p = Desc.ImplicitUses; *p; ++p) {
	O << ", " << RI.get(*p).Name;
      }
    }
    O << "\n";

    return;
  }

  case X86II::MRMS0m: case X86II::MRMS1m:
  case X86II::MRMS2m: case X86II::MRMS3m:
  case X86II::MRMS4m: case X86II::MRMS5m:
  case X86II::MRMS6m: case X86II::MRMS7m: {
    // In this form, the following are valid formats:
    //  1. sete [m]
    //  2. cmp [m], immediate
    //  2. shl [m], rinput  <implicit CL or 1>
    //  3. sbb [m], immediate
    //    
    assert(MI->getNumOperands() >= 4 && MI->getNumOperands() <= 5 &&
           isMem(MI, 0) && "Bad MRMSxM format!");
    assert((MI->getNumOperands() != 5 || MI->getOperand(4).isImmediate()) &&
           "Bad MRMSxM format!");

    O << TII.getName(MI->getOpCode()) << " ";
    O << sizePtr(Desc) << " ";
    printMemReference(O, MI, 0, RI);
    if (MI->getNumOperands() == 5) {
      O << ", ";
      printOp(O, MI->getOperand(4), RI);
    }
    O << "\n";
    return;
  }

  default:
    O << "\tUNKNOWN FORM:\t\t-"; MI->print(O, TM); break;
  }
}

bool Printer::doInitialization(Module &M)
{
  // Tell gas we are outputting Intel syntax (not AT&T syntax) assembly,
  // with no % decorations on register names.
  O << "\t.intel_syntax noprefix\n";

  // Ripped from CWriter:
  // Calculate which global values have names that will collide when we throw
  // away type information.
  {  // Scope to delete the FoundNames set when we are done with it...
    std::set<std::string> FoundNames;
    for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
      if (I->hasName())                      // If the global has a name...
        if (FoundNames.count(I->getName()))  // And the name is already used
          MangledGlobals.insert(I);          // Mangle the name
        else
          FoundNames.insert(I->getName());   // Otherwise, keep track of name

    for (Module::giterator I = M.gbegin(), E = M.gend(); I != E; ++I)
      if (I->hasName())                      // If the global has a name...
        if (FoundNames.count(I->getName()))  // And the name is already used
          MangledGlobals.insert(I);          // Mangle the name
        else
          FoundNames.insert(I->getName());   // Otherwise, keep track of name
  }

  return false; // success
}

bool Printer::doFinalization(Module &M)
{
  // Print out module-level global variables here.
  for (Module::const_giterator I = M.gbegin(), E = M.gend(); I != E; ++I) {
    std::string name(getValueName(I));
    if (I->hasInitializer()) {
      Constant *C = I->getInitializer();
      O << "\t.data\n";
      O << "\t.globl " << name << "\n";
      O << "\t.type " << name << ",@object\n";
      O << "\t.size " << name << ","
	<< (unsigned)TD->getTypeSize(I->getType()) << "\n";
      O << "\t.align " << (unsigned)TD->getTypeAlignment(C->getType()) << "\n";
      O << name << ":\t\t\t\t\t#" << *C << "\n";
      printConstantValueOnly (C);
    } else {
      O << "\t.globl " << name << "\n";
      O << "\t.comm " << name << ", "
        << (unsigned)TD->getTypeSize(I->getType()) << ", "
        << (unsigned)TD->getTypeAlignment(I->getType()) << "\n";
    }
  }
  MangledGlobals.clear();
  return false; // success
}


