//===-- Printer.cpp - Convert LLVM code to PowerPC assembly ---------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file contains a printer that converts from our internal representation
// of machine-dependent LLVM code to PowerPC assembly language. This printer is
// the output mechanism used by `llc' and `lli -print-machineinstrs'.
//
// Documentation at http://developer.apple.com/documentation/DeveloperTools/
// Reference/Assembler/ASMIntroduction/chapter_1_section_1.html
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "asmprinter"
#include "PowerPC.h"
#include "PowerPCInstrInfo.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/Mangler.h"
#include "Support/CommandLine.h"
#include "Support/Debug.h"
#include "Support/Statistic.h"
#include "Support/StringExtras.h"
#include <set>

namespace llvm {

namespace {
  Statistic<> EmittedInsts("asm-printer", "Number of machine instrs printed");

  struct Printer : public MachineFunctionPass {
    /// Output stream on which we're printing assembly code.
    ///
    std::ostream &O;

    /// Target machine description which we query for reg. names, data
    /// layout, etc.
    ///
    TargetMachine &TM;

    /// Name-mangler for global names.
    ///
    Mangler *Mang;
    std::set<std::string> FnStubs, GVStubs, LinkOnceStubs;
    std::set<std::string> Strings;

    Printer(std::ostream &o, TargetMachine &tm) : O(o), TM(tm), labelNumber(0)
      { }

    /// Cache of mangled name for current function. This is
    /// recalculated at the beginning of each call to
    /// runOnMachineFunction().
    ///
    std::string CurrentFnName;

    /// Unique incrementer for label values for referencing
    /// Global values.
    ///
    unsigned int labelNumber;

    virtual const char *getPassName() const {
      return "PowerPC Assembly Printer";
    }

    void printMachineInstruction(const MachineInstr *MI);
    void printOp(const MachineOperand &MO, bool elideOffsetKeyword = false);
    void printConstantPool(MachineConstantPool *MCP);
    bool runOnMachineFunction(MachineFunction &F);    
    bool doInitialization(Module &M);
    bool doFinalization(Module &M);
    void emitGlobalConstant(const Constant* CV);
    void emitConstantValueOnly(const Constant *CV);
  };
} // end of anonymous namespace

/// createPPCCodePrinterPass - Returns a pass that prints the PPC
/// assembly code for a MachineFunction to the given output stream,
/// using the given target machine description.  This should work
/// regardless of whether the function is in SSA form.
///
FunctionPass *createPPCCodePrinterPass(std::ostream &o,TargetMachine &tm) {
  return new Printer(o, tm);
}

/// isStringCompatible - Can we treat the specified array as a string?
/// Only if it is an array of ubytes or non-negative sbytes.
///
static bool isStringCompatible(const ConstantArray *CVA) {
  const Type *ETy = cast<ArrayType>(CVA->getType())->getElementType();
  if (ETy == Type::UByteTy) return true;
  if (ETy != Type::SByteTy) return false;

  for (unsigned i = 0; i < CVA->getNumOperands(); ++i)
    if (cast<ConstantSInt>(CVA->getOperand(i))->getValue() < 0)
      return false;

  return true;
}

/// toOctal - Convert the low order bits of X into an octal digit.
///
static inline char toOctal(int X) {
  return (X&7)+'0';
}

/// getAsCString - Return the specified array as a C compatible
/// string, only if the predicate isStringCompatible is true.
///
static void printAsCString(std::ostream &O, const ConstantArray *CVA) {
  assert(isStringCompatible(CVA) && "Array is not string compatible!");

  O << "\"";
  for (unsigned i = 0; i < CVA->getNumOperands(); ++i) {
    unsigned char C = cast<ConstantInt>(CVA->getOperand(i))->getRawValue();

    if (C == '"') {
      O << "\\\"";
    } else if (C == '\\') {
      O << "\\\\";
    } else if (isprint(C)) {
      O << C;
    } else {
      switch(C) {
      case '\b': O << "\\b"; break;
      case '\f': O << "\\f"; break;
      case '\n': O << "\\n"; break;
      case '\r': O << "\\r"; break;
      case '\t': O << "\\t"; break;
      default:
        O << '\\';
        O << toOctal(C >> 6);
        O << toOctal(C >> 3);
        O << toOctal(C >> 0);
        break;
      }
    }
  }
  O << "\"";
}

// Print out the specified constant, without a storage class.  Only the
// constants valid in constant expressions can occur here.
void Printer::emitConstantValueOnly(const Constant *CV) {
  if (CV->isNullValue())
    O << "0";
  else if (const ConstantBool *CB = dyn_cast<ConstantBool>(CV)) {
    assert(CB == ConstantBool::True);
    O << "1";
  } else if (const ConstantSInt *CI = dyn_cast<ConstantSInt>(CV))
    O << CI->getValue();
  else if (const ConstantUInt *CI = dyn_cast<ConstantUInt>(CV))
    O << CI->getValue();
  else if (const GlobalValue *GV = dyn_cast<GlobalValue>(CV))
    // This is a constant address for a global variable or function.  Use the
    // name of the variable or function as the address value.
    O << Mang->getValueName(GV);
  else if (const ConstantExpr *CE = dyn_cast<ConstantExpr>(CV)) {
    const TargetData &TD = TM.getTargetData();
    switch(CE->getOpcode()) {
    case Instruction::GetElementPtr: {
      // generate a symbolic expression for the byte address
      const Constant *ptrVal = CE->getOperand(0);
      std::vector<Value*> idxVec(CE->op_begin()+1, CE->op_end());
      if (unsigned Offset = TD.getIndexedOffset(ptrVal->getType(), idxVec)) {
        O << "(";
        emitConstantValueOnly(ptrVal);
        O << ") + " << Offset;
      } else {
        emitConstantValueOnly(ptrVal);
      }
      break;
    }
    case Instruction::Cast: {
      // Support only non-converting or widening casts for now, that is, ones
      // that do not involve a change in value.  This assertion is really gross,
      // and may not even be a complete check.
      Constant *Op = CE->getOperand(0);
      const Type *OpTy = Op->getType(), *Ty = CE->getType();

      // Remember, kids, pointers on x86 can be losslessly converted back and
      // forth into 32-bit or wider integers, regardless of signedness. :-P
      assert(((isa<PointerType>(OpTy)
               && (Ty == Type::LongTy || Ty == Type::ULongTy
                   || Ty == Type::IntTy || Ty == Type::UIntTy))
              || (isa<PointerType>(Ty)
                  && (OpTy == Type::LongTy || OpTy == Type::ULongTy
                      || OpTy == Type::IntTy || OpTy == Type::UIntTy))
              || (((TD.getTypeSize(Ty) >= TD.getTypeSize(OpTy))
                   && OpTy->isLosslesslyConvertibleTo(Ty))))
             && "FIXME: Don't yet support this kind of constant cast expr");
      O << "(";
      emitConstantValueOnly(Op);
      O << ")";
      break;
    }
    case Instruction::Add:
      O << "(";
      emitConstantValueOnly(CE->getOperand(0));
      O << ") + (";
      emitConstantValueOnly(CE->getOperand(1));
      O << ")";
      break;
    default:
      assert(0 && "Unsupported operator!");
    }
  } else {
    assert(0 && "Unknown constant value!");
  }
}

// Print a constant value or values, with the appropriate storage class as a
// prefix.
void Printer::emitGlobalConstant(const Constant *CV) {  
  const TargetData &TD = TM.getTargetData();

  if (const ConstantArray *CVA = dyn_cast<ConstantArray>(CV)) {
    if (isStringCompatible(CVA)) {
      O << "\t.ascii ";
      printAsCString(O, CVA);
      O << "\n";
    } else { // Not a string.  Print the values in successive locations
      const std::vector<Use> &constValues = CVA->getValues();
      for (unsigned i=0; i < constValues.size(); i++)
        emitGlobalConstant(cast<Constant>(constValues[i].get()));
    }
    return;
  } else if (const ConstantStruct *CVS = dyn_cast<ConstantStruct>(CV)) {
    // Print the fields in successive locations. Pad to align if needed!
    const StructLayout *cvsLayout = TD.getStructLayout(CVS->getType());
    const std::vector<Use>& constValues = CVS->getValues();
    unsigned sizeSoFar = 0;
    for (unsigned i=0, N = constValues.size(); i < N; i++) {
      const Constant* field = cast<Constant>(constValues[i].get());

      // Check if padding is needed and insert one or more 0s.
      unsigned fieldSize = TD.getTypeSize(field->getType());
      unsigned padSize = ((i == N-1? cvsLayout->StructSize
                           : cvsLayout->MemberOffsets[i+1])
                          - cvsLayout->MemberOffsets[i]) - fieldSize;
      sizeSoFar += fieldSize + padSize;

      // Now print the actual field value
      emitGlobalConstant(field);

      // Insert the field padding unless it's zero bytes...
      if (padSize)
        O << "\t.space\t " << padSize << "\n";      
    }
    assert(sizeSoFar == cvsLayout->StructSize &&
           "Layout of constant struct may be incorrect!");
    return;
  } else if (const ConstantFP *CFP = dyn_cast<ConstantFP>(CV)) {
    // FP Constants are printed as integer constants to avoid losing
    // precision...
    double Val = CFP->getValue();
    switch (CFP->getType()->getTypeID()) {
    default: assert(0 && "Unknown floating point type!");
    case Type::FloatTyID: {
      union FU {                            // Abide by C TBAA rules
        float FVal;
        unsigned UVal;
      } U;
      U.FVal = Val;
      O << ".long\t" << U.UVal << "\t; float " << Val << "\n";
      return;
    }
    case Type::DoubleTyID: {
      union DU {                            // Abide by C TBAA rules
        double FVal;
        uint64_t UVal;
        struct {
          uint32_t MSWord;
          uint32_t LSWord;
        } T;
      } U;
      U.FVal = Val;
      
      O << ".long\t" << U.T.MSWord << "\t; double most significant word " 
        << Val << "\n";
      O << ".long\t" << U.T.LSWord << "\t; double least significant word " 
        << Val << "\n";
      return;
    }
    }
  } else if (CV->getType()->getPrimitiveSize() == 64) {
    if (const ConstantInt *CI = dyn_cast<ConstantInt>(CV)) {
      union DU {                            // Abide by C TBAA rules
        int64_t UVal;
        struct {
          uint32_t MSWord;
          uint32_t LSWord;
        } T;
      } U;
      U.UVal = CI->getRawValue();
        
      O << ".long\t" << U.T.MSWord << "\t; Double-word most significant word " 
        << U.UVal << "\n";
      O << ".long\t" << U.T.LSWord << "\t; Double-word least significant word " 
        << U.UVal << "\n";
      return;
    }
  }

  const Type *type = CV->getType();
  O << "\t";
  switch (type->getTypeID()) {
  case Type::UByteTyID: case Type::SByteTyID:
    O << ".byte";
    break;
  case Type::UShortTyID: case Type::ShortTyID:
    O << ".short";
    break;
  case Type::BoolTyID: 
  case Type::PointerTyID:
  case Type::UIntTyID: case Type::IntTyID:
    O << ".long";
    break;
  case Type::ULongTyID: case Type::LongTyID:    
    assert (0 && "Should have already output double-word constant.");
  case Type::FloatTyID: case Type::DoubleTyID:
    assert (0 && "Should have already output floating point constant.");
  default:
    if (CV == Constant::getNullValue(type)) {  // Zero initializer?
      O << ".space\t" << TD.getTypeSize(type) << "\n";      
      return;
    }
    std::cerr << "Can't handle printing: " << *CV;
    abort();
    break;
  }
  O << "\t";
  emitConstantValueOnly(CV);
  O << "\n";
}

/// printConstantPool - Print to the current output stream assembly
/// representations of the constants in the constant pool MCP. This is
/// used to print out constants which have been "spilled to memory" by
/// the code generator.
///
void Printer::printConstantPool(MachineConstantPool *MCP) {
  const std::vector<Constant*> &CP = MCP->getConstants();
  const TargetData &TD = TM.getTargetData();
 
  if (CP.empty()) return;

  for (unsigned i = 0, e = CP.size(); i != e; ++i) {
    O << "\t.const\n";
    O << "\t.align " << (unsigned)TD.getTypeAlignment(CP[i]->getType())
      << "\n";
    O << ".CPI" << CurrentFnName << "_" << i << ":\t\t\t\t\t;"
      << *CP[i] << "\n";
    emitGlobalConstant(CP[i]);
  }
}

/// runOnMachineFunction - This uses the printMachineInstruction()
/// method to print assembly for each instruction.
///
bool Printer::runOnMachineFunction(MachineFunction &MF) {
  O << "\n\n";
  // What's my mangled name?
  CurrentFnName = Mang->getValueName(MF.getFunction());

  // Print out constants referenced by the function
  printConstantPool(MF.getConstantPool());

  // Print out labels for the function.
  O << "\t.text\n"; 
  O << "\t.globl\t" << CurrentFnName << "\n";
  O << "\t.align 2\n";
  O << CurrentFnName << ":\n";

  // Print out code for the function.
  for (MachineFunction::const_iterator I = MF.begin(), E = MF.end();
       I != E; ++I) {
    // Print a label for the basic block.
    O << ".LBB" << CurrentFnName << "_" << I->getNumber() << ":\t; "
      << I->getBasicBlock()->getName() << "\n";
    for (MachineBasicBlock::const_iterator II = I->begin(), E = I->end();
      II != E; ++II) {
      // Print the assembly for the instruction.
      O << "\t";
      printMachineInstruction(II);
    }
  }

  // We didn't modify anything.
  return false;
}

void Printer::printOp(const MachineOperand &MO,
                      bool elideOffsetKeyword /* = false */) {
  const MRegisterInfo &RI = *TM.getRegisterInfo();
  int new_symbol;
  
  switch (MO.getType()) {
  case MachineOperand::MO_VirtualRegister:
    if (Value *V = MO.getVRegValueOrNull()) {
      O << "<" << V->getName() << ">";
      return;
    }
    // FALLTHROUGH
  case MachineOperand::MO_MachineRegister:
  case MachineOperand::MO_CCRegister:
    O << LowercaseString(RI.get(MO.getReg()).Name);
    return;

  case MachineOperand::MO_SignExtendedImmed:
    O << (short)MO.getImmedValue();
    return;

  case MachineOperand::MO_UnextendedImmed:
    O << (unsigned short)MO.getImmedValue();
    return;
    
  case MachineOperand::MO_PCRelativeDisp:
    std::cerr << "Shouldn't use addPCDisp() when building PPC MachineInstrs";
    abort();
    return;
    
  case MachineOperand::MO_MachineBasicBlock: {
    MachineBasicBlock *MBBOp = MO.getMachineBasicBlock();
    O << ".LBB" << Mang->getValueName(MBBOp->getParent()->getFunction())
      << "_" << MBBOp->getNumber() << "\t; "
      << MBBOp->getBasicBlock()->getName();
    return;
  }

  case MachineOperand::MO_ConstantPoolIndex:
    O << ".CPI" << CurrentFnName << "_" << MO.getConstantPoolIndex();
    return;

  case MachineOperand::MO_ExternalSymbol:
    O << MO.getSymbolName();
    return;

  case MachineOperand::MO_GlobalAddress:
    if (!elideOffsetKeyword) {
      GlobalValue *GV = MO.getGlobal();
      std::string Name = Mang->getValueName(GV);
      // Dynamically-resolved functions need a stub for the function
      Function *F = dyn_cast<Function>(GV);
      if (F && F->isExternal()) {
        FnStubs.insert(Name);
        O << "L" << Name << "$stub";
      } else {
        GlobalVariable *GVar = dyn_cast<GlobalVariable>(GV);
        // External global variables need a non-lazily-resolved stub
        if (GVar && GVar->isExternal()) {
          GVStubs.insert(Name);
          O << "L" << Name << "$non_lazy_ptr";
        } else 
          O << Mang->getValueName(GV);
      }
    }
    return;
    
  default:
    O << "<unknown operand type: " << MO.getType() << ">";
    return;
  }
}

/// printMachineInstruction -- Print out a single PPC32 LLVM instruction
/// MI in Darwin syntax to the current output stream.
///
void Printer::printMachineInstruction(const MachineInstr *MI) {
  unsigned Opcode = MI->getOpcode();
  const TargetInstrInfo &TII = *TM.getInstrInfo();
  const TargetInstrDescriptor &Desc = TII.get(Opcode);
  unsigned int i;

  unsigned int ArgCount = MI->getNumOperands();
    //Desc.TSFlags & PPC32II::ArgCountMask;
  unsigned int ArgType[] = {
    (Desc.TSFlags >> PPC32II::Arg0TypeShift) & PPC32II::ArgTypeMask,
    (Desc.TSFlags >> PPC32II::Arg1TypeShift) & PPC32II::ArgTypeMask,
    (Desc.TSFlags >> PPC32II::Arg2TypeShift) & PPC32II::ArgTypeMask,
    (Desc.TSFlags >> PPC32II::Arg3TypeShift) & PPC32II::ArgTypeMask,
    (Desc.TSFlags >> PPC32II::Arg4TypeShift) & PPC32II::ArgTypeMask
  };
  assert(((Desc.TSFlags & PPC32II::VMX) == 0) &&
         "Instruction requires VMX support");
  assert(((Desc.TSFlags & PPC32II::PPC64) == 0) &&
         "Instruction requires 64 bit support");
  ++EmittedInsts;

  // CALLpcrel and CALLindirect are handled specially here to print only the
  // appropriate number of args that the assembler expects.  This is because
  // may have many arguments appended to record the uses of registers that are
  // holding arguments to the called function.
  if (Opcode == PPC32::IMPLICIT_DEF) {
    O << "; IMPLICIT DEF ";
    printOp(MI->getOperand(0));
    O << "\n";
    return;
  } else if (Opcode == PPC32::CALLpcrel) {
    O << TII.getName(MI->getOpcode()) << " ";
    printOp(MI->getOperand(0));
    O << "\n";
    return;
  } else if (Opcode == PPC32::CALLindirect) {
    O << TII.getName(MI->getOpcode()) << " ";
    printOp(MI->getOperand(0));
    O << ", ";
    printOp(MI->getOperand(1));
    O << "\n";
    return;
  } else if (Opcode == PPC32::MovePCtoLR) {
    // FIXME: should probably be converted to cout.width and cout.fill
    O << "bl \"L0000" << labelNumber << "$pb\"\n";
    O << "\"L0000" << labelNumber << "$pb\":\n";
    O << "\tmflr ";
    printOp(MI->getOperand(0));
    O << "\n";
    return;
  }

  O << TII.getName(MI->getOpcode()) << " ";
  if (Opcode == PPC32::LOADLoDirect || Opcode == PPC32::LOADLoIndirect) {
    printOp(MI->getOperand(0));
    O << ", lo16(";
    printOp(MI->getOperand(2));
    O << "-\"L0000" << labelNumber << "$pb\")";
    labelNumber++;
    O << "(";
    if (MI->getOperand(1).getReg() == PPC32::R0)
      O << "0";
    else
      printOp(MI->getOperand(1));
    O << ")\n";
  } else if (Opcode == PPC32::LOADHiAddr) {
    printOp(MI->getOperand(0));
    O << ", ";
    if (MI->getOperand(1).getReg() == PPC32::R0)
      O << "0";
    else
      printOp(MI->getOperand(1));
    O << ", ha16(" ;
    printOp(MI->getOperand(2));
     O << "-\"L0000" << labelNumber << "$pb\")\n";
  } else if (ArgCount == 3 && ArgType[1] == PPC32II::Disimm16) {
    printOp(MI->getOperand(0));
    O << ", ";
    printOp(MI->getOperand(1));
    O << "(";
    if (MI->getOperand(2).hasAllocatedReg() &&
        MI->getOperand(2).getReg() == PPC32::R0)
      O << "0";
    else
      printOp(MI->getOperand(2));
    O << ")\n";
  } else {
    for (i = 0; i < ArgCount; ++i) {
      if (i == 1 && ArgCount == 3 && ArgType[2] == PPC32II::Simm16 &&
          MI->getOperand(1).hasAllocatedReg() && 
          MI->getOperand(1).getReg() == PPC32::R0) {
        O << "0";
      } else {
        printOp(MI->getOperand(i));
      }
      if (ArgCount - 1 == i)
        O << "\n";
      else
        O << ", ";
    }
  }
}

bool Printer::doInitialization(Module &M) {
  Mang = new Mangler(M, true);
  return false; // success
}

// SwitchSection - Switch to the specified section of the executable if we are
// not already in it!
//
static void SwitchSection(std::ostream &OS, std::string &CurSection,
                          const char *NewSection) {
  if (CurSection != NewSection) {
    CurSection = NewSection;
    if (!CurSection.empty())
      OS << "\t" << NewSection << "\n";
  }
}

bool Printer::doFinalization(Module &M) {
  const TargetData &TD = TM.getTargetData();
  std::string CurSection;

  // Print out module-level global variables here.
  for (Module::const_giterator I = M.gbegin(), E = M.gend(); I != E; ++I)
    if (I->hasInitializer()) {   // External global require no code
      O << "\n\n";
      std::string name = Mang->getValueName(I);
      Constant *C = I->getInitializer();
      unsigned Size = TD.getTypeSize(C->getType());
      unsigned Align = TD.getTypeAlignment(C->getType());

      if (C->isNullValue() && /* FIXME: Verify correct */
          (I->hasInternalLinkage() || I->hasWeakLinkage())) {
        SwitchSection(O, CurSection, ".data");
        if (I->hasInternalLinkage())
          O << "\t.lcomm " << name << "," << TD.getTypeSize(C->getType())
            << "," << (unsigned)TD.getTypeAlignment(C->getType());
        else 
          O << "\t.comm " << name << "," << TD.getTypeSize(C->getType());
        O << "\t\t; ";
        WriteAsOperand(O, I, true, true, &M);
        O << "\n";
      } else {
        switch (I->getLinkage()) {
        case GlobalValue::LinkOnceLinkage:
          O << ".section __TEXT,__textcoal_nt,coalesced,no_toc\n"
            << ".weak_definition " << name << '\n'
            << ".private_extern " << name << '\n'
            << ".section __DATA,__datacoal_nt,coalesced,no_toc\n";
          LinkOnceStubs.insert(name);
          break;  
        case GlobalValue::WeakLinkage:   // FIXME: Verify correct for weak.
          // Nonnull linkonce -> weak
          O << "\t.weak " << name << "\n";
          SwitchSection(O, CurSection, "");
          O << "\t.section\t.llvm.linkonce.d." << name << ",\"aw\",@progbits\n";
          break;
        case GlobalValue::AppendingLinkage:
          // FIXME: appending linkage variables should go into a section of
          // their name or something.  For now, just emit them as external.
        case GlobalValue::ExternalLinkage:
          // If external or appending, declare as a global symbol
          O << "\t.globl " << name << "\n";
          // FALL THROUGH
        case GlobalValue::InternalLinkage:
          SwitchSection(O, CurSection, ".data");
          break;
        }

        O << "\t.align " << Align << "\n";
        O << name << ":\t\t\t\t; ";
        WriteAsOperand(O, I, true, true, &M);
        O << " = ";
        WriteAsOperand(O, C, false, false, &M);
        O << "\n";
        emitGlobalConstant(C);
      }
    }

  // Output stubs for link-once variables
  if (LinkOnceStubs.begin() != LinkOnceStubs.end())
    O << ".data\n.align 2\n";
  for (std::set<std::string>::iterator i = LinkOnceStubs.begin(), 
         e = LinkOnceStubs.end(); i != e; ++i)
  {
    O << *i << "$non_lazy_ptr:\n"
      << "\t.long\t" << *i << '\n';
  }
  
  // Output stubs for dynamically-linked functions
  for (std::set<std::string>::iterator i = FnStubs.begin(), e = FnStubs.end(); 
       i != e; ++i)
  {
    O << "\t.picsymbol_stub\n";
    O << "L" << *i << "$stub:\n";
    O << "\t.indirect_symbol " << *i << "\n";
    O << "\tmflr r0\n";
    O << "\tbl L0$" << *i << "\n";
    O << "L0$" << *i << ":\n";
    O << "\tmflr r11\n";
    O << "\taddis r11,r11,ha16(L" << *i << "$lazy_ptr-L0$" << *i << ")\n";
    O << "\tmtlr r0\n";
    O << "\tlwz r12,lo16(L" << *i << "$lazy_ptr-L0$" << *i << ")(r11)\n";
    O << "\tmtctr r12\n";
    O << "\taddi r11,r11,lo16(L" << *i << "$lazy_ptr - L0$" << *i << ")\n";
    O << "\tbctr\n";
    O << ".data\n";
    O << ".lazy_symbol_pointer\n";
    O << "L" << *i << "$lazy_ptr:\n";
    O << ".indirect_symbol " << *i << "\n";
    O << ".long dyld_stub_binding_helper\n";
  }

  O << "\n";

  // Output stubs for external global variables
  if (GVStubs.begin() != GVStubs.end())
    O << ".data\n\t.non_lazy_symbol_pointer\n";
  for (std::set<std::string>::iterator i = GVStubs.begin(), e = GVStubs.end(); 
       i != e; ++i) {
    O << "L" << *i << "$non_lazy_ptr:\n";
    O << "\t.indirect_symbol " << *i << "\n";
    O << "\t.long\t0\n";
  }
  
  delete Mang;
  return false; // success
}

} // End llvm namespace
