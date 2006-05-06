//===-- X86IntelAsmPrinter.cpp - Convert X86 LLVM code to Intel assembly --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a printer that converts from our internal representation
// of machine-dependent LLVM code to Intel format assembly language.
// This printer is the output mechanism used by `llc'.
//
//===----------------------------------------------------------------------===//

#include "X86IntelAsmPrinter.h"
#include "X86.h"
#include "llvm/Constants.h"
#include "llvm/Module.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Support/Mangler.h"
#include "llvm/Target/TargetOptions.h"
using namespace llvm;

X86IntelAsmPrinter::X86IntelAsmPrinter(std::ostream &O, X86TargetMachine &TM)
    : X86SharedAsmPrinter(O, TM) {
}

/// runOnMachineFunction - This uses the printMachineInstruction()
/// method to print assembly for each instruction.
///
bool X86IntelAsmPrinter::runOnMachineFunction(MachineFunction &MF) {
  SetupMachineFunction(MF);
  O << "\n\n";

  // Print out constants referenced by the function
  EmitConstantPool(MF.getConstantPool());

  // Print out labels for the function.
  SwitchSection(".code", MF.getFunction());
  EmitAlignment(4);
  if (MF.getFunction()->getLinkage() == GlobalValue::ExternalLinkage)
    O << "\tpublic " << CurrentFnName << "\n";
  O << CurrentFnName << "\tproc near\n";
  
  // Print out code for the function.
  for (MachineFunction::const_iterator I = MF.begin(), E = MF.end();
       I != E; ++I) {
    // Print a label for the basic block if there are any predecessors.
    if (I->pred_begin() != I->pred_end()) {
      printBasicBlockLabel(I, true);
      O << '\n';
    }
    for (MachineBasicBlock::const_iterator II = I->begin(), E = I->end();
         II != E; ++II) {
      // Print the assembly for the instruction.
      O << "\t";
      printMachineInstruction(II);
    }
  }

  O << CurrentFnName << "\tendp\n";

  // We didn't modify anything.
  return false;
}

void X86IntelAsmPrinter::printSSECC(const MachineInstr *MI, unsigned Op) {
  unsigned char value = MI->getOperand(Op).getImmedValue();
  assert(value <= 7 && "Invalid ssecc argument!");
  switch (value) {
  case 0: O << "eq"; break;
  case 1: O << "lt"; break;
  case 2: O << "le"; break;
  case 3: O << "unord"; break;
  case 4: O << "neq"; break;
  case 5: O << "nlt"; break;
  case 6: O << "nle"; break;
  case 7: O << "ord"; break;
  }
}

void X86IntelAsmPrinter::printOp(const MachineOperand &MO, 
                                 const char *Modifier) {
  const MRegisterInfo &RI = *TM.getRegisterInfo();
  switch (MO.getType()) {
  case MachineOperand::MO_Register:
    if (MRegisterInfo::isPhysicalRegister(MO.getReg())) {
      unsigned Reg = MO.getReg();
      if (Modifier && strncmp(Modifier, "trunc", strlen("trunc")) == 0) {
        MVT::ValueType VT = (strcmp(Modifier,"trunc16") == 0)
          ? MVT::i16 : MVT::i32;
        Reg = getX86SubSuperRegister(Reg, VT);
      }
      O << RI.get(Reg).Name;
    } else
      O << "reg" << MO.getReg();
    return;

  case MachineOperand::MO_Immediate:
    O << (int)MO.getImmedValue();
    return;
  case MachineOperand::MO_MachineBasicBlock:
    printBasicBlockLabel(MO.getMachineBasicBlock());
    return;
  case MachineOperand::MO_ConstantPoolIndex: {
    bool isMemOp  = Modifier && !strcmp(Modifier, "mem");
    if (!isMemOp) O << "OFFSET ";
    O << "[" << PrivateGlobalPrefix << "CPI" << getFunctionNumber() << "_"
      << MO.getConstantPoolIndex();
    int Offset = MO.getOffset();
    if (Offset > 0)
      O << " + " << Offset;
    else if (Offset < 0)
      O << Offset;
    O << "]";
    return;
  }
  case MachineOperand::MO_GlobalAddress: {
    bool isCallOp = Modifier && !strcmp(Modifier, "call");
    bool isMemOp  = Modifier && !strcmp(Modifier, "mem");
    if (!isMemOp && !isCallOp) O << "OFFSET ";
    O << Mang->getValueName(MO.getGlobal());
    int Offset = MO.getOffset();
    if (Offset > 0)
      O << " + " << Offset;
    else if (Offset < 0)
      O << Offset;
    return;
  }
  case MachineOperand::MO_ExternalSymbol: {
    bool isCallOp = Modifier && !strcmp(Modifier, "call");
    if (!isCallOp) O << "OFFSET ";
    O << GlobalPrefix << MO.getSymbolName();
    return;
  }
  default:
    O << "<unknown operand type>"; return;
  }
}

void X86IntelAsmPrinter::printMemReference(const MachineInstr *MI, unsigned Op){
  assert(isMem(MI, Op) && "Invalid memory reference!");

  const MachineOperand &BaseReg  = MI->getOperand(Op);
  int ScaleVal                   = MI->getOperand(Op+1).getImmedValue();
  const MachineOperand &IndexReg = MI->getOperand(Op+2);
  const MachineOperand &DispSpec = MI->getOperand(Op+3);

  if (BaseReg.isFrameIndex()) {
    O << "[frame slot #" << BaseReg.getFrameIndex();
    if (DispSpec.getImmedValue())
      O << " + " << DispSpec.getImmedValue();
    O << "]";
    return;
  }

  O << "[";
  bool NeedPlus = false;
  if (BaseReg.getReg()) {
    printOp(BaseReg, "mem");
    NeedPlus = true;
  }

  if (IndexReg.getReg()) {
    if (NeedPlus) O << " + ";
    if (ScaleVal != 1)
      O << ScaleVal << "*";
    printOp(IndexReg);
    NeedPlus = true;
  }

  if (DispSpec.isGlobalAddress() || DispSpec.isConstantPoolIndex()) {
    if (NeedPlus)
      O << " + ";
    printOp(DispSpec, "mem");
  } else {
    int DispVal = DispSpec.getImmedValue();
    if (DispVal || (!BaseReg.getReg() && !IndexReg.getReg())) {
      if (NeedPlus)
        if (DispVal > 0)
          O << " + ";
        else {
          O << " - ";
          DispVal = -DispVal;
        }
      O << DispVal;
    }
  }
  O << "]";
}

void X86IntelAsmPrinter::printPICLabel(const MachineInstr *MI, unsigned Op) {
  O << "\"L" << getFunctionNumber() << "$pb\"\n";
  O << "\"L" << getFunctionNumber() << "$pb\":";
}

bool X86IntelAsmPrinter::printAsmMRegister(const MachineOperand &MO,
                                           const char Mode) {
  const MRegisterInfo &RI = *TM.getRegisterInfo();
  unsigned Reg = MO.getReg();
  switch (Mode) {
  default: return true;  // Unknown mode.
  case 'b': // Print QImode register
    Reg = getX86SubSuperRegister(Reg, MVT::i8);
    break;
  case 'h': // Print QImode high register
    Reg = getX86SubSuperRegister(Reg, MVT::i8, true);
    break;
  case 'w': // Print HImode register
    Reg = getX86SubSuperRegister(Reg, MVT::i16);
    break;
  case 'k': // Print SImode register
    Reg = getX86SubSuperRegister(Reg, MVT::i32);
    break;
  }

  O << '%' << RI.get(Reg).Name;
  return false;
}

/// PrintAsmOperand - Print out an operand for an inline asm expression.
///
bool X86IntelAsmPrinter::PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                                         unsigned AsmVariant, 
                                         const char *ExtraCode) {
  // Does this asm operand have a single letter operand modifier?
  if (ExtraCode && ExtraCode[0]) {
    if (ExtraCode[1] != 0) return true; // Unknown modifier.
    
    switch (ExtraCode[0]) {
    default: return true;  // Unknown modifier.
    case 'b': // Print QImode register
    case 'h': // Print QImode high register
    case 'w': // Print HImode register
    case 'k': // Print SImode register
      return printAsmMRegister(MI->getOperand(OpNo), ExtraCode[0]);
    }
  }
  
  printOperand(MI, OpNo);
  return false;
}

bool X86IntelAsmPrinter::PrintAsmMemoryOperand(const MachineInstr *MI,
                                               unsigned OpNo,
                                               unsigned AsmVariant, 
                                               const char *ExtraCode) {
  if (ExtraCode && ExtraCode[0])
    return true; // Unknown modifier.
  printMemReference(MI, OpNo);
  return false;
}

/// printMachineInstruction -- Print out a single X86 LLVM instruction
/// MI in Intel syntax to the current output stream.
///
void X86IntelAsmPrinter::printMachineInstruction(const MachineInstr *MI) {
  ++EmittedInsts;

  // See if a truncate instruction can be turned into a nop.
  switch (MI->getOpcode()) {
  default: break;
  case X86::TRUNC_R32_R16:
  case X86::TRUNC_R32_R8:
  case X86::TRUNC_R16_R8: {
    const MachineOperand &MO0 = MI->getOperand(0);
    const MachineOperand &MO1 = MI->getOperand(1);
    unsigned Reg0 = MO0.getReg();
    unsigned Reg1 = MO1.getReg();
    if (MI->getOpcode() == X86::TRUNC_R16_R8)
      Reg0 = getX86SubSuperRegister(Reg0, MVT::i16);
    else
      Reg0 = getX86SubSuperRegister(Reg0, MVT::i32);
    if (Reg0 == Reg1)
      O << CommentString << " TRUNCATE ";
    break;
  }
  }

  // Call the autogenerated instruction printer routines.
  printInstruction(MI);
}

bool X86IntelAsmPrinter::doInitialization(Module &M) {
  MLSections = true;
  GlobalPrefix = "_";
  CommentString = ";";

  X86SharedAsmPrinter::doInitialization(M);

  PrivateGlobalPrefix = "$";
  AlignDirective = "\talign\t";
  ZeroDirective = "\tdb\t";
  ZeroDirectiveSuffix = " dup(0)";
  AsciiDirective = "\tdb\t";
  AscizDirective = 0;
  Data8bitsDirective = "\tdb\t";
  Data16bitsDirective = "\tdw\t";
  Data32bitsDirective = "\tdd\t";
  Data64bitsDirective = "\tdq\t";
  HasDotTypeDotSizeDirective = false;
  Mang->markCharUnacceptable('.');

  O << "\t.686\n\t.model flat\n\n";

  // Emit declarations for external functions.
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
    if (I->isExternal())
      O << "\textern " << Mang->getValueName(I) << ":near\n";

  // Emit declarations for external globals.  Note that VC++ always declares
  // external globals to have type byte, and if that's good enough for VC++...
  for (Module::const_global_iterator I = M.global_begin(), E = M.global_end();
       I != E; ++I) {
    if (I->isExternal())
      O << "\textern " << Mang->getValueName(I) << ":byte\n";
  }

  return false;
}

bool X86IntelAsmPrinter::doFinalization(Module &M) {
  const TargetData *TD = TM.getTargetData();

  // Print out module-level global variables here.
  for (Module::const_global_iterator I = M.global_begin(), E = M.global_end();
       I != E; ++I) {
    if (I->isExternal()) continue;   // External global require no code
    
    // Check to see if this is a special global used by LLVM, if so, emit it.
    if (EmitSpecialLLVMGlobal(I))
      continue;
    
    std::string name = Mang->getValueName(I);
    Constant *C = I->getInitializer();
    unsigned Size = TD->getTypeSize(C->getType());
    unsigned Align = getPreferredAlignmentLog(I);
    bool bCustomSegment = false;

    switch (I->getLinkage()) {
    case GlobalValue::LinkOnceLinkage:
    case GlobalValue::WeakLinkage:
      SwitchSection("", 0);
      O << name << "?\tsegment common 'COMMON'\n";
      bCustomSegment = true;
      // FIXME: the default alignment is 16 bytes, but 1, 2, 4, and 256
      // are also available.
      break;
    case GlobalValue::AppendingLinkage:
      SwitchSection("", 0);
      O << name << "?\tsegment public 'DATA'\n";
      bCustomSegment = true;
      // FIXME: the default alignment is 16 bytes, but 1, 2, 4, and 256
      // are also available.
      break;
    case GlobalValue::ExternalLinkage:
      O << "\tpublic " << name << "\n";
      // FALL THROUGH
    case GlobalValue::InternalLinkage:
      SwitchSection(".data", I);
      break;
    default:
      assert(0 && "Unknown linkage type!");
    }

    if (!bCustomSegment)
      EmitAlignment(Align, I);

    O << name << ":\t\t\t\t" << CommentString << " " << I->getName() << '\n';

    EmitGlobalConstant(C);

    if (bCustomSegment)
      O << name << "?\tends\n";
  }
  
  // Bypass X86SharedAsmPrinter::doFinalization().
  AsmPrinter::doFinalization(M);
  SwitchSection("", 0);
  O << "\tend\n";
  return false; // success
}

void X86IntelAsmPrinter::EmitString(const ConstantArray *CVA) const {
  unsigned NumElts = CVA->getNumOperands();
  if (NumElts) {
    // ML does not have escape sequences except '' for '.  It also has a maximum
    // string length of 255.
    unsigned len = 0;
    bool inString = false;
    for (unsigned i = 0; i < NumElts; i++) {
      int n = cast<ConstantInt>(CVA->getOperand(i))->getRawValue() & 255;
      if (len == 0)
        O << "\tdb ";

      if (n >= 32 && n <= 127) {
        if (!inString) {
          if (len > 0) {
            O << ",'";
            len += 2;
          } else {
            O << "'";
            len++;
          }
          inString = true;
        }
        if (n == '\'') {
          O << "'";
          len++;
        }
        O << char(n);
      } else {
        if (inString) {
          O << "'";
          len++;
          inString = false;
        }
        if (len > 0) {
          O << ",";
          len++;
        }
        O << n;
        len += 1 + (n > 9) + (n > 99);
      }

      if (len > 60) {
        if (inString) {
          O << "'";
          inString = false;
        }
        O << "\n";
        len = 0;
      }
    }

    if (len > 0) {
      if (inString)
        O << "'";
      O << "\n";
    }
  }
}

// Include the auto-generated portion of the assembly writer.
#include "X86GenAsmWriter1.inc"
