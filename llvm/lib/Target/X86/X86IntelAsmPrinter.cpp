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
  CommentString = ";";
  GlobalPrefix = "_";
  PrivateGlobalPrefix = "$";
  AlignDirective = "\talign\t";
  MLSections = true;
  ZeroDirective = "\tdb\t";
  ZeroDirectiveSuffix = " dup(0)";
  AsciiDirective = "\tdb\t";
  AscizDirective = 0;
  Data8bitsDirective = "\t.db\t";
  Data16bitsDirective = "\t.dw\t";
  Data32bitsDirective = "\t.dd\t";
  Data64bitsDirective = "\t.dq\t";
  HasDotTypeDotSizeDirective = false;
}

/// runOnMachineFunction - This uses the printMachineInstruction()
/// method to print assembly for each instruction.
///
bool X86IntelAsmPrinter::runOnMachineFunction(MachineFunction &MF) {
  if (forDarwin) {
    // Let PassManager know we need debug information and relay
    // the MachineDebugInfo address on to DwarfWriter.
    DW.SetDebugInfo(&getAnalysis<MachineDebugInfo>());
  }

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
  
  if (forDarwin) {
    // Emit pre-function debug information.
    DW.BeginFunction(&MF);
  }

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

  if (forDarwin) {
    // Emit post-function debug information.
    DW.EndFunction();
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
  case MachineOperand::MO_VirtualRegister:
    if (Value *V = MO.getVRegValueOrNull()) {
      O << "<" << V->getName() << ">";
      return;
    }
    // FALLTHROUGH
  case MachineOperand::MO_MachineRegister:
    if (MRegisterInfo::isPhysicalRegister(MO.getReg()))
      O << RI.get(MO.getReg()).Name;
    else
      O << "reg" << MO.getReg();
    return;

  case MachineOperand::MO_SignExtendedImmed:
  case MachineOperand::MO_UnextendedImmed:
    O << (int)MO.getImmedValue();
    return;
  case MachineOperand::MO_MachineBasicBlock:
    printBasicBlockLabel(MO.getMachineBasicBlock());
    return;
  case MachineOperand::MO_PCRelativeDisp:
    assert(0 && "Shouldn't use addPCDisp() when building X86 MachineInstrs");
    abort ();
    return;
  case MachineOperand::MO_ConstantPoolIndex: {
    bool isMemOp  = Modifier && !strcmp(Modifier, "mem");
    if (!isMemOp) O << "OFFSET ";
    O << "[" << PrivateGlobalPrefix << "CPI" << getFunctionNumber() << "_"
      << MO.getConstantPoolIndex();
    if (forDarwin && TM.getRelocationModel() == Reloc::PIC)
      O << "-\"L" << getFunctionNumber() << "$pb\"";
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
    if (forDarwin && TM.getRelocationModel() != Reloc::Static) {
      GlobalValue *GV = MO.getGlobal();
      std::string Name = Mang->getValueName(GV);
      if (!isMemOp && !isCallOp) O << '$';
      // Link-once, External, or Weakly-linked global variables need
      // non-lazily-resolved stubs
      if (GV->isExternal() || GV->hasWeakLinkage() ||
          GV->hasLinkOnceLinkage()) {
        // Dynamically-resolved functions need a stub for the function.
        if (isCallOp && isa<Function>(GV) && cast<Function>(GV)->isExternal()) {
          FnStubs.insert(Name);
          O << "L" << Name << "$stub";
        } else {
          GVStubs.insert(Name);
          O << "L" << Name << "$non_lazy_ptr";
        }
      } else {
        O << Mang->getValueName(GV);
      }
      if (!isCallOp && TM.getRelocationModel() == Reloc::PIC)
        O << "-\"L" << getFunctionNumber() << "$pb\"";
    } else
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
    if (isCallOp && forDarwin && TM.getRelocationModel() != Reloc::Static) {
      std::string Name(GlobalPrefix);
      Name += MO.getSymbolName();
      FnStubs.insert(Name);
      O << "L" << Name << "$stub";
      return;
    }
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
  const char *Name = RI.get(Reg).Name;
  switch (Mode) {
  default: return true;  // Unknown mode.
  case 'b': // Print QImode register
    switch (Reg) {
    default: return true;
    case X86::AH: case X86::AL: case X86::AX: case X86::EAX:
      Name = "AL";
      break;
    case X86::DH: case X86::DL: case X86::DX: case X86::EDX:
      Name = "DL";
      break;
    case X86::CH: case X86::CL: case X86::CX: case X86::ECX:
      Name = "CL";
      break;
    case X86::BH: case X86::BL: case X86::BX: case X86::EBX:
      Name = "BL";
      break;
    case X86::ESI:
      Name = "SIL";
      break;
    case X86::EDI:
      Name = "DIL";
      break;
    case X86::EBP:
      Name = "BPL";
      break;
    case X86::ESP:
      Name = "SPL";
      break;
    }
    break;
  case 'h': // Print QImode high register
    switch (Reg) {
    default: return true;
    case X86::AH: case X86::AL: case X86::AX: case X86::EAX:
      Name = "AL";
      break;
    case X86::DH: case X86::DL: case X86::DX: case X86::EDX:
      Name = "DL";
      break;
    case X86::CH: case X86::CL: case X86::CX: case X86::ECX:
      Name = "CL";
      break;
    case X86::BH: case X86::BL: case X86::BX: case X86::EBX:
      Name = "BL";
      break;
    }
    break;
  case 'w': // Print HImode register
    switch (Reg) {
    default: return true;
    case X86::AH: case X86::AL: case X86::AX: case X86::EAX:
      Name = "AX";
      break;
    case X86::DH: case X86::DL: case X86::DX: case X86::EDX:
      Name = "DX";
      break;
    case X86::CH: case X86::CL: case X86::CX: case X86::ECX:
      Name = "CX";
      break;
    case X86::BH: case X86::BL: case X86::BX: case X86::EBX:
      Name = "BX";
      break;
    case X86::ESI:
      Name = "SI";
      break;
    case X86::EDI:
      Name = "DI";
      break;
    case X86::EBP:
      Name = "BP";
      break;
    case X86::ESP:
      Name = "SP";
      break;
    }
    break;
  case 'k': // Print SImode register
    switch (Reg) {
    default: return true;
    case X86::AH: case X86::AL: case X86::AX: case X86::EAX:
      Name = "EAX";
      break;
    case X86::DH: case X86::DL: case X86::DX: case X86::EDX:
      Name = "EDX";
      break;
    case X86::CH: case X86::CL: case X86::CX: case X86::ECX:
      Name = "ECX";
      break;
    case X86::BH: case X86::BL: case X86::BX: case X86::EBX:
      Name = "EBX";
      break;
    case X86::ESI:
      Name = "ESI";
      break;
    case X86::EDI:
      Name = "EDI";
      break;
    case X86::EBP:
      Name = "EBP";
      break;
    case X86::ESP:
      Name = "ESP";
      break;
    }
    break;
  }

  O << Name;
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

  // Call the autogenerated instruction printer routines.
  printInstruction(MI);
}

bool X86IntelAsmPrinter::doInitialization(Module &M) {
  X86SharedAsmPrinter::doInitialization(M);
  Mang->markCharUnacceptable('.');
  PrivateGlobalPrefix = "$";  // need this here too :(
  O << "\t.686\n\t.model flat\n\n";

  // Emit declarations for external functions.
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
    if (I->isExternal())
      O << "\textern " << Mang->getValueName(I) << ":near\n";

  // Emit declarations for external globals.
  for (Module::const_global_iterator I = M.global_begin(), E = M.global_end();
       I != E; ++I) {
    if (I->isExternal())
      O << "\textern " << Mang->getValueName(I) << ":byte\n";
  }

  return false;
}

bool X86IntelAsmPrinter::doFinalization(Module &M) {
  X86SharedAsmPrinter::doFinalization(M);
  SwitchSection("", 0);
  O << "\tend\n";
  return false;
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
