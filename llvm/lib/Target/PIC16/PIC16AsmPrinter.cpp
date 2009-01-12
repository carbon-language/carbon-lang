//===-- PIC16AsmPrinter.cpp - PIC16 LLVM assembly writer ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a printer that converts from our internal representation
// of machine-dependent LLVM code to PIC16 assembly language.
//
//===----------------------------------------------------------------------===//

#include "PIC16AsmPrinter.h"
#include "PIC16TargetAsmInfo.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Mangler.h"
#include "llvm/Function.h"
#include "llvm/Module.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/DerivedTypes.h"

using namespace llvm;

#include "PIC16GenAsmWriter.inc"

bool PIC16AsmPrinter::printMachineInstruction(const MachineInstr *MI) {
  std::string NewBankselLabel;
  unsigned Operands = MI->getNumOperands();
  if (Operands > 1) {
    // Global address or external symbol should be second operand from last
    // if we want to print banksel for it.
    const MachineOperand &Op = MI->getOperand(Operands-2);
    unsigned OpType = Op.getType();
    if (OpType == MachineOperand::MO_GlobalAddress ||
        OpType == MachineOperand::MO_ExternalSymbol) { 
      if (OpType == MachineOperand::MO_GlobalAddress ) 
        NewBankselLabel =  Mang->getValueName(Op.getGlobal());
      else 
        NewBankselLabel =  Op.getSymbolName();

      // Operand after global address or external symbol should be  banksel.
      // Value 1 for this operand means we need to generate banksel else do not
      // generate banksel.
      const MachineOperand &BS = MI->getOperand(Operands-1);
      if (((int)BS.getImm() == 1) &&
          (strcmp (CurrentBankselLabelInBasicBlock.c_str(),
                   NewBankselLabel.c_str()))) {
        CurrentBankselLabelInBasicBlock = NewBankselLabel;
        O << "\tbanksel ";
        printOperand(MI, Operands-2);
        O << "\n";
      }
    }
  }
  printInstruction(MI);
  return true;
}

/// runOnMachineFunction - This uses the printInstruction()
/// method to print assembly for each instruction.
///
bool PIC16AsmPrinter::runOnMachineFunction(MachineFunction &MF) {
  // This calls the base class function required to be called at beginning
  // of runOnMachineFunction.
  SetupMachineFunction(MF);

  // Get the mangled name.
  const Function *F = MF.getFunction();
  CurrentFnName = Mang->getValueName(F);

  // Emit the function variables.
  emitFunctionData(MF);
  std::string codeSection;
  codeSection = "code." + CurrentFnName + ".#";
  O <<  "\n";
  SwitchToTextSection (codeSection.c_str(),F);

  // Print out code for the function.
  for (MachineFunction::const_iterator I = MF.begin(), E = MF.end();
       I != E; ++I) {
    // Print a label for the basic block.
    if (I != MF.begin()) {
      printBasicBlockLabel(I, true);
      O << '\n';
    }
    else
      O << CurrentFnName << ":\n";
    CurrentBankselLabelInBasicBlock = "";
    for (MachineBasicBlock::const_iterator II = I->begin(), E = I->end();
         II != E; ++II) {
      // Print the assembly for the instruction.
        printMachineInstruction(II);
    }
  }
  return false;  // we didn't modify anything.
}

/// createPIC16CodePrinterPass - Returns a pass that prints the PIC16
/// assembly code for a MachineFunction to the given output stream,
/// using the given target machine description.  This should work
/// regardless of whether the function is in SSA form.
///
FunctionPass *llvm::createPIC16CodePrinterPass(raw_ostream &o,
                                               PIC16TargetMachine &tm) {
  return new PIC16AsmPrinter(o, tm, tm.getTargetAsmInfo());
}

void PIC16AsmPrinter::printOperand(const MachineInstr *MI, int opNum) {
  const MachineOperand &MO = MI->getOperand(opNum);

  switch (MO.getType()) {
    case MachineOperand::MO_Register:
      if (TargetRegisterInfo::isPhysicalRegister(MO.getReg()))
        O << TM.getRegisterInfo()->get(MO.getReg()).AsmName;
      else
        assert(0 && "not implemented");
        return;

    case MachineOperand::MO_Immediate:
      O << (int)MO.getImm();
      return;

    case MachineOperand::MO_GlobalAddress:
      O << Mang->getValueName(MO.getGlobal());
      break;

    case MachineOperand::MO_ExternalSymbol:
      O << MO.getSymbolName();
      break;

    default:
      assert(0 && " Operand type not supported.");
  }
}

bool PIC16AsmPrinter::doInitialization (Module &M) {
  bool Result = AsmPrinter::doInitialization(M);
  // FIXME:: This is temporary solution to generate the include file.
  // The processor should be passed to llc as in input and the header file
  // should be generated accordingly.
  O << "\t#include P16F1937.INC\n";
  EmitExternsAndGlobals (M);
  EmitInitData (M);
  EmitUnInitData(M);
  EmitRomData(M);
  return Result;
}

void PIC16AsmPrinter::EmitExternsAndGlobals (Module &M) {
 // Emit declarations for external functions.
  O << "section.0" <<"\n";
  for (Module::iterator I = M.begin(), E = M.end(); I != E; I++) {
    std::string Name = Mang->getValueName(I);
    if (Name.compare("abort") == 0)
      continue;
    if (I->isDeclaration()) {
      O << "\textern " <<Name << "\n";
      O << "\textern " << Name << ".retval\n";
      O << "\textern " << Name << ".args\n";
    }
    else if (I->hasExternalLinkage()) {
      O << "\tglobal " << Name << "\n";
      O << "\tglobal " << Name << ".retval\n";
      O << "\tglobal " << Name << ".args\n";
    }
  }
  // Emit declarations for external globals.
  for (Module::const_global_iterator I = M.global_begin(), E = M.global_end();
       I != E; I++) {
    std::string Name = Mang->getValueName(I);
    if (I->isDeclaration())
      O << "\textern "<< Name << "\n";
    else if (I->getLinkage() == GlobalValue::CommonLinkage)
      O << "\tglobal "<< Name << "\n";
  }
}
void PIC16AsmPrinter::EmitInitData (Module &M) {
  std::string iDataSection = "idata.#";
  SwitchToDataSection(iDataSection.c_str());
  for (Module::const_global_iterator I = M.global_begin(), E = M.global_end();
       I != E; ++I) {
    if (!I->hasInitializer())   // External global require no code.
      continue;

    Constant *C = I->getInitializer();
    const PointerType *PtrTy = I->getType();
    int AddrSpace = PtrTy->getAddressSpace();

    if ((!C->isNullValue()) && (AddrSpace == PIC16ISD::RAM_SPACE)) {
    
      if (EmitSpecialLLVMGlobal(I)) 
        continue;

      // Any variables reaching here with "." in its name is a local scope
      // variable and should not be printed in global data section.
      std::string name = Mang->getValueName(I);
      if (name.find(".") != std::string::npos)
        continue;

      O << name;
      EmitGlobalConstant(C);
    }
  }
}

void PIC16AsmPrinter::EmitConstantValueOnly(const Constant* CV) {
  if (const ConstantInt *CI = dyn_cast<ConstantInt>(CV)) {
    unsigned BitWidth = CI->getBitWidth();
    int Val = CI->getZExtValue();
    if (BitWidth == 8) {
      // Expecting db directive here. In case of romdata we need to pad the
      // word with zeros.
      if (IsRomData)
        O << 0 <<", ";
      O << Val; 
    }
    else if (BitWidth == 16) {
      unsigned Element1, Element2;
      Element1 = 0x00ff & Val;
      Element2 = 0x00ff & (Val >> 8);
      if (IsRomData)
        O << 0 <<", "<<Element1 <<", "<< 0 <<", "<< Element2;
      else
        O << Element1 <<", "<< Element2;  
    }
    else if (BitWidth == 32) {
      unsigned Element1, Element2, Element3, Element4;
      Element1 = 0x00ff & Val;
      Element2 = 0x00ff & (Val >> 8);
      Element3 = 0x00ff & (Val >> 16);
      Element4 = 0x00ff & (Val >> 24);
      if (IsRomData)
        O << 0 <<", "<< Element1 <<", "<< 0 <<", "<< Element2 <<", "<< 0 
          <<", "<< Element3 <<", "<< 0 <<", "<< Element4;
      else 
        O << Element1 <<", "<< Element2 <<", "<< Element3 <<", "<< Element4;    
    }
    return;
  }
  AsmPrinter::EmitConstantValueOnly(CV);
}

void PIC16AsmPrinter::EmitRomData (Module &M)
{
  std::string romDataSection = "romdata.#";
  SwitchToRomDataSection(romDataSection.c_str());
  IsRomData = true;
  for (Module::const_global_iterator I = M.global_begin(), E = M.global_end();
       I != E; ++I) {
    if (!I->hasInitializer())   // External global require no code.
      continue;

    Constant *C = I->getInitializer();
    const PointerType *PtrTy = I->getType();
    int AddrSpace = PtrTy->getAddressSpace();
    if ((!C->isNullValue()) && (AddrSpace == PIC16ISD::ROM_SPACE)) {

      if (EmitSpecialLLVMGlobal(I))
        continue;

      // Any variables reaching here with "." in its name is a local scope
      // variable and should not be printed in global data section.
      std::string name = Mang->getValueName(I);
      if (name.find(".") != std::string::npos)
        continue;

      O << name;
      EmitGlobalConstant(C);
      O << "\n";
    }
  }
  IsRomData = false;
}


void PIC16AsmPrinter::EmitUnInitData (Module &M)
{
  std::string uDataSection = "udata.#";
  SwitchToUDataSection(uDataSection.c_str());
  const TargetData *TD = TM.getTargetData();

  for (Module::const_global_iterator I = M.global_begin(), E = M.global_end();
       I != E; ++I) {
    if (!I->hasInitializer())   // External global require no code.
      continue;

    Constant *C = I->getInitializer();
    if (C->isNullValue()) {

      if (EmitSpecialLLVMGlobal(I))
        continue;

      // Any variables reaching here with "." in its name is a local scope
      // variable and should not be printed in global data section.
      std::string name = Mang->getValueName(I);
      if (name.find(".") != std::string::npos)
        continue;

      const Type *Ty = C->getType();
      unsigned Size = TD->getTypePaddedSize(Ty);
      O << name << " " <<"RES"<< " " << Size ;
      O << "\n";
    }
  }
}

bool PIC16AsmPrinter::doFinalization(Module &M) {
  O << "\t" << "END\n";
  bool Result = AsmPrinter::doFinalization(M);
  return Result;
}

void PIC16AsmPrinter::emitFunctionData(MachineFunction &MF) {
  const Function *F = MF.getFunction();
  std::string FuncName = Mang->getValueName(F);
  const Module *M = F->getParent();
  const TargetData *TD = TM.getTargetData();

  // Emit the data section name.
  O << "\n"; 
  std::string fDataSection = "fdata." + CurrentFnName + ".#";
  SwitchToUDataSection(fDataSection.c_str(), F);
  
  //Emit function return value.
  O << CurrentFnName << ".retval:\n";
  const Type *RetType = F->getReturnType();
  if (RetType->getTypeID() != Type::VoidTyID) {
    unsigned RetSize = TD->getTypePaddedSize(RetType);
    if (RetSize > 0)
      O << CurrentFnName << ".retval" << " RES " << RetSize;
   }
  // Emit function arguments.
  O << CurrentFnName << ".args:\n";
  for (Function::const_arg_iterator AI = F->arg_begin(), AE = F->arg_end();
       AI != AE; ++AI) {
    std::string ArgName = Mang->getValueName(AI);
    const Type *ArgTy = AI->getType();
    unsigned ArgSize = TD->getTypePaddedSize(ArgTy);
    O << CurrentFnName << ".args." << ArgName << " RES " << ArgSize; 
  }
  // Emit the function variables. 
   
  // In PIC16 all the function arguments and local variables are global.
  // Therefore to get the variable belonging to this function entire
  // global list will be traversed and variables belonging to this function
  // will be emitted in the current data section.
  for (Module::const_global_iterator I = M->global_begin(), E = M->global_end();
       I != E; ++I) {
    std::string VarName = Mang->getValueName(I);
    
    // The variables of a function are of form FuncName.* . If this variable
    // does not belong to this function then continue. 
    if (!(VarName.find(FuncName + ".") == 0 ? true : false))
      continue;
   
    Constant *C = I->getInitializer();
    const Type *Ty = C->getType();
    unsigned Size = TD->getTypePaddedSize(Ty);
    // Emit memory reserve directive.
    O << VarName << "  RES  " << Size << "\n";
  }
  emitFunctionTempData(MF);
}

void PIC16AsmPrinter::emitFunctionTempData(MachineFunction &MF) {
  // Emit temporary variables.
  MachineFrameInfo *FrameInfo = MF.getFrameInfo();
  if (FrameInfo->hasStackObjects()) {
    int indexBegin = FrameInfo->getObjectIndexBegin();
    int indexEnd = FrameInfo->getObjectIndexEnd();

    if (indexBegin < indexEnd)
      O << CurrentFnName << ".tmp RES"<< " " 
        <<indexEnd - indexBegin <<"\n";
    /*
    while (indexBegin < indexEnd) {
        O << CurrentFnName << "_tmp_" << indexBegin << " " << "RES"<< " " 
          << 1 << "\n" ;
        indexBegin++;
    }
    */
  }
}

/// The function is same as AsmPrinter::SwitchtoDataSection except the call
/// to getUDataSectionStartSuffix.
void PIC16AsmPrinter::SwitchToUDataSection(const char *NewSection,
                                           const GlobalValue *GV) {
  std::string NS;
  if (GV && GV->hasSection())
    NS = TAI->getSwitchToSectionDirective() + GV->getSection();
  else
    NS = NewSection;

  // If we're already in this section, we're done.
  if (CurrentSection == NS) return;

  // Close the current section, if applicable.
  if (TAI->getSectionEndDirectiveSuffix() && !CurrentSection.empty())
    O << CurrentSection << TAI->getSectionEndDirectiveSuffix() << '\n';

  CurrentSection = NS;

  if (!CurrentSection.empty()){}
    O << CurrentSection << (static_cast<const PIC16TargetAsmInfo *>(TAI))->
                            getUDataSectionStartSuffix() << '\n';

  IsInTextSection = false;
}

/// The function is same as AsmPrinter::SwitchtoDataSection except the call
/// to getRomDataSectionStartSuffix.
void PIC16AsmPrinter::SwitchToRomDataSection(const char *NewSection,
                                             const GlobalValue *GV) {
  std::string NS;
  if (GV && GV->hasSection())
    NS = TAI->getSwitchToSectionDirective() + GV->getSection();
  else
    NS = NewSection;

  // If we're already in this section, we're done.
  if (CurrentSection == NS) return;

  // Close the current section, if applicable.
  if (TAI->getSectionEndDirectiveSuffix() && !CurrentSection.empty())
    O << CurrentSection << TAI->getSectionEndDirectiveSuffix() << '\n';

  CurrentSection = NS;

  if (!CurrentSection.empty()) {}
    O << CurrentSection << (static_cast< const PIC16TargetAsmInfo *>(TAI))->
                            getRomDataSectionStartSuffix() << '\n';

  IsInTextSection = false;
}

