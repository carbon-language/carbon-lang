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
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/Module.h"
#include "llvm/CodeGen/DwarfWriter.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Mangler.h"
#include "llvm/CodeGen/DwarfWriter.h"
#include "llvm/CodeGen/MachineModuleInfo.h"

using namespace llvm;

#include "PIC16GenAsmWriter.inc"

inline static bool isLocalToFunc (std::string &FuncName, std::string &VarName) {
  if (VarName.find(FuncName + ".auto.") != std::string::npos 
      || VarName.find(FuncName + ".arg.") != std::string::npos)
    return true;

  return false;
}

inline static bool isLocalName (std::string &Name) {
  if (Name.find(".auto.") != std::string::npos 
      || Name.find(".arg.") != std::string::npos) 
    return true;

  return false;
}

bool PIC16AsmPrinter::printMachineInstruction(const MachineInstr *MI) {
  std::string NewBank = "";
  unsigned Operands = MI->getNumOperands();
  if (Operands > 1) {
    // If we have a Global address or external symbol then we need to print 
    // banksel for it. 
    unsigned BankSelVar = 0;
    MachineOperand Op = MI->getOperand(BankSelVar);
    while (BankSelVar < Operands-1) {
      Op = MI->getOperand(BankSelVar);
      if ((Op.getType() ==  MachineOperand::MO_GlobalAddress) ||
          (Op.getType() ==  MachineOperand::MO_ExternalSymbol))
        break;
      BankSelVar++;
    }
    if (BankSelVar < Operands-1) {
      unsigned OpType = Op.getType();
      if (OpType == MachineOperand::MO_GlobalAddress ) 
        NewBank = Op.getGlobal()->getSection(); 
      else {
        // External Symbol is generated for temp data and arguments. They are 
        // in fpdata.<functionname>.# section.
        std::string ESName = Op.getSymbolName();
        int index = ESName.find_first_of(".");
        std::string FnName = ESName.substr(0,index);
        NewBank = "fpdata." + FnName +".#";
      }
      // Operand after global address or external symbol should be  banksel.
      // Value 1 for this operand means we need to generate banksel else do not
      // generate banksel.
      const MachineOperand &BS = MI->getOperand(BankSelVar+1);
      // If Section names are same then the variables are in same section.
      // This is not true for external variables as section names for global
      // variables in all files are same at this time. For eg. initialized 
      // data in put in idata.# section in all files. 
      if ((BS.getType() == MachineOperand::MO_Immediate 
          && (int)BS.getImm() == 1) 
          && ((Op.isGlobal() && Op.getGlobal()->hasExternalLinkage()) ||
           (NewBank.compare(CurBank) != 0))) { 
        O << "\tbanksel ";
        printOperand(MI, BankSelVar);
        O << "\n";
        CurBank = NewBank;
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
  this->MF = &MF;

  // This calls the base class function required to be called at beginning
  // of runOnMachineFunction.
  SetupMachineFunction(MF);

  // Get the mangled name.
  const Function *F = MF.getFunction();
  CurrentFnName = Mang->getValueName(F);

  // Emit the function variables.
  emitFunctionData(MF);
  std::string codeSection;
  codeSection = "code." + CurrentFnName + ".# " + "CODE";
  const Section *fCodeSection = TAI->getNamedSection(codeSection.c_str(),
                                               SectionFlags::Code);
  O <<  "\n";
  SwitchToSection (fCodeSection);

  // Emit the frame address of the function at the beginning of code.
  O << "    retlw  low(" << FunctionLabelBegin<< CurrentFnName << ".frame)\n";
  O << "    retlw  high(" << FunctionLabelBegin<< CurrentFnName << ".frame)\n"; 
  O << CurrentFnName << ":\n";


  // Print out code for the function.
  for (MachineFunction::const_iterator I = MF.begin(), E = MF.end();
       I != E; ++I) {
    // Print a label for the basic block.
    if (I != MF.begin()) {
      printBasicBlockLabel(I, true);
      O << '\n';
    }
    CurBank = "";
    
    // For emitting line directives, we need to keep track of the current
    // source line. When it changes then only emit the line directive.
    unsigned CurLine = 0;
    for (MachineBasicBlock::const_iterator II = I->begin(), E = I->end();
         II != E; ++II) {
      // Emit the line directive if source line changed.
      const DebugLoc DL = II->getDebugLoc();
      if (!DL.isUnknown()) {
        unsigned line = MF.getDebugLocTuple(DL).Line;
        if (line != CurLine) {
          O << "\t.line " << line << "\n";
          CurLine = line;
        }
      }
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
                                               PIC16TargetMachine &tm,
                                               CodeGenOpt::Level OptLevel,
                                               bool verbose) {
  return new PIC16AsmPrinter(o, tm, tm.getTargetAsmInfo(), OptLevel, verbose);
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

    case MachineOperand::MO_GlobalAddress: {
      std::string Name = Mang->getValueName(MO.getGlobal());
      if (isLocalName(Name)) 
        O << FunctionLabelBegin << Mang->getValueName(MO.getGlobal());
      else
         O << Mang->getValueName(MO.getGlobal());
      break;
    }
    case MachineOperand::MO_ExternalSymbol: {
      std::string Name = MO.getSymbolName(); 
      if (Name.find("__intrinsics.") != std::string::npos)
        O  << MO.getSymbolName();
      else
        O << FunctionLabelBegin << MO.getSymbolName();
      break;
    }
    case MachineOperand::MO_MachineBasicBlock:
      printBasicBlockLabel(MO.getMBB());
      return;

    default:
      assert(0 && " Operand type not supported.");
  }
}

void PIC16AsmPrinter::printCCOperand(const MachineInstr *MI, int opNum) {
  int CC = (int)MI->getOperand(opNum).getImm();
  O << PIC16CondCodeToString((PIC16CC::CondCodes)CC);
}


bool PIC16AsmPrinter::doInitialization (Module &M) {
  bool Result = AsmPrinter::doInitialization(M);
  // FIXME:: This is temporary solution to generate the include file.
  // The processor should be passed to llc as in input and the header file
  // should be generated accordingly.
  O << "\t#include P16F1937.INC\n";
  MachineModuleInfo *MMI = getAnalysisIfAvailable<MachineModuleInfo>();
  assert(MMI);
  DwarfWriter *DW = getAnalysisIfAvailable<DwarfWriter>();
  assert(DW && "Dwarf Writer is not available");
  DW->BeginModule(&M, MMI, O, this, TAI);

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
    
    // If it is llvm intrinsic call then don't emit
    if (Name.find("llvm.") != std::string::npos)
      continue;

    if (I->isDeclaration()) {
      O << "\textern " <<Name << "\n";
      O << "\textern "  << FunctionLabelBegin << Name << ".retval\n";
      O << "\textern " << FunctionLabelBegin << Name << ".args\n";
    }
    else if (I->hasExternalLinkage()) {
      O << "\tglobal " << Name << "\n";
      O << "\tglobal " << FunctionLabelBegin << Name << ".retval\n";
      O << "\tglobal " << FunctionLabelBegin<< Name << ".args\n";
    }
  }

  // Emit header file to include declaration of library functions
  O << "\t#include C16IntrinsicCalls.INC\n";

  // Emit declarations for external globals.
  for (Module::const_global_iterator I = M.global_begin(), E = M.global_end();
       I != E; I++) {
    // Any variables reaching here with ".auto." in its name is a local scope
    // variable and should not be printed in global data section.
    std::string Name = Mang->getValueName(I);
    if (isLocalName (Name))
      continue;

    if (I->isDeclaration())
      O << "\textern "<< Name << "\n";
    else if (I->hasCommonLinkage() || I->hasExternalLinkage())
      O << "\tglobal "<< Name << "\n";
  }
}

void PIC16AsmPrinter::EmitInitData (Module &M) {
  SwitchToSection(TAI->getDataSection());
  for (Module::global_iterator I = M.global_begin(), E = M.global_end();
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
      std::string Name = Mang->getValueName(I);
      if (isLocalName(Name))
        continue;

      I->setSection(TAI->getDataSection()->getName());
      O << Name;
      EmitGlobalConstant(C, AddrSpace);
    }
  }
}

void PIC16AsmPrinter::EmitRomData (Module &M)
{
  SwitchToSection(TAI->getReadOnlySection());
  IsRomData = true;
  for (Module::global_iterator I = M.global_begin(), E = M.global_end();
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

      I->setSection(TAI->getReadOnlySection()->getName());
      O << name;
      EmitGlobalConstant(C, AddrSpace);
      O << "\n";
    }
  }
  IsRomData = false;
}

void PIC16AsmPrinter::EmitUnInitData (Module &M)
{
  SwitchToSection(TAI->getBSSSection_());
  const TargetData *TD = TM.getTargetData();

  for (Module::global_iterator I = M.global_begin(), E = M.global_end();
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

      I->setSection(TAI->getBSSSection_()->getName());

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
  Module *M = const_cast<Module *>(F->getParent());
  const TargetData *TD = TM.getTargetData();
  unsigned FrameSize = 0;
  // Emit the data section name.
  O << "\n"; 
  std::string SectionName = "fpdata." + CurrentFnName + ".# " + "UDATA_OVR";

  const Section *fPDataSection = TAI->getNamedSection(SectionName.c_str(),
                                                      SectionFlags::Writeable);
  SwitchToSection(fPDataSection);
  

  // Emit function frame label
  O << FunctionLabelBegin << CurrentFnName << ".frame:\n";

  const Type *RetType = F->getReturnType();
  unsigned RetSize = 0; 
  if (RetType->getTypeID() != Type::VoidTyID) 
    RetSize = TD->getTypePaddedSize(RetType);
  
  //Emit function return value space
  if(RetSize > 0)
     O << FunctionLabelBegin << CurrentFnName << ".retval    RES  " << RetSize 
       << "\n";
  else
     O << FunctionLabelBegin << CurrentFnName << ".retval:\n";
   
  // Emit variable to hold the space for function arguments 
  unsigned ArgSize = 0;
  for (Function::const_arg_iterator argi = F->arg_begin(),
           arge = F->arg_end(); argi != arge ; ++argi) {
    const Type *Ty = argi->getType();
    ArgSize += TD->getTypePaddedSize(Ty);
   }
  O << FunctionLabelBegin << CurrentFnName << ".args      RES  " << ArgSize 
    << "\n";

  // Emit temporary space
  int TempSize = PTLI->GetTmpSize();
  if (TempSize > 0 )
    O << FunctionLabelBegin << CurrentFnName << ".tmp       RES  " << TempSize 
      <<"\n";

  // Emit the section name for local variables.
  O << "\n";
  std::string SecNameLocals = "fadata." + CurrentFnName + ".# " + "UDATA_OVR";

  const Section *fADataSection = TAI->getNamedSection(SecNameLocals.c_str(),
                                                      SectionFlags::Writeable);
  SwitchToSection(fADataSection);

  // Emit the function variables. 
   
  // In PIC16 all the function arguments and local variables are global.
  // Therefore to get the variable belonging to this function entire
  // global list will be traversed and variables belonging to this function
  // will be emitted in the current data section.
  for (Module::global_iterator I = M->global_begin(), E = M->global_end();
       I != E; ++I) {
    std::string VarName = Mang->getValueName(I);
    
    // The variables of a function are of form FuncName.* . If this variable
    // does not belong to this function then continue. 
    // Static local varilabes of a function does not have .auto. in their
    // name. They are not printed as part of function data but module
    // level global data.
    if (! isLocalToFunc(FuncName, VarName))
     continue;

    I->setSection("fadata." + CurrentFnName + ".#");
    Constant *C = I->getInitializer();
    const Type *Ty = C->getType();
    unsigned Size = TD->getTypePaddedSize(Ty);
    FrameSize += Size; 
    // Emit memory reserve directive.
    O << FunctionLabelBegin << VarName << "  RES  " << Size << "\n";
  }

}
