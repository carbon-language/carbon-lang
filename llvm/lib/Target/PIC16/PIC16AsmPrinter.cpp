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
#include "PIC16Section.h"
#include "PIC16TargetAsmInfo.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/Module.h"
#include "llvm/CodeGen/DwarfWriter.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/DwarfWriter.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/Target/TargetRegistry.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/Mangler.h"
#include <cstring>
using namespace llvm;

#include "PIC16GenAsmWriter.inc"

PIC16AsmPrinter::PIC16AsmPrinter(formatted_raw_ostream &O, TargetMachine &TM,
                                 const TargetAsmInfo *T, bool V)
: AsmPrinter(O, TM, T, V), DbgInfo(O, T) {
  PTLI = static_cast<PIC16TargetLowering*>(TM.getTargetLowering());
  PTAI = static_cast<const PIC16TargetAsmInfo*>(T);
  PTOF = (PIC16TargetObjectFile*)&PTLI->getObjFileLowering();
}

bool PIC16AsmPrinter::printMachineInstruction(const MachineInstr *MI) {
  printInstruction(MI);
  return true;
}

/// runOnMachineFunction - This emits the frame section, autos section and 
/// assembly for each instruction. Also takes care of function begin debug
/// directive and file begin debug directive (if required) for the function.
///
bool PIC16AsmPrinter::runOnMachineFunction(MachineFunction &MF) {
  this->MF = &MF;

  // This calls the base class function required to be called at beginning
  // of runOnMachineFunction.
  SetupMachineFunction(MF);

  // Get the mangled name.
  const Function *F = MF.getFunction();
  CurrentFnName = Mang->getMangledName(F);

  // Emit the function frame (args and temps).
  EmitFunctionFrame(MF);

  DbgInfo.BeginFunction(MF);

  // Emit the autos section of function.
  EmitAutos(CurrentFnName);

  // Now emit the instructions of function in its code section.
  const MCSection *fCodeSection = 
    getObjFileLowering().getSectionForFunction(CurrentFnName);
  // Start the Code Section.
  O <<  "\n";
  SwitchToSection(fCodeSection);

  // Emit the frame address of the function at the beginning of code.
  O << "\tretlw  low(" << PAN::getFrameLabel(CurrentFnName) << ")\n";
  O << "\tretlw  high(" << PAN::getFrameLabel(CurrentFnName) << ")\n";

  // Emit function start label.
  O << CurrentFnName << ":\n";

  DebugLoc CurDL;
  O << "\n"; 
  // Print out code for the function.
  for (MachineFunction::const_iterator I = MF.begin(), E = MF.end();
       I != E; ++I) {

    // Print a label for the basic block.
    if (I != MF.begin()) {
      printBasicBlockLabel(I, true);
      O << '\n';
    }
    
    // Print a basic block.
    for (MachineBasicBlock::const_iterator II = I->begin(), E = I->end();
         II != E; ++II) {

      // Emit the line directive if source line changed.
      const DebugLoc DL = II->getDebugLoc();
      if (!DL.isUnknown() && DL != CurDL) {
        DbgInfo.ChangeDebugLoc(MF, DL);
        CurDL = DL;
      }
        
      // Print the assembly for the instruction.
      printMachineInstruction(II);
    }
  }
  
  // Emit function end debug directives.
  DbgInfo.EndFunction(MF);

  return false;  // we didn't modify anything.
}


// printOperand - print operand of insn.
void PIC16AsmPrinter::printOperand(const MachineInstr *MI, int opNum) {
  const MachineOperand &MO = MI->getOperand(opNum);

  switch (MO.getType()) {
    case MachineOperand::MO_Register:
      if (TargetRegisterInfo::isPhysicalRegister(MO.getReg()))
        O << TM.getRegisterInfo()->get(MO.getReg()).AsmName;
      else
        llvm_unreachable("not implemented");
      return;

    case MachineOperand::MO_Immediate:
      O << (int)MO.getImm();
      return;

    case MachineOperand::MO_GlobalAddress: {
      std::string Sname = Mang->getMangledName(MO.getGlobal());
      // FIXME: currently we do not have a memcpy def coming in the module
      // by any chance, as we do not link in those as .bc lib. So these calls
      // are always external and it is safe to emit an extern.
      if (PAN::isMemIntrinsic(Sname)) {
        LibcallDecls.push_back(createESName(Sname));
      }

      O << Sname;
      break;
    }
    case MachineOperand::MO_ExternalSymbol: {
       const char *Sname = MO.getSymbolName();

      // If its a libcall name, record it to decls section.
      if (PAN::getSymbolTag(Sname) == PAN::LIBCALL) {
        LibcallDecls.push_back(Sname);
      }

      // Record a call to intrinsic to print the extern declaration for it.
      std::string Sym = Sname;  
      if (PAN::isMemIntrinsic(Sym)) {
        Sym = PAN::addPrefix(Sym);
        LibcallDecls.push_back(createESName(Sym));
      }

      O  << Sym;
      break;
    }
    case MachineOperand::MO_MachineBasicBlock:
      printBasicBlockLabel(MO.getMBB());
      return;

    default:
      llvm_unreachable(" Operand type not supported.");
  }
}

/// printCCOperand - Print the cond code operand.
///
void PIC16AsmPrinter::printCCOperand(const MachineInstr *MI, int opNum) {
  int CC = (int)MI->getOperand(opNum).getImm();
  O << PIC16CondCodeToString((PIC16CC::CondCodes)CC);
}

// This function is used to sort the decls list.
// should return true if s1 should come before s2.
static bool is_before(const char *s1, const char *s2) {
  return strcmp(s1, s2) <= 0;
}

// This is used by list::unique below. 
// unique will filter out duplicates if it knows them.
static bool is_duplicate(const char *s1, const char *s2) {
  return !strcmp(s1, s2);
}

/// printLibcallDecls - print the extern declarations for compiler 
/// intrinsics.
///
void PIC16AsmPrinter::printLibcallDecls() {
  // If no libcalls used, return.
  if (LibcallDecls.empty()) return;

  O << TAI->getCommentString() << "External decls for libcalls - BEGIN." <<"\n";
  // Remove duplicate entries.
  LibcallDecls.sort(is_before);
  LibcallDecls.unique(is_duplicate);

  for (std::list<const char*>::const_iterator I = LibcallDecls.begin(); 
       I != LibcallDecls.end(); I++) {
    O << TAI->getExternDirective() << *I << "\n";
    O << TAI->getExternDirective() << PAN::getArgsLabel(*I) << "\n";
    O << TAI->getExternDirective() << PAN::getRetvalLabel(*I) << "\n";
  }
  O << TAI->getCommentString() << "External decls for libcalls - END." <<"\n";
}

/// doInitialization - Perfrom Module level initializations here.
/// One task that we do here is to sectionize all global variables.
/// The MemSelOptimizer pass depends on the sectionizing.
///
bool PIC16AsmPrinter::doInitialization(Module &M) {
  bool Result = AsmPrinter::doInitialization(M);

  // FIXME:: This is temporary solution to generate the include file.
  // The processor should be passed to llc as in input and the header file
  // should be generated accordingly.
  O << "\n\t#include P16F1937.INC\n";

  // Set the section names for all globals.
  for (Module::global_iterator I = M.global_begin(), E = M.global_end();
       I != E; ++I)
    if (!I->isDeclaration() && !I->hasAvailableExternallyLinkage()) {
      const MCSection *S = getObjFileLowering().SectionForGlobal(I, Mang, TM);
      
      I->setSection(((const MCSectionPIC16*)S)->getName());
    }

  DbgInfo.BeginModule(M);
  EmitFunctionDecls(M);
  EmitUndefinedVars(M);
  EmitDefinedVars(M);
  EmitIData(M);
  EmitUData(M);
  EmitRomData(M);
  return Result;
}

/// Emit extern decls for functions imported from other modules, and emit
/// global declarations for function defined in this module and which are
/// available to other modules.
///
void PIC16AsmPrinter::EmitFunctionDecls(Module &M) {
 // Emit declarations for external functions.
  O <<"\n"<<TAI->getCommentString() << "Function Declarations - BEGIN." <<"\n";
  for (Module::iterator I = M.begin(), E = M.end(); I != E; I++) {
    if (I->isIntrinsic())
      continue;

    std::string Name = Mang->getMangledName(I);
    if (Name.compare("@abort") == 0)
      continue;
    
    if (!I->isDeclaration() && !I->hasExternalLinkage())
      continue;

    // Do not emit memcpy, memset, and memmove here.
    // Calls to these routines can be generated in two ways,
    // 1. User calling the standard lib function
    // 2. Codegen generating these calls for llvm intrinsics.
    // In the first case a prototype is alread availale, while in
    // second case the call is via and externalsym and the prototype is missing.
    // So declarations for these are currently always getting printing by
    // tracking both kind of references in printInstrunction.
    if (I->isDeclaration() && PAN::isMemIntrinsic(Name)) continue;

    const char *directive = I->isDeclaration() ? TAI->getExternDirective() :
                                                 TAI->getGlobalDirective();
      
    O << directive << Name << "\n";
    O << directive << PAN::getRetvalLabel(Name) << "\n";
    O << directive << PAN::getArgsLabel(Name) << "\n";
  }

  O << TAI->getCommentString() << "Function Declarations - END." <<"\n";
}

// Emit variables imported from other Modules.
void PIC16AsmPrinter::EmitUndefinedVars(Module &M) {
  std::vector<const GlobalVariable*> Items = PTOF->ExternalVarDecls->Items;
  if (!Items.size()) return;

  O << "\n" << TAI->getCommentString() << "Imported Variables - BEGIN" << "\n";
  for (unsigned j = 0; j < Items.size(); j++) {
    O << TAI->getExternDirective() << Mang->getMangledName(Items[j]) << "\n";
  }
  O << TAI->getCommentString() << "Imported Variables - END" << "\n";
}

// Emit variables defined in this module and are available to other modules.
void PIC16AsmPrinter::EmitDefinedVars(Module &M) {
  std::vector<const GlobalVariable*> Items = PTOF->ExternalVarDefs->Items;
  if (!Items.size()) return;

  O << "\n" << TAI->getCommentString() << "Exported Variables - BEGIN" << "\n";
  for (unsigned j = 0; j < Items.size(); j++) {
    O << TAI->getGlobalDirective() << Mang->getMangledName(Items[j]) << "\n";
  }
  O <<  TAI->getCommentString() << "Exported Variables - END" << "\n";
}

// Emit initialized data placed in ROM.
void PIC16AsmPrinter::EmitRomData(Module &M) {
  // Print ROM Data section.
  const std::vector<PIC16Section*> &ROSections = PTOF->ROSections;
  for (unsigned i = 0; i < ROSections.size(); i++) {
    const std::vector<const GlobalVariable*> &Items = ROSections[i]->Items;
    if (!Items.size()) continue;
    O << "\n";
    SwitchToSection(PTOF->ROSections[i]->S_);
    for (unsigned j = 0; j < Items.size(); j++) {
      O << Mang->getMangledName(Items[j]);
      Constant *C = Items[j]->getInitializer();
      int AddrSpace = Items[j]->getType()->getAddressSpace();
      EmitGlobalConstant(C, AddrSpace);
    }
  }
}

bool PIC16AsmPrinter::doFinalization(Module &M) {
  printLibcallDecls();
  EmitRemainingAutos();
  DbgInfo.EndModule(M);
  O << "\n\t" << "END\n";
  return AsmPrinter::doFinalization(M);
}

void PIC16AsmPrinter::EmitFunctionFrame(MachineFunction &MF) {
  const Function *F = MF.getFunction();
  std::string FuncName = Mang->getMangledName(F);
  const TargetData *TD = TM.getTargetData();
  // Emit the data section name.
  O << "\n"; 
  
  const MCSection *fPDataSection =
    getObjFileLowering().getSectionForFunctionFrame(CurrentFnName);
  SwitchToSection(fPDataSection);
  
  // Emit function frame label
  O << PAN::getFrameLabel(CurrentFnName) << ":\n";

  const Type *RetType = F->getReturnType();
  unsigned RetSize = 0; 
  if (RetType->getTypeID() != Type::VoidTyID) 
    RetSize = TD->getTypeAllocSize(RetType);
  
  //Emit function return value space
  // FIXME: Do not emit RetvalLable when retsize is zero. To do this
  // we will need to avoid printing a global directive for Retval label
  // in emitExternandGloblas.
  if(RetSize > 0)
     O << PAN::getRetvalLabel(CurrentFnName) << " RES " << RetSize << "\n";
  else
     O << PAN::getRetvalLabel(CurrentFnName) << ": \n";
   
  // Emit variable to hold the space for function arguments 
  unsigned ArgSize = 0;
  for (Function::const_arg_iterator argi = F->arg_begin(),
           arge = F->arg_end(); argi != arge ; ++argi) {
    const Type *Ty = argi->getType();
    ArgSize += TD->getTypeAllocSize(Ty);
   }

  O << PAN::getArgsLabel(CurrentFnName) << " RES " << ArgSize << "\n";

  // Emit temporary space
  int TempSize = PTLI->GetTmpSize();
  if (TempSize > 0)
    O << PAN::getTempdataLabel(CurrentFnName) << " RES  " << TempSize << '\n';
}

void PIC16AsmPrinter::EmitIData(Module &M) {

  // Print all IDATA sections.
  const std::vector<PIC16Section*> &IDATASections = PTOF->IDATASections;
  for (unsigned i = 0; i < IDATASections.size(); i++) {
    O << "\n";
    if (IDATASections[i]->S_->getName().find("llvm.") != std::string::npos)
      continue;
    SwitchToSection(IDATASections[i]->S_);
    std::vector<const GlobalVariable*> Items = IDATASections[i]->Items;
    for (unsigned j = 0; j < Items.size(); j++) {
      std::string Name = Mang->getMangledName(Items[j]);
      Constant *C = Items[j]->getInitializer();
      int AddrSpace = Items[j]->getType()->getAddressSpace();
      O << Name;
      EmitGlobalConstant(C, AddrSpace);
    }
  }
}

void PIC16AsmPrinter::EmitUData(Module &M) {
  const TargetData *TD = TM.getTargetData();

  // Print all BSS sections.
  const std::vector<PIC16Section*> &BSSSections = PTOF->BSSSections;
  for (unsigned i = 0; i < BSSSections.size(); i++) {
    O << "\n";
    SwitchToSection(BSSSections[i]->S_);
    std::vector<const GlobalVariable*> Items = BSSSections[i]->Items;
    for (unsigned j = 0; j < Items.size(); j++) {
      std::string Name = Mang->getMangledName(Items[j]);
      Constant *C = Items[j]->getInitializer();
      const Type *Ty = C->getType();
      unsigned Size = TD->getTypeAllocSize(Ty);

      O << Name << " RES " << Size << "\n";
    }
  }
}

void PIC16AsmPrinter::EmitAutos(std::string FunctName) {
  // Section names for all globals are already set.
  const TargetData *TD = TM.getTargetData();

  // Now print Autos section for this function.
  std::string SectionName = PAN::getAutosSectionName(FunctName);
  const std::vector<PIC16Section*> &AutosSections = PTOF->AutosSections;
  for (unsigned i = 0; i < AutosSections.size(); i++) {
    O << "\n";
    if (AutosSections[i]->S_->getName() == SectionName) { 
      // Set the printing status to true
      AutosSections[i]->setPrintedStatus(true);
      SwitchToSection(AutosSections[i]->S_);
      const std::vector<const GlobalVariable*> &Items = AutosSections[i]->Items;
      for (unsigned j = 0; j < Items.size(); j++) {
        std::string VarName = Mang->getMangledName(Items[j]);
        Constant *C = Items[j]->getInitializer();
        const Type *Ty = C->getType();
        unsigned Size = TD->getTypeAllocSize(Ty);
        // Emit memory reserve directive.
        O << VarName << "  RES  " << Size << "\n";
      }
      break;
    }
  }
}

// Print autos that were not printed during the code printing of functions.
// As the functions might themselves would have got deleted by the optimizer.
void PIC16AsmPrinter::EmitRemainingAutos() {
  const TargetData *TD = TM.getTargetData();

  // Now print Autos section for this function.
  std::vector <PIC16Section *>AutosSections = PTOF->AutosSections;
  for (unsigned i = 0; i < AutosSections.size(); i++) {
    
    // if the section is already printed then don't print again
    if (AutosSections[i]->isPrinted()) 
      continue;

    // Set status as printed
    AutosSections[i]->setPrintedStatus(true);

    O << "\n";
    SwitchToSection(AutosSections[i]->S_);
    const std::vector<const GlobalVariable*> &Items = AutosSections[i]->Items;
    for (unsigned j = 0; j < Items.size(); j++) {
      std::string VarName = Mang->getMangledName(Items[j]);
      Constant *C = Items[j]->getInitializer();
      const Type *Ty = C->getType();
      unsigned Size = TD->getTypeAllocSize(Ty);
      // Emit memory reserve directive.
      O << VarName << "  RES  " << Size << "\n";
    }
  }
}


extern "C" void LLVMInitializePIC16Target() { 
  // Register the targets
  RegisterTargetMachine<PIC16TargetMachine> A(ThePIC16Target);  
  RegisterTargetMachine<CooperTargetMachine> B(TheCooperTarget);
  RegisterAsmPrinter<PIC16AsmPrinter> C(ThePIC16Target);
  RegisterAsmPrinter<PIC16AsmPrinter> D(TheCooperTarget);

  RegisterAsmInfo<PIC16TargetAsmInfo> E(ThePIC16Target);
  RegisterAsmInfo<PIC16TargetAsmInfo> F(TheCooperTarget);
}
