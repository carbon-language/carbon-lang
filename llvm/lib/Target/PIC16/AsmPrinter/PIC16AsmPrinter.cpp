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

#include "PIC16ABINames.h"
#include "PIC16AsmPrinter.h"
#include "PIC16Section.h"
#include "PIC16MCAsmInfo.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/Module.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Target/Mangler.h"
#include "llvm/Target/TargetRegistry.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/SmallString.h"
#include <cstring>
using namespace llvm;

#include "PIC16GenAsmWriter.inc"

PIC16AsmPrinter::PIC16AsmPrinter(TargetMachine &TM, MCStreamer &Streamer)
: AsmPrinter(TM, Streamer), DbgInfo(Streamer, TM.getMCAsmInfo()) {
  PTLI = static_cast<PIC16TargetLowering*>(TM.getTargetLowering());
  PMAI = static_cast<const PIC16MCAsmInfo*>(TM.getMCAsmInfo());
  PTOF = (PIC16TargetObjectFile *)&PTLI->getObjFileLowering();
}

void PIC16AsmPrinter::EmitInstruction(const MachineInstr *MI) {
  SmallString<128> Str;
  raw_svector_ostream OS(Str);
  printInstruction(MI, OS);
  
  OutStreamer.EmitRawText(OS.str());
}

static int getFunctionColor(const Function *F) {
  if (F->hasSection()) {
    std::string Sectn = F->getSection();
    std::string StrToFind = "Overlay=";
    std::string::size_type Pos = Sectn.find(StrToFind);

    // Retreive the color number if the key is found.
    if (Pos != std::string::npos) {
      Pos += StrToFind.length();
      std::string Color = "";
      char c = Sectn.at(Pos);
      // A Color can only consist of digits.
      while (c >= '0' && c<= '9') {
        Color.append(1,c);
        Pos++;
        if (Pos >= Sectn.length())
          break;
        c = Sectn.at(Pos);
      }
      return atoi(Color.c_str());
    }
  }

  // Color was not set for function, so return -1.
  return -1;
}

// Color the Auto section of the given function. 
void PIC16AsmPrinter::ColorAutoSection(const Function *F) {
  std::string SectionName = PAN::getAutosSectionName(CurrentFnSym->getName());
  PIC16Section* Section = PTOF->findPIC16Section(SectionName);
  if (Section != NULL) {
    int Color = getFunctionColor(F);
    if (Color >= 0)
      Section->setColor(Color);
  }
}


/// runOnMachineFunction - This emits the frame section, autos section and 
/// assembly for each instruction. Also takes care of function begin debug
/// directive and file begin debug directive (if required) for the function.
///
bool PIC16AsmPrinter::runOnMachineFunction(MachineFunction &MF) {
  // This calls the base class function required to be called at beginning
  // of runOnMachineFunction.
  SetupMachineFunction(MF);

  // Put the color information from function to its auto section.
  const Function *F = MF.getFunction();
  ColorAutoSection(F);

  // Emit the function frame (args and temps).
  EmitFunctionFrame(MF);

  DbgInfo.BeginFunction(MF);

  // Now emit the instructions of function in its code section.
  const MCSection *fCodeSection = 
    getObjFileLowering().SectionForCode(CurrentFnSym->getName(), 
                                        PAN::isISR(F->getSection()));

  // Start the Code Section.
  OutStreamer.SwitchSection(fCodeSection);

  // Emit the frame address of the function at the beginning of code.
  OutStreamer.EmitRawText("\tretlw  low(" + 
                          Twine(PAN::getFrameLabel(CurrentFnSym->getName())) +
                          ")");
  OutStreamer.EmitRawText("\tretlw  high(" +
                          Twine(PAN::getFrameLabel(CurrentFnSym->getName())) +
                          ")");

  // Emit function start label.
  OutStreamer.EmitLabel(CurrentFnSym);

  DebugLoc CurDL;
  // Print out code for the function.
  for (MachineFunction::const_iterator I = MF.begin(), E = MF.end();
       I != E; ++I) {

    // Print a label for the basic block.
    if (I != MF.begin())
      EmitBasicBlockStart(I);
    
    // Print a basic block.
    for (MachineBasicBlock::const_iterator II = I->begin(), E = I->end();
         II != E; ++II) {
      // Emit the line directive if source line changed.
      DebugLoc DL = II->getDebugLoc();
      if (!DL.isUnknown() && DL != CurDL) {
        DbgInfo.ChangeDebugLoc(MF, DL);
        CurDL = DL;
      }
        
      // Print the assembly for the instruction.
      EmitInstruction(II);
    }
  }
  
  // Emit function end debug directives.
  DbgInfo.EndFunction(MF);

  return false;  // we didn't modify anything.
}


// printOperand - print operand of insn.
void PIC16AsmPrinter::printOperand(const MachineInstr *MI, int opNum,
                                   raw_ostream &O) {
  const MachineOperand &MO = MI->getOperand(opNum);
  const Function *F = MI->getParent()->getParent()->getFunction();

  switch (MO.getType()) {
    case MachineOperand::MO_Register:
      {
        // For indirect load/store insns, the fsr name is printed as INDF.
        std::string RegName = getRegisterName(MO.getReg());
        if ((MI->getOpcode() == PIC16::load_indirect) ||
            (MI->getOpcode() == PIC16::store_indirect))
          RegName.replace (0, 3, "INDF");
        O << RegName;
      }
      return;

    case MachineOperand::MO_Immediate:
      O << (int)MO.getImm();
      return;

    case MachineOperand::MO_GlobalAddress: {
      MCSymbol *Sym = Mang->getSymbol(MO.getGlobal());
      // FIXME: currently we do not have a memcpy def coming in the module
      // by any chance, as we do not link in those as .bc lib. So these calls
      // are always external and it is safe to emit an extern.
      if (PAN::isMemIntrinsic(Sym->getName()))
        LibcallDecls.insert(Sym->getName());

      O << *Sym;
      break;
    }
    case MachineOperand::MO_ExternalSymbol: {
       const char *Sname = MO.getSymbolName();
       std::string Printname = Sname;

      // Intrinsic stuff needs to be renamed if we are printing IL fn. 
      if (PAN::isIntrinsicStuff(Printname)) {
        if (PAN::isISR(F->getSection())) {
          Printname = PAN::Rename(Sname);
        }
        // Record these decls, we need to print them in asm as extern.
        LibcallDecls.insert(Printname);
      }

      O << Printname;
      break;
    }
    case MachineOperand::MO_MachineBasicBlock:
      O << *MO.getMBB()->getSymbol();
      return;

    default:
      llvm_unreachable(" Operand type not supported.");
  }
}

/// printCCOperand - Print the cond code operand.
///
void PIC16AsmPrinter::printCCOperand(const MachineInstr *MI, int opNum,
                                     raw_ostream &O) {
  int CC = (int)MI->getOperand(opNum).getImm();
  O << PIC16CondCodeToString((PIC16CC::CondCodes)CC);
}

/// printLibcallDecls - print the extern declarations for compiler 
/// intrinsics.
///
void PIC16AsmPrinter::printLibcallDecls() {
  // If no libcalls used, return.
  if (LibcallDecls.empty()) return;

  OutStreamer.AddComment("External decls for libcalls - BEGIN");
  OutStreamer.AddBlankLine();

  for (std::set<std::string>::const_iterator I = LibcallDecls.begin(),
       E = LibcallDecls.end(); I != E; I++)
    OutStreamer.EmitRawText(MAI->getExternDirective() + Twine(*I));

  OutStreamer.AddComment("External decls for libcalls - END");
  OutStreamer.AddBlankLine();
}

/// doInitialization - Perform Module level initializations here.
/// One task that we do here is to sectionize all global variables.
/// The MemSelOptimizer pass depends on the sectionizing.
///
bool PIC16AsmPrinter::doInitialization(Module &M) {
  bool Result = AsmPrinter::doInitialization(M);

  // Every asmbly contains these std headers. 
  OutStreamer.EmitRawText(StringRef("\n#include p16f1xxx.inc"));
  OutStreamer.EmitRawText(StringRef("#include stdmacros.inc"));

  // Set the section names for all globals.
  for (Module::global_iterator I = M.global_begin(), E = M.global_end();
       I != E; ++I) {

    // Record External Var Decls.
    if (I->isDeclaration()) {
      ExternalVarDecls.push_back(I);
      continue;
    }

    // Record Exteranl Var Defs.
    if (I->hasExternalLinkage() || I->hasCommonLinkage()) {
      ExternalVarDefs.push_back(I);
    }

    // Sectionify actual data.
    if (!I->hasAvailableExternallyLinkage()) {
      const MCSection *S = getObjFileLowering().SectionForGlobal(I, Mang, TM);
      
      I->setSection(((const PIC16Section *)S)->getName());
    }
  }

  DbgInfo.BeginModule(M);
  EmitFunctionDecls(M);
  EmitUndefinedVars(M);
  EmitDefinedVars(M);
  EmitIData(M);
  EmitUData(M);
  EmitRomData(M);
  EmitSharedUdata(M);
  EmitUserSections(M);
  return Result;
}

/// Emit extern decls for functions imported from other modules, and emit
/// global declarations for function defined in this module and which are
/// available to other modules.
///
void PIC16AsmPrinter::EmitFunctionDecls(Module &M) {
 // Emit declarations for external functions.
  OutStreamer.AddComment("Function Declarations - BEGIN");
  OutStreamer.AddBlankLine();
  for (Module::iterator I = M.begin(), E = M.end(); I != E; I++) {
    if (I->isIntrinsic() || I->getName() == "@abort")
      continue;
    
    if (!I->isDeclaration() && !I->hasExternalLinkage())
      continue;

    MCSymbol *Sym = Mang->getSymbol(I);
    
    // Do not emit memcpy, memset, and memmove here.
    // Calls to these routines can be generated in two ways,
    // 1. User calling the standard lib function
    // 2. Codegen generating these calls for llvm intrinsics.
    // In the first case a prototype is alread availale, while in
    // second case the call is via and externalsym and the prototype is missing.
    // So declarations for these are currently always getting printing by
    // tracking both kind of references in printInstrunction.
    if (I->isDeclaration() && PAN::isMemIntrinsic(Sym->getName())) continue;

    const char *directive = I->isDeclaration() ? MAI->getExternDirective() :
                                                 MAI->getGlobalDirective();
      
    OutStreamer.EmitRawText(directive + Twine(Sym->getName()));
    OutStreamer.EmitRawText(directive +
                            Twine(PAN::getRetvalLabel(Sym->getName())));
    OutStreamer.EmitRawText(directive +
                            Twine(PAN::getArgsLabel(Sym->getName())));
  }

  OutStreamer.AddComment("Function Declarations - END");
  OutStreamer.AddBlankLine();

}

// Emit variables imported from other Modules.
void PIC16AsmPrinter::EmitUndefinedVars(Module &M) {
  std::vector<const GlobalVariable*> Items = ExternalVarDecls;
  if (!Items.size()) return;

  OutStreamer.AddComment("Imported Variables - BEGIN");
  OutStreamer.AddBlankLine();
  for (unsigned j = 0; j < Items.size(); j++)
    OutStreamer.EmitRawText(MAI->getExternDirective() +
                            Twine(Mang->getSymbol(Items[j])->getName()));
  
  OutStreamer.AddComment("Imported Variables - END");
  OutStreamer.AddBlankLine();
}

// Emit variables defined in this module and are available to other modules.
void PIC16AsmPrinter::EmitDefinedVars(Module &M) {
  std::vector<const GlobalVariable*> Items = ExternalVarDefs;
  if (!Items.size()) return;

  OutStreamer.AddComment("Exported Variables - BEGIN");
  OutStreamer.AddBlankLine();

  for (unsigned j = 0; j < Items.size(); j++)
    OutStreamer.EmitRawText(MAI->getGlobalDirective() +
                            Twine(Mang->getSymbol(Items[j])->getName()));
  OutStreamer.AddComment("Exported Variables - END");
  OutStreamer.AddBlankLine();
}

// Emit initialized data placed in ROM.
void PIC16AsmPrinter::EmitRomData(Module &M) {
  EmitSingleSection(PTOF->ROMDATASection());
}

// Emit Shared section udata.
void PIC16AsmPrinter::EmitSharedUdata(Module &M) {
  EmitSingleSection(PTOF->SHAREDUDATASection());
}

bool PIC16AsmPrinter::doFinalization(Module &M) {
  EmitAllAutos(M);
  printLibcallDecls();
  DbgInfo.EndModule(M);
  OutStreamer.EmitRawText(StringRef("\tEND"));
  return AsmPrinter::doFinalization(M);
}

void PIC16AsmPrinter::EmitFunctionFrame(MachineFunction &MF) {
  const Function *F = MF.getFunction();
  const TargetData *TD = TM.getTargetData();
  // Emit the data section name.
  
  PIC16Section *fPDataSection =
    const_cast<PIC16Section *>(getObjFileLowering().
                                SectionForFrame(CurrentFnSym->getName()));
 
  fPDataSection->setColor(getFunctionColor(F)); 
  OutStreamer.SwitchSection(fPDataSection);
  
  // Emit function frame label
  OutStreamer.EmitRawText(PAN::getFrameLabel(CurrentFnSym->getName()) +
                          Twine(":"));

  const Type *RetType = F->getReturnType();
  unsigned RetSize = 0; 
  if (RetType->getTypeID() != Type::VoidTyID) 
    RetSize = TD->getTypeAllocSize(RetType);
  
  //Emit function return value space
  // FIXME: Do not emit RetvalLable when retsize is zero. To do this
  // we will need to avoid printing a global directive for Retval label
  // in emitExternandGloblas.
  if(RetSize > 0)
     OutStreamer.EmitRawText(PAN::getRetvalLabel(CurrentFnSym->getName()) +
                             Twine(" RES ") + Twine(RetSize));
  else
     OutStreamer.EmitRawText(PAN::getRetvalLabel(CurrentFnSym->getName()) +
                             Twine(":"));
   
  // Emit variable to hold the space for function arguments 
  unsigned ArgSize = 0;
  for (Function::const_arg_iterator argi = F->arg_begin(),
           arge = F->arg_end(); argi != arge ; ++argi) {
    const Type *Ty = argi->getType();
    ArgSize += TD->getTypeAllocSize(Ty);
   }

  OutStreamer.EmitRawText(PAN::getArgsLabel(CurrentFnSym->getName()) +
                          Twine(" RES ") + Twine(ArgSize));

  // Emit temporary space
  int TempSize = PTLI->GetTmpSize();
  if (TempSize > 0)
    OutStreamer.EmitRawText(PAN::getTempdataLabel(CurrentFnSym->getName()) +
                            Twine(" RES  ") + Twine(TempSize));
}


void PIC16AsmPrinter::EmitInitializedDataSection(const PIC16Section *S) {
  /// Emit Section header.
  OutStreamer.SwitchSection(S);

    std::vector<const GlobalVariable*> Items = S->Items;
    for (unsigned j = 0; j < Items.size(); j++) {
      Constant *C = Items[j]->getInitializer();
      int AddrSpace = Items[j]->getType()->getAddressSpace();
      OutStreamer.EmitRawText(Mang->getSymbol(Items[j])->getName());
      EmitGlobalConstant(C, AddrSpace);
   }
}

// Print all IDATA sections.
void PIC16AsmPrinter::EmitIData(Module &M) {
  EmitSectionList (M, PTOF->IDATASections());
}

void PIC16AsmPrinter::
EmitUninitializedDataSection(const PIC16Section *S) {
    const TargetData *TD = TM.getTargetData();
    OutStreamer.SwitchSection(S);
    std::vector<const GlobalVariable*> Items = S->Items;
    for (unsigned j = 0; j < Items.size(); j++) {
      Constant *C = Items[j]->getInitializer();
      const Type *Ty = C->getType();
      unsigned Size = TD->getTypeAllocSize(Ty);
      OutStreamer.EmitRawText(Mang->getSymbol(Items[j])->getName() +
                              Twine(" RES ") + Twine(Size));
    }
}

// Print all UDATA sections.
void PIC16AsmPrinter::EmitUData(Module &M) {
  EmitSectionList (M, PTOF->UDATASections());
}

// Print all USER sections.
void PIC16AsmPrinter::EmitUserSections(Module &M) {
  EmitSectionList (M, PTOF->USERSections());
}

// Print all AUTO sections.
void PIC16AsmPrinter::EmitAllAutos(Module &M) {
  EmitSectionList (M, PTOF->AUTOSections());
}

extern "C" void LLVMInitializePIC16AsmPrinter() { 
  RegisterAsmPrinter<PIC16AsmPrinter> X(ThePIC16Target);
}

// Emit one data section using correct section emitter based on section type.
void PIC16AsmPrinter::EmitSingleSection(const PIC16Section *S) {
  if (S == NULL) return;

  switch (S->getType()) {
    default: llvm_unreachable ("unknow user section type");
    case UDATA:
    case UDATA_SHR:
    case UDATA_OVR:
      EmitUninitializedDataSection(S);
      break;
    case IDATA:
    case ROMDATA:
      EmitInitializedDataSection(S);
      break;
  }
}

// Emit a list of sections.
void PIC16AsmPrinter::
EmitSectionList(Module &M, const std::vector<PIC16Section *> &SList) {
  for (unsigned i = 0; i < SList.size(); i++) {
    // Exclude llvm specific metadata sections.
    if (SList[i]->getName().find("llvm.") != std::string::npos)
      continue;
    OutStreamer.AddBlankLine();
    EmitSingleSection(SList[i]);
  }
}

