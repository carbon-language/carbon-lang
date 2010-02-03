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
#include "llvm/CodeGen/DwarfWriter.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/DwarfWriter.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Target/TargetRegistry.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormattedStream.h"
#include <cstring>
using namespace llvm;

#include "PIC16GenAsmWriter.inc"

PIC16AsmPrinter::PIC16AsmPrinter(formatted_raw_ostream &O, TargetMachine &TM,
                                 MCContext &Ctx, MCStreamer &Streamer,
                                 const MCAsmInfo *T)
: AsmPrinter(O, TM, Ctx, Streamer, T), DbgInfo(O, T) {
  PTLI = static_cast<PIC16TargetLowering*>(TM.getTargetLowering());
  PMAI = static_cast<const PIC16MCAsmInfo*>(T);
  PTOF = (PIC16TargetObjectFile *)&PTLI->getObjFileLowering();
}

void PIC16AsmPrinter::EmitInstruction(const MachineInstr *MI) {
  printInstruction(MI);
  O << '\n';
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
  const MCSection *fCodeSection 
    = getObjFileLowering().SectionForCode(CurrentFnSym->getName());

  // Start the Code Section.
  O <<  "\n";
  OutStreamer.SwitchSection(fCodeSection);

  // Emit the frame address of the function at the beginning of code.
  O << "\tretlw  low(" << PAN::getFrameLabel(CurrentFnSym->getName()) << ")\n";
  O << "\tretlw  high(" << PAN::getFrameLabel(CurrentFnSym->getName()) << ")\n";

  // Emit function start label.
  O << *CurrentFnSym << ":\n";

  DebugLoc CurDL;
  O << "\n"; 
  // Print out code for the function.
  for (MachineFunction::const_iterator I = MF.begin(), E = MF.end();
       I != E; ++I) {

    // Print a label for the basic block.
    if (I != MF.begin()) {
      EmitBasicBlockStart(I);
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
      EmitInstruction(II);
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
      {
        // For indirect load/store insns, the fsr name is printed as INDF.
        std::string RegName = getRegisterName(MO.getReg());
        if ((MI->getOpcode() == PIC16::load_indirect) ||
            (MI->getOpcode() == PIC16::store_indirect))
        {
          RegName.replace (0, 3, "INDF");
        }
        O << RegName;
      }
      return;

    case MachineOperand::MO_Immediate:
      O << (int)MO.getImm();
      return;

    case MachineOperand::MO_GlobalAddress: {
      MCSymbol *Sym = GetGlobalValueSymbol(MO.getGlobal());
      // FIXME: currently we do not have a memcpy def coming in the module
      // by any chance, as we do not link in those as .bc lib. So these calls
      // are always external and it is safe to emit an extern.
      if (PAN::isMemIntrinsic(Sym->getName()))
        LibcallDecls.push_back(createESName(Sym->getName()));

      O << *Sym;
      break;
    }
    case MachineOperand::MO_ExternalSymbol: {
       const char *Sname = MO.getSymbolName();

      // If its a libcall name, record it to decls section.
      if (PAN::getSymbolTag(Sname) == PAN::LIBCALL)
        LibcallDecls.push_back(Sname);

      // Record a call to intrinsic to print the extern declaration for it.
      std::string Sym = Sname;  
      if (PAN::isMemIntrinsic(Sym)) {
        Sym = PAN::addPrefix(Sym);
        LibcallDecls.push_back(createESName(Sym));
      }

      O << Sym;
      break;
    }
    case MachineOperand::MO_MachineBasicBlock:
      O << *MO.getMBB()->getSymbol(OutContext);
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

  O << MAI->getCommentString() << "External decls for libcalls - BEGIN." <<"\n";
  // Remove duplicate entries.
  LibcallDecls.sort(is_before);
  LibcallDecls.unique(is_duplicate);

  for (std::list<const char*>::const_iterator I = LibcallDecls.begin(); 
       I != LibcallDecls.end(); I++) {
    O << MAI->getExternDirective() << *I << "\n";
    O << MAI->getExternDirective() << PAN::getArgsLabel(*I) << "\n";
    O << MAI->getExternDirective() << PAN::getRetvalLabel(*I) << "\n";
  }
  O << MAI->getCommentString() << "External decls for libcalls - END." <<"\n";
}

/// doInitialization - Perform Module level initializations here.
/// One task that we do here is to sectionize all global variables.
/// The MemSelOptimizer pass depends on the sectionizing.
///
bool PIC16AsmPrinter::doInitialization(Module &M) {
  bool Result = AsmPrinter::doInitialization(M);

  // Every asmbly contains these std headers. 
  O << "\n#include p16f1xxx.inc";
  O << "\n#include stdmacros.inc";

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
  O <<"\n"<<MAI->getCommentString() << "Function Declarations - BEGIN." <<"\n";
  for (Module::iterator I = M.begin(), E = M.end(); I != E; I++) {
    if (I->isIntrinsic() || I->getName() == "@abort")
      continue;
    
    if (!I->isDeclaration() && !I->hasExternalLinkage())
      continue;

    MCSymbol *Sym = GetGlobalValueSymbol(I);
    
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
      
    O << directive << Sym->getName() << "\n";
    O << directive << PAN::getRetvalLabel(Sym->getName()) << "\n";
    O << directive << PAN::getArgsLabel(Sym->getName()) << "\n";
  }

  O << MAI->getCommentString() << "Function Declarations - END." <<"\n";
}

// Emit variables imported from other Modules.
void PIC16AsmPrinter::EmitUndefinedVars(Module &M) {
  std::vector<const GlobalVariable*> Items = ExternalVarDecls;
  if (!Items.size()) return;

  O << "\n" << MAI->getCommentString() << "Imported Variables - BEGIN" << "\n";
  for (unsigned j = 0; j < Items.size(); j++)
    O << MAI->getExternDirective() << *GetGlobalValueSymbol(Items[j]) << "\n";
  O << MAI->getCommentString() << "Imported Variables - END" << "\n";
}

// Emit variables defined in this module and are available to other modules.
void PIC16AsmPrinter::EmitDefinedVars(Module &M) {
  std::vector<const GlobalVariable*> Items = ExternalVarDefs;
  if (!Items.size()) return;

  O << "\n" << MAI->getCommentString() << "Exported Variables - BEGIN" << "\n";
  for (unsigned j = 0; j < Items.size(); j++)
    O << MAI->getGlobalDirective() << *GetGlobalValueSymbol(Items[j]) << "\n";
  O <<  MAI->getCommentString() << "Exported Variables - END" << "\n";
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
  O << "\n\t" << "END\n";
  return AsmPrinter::doFinalization(M);
}

void PIC16AsmPrinter::EmitFunctionFrame(MachineFunction &MF) {
  const Function *F = MF.getFunction();
  const TargetData *TD = TM.getTargetData();
  // Emit the data section name.
  O << "\n"; 
  
  PIC16Section *fPDataSection =
    const_cast<PIC16Section *>(getObjFileLowering().
                                SectionForFrame(CurrentFnSym->getName()));
 
  fPDataSection->setColor(getFunctionColor(F)); 
  OutStreamer.SwitchSection(fPDataSection);
  
  // Emit function frame label
  O << PAN::getFrameLabel(CurrentFnSym->getName()) << ":\n";

  const Type *RetType = F->getReturnType();
  unsigned RetSize = 0; 
  if (RetType->getTypeID() != Type::VoidTyID) 
    RetSize = TD->getTypeAllocSize(RetType);
  
  //Emit function return value space
  // FIXME: Do not emit RetvalLable when retsize is zero. To do this
  // we will need to avoid printing a global directive for Retval label
  // in emitExternandGloblas.
  if(RetSize > 0)
     O << PAN::getRetvalLabel(CurrentFnSym->getName())
       << " RES " << RetSize << "\n";
  else
     O << PAN::getRetvalLabel(CurrentFnSym->getName()) << ": \n";
   
  // Emit variable to hold the space for function arguments 
  unsigned ArgSize = 0;
  for (Function::const_arg_iterator argi = F->arg_begin(),
           arge = F->arg_end(); argi != arge ; ++argi) {
    const Type *Ty = argi->getType();
    ArgSize += TD->getTypeAllocSize(Ty);
   }

  O << PAN::getArgsLabel(CurrentFnSym->getName()) << " RES " << ArgSize << "\n";

  // Emit temporary space
  int TempSize = PTLI->GetTmpSize();
  if (TempSize > 0)
    O << PAN::getTempdataLabel(CurrentFnSym->getName()) << " RES  "
      << TempSize << '\n';
}


void PIC16AsmPrinter::EmitInitializedDataSection(const PIC16Section *S) {
  /// Emit Section header.
  OutStreamer.SwitchSection(S);

    std::vector<const GlobalVariable*> Items = S->Items;
    for (unsigned j = 0; j < Items.size(); j++) {
      Constant *C = Items[j]->getInitializer();
      int AddrSpace = Items[j]->getType()->getAddressSpace();
      O << *GetGlobalValueSymbol(Items[j]);
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
      O << *GetGlobalValueSymbol(Items[j]) << " RES " << Size << "\n";
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
    O << "\n";
    EmitSingleSection(SList[i]);
  }
}

