//===-- AsmPrinter.cpp - Common AsmPrinter code ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the AsmPrinter class.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "asm-printer"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Constants.h"
#include "llvm/Module.h"
#include "llvm/CodeGen/DwarfWriter.h"
#include "llvm/CodeGen/GCMetadataPrinter.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/Analysis/DebugInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/Target/Mangler.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormattedStream.h"
#include <cerrno>
using namespace llvm;

STATISTIC(EmittedInsts, "Number of machine instrs printed");

char AsmPrinter::ID = 0;
AsmPrinter::AsmPrinter(formatted_raw_ostream &o, TargetMachine &tm,
                       const MCAsmInfo *T, bool VerboseAsm)
  : MachineFunctionPass(&ID), O(o),
    TM(tm), MAI(T), TRI(tm.getRegisterInfo()),

    OutContext(*new MCContext()),
    // FIXME: Pass instprinter to streamer.
    OutStreamer(*createAsmStreamer(OutContext, O, *T,
                                   TM.getTargetData()->isLittleEndian(),
                                   VerboseAsm, 0)),

    LastMI(0), LastFn(0), Counter(~0U), PrevDLT(NULL) {
  DW = 0; MMI = 0;
  this->VerboseAsm = VerboseAsm;
}

AsmPrinter::~AsmPrinter() {
  for (gcp_iterator I = GCMetadataPrinters.begin(),
                    E = GCMetadataPrinters.end(); I != E; ++I)
    delete I->second;
  
  delete &OutStreamer;
  delete &OutContext;
}

/// getFunctionNumber - Return a unique ID for the current function.
///
unsigned AsmPrinter::getFunctionNumber() const {
  return MF->getFunctionNumber();
}

TargetLoweringObjectFile &AsmPrinter::getObjFileLowering() const {
  return TM.getTargetLowering()->getObjFileLowering();
}

/// getCurrentSection() - Return the current section we are emitting to.
const MCSection *AsmPrinter::getCurrentSection() const {
  return OutStreamer.getCurrentSection();
}


void AsmPrinter::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  MachineFunctionPass::getAnalysisUsage(AU);
  AU.addRequired<GCModuleInfo>();
  if (VerboseAsm)
    AU.addRequired<MachineLoopInfo>();
}

bool AsmPrinter::doInitialization(Module &M) {
  // Initialize TargetLoweringObjectFile.
  const_cast<TargetLoweringObjectFile&>(getObjFileLowering())
    .Initialize(OutContext, TM);
  
  Mang = new Mangler(*MAI);
  
  // Allow the target to emit any magic that it wants at the start of the file.
  EmitStartOfAsmFile(M);

  // Very minimal debug info. It is ignored if we emit actual debug info. If we
  // don't, this at least helps the user find where a global came from.
  if (MAI->hasSingleParameterDotFile()) {
    // .file "foo.c"
    OutStreamer.EmitFileDirective(M.getModuleIdentifier());
  }

  GCModuleInfo *MI = getAnalysisIfAvailable<GCModuleInfo>();
  assert(MI && "AsmPrinter didn't require GCModuleInfo?");
  for (GCModuleInfo::iterator I = MI->begin(), E = MI->end(); I != E; ++I)
    if (GCMetadataPrinter *MP = GetOrCreateGCPrinter(*I))
      MP->beginAssembly(O, *this, *MAI);
  
  if (!M.getModuleInlineAsm().empty())
    O << MAI->getCommentString() << " Start of file scope inline assembly\n"
      << M.getModuleInlineAsm()
      << '\n' << MAI->getCommentString()
      << " End of file scope inline assembly\n";

  MMI = getAnalysisIfAvailable<MachineModuleInfo>();
  if (MMI)
    MMI->AnalyzeModule(M);
  DW = getAnalysisIfAvailable<DwarfWriter>();
  if (DW)
    DW->BeginModule(&M, MMI, O, this, MAI);

  return false;
}

void AsmPrinter::EmitLinkage(unsigned Linkage, MCSymbol *GVSym) const {
  switch ((GlobalValue::LinkageTypes)Linkage) {
  case GlobalValue::CommonLinkage:
  case GlobalValue::LinkOnceAnyLinkage:
  case GlobalValue::LinkOnceODRLinkage:
  case GlobalValue::WeakAnyLinkage:
  case GlobalValue::WeakODRLinkage:
  case GlobalValue::LinkerPrivateLinkage:
    if (MAI->getWeakDefDirective() != 0) {
      // .globl _foo
      OutStreamer.EmitSymbolAttribute(GVSym, MCSA_Global);
      // .weak_definition _foo
      OutStreamer.EmitSymbolAttribute(GVSym, MCSA_WeakDefinition);
    } else if (const char *LinkOnce = MAI->getLinkOnceDirective()) {
      // .globl _foo
      OutStreamer.EmitSymbolAttribute(GVSym, MCSA_Global);
      // FIXME: linkonce should be a section attribute, handled by COFF Section
      // assignment.
      // http://sourceware.org/binutils/docs-2.20/as/Linkonce.html#Linkonce
      // .linkonce discard
      // FIXME: It would be nice to use .linkonce samesize for non-common
      // globals.
      O << LinkOnce;
    } else {
      // .weak _foo
      OutStreamer.EmitSymbolAttribute(GVSym, MCSA_Weak);
    }
    break;
  case GlobalValue::DLLExportLinkage:
  case GlobalValue::AppendingLinkage:
    // FIXME: appending linkage variables should go into a section of
    // their name or something.  For now, just emit them as external.
  case GlobalValue::ExternalLinkage:
    // If external or appending, declare as a global symbol.
    // .globl _foo
    OutStreamer.EmitSymbolAttribute(GVSym, MCSA_Global);
    break;
  case GlobalValue::PrivateLinkage:
  case GlobalValue::InternalLinkage:
    break;
  default:
    llvm_unreachable("Unknown linkage type!");
  }
}


/// EmitGlobalVariable - Emit the specified global variable to the .s file.
void AsmPrinter::EmitGlobalVariable(const GlobalVariable *GV) {
  if (!GV->hasInitializer())   // External globals require no code.
    return;
  
  // Check to see if this is a special global used by LLVM, if so, emit it.
  if (EmitSpecialLLVMGlobal(GV))
    return;

  MCSymbol *GVSym = GetGlobalValueSymbol(GV);
  EmitVisibility(GVSym, GV->getVisibility());

  if (MAI->hasDotTypeDotSizeDirective())
    OutStreamer.EmitSymbolAttribute(GVSym, MCSA_ELF_TypeObject);
  
  SectionKind GVKind = TargetLoweringObjectFile::getKindForGlobal(GV, TM);

  const TargetData *TD = TM.getTargetData();
  unsigned Size = TD->getTypeAllocSize(GV->getType()->getElementType());
  unsigned AlignLog = TD->getPreferredAlignmentLog(GV);
  
  // Handle common and BSS local symbols (.lcomm).
  if (GVKind.isCommon() || GVKind.isBSSLocal()) {
    if (Size == 0) Size = 1;   // .comm Foo, 0 is undefined, avoid it.
    
    if (VerboseAsm) {
      WriteAsOperand(OutStreamer.GetCommentOS(), GV,
                     /*PrintType=*/false, GV->getParent());
      OutStreamer.GetCommentOS() << '\n';
    }
    
    // Handle common symbols.
    if (GVKind.isCommon()) {
      // .comm _foo, 42, 4
      OutStreamer.EmitCommonSymbol(GVSym, Size, 1 << AlignLog);
      return;
    }
    
    // Handle local BSS symbols.
    if (MAI->hasMachoZeroFillDirective()) {
      const MCSection *TheSection =
        getObjFileLowering().SectionForGlobal(GV, GVKind, Mang, TM);
      // .zerofill __DATA, __bss, _foo, 400, 5
      OutStreamer.EmitZerofill(TheSection, GVSym, Size, 1 << AlignLog);
      return;
    }
    
    if (MAI->hasLCOMMDirective()) {
      // .lcomm _foo, 42
      OutStreamer.EmitLocalCommonSymbol(GVSym, Size);
      return;
    }
    
    // .local _foo
    OutStreamer.EmitSymbolAttribute(GVSym, MCSA_Local);
    // .comm _foo, 42, 4
    OutStreamer.EmitCommonSymbol(GVSym, Size, 1 << AlignLog);
    return;
  }
  
  const MCSection *TheSection =
    getObjFileLowering().SectionForGlobal(GV, GVKind, Mang, TM);

  // Handle the zerofill directive on darwin, which is a special form of BSS
  // emission.
  if (GVKind.isBSSExtern() && MAI->hasMachoZeroFillDirective()) {
    // .globl _foo
    OutStreamer.EmitSymbolAttribute(GVSym, MCSA_Global);
    // .zerofill __DATA, __common, _foo, 400, 5
    OutStreamer.EmitZerofill(TheSection, GVSym, Size, 1 << AlignLog);
    return;
  }

  OutStreamer.SwitchSection(TheSection);

  EmitLinkage(GV->getLinkage(), GVSym);
  EmitAlignment(AlignLog, GV);

  if (VerboseAsm) {
    WriteAsOperand(OutStreamer.GetCommentOS(), GV,
                   /*PrintType=*/false, GV->getParent());
    OutStreamer.GetCommentOS() << '\n';
  }
  OutStreamer.EmitLabel(GVSym);

  EmitGlobalConstant(GV->getInitializer());

  if (MAI->hasDotTypeDotSizeDirective())
    // .size foo, 42
    OutStreamer.EmitELFSize(GVSym, MCConstantExpr::Create(Size, OutContext));
  
  OutStreamer.AddBlankLine();
}

/// EmitFunctionHeader - This method emits the header for the current
/// function.
void AsmPrinter::EmitFunctionHeader() {
  // Print out constants referenced by the function
  EmitConstantPool();
  
  // Print the 'header' of function.
  const Function *F = MF->getFunction();

  OutStreamer.SwitchSection(getObjFileLowering().SectionForGlobal(F, Mang, TM));
  EmitVisibility(CurrentFnSym, F->getVisibility());

  EmitLinkage(F->getLinkage(), CurrentFnSym);
  EmitAlignment(MF->getAlignment(), F);

  if (MAI->hasDotTypeDotSizeDirective())
    OutStreamer.EmitSymbolAttribute(CurrentFnSym, MCSA_ELF_TypeFunction);

  if (VerboseAsm) {
    WriteAsOperand(OutStreamer.GetCommentOS(), F,
                   /*PrintType=*/false, F->getParent());
    OutStreamer.GetCommentOS() << '\n';
  }

  // Emit the CurrentFnSym.  This is is a virtual function to allow targets to
  // do their wild and crazy things as required.
  EmitFunctionEntryLabel();
  
  // Add some workaround for linkonce linkage on Cygwin\MinGW.
  if (MAI->getLinkOnceDirective() != 0 &&
      (F->hasLinkOnceLinkage() || F->hasWeakLinkage()))
    // FIXME: What is this?
    O << "Lllvm$workaround$fake$stub$" << *CurrentFnSym << ":\n";
  
  // Emit pre-function debug and/or EH information.
  if (MAI->doesSupportDebugInformation() || MAI->doesSupportExceptionHandling())
    DW->BeginFunction(MF);
}

/// EmitFunctionEntryLabel - Emit the label that is the entrypoint for the
/// function.  This can be overridden by targets as required to do custom stuff.
void AsmPrinter::EmitFunctionEntryLabel() {
  OutStreamer.EmitLabel(CurrentFnSym);
}


/// EmitFunctionBody - This method emits the body and trailer for a
/// function.
void AsmPrinter::EmitFunctionBody() {
  // Emit target-specific gunk before the function body.
  EmitFunctionBodyStart();
  
  // Print out code for the function.
  bool HasAnyRealCode = false;
  for (MachineFunction::const_iterator I = MF->begin(), E = MF->end();
       I != E; ++I) {
    // Print a label for the basic block.
    EmitBasicBlockStart(I);
    for (MachineBasicBlock::const_iterator II = I->begin(), IE = I->end();
         II != IE; ++II) {
      // Print the assembly for the instruction.
      if (!II->isLabel())
        HasAnyRealCode = true;
      
      ++EmittedInsts;
      
      // FIXME: Clean up processDebugLoc.
      processDebugLoc(II, true);
      
      EmitInstruction(II);
      
      if (VerboseAsm)
        EmitComments(*II);
      O << '\n';
      
      // FIXME: Clean up processDebugLoc.
      processDebugLoc(II, false);
    }
  }
  
  // If the function is empty and the object file uses .subsections_via_symbols,
  // then we need to emit *something* to the function body to prevent the
  // labels from collapsing together.  Just emit a 0 byte.
  if (MAI->hasSubsectionsViaSymbols() && !HasAnyRealCode)
    OutStreamer.EmitIntValue(0, 1, 0/*addrspace*/);
  
  // Emit target-specific gunk after the function body.
  EmitFunctionBodyEnd();
  
  if (MAI->hasDotTypeDotSizeDirective())
    O << "\t.size\t" << *CurrentFnSym << ", .-" << *CurrentFnSym << '\n';
  
  // Emit post-function debug information.
  if (MAI->doesSupportDebugInformation() || MAI->doesSupportExceptionHandling())
    DW->EndFunction(MF);
  
  // Print out jump tables referenced by the function.
  EmitJumpTableInfo();
}


bool AsmPrinter::doFinalization(Module &M) {
  // Emit global variables.
  for (Module::const_global_iterator I = M.global_begin(), E = M.global_end();
       I != E; ++I)
    EmitGlobalVariable(I);
  
  // Emit final debug information.
  if (MAI->doesSupportDebugInformation() || MAI->doesSupportExceptionHandling())
    DW->EndModule();
  
  // If the target wants to know about weak references, print them all.
  if (MAI->getWeakRefDirective()) {
    // FIXME: This is not lazy, it would be nice to only print weak references
    // to stuff that is actually used.  Note that doing so would require targets
    // to notice uses in operands (due to constant exprs etc).  This should
    // happen with the MC stuff eventually.

    // Print out module-level global variables here.
    for (Module::const_global_iterator I = M.global_begin(), E = M.global_end();
         I != E; ++I) {
      if (!I->hasExternalWeakLinkage()) continue;
      OutStreamer.EmitSymbolAttribute(GetGlobalValueSymbol(I),
                                      MCSA_WeakReference);
    }
    
    for (Module::const_iterator I = M.begin(), E = M.end(); I != E; ++I) {
      if (!I->hasExternalWeakLinkage()) continue;
      OutStreamer.EmitSymbolAttribute(GetGlobalValueSymbol(I),
                                      MCSA_WeakReference);
    }
  }

  if (MAI->hasSetDirective()) {
    OutStreamer.AddBlankLine();
    for (Module::const_alias_iterator I = M.alias_begin(), E = M.alias_end();
         I != E; ++I) {
      MCSymbol *Name = GetGlobalValueSymbol(I);

      const GlobalValue *GV = cast<GlobalValue>(I->getAliasedGlobal());
      MCSymbol *Target = GetGlobalValueSymbol(GV);

      if (I->hasExternalLinkage() || !MAI->getWeakRefDirective())
        OutStreamer.EmitSymbolAttribute(Name, MCSA_Global);
      else if (I->hasWeakLinkage())
        OutStreamer.EmitSymbolAttribute(Name, MCSA_WeakReference);
      else
        assert(I->hasLocalLinkage() && "Invalid alias linkage");

      EmitVisibility(Name, I->getVisibility());

      // Emit the directives as assignments aka .set:
      OutStreamer.EmitAssignment(Name, 
                                 MCSymbolRefExpr::Create(Target, OutContext));
    }
  }

  GCModuleInfo *MI = getAnalysisIfAvailable<GCModuleInfo>();
  assert(MI && "AsmPrinter didn't require GCModuleInfo?");
  for (GCModuleInfo::iterator I = MI->end(), E = MI->begin(); I != E; )
    if (GCMetadataPrinter *MP = GetOrCreateGCPrinter(*--I))
      MP->finishAssembly(O, *this, *MAI);

  // If we don't have any trampolines, then we don't require stack memory
  // to be executable. Some targets have a directive to declare this.
  Function *InitTrampolineIntrinsic = M.getFunction("llvm.init.trampoline");
  if (!InitTrampolineIntrinsic || InitTrampolineIntrinsic->use_empty())
    if (MCSection *S = MAI->getNonexecutableStackSection(OutContext))
      OutStreamer.SwitchSection(S);
  
  // Allow the target to emit any magic that it wants at the end of the file,
  // after everything else has gone out.
  EmitEndOfAsmFile(M);
  
  delete Mang; Mang = 0;
  DW = 0; MMI = 0;
  
  OutStreamer.Finish();
  return false;
}

void AsmPrinter::SetupMachineFunction(MachineFunction &MF) {
  this->MF = &MF;
  // Get the function symbol.
  CurrentFnSym = GetGlobalValueSymbol(MF.getFunction());

  if (VerboseAsm)
    LI = &getAnalysis<MachineLoopInfo>();
}

namespace {
  // SectionCPs - Keep track the alignment, constpool entries per Section.
  struct SectionCPs {
    const MCSection *S;
    unsigned Alignment;
    SmallVector<unsigned, 4> CPEs;
    SectionCPs(const MCSection *s, unsigned a) : S(s), Alignment(a) {}
  };
}

/// EmitConstantPool - Print to the current output stream assembly
/// representations of the constants in the constant pool MCP. This is
/// used to print out constants which have been "spilled to memory" by
/// the code generator.
///
void AsmPrinter::EmitConstantPool() {
  const MachineConstantPool *MCP = MF->getConstantPool();
  const std::vector<MachineConstantPoolEntry> &CP = MCP->getConstants();
  if (CP.empty()) return;

  // Calculate sections for constant pool entries. We collect entries to go into
  // the same section together to reduce amount of section switch statements.
  SmallVector<SectionCPs, 4> CPSections;
  for (unsigned i = 0, e = CP.size(); i != e; ++i) {
    const MachineConstantPoolEntry &CPE = CP[i];
    unsigned Align = CPE.getAlignment();
    
    SectionKind Kind;
    switch (CPE.getRelocationInfo()) {
    default: llvm_unreachable("Unknown section kind");
    case 2: Kind = SectionKind::getReadOnlyWithRel(); break;
    case 1:
      Kind = SectionKind::getReadOnlyWithRelLocal();
      break;
    case 0:
    switch (TM.getTargetData()->getTypeAllocSize(CPE.getType())) {
    case 4:  Kind = SectionKind::getMergeableConst4(); break;
    case 8:  Kind = SectionKind::getMergeableConst8(); break;
    case 16: Kind = SectionKind::getMergeableConst16();break;
    default: Kind = SectionKind::getMergeableConst(); break;
    }
    }

    const MCSection *S = getObjFileLowering().getSectionForConstant(Kind);
    
    // The number of sections are small, just do a linear search from the
    // last section to the first.
    bool Found = false;
    unsigned SecIdx = CPSections.size();
    while (SecIdx != 0) {
      if (CPSections[--SecIdx].S == S) {
        Found = true;
        break;
      }
    }
    if (!Found) {
      SecIdx = CPSections.size();
      CPSections.push_back(SectionCPs(S, Align));
    }

    if (Align > CPSections[SecIdx].Alignment)
      CPSections[SecIdx].Alignment = Align;
    CPSections[SecIdx].CPEs.push_back(i);
  }

  // Now print stuff into the calculated sections.
  for (unsigned i = 0, e = CPSections.size(); i != e; ++i) {
    OutStreamer.SwitchSection(CPSections[i].S);
    EmitAlignment(Log2_32(CPSections[i].Alignment));

    unsigned Offset = 0;
    for (unsigned j = 0, ee = CPSections[i].CPEs.size(); j != ee; ++j) {
      unsigned CPI = CPSections[i].CPEs[j];
      MachineConstantPoolEntry CPE = CP[CPI];

      // Emit inter-object padding for alignment.
      unsigned AlignMask = CPE.getAlignment() - 1;
      unsigned NewOffset = (Offset + AlignMask) & ~AlignMask;
      OutStreamer.EmitFill(NewOffset - Offset, 0/*fillval*/, 0/*addrspace*/);

      const Type *Ty = CPE.getType();
      Offset = NewOffset + TM.getTargetData()->getTypeAllocSize(Ty);

      // Emit the label with a comment on it.
      if (VerboseAsm) {
        OutStreamer.GetCommentOS() << "constant pool ";
        WriteTypeSymbolic(OutStreamer.GetCommentOS(), CPE.getType(),
                          MF->getFunction()->getParent());
        OutStreamer.GetCommentOS() << '\n';
      }
      OutStreamer.EmitLabel(GetCPISymbol(CPI));

      if (CPE.isMachineConstantPoolEntry())
        EmitMachineConstantPoolValue(CPE.Val.MachineCPVal);
      else
        EmitGlobalConstant(CPE.Val.ConstVal);
    }
  }
}

/// EmitJumpTableInfo - Print assembly representations of the jump tables used
/// by the current function to the current output stream.  
///
void AsmPrinter::EmitJumpTableInfo() {
  const MachineJumpTableInfo *MJTI = MF->getJumpTableInfo();
  if (MJTI == 0) return;
  const std::vector<MachineJumpTableEntry> &JT = MJTI->getJumpTables();
  if (JT.empty()) return;

  // Pick the directive to use to print the jump table entries, and switch to 
  // the appropriate section.
  const Function *F = MF->getFunction();
  bool JTInDiffSection = false;
  if (// In PIC mode, we need to emit the jump table to the same section as the
      // function body itself, otherwise the label differences won't make sense.
      // FIXME: Need a better predicate for this: what about custom entries?
      MJTI->getEntryKind() == MachineJumpTableInfo::EK_LabelDifference32 ||
      // We should also do if the section name is NULL or function is declared
      // in discardable section
      // FIXME: this isn't the right predicate, should be based on the MCSection
      // for the function.
      F->isWeakForLinker()) {
    OutStreamer.SwitchSection(getObjFileLowering().SectionForGlobal(F,Mang,TM));
  } else {
    // Otherwise, drop it in the readonly section.
    const MCSection *ReadOnlySection = 
      getObjFileLowering().getSectionForConstant(SectionKind::getReadOnly());
    OutStreamer.SwitchSection(ReadOnlySection);
    JTInDiffSection = true;
  }

  EmitAlignment(Log2_32(MJTI->getEntryAlignment(*TM.getTargetData())));
  
  for (unsigned JTI = 0, e = JT.size(); JTI != e; ++JTI) {
    const std::vector<MachineBasicBlock*> &JTBBs = JT[JTI].MBBs;
    
    // If this jump table was deleted, ignore it. 
    if (JTBBs.empty()) continue;

    // For the EK_LabelDifference32 entry, if the target supports .set, emit a
    // .set directive for each unique entry.  This reduces the number of
    // relocations the assembler will generate for the jump table.
    if (MJTI->getEntryKind() == MachineJumpTableInfo::EK_LabelDifference32 &&
        MAI->hasSetDirective()) {
      SmallPtrSet<const MachineBasicBlock*, 16> EmittedSets;
      const TargetLowering *TLI = TM.getTargetLowering();
      const MCExpr *Base = TLI->getPICJumpTableRelocBaseExpr(MF,JTI,OutContext);
      for (unsigned ii = 0, ee = JTBBs.size(); ii != ee; ++ii) {
        const MachineBasicBlock *MBB = JTBBs[ii];
        if (!EmittedSets.insert(MBB)) continue;
        
        // .set LJTSet, LBB32-base
        const MCExpr *LHS =
          MCSymbolRefExpr::Create(MBB->getSymbol(OutContext), OutContext);
        OutStreamer.EmitAssignment(GetJTSetSymbol(JTI, MBB->getNumber()),
                                MCBinaryExpr::CreateSub(LHS, Base, OutContext));
      }
    }          
    
    // On some targets (e.g. Darwin) we want to emit two consequtive labels
    // before each jump table.  The first label is never referenced, but tells
    // the assembler and linker the extents of the jump table object.  The
    // second label is actually referenced by the code.
    if (JTInDiffSection && MAI->getLinkerPrivateGlobalPrefix()[0])
      // FIXME: This doesn't have to have any specific name, just any randomly
      // named and numbered 'l' label would work.  Simplify GetJTISymbol.
      OutStreamer.EmitLabel(GetJTISymbol(JTI, true));

    OutStreamer.EmitLabel(GetJTISymbol(JTI));

    for (unsigned ii = 0, ee = JTBBs.size(); ii != ee; ++ii)
      EmitJumpTableEntry(MJTI, JTBBs[ii], JTI);
  }
}

/// EmitJumpTableEntry - Emit a jump table entry for the specified MBB to the
/// current stream.
void AsmPrinter::EmitJumpTableEntry(const MachineJumpTableInfo *MJTI,
                                    const MachineBasicBlock *MBB,
                                    unsigned UID) const {
  const MCExpr *Value = 0;
  switch (MJTI->getEntryKind()) {
  case MachineJumpTableInfo::EK_Custom32:
    Value = TM.getTargetLowering()->LowerCustomJumpTableEntry(MJTI, MBB, UID,
                                                              OutContext);
    break;
  case MachineJumpTableInfo::EK_BlockAddress:
    // EK_BlockAddress - Each entry is a plain address of block, e.g.:
    //     .word LBB123
    Value = MCSymbolRefExpr::Create(MBB->getSymbol(OutContext), OutContext);
    break;
  case MachineJumpTableInfo::EK_GPRel32BlockAddress: {
    // EK_GPRel32BlockAddress - Each entry is an address of block, encoded
    // with a relocation as gp-relative, e.g.:
    //     .gprel32 LBB123
    MCSymbol *MBBSym = MBB->getSymbol(OutContext);
    OutStreamer.EmitGPRel32Value(MCSymbolRefExpr::Create(MBBSym, OutContext));
    return;
  }

  case MachineJumpTableInfo::EK_LabelDifference32: {
    // EK_LabelDifference32 - Each entry is the address of the block minus
    // the address of the jump table.  This is used for PIC jump tables where
    // gprel32 is not supported.  e.g.:
    //      .word LBB123 - LJTI1_2
    // If the .set directive is supported, this is emitted as:
    //      .set L4_5_set_123, LBB123 - LJTI1_2
    //      .word L4_5_set_123
    
    // If we have emitted set directives for the jump table entries, print 
    // them rather than the entries themselves.  If we're emitting PIC, then
    // emit the table entries as differences between two text section labels.
    if (MAI->hasSetDirective()) {
      // If we used .set, reference the .set's symbol.
      Value = MCSymbolRefExpr::Create(GetJTSetSymbol(UID, MBB->getNumber()),
                                      OutContext);
      break;
    }
    // Otherwise, use the difference as the jump table entry.
    Value = MCSymbolRefExpr::Create(MBB->getSymbol(OutContext), OutContext);
    const MCExpr *JTI = MCSymbolRefExpr::Create(GetJTISymbol(UID), OutContext);
    Value = MCBinaryExpr::CreateSub(Value, JTI, OutContext);
    break;
  }
  }
  
  assert(Value && "Unknown entry kind!");
 
  unsigned EntrySize = MJTI->getEntrySize(*TM.getTargetData());
  OutStreamer.EmitValue(Value, EntrySize, /*addrspace*/0);
}


/// EmitSpecialLLVMGlobal - Check to see if the specified global is a
/// special global used by LLVM.  If so, emit it and return true, otherwise
/// do nothing and return false.
bool AsmPrinter::EmitSpecialLLVMGlobal(const GlobalVariable *GV) {
  if (GV->getName() == "llvm.used") {
    if (MAI->hasNoDeadStrip())    // No need to emit this at all.
      EmitLLVMUsedList(GV->getInitializer());
    return true;
  }

  // Ignore debug and non-emitted data.  This handles llvm.compiler.used.
  if (GV->getSection() == "llvm.metadata" ||
      GV->hasAvailableExternallyLinkage())
    return true;
  
  if (!GV->hasAppendingLinkage()) return false;

  assert(GV->hasInitializer() && "Not a special LLVM global!");
  
  const TargetData *TD = TM.getTargetData();
  unsigned Align = Log2_32(TD->getPointerPrefAlignment());
  if (GV->getName() == "llvm.global_ctors") {
    OutStreamer.SwitchSection(getObjFileLowering().getStaticCtorSection());
    EmitAlignment(Align, 0);
    EmitXXStructorList(GV->getInitializer());
    
    if (TM.getRelocationModel() == Reloc::Static &&
        MAI->hasStaticCtorDtorReferenceInStaticMode()) {
      StringRef Sym(".constructors_used");
      OutStreamer.EmitSymbolAttribute(OutContext.GetOrCreateSymbol(Sym),
                                      MCSA_Reference);
    }
    return true;
  } 
  
  if (GV->getName() == "llvm.global_dtors") {
    OutStreamer.SwitchSection(getObjFileLowering().getStaticDtorSection());
    EmitAlignment(Align, 0);
    EmitXXStructorList(GV->getInitializer());

    if (TM.getRelocationModel() == Reloc::Static &&
        MAI->hasStaticCtorDtorReferenceInStaticMode()) {
      StringRef Sym(".destructors_used");
      OutStreamer.EmitSymbolAttribute(OutContext.GetOrCreateSymbol(Sym),
                                      MCSA_Reference);
    }
    return true;
  }
  
  return false;
}

/// EmitLLVMUsedList - For targets that define a MAI::UsedDirective, mark each
/// global in the specified llvm.used list for which emitUsedDirectiveFor
/// is true, as being used with this directive.
void AsmPrinter::EmitLLVMUsedList(Constant *List) {
  // Should be an array of 'i8*'.
  ConstantArray *InitList = dyn_cast<ConstantArray>(List);
  if (InitList == 0) return;
  
  for (unsigned i = 0, e = InitList->getNumOperands(); i != e; ++i) {
    const GlobalValue *GV =
      dyn_cast<GlobalValue>(InitList->getOperand(i)->stripPointerCasts());
    if (GV && getObjFileLowering().shouldEmitUsedDirectiveFor(GV, Mang))
      OutStreamer.EmitSymbolAttribute(GetGlobalValueSymbol(GV),
                                      MCSA_NoDeadStrip);
  }
}

/// EmitXXStructorList - Emit the ctor or dtor list.  This just prints out the 
/// function pointers, ignoring the init priority.
void AsmPrinter::EmitXXStructorList(Constant *List) {
  // Should be an array of '{ int, void ()* }' structs.  The first value is the
  // init priority, which we ignore.
  if (!isa<ConstantArray>(List)) return;
  ConstantArray *InitList = cast<ConstantArray>(List);
  for (unsigned i = 0, e = InitList->getNumOperands(); i != e; ++i)
    if (ConstantStruct *CS = dyn_cast<ConstantStruct>(InitList->getOperand(i))){
      if (CS->getNumOperands() != 2) return;  // Not array of 2-element structs.

      if (CS->getOperand(1)->isNullValue())
        return;  // Found a null terminator, exit printing.
      // Emit the function pointer.
      EmitGlobalConstant(CS->getOperand(1));
    }
}

//===--------------------------------------------------------------------===//
// Emission and print routines
//

/// EmitInt8 - Emit a byte directive and value.
///
void AsmPrinter::EmitInt8(int Value) const {
  OutStreamer.EmitIntValue(Value, 1, 0/*addrspace*/);
}

/// EmitInt16 - Emit a short directive and value.
///
void AsmPrinter::EmitInt16(int Value) const {
  OutStreamer.EmitIntValue(Value, 2, 0/*addrspace*/);
}

/// EmitInt32 - Emit a long directive and value.
///
void AsmPrinter::EmitInt32(int Value) const {
  OutStreamer.EmitIntValue(Value, 4, 0/*addrspace*/);
}

/// EmitInt64 - Emit a long long directive and value.
///
void AsmPrinter::EmitInt64(uint64_t Value) const {
  OutStreamer.EmitIntValue(Value, 8, 0/*addrspace*/);
}

//===----------------------------------------------------------------------===//

// EmitAlignment - Emit an alignment directive to the specified power of
// two boundary.  For example, if you pass in 3 here, you will get an 8
// byte alignment.  If a global value is specified, and if that global has
// an explicit alignment requested, it will unconditionally override the
// alignment request.  However, if ForcedAlignBits is specified, this value
// has final say: the ultimate alignment will be the max of ForcedAlignBits
// and the alignment computed with NumBits and the global.
//
// The algorithm is:
//     Align = NumBits;
//     if (GV && GV->hasalignment) Align = GV->getalignment();
//     Align = std::max(Align, ForcedAlignBits);
//
void AsmPrinter::EmitAlignment(unsigned NumBits, const GlobalValue *GV,
                               unsigned ForcedAlignBits,
                               bool UseFillExpr) const {
  if (GV && GV->getAlignment())
    NumBits = Log2_32(GV->getAlignment());
  NumBits = std::max(NumBits, ForcedAlignBits);
  
  if (NumBits == 0) return;   // No need to emit alignment.
  
  unsigned FillValue = 0;
  if (getCurrentSection()->getKind().isText())
    FillValue = MAI->getTextAlignFillValue();
  
  OutStreamer.EmitValueToAlignment(1 << NumBits, FillValue, 1, 0);
}

/// LowerConstant - Lower the specified LLVM Constant to an MCExpr.
///
static const MCExpr *LowerConstant(const Constant *CV, AsmPrinter &AP) {
  MCContext &Ctx = AP.OutContext;
  
  if (CV->isNullValue() || isa<UndefValue>(CV))
    return MCConstantExpr::Create(0, Ctx);

  if (const ConstantInt *CI = dyn_cast<ConstantInt>(CV))
    return MCConstantExpr::Create(CI->getZExtValue(), Ctx);
  
  if (const GlobalValue *GV = dyn_cast<GlobalValue>(CV))
    return MCSymbolRefExpr::Create(AP.GetGlobalValueSymbol(GV), Ctx);
  if (const BlockAddress *BA = dyn_cast<BlockAddress>(CV))
    return MCSymbolRefExpr::Create(AP.GetBlockAddressSymbol(BA), Ctx);
  
  const ConstantExpr *CE = dyn_cast<ConstantExpr>(CV);
  if (CE == 0) {
    llvm_unreachable("Unknown constant value to lower!");
    return MCConstantExpr::Create(0, Ctx);
  }
  
  switch (CE->getOpcode()) {
  case Instruction::ZExt:
  case Instruction::SExt:
  case Instruction::FPTrunc:
  case Instruction::FPExt:
  case Instruction::UIToFP:
  case Instruction::SIToFP:
  case Instruction::FPToUI:
  case Instruction::FPToSI:
  default: llvm_unreachable("FIXME: Don't support this constant cast expr");
  case Instruction::GetElementPtr: {
    const TargetData &TD = *AP.TM.getTargetData();
    // Generate a symbolic expression for the byte address
    const Constant *PtrVal = CE->getOperand(0);
    SmallVector<Value*, 8> IdxVec(CE->op_begin()+1, CE->op_end());
    int64_t Offset = TD.getIndexedOffset(PtrVal->getType(), &IdxVec[0],
                                         IdxVec.size());
    
    const MCExpr *Base = LowerConstant(CE->getOperand(0), AP);
    if (Offset == 0)
      return Base;
    
    // Truncate/sext the offset to the pointer size.
    if (TD.getPointerSizeInBits() != 64) {
      int SExtAmount = 64-TD.getPointerSizeInBits();
      Offset = (Offset << SExtAmount) >> SExtAmount;
    }
    
    return MCBinaryExpr::CreateAdd(Base, MCConstantExpr::Create(Offset, Ctx),
                                   Ctx);
  }
      
  case Instruction::Trunc:
    // We emit the value and depend on the assembler to truncate the generated
    // expression properly.  This is important for differences between
    // blockaddress labels.  Since the two labels are in the same function, it
    // is reasonable to treat their delta as a 32-bit value.
    // FALL THROUGH.
  case Instruction::BitCast:
    return LowerConstant(CE->getOperand(0), AP);

  case Instruction::IntToPtr: {
    const TargetData &TD = *AP.TM.getTargetData();
    // Handle casts to pointers by changing them into casts to the appropriate
    // integer type.  This promotes constant folding and simplifies this code.
    Constant *Op = CE->getOperand(0);
    Op = ConstantExpr::getIntegerCast(Op, TD.getIntPtrType(CV->getContext()),
                                      false/*ZExt*/);
    return LowerConstant(Op, AP);
  }
    
  case Instruction::PtrToInt: {
    const TargetData &TD = *AP.TM.getTargetData();
    // Support only foldable casts to/from pointers that can be eliminated by
    // changing the pointer to the appropriately sized integer type.
    Constant *Op = CE->getOperand(0);
    const Type *Ty = CE->getType();

    const MCExpr *OpExpr = LowerConstant(Op, AP);

    // We can emit the pointer value into this slot if the slot is an
    // integer slot equal to the size of the pointer.
    if (TD.getTypeAllocSize(Ty) == TD.getTypeAllocSize(Op->getType()))
      return OpExpr;

    // Otherwise the pointer is smaller than the resultant integer, mask off
    // the high bits so we are sure to get a proper truncation if the input is
    // a constant expr.
    unsigned InBits = TD.getTypeAllocSizeInBits(Op->getType());
    const MCExpr *MaskExpr = MCConstantExpr::Create(~0ULL >> (64-InBits), Ctx);
    return MCBinaryExpr::CreateAnd(OpExpr, MaskExpr, Ctx);
  }
      
  case Instruction::Add:
  case Instruction::Sub:
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor: {
    const MCExpr *LHS = LowerConstant(CE->getOperand(0), AP);
    const MCExpr *RHS = LowerConstant(CE->getOperand(1), AP);
    switch (CE->getOpcode()) {
    default: llvm_unreachable("Unknown binary operator constant cast expr");
    case Instruction::Add: return MCBinaryExpr::CreateAdd(LHS, RHS, Ctx);
    case Instruction::Sub: return MCBinaryExpr::CreateSub(LHS, RHS, Ctx);
    case Instruction::And: return MCBinaryExpr::CreateAnd(LHS, RHS, Ctx);
    case Instruction::Or:  return MCBinaryExpr::CreateOr (LHS, RHS, Ctx);
    case Instruction::Xor: return MCBinaryExpr::CreateXor(LHS, RHS, Ctx);
    }
  }
  }
}

static void EmitGlobalConstantArray(const ConstantArray *CA, unsigned AddrSpace,
                                    AsmPrinter &AP) {
  if (AddrSpace != 0 || !CA->isString()) {
    // Not a string.  Print the values in successive locations
    for (unsigned i = 0, e = CA->getNumOperands(); i != e; ++i)
      AP.EmitGlobalConstant(CA->getOperand(i), AddrSpace);
    return;
  }
  
  // Otherwise, it can be emitted as .ascii.
  SmallVector<char, 128> TmpVec;
  TmpVec.reserve(CA->getNumOperands());
  for (unsigned i = 0, e = CA->getNumOperands(); i != e; ++i)
    TmpVec.push_back(cast<ConstantInt>(CA->getOperand(i))->getZExtValue());

  AP.OutStreamer.EmitBytes(StringRef(TmpVec.data(), TmpVec.size()), AddrSpace);
}

static void EmitGlobalConstantVector(const ConstantVector *CV,
                                     unsigned AddrSpace, AsmPrinter &AP) {
  for (unsigned i = 0, e = CV->getType()->getNumElements(); i != e; ++i)
    AP.EmitGlobalConstant(CV->getOperand(i), AddrSpace);
}

static void EmitGlobalConstantStruct(const ConstantStruct *CS,
                                     unsigned AddrSpace, AsmPrinter &AP) {
  // Print the fields in successive locations. Pad to align if needed!
  const TargetData *TD = AP.TM.getTargetData();
  unsigned Size = TD->getTypeAllocSize(CS->getType());
  const StructLayout *Layout = TD->getStructLayout(CS->getType());
  uint64_t SizeSoFar = 0;
  for (unsigned i = 0, e = CS->getNumOperands(); i != e; ++i) {
    const Constant *Field = CS->getOperand(i);

    // Check if padding is needed and insert one or more 0s.
    uint64_t FieldSize = TD->getTypeAllocSize(Field->getType());
    uint64_t PadSize = ((i == e-1 ? Size : Layout->getElementOffset(i+1))
                        - Layout->getElementOffset(i)) - FieldSize;
    SizeSoFar += FieldSize + PadSize;

    // Now print the actual field value.
    AP.EmitGlobalConstant(Field, AddrSpace);

    // Insert padding - this may include padding to increase the size of the
    // current field up to the ABI size (if the struct is not packed) as well
    // as padding to ensure that the next field starts at the right offset.
    AP.OutStreamer.EmitZeros(PadSize, AddrSpace);
  }
  assert(SizeSoFar == Layout->getSizeInBytes() &&
         "Layout of constant struct may be incorrect!");
}

static void EmitGlobalConstantFP(const ConstantFP *CFP, unsigned AddrSpace,
                                 AsmPrinter &AP) {
  // FP Constants are printed as integer constants to avoid losing
  // precision.
  if (CFP->getType()->isDoubleTy()) {
    if (AP.VerboseAsm) {
      double Val = CFP->getValueAPF().convertToDouble();
      AP.OutStreamer.GetCommentOS() << "double " << Val << '\n';
    }

    uint64_t Val = CFP->getValueAPF().bitcastToAPInt().getZExtValue();
    AP.OutStreamer.EmitIntValue(Val, 8, AddrSpace);
    return;
  }
  
  if (CFP->getType()->isFloatTy()) {
    if (AP.VerboseAsm) {
      float Val = CFP->getValueAPF().convertToFloat();
      AP.OutStreamer.GetCommentOS() << "float " << Val << '\n';
    }
    uint64_t Val = CFP->getValueAPF().bitcastToAPInt().getZExtValue();
    AP.OutStreamer.EmitIntValue(Val, 4, AddrSpace);
    return;
  }
  
  if (CFP->getType()->isX86_FP80Ty()) {
    // all long double variants are printed as hex
    // api needed to prevent premature destruction
    APInt API = CFP->getValueAPF().bitcastToAPInt();
    const uint64_t *p = API.getRawData();
    if (AP.VerboseAsm) {
      // Convert to double so we can print the approximate val as a comment.
      APFloat DoubleVal = CFP->getValueAPF();
      bool ignored;
      DoubleVal.convert(APFloat::IEEEdouble, APFloat::rmNearestTiesToEven,
                        &ignored);
      AP.OutStreamer.GetCommentOS() << "x86_fp80 ~= "
        << DoubleVal.convertToDouble() << '\n';
    }
    
    if (AP.TM.getTargetData()->isBigEndian()) {
      AP.OutStreamer.EmitIntValue(p[1], 2, AddrSpace);
      AP.OutStreamer.EmitIntValue(p[0], 8, AddrSpace);
    } else {
      AP.OutStreamer.EmitIntValue(p[0], 8, AddrSpace);
      AP.OutStreamer.EmitIntValue(p[1], 2, AddrSpace);
    }
    
    // Emit the tail padding for the long double.
    const TargetData &TD = *AP.TM.getTargetData();
    AP.OutStreamer.EmitZeros(TD.getTypeAllocSize(CFP->getType()) -
                             TD.getTypeStoreSize(CFP->getType()), AddrSpace);
    return;
  }
  
  assert(CFP->getType()->isPPC_FP128Ty() &&
         "Floating point constant type not handled");
  // All long double variants are printed as hex api needed to prevent
  // premature destruction.
  APInt API = CFP->getValueAPF().bitcastToAPInt();
  const uint64_t *p = API.getRawData();
  if (AP.TM.getTargetData()->isBigEndian()) {
    AP.OutStreamer.EmitIntValue(p[0], 8, AddrSpace);
    AP.OutStreamer.EmitIntValue(p[1], 8, AddrSpace);
  } else {
    AP.OutStreamer.EmitIntValue(p[1], 8, AddrSpace);
    AP.OutStreamer.EmitIntValue(p[0], 8, AddrSpace);
  }
}

static void EmitGlobalConstantLargeInt(const ConstantInt *CI,
                                       unsigned AddrSpace, AsmPrinter &AP) {
  const TargetData *TD = AP.TM.getTargetData();
  unsigned BitWidth = CI->getBitWidth();
  assert((BitWidth & 63) == 0 && "only support multiples of 64-bits");

  // We don't expect assemblers to support integer data directives
  // for more than 64 bits, so we emit the data in at most 64-bit
  // quantities at a time.
  const uint64_t *RawData = CI->getValue().getRawData();
  for (unsigned i = 0, e = BitWidth / 64; i != e; ++i) {
    uint64_t Val = TD->isBigEndian() ? RawData[e - i - 1] : RawData[i];
    AP.OutStreamer.EmitIntValue(Val, 8, AddrSpace);
  }
}

/// EmitGlobalConstant - Print a general LLVM constant to the .s file.
void AsmPrinter::EmitGlobalConstant(const Constant *CV, unsigned AddrSpace) {
  if (isa<ConstantAggregateZero>(CV) || isa<UndefValue>(CV)) {
    uint64_t Size = TM.getTargetData()->getTypeAllocSize(CV->getType());
    return OutStreamer.EmitZeros(Size, AddrSpace);
  }

  if (const ConstantInt *CI = dyn_cast<ConstantInt>(CV)) {
    unsigned Size = TM.getTargetData()->getTypeAllocSize(CV->getType());
    switch (Size) {
    case 1:
    case 2:
    case 4:
    case 8:
      if (VerboseAsm)
        OutStreamer.GetCommentOS() << format("0x%llx\n", CI->getZExtValue());
      OutStreamer.EmitIntValue(CI->getZExtValue(), Size, AddrSpace);
      return;
    default:
      EmitGlobalConstantLargeInt(CI, AddrSpace, *this);
      return;
    }
  }
  
  if (const ConstantArray *CVA = dyn_cast<ConstantArray>(CV))
    return EmitGlobalConstantArray(CVA, AddrSpace, *this);
  
  if (const ConstantStruct *CVS = dyn_cast<ConstantStruct>(CV))
    return EmitGlobalConstantStruct(CVS, AddrSpace, *this);

  if (const ConstantFP *CFP = dyn_cast<ConstantFP>(CV))
    return EmitGlobalConstantFP(CFP, AddrSpace, *this);
  
  if (const ConstantVector *V = dyn_cast<ConstantVector>(CV))
    return EmitGlobalConstantVector(V, AddrSpace, *this);

  if (isa<ConstantPointerNull>(CV)) {
    unsigned Size = TM.getTargetData()->getTypeAllocSize(CV->getType());
    OutStreamer.EmitIntValue(0, Size, AddrSpace);
    return;
  }
  
  // Otherwise, it must be a ConstantExpr.  Lower it to an MCExpr, then emit it
  // thread the streamer with EmitValue.
  OutStreamer.EmitValue(LowerConstant(CV, *this),
                        TM.getTargetData()->getTypeAllocSize(CV->getType()),
                        AddrSpace);
}

void AsmPrinter::EmitMachineConstantPoolValue(MachineConstantPoolValue *MCPV) {
  // Target doesn't support this yet!
  llvm_unreachable("Target does not support EmitMachineConstantPoolValue");
}

/// PrintSpecial - Print information related to the specified machine instr
/// that is independent of the operand, and may be independent of the instr
/// itself.  This can be useful for portably encoding the comment character
/// or other bits of target-specific knowledge into the asmstrings.  The
/// syntax used is ${:comment}.  Targets can override this to add support
/// for their own strange codes.
void AsmPrinter::PrintSpecial(const MachineInstr *MI, const char *Code) const {
  if (!strcmp(Code, "private")) {
    O << MAI->getPrivateGlobalPrefix();
  } else if (!strcmp(Code, "comment")) {
    if (VerboseAsm)
      O << MAI->getCommentString();
  } else if (!strcmp(Code, "uid")) {
    // Comparing the address of MI isn't sufficient, because machineinstrs may
    // be allocated to the same address across functions.
    const Function *ThisF = MI->getParent()->getParent()->getFunction();
    
    // If this is a new LastFn instruction, bump the counter.
    if (LastMI != MI || LastFn != ThisF) {
      ++Counter;
      LastMI = MI;
      LastFn = ThisF;
    }
    O << Counter;
  } else {
    std::string msg;
    raw_string_ostream Msg(msg);
    Msg << "Unknown special formatter '" << Code
         << "' for machine instr: " << *MI;
    llvm_report_error(Msg.str());
  }    
}

/// processDebugLoc - Processes the debug information of each machine
/// instruction's DebugLoc.
void AsmPrinter::processDebugLoc(const MachineInstr *MI, 
                                 bool BeforePrintingInsn) {
  if (!MAI || !DW || !MAI->doesSupportDebugInformation()
      || !DW->ShouldEmitDwarfDebug())
    return;
  DebugLoc DL = MI->getDebugLoc();
  if (DL.isUnknown())
    return;
  DILocation CurDLT = MF->getDILocation(DL);
  if (CurDLT.getScope().isNull())
    return;

  if (!BeforePrintingInsn) {
    // After printing instruction
    DW->EndScope(MI);
  } else if (CurDLT.getNode() != PrevDLT) {
    unsigned L = DW->RecordSourceLine(CurDLT.getLineNumber(), 
                                      CurDLT.getColumnNumber(),
                                      CurDLT.getScope().getNode());
    printLabel(L);
    O << '\n';
    DW->BeginScope(MI, L);
    PrevDLT = CurDLT.getNode();
  }
}


/// printInlineAsm - This method formats and prints the specified machine
/// instruction that is an inline asm.
void AsmPrinter::printInlineAsm(const MachineInstr *MI) const {
  unsigned NumOperands = MI->getNumOperands();
  
  // Count the number of register definitions.
  unsigned NumDefs = 0;
  for (; MI->getOperand(NumDefs).isReg() && MI->getOperand(NumDefs).isDef();
       ++NumDefs)
    assert(NumDefs != NumOperands-1 && "No asm string?");
  
  assert(MI->getOperand(NumDefs).isSymbol() && "No asm string?");

  // Disassemble the AsmStr, printing out the literal pieces, the operands, etc.
  const char *AsmStr = MI->getOperand(NumDefs).getSymbolName();

  O << '\t';

  // If this asmstr is empty, just print the #APP/#NOAPP markers.
  // These are useful to see where empty asm's wound up.
  if (AsmStr[0] == 0) {
    O << MAI->getCommentString() << MAI->getInlineAsmStart() << "\n\t";
    O << MAI->getCommentString() << MAI->getInlineAsmEnd() << '\n';
    return;
  }
  
  O << MAI->getCommentString() << MAI->getInlineAsmStart() << "\n\t";

  // The variant of the current asmprinter.
  int AsmPrinterVariant = MAI->getAssemblerDialect();

  int CurVariant = -1;            // The number of the {.|.|.} region we are in.
  const char *LastEmitted = AsmStr; // One past the last character emitted.
  
  while (*LastEmitted) {
    switch (*LastEmitted) {
    default: {
      // Not a special case, emit the string section literally.
      const char *LiteralEnd = LastEmitted+1;
      while (*LiteralEnd && *LiteralEnd != '{' && *LiteralEnd != '|' &&
             *LiteralEnd != '}' && *LiteralEnd != '$' && *LiteralEnd != '\n')
        ++LiteralEnd;
      if (CurVariant == -1 || CurVariant == AsmPrinterVariant)
        O.write(LastEmitted, LiteralEnd-LastEmitted);
      LastEmitted = LiteralEnd;
      break;
    }
    case '\n':
      ++LastEmitted;   // Consume newline character.
      O << '\n';       // Indent code with newline.
      break;
    case '$': {
      ++LastEmitted;   // Consume '$' character.
      bool Done = true;

      // Handle escapes.
      switch (*LastEmitted) {
      default: Done = false; break;
      case '$':     // $$ -> $
        if (CurVariant == -1 || CurVariant == AsmPrinterVariant)
          O << '$';
        ++LastEmitted;  // Consume second '$' character.
        break;
      case '(':             // $( -> same as GCC's { character.
        ++LastEmitted;      // Consume '(' character.
        if (CurVariant != -1) {
          llvm_report_error("Nested variants found in inline asm string: '"
                            + std::string(AsmStr) + "'");
        }
        CurVariant = 0;     // We're in the first variant now.
        break;
      case '|':
        ++LastEmitted;  // consume '|' character.
        if (CurVariant == -1)
          O << '|';       // this is gcc's behavior for | outside a variant
        else
          ++CurVariant;   // We're in the next variant.
        break;
      case ')':         // $) -> same as GCC's } char.
        ++LastEmitted;  // consume ')' character.
        if (CurVariant == -1)
          O << '}';     // this is gcc's behavior for } outside a variant
        else 
          CurVariant = -1;
        break;
      }
      if (Done) break;
      
      bool HasCurlyBraces = false;
      if (*LastEmitted == '{') {     // ${variable}
        ++LastEmitted;               // Consume '{' character.
        HasCurlyBraces = true;
      }
      
      // If we have ${:foo}, then this is not a real operand reference, it is a
      // "magic" string reference, just like in .td files.  Arrange to call
      // PrintSpecial.
      if (HasCurlyBraces && *LastEmitted == ':') {
        ++LastEmitted;
        const char *StrStart = LastEmitted;
        const char *StrEnd = strchr(StrStart, '}');
        if (StrEnd == 0) {
          llvm_report_error("Unterminated ${:foo} operand in inline asm string: '" 
                            + std::string(AsmStr) + "'");
        }
        
        std::string Val(StrStart, StrEnd);
        PrintSpecial(MI, Val.c_str());
        LastEmitted = StrEnd+1;
        break;
      }
            
      const char *IDStart = LastEmitted;
      char *IDEnd;
      errno = 0;
      long Val = strtol(IDStart, &IDEnd, 10); // We only accept numbers for IDs.
      if (!isdigit(*IDStart) || (Val == 0 && errno == EINVAL)) {
        llvm_report_error("Bad $ operand number in inline asm string: '" 
                          + std::string(AsmStr) + "'");
      }
      LastEmitted = IDEnd;
      
      char Modifier[2] = { 0, 0 };
      
      if (HasCurlyBraces) {
        // If we have curly braces, check for a modifier character.  This
        // supports syntax like ${0:u}, which correspond to "%u0" in GCC asm.
        if (*LastEmitted == ':') {
          ++LastEmitted;    // Consume ':' character.
          if (*LastEmitted == 0) {
            llvm_report_error("Bad ${:} expression in inline asm string: '" 
                              + std::string(AsmStr) + "'");
          }
          
          Modifier[0] = *LastEmitted;
          ++LastEmitted;    // Consume modifier character.
        }
        
        if (*LastEmitted != '}') {
          llvm_report_error("Bad ${} expression in inline asm string: '" 
                            + std::string(AsmStr) + "'");
        }
        ++LastEmitted;    // Consume '}' character.
      }
      
      if ((unsigned)Val >= NumOperands-1) {
        llvm_report_error("Invalid $ operand number in inline asm string: '" 
                          + std::string(AsmStr) + "'");
      }
      
      // Okay, we finally have a value number.  Ask the target to print this
      // operand!
      if (CurVariant == -1 || CurVariant == AsmPrinterVariant) {
        unsigned OpNo = 1;

        bool Error = false;

        // Scan to find the machine operand number for the operand.
        for (; Val; --Val) {
          if (OpNo >= MI->getNumOperands()) break;
          unsigned OpFlags = MI->getOperand(OpNo).getImm();
          OpNo += InlineAsm::getNumOperandRegisters(OpFlags) + 1;
        }

        if (OpNo >= MI->getNumOperands()) {
          Error = true;
        } else {
          unsigned OpFlags = MI->getOperand(OpNo).getImm();
          ++OpNo;  // Skip over the ID number.

          if (Modifier[0] == 'l')  // labels are target independent
            O << *MI->getOperand(OpNo).getMBB()->getSymbol(OutContext);
          else {
            AsmPrinter *AP = const_cast<AsmPrinter*>(this);
            if ((OpFlags & 7) == 4) {
              Error = AP->PrintAsmMemoryOperand(MI, OpNo, AsmPrinterVariant,
                                                Modifier[0] ? Modifier : 0);
            } else {
              Error = AP->PrintAsmOperand(MI, OpNo, AsmPrinterVariant,
                                          Modifier[0] ? Modifier : 0);
            }
          }
        }
        if (Error) {
          std::string msg;
          raw_string_ostream Msg(msg);
          Msg << "Invalid operand found in inline asm: '" << AsmStr << "'\n";
          MI->print(Msg);
          llvm_report_error(Msg.str());
        }
      }
      break;
    }
    }
  }
  O << "\n\t" << MAI->getCommentString() << MAI->getInlineAsmEnd();
}

/// printImplicitDef - This method prints the specified machine instruction
/// that is an implicit def.
void AsmPrinter::printImplicitDef(const MachineInstr *MI) const {
  if (!VerboseAsm) return;
  O.PadToColumn(MAI->getCommentColumn());
  O << MAI->getCommentString() << " implicit-def: "
    << TRI->getName(MI->getOperand(0).getReg());
}

void AsmPrinter::printKill(const MachineInstr *MI) const {
  if (!VerboseAsm) return;
  O.PadToColumn(MAI->getCommentColumn());
  O << MAI->getCommentString() << " kill:";
  for (unsigned n = 0, e = MI->getNumOperands(); n != e; ++n) {
    const MachineOperand &op = MI->getOperand(n);
    assert(op.isReg() && "KILL instruction must have only register operands");
    O << ' ' << TRI->getName(op.getReg()) << (op.isDef() ? "<def>" : "<kill>");
  }
}

/// printLabel - This method prints a local label used by debug and
/// exception handling tables.
void AsmPrinter::printLabel(const MachineInstr *MI) const {
  printLabel(MI->getOperand(0).getImm());
}

void AsmPrinter::printLabel(unsigned Id) const {
  O << MAI->getPrivateGlobalPrefix() << "label" << Id << ':';
}

/// PrintAsmOperand - Print the specified operand of MI, an INLINEASM
/// instruction, using the specified assembler variant.  Targets should
/// override this to format as appropriate.
bool AsmPrinter::PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                                 unsigned AsmVariant, const char *ExtraCode) {
  // Target doesn't support this yet!
  return true;
}

bool AsmPrinter::PrintAsmMemoryOperand(const MachineInstr *MI, unsigned OpNo,
                                       unsigned AsmVariant,
                                       const char *ExtraCode) {
  // Target doesn't support this yet!
  return true;
}

MCSymbol *AsmPrinter::GetBlockAddressSymbol(const BlockAddress *BA,
                                            const char *Suffix) const {
  return GetBlockAddressSymbol(BA->getFunction(), BA->getBasicBlock(), Suffix);
}

MCSymbol *AsmPrinter::GetBlockAddressSymbol(const Function *F,
                                            const BasicBlock *BB,
                                            const char *Suffix) const {
  assert(BB->hasName() &&
         "Address of anonymous basic block not supported yet!");

  // This code must use the function name itself, and not the function number,
  // since it must be possible to generate the label name from within other
  // functions.
  SmallString<60> FnName;
  Mang->getNameWithPrefix(FnName, F, false);

  // FIXME: THIS IS BROKEN IF THE LLVM BASIC BLOCK DOESN'T HAVE A NAME!
  SmallString<60> NameResult;
  Mang->getNameWithPrefix(NameResult,
                          StringRef("BA") + Twine((unsigned)FnName.size()) + 
                          "_" + FnName.str() + "_" + BB->getName() + Suffix, 
                          Mangler::Private);

  return OutContext.GetOrCreateSymbol(NameResult.str());
}

/// GetCPISymbol - Return the symbol for the specified constant pool entry.
MCSymbol *AsmPrinter::GetCPISymbol(unsigned CPID) const {
  SmallString<60> Name;
  raw_svector_ostream(Name) << MAI->getPrivateGlobalPrefix() << "CPI"
    << getFunctionNumber() << '_' << CPID;
  return OutContext.GetOrCreateSymbol(Name.str());
}

/// GetJTISymbol - Return the symbol for the specified jump table entry.
MCSymbol *AsmPrinter::GetJTISymbol(unsigned JTID, bool isLinkerPrivate) const {
  return MF->getJTISymbol(JTID, OutContext, isLinkerPrivate);
}

/// GetJTSetSymbol - Return the symbol for the specified jump table .set
/// FIXME: privatize to AsmPrinter.
MCSymbol *AsmPrinter::GetJTSetSymbol(unsigned UID, unsigned MBBID) const {
  SmallString<60> Name;
  raw_svector_ostream(Name) << MAI->getPrivateGlobalPrefix()
    << getFunctionNumber() << '_' << UID << "_set_" << MBBID;
  return OutContext.GetOrCreateSymbol(Name.str());
}

/// GetGlobalValueSymbol - Return the MCSymbol for the specified global
/// value.
MCSymbol *AsmPrinter::GetGlobalValueSymbol(const GlobalValue *GV) const {
  SmallString<60> NameStr;
  Mang->getNameWithPrefix(NameStr, GV, false);
  return OutContext.GetOrCreateSymbol(NameStr.str());
}

/// GetSymbolWithGlobalValueBase - Return the MCSymbol for a symbol with
/// global value name as its base, with the specified suffix, and where the
/// symbol is forced to have private linkage if ForcePrivate is true.
MCSymbol *AsmPrinter::GetSymbolWithGlobalValueBase(const GlobalValue *GV,
                                                   StringRef Suffix,
                                                   bool ForcePrivate) const {
  SmallString<60> NameStr;
  Mang->getNameWithPrefix(NameStr, GV, ForcePrivate);
  NameStr.append(Suffix.begin(), Suffix.end());
  return OutContext.GetOrCreateSymbol(NameStr.str());
}

/// GetExternalSymbolSymbol - Return the MCSymbol for the specified
/// ExternalSymbol.
MCSymbol *AsmPrinter::GetExternalSymbolSymbol(StringRef Sym) const {
  SmallString<60> NameStr;
  Mang->getNameWithPrefix(NameStr, Sym);
  return OutContext.GetOrCreateSymbol(NameStr.str());
}  



/// PrintParentLoopComment - Print comments about parent loops of this one.
static void PrintParentLoopComment(raw_ostream &OS, const MachineLoop *Loop,
                                   unsigned FunctionNumber) {
  if (Loop == 0) return;
  PrintParentLoopComment(OS, Loop->getParentLoop(), FunctionNumber);
  OS.indent(Loop->getLoopDepth()*2)
    << "Parent Loop BB" << FunctionNumber << "_"
    << Loop->getHeader()->getNumber()
    << " Depth=" << Loop->getLoopDepth() << '\n';
}


/// PrintChildLoopComment - Print comments about child loops within
/// the loop for this basic block, with nesting.
static void PrintChildLoopComment(raw_ostream &OS, const MachineLoop *Loop,
                                  unsigned FunctionNumber) {
  // Add child loop information
  for (MachineLoop::iterator CL = Loop->begin(), E = Loop->end();CL != E; ++CL){
    OS.indent((*CL)->getLoopDepth()*2)
      << "Child Loop BB" << FunctionNumber << "_"
      << (*CL)->getHeader()->getNumber() << " Depth " << (*CL)->getLoopDepth()
      << '\n';
    PrintChildLoopComment(OS, *CL, FunctionNumber);
  }
}

/// EmitComments - Pretty-print comments for basic blocks.
static void PrintBasicBlockLoopComments(const MachineBasicBlock &MBB,
                                        const MachineLoopInfo *LI,
                                        const AsmPrinter &AP) {
  // Add loop depth information
  const MachineLoop *Loop = LI->getLoopFor(&MBB);
  if (Loop == 0) return;
  
  MachineBasicBlock *Header = Loop->getHeader();
  assert(Header && "No header for loop");
  
  // If this block is not a loop header, just print out what is the loop header
  // and return.
  if (Header != &MBB) {
    AP.OutStreamer.AddComment("  in Loop: Header=BB" +
                              Twine(AP.getFunctionNumber())+"_" +
                              Twine(Loop->getHeader()->getNumber())+
                              " Depth="+Twine(Loop->getLoopDepth()));
    return;
  }
  
  // Otherwise, it is a loop header.  Print out information about child and
  // parent loops.
  raw_ostream &OS = AP.OutStreamer.GetCommentOS();
  
  PrintParentLoopComment(OS, Loop->getParentLoop(), AP.getFunctionNumber()); 
  
  OS << "=>";
  OS.indent(Loop->getLoopDepth()*2-2);
  
  OS << "This ";
  if (Loop->empty())
    OS << "Inner ";
  OS << "Loop Header: Depth=" + Twine(Loop->getLoopDepth()) << '\n';
  
  PrintChildLoopComment(OS, Loop, AP.getFunctionNumber());
}


/// EmitBasicBlockStart - This method prints the label for the specified
/// MachineBasicBlock, an alignment (if present) and a comment describing
/// it if appropriate.
void AsmPrinter::EmitBasicBlockStart(const MachineBasicBlock *MBB) const {
  // Emit an alignment directive for this block, if needed.
  if (unsigned Align = MBB->getAlignment())
    EmitAlignment(Log2_32(Align));

  // If the block has its address taken, emit a special label to satisfy
  // references to the block. This is done so that we don't need to
  // remember the number of this label, and so that we can make
  // forward references to labels without knowing what their numbers
  // will be.
  if (MBB->hasAddressTaken()) {
    const BasicBlock *BB = MBB->getBasicBlock();
    if (VerboseAsm)
      OutStreamer.AddComment("Address Taken");
    OutStreamer.EmitLabel(GetBlockAddressSymbol(BB->getParent(), BB));
  }

  // Print the main label for the block.
  if (MBB->pred_empty() || MBB->isOnlyReachableByFallthrough()) {
    if (VerboseAsm) {
      // NOTE: Want this comment at start of line.
      O << MAI->getCommentString() << " BB#" << MBB->getNumber() << ':';
      if (const BasicBlock *BB = MBB->getBasicBlock())
        if (BB->hasName())
          OutStreamer.AddComment("%" + BB->getName());
      
      PrintBasicBlockLoopComments(*MBB, LI, *this);
      OutStreamer.AddBlankLine();
    }
  } else {
    if (VerboseAsm) {
      if (const BasicBlock *BB = MBB->getBasicBlock())
        if (BB->hasName())
          OutStreamer.AddComment("%" + BB->getName());
      PrintBasicBlockLoopComments(*MBB, LI, *this);
    }

    OutStreamer.EmitLabel(MBB->getSymbol(OutContext));
  }
}

void AsmPrinter::EmitVisibility(MCSymbol *Sym, unsigned Visibility) const {
  MCSymbolAttr Attr = MCSA_Invalid;
  
  switch (Visibility) {
  default: break;
  case GlobalValue::HiddenVisibility:
    Attr = MAI->getHiddenVisibilityAttr();
    break;
  case GlobalValue::ProtectedVisibility:
    Attr = MAI->getProtectedVisibilityAttr();
    break;
  }

  if (Attr != MCSA_Invalid)
    OutStreamer.EmitSymbolAttribute(Sym, Attr);
}

void AsmPrinter::printOffset(int64_t Offset) const {
  if (Offset > 0)
    O << '+' << Offset;
  else if (Offset < 0)
    O << Offset;
}

GCMetadataPrinter *AsmPrinter::GetOrCreateGCPrinter(GCStrategy *S) {
  if (!S->usesMetadata())
    return 0;
  
  gcp_iterator GCPI = GCMetadataPrinters.find(S);
  if (GCPI != GCMetadataPrinters.end())
    return GCPI->second;
  
  const char *Name = S->getName().c_str();
  
  for (GCMetadataPrinterRegistry::iterator
         I = GCMetadataPrinterRegistry::begin(),
         E = GCMetadataPrinterRegistry::end(); I != E; ++I)
    if (strcmp(Name, I->getName()) == 0) {
      GCMetadataPrinter *GMP = I->instantiate();
      GMP->S = S;
      GCMetadataPrinters.insert(std::make_pair(S, GMP));
      return GMP;
    }
  
  llvm_report_error("no GCMetadataPrinter registered for GC: " + Twine(Name));
  return 0;
}

/// EmitComments - Pretty-print comments for instructions
void AsmPrinter::EmitComments(const MachineInstr &MI) const {
  if (!VerboseAsm)
    return;

  bool Newline = false;

  if (!MI.getDebugLoc().isUnknown()) {
    DILocation DLT = MF->getDILocation(MI.getDebugLoc());

    // Print source line info.
    O.PadToColumn(MAI->getCommentColumn());
    O << MAI->getCommentString() << ' ';
    DIScope Scope = DLT.getScope();
    // Omit the directory, because it's likely to be long and uninteresting.
    if (!Scope.isNull())
      O << Scope.getFilename();
    else
      O << "<unknown>";
    O << ':' << DLT.getLineNumber();
    if (DLT.getColumnNumber() != 0)
      O << ':' << DLT.getColumnNumber();
    Newline = true;
  }

  // Check for spills and reloads
  int FI;

  const MachineFrameInfo *FrameInfo =
    MI.getParent()->getParent()->getFrameInfo();

  // We assume a single instruction only has a spill or reload, not
  // both.
  const MachineMemOperand *MMO;
  if (TM.getInstrInfo()->isLoadFromStackSlotPostFE(&MI, FI)) {
    if (FrameInfo->isSpillSlotObjectIndex(FI)) {
      MMO = *MI.memoperands_begin();
      if (Newline) O << '\n';
      O.PadToColumn(MAI->getCommentColumn());
      O << MAI->getCommentString() << ' ' << MMO->getSize() << "-byte Reload";
      Newline = true;
    }
  }
  else if (TM.getInstrInfo()->hasLoadFromStackSlot(&MI, MMO, FI)) {
    if (FrameInfo->isSpillSlotObjectIndex(FI)) {
      if (Newline) O << '\n';
      O.PadToColumn(MAI->getCommentColumn());
      O << MAI->getCommentString() << ' '
        << MMO->getSize() << "-byte Folded Reload";
      Newline = true;
    }
  }
  else if (TM.getInstrInfo()->isStoreToStackSlotPostFE(&MI, FI)) {
    if (FrameInfo->isSpillSlotObjectIndex(FI)) {
      MMO = *MI.memoperands_begin();
      if (Newline) O << '\n';
      O.PadToColumn(MAI->getCommentColumn());
      O << MAI->getCommentString() << ' ' << MMO->getSize() << "-byte Spill";
      Newline = true;
    }
  }
  else if (TM.getInstrInfo()->hasStoreToStackSlot(&MI, MMO, FI)) {
    if (FrameInfo->isSpillSlotObjectIndex(FI)) {
      if (Newline) O << '\n';
      O.PadToColumn(MAI->getCommentColumn());
      O << MAI->getCommentString() << ' '
        << MMO->getSize() << "-byte Folded Spill";
      Newline = true;
    }
  }

  // Check for spill-induced copies
  unsigned SrcReg, DstReg, SrcSubIdx, DstSubIdx;
  if (TM.getInstrInfo()->isMoveInstr(MI, SrcReg, DstReg,
                                      SrcSubIdx, DstSubIdx)) {
    if (MI.getAsmPrinterFlag(ReloadReuse)) {
      if (Newline) O << '\n';
      O.PadToColumn(MAI->getCommentColumn());
      O << MAI->getCommentString() << " Reload Reuse";
    }
  }
}

