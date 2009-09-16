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

#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Constants.h"
#include "llvm/Module.h"
#include "llvm/CodeGen/GCMetadataPrinter.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/DwarfWriter.h"
#include "llvm/Analysis/DebugInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/Mangler.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include <cerrno>
using namespace llvm;

static cl::opt<cl::boolOrDefault>
AsmVerbose("asm-verbose", cl::desc("Add comments to directives."),
           cl::init(cl::BOU_UNSET));

char AsmPrinter::ID = 0;
AsmPrinter::AsmPrinter(formatted_raw_ostream &o, TargetMachine &tm,
                       const MCAsmInfo *T, bool VDef)
  : MachineFunctionPass(&ID), FunctionNumber(0), O(o),
    TM(tm), MAI(T), TRI(tm.getRegisterInfo()),

    OutContext(*new MCContext()),
    // FIXME: Pass instprinter to streamer.
    OutStreamer(*createAsmStreamer(OutContext, O, *T, 0)),

    LastMI(0), LastFn(0), Counter(~0U),
    PrevDLT(0, ~0U, ~0U) {
  DW = 0; MMI = 0;
  switch (AsmVerbose) {
  case cl::BOU_UNSET: VerboseAsm = VDef;  break;
  case cl::BOU_TRUE:  VerboseAsm = true;  break;
  case cl::BOU_FALSE: VerboseAsm = false; break;
  }
}

AsmPrinter::~AsmPrinter() {
  for (gcp_iterator I = GCMetadataPrinters.begin(),
                    E = GCMetadataPrinters.end(); I != E; ++I)
    delete I->second;
  
  delete &OutStreamer;
  delete &OutContext;
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
  
  Mang = new Mangler(M, MAI->getGlobalPrefix(), MAI->getPrivateGlobalPrefix(),
                     MAI->getLinkerPrivateGlobalPrefix());
  
  if (MAI->doesAllowQuotesInName())
    Mang->setUseQuotes(true);
  
  GCModuleInfo *MI = getAnalysisIfAvailable<GCModuleInfo>();
  assert(MI && "AsmPrinter didn't require GCModuleInfo?");

  if (MAI->hasSingleParameterDotFile()) {
    /* Very minimal debug info. It is ignored if we emit actual
       debug info. If we don't, this at helps the user find where
       a function came from. */
    O << "\t.file\t\"" << M.getModuleIdentifier() << "\"\n";
  }

  for (GCModuleInfo::iterator I = MI->begin(), E = MI->end(); I != E; ++I)
    if (GCMetadataPrinter *MP = GetOrCreateGCPrinter(*I))
      MP->beginAssembly(O, *this, *MAI);
  
  if (!M.getModuleInlineAsm().empty())
    O << MAI->getCommentString() << " Start of file scope inline assembly\n"
      << M.getModuleInlineAsm()
      << '\n' << MAI->getCommentString()
      << " End of file scope inline assembly\n";

  if (MAI->doesSupportDebugInformation() ||
      MAI->doesSupportExceptionHandling()) {
    MMI = getAnalysisIfAvailable<MachineModuleInfo>();
    if (MMI)
      MMI->AnalyzeModule(M);
    DW = getAnalysisIfAvailable<DwarfWriter>();
    if (DW)
      DW->BeginModule(&M, MMI, O, this, MAI);
  }

  return false;
}

bool AsmPrinter::doFinalization(Module &M) {
  // Emit global variables.
  for (Module::const_global_iterator I = M.global_begin(), E = M.global_end();
       I != E; ++I)
    PrintGlobalVariable(I);
  
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
      if (I->hasExternalWeakLinkage())
        O << MAI->getWeakRefDirective() << Mang->getMangledName(I) << '\n';
    }
    
    for (Module::const_iterator I = M.begin(), E = M.end(); I != E; ++I) {
      if (I->hasExternalWeakLinkage())
        O << MAI->getWeakRefDirective() << Mang->getMangledName(I) << '\n';
    }
  }

  if (MAI->getSetDirective()) {
    O << '\n';
    for (Module::const_alias_iterator I = M.alias_begin(), E = M.alias_end();
         I != E; ++I) {
      std::string Name = Mang->getMangledName(I);

      const GlobalValue *GV = cast<GlobalValue>(I->getAliasedGlobal());
      std::string Target = Mang->getMangledName(GV);

      if (I->hasExternalLinkage() || !MAI->getWeakRefDirective())
        O << "\t.globl\t" << Name << '\n';
      else if (I->hasWeakLinkage())
        O << MAI->getWeakRefDirective() << Name << '\n';
      else if (!I->hasLocalLinkage())
        llvm_unreachable("Invalid alias linkage");

      printVisibility(Name, I->getVisibility());

      O << MAI->getSetDirective() << ' ' << Name << ", " << Target << '\n';
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
    if (MAI->getNonexecutableStackDirective())
      O << MAI->getNonexecutableStackDirective() << '\n';

  delete Mang; Mang = 0;
  DW = 0; MMI = 0;
  
  OutStreamer.Finish();
  return false;
}

void AsmPrinter::SetupMachineFunction(MachineFunction &MF) {
  // What's my mangled name?
  CurrentFnName = Mang->getMangledName(MF.getFunction());
  IncrementFunctionNumber();

  if (VerboseAsm)
    LI = &getAnalysis<MachineLoopInfo>();
}

namespace {
  // SectionCPs - Keep track the alignment, constpool entries per Section.
  struct SectionCPs {
    const MCSection *S;
    unsigned Alignment;
    SmallVector<unsigned, 4> CPEs;
    SectionCPs(const MCSection *s, unsigned a) : S(s), Alignment(a) {};
  };
}

/// EmitConstantPool - Print to the current output stream assembly
/// representations of the constants in the constant pool MCP. This is
/// used to print out constants which have been "spilled to memory" by
/// the code generator.
///
void AsmPrinter::EmitConstantPool(MachineConstantPool *MCP) {
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
      EmitZeros(NewOffset - Offset);

      const Type *Ty = CPE.getType();
      Offset = NewOffset + TM.getTargetData()->getTypeAllocSize(Ty);

      O << MAI->getPrivateGlobalPrefix() << "CPI" << getFunctionNumber() << '_'
        << CPI << ':';
      if (VerboseAsm) {
        O.PadToColumn(MAI->getCommentColumn());
        O << MAI->getCommentString() << " constant ";
        WriteTypeSymbolic(O, CPE.getType(), MF->getFunction()->getParent());
      }
      O << '\n';
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
void AsmPrinter::EmitJumpTableInfo(MachineJumpTableInfo *MJTI,
                                   MachineFunction &MF) {
  const std::vector<MachineJumpTableEntry> &JT = MJTI->getJumpTables();
  if (JT.empty()) return;

  bool IsPic = TM.getRelocationModel() == Reloc::PIC_;
  
  // Pick the directive to use to print the jump table entries, and switch to 
  // the appropriate section.
  TargetLowering *LoweringInfo = TM.getTargetLowering();

  const Function *F = MF.getFunction();
  bool JTInDiffSection = false;
  if (F->isWeakForLinker() ||
      (IsPic && !LoweringInfo->usesGlobalOffsetTable())) {
    // In PIC mode, we need to emit the jump table to the same section as the
    // function body itself, otherwise the label differences won't make sense.
    // We should also do if the section name is NULL or function is declared in
    // discardable section.
    OutStreamer.SwitchSection(getObjFileLowering().SectionForGlobal(F, Mang,
                                                                    TM));
  } else {
    // Otherwise, drop it in the readonly section.
    const MCSection *ReadOnlySection = 
      getObjFileLowering().getSectionForConstant(SectionKind::getReadOnly());
    OutStreamer.SwitchSection(ReadOnlySection);
    JTInDiffSection = true;
  }
  
  EmitAlignment(Log2_32(MJTI->getAlignment()));
  
  for (unsigned i = 0, e = JT.size(); i != e; ++i) {
    const std::vector<MachineBasicBlock*> &JTBBs = JT[i].MBBs;
    
    // If this jump table was deleted, ignore it. 
    if (JTBBs.empty()) continue;

    // For PIC codegen, if possible we want to use the SetDirective to reduce
    // the number of relocations the assembler will generate for the jump table.
    // Set directives are all printed before the jump table itself.
    SmallPtrSet<MachineBasicBlock*, 16> EmittedSets;
    if (MAI->getSetDirective() && IsPic)
      for (unsigned ii = 0, ee = JTBBs.size(); ii != ee; ++ii)
        if (EmittedSets.insert(JTBBs[ii]))
          printPICJumpTableSetLabel(i, JTBBs[ii]);
    
    // On some targets (e.g. Darwin) we want to emit two consequtive labels
    // before each jump table.  The first label is never referenced, but tells
    // the assembler and linker the extents of the jump table object.  The
    // second label is actually referenced by the code.
    if (JTInDiffSection && MAI->getLinkerPrivateGlobalPrefix()[0]) {
      O << MAI->getLinkerPrivateGlobalPrefix()
        << "JTI" << getFunctionNumber() << '_' << i << ":\n";
    }
    
    O << MAI->getPrivateGlobalPrefix() << "JTI" << getFunctionNumber() 
      << '_' << i << ":\n";
    
    for (unsigned ii = 0, ee = JTBBs.size(); ii != ee; ++ii) {
      printPICJumpTableEntry(MJTI, JTBBs[ii], i);
      O << '\n';
    }
  }
}

void AsmPrinter::printPICJumpTableEntry(const MachineJumpTableInfo *MJTI,
                                        const MachineBasicBlock *MBB,
                                        unsigned uid)  const {
  bool isPIC = TM.getRelocationModel() == Reloc::PIC_;
  
  // Use JumpTableDirective otherwise honor the entry size from the jump table
  // info.
  const char *JTEntryDirective = MAI->getJumpTableDirective(isPIC);
  bool HadJTEntryDirective = JTEntryDirective != NULL;
  if (!HadJTEntryDirective) {
    JTEntryDirective = MJTI->getEntrySize() == 4 ?
      MAI->getData32bitsDirective() : MAI->getData64bitsDirective();
  }

  O << JTEntryDirective << ' ';

  // If we have emitted set directives for the jump table entries, print 
  // them rather than the entries themselves.  If we're emitting PIC, then
  // emit the table entries as differences between two text section labels.
  // If we're emitting non-PIC code, then emit the entries as direct
  // references to the target basic blocks.
  if (!isPIC) {
    GetMBBSymbol(MBB->getNumber())->print(O, MAI);
  } else if (MAI->getSetDirective()) {
    O << MAI->getPrivateGlobalPrefix() << getFunctionNumber()
      << '_' << uid << "_set_" << MBB->getNumber();
  } else {
    GetMBBSymbol(MBB->getNumber())->print(O, MAI);
    // If the arch uses custom Jump Table directives, don't calc relative to
    // JT
    if (!HadJTEntryDirective) 
      O << '-' << MAI->getPrivateGlobalPrefix() << "JTI"
        << getFunctionNumber() << '_' << uid;
  }
}


/// EmitSpecialLLVMGlobal - Check to see if the specified global is a
/// special global used by LLVM.  If so, emit it and return true, otherwise
/// do nothing and return false.
bool AsmPrinter::EmitSpecialLLVMGlobal(const GlobalVariable *GV) {
  if (GV->getName() == "llvm.used") {
    if (MAI->getUsedDirective() != 0)    // No need to emit this at all.
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
    return true;
  } 
  
  if (GV->getName() == "llvm.global_dtors") {
    OutStreamer.SwitchSection(getObjFileLowering().getStaticDtorSection());
    EmitAlignment(Align, 0);
    EmitXXStructorList(GV->getInitializer());
    return true;
  }
  
  return false;
}

/// EmitLLVMUsedList - For targets that define a MAI::UsedDirective, mark each
/// global in the specified llvm.used list for which emitUsedDirectiveFor
/// is true, as being used with this directive.
void AsmPrinter::EmitLLVMUsedList(Constant *List) {
  const char *Directive = MAI->getUsedDirective();

  // Should be an array of 'i8*'.
  ConstantArray *InitList = dyn_cast<ConstantArray>(List);
  if (InitList == 0) return;
  
  for (unsigned i = 0, e = InitList->getNumOperands(); i != e; ++i) {
    const GlobalValue *GV =
      dyn_cast<GlobalValue>(InitList->getOperand(i)->stripPointerCasts());
    if (GV && getObjFileLowering().shouldEmitUsedDirectiveFor(GV, Mang)) {
      O << Directive;
      EmitConstantValueOnly(InitList->getOperand(i));
      O << '\n';
    }
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


//===----------------------------------------------------------------------===//
/// LEB 128 number encoding.

/// PrintULEB128 - Print a series of hexidecimal values (separated by commas)
/// representing an unsigned leb128 value.
void AsmPrinter::PrintULEB128(unsigned Value) const {
  char Buffer[20];
  do {
    unsigned char Byte = static_cast<unsigned char>(Value & 0x7f);
    Value >>= 7;
    if (Value) Byte |= 0x80;
    O << "0x" << utohex_buffer(Byte, Buffer+20);
    if (Value) O << ", ";
  } while (Value);
}

/// PrintSLEB128 - Print a series of hexidecimal values (separated by commas)
/// representing a signed leb128 value.
void AsmPrinter::PrintSLEB128(int Value) const {
  int Sign = Value >> (8 * sizeof(Value) - 1);
  bool IsMore;
  char Buffer[20];

  do {
    unsigned char Byte = static_cast<unsigned char>(Value & 0x7f);
    Value >>= 7;
    IsMore = Value != Sign || ((Byte ^ Sign) & 0x40) != 0;
    if (IsMore) Byte |= 0x80;
    O << "0x" << utohex_buffer(Byte, Buffer+20);
    if (IsMore) O << ", ";
  } while (IsMore);
}

//===--------------------------------------------------------------------===//
// Emission and print routines
//

/// PrintHex - Print a value as a hexidecimal value.
///
void AsmPrinter::PrintHex(int Value) const { 
  char Buffer[20];
  O << "0x" << utohex_buffer(static_cast<unsigned>(Value), Buffer+20);
}

/// EOL - Print a newline character to asm stream.  If a comment is present
/// then it will be printed first.  Comments should not contain '\n'.
void AsmPrinter::EOL() const {
  O << '\n';
}

void AsmPrinter::EOL(const std::string &Comment) const {
  if (VerboseAsm && !Comment.empty()) {
    O.PadToColumn(MAI->getCommentColumn());
    O << MAI->getCommentString()
      << ' '
      << Comment;
  }
  O << '\n';
}

void AsmPrinter::EOL(const char* Comment) const {
  if (VerboseAsm && *Comment) {
    O.PadToColumn(MAI->getCommentColumn());
    O << MAI->getCommentString()
      << ' '
      << Comment;
  }
  O << '\n';
}

static const char *DecodeDWARFEncoding(unsigned Encoding) {
  switch (Encoding) {
  case dwarf::DW_EH_PE_absptr:
    return "absptr";
  case dwarf::DW_EH_PE_omit:
    return "omit";
  case dwarf::DW_EH_PE_pcrel:
    return "pcrel";
  case dwarf::DW_EH_PE_udata4:
    return "udata4";
  case dwarf::DW_EH_PE_udata8:
    return "udata8";
  case dwarf::DW_EH_PE_sdata4:
    return "sdata4";
  case dwarf::DW_EH_PE_sdata8:
    return "sdata8";
  case dwarf::DW_EH_PE_pcrel | dwarf::DW_EH_PE_udata4:
    return "pcrel udata4";
  case dwarf::DW_EH_PE_pcrel | dwarf::DW_EH_PE_sdata4:
    return "pcrel sdata4";
  case dwarf::DW_EH_PE_pcrel | dwarf::DW_EH_PE_udata8:
    return "pcrel udata8";
  case dwarf::DW_EH_PE_pcrel | dwarf::DW_EH_PE_sdata8:
    return "pcrel sdata8";
  case dwarf::DW_EH_PE_indirect | dwarf::DW_EH_PE_pcrel |dwarf::DW_EH_PE_udata4:
    return "indirect pcrel udata4";
  case dwarf::DW_EH_PE_indirect | dwarf::DW_EH_PE_pcrel |dwarf::DW_EH_PE_sdata4:
    return "indirect pcrel sdata4";
  case dwarf::DW_EH_PE_indirect | dwarf::DW_EH_PE_pcrel |dwarf::DW_EH_PE_udata8:
    return "indirect pcrel udata8";
  case dwarf::DW_EH_PE_indirect | dwarf::DW_EH_PE_pcrel |dwarf::DW_EH_PE_sdata8:
    return "indirect pcrel sdata8";
  }

  return 0;
}

void AsmPrinter::EOL(const char *Comment, unsigned Encoding) const {
  if (VerboseAsm && *Comment) {
    O.PadToColumn(MAI->getCommentColumn());
    O << MAI->getCommentString()
      << ' '
      << Comment;

    if (const char *EncStr = DecodeDWARFEncoding(Encoding))
      O << " (" << EncStr << ')';
  }
  O << '\n';
}

/// EmitULEB128Bytes - Emit an assembler byte data directive to compose an
/// unsigned leb128 value.
void AsmPrinter::EmitULEB128Bytes(unsigned Value) const {
  if (MAI->hasLEB128()) {
    O << "\t.uleb128\t"
      << Value;
  } else {
    O << MAI->getData8bitsDirective();
    PrintULEB128(Value);
  }
}

/// EmitSLEB128Bytes - print an assembler byte data directive to compose a
/// signed leb128 value.
void AsmPrinter::EmitSLEB128Bytes(int Value) const {
  if (MAI->hasLEB128()) {
    O << "\t.sleb128\t"
      << Value;
  } else {
    O << MAI->getData8bitsDirective();
    PrintSLEB128(Value);
  }
}

/// EmitInt8 - Emit a byte directive and value.
///
void AsmPrinter::EmitInt8(int Value) const {
  O << MAI->getData8bitsDirective();
  PrintHex(Value & 0xFF);
}

/// EmitInt16 - Emit a short directive and value.
///
void AsmPrinter::EmitInt16(int Value) const {
  O << MAI->getData16bitsDirective();
  PrintHex(Value & 0xFFFF);
}

/// EmitInt32 - Emit a long directive and value.
///
void AsmPrinter::EmitInt32(int Value) const {
  O << MAI->getData32bitsDirective();
  PrintHex(Value);
}

/// EmitInt64 - Emit a long long directive and value.
///
void AsmPrinter::EmitInt64(uint64_t Value) const {
  if (MAI->getData64bitsDirective()) {
    O << MAI->getData64bitsDirective();
    PrintHex(Value);
  } else {
    if (TM.getTargetData()->isBigEndian()) {
      EmitInt32(unsigned(Value >> 32)); O << '\n';
      EmitInt32(unsigned(Value));
    } else {
      EmitInt32(unsigned(Value)); O << '\n';
      EmitInt32(unsigned(Value >> 32));
    }
  }
}

/// toOctal - Convert the low order bits of X into an octal digit.
///
static inline char toOctal(int X) {
  return (X&7)+'0';
}

/// printStringChar - Print a char, escaped if necessary.
///
static void printStringChar(formatted_raw_ostream &O, unsigned char C) {
  if (C == '"') {
    O << "\\\"";
  } else if (C == '\\') {
    O << "\\\\";
  } else if (isprint((unsigned char)C)) {
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

/// EmitString - Emit a string with quotes and a null terminator.
/// Special characters are emitted properly.
/// \literal (Eg. '\t') \endliteral
void AsmPrinter::EmitString(const std::string &String) const {
  EmitString(String.c_str(), String.size());
}

void AsmPrinter::EmitString(const char *String, unsigned Size) const {
  const char* AscizDirective = MAI->getAscizDirective();
  if (AscizDirective)
    O << AscizDirective;
  else
    O << MAI->getAsciiDirective();
  O << '\"';
  for (unsigned i = 0; i < Size; ++i)
    printStringChar(O, String[i]);
  if (AscizDirective)
    O << '\"';
  else
    O << "\\0\"";
}


/// EmitFile - Emit a .file directive.
void AsmPrinter::EmitFile(unsigned Number, const std::string &Name) const {
  O << "\t.file\t" << Number << " \"";
  for (unsigned i = 0, N = Name.size(); i < N; ++i)
    printStringChar(O, Name[i]);
  O << '\"';
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

/// EmitZeros - Emit a block of zeros.
///
void AsmPrinter::EmitZeros(uint64_t NumZeros, unsigned AddrSpace) const {
  if (NumZeros) {
    if (MAI->getZeroDirective()) {
      O << MAI->getZeroDirective() << NumZeros;
      if (MAI->getZeroDirectiveSuffix())
        O << MAI->getZeroDirectiveSuffix();
      O << '\n';
    } else {
      for (; NumZeros; --NumZeros)
        O << MAI->getData8bitsDirective(AddrSpace) << "0\n";
    }
  }
}

// Print out the specified constant, without a storage class.  Only the
// constants valid in constant expressions can occur here.
void AsmPrinter::EmitConstantValueOnly(const Constant *CV) {
  if (CV->isNullValue() || isa<UndefValue>(CV))
    O << '0';
  else if (const ConstantInt *CI = dyn_cast<ConstantInt>(CV)) {
    O << CI->getZExtValue();
  } else if (const GlobalValue *GV = dyn_cast<GlobalValue>(CV)) {
    // This is a constant address for a global variable or function. Use the
    // name of the variable or function as the address value.
    O << Mang->getMangledName(GV);
  } else if (const ConstantExpr *CE = dyn_cast<ConstantExpr>(CV)) {
    const TargetData *TD = TM.getTargetData();
    unsigned Opcode = CE->getOpcode();    
    switch (Opcode) {
    case Instruction::Trunc:
    case Instruction::ZExt:
    case Instruction::SExt:
    case Instruction::FPTrunc:
    case Instruction::FPExt:
    case Instruction::UIToFP:
    case Instruction::SIToFP:
    case Instruction::FPToUI:
    case Instruction::FPToSI:
      llvm_unreachable("FIXME: Don't support this constant cast expr");
    case Instruction::GetElementPtr: {
      // generate a symbolic expression for the byte address
      const Constant *ptrVal = CE->getOperand(0);
      SmallVector<Value*, 8> idxVec(CE->op_begin()+1, CE->op_end());
      if (int64_t Offset = TD->getIndexedOffset(ptrVal->getType(), &idxVec[0],
                                                idxVec.size())) {
        // Truncate/sext the offset to the pointer size.
        if (TD->getPointerSizeInBits() != 64) {
          int SExtAmount = 64-TD->getPointerSizeInBits();
          Offset = (Offset << SExtAmount) >> SExtAmount;
        }
        
        if (Offset)
          O << '(';
        EmitConstantValueOnly(ptrVal);
        if (Offset > 0)
          O << ") + " << Offset;
        else if (Offset < 0)
          O << ") - " << -Offset;
      } else {
        EmitConstantValueOnly(ptrVal);
      }
      break;
    }
    case Instruction::BitCast:
      return EmitConstantValueOnly(CE->getOperand(0));

    case Instruction::IntToPtr: {
      // Handle casts to pointers by changing them into casts to the appropriate
      // integer type.  This promotes constant folding and simplifies this code.
      Constant *Op = CE->getOperand(0);
      Op = ConstantExpr::getIntegerCast(Op, TD->getIntPtrType(CV->getContext()),
                                        false/*ZExt*/);
      return EmitConstantValueOnly(Op);
    }
      
      
    case Instruction::PtrToInt: {
      // Support only foldable casts to/from pointers that can be eliminated by
      // changing the pointer to the appropriately sized integer type.
      Constant *Op = CE->getOperand(0);
      const Type *Ty = CE->getType();

      // We can emit the pointer value into this slot if the slot is an
      // integer slot greater or equal to the size of the pointer.
      if (TD->getTypeAllocSize(Ty) == TD->getTypeAllocSize(Op->getType()))
        return EmitConstantValueOnly(Op);

      O << "((";
      EmitConstantValueOnly(Op);
      APInt ptrMask =
        APInt::getAllOnesValue(TD->getTypeAllocSizeInBits(Op->getType()));
      
      SmallString<40> S;
      ptrMask.toStringUnsigned(S);
      O << ") & " << S.str() << ')';
      break;
    }
    case Instruction::Add:
    case Instruction::Sub:
    case Instruction::And:
    case Instruction::Or:
    case Instruction::Xor:
      O << '(';
      EmitConstantValueOnly(CE->getOperand(0));
      O << ')';
      switch (Opcode) {
      case Instruction::Add:
       O << " + ";
       break;
      case Instruction::Sub:
       O << " - ";
       break;
      case Instruction::And:
       O << " & ";
       break;
      case Instruction::Or:
       O << " | ";
       break;
      case Instruction::Xor:
       O << " ^ ";
       break;
      default:
       break;
      }
      O << '(';
      EmitConstantValueOnly(CE->getOperand(1));
      O << ')';
      break;
    default:
      llvm_unreachable("Unsupported operator!");
    }
  } else {
    llvm_unreachable("Unknown constant value!");
  }
}

/// printAsCString - Print the specified array as a C compatible string, only if
/// the predicate isString is true.
///
static void printAsCString(formatted_raw_ostream &O, const ConstantArray *CVA,
                           unsigned LastElt) {
  assert(CVA->isString() && "Array is not string compatible!");

  O << '\"';
  for (unsigned i = 0; i != LastElt; ++i) {
    unsigned char C =
        (unsigned char)cast<ConstantInt>(CVA->getOperand(i))->getZExtValue();
    printStringChar(O, C);
  }
  O << '\"';
}

/// EmitString - Emit a zero-byte-terminated string constant.
///
void AsmPrinter::EmitString(const ConstantArray *CVA) const {
  unsigned NumElts = CVA->getNumOperands();
  if (MAI->getAscizDirective() && NumElts && 
      cast<ConstantInt>(CVA->getOperand(NumElts-1))->getZExtValue() == 0) {
    O << MAI->getAscizDirective();
    printAsCString(O, CVA, NumElts-1);
  } else {
    O << MAI->getAsciiDirective();
    printAsCString(O, CVA, NumElts);
  }
  O << '\n';
}

void AsmPrinter::EmitGlobalConstantArray(const ConstantArray *CVA,
                                         unsigned AddrSpace) {
  if (CVA->isString()) {
    EmitString(CVA);
  } else { // Not a string.  Print the values in successive locations
    for (unsigned i = 0, e = CVA->getNumOperands(); i != e; ++i)
      EmitGlobalConstant(CVA->getOperand(i), AddrSpace);
  }
}

void AsmPrinter::EmitGlobalConstantVector(const ConstantVector *CP) {
  const VectorType *PTy = CP->getType();
  
  for (unsigned I = 0, E = PTy->getNumElements(); I < E; ++I)
    EmitGlobalConstant(CP->getOperand(I));
}

void AsmPrinter::EmitGlobalConstantStruct(const ConstantStruct *CVS,
                                          unsigned AddrSpace) {
  // Print the fields in successive locations. Pad to align if needed!
  const TargetData *TD = TM.getTargetData();
  unsigned Size = TD->getTypeAllocSize(CVS->getType());
  const StructLayout *cvsLayout = TD->getStructLayout(CVS->getType());
  uint64_t sizeSoFar = 0;
  for (unsigned i = 0, e = CVS->getNumOperands(); i != e; ++i) {
    const Constant* field = CVS->getOperand(i);

    // Check if padding is needed and insert one or more 0s.
    uint64_t fieldSize = TD->getTypeAllocSize(field->getType());
    uint64_t padSize = ((i == e-1 ? Size : cvsLayout->getElementOffset(i+1))
                        - cvsLayout->getElementOffset(i)) - fieldSize;
    sizeSoFar += fieldSize + padSize;

    // Now print the actual field value.
    EmitGlobalConstant(field, AddrSpace);

    // Insert padding - this may include padding to increase the size of the
    // current field up to the ABI size (if the struct is not packed) as well
    // as padding to ensure that the next field starts at the right offset.
    EmitZeros(padSize, AddrSpace);
  }
  assert(sizeSoFar == cvsLayout->getSizeInBytes() &&
         "Layout of constant struct may be incorrect!");
}

void AsmPrinter::EmitGlobalConstantFP(const ConstantFP *CFP, 
                                      unsigned AddrSpace) {
  // FP Constants are printed as integer constants to avoid losing
  // precision...
  LLVMContext &Context = CFP->getContext();
  const TargetData *TD = TM.getTargetData();
  if (CFP->getType() == Type::getDoubleTy(Context)) {
    double Val = CFP->getValueAPF().convertToDouble();  // for comment only
    uint64_t i = CFP->getValueAPF().bitcastToAPInt().getZExtValue();
    if (MAI->getData64bitsDirective(AddrSpace)) {
      O << MAI->getData64bitsDirective(AddrSpace) << i;
      if (VerboseAsm) {
        O.PadToColumn(MAI->getCommentColumn());
        O << MAI->getCommentString() << " double " << Val;
      }
      O << '\n';
    } else if (TD->isBigEndian()) {
      O << MAI->getData32bitsDirective(AddrSpace) << unsigned(i >> 32);
      if (VerboseAsm) {
        O.PadToColumn(MAI->getCommentColumn());
        O << MAI->getCommentString()
          << " most significant word of double " << Val;
      }
      O << '\n';
      O << MAI->getData32bitsDirective(AddrSpace) << unsigned(i);
      if (VerboseAsm) {
        O.PadToColumn(MAI->getCommentColumn());
        O << MAI->getCommentString()
          << " least significant word of double " << Val;
      }
      O << '\n';
    } else {
      O << MAI->getData32bitsDirective(AddrSpace) << unsigned(i);
      if (VerboseAsm) {
        O.PadToColumn(MAI->getCommentColumn());
        O << MAI->getCommentString()
          << " least significant word of double " << Val;
      }
      O << '\n';
      O << MAI->getData32bitsDirective(AddrSpace) << unsigned(i >> 32);
      if (VerboseAsm) {
        O.PadToColumn(MAI->getCommentColumn());
        O << MAI->getCommentString()
          << " most significant word of double " << Val;
      }
      O << '\n';
    }
    return;
  } else if (CFP->getType() == Type::getFloatTy(Context)) {
    float Val = CFP->getValueAPF().convertToFloat();  // for comment only
    O << MAI->getData32bitsDirective(AddrSpace)
      << CFP->getValueAPF().bitcastToAPInt().getZExtValue();
    if (VerboseAsm) {
      O.PadToColumn(MAI->getCommentColumn());
      O << MAI->getCommentString() << " float " << Val;
    }
    O << '\n';
    return;
  } else if (CFP->getType() == Type::getX86_FP80Ty(Context)) {
    // all long double variants are printed as hex
    // api needed to prevent premature destruction
    APInt api = CFP->getValueAPF().bitcastToAPInt();
    const uint64_t *p = api.getRawData();
    // Convert to double so we can print the approximate val as a comment.
    APFloat DoubleVal = CFP->getValueAPF();
    bool ignored;
    DoubleVal.convert(APFloat::IEEEdouble, APFloat::rmNearestTiesToEven,
                      &ignored);
    if (TD->isBigEndian()) {
      O << MAI->getData16bitsDirective(AddrSpace) << uint16_t(p[1]);
      if (VerboseAsm) {
        O.PadToColumn(MAI->getCommentColumn());
        O << MAI->getCommentString()
          << " most significant halfword of x86_fp80 ~"
          << DoubleVal.convertToDouble();
      }
      O << '\n';
      O << MAI->getData16bitsDirective(AddrSpace) << uint16_t(p[0] >> 48);
      if (VerboseAsm) {
        O.PadToColumn(MAI->getCommentColumn());
        O << MAI->getCommentString() << " next halfword";
      }
      O << '\n';
      O << MAI->getData16bitsDirective(AddrSpace) << uint16_t(p[0] >> 32);
      if (VerboseAsm) {
        O.PadToColumn(MAI->getCommentColumn());
        O << MAI->getCommentString() << " next halfword";
      }
      O << '\n';
      O << MAI->getData16bitsDirective(AddrSpace) << uint16_t(p[0] >> 16);
      if (VerboseAsm) {
        O.PadToColumn(MAI->getCommentColumn());
        O << MAI->getCommentString() << " next halfword";
      }
      O << '\n';
      O << MAI->getData16bitsDirective(AddrSpace) << uint16_t(p[0]);
      if (VerboseAsm) {
        O.PadToColumn(MAI->getCommentColumn());
        O << MAI->getCommentString()
          << " least significant halfword";
      }
      O << '\n';
     } else {
      O << MAI->getData16bitsDirective(AddrSpace) << uint16_t(p[0]);
      if (VerboseAsm) {
        O.PadToColumn(MAI->getCommentColumn());
        O << MAI->getCommentString()
          << " least significant halfword of x86_fp80 ~"
          << DoubleVal.convertToDouble();
      }
      O << '\n';
      O << MAI->getData16bitsDirective(AddrSpace) << uint16_t(p[0] >> 16);
      if (VerboseAsm) {
        O.PadToColumn(MAI->getCommentColumn());
        O << MAI->getCommentString()
          << " next halfword";
      }
      O << '\n';
      O << MAI->getData16bitsDirective(AddrSpace) << uint16_t(p[0] >> 32);
      if (VerboseAsm) {
        O.PadToColumn(MAI->getCommentColumn());
        O << MAI->getCommentString()
          << " next halfword";
      }
      O << '\n';
      O << MAI->getData16bitsDirective(AddrSpace) << uint16_t(p[0] >> 48);
      if (VerboseAsm) {
        O.PadToColumn(MAI->getCommentColumn());
        O << MAI->getCommentString()
          << " next halfword";
      }
      O << '\n';
      O << MAI->getData16bitsDirective(AddrSpace) << uint16_t(p[1]);
      if (VerboseAsm) {
        O.PadToColumn(MAI->getCommentColumn());
        O << MAI->getCommentString()
          << " most significant halfword";
      }
      O << '\n';
    }
    EmitZeros(TD->getTypeAllocSize(Type::getX86_FP80Ty(Context)) -
              TD->getTypeStoreSize(Type::getX86_FP80Ty(Context)), AddrSpace);
    return;
  } else if (CFP->getType() == Type::getPPC_FP128Ty(Context)) {
    // all long double variants are printed as hex
    // api needed to prevent premature destruction
    APInt api = CFP->getValueAPF().bitcastToAPInt();
    const uint64_t *p = api.getRawData();
    if (TD->isBigEndian()) {
      O << MAI->getData32bitsDirective(AddrSpace) << uint32_t(p[0] >> 32);
      if (VerboseAsm) {
        O.PadToColumn(MAI->getCommentColumn());
        O << MAI->getCommentString()
          << " most significant word of ppc_fp128";
      }
      O << '\n';
      O << MAI->getData32bitsDirective(AddrSpace) << uint32_t(p[0]);
      if (VerboseAsm) {
        O.PadToColumn(MAI->getCommentColumn());
        O << MAI->getCommentString()
        << " next word";
      }
      O << '\n';
      O << MAI->getData32bitsDirective(AddrSpace) << uint32_t(p[1] >> 32);
      if (VerboseAsm) {
        O.PadToColumn(MAI->getCommentColumn());
        O << MAI->getCommentString()
          << " next word";
      }
      O << '\n';
      O << MAI->getData32bitsDirective(AddrSpace) << uint32_t(p[1]);
      if (VerboseAsm) {
        O.PadToColumn(MAI->getCommentColumn());
        O << MAI->getCommentString()
          << " least significant word";
      }
      O << '\n';
     } else {
      O << MAI->getData32bitsDirective(AddrSpace) << uint32_t(p[1]);
      if (VerboseAsm) {
        O.PadToColumn(MAI->getCommentColumn());
        O << MAI->getCommentString()
          << " least significant word of ppc_fp128";
      }
      O << '\n';
      O << MAI->getData32bitsDirective(AddrSpace) << uint32_t(p[1] >> 32);
      if (VerboseAsm) {
        O.PadToColumn(MAI->getCommentColumn());
        O << MAI->getCommentString()
          << " next word";
      }
      O << '\n';
      O << MAI->getData32bitsDirective(AddrSpace) << uint32_t(p[0]);
      if (VerboseAsm) {
        O.PadToColumn(MAI->getCommentColumn());
        O << MAI->getCommentString()
          << " next word";
      }
      O << '\n';
      O << MAI->getData32bitsDirective(AddrSpace) << uint32_t(p[0] >> 32);
      if (VerboseAsm) {
        O.PadToColumn(MAI->getCommentColumn());
        O << MAI->getCommentString()
          << " most significant word";
      }
      O << '\n';
    }
    return;
  } else llvm_unreachable("Floating point constant type not handled");
}

void AsmPrinter::EmitGlobalConstantLargeInt(const ConstantInt *CI,
                                            unsigned AddrSpace) {
  const TargetData *TD = TM.getTargetData();
  unsigned BitWidth = CI->getBitWidth();
  assert(isPowerOf2_32(BitWidth) &&
         "Non-power-of-2-sized integers not handled!");

  // We don't expect assemblers to support integer data directives
  // for more than 64 bits, so we emit the data in at most 64-bit
  // quantities at a time.
  const uint64_t *RawData = CI->getValue().getRawData();
  for (unsigned i = 0, e = BitWidth / 64; i != e; ++i) {
    uint64_t Val;
    if (TD->isBigEndian())
      Val = RawData[e - i - 1];
    else
      Val = RawData[i];

    if (MAI->getData64bitsDirective(AddrSpace))
      O << MAI->getData64bitsDirective(AddrSpace) << Val << '\n';
    else if (TD->isBigEndian()) {
      O << MAI->getData32bitsDirective(AddrSpace) << unsigned(Val >> 32);
      if (VerboseAsm) {
        O.PadToColumn(MAI->getCommentColumn());
        O << MAI->getCommentString()
          << " most significant half of i64 " << Val;
      }
      O << '\n';
      O << MAI->getData32bitsDirective(AddrSpace) << unsigned(Val);
      if (VerboseAsm) {
        O.PadToColumn(MAI->getCommentColumn());
        O << MAI->getCommentString()
          << " least significant half of i64 " << Val;
      }
      O << '\n';
    } else {
      O << MAI->getData32bitsDirective(AddrSpace) << unsigned(Val);
      if (VerboseAsm) {
        O.PadToColumn(MAI->getCommentColumn());
        O << MAI->getCommentString()
          << " least significant half of i64 " << Val;
      }
      O << '\n';
      O << MAI->getData32bitsDirective(AddrSpace) << unsigned(Val >> 32);
      if (VerboseAsm) {
        O.PadToColumn(MAI->getCommentColumn());
        O << MAI->getCommentString()
          << " most significant half of i64 " << Val;
      }
      O << '\n';
    }
  }
}

/// EmitGlobalConstant - Print a general LLVM constant to the .s file.
void AsmPrinter::EmitGlobalConstant(const Constant *CV, unsigned AddrSpace) {
  const TargetData *TD = TM.getTargetData();
  const Type *type = CV->getType();
  unsigned Size = TD->getTypeAllocSize(type);

  if (CV->isNullValue() || isa<UndefValue>(CV)) {
    EmitZeros(Size, AddrSpace);
    return;
  } else if (const ConstantArray *CVA = dyn_cast<ConstantArray>(CV)) {
    EmitGlobalConstantArray(CVA , AddrSpace);
    return;
  } else if (const ConstantStruct *CVS = dyn_cast<ConstantStruct>(CV)) {
    EmitGlobalConstantStruct(CVS, AddrSpace);
    return;
  } else if (const ConstantFP *CFP = dyn_cast<ConstantFP>(CV)) {
    EmitGlobalConstantFP(CFP, AddrSpace);
    return;
  } else if (const ConstantInt *CI = dyn_cast<ConstantInt>(CV)) {
    // Small integers are handled below; large integers are handled here.
    if (Size > 4) {
      EmitGlobalConstantLargeInt(CI, AddrSpace);
      return;
    }
  } else if (const ConstantVector *CP = dyn_cast<ConstantVector>(CV)) {
    EmitGlobalConstantVector(CP);
    return;
  }

  printDataDirective(type, AddrSpace);
  EmitConstantValueOnly(CV);
  if (VerboseAsm) {
    if (const ConstantInt *CI = dyn_cast<ConstantInt>(CV)) {
      SmallString<40> S;
      CI->getValue().toStringUnsigned(S, 16);
      O.PadToColumn(MAI->getCommentColumn());
      O << MAI->getCommentString() << " 0x" << S.str();
    }
  }
  O << '\n';
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
void AsmPrinter::processDebugLoc(DebugLoc DL) {
  if (!MAI || !DW)
    return;
  
  if (MAI->doesSupportDebugInformation() && DW->ShouldEmitDwarfDebug()) {
    if (!DL.isUnknown()) {
      DebugLocTuple CurDLT = MF->getDebugLocTuple(DL);

      if (CurDLT.CompileUnit != 0 && PrevDLT != CurDLT) {
        printLabel(DW->RecordSourceLine(CurDLT.Line, CurDLT.Col,
                                        DICompileUnit(CurDLT.CompileUnit)));
        O << '\n';
      }

      PrevDLT = CurDLT;
    }
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

          if (Modifier[0]=='l')  // labels are target independent
            GetMBBSymbol(MI->getOperand(OpNo).getMBB()
                           ->getNumber())->print(O, MAI);
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
          Msg << "Invalid operand found in inline asm: '"
               << AsmStr << "'\n";
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
/// overried this to format as appropriate.
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

MCSymbol *AsmPrinter::GetMBBSymbol(unsigned MBBID) const {
  SmallString<60> Name;
  raw_svector_ostream(Name) << MAI->getPrivateGlobalPrefix() << "BB"
    << getFunctionNumber() << '_' << MBBID;
  
  return OutContext.GetOrCreateSymbol(Name.str());
}


/// EmitBasicBlockStart - This method prints the label for the specified
/// MachineBasicBlock, an alignment (if present) and a comment describing
/// it if appropriate.
void AsmPrinter::EmitBasicBlockStart(const MachineBasicBlock *MBB) const {
  if (unsigned Align = MBB->getAlignment())
    EmitAlignment(Log2_32(Align));

  GetMBBSymbol(MBB->getNumber())->print(O, MAI);
  O << ':';
  
  if (VerboseAsm) {
    if (const BasicBlock *BB = MBB->getBasicBlock())
      if (BB->hasName()) {
        O.PadToColumn(MAI->getCommentColumn());
        O << MAI->getCommentString() << ' ';
        WriteAsOperand(O, BB, /*PrintType=*/false);
      }

    EmitComments(*MBB);
  }
}

/// printPICJumpTableSetLabel - This method prints a set label for the
/// specified MachineBasicBlock for a jumptable entry.
void AsmPrinter::printPICJumpTableSetLabel(unsigned uid, 
                                           const MachineBasicBlock *MBB) const {
  if (!MAI->getSetDirective())
    return;
  
  O << MAI->getSetDirective() << ' ' << MAI->getPrivateGlobalPrefix()
    << getFunctionNumber() << '_' << uid << "_set_" << MBB->getNumber() << ',';
  GetMBBSymbol(MBB->getNumber())->print(O, MAI);
  O << '-' << MAI->getPrivateGlobalPrefix() << "JTI" << getFunctionNumber() 
    << '_' << uid << '\n';
}

void AsmPrinter::printPICJumpTableSetLabel(unsigned uid, unsigned uid2,
                                           const MachineBasicBlock *MBB) const {
  if (!MAI->getSetDirective())
    return;
  
  O << MAI->getSetDirective() << ' ' << MAI->getPrivateGlobalPrefix()
    << getFunctionNumber() << '_' << uid << '_' << uid2
    << "_set_" << MBB->getNumber() << ',';
  GetMBBSymbol(MBB->getNumber())->print(O, MAI);
  O << '-' << MAI->getPrivateGlobalPrefix() << "JTI" << getFunctionNumber() 
    << '_' << uid << '_' << uid2 << '\n';
}

/// printDataDirective - This method prints the asm directive for the
/// specified type.
void AsmPrinter::printDataDirective(const Type *type, unsigned AddrSpace) {
  const TargetData *TD = TM.getTargetData();
  switch (type->getTypeID()) {
  case Type::FloatTyID: case Type::DoubleTyID:
  case Type::X86_FP80TyID: case Type::FP128TyID: case Type::PPC_FP128TyID:
    assert(0 && "Should have already output floating point constant.");
  default:
    assert(0 && "Can't handle printing this type of thing");
  case Type::IntegerTyID: {
    unsigned BitWidth = cast<IntegerType>(type)->getBitWidth();
    if (BitWidth <= 8)
      O << MAI->getData8bitsDirective(AddrSpace);
    else if (BitWidth <= 16)
      O << MAI->getData16bitsDirective(AddrSpace);
    else if (BitWidth <= 32)
      O << MAI->getData32bitsDirective(AddrSpace);
    else if (BitWidth <= 64) {
      assert(MAI->getData64bitsDirective(AddrSpace) &&
             "Target cannot handle 64-bit constant exprs!");
      O << MAI->getData64bitsDirective(AddrSpace);
    } else {
      llvm_unreachable("Target cannot handle given data directive width!");
    }
    break;
  }
  case Type::PointerTyID:
    if (TD->getPointerSize() == 8) {
      assert(MAI->getData64bitsDirective(AddrSpace) &&
             "Target cannot handle 64-bit pointer exprs!");
      O << MAI->getData64bitsDirective(AddrSpace);
    } else if (TD->getPointerSize() == 2) {
      O << MAI->getData16bitsDirective(AddrSpace);
    } else if (TD->getPointerSize() == 1) {
      O << MAI->getData8bitsDirective(AddrSpace);
    } else {
      O << MAI->getData32bitsDirective(AddrSpace);
    }
    break;
  }
}

void AsmPrinter::printVisibility(const std::string& Name,
                                 unsigned Visibility) const {
  if (Visibility == GlobalValue::HiddenVisibility) {
    if (const char *Directive = MAI->getHiddenDirective())
      O << Directive << Name << '\n';
  } else if (Visibility == GlobalValue::ProtectedVisibility) {
    if (const char *Directive = MAI->getProtectedDirective())
      O << Directive << Name << '\n';
  }
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
  
  errs() << "no GCMetadataPrinter registered for GC: " << Name << "\n";
  llvm_unreachable(0);
}

/// EmitComments - Pretty-print comments for instructions
void AsmPrinter::EmitComments(const MachineInstr &MI) const {
  assert(VerboseAsm && !MI.getDebugLoc().isUnknown());
  
  DebugLocTuple DLT = MF->getDebugLocTuple(MI.getDebugLoc());

  // Print source line info.
  O.PadToColumn(MAI->getCommentColumn());
  O << MAI->getCommentString() << " SrcLine ";
  if (DLT.CompileUnit) {
    std::string Str;
    DICompileUnit CU(DLT.CompileUnit);
    O << CU.getFilename(Str) << " ";
  }
  O << DLT.Line;
  if (DLT.Col != 0) 
    O << ":" << DLT.Col;
}

/// PrintChildLoopComment - Print comments about child loops within
/// the loop for this basic block, with nesting.
///
static void PrintChildLoopComment(formatted_raw_ostream &O,
                                  const MachineLoop *loop,
                                  const MCAsmInfo *MAI,
                                  int FunctionNumber) {
  // Add child loop information
  for(MachineLoop::iterator cl = loop->begin(),
        clend = loop->end();
      cl != clend;
      ++cl) {
    MachineBasicBlock *Header = (*cl)->getHeader();
    assert(Header && "No header for loop");

    O << '\n';
    O.PadToColumn(MAI->getCommentColumn());

    O << MAI->getCommentString();
    O.indent(((*cl)->getLoopDepth()-1)*2)
      << " Child Loop BB" << FunctionNumber << "_"
      << Header->getNumber() << " Depth " << (*cl)->getLoopDepth();

    PrintChildLoopComment(O, *cl, MAI, FunctionNumber);
  }
}

/// EmitComments - Pretty-print comments for basic blocks
void AsmPrinter::EmitComments(const MachineBasicBlock &MBB) const
{
  if (VerboseAsm) {
    // Add loop depth information
    const MachineLoop *loop = LI->getLoopFor(&MBB);

    if (loop) {
      // Print a newline after bb# annotation.
      O << "\n";
      O.PadToColumn(MAI->getCommentColumn());
      O << MAI->getCommentString() << " Loop Depth " << loop->getLoopDepth()
        << '\n';

      O.PadToColumn(MAI->getCommentColumn());

      MachineBasicBlock *Header = loop->getHeader();
      assert(Header && "No header for loop");
      
      if (Header == &MBB) {
        O << MAI->getCommentString() << " Loop Header";
        PrintChildLoopComment(O, loop, MAI, getFunctionNumber());
      }
      else {
        O << MAI->getCommentString() << " Loop Header is BB"
          << getFunctionNumber() << "_" << loop->getHeader()->getNumber();
      }

      if (loop->empty()) {
        O << '\n';
        O.PadToColumn(MAI->getCommentColumn());
        O << MAI->getCommentString() << " Inner Loop";
      }

      // Add parent loop information
      for (const MachineLoop *CurLoop = loop->getParentLoop();
           CurLoop;
           CurLoop = CurLoop->getParentLoop()) {
        MachineBasicBlock *Header = CurLoop->getHeader();
        assert(Header && "No header for loop");

        O << '\n';
        O.PadToColumn(MAI->getCommentColumn());
        O << MAI->getCommentString();
        O.indent((CurLoop->getLoopDepth()-1)*2)
          << " Inside Loop BB" << getFunctionNumber() << "_"
          << Header->getNumber() << " Depth " << CurLoop->getLoopDepth();
      }
    }
  }
}
