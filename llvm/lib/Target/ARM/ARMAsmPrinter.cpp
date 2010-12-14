//===-- ARMAsmPrinter.cpp - Print machine code to an ARM .s file ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a printer that converts from our internal representation
// of machine-dependent LLVM code to GAS-format ARM assembly language.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "asm-printer"
#include "ARM.h"
#include "ARMBuildAttrs.h"
#include "ARMAddressingModes.h"
#include "ARMConstantPoolValue.h"
#include "InstPrinter/ARMInstPrinter.h"
#include "ARMAsmPrinter.h"
#include "ARMMachineFunctionInfo.h"
#include "ARMTargetMachine.h"
#include "ARMTargetObjectFile.h"
#include "llvm/Analysis/DebugInfo.h"
#include "llvm/Constants.h"
#include "llvm/Module.h"
#include "llvm/Type.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/CodeGen/MachineModuleInfoImpls.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/MC/MCObjectStreamer.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Target/Mangler.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetRegistry.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <cctype>
using namespace llvm;

namespace {

  // Per section and per symbol attributes are not supported.
  // To implement them we would need the ability to delay this emission
  // until the assembly file is fully parsed/generated as only then do we
  // know the symbol and section numbers.
  class AttributeEmitter {
  public:
    virtual void MaybeSwitchVendor(StringRef Vendor) = 0;
    virtual void EmitAttribute(unsigned Attribute, unsigned Value) = 0;
    virtual void Finish() = 0;
    virtual ~AttributeEmitter() {}
  };

  class AsmAttributeEmitter : public AttributeEmitter {
    MCStreamer &Streamer;

  public:
    AsmAttributeEmitter(MCStreamer &Streamer_) : Streamer(Streamer_) {}
    void MaybeSwitchVendor(StringRef Vendor) { }

    void EmitAttribute(unsigned Attribute, unsigned Value) {
      Streamer.EmitRawText("\t.eabi_attribute " +
                           Twine(Attribute) + ", " + Twine(Value));
    }

    void Finish() { }
  };

  class ObjectAttributeEmitter : public AttributeEmitter {
    MCObjectStreamer &Streamer;
    StringRef CurrentVendor;
    SmallString<64> Contents;

  public:
    ObjectAttributeEmitter(MCObjectStreamer &Streamer_) :
      Streamer(Streamer_), CurrentVendor("") { }

    void MaybeSwitchVendor(StringRef Vendor) {
      assert(!Vendor.empty() && "Vendor cannot be empty.");

      if (CurrentVendor.empty())
        CurrentVendor = Vendor;
      else if (CurrentVendor == Vendor)
        return;
      else
        Finish();

      CurrentVendor = Vendor;

      assert(Contents.size() == 0);
    }

    void EmitAttribute(unsigned Attribute, unsigned Value) {
      // FIXME: should be ULEB
      Contents += Attribute;
      Contents += Value;
    }

    void Finish() {
      const size_t ContentsSize = Contents.size();

      // Vendor size + Vendor name + '\0'
      const size_t VendorHeaderSize = 4 + CurrentVendor.size() + 1;

      // Tag + Tag Size
      const size_t TagHeaderSize = 1 + 4;

      Streamer.EmitIntValue(VendorHeaderSize + TagHeaderSize + ContentsSize, 4);
      Streamer.EmitBytes(CurrentVendor, 0);
      Streamer.EmitIntValue(0, 1); // '\0'

      Streamer.EmitIntValue(ARMBuildAttrs::File, 1);
      Streamer.EmitIntValue(TagHeaderSize + ContentsSize, 4);

      Streamer.EmitBytes(Contents, 0);

      Contents.clear();
    }
  };

} // end of anonymous namespace

MachineLocation ARMAsmPrinter::
getDebugValueLocation(const MachineInstr *MI) const {
  MachineLocation Location;
  assert(MI->getNumOperands() == 4 && "Invalid no. of machine operands!");
  // Frame address.  Currently handles register +- offset only.
  if (MI->getOperand(0).isReg() && MI->getOperand(1).isImm())
    Location.set(MI->getOperand(0).getReg(), MI->getOperand(1).getImm());
  else {
    DEBUG(dbgs() << "DBG_VALUE instruction ignored! " << *MI << "\n");
  }
  return Location;
}

void ARMAsmPrinter::EmitFunctionEntryLabel() {
  if (AFI->isThumbFunction()) {
    OutStreamer.EmitAssemblerFlag(MCAF_Code16);
    OutStreamer.EmitThumbFunc(Subtarget->isTargetDarwin()? CurrentFnSym : 0);
  }

  OutStreamer.EmitLabel(CurrentFnSym);
}

/// runOnMachineFunction - This uses the EmitInstruction()
/// method to print assembly for each instruction.
///
bool ARMAsmPrinter::runOnMachineFunction(MachineFunction &MF) {
  AFI = MF.getInfo<ARMFunctionInfo>();
  MCP = MF.getConstantPool();

  return AsmPrinter::runOnMachineFunction(MF);
}

void ARMAsmPrinter::printOperand(const MachineInstr *MI, int OpNum,
                                 raw_ostream &O, const char *Modifier) {
  const MachineOperand &MO = MI->getOperand(OpNum);
  unsigned TF = MO.getTargetFlags();

  switch (MO.getType()) {
  default:
    assert(0 && "<unknown operand type>");
  case MachineOperand::MO_Register: {
    unsigned Reg = MO.getReg();
    assert(TargetRegisterInfo::isPhysicalRegister(Reg));
    assert(!MO.getSubReg() && "Subregs should be eliminated!");
    O << ARMInstPrinter::getRegisterName(Reg);
    break;
  }
  case MachineOperand::MO_Immediate: {
    int64_t Imm = MO.getImm();
    O << '#';
    if ((Modifier && strcmp(Modifier, "lo16") == 0) ||
        (TF == ARMII::MO_LO16))
      O << ":lower16:";
    else if ((Modifier && strcmp(Modifier, "hi16") == 0) ||
             (TF == ARMII::MO_HI16))
      O << ":upper16:";
    O << Imm;
    break;
  }
  case MachineOperand::MO_MachineBasicBlock:
    O << *MO.getMBB()->getSymbol();
    return;
  case MachineOperand::MO_GlobalAddress: {
    const GlobalValue *GV = MO.getGlobal();
    if ((Modifier && strcmp(Modifier, "lo16") == 0) ||
        (TF & ARMII::MO_LO16))
      O << ":lower16:";
    else if ((Modifier && strcmp(Modifier, "hi16") == 0) ||
             (TF & ARMII::MO_HI16))
      O << ":upper16:";
    O << *Mang->getSymbol(GV);

    printOffset(MO.getOffset(), O);
    if (TF == ARMII::MO_PLT)
      O << "(PLT)";
    break;
  }
  case MachineOperand::MO_ExternalSymbol: {
    O << *GetExternalSymbolSymbol(MO.getSymbolName());
    if (TF == ARMII::MO_PLT)
      O << "(PLT)";
    break;
  }
  case MachineOperand::MO_ConstantPoolIndex:
    O << *GetCPISymbol(MO.getIndex());
    break;
  case MachineOperand::MO_JumpTableIndex:
    O << *GetJTISymbol(MO.getIndex());
    break;
  }
}

//===--------------------------------------------------------------------===//

MCSymbol *ARMAsmPrinter::
GetARMSetPICJumpTableLabel2(unsigned uid, unsigned uid2,
                            const MachineBasicBlock *MBB) const {
  SmallString<60> Name;
  raw_svector_ostream(Name) << MAI->getPrivateGlobalPrefix()
    << getFunctionNumber() << '_' << uid << '_' << uid2
    << "_set_" << MBB->getNumber();
  return OutContext.GetOrCreateSymbol(Name.str());
}

MCSymbol *ARMAsmPrinter::
GetARMJTIPICJumpTableLabel2(unsigned uid, unsigned uid2) const {
  SmallString<60> Name;
  raw_svector_ostream(Name) << MAI->getPrivateGlobalPrefix() << "JTI"
    << getFunctionNumber() << '_' << uid << '_' << uid2;
  return OutContext.GetOrCreateSymbol(Name.str());
}


MCSymbol *ARMAsmPrinter::GetARMSJLJEHLabel(void) const {
  SmallString<60> Name;
  raw_svector_ostream(Name) << MAI->getPrivateGlobalPrefix() << "SJLJEH"
    << getFunctionNumber();
  return OutContext.GetOrCreateSymbol(Name.str());
}

bool ARMAsmPrinter::PrintAsmOperand(const MachineInstr *MI, unsigned OpNum,
                                    unsigned AsmVariant, const char *ExtraCode,
                                    raw_ostream &O) {
  // Does this asm operand have a single letter operand modifier?
  if (ExtraCode && ExtraCode[0]) {
    if (ExtraCode[1] != 0) return true; // Unknown modifier.

    switch (ExtraCode[0]) {
    default: return true;  // Unknown modifier.
    case 'a': // Print as a memory address.
      if (MI->getOperand(OpNum).isReg()) {
        O << "["
          << ARMInstPrinter::getRegisterName(MI->getOperand(OpNum).getReg())
          << "]";
        return false;
      }
      // Fallthrough
    case 'c': // Don't print "#" before an immediate operand.
      if (!MI->getOperand(OpNum).isImm())
        return true;
      O << MI->getOperand(OpNum).getImm();
      return false;
    case 'P': // Print a VFP double precision register.
    case 'q': // Print a NEON quad precision register.
      printOperand(MI, OpNum, O);
      return false;
    case 'Q':
    case 'R':
    case 'H':
      report_fatal_error("llvm does not support 'Q', 'R', and 'H' modifiers!");
      return true;
    }
  }

  printOperand(MI, OpNum, O);
  return false;
}

bool ARMAsmPrinter::PrintAsmMemoryOperand(const MachineInstr *MI,
                                          unsigned OpNum, unsigned AsmVariant,
                                          const char *ExtraCode,
                                          raw_ostream &O) {
  if (ExtraCode && ExtraCode[0])
    return true; // Unknown modifier.

  const MachineOperand &MO = MI->getOperand(OpNum);
  assert(MO.isReg() && "unexpected inline asm memory operand");
  O << "[" << ARMInstPrinter::getRegisterName(MO.getReg()) << "]";
  return false;
}

void ARMAsmPrinter::EmitStartOfAsmFile(Module &M) {
  if (Subtarget->isTargetDarwin()) {
    Reloc::Model RelocM = TM.getRelocationModel();
    if (RelocM == Reloc::PIC_ || RelocM == Reloc::DynamicNoPIC) {
      // Declare all the text sections up front (before the DWARF sections
      // emitted by AsmPrinter::doInitialization) so the assembler will keep
      // them together at the beginning of the object file.  This helps
      // avoid out-of-range branches that are due a fundamental limitation of
      // the way symbol offsets are encoded with the current Darwin ARM
      // relocations.
      const TargetLoweringObjectFileMachO &TLOFMacho =
        static_cast<const TargetLoweringObjectFileMachO &>(
          getObjFileLowering());
      OutStreamer.SwitchSection(TLOFMacho.getTextSection());
      OutStreamer.SwitchSection(TLOFMacho.getTextCoalSection());
      OutStreamer.SwitchSection(TLOFMacho.getConstTextCoalSection());
      if (RelocM == Reloc::DynamicNoPIC) {
        const MCSection *sect =
          OutContext.getMachOSection("__TEXT", "__symbol_stub4",
                                     MCSectionMachO::S_SYMBOL_STUBS,
                                     12, SectionKind::getText());
        OutStreamer.SwitchSection(sect);
      } else {
        const MCSection *sect =
          OutContext.getMachOSection("__TEXT", "__picsymbolstub4",
                                     MCSectionMachO::S_SYMBOL_STUBS,
                                     16, SectionKind::getText());
        OutStreamer.SwitchSection(sect);
      }
      const MCSection *StaticInitSect =
        OutContext.getMachOSection("__TEXT", "__StaticInit",
                                   MCSectionMachO::S_REGULAR |
                                   MCSectionMachO::S_ATTR_PURE_INSTRUCTIONS,
                                   SectionKind::getText());
      OutStreamer.SwitchSection(StaticInitSect);
    }
  }

  // Use unified assembler syntax.
  OutStreamer.EmitAssemblerFlag(MCAF_SyntaxUnified);

  // Emit ARM Build Attributes
  if (Subtarget->isTargetELF()) {

    emitAttributes();
  }
}


void ARMAsmPrinter::EmitEndOfAsmFile(Module &M) {
  if (Subtarget->isTargetDarwin()) {
    // All darwin targets use mach-o.
    const TargetLoweringObjectFileMachO &TLOFMacho =
      static_cast<const TargetLoweringObjectFileMachO &>(getObjFileLowering());
    MachineModuleInfoMachO &MMIMacho =
      MMI->getObjFileInfo<MachineModuleInfoMachO>();

    // Output non-lazy-pointers for external and common global variables.
    MachineModuleInfoMachO::SymbolListTy Stubs = MMIMacho.GetGVStubList();

    if (!Stubs.empty()) {
      // Switch with ".non_lazy_symbol_pointer" directive.
      OutStreamer.SwitchSection(TLOFMacho.getNonLazySymbolPointerSection());
      EmitAlignment(2);
      for (unsigned i = 0, e = Stubs.size(); i != e; ++i) {
        // L_foo$stub:
        OutStreamer.EmitLabel(Stubs[i].first);
        //   .indirect_symbol _foo
        MachineModuleInfoImpl::StubValueTy &MCSym = Stubs[i].second;
        OutStreamer.EmitSymbolAttribute(MCSym.getPointer(),MCSA_IndirectSymbol);

        if (MCSym.getInt())
          // External to current translation unit.
          OutStreamer.EmitIntValue(0, 4/*size*/, 0/*addrspace*/);
        else
          // Internal to current translation unit.
          //
          // When we place the LSDA into the TEXT section, the type info
          // pointers need to be indirect and pc-rel. We accomplish this by
          // using NLPs; however, sometimes the types are local to the file.
          // We need to fill in the value for the NLP in those cases.
          OutStreamer.EmitValue(MCSymbolRefExpr::Create(MCSym.getPointer(),
                                                        OutContext),
                                4/*size*/, 0/*addrspace*/);
      }

      Stubs.clear();
      OutStreamer.AddBlankLine();
    }

    Stubs = MMIMacho.GetHiddenGVStubList();
    if (!Stubs.empty()) {
      OutStreamer.SwitchSection(getObjFileLowering().getDataSection());
      EmitAlignment(2);
      for (unsigned i = 0, e = Stubs.size(); i != e; ++i) {
        // L_foo$stub:
        OutStreamer.EmitLabel(Stubs[i].first);
        //   .long _foo
        OutStreamer.EmitValue(MCSymbolRefExpr::
                              Create(Stubs[i].second.getPointer(),
                                     OutContext),
                              4/*size*/, 0/*addrspace*/);
      }

      Stubs.clear();
      OutStreamer.AddBlankLine();
    }

    // Funny Darwin hack: This flag tells the linker that no global symbols
    // contain code that falls through to other global symbols (e.g. the obvious
    // implementation of multiple entry points).  If this doesn't occur, the
    // linker can safely perform dead code stripping.  Since LLVM never
    // generates code that does this, it is always safe to set.
    OutStreamer.EmitAssemblerFlag(MCAF_SubsectionsViaSymbols);
  }
}

//===----------------------------------------------------------------------===//
// Helper routines for EmitStartOfAsmFile() and EmitEndOfAsmFile()
// FIXME:
// The following seem like one-off assembler flags, but they actually need
// to appear in the .ARM.attributes section in ELF.
// Instead of subclassing the MCELFStreamer, we do the work here.

void ARMAsmPrinter::emitAttributes() {

  emitARMAttributeSection();

  AttributeEmitter *AttrEmitter;
  if (OutStreamer.hasRawTextSupport())
    AttrEmitter = new AsmAttributeEmitter(OutStreamer);
  else {
    MCObjectStreamer &O = static_cast<MCObjectStreamer&>(OutStreamer);
    AttrEmitter = new ObjectAttributeEmitter(O);
  }

  AttrEmitter->MaybeSwitchVendor("aeabi");

  std::string CPUString = Subtarget->getCPUString();
  if (OutStreamer.hasRawTextSupport()) {
    if (CPUString != "generic")
      OutStreamer.EmitRawText(StringRef("\t.cpu ") + CPUString);
  } else {
    assert(CPUString == "generic" && "Unsupported .cpu attribute for ELF/.o");
    // FIXME: Why these defaults?
    AttrEmitter->EmitAttribute(ARMBuildAttrs::CPU_arch, ARMBuildAttrs::v4T);
    AttrEmitter->EmitAttribute(ARMBuildAttrs::ARM_ISA_use, 1);
    AttrEmitter->EmitAttribute(ARMBuildAttrs::THUMB_ISA_use, 1);
  }

  // FIXME: Emit FPU type
  if (Subtarget->hasVFP2())
    AttrEmitter->EmitAttribute(ARMBuildAttrs::VFP_arch, 2);

  // Signal various FP modes.
  if (!UnsafeFPMath) {
    AttrEmitter->EmitAttribute(ARMBuildAttrs::ABI_FP_denormal, 1);
    AttrEmitter->EmitAttribute(ARMBuildAttrs::ABI_FP_exceptions, 1);
  }

  if (NoInfsFPMath && NoNaNsFPMath)
    AttrEmitter->EmitAttribute(ARMBuildAttrs::ABI_FP_number_model, 1);
  else
    AttrEmitter->EmitAttribute(ARMBuildAttrs::ABI_FP_number_model, 3);

  // 8-bytes alignment stuff.
  AttrEmitter->EmitAttribute(ARMBuildAttrs::ABI_align8_needed, 1);
  AttrEmitter->EmitAttribute(ARMBuildAttrs::ABI_align8_preserved, 1);

  // Hard float.  Use both S and D registers and conform to AAPCS-VFP.
  if (Subtarget->isAAPCS_ABI() && FloatABIType == FloatABI::Hard) {
    AttrEmitter->EmitAttribute(ARMBuildAttrs::ABI_HardFP_use, 3);
    AttrEmitter->EmitAttribute(ARMBuildAttrs::ABI_VFP_args, 1);
  }
  // FIXME: Should we signal R9 usage?

  AttrEmitter->EmitAttribute(ARMBuildAttrs::DIV_use, 1);

  AttrEmitter->Finish();
  delete AttrEmitter;
}

void ARMAsmPrinter::emitARMAttributeSection() {
  // <format-version>
  // [ <section-length> "vendor-name"
  // [ <file-tag> <size> <attribute>*
  //   | <section-tag> <size> <section-number>* 0 <attribute>*
  //   | <symbol-tag> <size> <symbol-number>* 0 <attribute>*
  //   ]+
  // ]*

  if (OutStreamer.hasRawTextSupport())
    return;

  const ARMElfTargetObjectFile &TLOFELF =
    static_cast<const ARMElfTargetObjectFile &>
    (getObjFileLowering());

  OutStreamer.SwitchSection(TLOFELF.getAttributesSection());

  // Format version
  OutStreamer.EmitIntValue(0x41, 1);
}

//===----------------------------------------------------------------------===//

static MCSymbol *getPICLabel(const char *Prefix, unsigned FunctionNumber,
                             unsigned LabelId, MCContext &Ctx) {

  MCSymbol *Label = Ctx.GetOrCreateSymbol(Twine(Prefix)
                       + "PC" + Twine(FunctionNumber) + "_" + Twine(LabelId));
  return Label;
}

static MCSymbolRefExpr::VariantKind
getModifierVariantKind(ARMCP::ARMCPModifier Modifier) {
  switch (Modifier) {
  default: llvm_unreachable("Unknown modifier!");
  case ARMCP::no_modifier: return MCSymbolRefExpr::VK_None;
  case ARMCP::TLSGD:       return MCSymbolRefExpr::VK_ARM_TLSGD;
  case ARMCP::TPOFF:       return MCSymbolRefExpr::VK_ARM_TPOFF;
  case ARMCP::GOTTPOFF:    return MCSymbolRefExpr::VK_ARM_GOTTPOFF;
  case ARMCP::GOT:         return MCSymbolRefExpr::VK_ARM_GOT;
  case ARMCP::GOTOFF:      return MCSymbolRefExpr::VK_ARM_GOTOFF;
  }
  return MCSymbolRefExpr::VK_None;
}

void ARMAsmPrinter::
EmitMachineConstantPoolValue(MachineConstantPoolValue *MCPV) {
  int Size = TM.getTargetData()->getTypeAllocSize(MCPV->getType());

  ARMConstantPoolValue *ACPV = static_cast<ARMConstantPoolValue*>(MCPV);

  MCSymbol *MCSym;
  if (ACPV->isLSDA()) {
    SmallString<128> Str;
    raw_svector_ostream OS(Str);
    OS << MAI->getPrivateGlobalPrefix() << "_LSDA_" << getFunctionNumber();
    MCSym = OutContext.GetOrCreateSymbol(OS.str());
  } else if (ACPV->isBlockAddress()) {
    MCSym = GetBlockAddressSymbol(ACPV->getBlockAddress());
  } else if (ACPV->isGlobalValue()) {
    const GlobalValue *GV = ACPV->getGV();
    bool isIndirect = Subtarget->isTargetDarwin() &&
      Subtarget->GVIsIndirectSymbol(GV, TM.getRelocationModel());
    if (!isIndirect)
      MCSym = Mang->getSymbol(GV);
    else {
      // FIXME: Remove this when Darwin transition to @GOT like syntax.
      MCSym = GetSymbolWithGlobalValueBase(GV, "$non_lazy_ptr");

      MachineModuleInfoMachO &MMIMachO =
        MMI->getObjFileInfo<MachineModuleInfoMachO>();
      MachineModuleInfoImpl::StubValueTy &StubSym =
        GV->hasHiddenVisibility() ? MMIMachO.getHiddenGVStubEntry(MCSym) :
        MMIMachO.getGVStubEntry(MCSym);
      if (StubSym.getPointer() == 0)
        StubSym = MachineModuleInfoImpl::
          StubValueTy(Mang->getSymbol(GV), !GV->hasInternalLinkage());
    }
  } else {
    assert(ACPV->isExtSymbol() && "unrecognized constant pool value");
    MCSym = GetExternalSymbolSymbol(ACPV->getSymbol());
  }

  // Create an MCSymbol for the reference.
  const MCExpr *Expr =
    MCSymbolRefExpr::Create(MCSym, getModifierVariantKind(ACPV->getModifier()),
                            OutContext);

  if (ACPV->getPCAdjustment()) {
    MCSymbol *PCLabel = getPICLabel(MAI->getPrivateGlobalPrefix(),
                                    getFunctionNumber(),
                                    ACPV->getLabelId(),
                                    OutContext);
    const MCExpr *PCRelExpr = MCSymbolRefExpr::Create(PCLabel, OutContext);
    PCRelExpr =
      MCBinaryExpr::CreateAdd(PCRelExpr,
                              MCConstantExpr::Create(ACPV->getPCAdjustment(),
                                                     OutContext),
                              OutContext);
    if (ACPV->mustAddCurrentAddress()) {
      // We want "(<expr> - .)", but MC doesn't have a concept of the '.'
      // label, so just emit a local label end reference that instead.
      MCSymbol *DotSym = OutContext.CreateTempSymbol();
      OutStreamer.EmitLabel(DotSym);
      const MCExpr *DotExpr = MCSymbolRefExpr::Create(DotSym, OutContext);
      PCRelExpr = MCBinaryExpr::CreateSub(PCRelExpr, DotExpr, OutContext);
    }
    Expr = MCBinaryExpr::CreateSub(Expr, PCRelExpr, OutContext);
  }
  OutStreamer.EmitValue(Expr, Size);
}

void ARMAsmPrinter::EmitJumpTable(const MachineInstr *MI) {
  unsigned Opcode = MI->getOpcode();
  int OpNum = 1;
  if (Opcode == ARM::BR_JTadd)
    OpNum = 2;
  else if (Opcode == ARM::BR_JTm)
    OpNum = 3;

  const MachineOperand &MO1 = MI->getOperand(OpNum);
  const MachineOperand &MO2 = MI->getOperand(OpNum+1); // Unique Id
  unsigned JTI = MO1.getIndex();

  // Emit a label for the jump table.
  MCSymbol *JTISymbol = GetARMJTIPICJumpTableLabel2(JTI, MO2.getImm());
  OutStreamer.EmitLabel(JTISymbol);

  // Emit each entry of the table.
  const MachineJumpTableInfo *MJTI = MF->getJumpTableInfo();
  const std::vector<MachineJumpTableEntry> &JT = MJTI->getJumpTables();
  const std::vector<MachineBasicBlock*> &JTBBs = JT[JTI].MBBs;

  for (unsigned i = 0, e = JTBBs.size(); i != e; ++i) {
    MachineBasicBlock *MBB = JTBBs[i];
    // Construct an MCExpr for the entry. We want a value of the form:
    // (BasicBlockAddr - TableBeginAddr)
    //
    // For example, a table with entries jumping to basic blocks BB0 and BB1
    // would look like:
    // LJTI_0_0:
    //    .word (LBB0 - LJTI_0_0)
    //    .word (LBB1 - LJTI_0_0)
    const MCExpr *Expr = MCSymbolRefExpr::Create(MBB->getSymbol(), OutContext);

    if (TM.getRelocationModel() == Reloc::PIC_)
      Expr = MCBinaryExpr::CreateSub(Expr, MCSymbolRefExpr::Create(JTISymbol,
                                                                   OutContext),
                                     OutContext);
    OutStreamer.EmitValue(Expr, 4);
  }
}

void ARMAsmPrinter::EmitJump2Table(const MachineInstr *MI) {
  unsigned Opcode = MI->getOpcode();
  int OpNum = (Opcode == ARM::t2BR_JT) ? 2 : 1;
  const MachineOperand &MO1 = MI->getOperand(OpNum);
  const MachineOperand &MO2 = MI->getOperand(OpNum+1); // Unique Id
  unsigned JTI = MO1.getIndex();

  // Emit a label for the jump table.
  MCSymbol *JTISymbol = GetARMJTIPICJumpTableLabel2(JTI, MO2.getImm());
  OutStreamer.EmitLabel(JTISymbol);

  // Emit each entry of the table.
  const MachineJumpTableInfo *MJTI = MF->getJumpTableInfo();
  const std::vector<MachineJumpTableEntry> &JT = MJTI->getJumpTables();
  const std::vector<MachineBasicBlock*> &JTBBs = JT[JTI].MBBs;
  unsigned OffsetWidth = 4;
  if (MI->getOpcode() == ARM::t2TBB_JT)
    OffsetWidth = 1;
  else if (MI->getOpcode() == ARM::t2TBH_JT)
    OffsetWidth = 2;

  for (unsigned i = 0, e = JTBBs.size(); i != e; ++i) {
    MachineBasicBlock *MBB = JTBBs[i];
    const MCExpr *MBBSymbolExpr = MCSymbolRefExpr::Create(MBB->getSymbol(),
                                                      OutContext);
    // If this isn't a TBB or TBH, the entries are direct branch instructions.
    if (OffsetWidth == 4) {
      MCInst BrInst;
      BrInst.setOpcode(ARM::t2B);
      BrInst.addOperand(MCOperand::CreateExpr(MBBSymbolExpr));
      OutStreamer.EmitInstruction(BrInst);
      continue;
    }
    // Otherwise it's an offset from the dispatch instruction. Construct an
    // MCExpr for the entry. We want a value of the form:
    // (BasicBlockAddr - TableBeginAddr) / 2
    //
    // For example, a TBB table with entries jumping to basic blocks BB0 and BB1
    // would look like:
    // LJTI_0_0:
    //    .byte (LBB0 - LJTI_0_0) / 2
    //    .byte (LBB1 - LJTI_0_0) / 2
    const MCExpr *Expr =
      MCBinaryExpr::CreateSub(MBBSymbolExpr,
                              MCSymbolRefExpr::Create(JTISymbol, OutContext),
                              OutContext);
    Expr = MCBinaryExpr::CreateDiv(Expr, MCConstantExpr::Create(2, OutContext),
                                   OutContext);
    OutStreamer.EmitValue(Expr, OffsetWidth);
  }
}

void ARMAsmPrinter::PrintDebugValueComment(const MachineInstr *MI,
                                           raw_ostream &OS) {
  unsigned NOps = MI->getNumOperands();
  assert(NOps==4);
  OS << '\t' << MAI->getCommentString() << "DEBUG_VALUE: ";
  // cast away const; DIetc do not take const operands for some reason.
  DIVariable V(const_cast<MDNode *>(MI->getOperand(NOps-1).getMetadata()));
  OS << V.getName();
  OS << " <- ";
  // Frame address.  Currently handles register +- offset only.
  assert(MI->getOperand(0).isReg() && MI->getOperand(1).isImm());
  OS << '['; printOperand(MI, 0, OS); OS << '+'; printOperand(MI, 1, OS);
  OS << ']';
  OS << "+";
  printOperand(MI, NOps-2, OS);
}

static void populateADROperands(MCInst &Inst, unsigned Dest,
                                const MCSymbol *Label,
                                unsigned pred, unsigned ccreg,
                                MCContext &Ctx) {
  const MCExpr *SymbolExpr = MCSymbolRefExpr::Create(Label, Ctx);
  Inst.addOperand(MCOperand::CreateReg(Dest));
  Inst.addOperand(MCOperand::CreateExpr(SymbolExpr));
  // Add predicate operands.
  Inst.addOperand(MCOperand::CreateImm(pred));
  Inst.addOperand(MCOperand::CreateReg(ccreg));
}

void ARMAsmPrinter::EmitInstruction(const MachineInstr *MI) {
  switch (MI->getOpcode()) {
  default: break;
  case ARM::t2ADDrSPi:
  case ARM::t2ADDrSPi12:
  case ARM::t2SUBrSPi:
  case ARM::t2SUBrSPi12:
    assert ((MI->getOperand(1).getReg() == ARM::SP) &&
            "Unexpected source register!");
    break;

  case ARM::t2MOVi32imm: assert(0 && "Should be lowered by thumb2it pass");
  case ARM::DBG_VALUE: {
    if (isVerbose() && OutStreamer.hasRawTextSupport()) {
      SmallString<128> TmpStr;
      raw_svector_ostream OS(TmpStr);
      PrintDebugValueComment(MI, OS);
      OutStreamer.EmitRawText(StringRef(OS.str()));
    }
    return;
  }
  case ARM::LEApcrel:
  case ARM::t2LEApcrel: {
    // FIXME: Need to also handle globals and externals
    MCInst TmpInst;
    TmpInst.setOpcode(MI->getOpcode() == ARM::t2LEApcrel
                      ? ARM::t2ADR : ARM::ADR);
    populateADROperands(TmpInst, MI->getOperand(0).getReg(),
                        GetCPISymbol(MI->getOperand(1).getIndex()),
                        MI->getOperand(2).getImm(), MI->getOperand(3).getReg(),
                        OutContext);
    OutStreamer.EmitInstruction(TmpInst);
    return;
  }
  case ARM::t2LEApcrelJT:
  case ARM::LEApcrelJT: {
    MCInst TmpInst;
    TmpInst.setOpcode(MI->getOpcode() == ARM::t2LEApcrelJT
                      ? ARM::t2ADR : ARM::ADR);
    populateADROperands(TmpInst, MI->getOperand(0).getReg(),
                      GetARMJTIPICJumpTableLabel2(MI->getOperand(1).getIndex(),
                                                  MI->getOperand(2).getImm()),
                      MI->getOperand(3).getImm(), MI->getOperand(4).getReg(),
                      OutContext);
    OutStreamer.EmitInstruction(TmpInst);
    return;
  }
  case ARM::MOVPCRX: {
    MCInst TmpInst;
    TmpInst.setOpcode(ARM::MOVr);
    TmpInst.addOperand(MCOperand::CreateReg(ARM::PC));
    TmpInst.addOperand(MCOperand::CreateReg(MI->getOperand(0).getReg()));
    // Add predicate operands.
    TmpInst.addOperand(MCOperand::CreateImm(ARMCC::AL));
    TmpInst.addOperand(MCOperand::CreateReg(0));
    // Add 's' bit operand (always reg0 for this)
    TmpInst.addOperand(MCOperand::CreateReg(0));
    OutStreamer.EmitInstruction(TmpInst);
    return;
  }
  case ARM::BXr9_CALL:
  case ARM::BX_CALL: {
    {
      MCInst TmpInst;
      TmpInst.setOpcode(ARM::MOVr);
      TmpInst.addOperand(MCOperand::CreateReg(ARM::LR));
      TmpInst.addOperand(MCOperand::CreateReg(ARM::PC));
      // Add predicate operands.
      TmpInst.addOperand(MCOperand::CreateImm(ARMCC::AL));
      TmpInst.addOperand(MCOperand::CreateReg(0));
      // Add 's' bit operand (always reg0 for this)
      TmpInst.addOperand(MCOperand::CreateReg(0));
      OutStreamer.EmitInstruction(TmpInst);
    }
    {
      MCInst TmpInst;
      TmpInst.setOpcode(ARM::BX);
      TmpInst.addOperand(MCOperand::CreateReg(MI->getOperand(0).getReg()));
      OutStreamer.EmitInstruction(TmpInst);
    }
    return;
  }
  case ARM::BMOVPCRXr9_CALL:
  case ARM::BMOVPCRX_CALL: {
    {
      MCInst TmpInst;
      TmpInst.setOpcode(ARM::MOVr);
      TmpInst.addOperand(MCOperand::CreateReg(ARM::LR));
      TmpInst.addOperand(MCOperand::CreateReg(ARM::PC));
      // Add predicate operands.
      TmpInst.addOperand(MCOperand::CreateImm(ARMCC::AL));
      TmpInst.addOperand(MCOperand::CreateReg(0));
      // Add 's' bit operand (always reg0 for this)
      TmpInst.addOperand(MCOperand::CreateReg(0));
      OutStreamer.EmitInstruction(TmpInst);
    }
    {
      MCInst TmpInst;
      TmpInst.setOpcode(ARM::MOVr);
      TmpInst.addOperand(MCOperand::CreateReg(ARM::PC));
      TmpInst.addOperand(MCOperand::CreateReg(MI->getOperand(0).getReg()));
      // Add predicate operands.
      TmpInst.addOperand(MCOperand::CreateImm(ARMCC::AL));
      TmpInst.addOperand(MCOperand::CreateReg(0));
      // Add 's' bit operand (always reg0 for this)
      TmpInst.addOperand(MCOperand::CreateReg(0));
      OutStreamer.EmitInstruction(TmpInst);
    }
    return;
  }
  case ARM::tPICADD: {
    // This is a pseudo op for a label + instruction sequence, which looks like:
    // LPC0:
    //     add r0, pc
    // This adds the address of LPC0 to r0.

    // Emit the label.
    OutStreamer.EmitLabel(getPICLabel(MAI->getPrivateGlobalPrefix(),
                          getFunctionNumber(), MI->getOperand(2).getImm(),
                          OutContext));

    // Form and emit the add.
    MCInst AddInst;
    AddInst.setOpcode(ARM::tADDhirr);
    AddInst.addOperand(MCOperand::CreateReg(MI->getOperand(0).getReg()));
    AddInst.addOperand(MCOperand::CreateReg(MI->getOperand(0).getReg()));
    AddInst.addOperand(MCOperand::CreateReg(ARM::PC));
    // Add predicate operands.
    AddInst.addOperand(MCOperand::CreateImm(ARMCC::AL));
    AddInst.addOperand(MCOperand::CreateReg(0));
    OutStreamer.EmitInstruction(AddInst);
    return;
  }
  case ARM::PICADD: {
    // This is a pseudo op for a label + instruction sequence, which looks like:
    // LPC0:
    //     add r0, pc, r0
    // This adds the address of LPC0 to r0.

    // Emit the label.
    OutStreamer.EmitLabel(getPICLabel(MAI->getPrivateGlobalPrefix(),
                          getFunctionNumber(), MI->getOperand(2).getImm(),
                          OutContext));

    // Form and emit the add.
    MCInst AddInst;
    AddInst.setOpcode(ARM::ADDrr);
    AddInst.addOperand(MCOperand::CreateReg(MI->getOperand(0).getReg()));
    AddInst.addOperand(MCOperand::CreateReg(ARM::PC));
    AddInst.addOperand(MCOperand::CreateReg(MI->getOperand(1).getReg()));
    // Add predicate operands.
    AddInst.addOperand(MCOperand::CreateImm(MI->getOperand(3).getImm()));
    AddInst.addOperand(MCOperand::CreateReg(MI->getOperand(4).getReg()));
    // Add 's' bit operand (always reg0 for this)
    AddInst.addOperand(MCOperand::CreateReg(0));
    OutStreamer.EmitInstruction(AddInst);
    return;
  }
  case ARM::PICSTR:
  case ARM::PICSTRB:
  case ARM::PICSTRH:
  case ARM::PICLDR:
  case ARM::PICLDRB:
  case ARM::PICLDRH:
  case ARM::PICLDRSB:
  case ARM::PICLDRSH: {
    // This is a pseudo op for a label + instruction sequence, which looks like:
    // LPC0:
    //     OP r0, [pc, r0]
    // The LCP0 label is referenced by a constant pool entry in order to get
    // a PC-relative address at the ldr instruction.

    // Emit the label.
    OutStreamer.EmitLabel(getPICLabel(MAI->getPrivateGlobalPrefix(),
                          getFunctionNumber(), MI->getOperand(2).getImm(),
                          OutContext));

    // Form and emit the load
    unsigned Opcode;
    switch (MI->getOpcode()) {
    default:
      llvm_unreachable("Unexpected opcode!");
    case ARM::PICSTR:   Opcode = ARM::STRrs; break;
    case ARM::PICSTRB:  Opcode = ARM::STRBrs; break;
    case ARM::PICSTRH:  Opcode = ARM::STRH; break;
    case ARM::PICLDR:   Opcode = ARM::LDRrs; break;
    case ARM::PICLDRB:  Opcode = ARM::LDRBrs; break;
    case ARM::PICLDRH:  Opcode = ARM::LDRH; break;
    case ARM::PICLDRSB: Opcode = ARM::LDRSB; break;
    case ARM::PICLDRSH: Opcode = ARM::LDRSH; break;
    }
    MCInst LdStInst;
    LdStInst.setOpcode(Opcode);
    LdStInst.addOperand(MCOperand::CreateReg(MI->getOperand(0).getReg()));
    LdStInst.addOperand(MCOperand::CreateReg(ARM::PC));
    LdStInst.addOperand(MCOperand::CreateReg(MI->getOperand(1).getReg()));
    LdStInst.addOperand(MCOperand::CreateImm(0));
    // Add predicate operands.
    LdStInst.addOperand(MCOperand::CreateImm(MI->getOperand(3).getImm()));
    LdStInst.addOperand(MCOperand::CreateReg(MI->getOperand(4).getReg()));
    OutStreamer.EmitInstruction(LdStInst);

    return;
  }
  case ARM::CONSTPOOL_ENTRY: {
    /// CONSTPOOL_ENTRY - This instruction represents a floating constant pool
    /// in the function.  The first operand is the ID# for this instruction, the
    /// second is the index into the MachineConstantPool that this is, the third
    /// is the size in bytes of this constant pool entry.
    unsigned LabelId = (unsigned)MI->getOperand(0).getImm();
    unsigned CPIdx   = (unsigned)MI->getOperand(1).getIndex();

    EmitAlignment(2);
    OutStreamer.EmitLabel(GetCPISymbol(LabelId));

    const MachineConstantPoolEntry &MCPE = MCP->getConstants()[CPIdx];
    if (MCPE.isMachineConstantPoolEntry())
      EmitMachineConstantPoolValue(MCPE.Val.MachineCPVal);
    else
      EmitGlobalConstant(MCPE.Val.ConstVal);

    return;
  }
  case ARM::t2BR_JT: {
    // Lower and emit the instruction itself, then the jump table following it.
    MCInst TmpInst;
    TmpInst.setOpcode(ARM::tMOVgpr2gpr);
    TmpInst.addOperand(MCOperand::CreateReg(ARM::PC));
    TmpInst.addOperand(MCOperand::CreateReg(MI->getOperand(0).getReg()));
    // Add predicate operands.
    TmpInst.addOperand(MCOperand::CreateImm(ARMCC::AL));
    TmpInst.addOperand(MCOperand::CreateReg(0));
    OutStreamer.EmitInstruction(TmpInst);
    // Output the data for the jump table itself
    EmitJump2Table(MI);
    return;
  }
  case ARM::t2TBB_JT: {
    // Lower and emit the instruction itself, then the jump table following it.
    MCInst TmpInst;

    TmpInst.setOpcode(ARM::t2TBB);
    TmpInst.addOperand(MCOperand::CreateReg(ARM::PC));
    TmpInst.addOperand(MCOperand::CreateReg(MI->getOperand(0).getReg()));
    // Add predicate operands.
    TmpInst.addOperand(MCOperand::CreateImm(ARMCC::AL));
    TmpInst.addOperand(MCOperand::CreateReg(0));
    OutStreamer.EmitInstruction(TmpInst);
    // Output the data for the jump table itself
    EmitJump2Table(MI);
    // Make sure the next instruction is 2-byte aligned.
    EmitAlignment(1);
    return;
  }
  case ARM::t2TBH_JT: {
    // Lower and emit the instruction itself, then the jump table following it.
    MCInst TmpInst;

    TmpInst.setOpcode(ARM::t2TBH);
    TmpInst.addOperand(MCOperand::CreateReg(ARM::PC));
    TmpInst.addOperand(MCOperand::CreateReg(MI->getOperand(0).getReg()));
    // Add predicate operands.
    TmpInst.addOperand(MCOperand::CreateImm(ARMCC::AL));
    TmpInst.addOperand(MCOperand::CreateReg(0));
    OutStreamer.EmitInstruction(TmpInst);
    // Output the data for the jump table itself
    EmitJump2Table(MI);
    return;
  }
  case ARM::tBR_JTr:
  case ARM::BR_JTr: {
    // Lower and emit the instruction itself, then the jump table following it.
    // mov pc, target
    MCInst TmpInst;
    unsigned Opc = MI->getOpcode() == ARM::BR_JTr ?
      ARM::MOVr : ARM::tMOVgpr2gpr;
    TmpInst.setOpcode(Opc);
    TmpInst.addOperand(MCOperand::CreateReg(ARM::PC));
    TmpInst.addOperand(MCOperand::CreateReg(MI->getOperand(0).getReg()));
    // Add predicate operands.
    TmpInst.addOperand(MCOperand::CreateImm(ARMCC::AL));
    TmpInst.addOperand(MCOperand::CreateReg(0));
    // Add 's' bit operand (always reg0 for this)
    if (Opc == ARM::MOVr)
      TmpInst.addOperand(MCOperand::CreateReg(0));
    OutStreamer.EmitInstruction(TmpInst);

    // Make sure the Thumb jump table is 4-byte aligned.
    if (Opc == ARM::tMOVr)
      EmitAlignment(2);

    // Output the data for the jump table itself
    EmitJumpTable(MI);
    return;
  }
  case ARM::BR_JTm: {
    // Lower and emit the instruction itself, then the jump table following it.
    // ldr pc, target
    MCInst TmpInst;
    if (MI->getOperand(1).getReg() == 0) {
      // literal offset
      TmpInst.setOpcode(ARM::LDRi12);
      TmpInst.addOperand(MCOperand::CreateReg(ARM::PC));
      TmpInst.addOperand(MCOperand::CreateReg(MI->getOperand(0).getReg()));
      TmpInst.addOperand(MCOperand::CreateImm(MI->getOperand(2).getImm()));
    } else {
      TmpInst.setOpcode(ARM::LDRrs);
      TmpInst.addOperand(MCOperand::CreateReg(ARM::PC));
      TmpInst.addOperand(MCOperand::CreateReg(MI->getOperand(0).getReg()));
      TmpInst.addOperand(MCOperand::CreateReg(MI->getOperand(1).getReg()));
      TmpInst.addOperand(MCOperand::CreateImm(0));
    }
    // Add predicate operands.
    TmpInst.addOperand(MCOperand::CreateImm(ARMCC::AL));
    TmpInst.addOperand(MCOperand::CreateReg(0));
    OutStreamer.EmitInstruction(TmpInst);

    // Output the data for the jump table itself
    EmitJumpTable(MI);
    return;
  }
  case ARM::BR_JTadd: {
    // Lower and emit the instruction itself, then the jump table following it.
    // add pc, target, idx
    MCInst TmpInst;
    TmpInst.setOpcode(ARM::ADDrr);
    TmpInst.addOperand(MCOperand::CreateReg(ARM::PC));
    TmpInst.addOperand(MCOperand::CreateReg(MI->getOperand(0).getReg()));
    TmpInst.addOperand(MCOperand::CreateReg(MI->getOperand(1).getReg()));
    // Add predicate operands.
    TmpInst.addOperand(MCOperand::CreateImm(ARMCC::AL));
    TmpInst.addOperand(MCOperand::CreateReg(0));
    // Add 's' bit operand (always reg0 for this)
    TmpInst.addOperand(MCOperand::CreateReg(0));
    OutStreamer.EmitInstruction(TmpInst);

    // Output the data for the jump table itself
    EmitJumpTable(MI);
    return;
  }
  case ARM::TRAP: {
    // Non-Darwin binutils don't yet support the "trap" mnemonic.
    // FIXME: Remove this special case when they do.
    if (!Subtarget->isTargetDarwin()) {
      //.long 0xe7ffdefe @ trap
      uint32_t Val = 0xe7ffdefeUL;
      OutStreamer.AddComment("trap");
      OutStreamer.EmitIntValue(Val, 4);
      return;
    }
    break;
  }
  case ARM::tTRAP: {
    // Non-Darwin binutils don't yet support the "trap" mnemonic.
    // FIXME: Remove this special case when they do.
    if (!Subtarget->isTargetDarwin()) {
      //.short 57086 @ trap
      uint16_t Val = 0xdefe;
      OutStreamer.AddComment("trap");
      OutStreamer.EmitIntValue(Val, 2);
      return;
    }
    break;
  }
  case ARM::t2Int_eh_sjlj_setjmp:
  case ARM::t2Int_eh_sjlj_setjmp_nofp:
  case ARM::tInt_eh_sjlj_setjmp: {
    // Two incoming args: GPR:$src, GPR:$val
    // mov $val, pc
    // adds $val, #7
    // str $val, [$src, #4]
    // movs r0, #0
    // b 1f
    // movs r0, #1
    // 1:
    unsigned SrcReg = MI->getOperand(0).getReg();
    unsigned ValReg = MI->getOperand(1).getReg();
    MCSymbol *Label = GetARMSJLJEHLabel();
    {
      MCInst TmpInst;
      TmpInst.setOpcode(ARM::tMOVgpr2tgpr);
      TmpInst.addOperand(MCOperand::CreateReg(ValReg));
      TmpInst.addOperand(MCOperand::CreateReg(ARM::PC));
      // 's' bit operand
      TmpInst.addOperand(MCOperand::CreateReg(ARM::CPSR));
      OutStreamer.AddComment("eh_setjmp begin");
      OutStreamer.EmitInstruction(TmpInst);
    }
    {
      MCInst TmpInst;
      TmpInst.setOpcode(ARM::tADDi3);
      TmpInst.addOperand(MCOperand::CreateReg(ValReg));
      // 's' bit operand
      TmpInst.addOperand(MCOperand::CreateReg(ARM::CPSR));
      TmpInst.addOperand(MCOperand::CreateReg(ValReg));
      TmpInst.addOperand(MCOperand::CreateImm(7));
      // Predicate.
      TmpInst.addOperand(MCOperand::CreateImm(ARMCC::AL));
      TmpInst.addOperand(MCOperand::CreateReg(0));
      OutStreamer.EmitInstruction(TmpInst);
    }
    {
      MCInst TmpInst;
      TmpInst.setOpcode(ARM::tSTRi);
      TmpInst.addOperand(MCOperand::CreateReg(ValReg));
      TmpInst.addOperand(MCOperand::CreateReg(SrcReg));
      // The offset immediate is #4. The operand value is scaled by 4 for the
      // tSTR instruction.
      TmpInst.addOperand(MCOperand::CreateImm(1));
      // Predicate.
      TmpInst.addOperand(MCOperand::CreateImm(ARMCC::AL));
      TmpInst.addOperand(MCOperand::CreateReg(0));
      OutStreamer.EmitInstruction(TmpInst);
    }
    {
      MCInst TmpInst;
      TmpInst.setOpcode(ARM::tMOVi8);
      TmpInst.addOperand(MCOperand::CreateReg(ARM::R0));
      TmpInst.addOperand(MCOperand::CreateReg(ARM::CPSR));
      TmpInst.addOperand(MCOperand::CreateImm(0));
      // Predicate.
      TmpInst.addOperand(MCOperand::CreateImm(ARMCC::AL));
      TmpInst.addOperand(MCOperand::CreateReg(0));
      OutStreamer.EmitInstruction(TmpInst);
    }
    {
      const MCExpr *SymbolExpr = MCSymbolRefExpr::Create(Label, OutContext);
      MCInst TmpInst;
      TmpInst.setOpcode(ARM::tB);
      TmpInst.addOperand(MCOperand::CreateExpr(SymbolExpr));
      OutStreamer.EmitInstruction(TmpInst);
    }
    {
      MCInst TmpInst;
      TmpInst.setOpcode(ARM::tMOVi8);
      TmpInst.addOperand(MCOperand::CreateReg(ARM::R0));
      TmpInst.addOperand(MCOperand::CreateReg(ARM::CPSR));
      TmpInst.addOperand(MCOperand::CreateImm(1));
      // Predicate.
      TmpInst.addOperand(MCOperand::CreateImm(ARMCC::AL));
      TmpInst.addOperand(MCOperand::CreateReg(0));
      OutStreamer.AddComment("eh_setjmp end");
      OutStreamer.EmitInstruction(TmpInst);
    }
    OutStreamer.EmitLabel(Label);
    return;
  }

  case ARM::Int_eh_sjlj_setjmp_nofp:
  case ARM::Int_eh_sjlj_setjmp: {
    // Two incoming args: GPR:$src, GPR:$val
    // add $val, pc, #8
    // str $val, [$src, #+4]
    // mov r0, #0
    // add pc, pc, #0
    // mov r0, #1
    unsigned SrcReg = MI->getOperand(0).getReg();
    unsigned ValReg = MI->getOperand(1).getReg();

    {
      MCInst TmpInst;
      TmpInst.setOpcode(ARM::ADDri);
      TmpInst.addOperand(MCOperand::CreateReg(ValReg));
      TmpInst.addOperand(MCOperand::CreateReg(ARM::PC));
      TmpInst.addOperand(MCOperand::CreateImm(8));
      // Predicate.
      TmpInst.addOperand(MCOperand::CreateImm(ARMCC::AL));
      TmpInst.addOperand(MCOperand::CreateReg(0));
      // 's' bit operand (always reg0 for this).
      TmpInst.addOperand(MCOperand::CreateReg(0));
      OutStreamer.AddComment("eh_setjmp begin");
      OutStreamer.EmitInstruction(TmpInst);
    }
    {
      MCInst TmpInst;
      TmpInst.setOpcode(ARM::STRi12);
      TmpInst.addOperand(MCOperand::CreateReg(ValReg));
      TmpInst.addOperand(MCOperand::CreateReg(SrcReg));
      TmpInst.addOperand(MCOperand::CreateImm(4));
      // Predicate.
      TmpInst.addOperand(MCOperand::CreateImm(ARMCC::AL));
      TmpInst.addOperand(MCOperand::CreateReg(0));
      OutStreamer.EmitInstruction(TmpInst);
    }
    {
      MCInst TmpInst;
      TmpInst.setOpcode(ARM::MOVi);
      TmpInst.addOperand(MCOperand::CreateReg(ARM::R0));
      TmpInst.addOperand(MCOperand::CreateImm(0));
      // Predicate.
      TmpInst.addOperand(MCOperand::CreateImm(ARMCC::AL));
      TmpInst.addOperand(MCOperand::CreateReg(0));
      // 's' bit operand (always reg0 for this).
      TmpInst.addOperand(MCOperand::CreateReg(0));
      OutStreamer.EmitInstruction(TmpInst);
    }
    {
      MCInst TmpInst;
      TmpInst.setOpcode(ARM::ADDri);
      TmpInst.addOperand(MCOperand::CreateReg(ARM::PC));
      TmpInst.addOperand(MCOperand::CreateReg(ARM::PC));
      TmpInst.addOperand(MCOperand::CreateImm(0));
      // Predicate.
      TmpInst.addOperand(MCOperand::CreateImm(ARMCC::AL));
      TmpInst.addOperand(MCOperand::CreateReg(0));
      // 's' bit operand (always reg0 for this).
      TmpInst.addOperand(MCOperand::CreateReg(0));
      OutStreamer.EmitInstruction(TmpInst);
    }
    {
      MCInst TmpInst;
      TmpInst.setOpcode(ARM::MOVi);
      TmpInst.addOperand(MCOperand::CreateReg(ARM::R0));
      TmpInst.addOperand(MCOperand::CreateImm(1));
      // Predicate.
      TmpInst.addOperand(MCOperand::CreateImm(ARMCC::AL));
      TmpInst.addOperand(MCOperand::CreateReg(0));
      // 's' bit operand (always reg0 for this).
      TmpInst.addOperand(MCOperand::CreateReg(0));
      OutStreamer.AddComment("eh_setjmp end");
      OutStreamer.EmitInstruction(TmpInst);
    }
    return;
  }
  case ARM::Int_eh_sjlj_longjmp: {
    // ldr sp, [$src, #8]
    // ldr $scratch, [$src, #4]
    // ldr r7, [$src]
    // bx $scratch
    unsigned SrcReg = MI->getOperand(0).getReg();
    unsigned ScratchReg = MI->getOperand(1).getReg();
    {
      MCInst TmpInst;
      TmpInst.setOpcode(ARM::LDRi12);
      TmpInst.addOperand(MCOperand::CreateReg(ARM::SP));
      TmpInst.addOperand(MCOperand::CreateReg(SrcReg));
      TmpInst.addOperand(MCOperand::CreateImm(8));
      // Predicate.
      TmpInst.addOperand(MCOperand::CreateImm(ARMCC::AL));
      TmpInst.addOperand(MCOperand::CreateReg(0));
      OutStreamer.EmitInstruction(TmpInst);
    }
    {
      MCInst TmpInst;
      TmpInst.setOpcode(ARM::LDRi12);
      TmpInst.addOperand(MCOperand::CreateReg(ScratchReg));
      TmpInst.addOperand(MCOperand::CreateReg(SrcReg));
      TmpInst.addOperand(MCOperand::CreateImm(4));
      // Predicate.
      TmpInst.addOperand(MCOperand::CreateImm(ARMCC::AL));
      TmpInst.addOperand(MCOperand::CreateReg(0));
      OutStreamer.EmitInstruction(TmpInst);
    }
    {
      MCInst TmpInst;
      TmpInst.setOpcode(ARM::LDRi12);
      TmpInst.addOperand(MCOperand::CreateReg(ARM::R7));
      TmpInst.addOperand(MCOperand::CreateReg(SrcReg));
      TmpInst.addOperand(MCOperand::CreateImm(0));
      // Predicate.
      TmpInst.addOperand(MCOperand::CreateImm(ARMCC::AL));
      TmpInst.addOperand(MCOperand::CreateReg(0));
      OutStreamer.EmitInstruction(TmpInst);
    }
    {
      MCInst TmpInst;
      TmpInst.setOpcode(ARM::BX);
      TmpInst.addOperand(MCOperand::CreateReg(ScratchReg));
      // Predicate.
      TmpInst.addOperand(MCOperand::CreateImm(ARMCC::AL));
      TmpInst.addOperand(MCOperand::CreateReg(0));
      OutStreamer.EmitInstruction(TmpInst);
    }
    return;
  }
  case ARM::tInt_eh_sjlj_longjmp: {
    // ldr $scratch, [$src, #8]
    // mov sp, $scratch
    // ldr $scratch, [$src, #4]
    // ldr r7, [$src]
    // bx $scratch
    unsigned SrcReg = MI->getOperand(0).getReg();
    unsigned ScratchReg = MI->getOperand(1).getReg();
    {
      MCInst TmpInst;
      TmpInst.setOpcode(ARM::tLDRi);
      TmpInst.addOperand(MCOperand::CreateReg(ScratchReg));
      TmpInst.addOperand(MCOperand::CreateReg(SrcReg));
      // The offset immediate is #8. The operand value is scaled by 4 for the
      // tLDR instruction.
      TmpInst.addOperand(MCOperand::CreateImm(2));
      // Predicate.
      TmpInst.addOperand(MCOperand::CreateImm(ARMCC::AL));
      TmpInst.addOperand(MCOperand::CreateReg(0));
      OutStreamer.EmitInstruction(TmpInst);
    }
    {
      MCInst TmpInst;
      TmpInst.setOpcode(ARM::tMOVtgpr2gpr);
      TmpInst.addOperand(MCOperand::CreateReg(ARM::SP));
      TmpInst.addOperand(MCOperand::CreateReg(ScratchReg));
      // Predicate.
      TmpInst.addOperand(MCOperand::CreateImm(ARMCC::AL));
      TmpInst.addOperand(MCOperand::CreateReg(0));
      OutStreamer.EmitInstruction(TmpInst);
    }
    {
      MCInst TmpInst;
      TmpInst.setOpcode(ARM::tLDRi);
      TmpInst.addOperand(MCOperand::CreateReg(ScratchReg));
      TmpInst.addOperand(MCOperand::CreateReg(SrcReg));
      TmpInst.addOperand(MCOperand::CreateImm(1));
      // Predicate.
      TmpInst.addOperand(MCOperand::CreateImm(ARMCC::AL));
      TmpInst.addOperand(MCOperand::CreateReg(0));
      OutStreamer.EmitInstruction(TmpInst);
    }
    {
      MCInst TmpInst;
      TmpInst.setOpcode(ARM::tLDRr);
      TmpInst.addOperand(MCOperand::CreateReg(ARM::R7));
      TmpInst.addOperand(MCOperand::CreateReg(SrcReg));
      TmpInst.addOperand(MCOperand::CreateReg(0));
      // Predicate.
      TmpInst.addOperand(MCOperand::CreateImm(ARMCC::AL));
      TmpInst.addOperand(MCOperand::CreateReg(0));
      OutStreamer.EmitInstruction(TmpInst);
    }
    {
      MCInst TmpInst;
      TmpInst.setOpcode(ARM::tBX_RET_vararg);
      TmpInst.addOperand(MCOperand::CreateReg(ScratchReg));
      // Predicate.
      TmpInst.addOperand(MCOperand::CreateImm(ARMCC::AL));
      TmpInst.addOperand(MCOperand::CreateReg(0));
      OutStreamer.EmitInstruction(TmpInst);
    }
    return;
  }
  }

  MCInst TmpInst;
  LowerARMMachineInstrToMCInst(MI, TmpInst, *this);
  OutStreamer.EmitInstruction(TmpInst);
}

//===----------------------------------------------------------------------===//
// Target Registry Stuff
//===----------------------------------------------------------------------===//

static MCInstPrinter *createARMMCInstPrinter(const Target &T,
                                             unsigned SyntaxVariant,
                                             const MCAsmInfo &MAI) {
  if (SyntaxVariant == 0)
    return new ARMInstPrinter(MAI);
  return 0;
}

// Force static initialization.
extern "C" void LLVMInitializeARMAsmPrinter() {
  RegisterAsmPrinter<ARMAsmPrinter> X(TheARMTarget);
  RegisterAsmPrinter<ARMAsmPrinter> Y(TheThumbTarget);

  TargetRegistry::RegisterMCInstPrinter(TheARMTarget, createARMMCInstPrinter);
  TargetRegistry::RegisterMCInstPrinter(TheThumbTarget, createARMMCInstPrinter);
}

