//===-- X86AsmPrinter.cpp - Convert X86 LLVM IR to X86 assembly -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file the shared super class printer that converts from our internal
// representation of machine-dependent LLVM code to Intel and AT&T format
// assembly language.
// This printer is the output mechanism used by `llc'.
//
//===----------------------------------------------------------------------===//

#include "X86AsmPrinter.h"
#include "X86ATTAsmPrinter.h"
#include "X86IntelAsmPrinter.h"
#include "X86Subtarget.h"
#include "llvm/Constants.h"
#include "llvm/Module.h"
#include "llvm/Type.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Support/Mangler.h"
#include "llvm/Support/CommandLine.h"
using namespace llvm;

enum AsmWriterFlavorTy { att, intel };

Statistic<> llvm::EmittedInsts("asm-printer",
                               "Number of machine instrs printed");

cl::opt<AsmWriterFlavorTy>
AsmWriterFlavor("x86-asm-syntax",
                cl::desc("Choose style of code to emit from X86 backend:"),
                cl::values(
                           clEnumVal(att,   "  Emit AT&T-style assembly"),
                           clEnumVal(intel, "  Emit Intel-style assembly"),
                           clEnumValEnd),
#ifdef _MSC_VER
                cl::init(intel)
#else
                cl::init(att)
#endif
                );

X86TargetAsmInfo::X86TargetAsmInfo(X86TargetMachine &TM) {
  const X86Subtarget *Subtarget = &TM.getSubtarget<X86Subtarget>();
  
  //FIXME - Should to be simplified.
   
  switch (Subtarget->TargetType) {
  case X86Subtarget::isDarwin:
    AlignmentIsInBytes = false;
    GlobalPrefix = "_";
    Data64bitsDirective = 0;       // we can't emit a 64-bit unit
    ZeroDirective = "\t.space\t";  // ".space N" emits N zeros.
    PrivateGlobalPrefix = "L";     // Marker for constant pool idxs
    ConstantPoolSection = "\t.const\n";
    JumpTableDataSection = "\t.const\n"; // FIXME: depends on PIC mode
    FourByteConstantSection = "\t.literal4\n";
    EightByteConstantSection = "\t.literal8\n";
    LCOMMDirective = "\t.lcomm\t";
    COMMDirectiveTakesAlignment = false;
    HasDotTypeDotSizeDirective = false;
    StaticCtorsSection = ".mod_init_func";
    StaticDtorsSection = ".mod_term_func";
    InlineAsmStart = "# InlineAsm Start";
    InlineAsmEnd = "# InlineAsm End";
    SetDirective = "\t.set";
    
    NeedsSet = true;
    DwarfAbbrevSection = ".section __DWARF,__debug_abbrev,regular,debug";
    DwarfInfoSection = ".section __DWARF,__debug_info,regular,debug";
    DwarfLineSection = ".section __DWARF,__debug_line,regular,debug";
    DwarfFrameSection = ".section __DWARF,__debug_frame,regular,debug";
    DwarfPubNamesSection = ".section __DWARF,__debug_pubnames,regular,debug";
    DwarfPubTypesSection = ".section __DWARF,__debug_pubtypes,regular,debug";
    DwarfStrSection = ".section __DWARF,__debug_str,regular,debug";
    DwarfLocSection = ".section __DWARF,__debug_loc,regular,debug";
    DwarfARangesSection = ".section __DWARF,__debug_aranges,regular,debug";
    DwarfRangesSection = ".section __DWARF,__debug_ranges,regular,debug";
    DwarfMacInfoSection = ".section __DWARF,__debug_macinfo,regular,debug";
    break;
  case X86Subtarget::isCygwin:
    GlobalPrefix = "_";
    COMMDirectiveTakesAlignment = false;
    HasDotTypeDotSizeDirective = false;
    StaticCtorsSection = "\t.section .ctors,\"aw\"";
    StaticDtorsSection = "\t.section .dtors,\"aw\"";
    break;
  case X86Subtarget::isWindows:
    GlobalPrefix = "_";
    HasDotTypeDotSizeDirective = false;
    break;
  default: break;
  }
  
  if (AsmWriterFlavor == intel) {
    GlobalPrefix = "_";
    CommentString = ";";
  
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
    
    TextSection = "_text";
    DataSection = "_data";
    SwitchToSectionDirective = "";
    TextSectionStartSuffix = "\tsegment 'CODE'";
    DataSectionStartSuffix = "\tsegment 'DATA'";
    SectionEndDirectiveSuffix = "\tends\n";
  }
}

/// doInitialization
bool X86SharedAsmPrinter::doInitialization(Module &M) {  
  if (Subtarget->isTargetDarwin()) {
    // Emit initial debug information.
    DW.BeginModule(&M);
  }

  return AsmPrinter::doInitialization(M);
}

bool X86SharedAsmPrinter::doFinalization(Module &M) {
  // Note: this code is not shared by the Intel printer as it is too different
  // from how MASM does things.  When making changes here don't forget to look
  // at X86IntelAsmPrinter::doFinalization().
  const TargetData *TD = TM.getTargetData();

  // Print out module-level global variables here.
  for (Module::const_global_iterator I = M.global_begin(), E = M.global_end();
       I != E; ++I) {
    if (!I->hasInitializer()) continue;   // External global require no code
    
    // Check to see if this is a special global used by LLVM, if so, emit it.
    if (EmitSpecialLLVMGlobal(I))
      continue;
    
    std::string name = Mang->getValueName(I);
    Constant *C = I->getInitializer();
    unsigned Size = TD->getTypeSize(C->getType());
    unsigned Align = getPreferredAlignmentLog(I);

    if (C->isNullValue() && /* FIXME: Verify correct */
        (I->hasInternalLinkage() || I->hasWeakLinkage() ||
         I->hasLinkOnceLinkage() ||
         (Subtarget->isTargetDarwin() && 
          I->hasExternalLinkage() && !I->hasSection()))) {
      if (Size == 0) Size = 1;   // .comm Foo, 0 is undefined, avoid it.
      if (I->hasExternalLinkage()) {
          O << "\t.globl\t" << name << "\n";
          O << "\t.zerofill __DATA__, __common, " << name << ", "
            << Size << ", " << Align;
      } else {
        SwitchToDataSection(TAI->getDataSection(), I);
        if (TAI->getLCOMMDirective() != NULL) {
          if (I->hasInternalLinkage()) {
            O << TAI->getLCOMMDirective() << name << "," << Size;
            if (Subtarget->isTargetDarwin())
              O << "," << (TAI->getAlignmentIsInBytes() ? (1 << Align) : Align);
          } else
            O << TAI->getCOMMDirective()  << name << "," << Size;
        } else {
          if (Subtarget->TargetType != X86Subtarget::isCygwin) {
            if (I->hasInternalLinkage())
              O << "\t.local\t" << name << "\n";
          }
          O << TAI->getCOMMDirective()  << name << "," << Size;
          if (TAI->getCOMMDirectiveTakesAlignment())
            O << "," << (TAI->getAlignmentIsInBytes() ? (1 << Align) : Align);
        }
      }
      O << "\t\t" << TAI->getCommentString() << " " << I->getName() << "\n";
    } else {
      switch (I->getLinkage()) {
      case GlobalValue::LinkOnceLinkage:
      case GlobalValue::WeakLinkage:
        if (Subtarget->isTargetDarwin()) {
          O << "\t.globl " << name << "\n"
            << "\t.weak_definition " << name << "\n";
          SwitchToDataSection(".section __DATA,__const_coal,coalesced", I);
        } else if (Subtarget->TargetType == X86Subtarget::isCygwin) {
          O << "\t.section\t.llvm.linkonce.d." << name << ",\"aw\"\n"
            << "\t.weak " << name << "\n";
        } else {
          O << "\t.section\t.llvm.linkonce.d." << name << ",\"aw\",@progbits\n"
            << "\t.weak " << name << "\n";
        }
        break;
      case GlobalValue::AppendingLinkage:
        // FIXME: appending linkage variables should go into a section of
        // their name or something.  For now, just emit them as external.
      case GlobalValue::ExternalLinkage:
        // If external or appending, declare as a global symbol
        O << "\t.globl " << name << "\n";
        // FALL THROUGH
      case GlobalValue::InternalLinkage:
        SwitchToDataSection(TAI->getDataSection(), I);
        break;
      default:
        assert(0 && "Unknown linkage type!");
      }

      EmitAlignment(Align, I);
      O << name << ":\t\t\t\t" << TAI->getCommentString() << " " << I->getName()
        << "\n";
      if (TAI->hasDotTypeDotSizeDirective())
        O << "\t.size " << name << ", " << Size << "\n";

      EmitGlobalConstant(C);
      O << '\n';
    }
  }
  
  if (Subtarget->isTargetDarwin()) {
    SwitchToDataSection("", 0);

    // Output stubs for dynamically-linked functions
    unsigned j = 1;
    for (std::set<std::string>::iterator i = FnStubs.begin(), e = FnStubs.end();
         i != e; ++i, ++j) {
      SwitchToDataSection(".section __IMPORT,__jump_table,symbol_stubs,"
                          "self_modifying_code+pure_instructions,5", 0);
      O << "L" << *i << "$stub:\n";
      O << "\t.indirect_symbol " << *i << "\n";
      O << "\thlt ; hlt ; hlt ; hlt ; hlt\n";
    }

    O << "\n";

    // Output stubs for external and common global variables.
    if (GVStubs.begin() != GVStubs.end())
      SwitchToDataSection(
                    ".section __IMPORT,__pointers,non_lazy_symbol_pointers", 0);
    for (std::set<std::string>::iterator i = GVStubs.begin(), e = GVStubs.end();
         i != e; ++i) {
      O << "L" << *i << "$non_lazy_ptr:\n";
      O << "\t.indirect_symbol " << *i << "\n";
      O << "\t.long\t0\n";
    }

    // Emit initial debug information.
    DW.EndModule();

    // Funny Darwin hack: This flag tells the linker that no global symbols
    // contain code that falls through to other global symbols (e.g. the obvious
    // implementation of multiple entry points).  If this doesn't occur, the
    // linker can safely perform dead code stripping.  Since LLVM never
    // generates code that does this, it is always safe to set.
    O << "\t.subsections_via_symbols\n";
  }

  AsmPrinter::doFinalization(M);
  return false; // success
}

/// createX86CodePrinterPass - Returns a pass that prints the X86 assembly code
/// for a MachineFunction to the given output stream, using the given target
/// machine description.
///
FunctionPass *llvm::createX86CodePrinterPass(std::ostream &o,
                                             X86TargetMachine &tm) {
  TargetAsmInfo *TAI = new X86TargetAsmInfo(tm);

  switch (AsmWriterFlavor) {
  default:
    assert(0 && "Unknown asm flavor!");
  case intel: return new X86IntelAsmPrinter(o, tm, TAI);
  case att: return new X86ATTAsmPrinter(o, tm, TAI);
  }
}
