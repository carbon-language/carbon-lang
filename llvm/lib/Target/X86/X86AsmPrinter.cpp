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

Statistic<> llvm::EmittedInsts("asm-printer",
                               "Number of machine instrs printed");

enum AsmWriterFlavorTy { att, intel };
cl::opt<AsmWriterFlavorTy>
AsmWriterFlavor("x86-asm-syntax",
                cl::desc("Choose style of code to emit from X86 backend:"),
                cl::values(
                           clEnumVal(att,   "  Emit AT&T-style assembly"),
                           clEnumVal(intel, "  Emit Intel-style assembly"),
                           clEnumValEnd),
                cl::init(att));

/// doInitialization
bool X86SharedAsmPrinter::doInitialization(Module &M) {
  const X86Subtarget *Subtarget = &TM.getSubtarget<X86Subtarget>();
  
  forDarwin = false;
  PrivateGlobalPrefix = ".L";
  
  switch (Subtarget->TargetType) {
  case X86Subtarget::isDarwin:
    AlignmentIsInBytes = false;
    GlobalPrefix = "_";
    Data64bitsDirective = 0;       // we can't emit a 64-bit unit
    ZeroDirective = "\t.space\t";  // ".space N" emits N zeros.
    PrivateGlobalPrefix = "L";     // Marker for constant pool idxs
    ConstantPoolSection = "\t.const\n";
    JumpTableSection = "\t.const\n"; // FIXME: depends on PIC mode
    LCOMMDirective = "\t.lcomm\t";
    COMMDirectiveTakesAlignment = false;
    HasDotTypeDotSizeDirective = false;
    forDarwin = true;
    StaticCtorsSection = ".mod_init_func";
    StaticDtorsSection = ".mod_term_func";
    InlineAsmStart = InlineAsmEnd = "";  // Don't use #APP/#NO_APP
    break;
  case X86Subtarget::isCygwin:
    GlobalPrefix = "_";
    COMMDirectiveTakesAlignment = false;
    HasDotTypeDotSizeDirective = false;
    break;
  case X86Subtarget::isWindows:
    GlobalPrefix = "_";
    HasDotTypeDotSizeDirective = false;
    break;
  default: break;
  }
  
  if (forDarwin) {
    // Emit initial debug information.
    DW.BeginModule(&M);
  }

  return AsmPrinter::doInitialization(M);
}

bool X86SharedAsmPrinter::doFinalization(Module &M) {
  const TargetData &TD = TM.getTargetData();

  // Print out module-level global variables here.
  for (Module::const_global_iterator I = M.global_begin(), E = M.global_end();
       I != E; ++I) {
    if (!I->hasInitializer()) continue;   // External global require no code
    
    // Check to see if this is a special global used by LLVM, if so, emit it.
    if (EmitSpecialLLVMGlobal(I))
      continue;
    
    std::string name = Mang->getValueName(I);
    Constant *C = I->getInitializer();
    unsigned Size = TD.getTypeSize(C->getType());
    unsigned Align = getPreferredAlignmentLog(I);

    if (C->isNullValue() && /* FIXME: Verify correct */
        (I->hasInternalLinkage() || I->hasWeakLinkage() ||
         I->hasLinkOnceLinkage() ||
         (forDarwin && I->hasExternalLinkage() && !I->hasSection()))) {
      if (Size == 0) Size = 1;   // .comm Foo, 0 is undefined, avoid it.
      if (I->hasExternalLinkage()) {
          O << "\t.globl\t" << name << "\n";
          O << "\t.zerofill __DATA__, __common, " << name << ", "
            << Size << ", " << Align;
      } else {
        SwitchSection(".data", I);
        if (LCOMMDirective != NULL) {
          if (I->hasInternalLinkage()) {
            O << LCOMMDirective << name << "," << Size;
            if (forDarwin)
              O << "," << (AlignmentIsInBytes ? (1 << Align) : Align);
          } else
            O << COMMDirective  << name << "," << Size;
        } else {
          if (I->hasInternalLinkage())
            O << "\t.local\t" << name << "\n";
          O << COMMDirective  << name << "," << Size;
          if (COMMDirectiveTakesAlignment)
            O << "," << (AlignmentIsInBytes ? (1 << Align) : Align);
        }
      }
      O << "\t\t" << CommentString << " " << I->getName() << "\n";
    } else {
      switch (I->getLinkage()) {
      case GlobalValue::LinkOnceLinkage:
      case GlobalValue::WeakLinkage:
        if (forDarwin) {
          O << "\t.globl " << name << "\n"
            << "\t.weak_definition " << name << "\n";
          SwitchSection(".section __DATA,__datacoal_nt,coalesced", I);
        } else {
          O << "\t.section\t.llvm.linkonce.d." << name << ",\"aw\",@progbits\n";
          O << "\t.weak " << name << "\n";
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
        SwitchSection(".data", I);
        break;
      default:
        assert(0 && "Unknown linkage type!");
      }

      EmitAlignment(Align, I);
      O << name << ":\t\t\t\t" << CommentString << " " << I->getName()
        << "\n";
      if (HasDotTypeDotSizeDirective)
        O << "\t.size " << name << ", " << Size << "\n";

      EmitGlobalConstant(C);
      O << '\n';
    }
  }
  
  if (forDarwin) {
    SwitchSection("", 0);

    // Output stubs for dynamically-linked functions
    unsigned j = 1;
    for (std::set<std::string>::iterator i = FnStubs.begin(), e = FnStubs.end();
         i != e; ++i, ++j) {
      SwitchSection(".section __IMPORT,__jump_table,symbol_stubs,"
                    "self_modifying_code+pure_instructions,5", 0);
      O << "L" << *i << "$stub:\n";
      O << "\t.indirect_symbol " << *i << "\n";
      O << "\thlt ; hlt ; hlt ; hlt ; hlt\n";
    }

    O << "\n";

    // Output stubs for external and common global variables.
    if (GVStubs.begin() != GVStubs.end())
      SwitchSection(".section __IMPORT,__pointers,non_lazy_symbol_pointers", 0);
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

void X86SharedAsmPrinter::printBasicBlockLabel(const MachineBasicBlock *MBB,
                                               bool printColon,
                                               bool printComment) const {
  O << PrivateGlobalPrefix << "BB" 
    << Mang->getValueName(MBB->getParent()->getFunction()) << "_" 
    << MBB->getNumber();
  if (printColon)
    O << ':';
  if (printComment)
    O << '\t' << CommentString << MBB->getBasicBlock()->getName();
}

/// createX86CodePrinterPass - Returns a pass that prints the X86 assembly code
/// for a MachineFunction to the given output stream, using the given target
/// machine description.
///
FunctionPass *llvm::createX86CodePrinterPass(std::ostream &o,
                                             X86TargetMachine &tm){
  switch (AsmWriterFlavor) {
  default:
    assert(0 && "Unknown asm flavor!");
  case intel:
    return new X86IntelAsmPrinter(o, tm);
  case att:
    return new X86ATTAsmPrinter(o, tm);
  }
}
