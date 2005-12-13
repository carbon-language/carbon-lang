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

#include "X86ATTAsmPrinter.h"
#include "X86IntelAsmPrinter.h"
#include "X86Subtarget.h"
#include "X86.h"
#include "llvm/Constants.h"
#include "llvm/Module.h"
#include "llvm/Type.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Support/Mangler.h"
#include "llvm/Support/CommandLine.h"
using namespace llvm;
using namespace x86;

Statistic<> llvm::x86::EmittedInsts("asm-printer",
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
  
  switch (Subtarget->TargetType) {
  case X86Subtarget::isDarwin:
    AlignmentIsInBytes = false;
    GlobalPrefix = "_";
    Data64bitsDirective = 0;       // we can't emit a 64-bit unit
    ZeroDirective = "\t.space\t";  // ".space N" emits N zeros.
    PrivateGlobalPrefix = "L";     // Marker for constant pool idxs
    ConstantPoolSection = "\t.const\n";
    LCOMMDirective = "\t.lcomm\t";
    COMMDirectiveTakesAlignment = false;
    HasDotTypeDotSizeDirective = false;
    forDarwin = true;
    StaticCtorsSection = ".mod_init_func";
    StaticDtorsSection = ".mod_term_func";
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
  
  return AsmPrinter::doInitialization(M);
}

bool X86SharedAsmPrinter::doFinalization(Module &M) {
  const TargetData &TD = TM.getTargetData();

  // Print out module-level global variables here.
  for (Module::const_global_iterator I = M.global_begin(),
       E = M.global_end(); I != E; ++I) {
    if (!I->hasInitializer()) continue;   // External global require no code
    
    // Check to see if this is a special global used by LLVM, if so, emit it.
    if (I->hasAppendingLinkage() && EmitSpecialLLVMGlobal(I))
      continue;
    
    O << "\n\n";
    std::string name = Mang->getValueName(I);
    Constant *C = I->getInitializer();
    unsigned Size = TD.getTypeSize(C->getType());
    unsigned Align = TD.getTypeAlignmentShift(C->getType());

    switch (I->getLinkage()) {
    default: assert(0 && "Unknown linkage type!");
    case GlobalValue::LinkOnceLinkage:
    case GlobalValue::WeakLinkage:   // FIXME: Verify correct for weak.
      if (C->isNullValue()) {
        O << COMMDirective << name << "," << Size;
        if (COMMDirectiveTakesAlignment)
          O << "," << (1 << Align);
        O << "\t\t" << CommentString << " " << I->getName() << "\n";
        continue;
      }
      
      // Nonnull linkonce -> weak
      O << "\t.weak " << name << "\n";
      O << "\t.section\t.llvm.linkonce.d." << name << ",\"aw\",@progbits\n";
      SwitchSection("", I);
      break;
    case GlobalValue::InternalLinkage:
      if (C->isNullValue()) {
        if (LCOMMDirective) {
          O << LCOMMDirective << name << "," << Size << "," << Align;
          continue;
        } else {
          SwitchSection(".bss", I);
          O << "\t.local " << name << "\n";
          O << COMMDirective << name << "," << Size;
          if (COMMDirectiveTakesAlignment)
            O << "," << (1 << Align);
          O << "\t\t# ";
          WriteAsOperand(O, I, true, true, &M);
          O << "\n";
          continue;
        }
      }
      SwitchSection(".data", I);
      break;
    case GlobalValue::AppendingLinkage:
      // FIXME: appending linkage variables should go into a section of
      // their name or something.  For now, just emit them as external.
    case GlobalValue::ExternalLinkage:
      SwitchSection(C->isNullValue() ? ".bss" : ".data", I);
      // If external or appending, declare as a global symbol
      O << "\t.globl " << name << "\n";
      break;
    }

    EmitAlignment(Align);
    if (HasDotTypeDotSizeDirective) {
      O << "\t.type " << name << ",@object\n";
      O << "\t.size " << name << "," << Size << "\n";
    }
    O << name << ":\t\t\t" << CommentString << ' ' << I->getName() << '\n';
    EmitGlobalConstant(C);
  }
  
  if (forDarwin) {
    SwitchSection("", 0);
    // Output stubs for external global variables
    if (GVStubs.begin() != GVStubs.end())
      O << "\t.non_lazy_symbol_pointer\n";
    for (std::set<std::string>::iterator i = GVStubs.begin(), e = GVStubs.end();
         i != e; ++i) {
      O << "L" << *i << "$non_lazy_ptr:\n";
      O << "\t.indirect_symbol " << *i << "\n";
      O << "\t.long\t0\n";
    }

    // Output stubs for dynamically-linked functions
    unsigned j = 1;
    for (std::set<std::string>::iterator i = FnStubs.begin(), e = FnStubs.end();
         i != e; ++i, ++j) {
      O << "\t.symbol_stub\n";
      O << "L" << *i << "$stub:\n";
      O << "\t.indirect_symbol " << *i << "\n";
      O << "\tjmp\t*L" << j << "$lz\n";
      O << "L" << *i << "$stub_binder:\n";
      O << "\tpushl\t$L" << j << "$lz\n";
      O << "\tjmp\tdyld_stub_binding_helper\n";
      O << "\t.section __DATA, __la_sym_ptr3,lazy_symbol_pointers\n";
      O << "L" << j << "$lz:\n";
      O << "\t.indirect_symbol " << *i << "\n";
      O << "\t.long\tL" << *i << "$stub_binder\n";
    }

    O << "\n";

    // Output stubs for link-once variables
    if (LinkOnceStubs.begin() != LinkOnceStubs.end())
      O << ".data\n.align 2\n";
    for (std::set<std::string>::iterator i = LinkOnceStubs.begin(),
         e = LinkOnceStubs.end(); i != e; ++i) {
      O << "L" << *i << "$non_lazy_ptr:\n"
        << "\t.long\t" << *i << '\n';
    }
  }

  AsmPrinter::doFinalization(M);
  return false; // success
}

/// createX86CodePrinterPass - Returns a pass that prints the X86 assembly code
/// for a MachineFunction to the given output stream, using the given target
/// machine description.
///
FunctionPass *llvm::createX86CodePrinterPass(std::ostream &o,TargetMachine &tm){
  switch (AsmWriterFlavor) {
  default:
    assert(0 && "Unknown asm flavor!");
  case intel:
    return new X86IntelAsmPrinter(o, tm);
  case att:
    return new X86ATTAsmPrinter(o, tm);
  }
}
