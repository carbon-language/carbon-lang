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
#include "X86.h"
#include "llvm/Module.h"
#include "llvm/Type.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/CodeGen/MachineConstantPool.h"
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
bool X86SharedAsmPrinter::doInitialization(Module& M) {
  bool leadingUnderscore = false;
  forCygwin = false;
  const std::string& TT = M.getTargetTriple();
  if (TT.length() > 5) {
    forCygwin = TT.find("cygwin") != std::string::npos ||
                TT.find("mingw")  != std::string::npos;
    forDarwin = TT.find("darwin") != std::string::npos;
  } else if (TT.empty()) {
#if defined(__CYGWIN__) || defined(__MINGW32__)
    forCygwin = true;
#elif defined(__APPLE__)
    forDarwin = true;
#elif defined(_WIN32)
    leadingUnderscore = true;
#else
    leadingUnderscore = false;
#endif
  }

  if (leadingUnderscore || forCygwin || forDarwin)
    GlobalPrefix = "_";

  if (forDarwin) {
    AlignmentIsInBytes = false;
    Data64bitsDirective = 0;       // we can't emit a 64-bit unit
    ZeroDirective = "\t.space\t";  // ".space N" emits N zeros.
    PrivateGlobalPrefix = "L";     // Marker for constant pool idxs
  }

  return AsmPrinter::doInitialization(M);
}

/// printConstantPool - Print to the current output stream assembly
/// representations of the constants in the constant pool MCP. This is
/// used to print out constants which have been "spilled to memory" by
/// the code generator.
///
void X86SharedAsmPrinter::printConstantPool(MachineConstantPool *MCP) {
  const std::vector<Constant*> &CP = MCP->getConstants();
  const TargetData &TD = TM.getTargetData();

  if (CP.empty()) return;

  SwitchSection(forDarwin ? "\t.const\n" : "\t.section .rodata\n", 0);
  
  for (unsigned i = 0, e = CP.size(); i != e; ++i) {
    // FIXME: force doubles to be naturally aligned.  We should handle this
    // more correctly in the future.
    if (CP[i]->getType() == Type::DoubleTy)
      emitAlignment(3);
    else
      emitAlignment(TD.getTypeAlignmentShift(CP[i]->getType()));
    O << PrivateGlobalPrefix << "CPI" << CurrentFnName << "_" << i
      << ":\t\t\t\t\t" << CommentString << *CP[i] << "\n";
    emitGlobalConstant(CP[i]);
  }
}

bool X86SharedAsmPrinter::doFinalization(Module &M) {
  const TargetData &TD = TM.getTargetData();

  // Print out module-level global variables here.
  for (Module::const_global_iterator I = M.global_begin(),
         E = M.global_end(); I != E; ++I)
    if (I->hasInitializer()) {   // External global require no code
      O << "\n\n";
      std::string name = Mang->getValueName(I);
      Constant *C = I->getInitializer();
      unsigned Size = TD.getTypeSize(C->getType());
      unsigned Align = TD.getTypeAlignmentShift(C->getType());

      if (C->isNullValue() &&
          (I->hasLinkOnceLinkage() || I->hasInternalLinkage() ||
           I->hasWeakLinkage() /* FIXME: Verify correct */)) {
        SwitchSection(".data", I);
        if (!forCygwin && !forDarwin && I->hasInternalLinkage())
          O << "\t.local " << name << "\n";
        if (forDarwin && I->hasInternalLinkage())
          O << "\t.lcomm " << name << "," << Size << "," << Align;
        else
          O << "\t.comm " << name << "," << Size;
        if (!forCygwin && !forDarwin)
          O << "," << (1 << Align);
        O << "\t\t# ";
        WriteAsOperand(O, I, true, true, &M);
        O << "\n";
      } else {
        switch (I->getLinkage()) {
        default: assert(0 && "Unknown linkage type!");
        case GlobalValue::LinkOnceLinkage:
        case GlobalValue::WeakLinkage:   // FIXME: Verify correct for weak.
          // Nonnull linkonce -> weak
          O << "\t.weak " << name << "\n";
          O << "\t.section\t.llvm.linkonce.d." << name << ",\"aw\",@progbits\n";
          SwitchSection("", I);
          break;
        case GlobalValue::AppendingLinkage:
          // FIXME: appending linkage variables should go into a section of
          // their name or something.  For now, just emit them as external.
        case GlobalValue::ExternalLinkage:
          // If external or appending, declare as a global symbol
          O << "\t.globl " << name << "\n";
          // FALL THROUGH
        case GlobalValue::InternalLinkage:
          SwitchSection(C->isNullValue() ? ".bss" : ".data", I);
          break;
        }

        emitAlignment(Align);
        if (!forCygwin && !forDarwin) {
          O << "\t.type " << name << ",@object\n";
          O << "\t.size " << name << "," << Size << "\n";
        }
        O << name << ":\t\t\t\t# ";
        WriteAsOperand(O, I, true, true, &M);
        O << " = ";
        WriteAsOperand(O, C, false, false, &M);
        O << "\n";
        emitGlobalConstant(C);
      }
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
