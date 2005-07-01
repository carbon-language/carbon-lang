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
#include "llvm/Assembly/Writer.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/Support/Mangler.h"
#include "llvm/Support/CommandLine.h"
using namespace llvm;
using namespace x86;

Statistic<> EmittedInsts("asm-printer", "Number of machine instrs printed");

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
  #elif defined(__MACOSX__)
    forDarwin = true;
  #elif defined(_WIN32)
    leadingUnderscore = true;
  #else
    leadingUnderscore = false;
  #endif
  }
  if (leadingUnderscore || forCygwin || forDarwin)
    GlobalPrefix = "_";

  if (forDarwin)
    AlignmentIsInBytes = false;

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

  for (unsigned i = 0, e = CP.size(); i != e; ++i) {
    O << "\t.section .rodata\n";
    emitAlignment(TD.getTypeAlignmentShift(CP[i]->getType()));
    O << ".CPI" << CurrentFnName << "_" << i << ":\t\t\t\t\t" << CommentString
      << *CP[i] << "\n";
    emitGlobalConstant(CP[i]);
  }
}

bool X86SharedAsmPrinter::doFinalization(Module &M) {
  const TargetData &TD = TM.getTargetData();
  std::string CurSection;

  // Print out module-level global variables here.
  for (Module::const_global_iterator I = M.global_begin(), E = M.global_end(); I != E; ++I)
  	if (I->hasInitializer()) {   // External global require no code
  	  O << "\n\n";
  	  std::string name = Mang->getValueName(I);
  	  Constant *C = I->getInitializer();
  	  unsigned Size = TD.getTypeSize(C->getType());
  	  unsigned Align = TD.getTypeAlignmentShift(C->getType());

  	  if (C->isNullValue() &&
  		(I->hasLinkOnceLinkage() || I->hasInternalLinkage() ||
  			I->hasWeakLinkage() /* FIXME: Verify correct */)) {
  		SwitchSection(O, CurSection, ".data");
  		if (!forCygwin && I->hasInternalLinkage())
  		  O << "\t.local " << name << "\n";
  		O << "\t.comm " << name << "," << TD.getTypeSize(C->getType());
  		if (!forCygwin)
  		  O << "," << (1 << Align);
  		O << "\t\t# ";
  		WriteAsOperand(O, I, true, true, &M);
  		O << "\n";
  	  } else {
  		switch (I->getLinkage()) {
  		case GlobalValue::LinkOnceLinkage:
  		case GlobalValue::WeakLinkage:   // FIXME: Verify correct for weak.
  		  // Nonnull linkonce -> weak
  		  O << "\t.weak " << name << "\n";
  		  SwitchSection(O, CurSection, "");
  		  O << "\t.section\t.llvm.linkonce.d." << name << ",\"aw\",@progbits\n";
  		  break;
  		case GlobalValue::AppendingLinkage:
  		  // FIXME: appending linkage variables should go into a section of
  		  // their name or something.  For now, just emit them as external.
  		case GlobalValue::ExternalLinkage:
  		  // If external or appending, declare as a global symbol
  		  O << "\t.globl " << name << "\n";
  		  // FALL THROUGH
  		case GlobalValue::InternalLinkage:
  		  if (C->isNullValue())
  		    SwitchSection(O, CurSection, ".bss");
  		  else
  		    SwitchSection(O, CurSection, ".data");
  		  break;
  		case GlobalValue::GhostLinkage:
  		  std::cerr << "GhostLinkage cannot appear in X86AsmPrinter!\n";
  		  abort();
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
