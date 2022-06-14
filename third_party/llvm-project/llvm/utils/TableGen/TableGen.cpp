//===- TableGen.cpp - Top-Level TableGen implementation for LLVM ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the main function for LLVM's TableGen.
//
//===----------------------------------------------------------------------===//

#include "TableGenBackends.h" // Declares all backends.
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/TableGen/Main.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/SetTheory.h"

using namespace llvm;

enum ActionType {
  PrintRecords,
  PrintDetailedRecords,
  NullBackend,
  DumpJSON,
  GenEmitter,
  GenRegisterInfo,
  GenInstrInfo,
  GenInstrDocs,
  GenAsmWriter,
  GenAsmMatcher,
  GenDisassembler,
  GenPseudoLowering,
  GenCompressInst,
  GenCallingConv,
  GenDAGISel,
  GenDFAPacketizer,
  GenFastISel,
  GenSubtarget,
  GenIntrinsicEnums,
  GenIntrinsicImpl,
  PrintEnums,
  PrintSets,
  GenOptParserDefs,
  GenOptRST,
  GenCTags,
  GenAttributes,
  GenSearchableTables,
  GenGlobalISel,
  GenGICombiner,
  GenX86EVEX2VEXTables,
  GenX86FoldTables,
  GenX86MnemonicTables,
  GenRegisterBank,
  GenExegesis,
  GenAutomata,
  GenDirectivesEnumDecl,
  GenDirectivesEnumImpl,
};

namespace llvm {
cl::opt<bool> EmitLongStrLiterals(
    "long-string-literals",
    cl::desc("when emitting large string tables, prefer string literals over "
             "comma-separated char literals. This can be a readability and "
             "compile-time performance win, but upsets some compilers"),
    cl::Hidden, cl::init(true));
} // end namespace llvm

namespace {
cl::opt<ActionType> Action(
    cl::desc("Action to perform:"),
    cl::values(
        clEnumValN(PrintRecords, "print-records",
                   "Print all records to stdout (default)"),
        clEnumValN(PrintDetailedRecords, "print-detailed-records",
                   "Print full details of all records to stdout"),
        clEnumValN(NullBackend, "null-backend",
                   "Do nothing after parsing (useful for timing)"),
        clEnumValN(DumpJSON, "dump-json",
                   "Dump all records as machine-readable JSON"),
        clEnumValN(GenEmitter, "gen-emitter", "Generate machine code emitter"),
        clEnumValN(GenRegisterInfo, "gen-register-info",
                   "Generate registers and register classes info"),
        clEnumValN(GenInstrInfo, "gen-instr-info",
                   "Generate instruction descriptions"),
        clEnumValN(GenInstrDocs, "gen-instr-docs",
                   "Generate instruction documentation"),
        clEnumValN(GenCallingConv, "gen-callingconv",
                   "Generate calling convention descriptions"),
        clEnumValN(GenAsmWriter, "gen-asm-writer", "Generate assembly writer"),
        clEnumValN(GenDisassembler, "gen-disassembler",
                   "Generate disassembler"),
        clEnumValN(GenPseudoLowering, "gen-pseudo-lowering",
                   "Generate pseudo instruction lowering"),
        clEnumValN(GenCompressInst, "gen-compress-inst-emitter",
                   "Generate RISCV compressed instructions."),
        clEnumValN(GenAsmMatcher, "gen-asm-matcher",
                   "Generate assembly instruction matcher"),
        clEnumValN(GenDAGISel, "gen-dag-isel",
                   "Generate a DAG instruction selector"),
        clEnumValN(GenDFAPacketizer, "gen-dfa-packetizer",
                   "Generate DFA Packetizer for VLIW targets"),
        clEnumValN(GenFastISel, "gen-fast-isel",
                   "Generate a \"fast\" instruction selector"),
        clEnumValN(GenSubtarget, "gen-subtarget",
                   "Generate subtarget enumerations"),
        clEnumValN(GenIntrinsicEnums, "gen-intrinsic-enums",
                   "Generate intrinsic enums"),
        clEnumValN(GenIntrinsicImpl, "gen-intrinsic-impl",
                   "Generate intrinsic information"),
        clEnumValN(PrintEnums, "print-enums", "Print enum values for a class"),
        clEnumValN(PrintSets, "print-sets",
                   "Print expanded sets for testing DAG exprs"),
        clEnumValN(GenOptParserDefs, "gen-opt-parser-defs",
                   "Generate option definitions"),
        clEnumValN(GenOptRST, "gen-opt-rst", "Generate option RST"),
        clEnumValN(GenCTags, "gen-ctags", "Generate ctags-compatible index"),
        clEnumValN(GenAttributes, "gen-attrs", "Generate attributes"),
        clEnumValN(GenSearchableTables, "gen-searchable-tables",
                   "Generate generic binary-searchable table"),
        clEnumValN(GenGlobalISel, "gen-global-isel",
                   "Generate GlobalISel selector"),
        clEnumValN(GenGICombiner, "gen-global-isel-combiner",
                   "Generate GlobalISel combiner"),
        clEnumValN(GenX86EVEX2VEXTables, "gen-x86-EVEX2VEX-tables",
                   "Generate X86 EVEX to VEX compress tables"),
        clEnumValN(GenX86FoldTables, "gen-x86-fold-tables",
                   "Generate X86 fold tables"),
        clEnumValN(GenX86MnemonicTables, "gen-x86-mnemonic-tables",
                   "Generate X86 mnemonic tables"),
        clEnumValN(GenRegisterBank, "gen-register-bank",
                   "Generate registers bank descriptions"),
        clEnumValN(GenExegesis, "gen-exegesis",
                   "Generate llvm-exegesis tables"),
        clEnumValN(GenAutomata, "gen-automata", "Generate generic automata"),
        clEnumValN(GenDirectivesEnumDecl, "gen-directive-decl",
                   "Generate directive related declaration code (header file)"),
        clEnumValN(GenDirectivesEnumImpl, "gen-directive-impl",
                   "Generate directive related implementation code")));

cl::OptionCategory PrintEnumsCat("Options for -print-enums");
cl::opt<std::string> Class("class", cl::desc("Print Enum list for this class"),
                           cl::value_desc("class name"),
                           cl::cat(PrintEnumsCat));

bool LLVMTableGenMain(raw_ostream &OS, RecordKeeper &Records) {
  switch (Action) {
  case PrintRecords:
    OS << Records;              // No argument, dump all contents
    break;
  case PrintDetailedRecords:
    EmitDetailedRecords(Records, OS);
    break;
  case NullBackend:             // No backend at all.
    break;
  case DumpJSON:
    EmitJSON(Records, OS);
    break;
  case GenEmitter:
    EmitCodeEmitter(Records, OS);
    break;
  case GenRegisterInfo:
    EmitRegisterInfo(Records, OS);
    break;
  case GenInstrInfo:
    EmitInstrInfo(Records, OS);
    break;
  case GenInstrDocs:
    EmitInstrDocs(Records, OS);
    break;
  case GenCallingConv:
    EmitCallingConv(Records, OS);
    break;
  case GenAsmWriter:
    EmitAsmWriter(Records, OS);
    break;
  case GenAsmMatcher:
    EmitAsmMatcher(Records, OS);
    break;
  case GenDisassembler:
    EmitDisassembler(Records, OS);
    break;
  case GenPseudoLowering:
    EmitPseudoLowering(Records, OS);
    break;
  case GenCompressInst:
    EmitCompressInst(Records, OS);
    break;
  case GenDAGISel:
    EmitDAGISel(Records, OS);
    break;
  case GenDFAPacketizer:
    EmitDFAPacketizer(Records, OS);
    break;
  case GenFastISel:
    EmitFastISel(Records, OS);
    break;
  case GenSubtarget:
    EmitSubtarget(Records, OS);
    break;
  case GenIntrinsicEnums:
    EmitIntrinsicEnums(Records, OS);
    break;
  case GenIntrinsicImpl:
    EmitIntrinsicImpl(Records, OS);
    break;
  case GenOptParserDefs:
    EmitOptParser(Records, OS);
    break;
  case GenOptRST:
    EmitOptRST(Records, OS);
    break;
  case PrintEnums:
  {
    for (Record *Rec : Records.getAllDerivedDefinitions(Class))
      OS << Rec->getName() << ", ";
    OS << "\n";
    break;
  }
  case PrintSets:
  {
    SetTheory Sets;
    Sets.addFieldExpander("Set", "Elements");
    for (Record *Rec : Records.getAllDerivedDefinitions("Set")) {
      OS << Rec->getName() << " = [";
      const std::vector<Record*> *Elts = Sets.expand(Rec);
      assert(Elts && "Couldn't expand Set instance");
      for (Record *Elt : *Elts)
        OS << ' ' << Elt->getName();
      OS << " ]\n";
    }
    break;
  }
  case GenCTags:
    EmitCTags(Records, OS);
    break;
  case GenAttributes:
    EmitAttributes(Records, OS);
    break;
  case GenSearchableTables:
    EmitSearchableTables(Records, OS);
    break;
  case GenGlobalISel:
    EmitGlobalISel(Records, OS);
    break;
  case GenGICombiner:
    EmitGICombiner(Records, OS);
    break;
  case GenRegisterBank:
    EmitRegisterBank(Records, OS);
    break;
  case GenX86EVEX2VEXTables:
    EmitX86EVEX2VEXTables(Records, OS);
    break;
  case GenX86MnemonicTables:
    EmitX86MnemonicTables(Records, OS);
    break;
  case GenX86FoldTables:
    EmitX86FoldTables(Records, OS);
    break;
  case GenExegesis:
    EmitExegesis(Records, OS);
    break;
  case GenAutomata:
    EmitAutomata(Records, OS);
    break;
  case GenDirectivesEnumDecl:
    EmitDirectivesDecl(Records, OS);
    break;
  case GenDirectivesEnumImpl:
    EmitDirectivesImpl(Records, OS);
    break;
  }

  return false;
}
}

int main(int argc, char **argv) {
  InitLLVM X(argc, argv);
  cl::ParseCommandLineOptions(argc, argv);

  return TableGenMain(argv[0], &LLVMTableGenMain);
}

#ifndef __has_feature
#define __has_feature(x) 0
#endif

#if __has_feature(address_sanitizer) ||                                        \
    (defined(__SANITIZE_ADDRESS__) && defined(__GNUC__)) ||                    \
    __has_feature(leak_sanitizer)

#include <sanitizer/lsan_interface.h>
// Disable LeakSanitizer for this binary as it has too many leaks that are not
// very interesting to fix. See compiler-rt/include/sanitizer/lsan_interface.h .
LLVM_ATTRIBUTE_USED int __lsan_is_turned_off() { return 1; }

#endif
