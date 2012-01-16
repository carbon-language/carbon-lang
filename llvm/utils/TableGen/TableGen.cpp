//===- TableGen.cpp - Top-Level TableGen implementation for LLVM ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the main function for LLVM's TableGen.
//
//===----------------------------------------------------------------------===//

#include "AsmMatcherEmitter.h"
#include "AsmWriterEmitter.h"
#include "CallingConvEmitter.h"
#include "CodeEmitterGen.h"
#include "DAGISelEmitter.h"
#include "DFAPacketizerEmitter.h"
#include "DisassemblerEmitter.h"
#include "EDEmitter.h"
#include "FastISelEmitter.h"
#include "InstrInfoEmitter.h"
#include "IntrinsicEmitter.h"
#include "PseudoLoweringEmitter.h"
#include "RegisterInfoEmitter.h"
#include "SubtargetEmitter.h"
#include "SetTheory.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Main.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenAction.h"

using namespace llvm;

enum ActionType {
  PrintRecords,
  GenEmitter,
  GenRegisterInfo,
  GenInstrInfo,
  GenAsmWriter,
  GenAsmMatcher,
  GenDisassembler,
  GenPseudoLowering,
  GenCallingConv,
  GenDAGISel,
  GenDFAPacketizer,
  GenFastISel,
  GenSubtarget,
  GenIntrinsic,
  GenTgtIntrinsic,
  GenEDInfo,
  PrintEnums,
  PrintSets
};

namespace {
  cl::opt<ActionType>
  Action(cl::desc("Action to perform:"),
         cl::values(clEnumValN(PrintRecords, "print-records",
                               "Print all records to stdout (default)"),
                    clEnumValN(GenEmitter, "gen-emitter",
                               "Generate machine code emitter"),
                    clEnumValN(GenRegisterInfo, "gen-register-info",
                               "Generate registers and register classes info"),
                    clEnumValN(GenInstrInfo, "gen-instr-info",
                               "Generate instruction descriptions"),
                    clEnumValN(GenCallingConv, "gen-callingconv",
                               "Generate calling convention descriptions"),
                    clEnumValN(GenAsmWriter, "gen-asm-writer",
                               "Generate assembly writer"),
                    clEnumValN(GenDisassembler, "gen-disassembler",
                               "Generate disassembler"),
                    clEnumValN(GenPseudoLowering, "gen-pseudo-lowering",
                               "Generate pseudo instruction lowering"),
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
                    clEnumValN(GenIntrinsic, "gen-intrinsic",
                               "Generate intrinsic information"),
                    clEnumValN(GenTgtIntrinsic, "gen-tgt-intrinsic",
                               "Generate target intrinsic information"),
                    clEnumValN(GenEDInfo, "gen-enhanced-disassembly-info",
                               "Generate enhanced disassembly info"),
                    clEnumValN(PrintEnums, "print-enums",
                               "Print enum values for a class"),
                    clEnumValN(PrintSets, "print-sets",
                               "Print expanded sets for testing DAG exprs"),
                    clEnumValEnd));

  cl::opt<std::string>
  Class("class", cl::desc("Print Enum list for this class"),
          cl::value_desc("class name"));
  
  class LLVMTableGenAction : public TableGenAction {
  public:
    bool operator()(raw_ostream &OS, RecordKeeper &Records) {
      switch (Action) {
      case PrintRecords:
        OS << Records;           // No argument, dump all contents
        break;
      case GenEmitter:
        CodeEmitterGen(Records).run(OS);
        break;
      case GenRegisterInfo:
        RegisterInfoEmitter(Records).run(OS);
        break;
      case GenInstrInfo:
        InstrInfoEmitter(Records).run(OS);
        break;
      case GenCallingConv:
        CallingConvEmitter(Records).run(OS);
        break;
      case GenAsmWriter:
        AsmWriterEmitter(Records).run(OS);
        break;
      case GenAsmMatcher:
        AsmMatcherEmitter(Records).run(OS);
        break;
      case GenDisassembler:
        DisassemblerEmitter(Records).run(OS);
        break;
      case GenPseudoLowering:
        PseudoLoweringEmitter(Records).run(OS);
        break;
      case GenDAGISel:
        DAGISelEmitter(Records).run(OS);
        break;
      case GenDFAPacketizer:
        DFAGen(Records).run(OS);
        break;
      case GenFastISel:
        FastISelEmitter(Records).run(OS);
        break;
      case GenSubtarget:
        SubtargetEmitter(Records).run(OS);
        break;
      case GenIntrinsic:
        IntrinsicEmitter(Records).run(OS);
        break;
      case GenTgtIntrinsic:
        IntrinsicEmitter(Records, true).run(OS);
        break;
      case GenEDInfo:
        EDEmitter(Records).run(OS);
        break;
      case PrintEnums:
      {
        std::vector<Record*> Recs = Records.getAllDerivedDefinitions(Class);
        for (unsigned i = 0, e = Recs.size(); i != e; ++i)
          OS << Recs[i]->getName() << ", ";
        OS << "\n";
        break;
      }
      case PrintSets:
      {
        SetTheory Sets;
        Sets.addFieldExpander("Set", "Elements");
        std::vector<Record*> Recs = Records.getAllDerivedDefinitions("Set");
        for (unsigned i = 0, e = Recs.size(); i != e; ++i) {
          OS << Recs[i]->getName() << " = [";
          const std::vector<Record*> *Elts = Sets.expand(Recs[i]);
          assert(Elts && "Couldn't expand Set instance");
          for (unsigned ei = 0, ee = Elts->size(); ei != ee; ++ei)
            OS << ' ' << (*Elts)[ei]->getName();
          OS << " ]\n";
        }
        break;
      }
      }
  
      return false;
    }
  };
}

int main(int argc, char **argv) {
  sys::PrintStackTraceOnErrorSignal();
  PrettyStackTraceProgram X(argc, argv);
  cl::ParseCommandLineOptions(argc, argv);

  LLVMTableGenAction Action;
  return TableGenMain(argv[0], Action);
}
