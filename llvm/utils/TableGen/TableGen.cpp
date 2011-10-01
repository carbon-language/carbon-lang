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
#include "ClangASTNodesEmitter.h"
#include "ClangAttrEmitter.h"
#include "ClangDiagnosticsEmitter.h"
#include "ClangSACheckersEmitter.h"
#include "CodeEmitterGen.h"
#include "DAGISelEmitter.h"
#include "DisassemblerEmitter.h"
#include "EDEmitter.h"
#include "FastISelEmitter.h"
#include "InstrInfoEmitter.h"
#include "IntrinsicEmitter.h"
#include "NeonEmitter.h"
#include "OptParserEmitter.h"
#include "PseudoLoweringEmitter.h"
#include "RegisterInfoEmitter.h"
#include "ARMDecoderEmitter.h"
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
  GenARMDecoder,
  GenDisassembler,
  GenPseudoLowering,
  GenCallingConv,
  GenClangAttrClasses,
  GenClangAttrImpl,
  GenClangAttrList,
  GenClangAttrPCHRead,
  GenClangAttrPCHWrite,
  GenClangAttrSpellingList,
  GenClangAttrLateParsedList,
  GenClangDiagsDefs,
  GenClangDiagGroups,
  GenClangDiagsIndexName,
  GenClangDeclNodes,
  GenClangStmtNodes,
  GenClangSACheckers,
  GenDAGISel,
  GenFastISel,
  GenOptParserDefs, GenOptParserImpl,
  GenSubtarget,
  GenIntrinsic,
  GenTgtIntrinsic,
  GenEDInfo,
  GenArmNeon,
  GenArmNeonSema,
  GenArmNeonTest,
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
                    clEnumValN(GenARMDecoder, "gen-arm-decoder",
                               "Generate decoders for ARM/Thumb"),
                    clEnumValN(GenDisassembler, "gen-disassembler",
                               "Generate disassembler"),
                    clEnumValN(GenPseudoLowering, "gen-pseudo-lowering",
                               "Generate pseudo instruction lowering"),
                    clEnumValN(GenAsmMatcher, "gen-asm-matcher",
                               "Generate assembly instruction matcher"),
                    clEnumValN(GenDAGISel, "gen-dag-isel",
                               "Generate a DAG instruction selector"),
                    clEnumValN(GenFastISel, "gen-fast-isel",
                               "Generate a \"fast\" instruction selector"),
                    clEnumValN(GenOptParserDefs, "gen-opt-parser-defs",
                               "Generate option definitions"),
                    clEnumValN(GenOptParserImpl, "gen-opt-parser-impl",
                               "Generate option parser implementation"),
                    clEnumValN(GenSubtarget, "gen-subtarget",
                               "Generate subtarget enumerations"),
                    clEnumValN(GenIntrinsic, "gen-intrinsic",
                               "Generate intrinsic information"),
                    clEnumValN(GenTgtIntrinsic, "gen-tgt-intrinsic",
                               "Generate target intrinsic information"),
                    clEnumValN(GenClangAttrClasses, "gen-clang-attr-classes",
                               "Generate clang attribute clases"),
                    clEnumValN(GenClangAttrImpl, "gen-clang-attr-impl",
                               "Generate clang attribute implementations"),
                    clEnumValN(GenClangAttrList, "gen-clang-attr-list",
                               "Generate a clang attribute list"),
                    clEnumValN(GenClangAttrPCHRead, "gen-clang-attr-pch-read",
                               "Generate clang PCH attribute reader"),
                    clEnumValN(GenClangAttrPCHWrite, "gen-clang-attr-pch-write",
                               "Generate clang PCH attribute writer"),
                    clEnumValN(GenClangAttrSpellingList,
                               "gen-clang-attr-spelling-list",
                               "Generate a clang attribute spelling list"),
                    clEnumValN(GenClangAttrLateParsedList,
                               "gen-clang-attr-late-parsed-list",
                               "Generate a clang attribute LateParsed list"),
                    clEnumValN(GenClangDiagsDefs, "gen-clang-diags-defs",
                               "Generate Clang diagnostics definitions"),
                    clEnumValN(GenClangDiagGroups, "gen-clang-diag-groups",
                               "Generate Clang diagnostic groups"),
                    clEnumValN(GenClangDiagsIndexName,
                               "gen-clang-diags-index-name",
                               "Generate Clang diagnostic name index"),
                    clEnumValN(GenClangDeclNodes, "gen-clang-decl-nodes",
                               "Generate Clang AST declaration nodes"),
                    clEnumValN(GenClangStmtNodes, "gen-clang-stmt-nodes",
                               "Generate Clang AST statement nodes"),
                    clEnumValN(GenClangSACheckers, "gen-clang-sa-checkers",
                               "Generate Clang Static Analyzer checkers"),
                    clEnumValN(GenEDInfo, "gen-enhanced-disassembly-info",
                               "Generate enhanced disassembly info"),
                    clEnumValN(GenArmNeon, "gen-arm-neon",
                               "Generate arm_neon.h for clang"),
                    clEnumValN(GenArmNeonSema, "gen-arm-neon-sema",
                               "Generate ARM NEON sema support for clang"),
                    clEnumValN(GenArmNeonTest, "gen-arm-neon-test",
                               "Generate ARM NEON tests for clang"),
                    clEnumValN(PrintEnums, "print-enums",
                               "Print enum values for a class"),
                    clEnumValN(PrintSets, "print-sets",
                               "Print expanded sets for testing DAG exprs"),
                    clEnumValEnd));

  cl::opt<std::string>
  Class("class", cl::desc("Print Enum list for this class"),
        cl::value_desc("class name"));

  cl::opt<std::string>
  ClangComponent("clang-component",
                 cl::desc("Only use warnings from specified component"),
                 cl::value_desc("component"), cl::Hidden);
}

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
    case GenARMDecoder:
      ARMDecoderEmitter(Records).run(OS);
      break;
    case GenAsmMatcher:
      AsmMatcherEmitter(Records).run(OS);
      break;
    case GenClangAttrClasses:
      ClangAttrClassEmitter(Records).run(OS);
      break;
    case GenClangAttrImpl:
      ClangAttrImplEmitter(Records).run(OS);
      break;
    case GenClangAttrList:
      ClangAttrListEmitter(Records).run(OS);
      break;
    case GenClangAttrPCHRead:
      ClangAttrPCHReadEmitter(Records).run(OS);
      break;
    case GenClangAttrPCHWrite:
      ClangAttrPCHWriteEmitter(Records).run(OS);
      break;
    case GenClangAttrSpellingList:
      ClangAttrSpellingListEmitter(Records).run(OS);
      break;
    case GenClangAttrLateParsedList:
      ClangAttrLateParsedListEmitter(Records).run(OS);
      break;
    case GenClangDiagsDefs:
      ClangDiagsDefsEmitter(Records, ClangComponent).run(OS);
      break;
    case GenClangDiagGroups:
      ClangDiagGroupsEmitter(Records).run(OS);
      break;
    case GenClangDiagsIndexName:
      ClangDiagsIndexNameEmitter(Records).run(OS);
      break;
    case GenClangDeclNodes:
      ClangASTNodesEmitter(Records, "Decl", "Decl").run(OS);
      ClangDeclContextEmitter(Records).run(OS);
      break;
    case GenClangStmtNodes:
      ClangASTNodesEmitter(Records, "Stmt", "").run(OS);
      break;
    case GenClangSACheckers:
      ClangSACheckersEmitter(Records).run(OS);
      break;
    case GenDisassembler:
      DisassemblerEmitter(Records).run(OS);
      break;
    case GenPseudoLowering:
      PseudoLoweringEmitter(Records).run(OS);
      break;
    case GenOptParserDefs:
      OptParserEmitter(Records, true).run(OS);
      break;
    case GenOptParserImpl:
      OptParserEmitter(Records, false).run(OS);
      break;
    case GenDAGISel:
      DAGISelEmitter(Records).run(OS);
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
    case GenArmNeon:
      NeonEmitter(Records).run(OS);
      break;
    case GenArmNeonSema:
      NeonEmitter(Records).runHeader(OS);
      break;
    case GenArmNeonTest:
      NeonEmitter(Records).runTests(OS);
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
    default:
      assert(1 && "Invalid Action");
      return true;
    }

    return false;
  }
};

int main(int argc, char **argv) {
  sys::PrintStackTraceOnErrorSignal();
  PrettyStackTraceProgram X(argc, argv);
  cl::ParseCommandLineOptions(argc, argv);

  LLVMTableGenAction Action;
  return TableGenMain(argv[0], Action);
}
