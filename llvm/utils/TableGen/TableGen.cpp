//===- TableGen.cpp - Top-Level TableGen implementation -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// TableGen is a tool which can be used to build up a description of something,
// then invoke one or more "tablegen backends" to emit information about the
// description in some predefined format.  In practice, this is used by the LLVM
// code generators to automate generation of a code generator through a
// high-level description of the target.
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
#include "Error.h"
#include "FastISelEmitter.h"
#include "InstrInfoEmitter.h"
#include "IntrinsicEmitter.h"
#include "LLVMCConfigurationEmitter.h"
#include "NeonEmitter.h"
#include "OptParserEmitter.h"
#include "PseudoLoweringEmitter.h"
#include "Record.h"
#include "RegisterInfoEmitter.h"
#include "ARMDecoderEmitter.h"
#include "SubtargetEmitter.h"
#include "SetTheory.h"
#include "TGParser.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/system_error.h"
#include <algorithm>
#include <cstdio>
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
  GenLLVMCConf,
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
                    clEnumValN(GenLLVMCConf, "gen-llvmc",
                               "Generate LLVMC configuration library"),
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
  OutputFilename("o", cl::desc("Output filename"), cl::value_desc("filename"),
                 cl::init("-"));

  cl::opt<std::string>
  DependFilename("d", cl::desc("Dependency filename"), cl::value_desc("filename"),
                 cl::init(""));

  cl::opt<std::string>
  InputFilename(cl::Positional, cl::desc("<input file>"), cl::init("-"));

  cl::list<std::string>
  IncludeDirs("I", cl::desc("Directory of include files"),
              cl::value_desc("directory"), cl::Prefix);

  cl::opt<std::string>
  ClangComponent("clang-component",
                 cl::desc("Only use warnings from specified component"),
                 cl::value_desc("component"), cl::Hidden);
}


int main(int argc, char **argv) {
  RecordKeeper Records;

  sys::PrintStackTraceOnErrorSignal();
  PrettyStackTraceProgram X(argc, argv);
  cl::ParseCommandLineOptions(argc, argv);


  try {
    // Parse the input file.
    OwningPtr<MemoryBuffer> File;
    if (error_code ec = MemoryBuffer::getFileOrSTDIN(InputFilename.c_str(), File)) {
      errs() << "Could not open input file '" << InputFilename << "': "
             << ec.message() <<"\n";
      Init::ReleaseMemory();
      return 1;
    }
    MemoryBuffer *F = File.take();

    // Tell SrcMgr about this buffer, which is what TGParser will pick up.
    SrcMgr.AddNewSourceBuffer(F, SMLoc());

    // Record the location of the include directory so that the lexer can find
    // it later.
    SrcMgr.setIncludeDirs(IncludeDirs);

    TGParser Parser(SrcMgr, Records);

    if (Parser.ParseFile()) {
      Init::ReleaseMemory();
      return 1;
    }

    std::string Error;
    tool_output_file Out(OutputFilename.c_str(), Error);
    if (!Error.empty()) {
      errs() << argv[0] << ": error opening " << OutputFilename
        << ":" << Error << "\n";
      Init::ReleaseMemory();
      return 1;
    }
    if (!DependFilename.empty()) {
      if (OutputFilename == "-") {
        errs() << argv[0] << ": the option -d must be used together with -o\n";
        Init::ReleaseMemory();
        return 1;
      }
      tool_output_file DepOut(DependFilename.c_str(), Error);
      if (!Error.empty()) {
        errs() << argv[0] << ": error opening " << DependFilename
          << ":" << Error << "\n";
        Init::ReleaseMemory();
        return 1;
      }
      DepOut.os() << DependFilename << ":";
      const std::vector<std::string> &Dependencies = Parser.getDependencies();
      for (std::vector<std::string>::const_iterator I = Dependencies.begin(),
                                                          E = Dependencies.end();
           I != E; ++I) {
        DepOut.os() << " " << (*I);
      }
      DepOut.os() << "\n";
      DepOut.keep();
    }

    switch (Action) {
    case PrintRecords:
      Out.os() << Records;           // No argument, dump all contents
      break;
    case GenEmitter:
      CodeEmitterGen(Records).run(Out.os());
      break;
    case GenRegisterInfo:
      RegisterInfoEmitter(Records).run(Out.os());
      break;
    case GenInstrInfo:
      InstrInfoEmitter(Records).run(Out.os());
      break;
    case GenCallingConv:
      CallingConvEmitter(Records).run(Out.os());
      break;
    case GenAsmWriter:
      AsmWriterEmitter(Records).run(Out.os());
      break;
    case GenARMDecoder:
      ARMDecoderEmitter(Records).run(Out.os());
      break;
    case GenAsmMatcher:
      AsmMatcherEmitter(Records).run(Out.os());
      break;
    case GenClangAttrClasses:
      ClangAttrClassEmitter(Records).run(Out.os());
      break;
    case GenClangAttrImpl:
      ClangAttrImplEmitter(Records).run(Out.os());
      break;
    case GenClangAttrList:
      ClangAttrListEmitter(Records).run(Out.os());
      break;
    case GenClangAttrPCHRead:
      ClangAttrPCHReadEmitter(Records).run(Out.os());
      break;
    case GenClangAttrPCHWrite:
      ClangAttrPCHWriteEmitter(Records).run(Out.os());
      break;
    case GenClangAttrSpellingList:
      ClangAttrSpellingListEmitter(Records).run(Out.os());
      break;
    case GenClangDiagsDefs:
      ClangDiagsDefsEmitter(Records, ClangComponent).run(Out.os());
      break;
    case GenClangDiagGroups:
      ClangDiagGroupsEmitter(Records).run(Out.os());
      break;
    case GenClangDiagsIndexName:
      ClangDiagsIndexNameEmitter(Records).run(Out.os());
      break;
    case GenClangDeclNodes:
      ClangASTNodesEmitter(Records, "Decl", "Decl").run(Out.os());
      ClangDeclContextEmitter(Records).run(Out.os());
      break;
    case GenClangStmtNodes:
      ClangASTNodesEmitter(Records, "Stmt", "").run(Out.os());
      break;
    case GenClangSACheckers:
      ClangSACheckersEmitter(Records).run(Out.os());
      break;
    case GenDisassembler:
      DisassemblerEmitter(Records).run(Out.os());
      break;
    case GenPseudoLowering:
      PseudoLoweringEmitter(Records).run(Out.os());
      break;
    case GenOptParserDefs:
      OptParserEmitter(Records, true).run(Out.os());
      break;
    case GenOptParserImpl:
      OptParserEmitter(Records, false).run(Out.os());
      break;
    case GenDAGISel:
      DAGISelEmitter(Records).run(Out.os());
      break;
    case GenFastISel:
      FastISelEmitter(Records).run(Out.os());
      break;
    case GenSubtarget:
      SubtargetEmitter(Records).run(Out.os());
      break;
    case GenIntrinsic:
      IntrinsicEmitter(Records).run(Out.os());
      break;
    case GenTgtIntrinsic:
      IntrinsicEmitter(Records, true).run(Out.os());
      break;
    case GenLLVMCConf:
      LLVMCConfigurationEmitter(Records).run(Out.os());
      break;
    case GenEDInfo:
      EDEmitter(Records).run(Out.os());
      break;
    case GenArmNeon:
      NeonEmitter(Records).run(Out.os());
      break;
    case GenArmNeonSema:
      NeonEmitter(Records).runHeader(Out.os());
      break;
    case GenArmNeonTest:
      NeonEmitter(Records).runTests(Out.os());
      break;
    case PrintEnums:
    {
      std::vector<Record*> Recs = Records.getAllDerivedDefinitions(Class);
      for (unsigned i = 0, e = Recs.size(); i != e; ++i)
        Out.os() << Recs[i]->getName() << ", ";
      Out.os() << "\n";
      break;
    }
    case PrintSets:
    {
      SetTheory Sets;
      Sets.addFieldExpander("Set", "Elements");
      std::vector<Record*> Recs = Records.getAllDerivedDefinitions("Set");
      for (unsigned i = 0, e = Recs.size(); i != e; ++i) {
        Out.os() << Recs[i]->getName() << " = [";
        const std::vector<Record*> *Elts = Sets.expand(Recs[i]);
        assert(Elts && "Couldn't expand Set instance");
        for (unsigned ei = 0, ee = Elts->size(); ei != ee; ++ei)
          Out.os() << ' ' << (*Elts)[ei]->getName();
        Out.os() << " ]\n";
      }
      break;
    }
    default:
      assert(1 && "Invalid Action");
      Init::ReleaseMemory();
      return 1;
    }

    // Declare success.
    Out.keep();

    Init::ReleaseMemory();
    return 0;

  } catch (const TGError &Error) {
    PrintError(Error);
  } catch (const std::string &Error) {
    PrintError(Error);
  } catch (const char *Error) {
    PrintError(Error);
  } catch (...) {
    errs() << argv[0] << ": Unknown unexpected exception occurred.\n";
  }

  Init::ReleaseMemory();

  return 1;
}
