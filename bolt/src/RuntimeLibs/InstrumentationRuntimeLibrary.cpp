//=  InstrumentationRuntimeLibrary.cpp - The Instrumentation Runtime Library =//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "InstrumentationRuntimeLibrary.h"
#include "BinaryFunction.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"

using namespace llvm;
using namespace bolt;

namespace opts {

extern cl::OptionCategory BoltOptCategory;

extern cl::opt<bool> InstrumentationFileAppendPID;
extern cl::opt<std::string> InstrumentationFilename;
extern cl::opt<uint32_t> InstrumentationSleepTime;

cl::opt<bool>
    Instrument("instrument",
               cl::desc("instrument code to generate accurate profile data"),
               cl::ZeroOrMore, cl::cat(BoltOptCategory));

static cl::opt<std::string> RuntimeInstrumentationLib(
    "runtime-instrumentation-lib",
    cl::desc("specify file name of the runtime instrumentation library"),
    cl::ZeroOrMore, cl::init("libbolt_rt_instr.a"), cl::cat(BoltOptCategory));

} // namespace opts

void InstrumentationRuntimeLibrary::adjustCommandLineOptions(
    const BinaryContext &BC) const {
  if (!BC.HasRelocations) {
    errs() << "BOLT-ERROR: instrumentation runtime libraries require "
              "relocations\n";
    exit(1);
  }
  if (!BC.StartFunctionAddress) {
    errs() << "BOLT-ERROR: instrumentation runtime libraries require a known "
              "entry point of "
              "the input binary\n";
    exit(1);
  }
  if (!BC.FiniFunctionAddress) {
    errs() << "BOLT-ERROR: input binary lacks DT_FINI entry in the dynamic "
              "section but instrumentation currently relies on patching "
              "DT_FINI to write the profile\n";
    exit(1);
  }
}

void InstrumentationRuntimeLibrary::emitBinary(BinaryContext &BC,
                                               MCStreamer &Streamer) {
  const auto *StartFunction =
      BC.getBinaryFunctionAtAddress(*BC.StartFunctionAddress);
  if (!StartFunction) {
    errs() << "BOLT-ERROR: failed to locate function at binary start address\n";
    exit(1);
  }
  const auto *FiniFunction =
      BC.getBinaryFunctionAtAddress(*BC.FiniFunctionAddress);
  if (!FiniFunction) {
    errs() << "BOLT-ERROR: failed to locate function at binary fini address\n";
    exit(1);
  }

  const auto Flags = BinarySection::getFlags(/*IsReadOnly=*/false,
                                             /*IsText=*/false,
                                             /*IsAllocatable=*/true);
  auto *Section =
      BC.Ctx->getELFSection(".bolt.instr.counters", ELF::SHT_PROGBITS, Flags);

  // All of the following symbols will be exported as globals to be used by the
  // instrumentation runtime library to dump the instrumentation data to disk.
  // Label marking start of the memory region containing instrumentation
  // counters, total vector size is Counters.size() 8-byte counters
  MCSymbol *Locs = BC.Ctx->getOrCreateSymbol("__bolt_instr_locations");
  MCSymbol *NumLocs = BC.Ctx->getOrCreateSymbol("__bolt_num_counters");
  MCSymbol *NumIndCalls =
      BC.Ctx->getOrCreateSymbol("__bolt_instr_num_ind_calls");
  MCSymbol *NumIndCallTargets =
      BC.Ctx->getOrCreateSymbol("__bolt_instr_num_ind_targets");
  MCSymbol *NumFuncs = BC.Ctx->getOrCreateSymbol("__bolt_instr_num_funcs");
  /// File name where profile is going to written to after target binary
  /// finishes a run
  MCSymbol *FilenameSym = BC.Ctx->getOrCreateSymbol("__bolt_instr_filename");
  MCSymbol *UsePIDSym = BC.Ctx->getOrCreateSymbol("__bolt_instr_use_pid");
  MCSymbol *InitPtr = BC.Ctx->getOrCreateSymbol("__bolt_instr_init_ptr");
  MCSymbol *FiniPtr = BC.Ctx->getOrCreateSymbol("__bolt_instr_fini_ptr");
  MCSymbol *SleepSym = BC.Ctx->getOrCreateSymbol("__bolt_instr_sleep_time");

  Section->setAlignment(BC.RegularPageSize);
  Streamer.SwitchSection(Section);
  Streamer.EmitLabel(Locs);
  Streamer.EmitSymbolAttribute(Locs, MCSymbolAttr::MCSA_Global);
  for (const auto &Label : Summary->Counters) {
    Streamer.EmitLabel(Label);
    Streamer.emitFill(8, 0);
  }
  const uint64_t Padding =
      alignTo(8 * Summary->Counters.size(), BC.RegularPageSize) -
      8 * Summary->Counters.size();
  if (Padding)
    Streamer.emitFill(Padding, 0);
  Streamer.EmitLabel(SleepSym);
  Streamer.EmitSymbolAttribute(SleepSym, MCSymbolAttr::MCSA_Global);
  Streamer.EmitIntValue(opts::InstrumentationSleepTime, /*Size=*/4);
  Streamer.EmitLabel(NumLocs);
  Streamer.EmitSymbolAttribute(NumLocs, MCSymbolAttr::MCSA_Global);
  Streamer.EmitIntValue(Summary->Counters.size(), /*Size=*/4);
  Streamer.EmitLabel(Summary->IndCallHandlerFunc);
  Streamer.EmitSymbolAttribute(Summary->IndCallHandlerFunc,
                               MCSymbolAttr::MCSA_Global);
  Streamer.EmitValue(
      MCSymbolRefExpr::create(
          Summary->InitialIndCallHandlerFunction->getSymbol(), *BC.Ctx),
      /*Size=*/8);
  Streamer.EmitLabel(Summary->IndTailCallHandlerFunc);
  Streamer.EmitSymbolAttribute(Summary->IndTailCallHandlerFunc,
                               MCSymbolAttr::MCSA_Global);
  Streamer.EmitValue(
      MCSymbolRefExpr::create(
          Summary->InitialIndTailCallHandlerFunction->getSymbol(), *BC.Ctx),
      /*Size=*/8);
  Streamer.EmitLabel(NumIndCalls);
  Streamer.EmitSymbolAttribute(NumIndCalls, MCSymbolAttr::MCSA_Global);
  Streamer.EmitIntValue(Summary->IndCallDescriptions.size(), /*Size=*/4);
  Streamer.EmitLabel(NumIndCallTargets);
  Streamer.EmitSymbolAttribute(NumIndCallTargets, MCSymbolAttr::MCSA_Global);
  Streamer.EmitIntValue(Summary->IndCallTargetDescriptions.size(), /*Size=*/4);
  Streamer.EmitLabel(NumFuncs);
  Streamer.EmitSymbolAttribute(NumFuncs, MCSymbolAttr::MCSA_Global);
  Streamer.EmitIntValue(Summary->FunctionDescriptions.size(), /*Size=*/4);
  Streamer.EmitLabel(FilenameSym);
  Streamer.EmitBytes(opts::InstrumentationFilename);
  Streamer.emitFill(1, 0);
  Streamer.EmitLabel(UsePIDSym);
  Streamer.EmitIntValue(opts::InstrumentationFileAppendPID ? 1 : 0, /*Size=*/1);

  Streamer.EmitLabel(InitPtr);
  Streamer.EmitSymbolAttribute(InitPtr, MCSymbolAttr::MCSA_Global);
  Streamer.EmitValue(
      MCSymbolRefExpr::create(StartFunction->getSymbol(), *BC.Ctx), /*Size=*/8);
  Streamer.EmitLabel(FiniPtr);
  Streamer.EmitSymbolAttribute(FiniPtr, MCSymbolAttr::MCSA_Global);
  Streamer.EmitValue(
      MCSymbolRefExpr::create(FiniFunction->getSymbol(), *BC.Ctx), /*Size=*/8);
}

void InstrumentationRuntimeLibrary::link(BinaryContext &BC, StringRef ToolPath,
                                         orc::ExecutionSession &ES,
                                         orc::RTDyldObjectLinkingLayer &OLT) {
  auto LibPath = getLibPath(ToolPath, opts::RuntimeInstrumentationLib);
  loadLibraryToOLT(LibPath, ES, OLT);

  RuntimeFiniAddress =
      cantFail(OLT.findSymbol("__bolt_instr_fini", false).getAddress());
  if (!RuntimeFiniAddress) {
    errs() << "BOLT-ERROR: instrumentation library does not define "
              "__bolt_instr_fini: "
           << LibPath << "\n";
    exit(1);
  }
  RuntimeStartAddress =
      cantFail(OLT.findSymbol("__bolt_instr_start", false).getAddress());
  if (!RuntimeStartAddress) {
    errs() << "BOLT-ERROR: instrumentation library does not define "
              "__bolt_instr_start: "
           << LibPath << "\n";
    exit(1);
  }
  outs() << "BOLT-INFO: output linked against instrumentation runtime "
            "library, lib entry point is 0x"
         << Twine::utohexstr(RuntimeFiniAddress) << "\n";
  outs()
      << "BOLT-INFO: clear procedure is 0x"
      << Twine::utohexstr(cantFail(
             OLT.findSymbol("__bolt_instr_clear_counters", false).getAddress()))
      << "\n";

  emitTablesAsELFNote(BC);
}

void InstrumentationRuntimeLibrary::emitTablesAsELFNote(BinaryContext &BC) {
  std::string TablesStr;
  raw_string_ostream OS(TablesStr);
  // This is sync'ed with runtime/instr.cpp:readDescriptions()

  auto getOutputAddress = [](const BinaryFunction &Func,
                             uint64_t Offset) -> uint64_t {
    return Offset == 0
               ? Func.getOutputAddress()
               : Func.translateInputToOutputAddress(Func.getAddress() + Offset);
  };

  // Indirect targets need to be sorted for fast lookup during runtime
  std::sort(Summary->IndCallTargetDescriptions.begin(),
            Summary->IndCallTargetDescriptions.end(),
            [&](const IndCallTargetDescription &A,
                const IndCallTargetDescription &B) {
              return getOutputAddress(*A.Target, A.ToLoc.Offset) <
                     getOutputAddress(*B.Target, B.ToLoc.Offset);
            });

  // Start of the vector with descriptions (one CounterDescription for each
  // counter), vector size is Counters.size() CounterDescription-sized elmts
  const auto IDSize =
      Summary->IndCallDescriptions.size() * sizeof(IndCallDescription);
  OS.write(reinterpret_cast<const char *>(&IDSize), 4);
  for (const auto &Desc : Summary->IndCallDescriptions) {
    OS.write(reinterpret_cast<const char *>(&Desc.FromLoc.FuncString), 4);
    OS.write(reinterpret_cast<const char *>(&Desc.FromLoc.Offset), 4);
  }
  const auto ITDSize = Summary->IndCallTargetDescriptions.size() *
                       sizeof(IndCallTargetDescription);
  OS.write(reinterpret_cast<const char *>(&ITDSize), 4);
  for (const auto &Desc : Summary->IndCallTargetDescriptions) {
    OS.write(reinterpret_cast<const char *>(&Desc.ToLoc.FuncString), 4);
    OS.write(reinterpret_cast<const char *>(&Desc.ToLoc.Offset), 4);
    uint64_t TargetFuncAddress =
        getOutputAddress(*Desc.Target, Desc.ToLoc.Offset);
    OS.write(reinterpret_cast<const char *>(&TargetFuncAddress), 8);
  }
  auto FuncDescSize = Summary->getFDSize();
  OS.write(reinterpret_cast<const char *>(&FuncDescSize), 4);
  for (const auto &Desc : Summary->FunctionDescriptions) {
    const auto LeafNum = Desc.LeafNodes.size();
    OS.write(reinterpret_cast<const char *>(&LeafNum), 4);
    for (const auto &LeafNode : Desc.LeafNodes) {
      OS.write(reinterpret_cast<const char *>(&LeafNode.Node), 4);
      OS.write(reinterpret_cast<const char *>(&LeafNode.Counter), 4);
    }
    const auto EdgesNum = Desc.Edges.size();
    OS.write(reinterpret_cast<const char *>(&EdgesNum), 4);
    for (const auto &Edge : Desc.Edges) {
      OS.write(reinterpret_cast<const char *>(&Edge.FromLoc.FuncString), 4);
      OS.write(reinterpret_cast<const char *>(&Edge.FromLoc.Offset), 4);
      OS.write(reinterpret_cast<const char *>(&Edge.FromNode), 4);
      OS.write(reinterpret_cast<const char *>(&Edge.ToLoc.FuncString), 4);
      OS.write(reinterpret_cast<const char *>(&Edge.ToLoc.Offset), 4);
      OS.write(reinterpret_cast<const char *>(&Edge.ToNode), 4);
      OS.write(reinterpret_cast<const char *>(&Edge.Counter), 4);
    }
    const auto CallsNum = Desc.Calls.size();
    OS.write(reinterpret_cast<const char *>(&CallsNum), 4);
    for (const auto &Call : Desc.Calls) {
      OS.write(reinterpret_cast<const char *>(&Call.FromLoc.FuncString), 4);
      OS.write(reinterpret_cast<const char *>(&Call.FromLoc.Offset), 4);
      OS.write(reinterpret_cast<const char *>(&Call.FromNode), 4);
      OS.write(reinterpret_cast<const char *>(&Call.ToLoc.FuncString), 4);
      OS.write(reinterpret_cast<const char *>(&Call.ToLoc.Offset), 4);
      OS.write(reinterpret_cast<const char *>(&Call.Counter), 4);
      uint64_t TargetFuncAddress =
          getOutputAddress(*Call.Target, Call.ToLoc.Offset);
      OS.write(reinterpret_cast<const char *>(&TargetFuncAddress), 8);
    }
    const auto EntryNum = Desc.EntryNodes.size();
    OS.write(reinterpret_cast<const char *>(&EntryNum), 4);
    for (const auto &EntryNode : Desc.EntryNodes) {
      OS.write(reinterpret_cast<const char *>(&EntryNode.Node), 8);
      uint64_t TargetFuncAddress =
          getOutputAddress(*Desc.Function, EntryNode.Address);
      OS.write(reinterpret_cast<const char *>(&TargetFuncAddress), 8);
    }
  }
  // Our string table lives immediately after descriptions vector
  OS << Summary->StringTable;
  OS.flush();
  const auto BoltInfo = BinarySection::encodeELFNote(
      "BOLT", TablesStr, BinarySection::NT_BOLT_INSTRUMENTATION_TABLES);
  BC.registerOrUpdateNoteSection(".bolt.instr.tables", copyByteArray(BoltInfo),
                                 BoltInfo.size(),
                                 /*Alignment=*/1,
                                 /*IsReadOnly=*/true, ELF::SHT_NOTE);
}
