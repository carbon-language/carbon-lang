//===--------------------- BackendStatistics.cpp ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// Functionalities used by the BackendPrinter to print out histograms
/// related to number of {dispatch/issue/retire} per number of cycles.
///
//===----------------------------------------------------------------------===//

#include "BackendStatistics.h"
#include "llvm/Support/Format.h"

using namespace llvm;

namespace mca {

void BackendStatistics::initializeRegisterFileInfo() {
  const MCSchedModel &SM = STI.getSchedModel();
  RegisterFileUsage Empty = {0, 0, 0};
  if (!SM.hasExtraProcessorInfo()) {
    // Assume a single register file.
    RegisterFiles.emplace_back(Empty);
    return;
  }

  // Initialize a RegisterFileUsage for every user defined register file, plus
  // the default register file which is always at index #0.
  const MCExtraProcessorInfo &PI = SM.getExtraProcessorInfo();
  // There is always an "InvalidRegisterFile" entry in tablegen. That entry can
  // be skipped. If there are no user defined register files, then reserve a
  // single entry for the default register file at index #0.
  unsigned NumRegFiles = std::max(PI.NumRegisterFiles, 1U);
  RegisterFiles.resize(NumRegFiles);
  std::fill(RegisterFiles.begin(), RegisterFiles.end(), Empty);
}

void BackendStatistics::onInstructionEvent(const HWInstructionEvent &Event) {
  switch (Event.Type) {
  default:
    break;
  case HWInstructionEvent::Retired: {
    const auto &RE = static_cast<const HWInstructionRetiredEvent &>(Event);
    for (unsigned I = 0, E = RegisterFiles.size(); I < E; ++I)
      RegisterFiles[I].CurrentlyUsedMappings -= RE.FreedPhysRegs[I];

    ++NumRetired;
    break;
  }
  case HWInstructionEvent::Issued:
    ++NumIssued;
    break;
  case HWInstructionEvent::Dispatched: {
    const auto &DE = static_cast<const HWInstructionDispatchedEvent &>(Event);
    for (unsigned I = 0, E = RegisterFiles.size(); I < E; ++I) {
      RegisterFileUsage &RFU = RegisterFiles[I];
      unsigned NumUsedPhysRegs = DE.UsedPhysRegs[I];
      RFU.CurrentlyUsedMappings += NumUsedPhysRegs;
      RFU.TotalMappings += NumUsedPhysRegs;
      RFU.MaxUsedMappings =
          std::max(RFU.MaxUsedMappings, RFU.CurrentlyUsedMappings);
    }

    ++NumDispatched;
  }
  }
}

void BackendStatistics::onReservedBuffers(ArrayRef<unsigned> Buffers) {
  for (const unsigned Buffer : Buffers) {
    if (BufferedResources.find(Buffer) != BufferedResources.end()) {
      BufferUsage &BU = BufferedResources[Buffer];
      BU.SlotsInUse++;
      BU.MaxUsedSlots = std::max(BU.MaxUsedSlots, BU.SlotsInUse);
      continue;
    }

    BufferedResources.insert(
        std::pair<unsigned, BufferUsage>(Buffer, {1U, 1U}));
  }
}

void BackendStatistics::onReleasedBuffers(ArrayRef<unsigned> Buffers) {
  for (const unsigned Buffer : Buffers) {
    assert(BufferedResources.find(Buffer) != BufferedResources.end() &&
           "Buffered resource not in map?");
    BufferUsage &BU = BufferedResources[Buffer];
    BU.SlotsInUse--;
  }
}

void BackendStatistics::printRetireUnitStatistics(llvm::raw_ostream &OS) const {
  std::string Buffer;
  raw_string_ostream TempStream(Buffer);
  TempStream << "\n\nRetire Control Unit - "
             << "number of cycles where we saw N instructions retired:\n";
  TempStream << "[# retired], [# cycles]\n";

  for (const std::pair<unsigned, unsigned> &Entry : RetiredPerCycle) {
    TempStream << " " << Entry.first;
    if (Entry.first < 10)
      TempStream << ",           ";
    else
      TempStream << ",          ";
    TempStream << Entry.second << "  ("
               << format("%.1f", ((double)Entry.second / NumCycles) * 100.0)
               << "%)\n";
  }

  TempStream.flush();
  OS << Buffer;
}

void BackendStatistics::printDispatchUnitStatistics(
    llvm::raw_ostream &OS) const {
  std::string Buffer;
  raw_string_ostream TempStream(Buffer);
  TempStream << "\n\nDispatch Logic - "
             << "number of cycles where we saw N instructions dispatched:\n";
  TempStream << "[# dispatched], [# cycles]\n";
  for (const std::pair<unsigned, unsigned> &Entry : DispatchGroupSizePerCycle) {
    TempStream << " " << Entry.first << ",              " << Entry.second
               << "  ("
               << format("%.1f", ((double)Entry.second / NumCycles) * 100.0)
               << "%)\n";
  }

  TempStream.flush();
  OS << Buffer;
}

void BackendStatistics::printSchedulerStatistics(llvm::raw_ostream &OS) const {
  std::string Buffer;
  raw_string_ostream TempStream(Buffer);
  TempStream << "\n\nSchedulers - number of cycles where we saw N instructions "
                "issued:\n";
  TempStream << "[# issued], [# cycles]\n";
  for (const std::pair<unsigned, unsigned> &Entry : IssuedPerCycle) {
    TempStream << " " << Entry.first << ",          " << Entry.second << "  ("
               << format("%.1f", ((double)Entry.second / NumCycles) * 100)
               << "%)\n";
  }

  TempStream.flush();
  OS << Buffer;
}

void BackendStatistics::printRATStatistics(raw_ostream &OS) const {
  std::string Buffer;
  raw_string_ostream TempStream(Buffer);

  TempStream << "\n\nRegister File statistics:";
  const RegisterFileUsage &GlobalUsage = RegisterFiles[0];
  TempStream << "\nTotal number of mappings created:   "
             << GlobalUsage.TotalMappings;
  TempStream << "\nMax number of mappings used:        "
             << GlobalUsage.MaxUsedMappings << '\n';

  for (unsigned I = 1, E = RegisterFiles.size(); I < E; ++I) {
    const RegisterFileUsage &RFU = RegisterFiles[I];
    // Obtain the register file descriptor from the scheduling model.
    assert(STI.getSchedModel().hasExtraProcessorInfo() &&
           "Unable to find register file info!");
    const MCExtraProcessorInfo &PI =
        STI.getSchedModel().getExtraProcessorInfo();
    assert(I <= PI.NumRegisterFiles && "Unexpected register file index!");
    const MCRegisterFileDesc &RFDesc = PI.RegisterFiles[I];
    // Skip invalid register files.
    if (!RFDesc.NumPhysRegs)
      continue;

    TempStream << "\n*  Register File #" << I;
    TempStream << " -- " << StringRef(RFDesc.Name) << ':';
    TempStream << "\n   Number of physical registers:     ";
    if (!RFDesc.NumPhysRegs)
      TempStream << "unbounded";
    else
      TempStream << RFDesc.NumPhysRegs;
    TempStream << "\n   Total number of mappings created: " << RFU.TotalMappings;
    TempStream << "\n   Max number of mappings used:      "
               << RFU.MaxUsedMappings << '\n';
  }

  TempStream.flush();
  OS << Buffer;
}

void BackendStatistics::printDispatchStalls(raw_ostream &OS) const {
  std::string Buffer;
  raw_string_ostream TempStream(Buffer);
  TempStream << "\n\nDynamic Dispatch Stall Cycles:\n";
  TempStream << "RAT     - Register unavailable:                      "
             << HWStalls[HWStallEvent::RegisterFileStall];
  TempStream << "\nRCU     - Retire tokens unavailable:                 "
             << HWStalls[HWStallEvent::RetireControlUnitStall];
  TempStream << "\nSCHEDQ  - Scheduler full:                            "
             << HWStalls[HWStallEvent::SchedulerQueueFull];
  TempStream << "\nLQ      - Load queue full:                           "
             << HWStalls[HWStallEvent::LoadQueueFull];
  TempStream << "\nSQ      - Store queue full:                          "
             << HWStalls[HWStallEvent::StoreQueueFull];
  TempStream << "\nGROUP   - Static restrictions on the dispatch group: "
             << HWStalls[HWStallEvent::DispatchGroupStall];
  TempStream << '\n';
  TempStream.flush();
  OS << Buffer;
}

void BackendStatistics::printSchedulerUsage(raw_ostream &OS,
                                            const MCSchedModel &SM) const {
  std::string Buffer;
  raw_string_ostream TempStream(Buffer);
  TempStream << "\n\nScheduler's queue usage:\n";
  // Early exit if no buffered resources were consumed.
  if (BufferedResources.empty()) {
    TempStream << "No scheduler resources used.\n";
    TempStream.flush();
    OS << Buffer;
    return;
  }

  for (unsigned I = 0, E = SM.getNumProcResourceKinds(); I < E; ++I) {
    const MCProcResourceDesc &ProcResource = *SM.getProcResource(I);
    if (ProcResource.BufferSize <= 0)
      continue;

    const BufferUsage &BU = BufferedResources.lookup(I);
    TempStream << ProcResource.Name << ",  " << BU.MaxUsedSlots << '/'
               << ProcResource.BufferSize << '\n';
  }

  TempStream.flush();
  OS << Buffer;
}
} // namespace mca
