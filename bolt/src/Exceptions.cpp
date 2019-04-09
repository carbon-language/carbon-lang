//===-- Exceptions.cpp - Helpers for processing C++ exceptions ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Some of the code is taken from examples/ExceptionDemo
//
//===----------------------------------------------------------------------===//

#include "Exceptions.h"
#include "BinaryFunction.h"
#include "RewriteInstance.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugFrame.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <functional>
#include <map>

#undef  DEBUG_TYPE
#define DEBUG_TYPE "bolt-exceptions"

using namespace llvm::dwarf;

namespace opts {

extern llvm::cl::OptionCategory BoltCategory;

extern llvm::cl::opt<unsigned> Verbosity;

static llvm::cl::opt<bool>
PrintExceptions("print-exceptions",
  llvm::cl::desc("print exception handling data"),
  llvm::cl::ZeroOrMore,
  llvm::cl::Hidden,
  llvm::cl::cat(BoltCategory));

} // namespace opts

namespace llvm {
namespace bolt {

namespace {

unsigned getEncodingSize(unsigned Encoding, BinaryContext &BC) {
  switch (Encoding & 0x0f) {
  default: llvm_unreachable("unknown encoding");
  case dwarf::DW_EH_PE_absptr:
  case dwarf::DW_EH_PE_signed:
    return BC.AsmInfo->getCodePointerSize();
  case dwarf::DW_EH_PE_udata2:
  case dwarf::DW_EH_PE_sdata2:
    return 2;
  case dwarf::DW_EH_PE_udata4:
  case dwarf::DW_EH_PE_sdata4:
    return 4;
  case dwarf::DW_EH_PE_udata8:
  case dwarf::DW_EH_PE_sdata8:
    return 8;
  }
}

} // anonymous namespace

// Read and dump the .gcc_exception_table section entry.
//
// .gcc_except_table section contains a set of Language-Specific Data Areas -
// a fancy name for exception handling tables. There's one  LSDA entry per
// function. However, we can't actually tell which function LSDA refers to
// unless we parse .eh_frame entry that refers to the LSDA.
// Then inside LSDA most addresses are encoded relative to the function start,
// so we need the function context in order to get to real addresses.
//
// The best visual representation of the tables comprising LSDA and
// relationships between them is illustrated at:
//   https://github.com/itanium-cxx-abi/cxx-abi/blob/master/exceptions.pdf
// Keep in mind that GCC implementation deviates slightly from that document.
//
// To summarize, there are 4 tables in LSDA: call site table, actions table,
// types table, and types index table (for indirection). The main table contains
// call site entries. Each call site includes a PC range that can throw an
// exception, a handler (landing pad), and a reference to an entry in the action
// table. The handler and/or action could be 0. The action entry is a head
// of a list of actions associated with a call site. The action table contains
// all such lists (it could be optimized to share list tails). Each action could
// be either to catch an exception of a given type, to perform a cleanup, or to
// propagate the exception after filtering it out (e.g. to make sure function
// exception specification is not violated). Catch action contains a reference
// to an entry in the type table, and filter action refers to an entry in the
// type index table to encode a set of types to filter.
//
// Call site table follows LSDA header. Action table immediately follows the
// call site table.
//
// Both types table and type index table start at the same location, but they
// grow in opposite directions (types go up, indices go down). The beginning of
// these tables is encoded in LSDA header. Sizes for both of the tables are not
// included anywhere.
//
// We have to parse all of the tables to determine their sizes. Then we have
// to parse the call site table and associate discovered information with
// actual call instructions and landing pad blocks.
//
// For the purpose of rewriting exception handling tables, we can reuse action,
// and type index tables in their original binary format.
//
// Type table could be encoded using position-independent references, and thus
// may require relocation.
//
// Ideally we should be able to re-write LSDA in-place, without the need to
// allocate a new space for it. Sadly there's no guarantee that the new call
// site table will be the same size as GCC uses uleb encodings for PC offsets.
//
// Note: some functions have LSDA entries with 0 call site entries.
void BinaryFunction::parseLSDA(ArrayRef<uint8_t> LSDASectionData,
                               uint64_t LSDASectionAddress) {
  assert(CurrentState == State::Disassembled && "unexpected function state");

  if (!getLSDAAddress())
    return;

  DWARFDataExtractor Data(
      StringRef(reinterpret_cast<const char *>(LSDASectionData.data()),
                LSDASectionData.size()),
      BC.DwCtx->getDWARFObj().isLittleEndian(), 8);
  uint32_t Offset = getLSDAAddress() - LSDASectionAddress;
  assert(Data.isValidOffset(Offset) && "wrong LSDA address");

  uint8_t LPStartEncoding = Data.getU8(&Offset);
  uint64_t LPStart = 0;
  if (auto MaybeLPStart = Data.getEncodedPointer(&Offset, LPStartEncoding,
                                                 Offset + LSDASectionAddress))
    LPStart = *MaybeLPStart;

  assert(LPStart == 0 && "support for split functions not implemented");

  const auto TTypeEncoding = Data.getU8(&Offset);
  size_t TTypeEncodingSize = 0;
  uintptr_t TTypeEnd = 0;
  if (TTypeEncoding != DW_EH_PE_omit) {
    TTypeEnd = Data.getULEB128(&Offset);
    TTypeEncodingSize = getEncodingSize(TTypeEncoding, BC);
  }

  if (opts::PrintExceptions) {
    outs() << "[LSDA at 0x" << Twine::utohexstr(getLSDAAddress())
           << " for function " << *this << "]:\n";
    outs() << "LPStart Encoding = 0x"
           << Twine::utohexstr(LPStartEncoding) << '\n';
    outs() << "LPStart = 0x" << Twine::utohexstr(LPStart) << '\n';
    outs() << "TType Encoding = 0x" << Twine::utohexstr(TTypeEncoding) << '\n';
    outs() << "TType End = " << TTypeEnd << '\n';
  }

  // Table to store list of indices in type table. Entries are uleb128 values.
  const uint32_t TypeIndexTableStart = Offset + TTypeEnd;

  // Offset past the last decoded index.
  uint32_t MaxTypeIndexTableOffset = 0;

  // Max positive index used in type table.
  unsigned MaxTypeIndex = 0;

  // The actual type info table starts at the same location, but grows in
  // opposite direction. TTypeEncoding is used to encode stored values.
  const auto TypeTableStart = Offset + TTypeEnd;

  uint8_t CallSiteEncoding = Data.getU8(&Offset);
  uint32_t CallSiteTableLength = Data.getULEB128(&Offset);
  auto CallSiteTableStart = Offset;
  auto CallSiteTableEnd = CallSiteTableStart + CallSiteTableLength;
  auto CallSitePtr = CallSiteTableStart;
  auto ActionTableStart = CallSiteTableEnd;

  if (opts::PrintExceptions) {
    outs() << "CallSite Encoding = " << (unsigned)CallSiteEncoding << '\n';
    outs() << "CallSite table length = " << CallSiteTableLength << '\n';
    outs() << '\n';
  }

  HasEHRanges = CallSitePtr < CallSiteTableEnd;
  uint64_t RangeBase = getAddress();
  while (CallSitePtr < CallSiteTableEnd) {
    uint64_t Start = *Data.getEncodedPointer(&CallSitePtr, CallSiteEncoding,
                                              CallSitePtr + LSDASectionAddress);
    uint64_t Length = *Data.getEncodedPointer(
        &CallSitePtr, CallSiteEncoding, CallSitePtr + LSDASectionAddress);
    uint64_t LandingPad = *Data.getEncodedPointer(
        &CallSitePtr, CallSiteEncoding, CallSitePtr + LSDASectionAddress);
    uint64_t ActionEntry = Data.getULEB128(&CallSitePtr);

    if (opts::PrintExceptions) {
      outs() << "Call Site: [0x" << Twine::utohexstr(RangeBase + Start)
             << ", 0x" << Twine::utohexstr(RangeBase + Start + Length)
             << "); landing pad: 0x" << Twine::utohexstr(LPStart + LandingPad)
             << "; action entry: 0x" << Twine::utohexstr(ActionEntry) << "\n";
      outs() << "  current offset is " << (CallSitePtr - CallSiteTableStart)
             << '\n';
    }

    // Create a handler entry if necessary.
    MCSymbol *LPSymbol{nullptr};
    if (LandingPad) {
      if (Instructions.find(LandingPad) == Instructions.end()) {
        if (opts::Verbosity >= 1) {
          errs() << "BOLT-WARNING: landing pad " << Twine::utohexstr(LandingPad)
                 << " not pointing to an instruction in function "
                 << *this << " - ignoring.\n";
        }
      } else {
        auto Label = Labels.find(LandingPad);
        if (Label != Labels.end()) {
          LPSymbol = Label->second;
        } else {
          LPSymbol = BC.Ctx->createTempSymbol("LP", true);
          Labels[LandingPad] = LPSymbol;
        }
      }
    }

    // Mark all call instructions in the range.
    auto II = Instructions.find(Start);
    auto IE = Instructions.end();
    assert(II != IE && "exception range not pointing to an instruction");
    do {
      auto &Instruction = II->second;
      if (BC.MIB->isCall(Instruction) &&
          !BC.MIB->getConditionalTailCall(Instruction)) {
        assert(!BC.MIB->isInvoke(Instruction) &&
               "overlapping exception ranges detected");
        // Add extra operands to a call instruction making it an invoke from
        // now on.
        BC.MIB->addEHInfo(Instruction,
                          MCPlus::MCLandingPad(LPSymbol, ActionEntry));
      }
      ++II;
    } while (II != IE && II->first < Start + Length);

    if (ActionEntry != 0) {
      auto printType = [&] (int Index, raw_ostream &OS) {
        assert(Index > 0 && "only positive indices are valid");
        uint32_t TTEntry = TypeTableStart - Index * TTypeEncodingSize;
        const auto TTEntryAddress = TTEntry + LSDASectionAddress;
        uint32_t TypeAddress =
            *Data.getEncodedPointer(&TTEntry, TTypeEncoding, TTEntryAddress);
        if ((TTypeEncoding & DW_EH_PE_pcrel) &&
            (TypeAddress == TTEntryAddress)) {
          TypeAddress = 0;
        }
        if (TypeAddress == 0) {
          OS << "<all>";
          return;
        }
        if (TTypeEncoding & DW_EH_PE_indirect) {
          auto PointerOrErr = BC.getPointerAtAddress(TypeAddress);
          assert(PointerOrErr && "failed to decode indirect address");
          TypeAddress = *PointerOrErr;
        }
        if (auto *TypeSymBD = BC.getBinaryDataAtAddress(TypeAddress)) {
          OS << TypeSymBD->getName();
        } else {
          OS << "0x" << Twine::utohexstr(TypeAddress);
        }
      };
      if (opts::PrintExceptions)
        outs() << "    actions: ";
      uint32_t ActionPtr = ActionTableStart + ActionEntry - 1;
      long long ActionType;
      long long ActionNext;
      auto Sep = "";
      do {
        ActionType = Data.getSLEB128(&ActionPtr);
        auto Self = ActionPtr;
        ActionNext = Data.getSLEB128(&ActionPtr);
        if (opts::PrintExceptions)
          outs() << Sep << "(" << ActionType << ", " << ActionNext << ") ";
        if (ActionType == 0) {
          if (opts::PrintExceptions)
            outs() << "cleanup";
        } else if (ActionType > 0) {
          // It's an index into a type table.
          MaxTypeIndex = std::max(MaxTypeIndex,
                                  static_cast<unsigned>(ActionType));
          if (opts::PrintExceptions) {
            outs() << "catch type ";
            printType(ActionType, outs());
          }
        } else { // ActionType < 0
          if (opts::PrintExceptions)
            outs() << "filter exception types ";
          auto TSep = "";
          // ActionType is a negative *byte* offset into *uleb128-encoded* table
          // of indices with base 1.
          // E.g. -1 means offset 0, -2 is offset 1, etc. The indices are
          // encoded using uleb128 thus we cannot directly dereference them.
          uint32_t TypeIndexTablePtr = TypeIndexTableStart - ActionType - 1;
          while (auto Index = Data.getULEB128(&TypeIndexTablePtr)) {
            MaxTypeIndex = std::max(MaxTypeIndex, static_cast<unsigned>(Index));
            if (opts::PrintExceptions) {
              outs() << TSep;
              printType(Index, outs());
              TSep = ", ";
            }
          }
          MaxTypeIndexTableOffset =
              std::max(MaxTypeIndexTableOffset,
                       TypeIndexTablePtr - TypeIndexTableStart);
        }

        Sep = "; ";

        ActionPtr = Self + ActionNext;
      } while (ActionNext);
      if (opts::PrintExceptions)
        outs() << '\n';
    }
  }
  if (opts::PrintExceptions)
    outs() << '\n';

  assert(TypeIndexTableStart + MaxTypeIndexTableOffset <=
             Data.getData().size() &&
         "LSDA entry has crossed section boundary");

  if (TTypeEnd) {
    LSDAActionTable = LSDASectionData.slice(
        ActionTableStart, TypeIndexTableStart -
                              MaxTypeIndex * TTypeEncodingSize -
                              ActionTableStart);
    for (unsigned Index = 1; Index <= MaxTypeIndex; ++Index) {
      uint32_t TTEntry = TypeTableStart - Index * TTypeEncodingSize;
      const auto TTEntryAddress = TTEntry + LSDASectionAddress;
      uint64_t TypeAddress =
          *Data.getEncodedPointer(&TTEntry, TTypeEncoding, TTEntryAddress);
      if ((TTypeEncoding & DW_EH_PE_pcrel) && (TypeAddress == TTEntryAddress)) {
        TypeAddress = 0;
      }
      if (TypeAddress && (TTypeEncoding & DW_EH_PE_indirect)) {
        auto PointerOrErr = BC.getPointerAtAddress(TypeAddress);
        assert(PointerOrErr && "failed to decode indirect address");
        TypeAddress = *PointerOrErr;
      }
      LSDATypeTable.emplace_back(TypeAddress);
    }
    LSDATypeIndexTable =
        LSDASectionData.slice(TypeIndexTableStart, MaxTypeIndexTableOffset);
  }
}

void BinaryFunction::updateEHRanges() {
  if (getSize() == 0)
    return;

  assert(CurrentState == State::CFG_Finalized && "unexpected state");

  // Build call sites table.
  struct EHInfo {
    const MCSymbol *LP; // landing pad
    uint64_t Action;
  };

  // If previous call can throw, this is its exception handler.
  EHInfo PreviousEH = {nullptr, 0};

  // Marker for the beginning of exceptions range.
  const MCSymbol *StartRange = nullptr;

  // Indicates whether the start range is located in a cold part.
  bool IsStartInCold = false;

  // Have we crossed hot/cold border for split functions?
  bool SeenCold = false;

  // Sites to update - either regular or cold.
  auto *Sites = &CallSites;

  for (auto &BB : BasicBlocksLayout) {

    if (BB->isCold() && !SeenCold) {
      SeenCold = true;

      // Close the range (if any) and change the target call sites.
      if (StartRange) {
        Sites->emplace_back(CallSite{StartRange, getFunctionEndLabel(),
                                     PreviousEH.LP, PreviousEH.Action});
      }
      Sites = &ColdCallSites;

      // Reset the range.
      StartRange = nullptr;
      PreviousEH = {nullptr, 0};
    }

    for (auto II = BB->begin(); II != BB->end(); ++II) {
      if (!BC.MIB->isCall(*II))
        continue;

      // Instruction can throw an exception that should be handled.
      const bool Throws = BC.MIB->isInvoke(*II);

      // Ignore the call if it's a continuation of a no-throw gap.
      if (!Throws && !StartRange)
        continue;

      // Extract exception handling information from the instruction.
      const MCSymbol *LP = nullptr;
      uint64_t Action = 0;
      if (const auto EHInfo = BC.MIB->getEHInfo(*II))
        std::tie(LP, Action) = *EHInfo;

      // No action if the exception handler has not changed.
      if (Throws &&
          StartRange &&
          PreviousEH.LP == LP &&
          PreviousEH.Action == Action)
        continue;

      // Same symbol is used for the beginning and the end of the range.
      const MCSymbol *EHSymbol = BC.Ctx->createTempSymbol("EH", true);
      MCInst EHLabel;
      BC.MIB->createEHLabel(EHLabel, EHSymbol, BC.Ctx.get());
      II = std::next(BB->insertPseudoInstr(II, EHLabel));

      // At this point we could be in one of the following states:
      //
      // I. Exception handler has changed and we need to close previous range
      //    and start a new one.
      //
      // II. Start a new exception range after the gap.
      //
      // III. Close current exception range and start a new gap.
      const MCSymbol *EndRange;
      if (StartRange) {
        // I, III:
        EndRange = EHSymbol;
      } else {
        // II:
        StartRange = EHSymbol;
        IsStartInCold = SeenCold;
        EndRange = nullptr;
      }

      // Close the previous range.
      if (EndRange) {
        Sites->emplace_back(CallSite{StartRange, EndRange,
                                     PreviousEH.LP, PreviousEH.Action});
      }

      if (Throws) {
        // I, II:
        StartRange = EHSymbol;
        IsStartInCold = SeenCold;
        PreviousEH = EHInfo{LP, Action};
      } else {
        StartRange = nullptr;
      }
    }
  }

  // Check if we need to close the range.
  if (StartRange) {
    assert((!isSplit() || Sites == &ColdCallSites) && "sites mismatch");
    const auto *EndRange = IsStartInCold ? getFunctionColdEndLabel()
                                         : getFunctionEndLabel();
    Sites->emplace_back(CallSite{StartRange, EndRange,
                                 PreviousEH.LP, PreviousEH.Action});
  }
}

// The code is based on EHStreamer::emitExceptionTable().
void BinaryFunction::emitLSDA(MCStreamer *Streamer, bool EmitColdPart) {
  const auto *Sites = EmitColdPart ? &ColdCallSites : &CallSites;
  if (Sites->empty()) {
    return;
  }


  // Calculate callsite table size. Size of each callsite entry is:
  //
  //  sizeof(start) + sizeof(length) + sizeof(LP) + sizeof(uleb128(action))
  //
  // or
  //
  //  sizeof(dwarf::DW_EH_PE_data4) * 3 + sizeof(uleb128(action))
  uint64_t CallSiteTableLength = Sites->size() * 4 * 3;
  for (const auto &CallSite : *Sites) {
    CallSiteTableLength += getULEB128Size(CallSite.Action);
  }

  Streamer->SwitchSection(BC.MOFI->getLSDASection());

  const auto TTypeEncoding = BC.MOFI->getTTypeEncoding();
  const auto TTypeEncodingSize = getEncodingSize(TTypeEncoding, BC);
  const auto TTypeAlignment = 4;

  // Type tables have to be aligned at 4 bytes.
  Streamer->EmitValueToAlignment(TTypeAlignment);

  // Emit the LSDA label.
  auto LSDASymbol = EmitColdPart ? getColdLSDASymbol() : getLSDASymbol();
  assert(LSDASymbol && "no LSDA symbol set");
  Streamer->EmitLabel(LSDASymbol);

  // Corresponding FDE start.
  const auto *StartSymbol = EmitColdPart ? getColdSymbol() : getSymbol();

  // Emit the LSDA header.

  // If LPStart is omitted, then the start of the FDE is used as a base for
  // landing pad displacements. Then if a cold fragment starts with
  // a landing pad, this means that the first landing pad offset will be 0.
  // As a result, an exception handling runtime will ignore this landing pad,
  // because zero offset denotes the absence of a landing pad.
  // For this reason, we emit LPStart value of 0 and output an absolute value
  // of the landing pad in the table.
  //
  // FIXME: this may break PIEs and DSOs where the base address is not 0.
  Streamer->EmitIntValue(dwarf::DW_EH_PE_udata4, 1); // LPStart format
  Streamer->EmitIntValue(0, 4);
  auto emitLandingPad = [&](const MCSymbol *LPSymbol) {
    if (!LPSymbol) {
      Streamer->EmitIntValue(0, 4);
      return;
    }
    Streamer->EmitSymbolValue(LPSymbol, 4);
  };

  Streamer->EmitIntValue(TTypeEncoding, 1);        // TType format

  // See the comment in EHStreamer::emitExceptionTable() on to use
  // uleb128 encoding (which can use variable number of bytes to encode the same
  // value) to ensure type info table is properly aligned at 4 bytes without
  // iteratively fixing sizes of the tables.
  unsigned CallSiteTableLengthSize = getULEB128Size(CallSiteTableLength);
  unsigned TTypeBaseOffset =
    sizeof(int8_t) +                            // Call site format
    CallSiteTableLengthSize +                   // Call site table length size
    CallSiteTableLength +                       // Call site table length
    LSDAActionTable.size() +                    // Actions table size
    LSDATypeTable.size() * TTypeEncodingSize;   // Types table size
  unsigned TTypeBaseOffsetSize = getULEB128Size(TTypeBaseOffset);
  unsigned TotalSize =
    sizeof(int8_t) +                            // LPStart format
    sizeof(int8_t) +                            // TType format
    TTypeBaseOffsetSize +                       // TType base offset size
    TTypeBaseOffset;                            // TType base offset
  unsigned SizeAlign = (4 - TotalSize) & 3;

  // Account for any extra padding that will be added to the call site table
  // length.
  Streamer->EmitPaddedULEB128IntValue(TTypeBaseOffset,
                                      TTypeBaseOffsetSize + SizeAlign);

  // Emit the landing pad call site table. We use signed data4 since we can emit
  // a landing pad in a different part of the split function that could appear
  // earlier in the address space than LPStart.
  Streamer->EmitIntValue(dwarf::DW_EH_PE_sdata4, 1);
  Streamer->EmitULEB128IntValue(CallSiteTableLength);

  for (const auto &CallSite : *Sites) {

    const auto *BeginLabel = CallSite.Start;
    const auto *EndLabel = CallSite.End;

    assert(BeginLabel && "start EH label expected");
    assert(EndLabel && "end EH label expected");

    // Start of the range is emitted relative to the start of current
    // function split part.
    Streamer->emitAbsoluteSymbolDiff(BeginLabel, StartSymbol, 4);
    Streamer->emitAbsoluteSymbolDiff(EndLabel, BeginLabel, 4);
    emitLandingPad(CallSite.LP);
    Streamer->EmitULEB128IntValue(CallSite.Action);
  }

  // Write out action, type, and type index tables at the end.
  //
  // For action and type index tables there's no need to change the original
  // table format unless we are doing function splitting, in which case we can
  // split and optimize the tables.
  //
  // For type table we (re-)encode the table using TTypeEncoding matching
  // the current assembler mode.
  for (auto const &Byte : LSDAActionTable) {
    Streamer->EmitIntValue(Byte, 1);
  }
  assert(!(TTypeEncoding & dwarf::DW_EH_PE_indirect) &&
         "indirect type info encoding is not supported yet");
  for (int Index = LSDATypeTable.size() - 1; Index >= 0; --Index) {
    // Note: the address could be an indirect one.
    const auto TypeAddress = LSDATypeTable[Index];
    switch (TTypeEncoding & 0x70) {
    default:
      llvm_unreachable("unsupported TTypeEncoding");
    case 0:
      Streamer->EmitIntValue(TypeAddress, TTypeEncodingSize);
      break;
    case dwarf::DW_EH_PE_pcrel: {
      if (TypeAddress) {
        const auto *TypeSymbol =
          BC.getOrCreateGlobalSymbol(TypeAddress,
                                     "TI",
                                     TTypeEncodingSize,
                                     TTypeAlignment);
        auto *DotSymbol = BC.Ctx->createTempSymbol();
        Streamer->EmitLabel(DotSymbol);
        const auto *SubDotExpr = MCBinaryExpr::createSub(
            MCSymbolRefExpr::create(TypeSymbol, *BC.Ctx),
            MCSymbolRefExpr::create(DotSymbol, *BC.Ctx),
            *BC.Ctx);
        Streamer->EmitValue(SubDotExpr, TTypeEncodingSize);
      } else {
        Streamer->EmitIntValue(0, TTypeEncodingSize);
      }
      break;
    }
    }
  }
  for (auto const &Byte : LSDATypeIndexTable) {
    Streamer->EmitIntValue(Byte, 1);
  }
}

const uint8_t DWARF_CFI_PRIMARY_OPCODE_MASK = 0xc0;

CFIReaderWriter::CFIReaderWriter(const DWARFDebugFrame &EHFrame) {
  // Prepare FDEs for fast lookup
  for (const auto &Entry : EHFrame.entries()) {
    const auto *CurFDE = dyn_cast<dwarf::FDE>(&Entry);
    // Skip CIEs.
    if (!CurFDE)
      continue;
    // There could me multiple FDEs with the same initial address, and perhaps
    // different sizes (address ranges). Use the first entry with non-zero size.
    auto FDEI = FDEs.lower_bound(CurFDE->getInitialLocation());
    if (FDEI != FDEs.end() && FDEI->first == CurFDE->getInitialLocation()) {
      if (CurFDE->getAddressRange()) {
        if (FDEI->second->getAddressRange() == 0) {
          FDEI->second = CurFDE;
        } else if (opts::Verbosity > 0) {
          errs() << "BOLT-WARNING: different FDEs for function at 0x"
                 << Twine::utohexstr(FDEI->first)
                 << " detected; sizes: "
                 << FDEI->second->getAddressRange() << " and "
                 << CurFDE->getAddressRange() << '\n';
        }
      }
    } else {
      FDEs.emplace_hint(FDEI, CurFDE->getInitialLocation(), CurFDE);
    }
  }
}

bool CFIReaderWriter::fillCFIInfoFor(BinaryFunction &Function) const {
  uint64_t Address = Function.getAddress();
  auto I = FDEs.find(Address);
  // Ignore zero-length FDE ranges.
  if (I == FDEs.end() || !I->second->getAddressRange())
    return true;

  const FDE &CurFDE = *I->second;
  if (Function.getSize() != CurFDE.getAddressRange()) {
    if (opts::Verbosity >= 1) {
      errs() << "BOLT-WARNING: CFI information size mismatch for function \""
             << Function << "\""
             << format(": Function size is %dB, CFI covers "
                       "%dB\n",
                       Function.getSize(), CurFDE.getAddressRange());
    }
    return false;
  }

  auto LSDA = CurFDE.getLSDAAddress();
  Function.setLSDAAddress(LSDA ? *LSDA : 0);

  uint64_t Offset = 0;
  uint64_t CodeAlignment = CurFDE.getLinkedCIE()->getCodeAlignmentFactor();
  uint64_t DataAlignment = CurFDE.getLinkedCIE()->getDataAlignmentFactor();
  if (CurFDE.getLinkedCIE()->getPersonalityAddress()) {
    Function.setPersonalityFunction(
        *CurFDE.getLinkedCIE()->getPersonalityAddress());
    Function.setPersonalityEncoding(
        *CurFDE.getLinkedCIE()->getPersonalityEncoding());
  }

  auto decodeFrameInstruction =
      [&Function, &Offset, Address, CodeAlignment, DataAlignment](
          const CFIProgram::Instruction &Instr) {
        uint8_t Opcode = Instr.Opcode;
        if (Opcode & DWARF_CFI_PRIMARY_OPCODE_MASK)
          Opcode &= DWARF_CFI_PRIMARY_OPCODE_MASK;
        switch (Instr.Opcode) {
        case DW_CFA_nop:
          break;
        case DW_CFA_advance_loc4:
        case DW_CFA_advance_loc2:
        case DW_CFA_advance_loc1:
        case DW_CFA_advance_loc:
          // Advance our current address
          Offset += CodeAlignment * int64_t(Instr.Ops[0]);
          break;
        case DW_CFA_offset_extended_sf:
          Function.addCFIInstruction(
              Offset, MCCFIInstruction::createOffset(
                          nullptr, Instr.Ops[0],
                          DataAlignment * int64_t(Instr.Ops[1])));
          break;
        case DW_CFA_offset_extended:
        case DW_CFA_offset:
          Function.addCFIInstruction(
              Offset, MCCFIInstruction::createOffset(
                          nullptr, Instr.Ops[0], DataAlignment * Instr.Ops[1]));
          break;
        case DW_CFA_restore_extended:
        case DW_CFA_restore:
          Function.addCFIInstruction(
              Offset, MCCFIInstruction::createRestore(nullptr, Instr.Ops[0]));
          break;
        case DW_CFA_set_loc:
          assert(Instr.Ops[0] >= Address && "set_loc out of function bounds");
          assert(Instr.Ops[0] <= Address + Function.getSize() &&
                 "set_loc out of function bounds");
          Offset = Instr.Ops[0] - Address;
          break;

        case DW_CFA_undefined:
          Function.addCFIInstruction(
              Offset, MCCFIInstruction::createUndefined(nullptr, Instr.Ops[0]));
          break;
        case DW_CFA_same_value:
          Function.addCFIInstruction(
              Offset, MCCFIInstruction::createSameValue(nullptr, Instr.Ops[0]));
          break;
        case DW_CFA_register:
          Function.addCFIInstruction(
              Offset, MCCFIInstruction::createRegister(nullptr, Instr.Ops[0],
                                                       Instr.Ops[1]));
          break;
        case DW_CFA_remember_state:
          Function.addCFIInstruction(
              Offset, MCCFIInstruction::createRememberState(nullptr));
          break;
        case DW_CFA_restore_state:
          Function.addCFIInstruction(
              Offset, MCCFIInstruction::createRestoreState(nullptr));
          break;
        case DW_CFA_def_cfa:
          Function.addCFIInstruction(
              Offset, MCCFIInstruction::createDefCfa(nullptr, Instr.Ops[0],
                                                     Instr.Ops[1]));
          break;
        case DW_CFA_def_cfa_sf:
          Function.addCFIInstruction(
              Offset, MCCFIInstruction::createDefCfa(
                          nullptr, Instr.Ops[0],
                          DataAlignment * int64_t(Instr.Ops[1])));
          break;
        case DW_CFA_def_cfa_register:
          Function.addCFIInstruction(
              Offset,
              MCCFIInstruction::createDefCfaRegister(nullptr, Instr.Ops[0]));
          break;
        case DW_CFA_def_cfa_offset:
          Function.addCFIInstruction(
              Offset,
              MCCFIInstruction::createDefCfaOffset(nullptr, Instr.Ops[0]));
          break;
        case DW_CFA_def_cfa_offset_sf:
          Function.addCFIInstruction(
              Offset, MCCFIInstruction::createDefCfaOffset(
                          nullptr, DataAlignment * int64_t(Instr.Ops[0])));
          break;
        case DW_CFA_GNU_args_size:
          Function.addCFIInstruction(
              Offset,
              MCCFIInstruction::createGnuArgsSize(nullptr, Instr.Ops[0]));
          Function.setUsesGnuArgsSize();
          break;
        case DW_CFA_val_offset_sf:
        case DW_CFA_val_offset:
          if (opts::Verbosity >= 1) {
            errs() << "BOLT-WARNING: DWARF val_offset() unimplemented\n";
          }
          return false;
        case DW_CFA_expression:
        case DW_CFA_def_cfa_expression:
        case DW_CFA_val_expression: {
          MCDwarfExprBuilder Builder;
          for (auto &ExprOp : *Instr.Expression) {
            const DWARFExpression::Operation::Description &Desc =
                ExprOp.getDescription();
            if (Desc.Op[0] == DWARFExpression::Operation::SizeNA) {
              Builder.appendOperation(ExprOp.getCode());
            } else if (Desc.Op[1] == DWARFExpression::Operation::SizeNA) {
              Builder.appendOperation(ExprOp.getCode(),
                                      ExprOp.getRawOperand(0));
            } else {
              Builder.appendOperation(ExprOp.getCode(), ExprOp.getRawOperand(0),
                                      ExprOp.getRawOperand(1));
            }
          }
          if (Opcode == DW_CFA_expression) {
            Function.addCFIInstruction(
                Offset, MCCFIInstruction::createExpression(
                            nullptr, Instr.Ops[0], Builder.take()));
          } else if (Opcode == DW_CFA_def_cfa_expression) {
            Function.addCFIInstruction(Offset,
                                       MCCFIInstruction::createDefCfaExpression(
                                           nullptr, Builder.take()));
          } else {
            assert(Opcode == DW_CFA_val_expression && "Unexpected opcode");
            Function.addCFIInstruction(
                Offset, MCCFIInstruction::createValExpression(
                            nullptr, Instr.Ops[0], Builder.take()));
          }
          break;
        }
        case DW_CFA_MIPS_advance_loc8:
          if (opts::Verbosity >= 1) {
            errs() << "BOLT-WARNING: DW_CFA_MIPS_advance_loc unimplemented\n";
          }
          return false;
        case DW_CFA_GNU_window_save:
        case DW_CFA_lo_user:
        case DW_CFA_hi_user:
          if (opts::Verbosity >= 1) {
            errs() << "BOLT-WARNING: DW_CFA_GNU_* and DW_CFA_*_user "
                      "unimplemented\n";
          }
          return false;
        default:
          if (opts::Verbosity >= 1) {
            errs() << "BOLT-WARNING: Unrecognized CFI instruction\n";
          }
          return false;
        }

        return true;
      };

  for (const CFIProgram::Instruction &Instr : CurFDE.getLinkedCIE()->cfis()) {
    if (!decodeFrameInstruction(Instr))
      return false;
  }

  for (const CFIProgram::Instruction &Instr : CurFDE.cfis()) {
    if (!decodeFrameInstruction(Instr))
      return false;
  }

  return true;
}

std::vector<char> CFIReaderWriter::generateEHFrameHeader(
    const DWARFDebugFrame &OldEHFrame,
    const DWARFDebugFrame &NewEHFrame,
    uint64_t EHFrameHeaderAddress,
    std::vector<uint64_t> &FailedAddresses) const {
  // Common PC -> FDE map to be written into .eh_frame_hdr.
  std::map<uint64_t, uint64_t> PCToFDE;

  // Presort array for binary search.
  std::sort(FailedAddresses.begin(), FailedAddresses.end());

  // Initialize PCToFDE using NewEHFrame.
  NewEHFrame.for_each_FDE([&](const dwarf::FDE *FDE) {
    const auto FuncAddress = FDE->getInitialLocation();
    const auto FDEAddress = NewEHFrame.getEHFrameAddress() + FDE->getOffset();

    // Ignore unused FDEs.
    if (FuncAddress == 0)
      return;

    // Add the address to the map unless we failed to write it.
    if (!std::binary_search(FailedAddresses.begin(), FailedAddresses.end(),
                            FuncAddress)) {
      DEBUG(dbgs() << "BOLT-DEBUG: FDE for function at 0x"
                   << Twine::utohexstr(FuncAddress) << " is at 0x"
                   << Twine::utohexstr(FDEAddress) << '\n');
      PCToFDE[FuncAddress] = FDEAddress;
    }
  });

  DEBUG(dbgs() << "BOLT-DEBUG: new .eh_frame contains "
               << std::distance(NewEHFrame.entries().begin(),
                                NewEHFrame.entries().end())
               << " entries\n");

  // Add entries from the original .eh_frame corresponding to the functions
  // that we did not update.
  OldEHFrame.for_each_FDE([&](const dwarf::FDE *FDE) {
    const auto FuncAddress = FDE->getInitialLocation();
    const auto FDEAddress = OldEHFrame.getEHFrameAddress() + FDE->getOffset();

    // Add the address if we failed to write it.
    if (PCToFDE.count(FuncAddress) == 0) {
      DEBUG(dbgs() << "BOLT-DEBUG: old FDE for function at 0x"
                   << Twine::utohexstr(FuncAddress) << " is at 0x"
                   << Twine::utohexstr(FDEAddress) << '\n');
      PCToFDE[FuncAddress] = FDEAddress;
    }
  });

  DEBUG(dbgs() << "BOLT-DEBUG: old .eh_frame contains "
               << std::distance(OldEHFrame.entries().begin(),
                                OldEHFrame.entries().end())
               << " entries\n");

  // Generate a new .eh_frame_hdr based on the new map.

  // Header plus table of entries of size 8 bytes.
  std::vector<char> EHFrameHeader(12 + PCToFDE.size() * 8);

  // Version is 1.
  EHFrameHeader[0] = 1;
  // Encoding of the eh_frame pointer.
  EHFrameHeader[1] = DW_EH_PE_pcrel | DW_EH_PE_sdata4;
  // Encoding of the count field to follow.
  EHFrameHeader[2] = DW_EH_PE_udata4;
  // Encoding of the table entries - 4-byte offset from the start of the header.
  EHFrameHeader[3] = DW_EH_PE_datarel | DW_EH_PE_sdata4;

  // Address of eh_frame. Use the new one.
  support::ulittle32_t::ref(EHFrameHeader.data() + 4) =
    NewEHFrame.getEHFrameAddress() - (EHFrameHeaderAddress + 4);

  // Number of entries in the table (FDE count).
  support::ulittle32_t::ref(EHFrameHeader.data() + 8) = PCToFDE.size();

  // Write the table at offset 12.
  auto *Ptr = EHFrameHeader.data();
  uint32_t Offset = 12;
  for (const auto &PCI : PCToFDE) {
    int64_t InitialPCOffset = PCI.first - EHFrameHeaderAddress;
    assert(isInt<32>(InitialPCOffset) && "PC offset out of bounds");
    support::ulittle32_t::ref(Ptr + Offset) = InitialPCOffset;
    Offset += 4;
    int64_t FDEOffset = PCI.second - EHFrameHeaderAddress;
    assert(isInt<32>(FDEOffset) && "FDE offset out of bounds");
    support::ulittle32_t::ref(Ptr + Offset) = FDEOffset;
    Offset += 4;
  }

  return EHFrameHeader;
}

} // namespace bolt
} // namespace llvm
