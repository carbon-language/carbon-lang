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
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#undef  DEBUG_TYPE
#define DEBUG_TYPE "flo-exceptions"

using namespace llvm::dwarf;

namespace llvm {
namespace flo {

namespace opts {

static cl::opt<bool>
PrintExceptions("print-exceptions",
                cl::desc("print exception handling data"),
                cl::Hidden);

} // namespace opts

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
//   http://mentorembedded.github.io/cxx-abi/exceptions.pdf
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
// For the purpose of rewriting exception handling tables, we can reuse action,
// types, and type index tables in their original binary format.
// This is only possible when type references are encoded as absolute addresses.
// We still have to parse all the tables to determine their sizes. Then we have
// to parse the call site table and associate discovered information with
// actual call instructions and landing pad blocks.
//
// Ideally we should be able to re-write LSDA in-place, without the need to
// allocate a new space for it. Sadly there's no guarantee that the new call
// site table will be the same size as GCC uses uleb encodings for PC offsets.
//
// For split function re-writing we would need to split LSDA too.
//
// Note: some functions have LSDA entries with 0 call site entries.
void BinaryFunction::parseLSDA(ArrayRef<uint8_t> LSDASectionData,
                               uint64_t LSDASectionAddress) {
  assert(CurrentState == State::Disassembled && "unexpected function state");

  if (!getLSDAAddress())
    return;

  assert(getLSDAAddress() < LSDASectionAddress + LSDASectionData.size() &&
         "wrong LSDA address");

  const uint8_t *Ptr =
      LSDASectionData.data() + getLSDAAddress() - LSDASectionAddress;

  uint8_t LPStartEncoding = *Ptr++;
  uintptr_t LPStart = 0;
  if (LPStartEncoding != DW_EH_PE_omit) {
    LPStart = readEncodedPointer(Ptr, LPStartEncoding);
  }

  assert(LPStart == 0 && "support for split functions not implemented");

  uint8_t TTypeEncoding = *Ptr++;
  uintptr_t TTypeEnd = 0;
  if (TTypeEncoding != DW_EH_PE_omit) {
    TTypeEnd = readULEB128(Ptr);
  }

  if (opts::PrintExceptions) {
    errs() << "[LSDA at 0x" << Twine::utohexstr(getLSDAAddress())
           << " for function " << getName() << "]:\n";
    errs() << "LPStart Encoding = " << (unsigned)LPStartEncoding << '\n';
    errs() << "LPStart = 0x" << Twine::utohexstr(LPStart) << '\n';
    errs() << "TType Encoding = " << (unsigned)TTypeEncoding << '\n';
    errs() << "TType End = " << TTypeEnd << '\n';
  }

  // Table to store list of indices in type table. Entries are uleb128 values.
  auto TypeIndexTableStart = Ptr + TTypeEnd;

  // Offset past the last decoded index.
  intptr_t MaxTypeIndexTableOffset = 0;

  // The actual type info table starts at the same location, but grows in
  // different direction. Encoding is different too (TTypeEncoding).
  auto TypeTableStart = reinterpret_cast<const uint32_t *>(Ptr + TTypeEnd);

  uint8_t       CallSiteEncoding = *Ptr++;
  uint32_t      CallSiteTableLength = readULEB128(Ptr);
  const uint8_t *CallSiteTableStart = Ptr;
  const uint8_t *CallSiteTableEnd = CallSiteTableStart + CallSiteTableLength;
  const uint8_t *CallSitePtr = CallSiteTableStart;
  const uint8_t *ActionTableStart = CallSiteTableEnd;

  if (opts::PrintExceptions) {
    errs() << "CallSite Encoding = " << (unsigned)CallSiteEncoding << '\n';
    errs() << "CallSite table length = " << CallSiteTableLength << '\n';
    errs() << '\n';
  }

  unsigned NumCallSites = 0;
  uint64_t RangeBase = getAddress();
  while (CallSitePtr < CallSiteTableEnd) {
    ++NumCallSites;

    uintptr_t Start = readEncodedPointer(CallSitePtr, CallSiteEncoding);
    uintptr_t Length = readEncodedPointer(CallSitePtr, CallSiteEncoding);
    uintptr_t LandingPad = readEncodedPointer(CallSitePtr, CallSiteEncoding);
    uintptr_t ActionEntry = readULEB128(CallSitePtr);

    if (opts::PrintExceptions) {
      errs() << "Call Site: [0x" << Twine::utohexstr(RangeBase + Start)
             << ", 0x" << Twine::utohexstr(RangeBase + Start + Length)
             << "); landing pad: 0x" << Twine::utohexstr(LPStart + LandingPad)
             << "; action entry: 0x" << Twine::utohexstr(ActionEntry) << "\n";
    }

    // Create a handler entry if necessary.
    MCSymbol *LPSymbol{nullptr};
    if (LandingPad) {
      auto Label = Labels.find(LandingPad);
      if (Label != Labels.end()) {
        LPSymbol = Label->second;
      } else {
        LPSymbol = BC.Ctx->createTempSymbol("LP", true);
        Labels[LandingPad] = LPSymbol;
      }
      LandingPads.insert(LPSymbol);
    }

    // Mark all call instructions in the range.
    auto II = Instructions.find(Start);
    auto IE = Instructions.end();
    assert(II != IE && "exception range not pointing to an instruction");
    do {
      auto &Instruction = II->second;
      if (BC.MIA->isCall(Instruction)) {
        assert(!BC.MIA->isInvoke(Instruction) &&
               "overlapping exception ranges detected");
        // Add extra operands to a call instruction making it an invoke from
        // now on.
        if (LPSymbol) {
          Instruction.addOperand(MCOperand::createExpr(
              MCSymbolRefExpr::create(LPSymbol,
                                      MCSymbolRefExpr::VK_None,
                                      *BC.Ctx)));
        } else {
          Instruction.addOperand(MCOperand::createImm(0));
        }
        Instruction.addOperand(MCOperand::createImm(ActionEntry));
      }
      ++II;
    } while (II != IE && II->first < Start + Length);

    if (ActionEntry != 0) {
      auto printType = [&] (int Index, raw_ostream &OS) {
        assert(Index > 0 && "only positive indices are valid");
        assert(TTypeEncoding == DW_EH_PE_udata4 &&
               "only udata4 supported for TTypeEncoding");
        auto TypeAddress = *(TypeTableStart - Index);
        if (TypeAddress == 0) {
          OS << "<all>";
          return;
        }
        auto NI = BC.GlobalAddresses.find(TypeAddress);
        if (NI != BC.GlobalAddresses.end()) {
          OS << NI->second;
        } else {
          OS << "0x" << Twine::utohexstr(TypeAddress);
        }
      };
      if (opts::PrintExceptions)
        errs() << "    actions: ";
      const uint8_t *ActionPtr = ActionTableStart + ActionEntry - 1;
      long long ActionType;
      long long ActionNext;
      auto Sep = "";
      do {
        ActionType = readSLEB128(ActionPtr);
        auto Self = ActionPtr;
        ActionNext = readSLEB128(ActionPtr);
        if (opts::PrintExceptions)
          errs() << Sep << "(" << ActionType << ", " << ActionNext << ") ";
        if (ActionType == 0) {
          if (opts::PrintExceptions)
            errs() << "cleanup";
        } else if (ActionType > 0) {
          // It's an index into a type table.
          if (opts::PrintExceptions) {
            errs() << "catch type ";
            printType(ActionType, errs());
          }
        } else { // ActionType < 0
          if (opts::PrintExceptions)
            errs() << "filter exception types ";
          auto TSep = "";
          // ActionType is a negative byte offset into uleb128-encoded table
          // of indices with base 1.
          // E.g. -1 means offset 0, -2 is offset 1, etc. The indices are
          // encoded using uleb128 thus we cannot directly dereference them.
          auto TypeIndexTablePtr = TypeIndexTableStart - ActionType - 1;
          while (auto Index = readULEB128(TypeIndexTablePtr)) {
            if (opts::PrintExceptions) {
              errs() << TSep;
              printType(Index, errs());
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
        errs() << '\n';
    }
  }
  if (opts::PrintExceptions)
    errs() << '\n';

  assert(reinterpret_cast<uint8_t *>(MaxTypeIndexTableOffset) <=
         LSDASectionData.data() + LSDASectionData.size() &&
         "LSDA entry has crossed section boundary");

  LSDATables =
    ArrayRef<uint8_t>(ActionTableStart,
                      reinterpret_cast<uint8_t *>(MaxTypeIndexTableOffset));
  LSDATablesTypeOffset =
    reinterpret_cast<const uint8_t *>(TypeTableStart) - ActionTableStart;
}

void BinaryFunction::updateEHRanges() {
  assert(CurrentState == State::CFG && "unexpected state");

  // Build call sites table.
  struct EHInfo {
    const MCSymbol *LP; // landing pad
    uint64_t Action;
  };

  // Markers for begining and the end of exceptions range.
  const MCSymbol *StartRange{nullptr};
  const MCSymbol *EndRange{nullptr};

  // If previous call can throw, this is its exception handler.
  EHInfo PreviousEH = {nullptr, 0};

  for(auto &BB : BasicBlocksLayout) {
    for (auto II = BB->begin(); II != BB->end(); ++II) {
      auto Instr = *II;

      if (!BC.MIA->isCall(Instr))
        continue;

      // Instruction can throw an exception that should be handled.
      bool Throws = BC.MIA->isInvoke(Instr);

      // Ignore the call if it's a continuation of a no-throw gap.
      if (!Throws && !StartRange)
        continue;

      // Extract exception handling information from the instruction.
      const MCSymbol *LP = nullptr;
      uint64_t Action = 0;
      std::tie(LP, Action) = BC.MIA->getEHInfo(Instr);

      // No action if the exception handler has not changed.
      if (Throws &&
          StartRange &&
          PreviousEH.LP == LP &&
          PreviousEH.Action == Action)
        continue;

      // Same symbol is used for the beginning and the end of the range.
      const MCSymbol *EHSymbol = BC.Ctx->createTempSymbol("EH", true);
      MCInst EHLabel;
      BC.MIA->createEHLabel(EHLabel, EHSymbol, BC.Ctx.get());
      II = BB->Instructions.insert(II, EHLabel);
      ++II;

      // At this point we could be in the one of the following states:
      //
      // I. Exception handler has changed and we need to close the prev range
      //    and start the new one.
      //
      // II. Start the new exception range after the gap.
      //
      // III. Close exception range and start the new gap.

      if (StartRange) {
        // I, III:
        EndRange = EHSymbol;
      } else {
        // II:
        StartRange = EHSymbol;
        EndRange = nullptr;
      }

      // Close the previous range.
      if (EndRange) {
        assert(StartRange && "beginning of the range expected");
        CallSites.emplace_back(CallSite{StartRange, EndRange,
                                        PreviousEH.LP, PreviousEH.Action});
        EndRange = nullptr;
      }

      if (Throws) {
        // I, II:
        StartRange = EHSymbol;
        PreviousEH = EHInfo{LP, Action};
      } else {
        StartRange = nullptr;
      }
    }
  }

  // Check if we need to close the range.
  if (StartRange) {
    assert(!EndRange && "unexpected end of range");
    EndRange = BC.Ctx->createTempSymbol("EH", true);
    MCInst EHLabel;
    BC.MIA->createEHLabel(EHLabel, EndRange, BC.Ctx.get());
    BasicBlocksLayout.back()->Instructions.emplace_back(EHLabel);

    CallSites.emplace_back(CallSite{StartRange, EndRange,
                                    PreviousEH.LP, PreviousEH.Action});
  }
}

const uint8_t DWARF_CFI_PRIMARY_OPCODE_MASK = 0xc0;
const uint8_t DWARF_CFI_PRIMARY_OPERAND_MASK = 0x3f;

void CFIReaderWriter::fillCFIInfoFor(BinaryFunction &Function) const {
  uint64_t Address = Function.getAddress();
  auto I = FDEs.find(Address);
  if (I == FDEs.end())
    return;

  const FDE &CurFDE = *I->second;
  if (Function.getSize() != CurFDE.getAddressRange()) {
    errs() << "FLO-WARNING: CFI information size mismatch for function \""
           << Function.getName() << "\""
           << format(": Function size is %dB, CFI covers "
                     "%dB\n",
                     Function.getSize(), CurFDE.getAddressRange());
  }

  Function.setLSDAAddress(CurFDE.getLSDAAddress());

  uint64_t Offset = 0;
  uint64_t CodeAlignment = CurFDE.getLinkedCIE()->getCodeAlignmentFactor();
  uint64_t DataAlignment = CurFDE.getLinkedCIE()->getDataAlignmentFactor();
  if (CurFDE.getLinkedCIE()->getPersonalityAddress() != 0) {
    Function.setPersonalityFunction(
        CurFDE.getLinkedCIE()->getPersonalityAddress());
    Function.setPersonalityEncoding(
        CurFDE.getLinkedCIE()->getPersonalityEncoding());
  }

  auto decodeFrameInstruction =
      [&Function, &Offset, Address, CodeAlignment, DataAlignment](
          const FrameEntry::Instruction &Instr) {
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
          assert(Instr.Ops[0] < Address && "set_loc out of function bounds");
          assert(Instr.Ops[0] > Address + Function.getSize() &&
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
              Offset, MCCFIInstruction::createDefCfa(nullptr, Instr.Ops[1],
                                                     -Instr.Ops[0]));
          break;
        case DW_CFA_def_cfa_sf:
          Function.addCFIInstruction(
              Offset, MCCFIInstruction::createDefCfa(
                          nullptr, Instr.Ops[1],
                          -(DataAlignment * int64_t(Instr.Ops[0]))));
          break;
        case DW_CFA_def_cfa_register:
          Function.addCFIInstruction(
              Offset,
              MCCFIInstruction::createDefCfaRegister(nullptr, Instr.Ops[0]));
          break;
        case DW_CFA_def_cfa_offset:
          Function.addCFIInstruction(
              Offset,
              MCCFIInstruction::createDefCfaOffset(nullptr, -Instr.Ops[0]));
          break;
        case DW_CFA_def_cfa_offset_sf:
          Function.addCFIInstruction(
              Offset, MCCFIInstruction::createDefCfaOffset(
                          nullptr, -(DataAlignment * int64_t(Instr.Ops[0]))));
          break;
        case DW_CFA_val_offset_sf:
        case DW_CFA_val_offset:
          llvm_unreachable("DWARF val_offset() unimplemented");
          break;
        case DW_CFA_expression:
        case DW_CFA_def_cfa_expression:
        case DW_CFA_val_expression:
          llvm_unreachable("DWARF CFA expressions unimplemented");
          break;
        case DW_CFA_MIPS_advance_loc8:
          llvm_unreachable("DW_CFA_MIPS_advance_loc unimplemented");
          break;
        case DW_CFA_GNU_args_size:
        case DW_CFA_GNU_window_save:
        case DW_CFA_lo_user:
        case DW_CFA_hi_user:
          llvm_unreachable("DW_CFA_GNU_* and DW_CFA_*_use unimplemented");
          break;
        default:
          llvm_unreachable("Unrecognized CFI instruction");
        }
      };

  for (const FrameEntry::Instruction &Instr : *(CurFDE.getLinkedCIE())) {
    decodeFrameInstruction(Instr);
  }

  for (const FrameEntry::Instruction &Instr : CurFDE) {
    decodeFrameInstruction(Instr);
  }
}

void CFIReaderWriter::rewriteHeaderFor(StringRef EHFrame,
                                       uint64_t NewEHFrameAddress,
                                       uint64_t NewFrameHdrAddress,
                                       ArrayRef<uint64_t> FailedAddresses) {
  DataExtractor Data(EHFrame,
                     /*IsLittleEndian=*/true,
                     /*PtrSize=*/4);
  uint32_t Offset = 0;
  std::map<uint64_t, uint64_t> PCToFDE;

  DEBUG(dbgs() << format(
            "CFIReaderWriter: Starting to patch .eh_frame_hdr.\n"
            "New .eh_frame address = %08x\nNew .eh_frame_hdr address = %08x\n",
            NewEHFrameAddress, NewFrameHdrAddress));

  // Scans the EHFrame, parsing start addresses for each function
  while (Data.isValidOffset(Offset)) {
    uint32_t StartOffset = Offset;

    uint64_t Length = Data.getU32(&Offset);

    if (Length == 0)
      break;

    uint32_t EndStructureOffset = Offset + static_cast<uint32_t>(Length);
    uint64_t Id = Data.getUnsigned(&Offset, 4);
    if (Id == 0) {
      Offset = EndStructureOffset;
      continue;
    }

    const uint8_t *DataStart =
        reinterpret_cast<const uint8_t *>(Data.getData().substr(Offset).data());
    const uint8_t *DataEnd = DataStart;
    uint64_t FuncAddress =
        readEncodedPointer(DataEnd, DW_EH_PE_sdata4 | DW_EH_PE_pcrel,
                           NewEHFrameAddress + Offset - (uintptr_t)DataEnd);
    Offset += DataEnd - DataStart;

    auto I = std::lower_bound(FailedAddresses.begin(), FailedAddresses.end(),
                              FuncAddress);
    if (I != FailedAddresses.end() && *I == FuncAddress) {
      Offset = EndStructureOffset;
      continue;
    }

    PCToFDE[FuncAddress] = NewEHFrameAddress + StartOffset;
    Offset = EndStructureOffset;
  }

  //Updates the EHFrameHdr
  DataExtractor HDRData(
      StringRef(FrameHdrContents.data(), FrameHdrContents.size()),
      /*IsLittleEndian=*/true,
      /*PtrSize=*/4);
  Offset = 0;
  uint8_t Version = HDRData.getU8(&Offset);
  assert(Version == 1 &&
         "Don't know how to handle this version of .eh_frame_hdr");

  uint8_t EhFrameAddrEncoding = HDRData.getU8(&Offset);
  uint8_t FDECntEncoding = HDRData.getU8(&Offset);
  uint8_t TableEncoding = HDRData.getU8(&Offset);
  const uint8_t *DataStart = reinterpret_cast<const uint8_t *>(
      HDRData.getData().substr(Offset).data());
  const uint8_t *DataEnd = DataStart;

  uint64_t EHFrameAddrOffset = Offset;
  uint64_t EHFrameAddress = readEncodedPointer(
      DataEnd, EhFrameAddrEncoding,
      FrameHdrAddress + Offset - (uintptr_t)DataEnd, FrameHdrAddress);
  Offset += DataEnd - DataStart;

  DataStart = reinterpret_cast<const uint8_t *>(
      HDRData.getData().substr(Offset).data());
  DataEnd = DataStart;
  uint64_t FDECountOffset = Offset;
  uint64_t FDECount = readEncodedPointer(
      DataEnd, FDECntEncoding, FrameHdrAddress + Offset - (uintptr_t)DataEnd,
      FrameHdrAddress);
  Offset += DataEnd - DataStart;

  assert(FDECount > 0 && "Empty binary search table in .eh_frame_hdr!");
  assert(EhFrameAddrEncoding == (DW_EH_PE_pcrel | DW_EH_PE_sdata4) &&
         "Don't know how to handle other .eh_frame address encoding!");
  assert(FDECntEncoding == DW_EH_PE_udata4 &&
         "Don't know how to thandle other .eh_frame_hdr encoding!");
  assert(TableEncoding == (DW_EH_PE_datarel | DW_EH_PE_sdata4) &&
         "Don't know how to handle other .eh_frame_hdr encoding!");

  // Update .eh_frame address
  // Write address using signed 4-byte pc-relative encoding
  DEBUG(dbgs() << format("CFIReaderWriter: Patching .eh_frame_hdr contents "
                         "(.eh_frame pointer) with %08x\n",
                         EHFrameAddress));
  int64_t RealOffset = EHFrameAddress - EHFrameAddrOffset - NewFrameHdrAddress;
  assert(isInt<32>(RealOffset));
  support::ulittle32_t::ref(FrameHdrContents.data() + EHFrameAddrOffset) =
    RealOffset;

  // Offset now points to the binary search table. Update it.
  uint64_t LastPC = 0;
  for (uint64_t I = 0; I != FDECount; ++I) {
    assert(HDRData.isValidOffset(Offset) &&
           ".eh_frame_hdr table finished earlier than we expected");
    DataStart = reinterpret_cast<const uint8_t *>(
        HDRData.getData().substr(Offset).data());
    DataEnd = DataStart;
    uint64_t InitialPCOffset = Offset;
    uint64_t InitialPC = readEncodedPointer(
        DataEnd, TableEncoding, FrameHdrAddress + Offset - (uintptr_t)DataEnd,
        FrameHdrAddress);
    LastPC = InitialPC;
    Offset += DataEnd - DataStart;

    uint64_t FDEPtrOffset = Offset;
    DataStart = reinterpret_cast<const uint8_t *>(
        HDRData.getData().substr(Offset).data());
    DataEnd = DataStart;
    // Advance Offset past FDEPtr
    uint64_t FDEPtr = readEncodedPointer(
        DataEnd, TableEncoding, FrameHdrAddress + Offset - (uintptr_t)DataEnd,
        FrameHdrAddress);
    Offset += DataEnd - DataStart;

    // Update InitialPC according to new eh_frame_hdr address
    // Write using signed 4-byte "data relative" (relative to .eh_frame_addr)
    // encoding
    int64_t RealOffset = InitialPC - NewFrameHdrAddress;
    assert(isInt<32>(RealOffset));
    support::ulittle32_t::ref(FrameHdrContents.data() + InitialPCOffset) =
        RealOffset;

    if (uint64_t NewPtr = PCToFDE[InitialPC])
      RealOffset = NewPtr - NewFrameHdrAddress;
    else
      RealOffset = FDEPtr - NewFrameHdrAddress;

    assert(isInt<32>(RealOffset));
    DEBUG(dbgs() << format("CFIReaderWriter: Patching .eh_frame_hdr contents "
                           "@offset %08x with new FDE ptr %08x\n",
                           FDEPtrOffset, RealOffset + NewFrameHdrAddress));
    support::ulittle32_t::ref(FrameHdrContents.data() + FDEPtrOffset) =
      RealOffset;
  }
  // Add new entries (for cold function parts)
  uint64_t ExtraEntries = 0;
  for (auto I = PCToFDE.upper_bound(LastPC), E = PCToFDE.end(); I != E; ++I) {
    ++ExtraEntries;
  }
  if (ExtraEntries == 0)
    return;
  FrameHdrContents.resize(FrameHdrContents.size() + (ExtraEntries * 8));
  // Update FDE count
  DEBUG(dbgs() << "CFIReaderWriter: Updating .eh_frame_hdr FDE count from "
               << FDECount << " to " << (FDECount + ExtraEntries) << "\n");
  support::ulittle32_t::ref(FrameHdrContents.data() + FDECountOffset) =
      FDECount + ExtraEntries;

  for (auto I = PCToFDE.upper_bound(LastPC), E = PCToFDE.end(); I != E; ++I) {
    // Write PC
    DEBUG(dbgs() << format("CFIReaderWriter: Writing extra FDE entry for PC "
                           "0x%x, FDE pointer 0x%x\n",
                           I->first, I->second));
    uint64_t InitialPC = I->first;
    int64_t RealOffset = InitialPC - NewFrameHdrAddress;
    assert(isInt<32>(RealOffset));
    support::ulittle32_t::ref(FrameHdrContents.data() + Offset) = RealOffset;
    Offset += 4;

    // Write FDE pointer
    uint64_t FDEPtr = I->second;
    RealOffset = FDEPtr - NewFrameHdrAddress;
    assert(isInt<32>(RealOffset));
    support::ulittle32_t::ref(FrameHdrContents.data() + Offset) = RealOffset;
    Offset += 4;
  }
}

} // namespace flo
} // namespace llvm
