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

STATISTIC(NumLSDAs, "Number of all LSDAs");
STATISTIC(NumTrivialLSDAs,
          "Number of LSDAs with single call site without landing pad or action");

using namespace llvm::dwarf;

namespace llvm {
namespace flo {

namespace opts {

static cl::opt<bool>
PrintExceptions("print-exceptions",
                cl::desc("print exception handling data"),
                cl::Hidden);

} // namespace opts

namespace {

/// Read an unsigned LEB128 value from data, advancing it past the value.
uintptr_t readULEB128(const uint8_t *&Data) {
  uintptr_t Result = 0;
  uintptr_t Shift = 0;
  unsigned char Byte;

  do {
    Byte = *Data++;
    Result |= (Byte & 0x7f) << Shift;
    Shift += 7;
  } while (Byte & 0x80);

  return Result;
}

/// Read a signed LEB128 value from data, advancing it past the value.
uintptr_t readSLEB128(const uint8_t *&Data) {
  uintptr_t Result = 0;
  uintptr_t Shift = 0;
  unsigned char Byte;

  do {
    Byte = *Data++;
    Result |= (Byte & 0x7f) << Shift;
    Shift += 7;
  } while (Byte & 0x80);

  if ((Byte & 0x40) && (Shift < (sizeof(Result) << 3))) {
    Result |= (~0 << Shift);
  }

  return Result;
}

/// Read and return a T from data, advancing it past the read item.
template<typename T>
T readValue(const uint8_t *&Data) {
  T Val;
  memcpy(&Val, Data, sizeof(T));
  Data += sizeof(T);
  return Val;
}

/// Read an encoded DWARF value from data, advancing it past any data read. This
/// function was adapted from the ExceptionDemo.cpp example in llvm.
uintptr_t readEncodedPointer(const uint8_t *&Data, uint8_t Encoding) {
  uintptr_t Result = 0;
  auto const Start = Data;

  if (Encoding == DW_EH_PE_omit)
    return Result;

  // first get value
  switch (Encoding & 0x0F) {
  case DW_EH_PE_absptr:
    Result = readValue<uintptr_t>(Data);
    break;
  case DW_EH_PE_uleb128:
    Result = readULEB128(Data);
    break;
  case DW_EH_PE_sleb128:
    Result = readSLEB128(Data);
    break;
  case DW_EH_PE_udata2:
    Result = readValue<uint16_t>(Data);
    break;
  case DW_EH_PE_udata4:
    Result = readValue<uint32_t>(Data);
    break;
  case DW_EH_PE_udata8:
    Result = readValue<uint64_t>(Data);
    break;
  case DW_EH_PE_sdata2:
    Result = readValue<int16_t>(Data);
    break;
  case DW_EH_PE_sdata4:
    Result = readValue<int32_t>(Data);
    break;
  case DW_EH_PE_sdata8:
    Result = readValue<int64_t>(Data);
    break;
  default:
    llvm_unreachable("not implemented");
  }

  // then add relative offset
  switch (Encoding & 0x70) {
  case DW_EH_PE_absptr:
    // do nothing
    break;
  case DW_EH_PE_pcrel:
    Result += reinterpret_cast<uintptr_t>(Start);
    break;
  case DW_EH_PE_textrel:
  case DW_EH_PE_datarel:
  case DW_EH_PE_funcrel:
  case DW_EH_PE_aligned:
  default:
    llvm_unreachable("not implemented");
  }

  // then apply indirection
  if (Encoding & 0x80 /*DW_EH_PE_indirect*/) {
    Result = *((uintptr_t*)Result);
  }

  return Result;
}

} // namespace

// readLSDA is reading and dumping the whole .gcc_exception_table section
// at once.
//
// .gcc_except_table section contains a set of Language-Specific Data Areas
// which are basically exception handling tables. One LSDA per function.
// One important observation - you can't actually tell which function LSDA
// refers to, and most addresses are relative to the function start. So you
// have to start with parsing .eh_frame entries that refers to LSDA to obtain
// a function context.
//
// The best visual representation of the tables comprising LSDA and relationship
// between them is illustrated at:
//   http://mentorembedded.github.io/cxx-abi/exceptions.pdf
// Keep in mind that GCC implementation deviates slightly from that document.
//
// To summarize, there are 4 tables in LSDA: call site table, actions table,
// types table, and types index table (indirection). The main table contains
// call site entries. Each call site includes a range that can throw an exception,
// a handler (landing pad), and a reference to an entry in the action table.
// A handler and/or action could be 0. An action entry is in fact a head
// of a list of actions associated with a call site and an action table contains
// all such lists (it could be optimize to share list tails). Each action could be
// either to catch an exception of a given type, to perform a cleanup, or to
// propagate an exception after filtering it out (e.g. to make sure function
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
// For the purpose of rewriting exception handling tables, we can reuse action
// table, types table, and type index table in a binary format when type
// references are hard-coded absolute addresses. We still have to parse all the
// table to determine their size. We have to parse call site table and associate
// discovered information with actual call instructions and landing pad blocks.
void readLSDA(ArrayRef<uint8_t> LSDAData, BinaryContext &BC) {
  const uint8_t *Ptr = LSDAData.data();

  while (Ptr < LSDAData.data() + LSDAData.size()) {
    uint8_t LPStartEncoding = *Ptr++;
    // Some of LSDAs are aligned while other are not. We use the hack below
    // to work around 0-filled alignment. However it could also mean 
    // DW_EH_PE_absptr format.
    //
    // FIXME: the proper way to parse these tables is to get the pointer
    //        from .eh_frame and parse one entry at a time.
    while (!LPStartEncoding)
      LPStartEncoding = *Ptr++;
    if (opts::PrintExceptions) {
      errs() << "[LSDA at 0x"
             << Twine::utohexstr(reinterpret_cast<uint64_t>(Ptr-1)) << "]:\n";
    }

    ++NumLSDAs;
    bool IsTrivial = true;

    uintptr_t LPStart = 0;
    if (LPStartEncoding != DW_EH_PE_omit) {
      LPStart = readEncodedPointer(Ptr, LPStartEncoding);
    }

    uint8_t TTypeEncoding = *Ptr++;
    uintptr_t TTypeEnd = 0;
    if (TTypeEncoding != DW_EH_PE_omit) {
      TTypeEnd = readULEB128(Ptr);
    }

    if (opts::PrintExceptions) {
      errs() << "LPStart Encoding = " << (unsigned)LPStartEncoding << '\n';
      errs() << "LPStart = 0x" << Twine::utohexstr(LPStart) << '\n';
      errs() << "TType Encoding = " << (unsigned)TTypeEncoding << '\n';
      errs() << "TType End = " << TTypeEnd << '\n';
    }

    // Table to store list of indices in type table. Entries are uleb128s values.
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
    while (CallSitePtr < CallSiteTableEnd) {
      ++NumCallSites;
      uintptr_t Start = readEncodedPointer(CallSitePtr, CallSiteEncoding);
      uintptr_t Length = readEncodedPointer(CallSitePtr, CallSiteEncoding);
      uintptr_t LandingPad = readEncodedPointer(CallSitePtr, CallSiteEncoding);

      uintptr_t ActionEntry = readULEB128(CallSitePtr);
      uint64_t RangeBase = 0;
      if (opts::PrintExceptions) {
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
        errs() << "Call Site: [0x" << Twine::utohexstr(RangeBase + Start)
               << ", 0x" << Twine::utohexstr(RangeBase + Start + Length)
               << "); landing pad: 0x" << Twine::utohexstr(LPStart + LandingPad)
               << "; action entry: 0x" << Twine::utohexstr(ActionEntry) << "\n";
        if (ActionEntry != 0) {
          errs() << "    actions: ";
          const uint8_t *ActionPtr = ActionTableStart + ActionEntry - 1;
          long long ActionType;
          long long ActionNext;
          auto Sep = "";
          do {
            ActionType = readSLEB128(ActionPtr);
            auto Self = ActionPtr;
            ActionNext = readSLEB128(ActionPtr);
            errs() << Sep << "(" << ActionType << ", " << ActionNext << ") ";
            if (ActionType == 0) {
              errs() << "cleanup";
            } else if (ActionType > 0) {
              // It's an index into a type table.
              errs() << "catch type ";
              printType(ActionType, errs());
            } else { // ActionType < 0
              errs() << "filter exception types ";
              auto TSep = "";
              // ActionType is a negative byte offset into uleb128-encoded table
              // of indices with base 1.
              // E.g. -1 means offset 0, -2 is offset 1, etc. The indices are
              // encoded using uleb128 so we cannot directly dereference them.
              auto TypeIndexTablePtr = TypeIndexTableStart - ActionType - 1;
              while (auto Index = readULEB128(TypeIndexTablePtr)) {
                errs() << TSep;
                printType(Index, errs());
                TSep = ", ";
              }
              MaxTypeIndexTableOffset =
                  std::max(MaxTypeIndexTableOffset,
                           TypeIndexTablePtr - TypeIndexTableStart);
            }

            Sep = "; ";

            ActionPtr = Self + ActionNext;
          } while (ActionNext);
          errs() << '\n';
        }
      }

      if (LandingPad != 0 || ActionEntry != 0)
        IsTrivial = false;
    }
    Ptr = CallSiteTableEnd;

    if (NumCallSites > 1)
      IsTrivial = false;

    if (IsTrivial)
      ++NumTrivialLSDAs;

    if (opts::PrintExceptions)
      errs() << '\n';

    if (CallSiteTableLength == 0 || TTypeEnd == 0)
      continue;

    Ptr = TypeIndexTableStart + MaxTypeIndexTableOffset;
  }
}

} // namespace flo
} // namespace llvm
