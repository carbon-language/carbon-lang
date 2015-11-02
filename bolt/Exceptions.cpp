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
    assert(0 && "not implemented");
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
    assert(0 && "not implemented");
  }

  // then apply indirection
  if (Encoding & 0x80 /*DW_EH_PE_indirect*/) {
    Result = *((uintptr_t*)Result);
  }

  return Result;
}

} // namespace

void readLSDA(ArrayRef<uint8_t> LSDAData) {
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
    const uint8_t *NextLSDA = Ptr + TTypeEnd;

    if (opts::PrintExceptions) {
      errs() << "LPStart Encoding = " << (unsigned)LPStartEncoding << '\n';
      errs() << "LPStart = 0x" << Twine::utohexstr(LPStart) << '\n';
      errs() << "TType Encoding = " << (unsigned)TTypeEncoding << '\n';
      errs() << "TType End = " << TTypeEnd << '\n';
    }

    uint8_t       CallSiteEncoding = *Ptr++;
    uint32_t      CallSiteTableLength = readULEB128(Ptr);
    const uint8_t *CallSiteTableStart = Ptr;
    const uint8_t *CallSiteTableEnd = CallSiteTableStart + CallSiteTableLength;
    const uint8_t *CallSitePtr = CallSiteTableStart;

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
        errs() << "Call Site: [0x" << Twine::utohexstr(RangeBase + Start)
               << ", 0x" << Twine::utohexstr(RangeBase + Start + Length)
               << "); landing pad: 0x" << Twine::utohexstr(LPStart + LandingPad)
               << "; action entry: 0x" << Twine::utohexstr(ActionEntry) << "\n";
      }

      if (LandingPad != 0 || ActionEntry != 0)
        IsTrivial = false;
    }
    Ptr = CallSiteTableEnd;

    if (NumCallSites > 1)
      IsTrivial = false;

    if (opts::PrintExceptions)
      errs() << '\n';

    if (IsTrivial)
      ++NumTrivialLSDAs;

    if (CallSiteTableLength == 0 || TTypeEnd == 0)
      continue;

    const uint8_t *ActionPtr = Ptr;
    uintptr_t ActionOffset = 0;
    do {
      uintptr_t ActionType = readULEB128(ActionPtr);
      ActionOffset = readULEB128(ActionPtr);
      if (opts::PrintExceptions) {
        errs() << "ActionType: " << ActionType
               << "; ActionOffset: " << ActionOffset << "\n";
      }
    } while (ActionOffset != 0);

    if (opts::PrintExceptions)
      errs() << '\n';

    Ptr = NextLSDA;
  }
}

} // namespace flo
} // namespace llvm
