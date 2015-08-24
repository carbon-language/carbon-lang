//===-- ubsan_diag.cc -----------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Diagnostic reporting for the UBSan runtime.
//
//===----------------------------------------------------------------------===//

#include "ubsan_platform.h"
#if CAN_SANITIZE_UB
#include "ubsan_diag.h"
#include "ubsan_init.h"
#include "ubsan_flags.h"
#include "sanitizer_common/sanitizer_placement_new.h"
#include "sanitizer_common/sanitizer_report_decorator.h"
#include "sanitizer_common/sanitizer_stacktrace.h"
#include "sanitizer_common/sanitizer_stacktrace_printer.h"
#include "sanitizer_common/sanitizer_suppressions.h"
#include "sanitizer_common/sanitizer_symbolizer.h"
#include <stdio.h>

using namespace __ubsan;

static void MaybePrintStackTrace(uptr pc, uptr bp) {
  // We assume that flags are already parsed, as UBSan runtime
  // will definitely be called when we print the first diagnostics message.
  if (!flags()->print_stacktrace)
    return;
  // We can only use slow unwind, as we don't have any information about stack
  // top/bottom.
  // FIXME: It's better to respect "fast_unwind_on_fatal" runtime flag and
  // fetch stack top/bottom information if we have it (e.g. if we're running
  // under ASan).
  if (StackTrace::WillUseFastUnwind(false))
    return;
  BufferedStackTrace stack;
  stack.Unwind(kStackTraceMax, pc, bp, 0, 0, 0, false);
  stack.Print();
}

static const char *ConvertTypeToString(ErrorType Type) {
  switch (Type) {
#define UBSAN_CHECK(Name, SummaryKind, FlagName)                               \
  case ErrorType::Name:                                                        \
    return SummaryKind;
#include "ubsan_checks.inc"
#undef UBSAN_CHECK
  }
  UNREACHABLE("unknown ErrorType!");
}

static void MaybeReportErrorSummary(Location Loc, ErrorType Type) {
  if (!common_flags()->print_summary)
    return;
  if (!flags()->report_error_type)
    Type = ErrorType::GenericUB;
  const char *ErrorKind = ConvertTypeToString(Type);
  if (Loc.isSourceLocation()) {
    SourceLocation SLoc = Loc.getSourceLocation();
    if (!SLoc.isInvalid()) {
      AddressInfo AI;
      AI.file = internal_strdup(SLoc.getFilename());
      AI.line = SLoc.getLine();
      AI.column = SLoc.getColumn();
      AI.function = internal_strdup("");  // Avoid printing ?? as function name.
      ReportErrorSummary(ErrorKind, AI);
      AI.Clear();
      return;
    }
  } else if (Loc.isSymbolizedStack()) {
    const AddressInfo &AI = Loc.getSymbolizedStack()->info;
    ReportErrorSummary(ErrorKind, AI);
    return;
  }
  ReportErrorSummary(ErrorKind);
}

namespace {
class Decorator : public SanitizerCommonDecorator {
 public:
  Decorator() : SanitizerCommonDecorator() {}
  const char *Highlight() const { return Green(); }
  const char *EndHighlight() const { return Default(); }
  const char *Note() const { return Black(); }
  const char *EndNote() const { return Default(); }
};
}

SymbolizedStack *__ubsan::getSymbolizedLocation(uptr PC) {
  InitAsStandaloneIfNecessary();
  return Symbolizer::GetOrInit()->SymbolizePC(PC);
}

Diag &Diag::operator<<(const TypeDescriptor &V) {
  return AddArg(V.getTypeName());
}

Diag &Diag::operator<<(const Value &V) {
  if (V.getType().isSignedIntegerTy())
    AddArg(V.getSIntValue());
  else if (V.getType().isUnsignedIntegerTy())
    AddArg(V.getUIntValue());
  else if (V.getType().isFloatTy())
    AddArg(V.getFloatValue());
  else
    AddArg("<unknown>");
  return *this;
}

/// Hexadecimal printing for numbers too large for Printf to handle directly.
static void PrintHex(UIntMax Val) {
#if HAVE_INT128_T
  Printf("0x%08x%08x%08x%08x",
          (unsigned int)(Val >> 96),
          (unsigned int)(Val >> 64),
          (unsigned int)(Val >> 32),
          (unsigned int)(Val));
#else
  UNREACHABLE("long long smaller than 64 bits?");
#endif
}

static void renderLocation(Location Loc) {
  InternalScopedString LocBuffer(1024);
  switch (Loc.getKind()) {
  case Location::LK_Source: {
    SourceLocation SLoc = Loc.getSourceLocation();
    if (SLoc.isInvalid())
      LocBuffer.append("<unknown>");
    else
      RenderSourceLocation(&LocBuffer, SLoc.getFilename(), SLoc.getLine(),
                           SLoc.getColumn(), common_flags()->symbolize_vs_style,
                           common_flags()->strip_path_prefix);
    break;
  }
  case Location::LK_Memory:
    LocBuffer.append("%p", Loc.getMemoryLocation());
    break;
  case Location::LK_Symbolized: {
    const AddressInfo &Info = Loc.getSymbolizedStack()->info;
    if (Info.file) {
      RenderSourceLocation(&LocBuffer, Info.file, Info.line, Info.column,
                           common_flags()->symbolize_vs_style,
                           common_flags()->strip_path_prefix);
    } else if (Info.module) {
      RenderModuleLocation(&LocBuffer, Info.module, Info.module_offset,
                           common_flags()->strip_path_prefix);
    } else {
      LocBuffer.append("%p", Info.address);
    }
    break;
  }
  case Location::LK_Null:
    LocBuffer.append("<unknown>");
    break;
  }
  Printf("%s:", LocBuffer.data());
}

static void renderText(const char *Message, const Diag::Arg *Args) {
  for (const char *Msg = Message; *Msg; ++Msg) {
    if (*Msg != '%') {
      char Buffer[64];
      unsigned I;
      for (I = 0; Msg[I] && Msg[I] != '%' && I != 63; ++I)
        Buffer[I] = Msg[I];
      Buffer[I] = '\0';
      Printf(Buffer);
      Msg += I - 1;
    } else {
      const Diag::Arg &A = Args[*++Msg - '0'];
      switch (A.Kind) {
      case Diag::AK_String:
        Printf("%s", A.String);
        break;
      case Diag::AK_TypeName: {
        if (SANITIZER_WINDOWS)
          // The Windows implementation demangles names early.
          Printf("'%s'", A.String);
        else
          Printf("'%s'", Symbolizer::GetOrInit()->Demangle(A.String));
        break;
      }
      case Diag::AK_SInt:
        // 'long long' is guaranteed to be at least 64 bits wide.
        if (A.SInt >= INT64_MIN && A.SInt <= INT64_MAX)
          Printf("%lld", (long long)A.SInt);
        else
          PrintHex(A.SInt);
        break;
      case Diag::AK_UInt:
        if (A.UInt <= UINT64_MAX)
          Printf("%llu", (unsigned long long)A.UInt);
        else
          PrintHex(A.UInt);
        break;
      case Diag::AK_Float: {
        // FIXME: Support floating-point formatting in sanitizer_common's
        //        printf, and stop using snprintf here.
        char Buffer[32];
#if SANITIZER_WINDOWS
        sprintf_s(Buffer, sizeof(Buffer), "%Lg", (long double)A.Float);
#else
        snprintf(Buffer, sizeof(Buffer), "%Lg", (long double)A.Float);
#endif
        Printf("%s", Buffer);
        break;
      }
      case Diag::AK_Pointer:
        Printf("%p", A.Pointer);
        break;
      }
    }
  }
}

/// Find the earliest-starting range in Ranges which ends after Loc.
static Range *upperBound(MemoryLocation Loc, Range *Ranges,
                         unsigned NumRanges) {
  Range *Best = 0;
  for (unsigned I = 0; I != NumRanges; ++I)
    if (Ranges[I].getEnd().getMemoryLocation() > Loc &&
        (!Best ||
         Best->getStart().getMemoryLocation() >
         Ranges[I].getStart().getMemoryLocation()))
      Best = &Ranges[I];
  return Best;
}

static inline uptr subtractNoOverflow(uptr LHS, uptr RHS) {
  return (LHS < RHS) ? 0 : LHS - RHS;
}

static inline uptr addNoOverflow(uptr LHS, uptr RHS) {
  const uptr Limit = (uptr)-1;
  return (LHS > Limit - RHS) ? Limit : LHS + RHS;
}

/// Render a snippet of the address space near a location.
static void renderMemorySnippet(const Decorator &Decor, MemoryLocation Loc,
                                Range *Ranges, unsigned NumRanges,
                                const Diag::Arg *Args) {
  // Show at least the 8 bytes surrounding Loc.
  const unsigned MinBytesNearLoc = 4;
  MemoryLocation Min = subtractNoOverflow(Loc, MinBytesNearLoc);
  MemoryLocation Max = addNoOverflow(Loc, MinBytesNearLoc);
  MemoryLocation OrigMin = Min;
  for (unsigned I = 0; I < NumRanges; ++I) {
    Min = __sanitizer::Min(Ranges[I].getStart().getMemoryLocation(), Min);
    Max = __sanitizer::Max(Ranges[I].getEnd().getMemoryLocation(), Max);
  }

  // If we have too many interesting bytes, prefer to show bytes after Loc.
  const unsigned BytesToShow = 32;
  if (Max - Min > BytesToShow)
    Min = __sanitizer::Min(Max - BytesToShow, OrigMin);
  Max = addNoOverflow(Min, BytesToShow);

  if (!IsAccessibleMemoryRange(Min, Max - Min)) {
    Printf("<memory cannot be printed>\n");
    return;
  }

  // Emit data.
  for (uptr P = Min; P != Max; ++P) {
    unsigned char C = *reinterpret_cast<const unsigned char*>(P);
    Printf("%s%02x", (P % 8 == 0) ? "  " : " ", C);
  }
  Printf("\n");

  // Emit highlights.
  Printf(Decor.Highlight());
  Range *InRange = upperBound(Min, Ranges, NumRanges);
  for (uptr P = Min; P != Max; ++P) {
    char Pad = ' ', Byte = ' ';
    if (InRange && InRange->getEnd().getMemoryLocation() == P)
      InRange = upperBound(P, Ranges, NumRanges);
    if (!InRange && P > Loc)
      break;
    if (InRange && InRange->getStart().getMemoryLocation() < P)
      Pad = '~';
    if (InRange && InRange->getStart().getMemoryLocation() <= P)
      Byte = '~';
    char Buffer[] = { Pad, Pad, P == Loc ? '^' : Byte, Byte, 0 };
    Printf((P % 8 == 0) ? Buffer : &Buffer[1]);
  }
  Printf("%s\n", Decor.EndHighlight());

  // Go over the line again, and print names for the ranges.
  InRange = 0;
  unsigned Spaces = 0;
  for (uptr P = Min; P != Max; ++P) {
    if (!InRange || InRange->getEnd().getMemoryLocation() == P)
      InRange = upperBound(P, Ranges, NumRanges);
    if (!InRange)
      break;

    Spaces += (P % 8) == 0 ? 2 : 1;

    if (InRange && InRange->getStart().getMemoryLocation() == P) {
      while (Spaces--)
        Printf(" ");
      renderText(InRange->getText(), Args);
      Printf("\n");
      // FIXME: We only support naming one range for now!
      break;
    }

    Spaces += 2;
  }

  // FIXME: Print names for anything we can identify within the line:
  //
  //  * If we can identify the memory itself as belonging to a particular
  //    global, stack variable, or dynamic allocation, then do so.
  //
  //  * If we have a pointer-size, pointer-aligned range highlighted,
  //    determine whether the value of that range is a pointer to an
  //    entity which we can name, and if so, print that name.
  //
  // This needs an external symbolizer, or (preferably) ASan instrumentation.
}

Diag::~Diag() {
  // All diagnostics should be printed under report mutex.
  CommonSanitizerReportMutex.CheckLocked();
  Decorator Decor;
  Printf(Decor.Bold());

  renderLocation(Loc);

  switch (Level) {
  case DL_Error:
    Printf("%s runtime error: %s%s",
           Decor.Warning(), Decor.EndWarning(), Decor.Bold());
    break;

  case DL_Note:
    Printf("%s note: %s", Decor.Note(), Decor.EndNote());
    break;
  }

  renderText(Message, Args);

  Printf("%s\n", Decor.Default());

  if (Loc.isMemoryLocation())
    renderMemorySnippet(Decor, Loc.getMemoryLocation(), Ranges,
                        NumRanges, Args);
}

ScopedReport::ScopedReport(ReportOptions Opts, Location SummaryLoc,
                           ErrorType Type)
    : Opts(Opts), SummaryLoc(SummaryLoc), Type(Type) {
  InitAsStandaloneIfNecessary();
  CommonSanitizerReportMutex.Lock();
}

ScopedReport::~ScopedReport() {
  MaybePrintStackTrace(Opts.pc, Opts.bp);
  MaybeReportErrorSummary(SummaryLoc, Type);
  CommonSanitizerReportMutex.Unlock();
  if (Opts.DieAfterReport || flags()->halt_on_error)
    Die();
}

ALIGNED(64) static char suppression_placeholder[sizeof(SuppressionContext)];
static SuppressionContext *suppression_ctx = nullptr;
static const char kVptrCheck[] = "vptr_check";
static const char *kSuppressionTypes[] = { kVptrCheck };

void __ubsan::InitializeSuppressions() {
  CHECK_EQ(nullptr, suppression_ctx);
  suppression_ctx = new (suppression_placeholder) // NOLINT
      SuppressionContext(kSuppressionTypes, ARRAY_SIZE(kSuppressionTypes));
  suppression_ctx->ParseFromFile(flags()->suppressions);
}

bool __ubsan::IsVptrCheckSuppressed(const char *TypeName) {
  InitAsStandaloneIfNecessary();
  CHECK(suppression_ctx);
  Suppression *s;
  return suppression_ctx->Match(TypeName, kVptrCheck, &s);
}

#endif  // CAN_SANITIZE_UB
