//= ScanfFormatString.cpp - Analysis of printf format strings --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Handling of format string in scanf and friends.  The structure of format
// strings for fscanf() are described in C99 7.19.6.2.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Analyses/FormatString.h"
#include "FormatStringParsing.h"

using clang::analyze_format_string::ArgTypeResult;
using clang::analyze_format_string::FormatStringHandler;
using clang::analyze_format_string::LengthModifier;
using clang::analyze_format_string::OptionalAmount;
using clang::analyze_format_string::ConversionSpecifier;
using clang::analyze_scanf::ScanfConversionSpecifier;
using clang::analyze_scanf::ScanfSpecifier;
using clang::UpdateOnReturn;

typedef clang::analyze_format_string::SpecifierResult<ScanfSpecifier>
        ScanfSpecifierResult;

static bool ParseScanList(FormatStringHandler &H,
                          ScanfConversionSpecifier &CS,
                          const char *&Beg, const char *E) {
  const char *I = Beg;
  const char *start = I - 1;
  UpdateOnReturn <const char*> UpdateBeg(Beg, I);

  // No more characters?
  if (I == E) {
    H.HandleIncompleteScanList(start, I);
    return true;
  }
  
  // Special case: ']' is the first character.
  if (*I == ']') {
    if (++I == E) {
      H.HandleIncompleteScanList(start, I - 1);
      return true;
    }
  }

  // Look for a ']' character which denotes the end of the scan list.
  while (*I != ']') {
    if (++I == E) {
      H.HandleIncompleteScanList(start, I - 1);
      return true;
    }
  }    

  CS.setEndScanList(I);
  return false;
}

// FIXME: Much of this is copy-paste from ParsePrintfSpecifier.
// We can possibly refactor.
static ScanfSpecifierResult ParseScanfSpecifier(FormatStringHandler &H,
                                                const char *&Beg,
                                                const char *E,
                                                unsigned &argIndex) {
  
  using namespace clang::analyze_scanf;
  const char *I = Beg;
  const char *Start = 0;
  UpdateOnReturn <const char*> UpdateBeg(Beg, I);

    // Look for a '%' character that indicates the start of a format specifier.
  for ( ; I != E ; ++I) {
    char c = *I;
    if (c == '\0') {
        // Detect spurious null characters, which are likely errors.
      H.HandleNullChar(I);
      return true;
    }
    if (c == '%') {
      Start = I++;  // Record the start of the format specifier.
      break;
    }
  }
  
    // No format specifier found?
  if (!Start)
    return false;
  
  if (I == E) {
      // No more characters left?
    H.HandleIncompleteSpecifier(Start, E - Start);
    return true;
  }
  
  ScanfSpecifier FS;
  if (ParseArgPosition(H, FS, Start, I, E))
    return true;

  if (I == E) {
      // No more characters left?
    H.HandleIncompleteSpecifier(Start, E - Start);
    return true;
  }
  
  // Look for '*' flag if it is present.
  if (*I == '*') {
    FS.setSuppressAssignment(I);
    if (++I == E) {
      H.HandleIncompleteSpecifier(Start, E - Start);
      return true;
    }
  }
  
  // Look for the field width (if any).  Unlike printf, this is either
  // a fixed integer or isn't present.
  const OptionalAmount &Amt = clang::analyze_format_string::ParseAmount(I, E);
  if (Amt.getHowSpecified() != OptionalAmount::NotSpecified) {
    assert(Amt.getHowSpecified() == OptionalAmount::Constant);
    FS.setFieldWidth(Amt);

    if (I == E) {
      // No more characters left?
      H.HandleIncompleteSpecifier(Start, E - Start);
      return true;
    }
  }
  
  // Look for the length modifier.
  if (ParseLengthModifier(FS, I, E) && I == E) {
      // No more characters left?
    H.HandleIncompleteSpecifier(Start, E - Start);
    return true;
  }
  
  // Detect spurious null characters, which are likely errors.
  if (*I == '\0') {
    H.HandleNullChar(I);
    return true;
  }
  
  // Finally, look for the conversion specifier.
  const char *conversionPosition = I++;
  ScanfConversionSpecifier::Kind k = ScanfConversionSpecifier::InvalidSpecifier;
  switch (*conversionPosition) {
    default:
      break;
    case '%': k = ConversionSpecifier::PercentArg;   break;
    case 'A': k = ConversionSpecifier::AArg; break;
    case 'E': k = ConversionSpecifier::EArg; break;
    case 'F': k = ConversionSpecifier::FArg; break;
    case 'G': k = ConversionSpecifier::GArg; break;
    case 'X': k = ConversionSpecifier::XArg; break;
    case 'a': k = ConversionSpecifier::aArg; break;
    case 'd': k = ConversionSpecifier::dArg; break;
    case 'e': k = ConversionSpecifier::eArg; break;
    case 'f': k = ConversionSpecifier::fArg; break;
    case 'g': k = ConversionSpecifier::gArg; break;
    case 'i': k = ConversionSpecifier::iArg; break;
    case 'n': k = ConversionSpecifier::nArg; break;
    case 'c': k = ConversionSpecifier::cArg; break;
    case 'C': k = ConversionSpecifier::CArg; break;
    case 'S': k = ConversionSpecifier::SArg; break;
    case '[': k = ConversionSpecifier::ScanListArg; break;
    case 'u': k = ConversionSpecifier::uArg; break;
    case 'x': k = ConversionSpecifier::xArg; break;
    case 'o': k = ConversionSpecifier::oArg; break;
    case 's': k = ConversionSpecifier::sArg; break;
    case 'p': k = ConversionSpecifier::pArg; break;
  }
  ScanfConversionSpecifier CS(conversionPosition, k);
  if (k == ScanfConversionSpecifier::ScanListArg) {
    if (!ParseScanList(H, CS, I, E))
      return true;
  }
  FS.setConversionSpecifier(CS);
  if (CS.consumesDataArgument() && !FS.getSuppressAssignment()
      && !FS.usesPositionalArg())
    FS.setArgIndex(argIndex++);
  
  // FIXME: '%' and '*' doesn't make sense.  Issue a warning.
  // FIXME: 'ConsumedSoFar' and '*' doesn't make sense.
  
  if (k == ScanfConversionSpecifier::InvalidSpecifier) {
    // Assume the conversion takes one argument.
    return !H.HandleInvalidScanfConversionSpecifier(FS, Beg, I - Beg);
  }
  return ScanfSpecifierResult(Start, FS);
}
  
bool clang::analyze_format_string::ParseScanfString(FormatStringHandler &H,
                                                    const char *I,
                                                    const char *E) {
  
  unsigned argIndex = 0;
  
  // Keep looking for a format specifier until we have exhausted the string.
  while (I != E) {
    const ScanfSpecifierResult &FSR = ParseScanfSpecifier(H, I, E, argIndex);
    // Did a fail-stop error of any kind occur when parsing the specifier?
    // If so, don't do any more processing.
    if (FSR.shouldStop())
      return true;;
      // Did we exhaust the string or encounter an error that
      // we can recover from?
    if (!FSR.hasValue())
      continue;
      // We have a format specifier.  Pass it to the callback.
    if (!H.HandleScanfSpecifier(FSR.getValue(), FSR.getStart(),
                                I - FSR.getStart())) {
      return true;
    }
  }
  assert(I == E && "Format string not exhausted");
  return false;
}


