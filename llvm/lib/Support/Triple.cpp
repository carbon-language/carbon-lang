//===--- Triple.cpp - Target triple helper class --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Triple.h"
#include <cassert>
#include <cstring>
using namespace llvm;

//

const char *Triple::getArchTypeName(ArchType Kind) {
  switch (Kind) {
  case InvalidArch: return "<invalid>";
  case UnknownArch: return "unknown";

  case x86: return "i386";
  case x86_64: return "x86_64";
  case ppc: return "powerpc";
  case ppc64: return "powerpc64";
  }

  return "<invalid>";
}

const char *Triple::getVendorTypeName(VendorType Kind) {
  switch (Kind) {
  case UnknownVendor: return "unknown";

  case Apple: return "apple";
  case PC: return "PC";
  }

  return "<invalid>";
}

const char *Triple::getOSTypeName(OSType Kind) {
  switch (Kind) {
  case UnknownOS: return "unknown";

  case Darwin: return "darwin";
  case FreeBSD: return "freebsd";
  case Linux: return "linux";
  }

  return "<invalid>";
}

//

void Triple::Parse() const {
  assert(!isInitialized() && "Invalid parse call.");

  std::string ArchName = getArchName();
  if (ArchName.size() == 4 && ArchName[0] == 'i' && 
      ArchName[2] == '8' && ArchName[3] == '6')
    Arch = x86;
  else if (ArchName == "amd64" || ArchName == "x86_64")
    Arch = x86_64;
  else if (ArchName == "powerpc")
    Arch = ppc;
  else if (ArchName == "powerpc64")
    Arch = ppc64;
  else
    Arch = UnknownArch;

  std::string VendorName = getVendorName();
  if (VendorName == "apple")
    Vendor = Apple;
  else if (VendorName == "pc")
    Vendor = PC;
  else
    Vendor = UnknownVendor;

  std::string OSName = getOSName();
  if (memcmp(&OSName[0], "darwin", 6) == 0)
    OS = Darwin;
  else if (memcmp(&OSName[0], "freebsd", 7) == 0)
    OS = FreeBSD;
  else if (memcmp(&OSName[0], "linux", 5) == 0)
    OS = Linux;
  else
    OS = UnknownOS;

  assert(isInitialized() && "Failed to initialize!");
}

static std::string extract(const std::string &A,
                           std::string::size_type begin,
                           std::string::size_type end) {
  if (begin == std::string::npos)
    return "";
  if (end == std::string::npos)
    return A.substr(begin);
  return A.substr(begin, end - begin);
}

static std::string extract1(const std::string &A,
                           std::string::size_type begin,
                           std::string::size_type end) {
  if (begin == std::string::npos || begin == end)
    return "";
  return extract(A, begin + 1, end);
}

std::string Triple::getArchName() const {
  std::string Tmp = Data;
  return extract(Tmp, 0, Tmp.find('-'));
}

std::string Triple::getVendorName() const {
  std::string Tmp = Data;
  Tmp = extract1(Tmp, Tmp.find('-'), std::string::npos);
  return extract(Tmp, 0, Tmp.find('-'));
}

std::string Triple::getOSName() const {
  std::string Tmp = Data;
  Tmp = extract1(Tmp, Tmp.find('-'), std::string::npos);
  Tmp = extract1(Tmp, Tmp.find('-'), std::string::npos);
  return extract(Tmp, 0, Tmp.find('-'));
}

std::string Triple::getEnvironmentName() const {
  std::string Tmp = Data;
  Tmp = extract1(Tmp, Tmp.find('-'), std::string::npos);
  Tmp = extract1(Tmp, Tmp.find('-'), std::string::npos);
  Tmp = extract1(Tmp, Tmp.find('-'), std::string::npos);
  return extract(Tmp, 0, std::string::npos);
}

std::string Triple::getOSAndEnvironmentName() const {
  std::string Tmp = Data;
  Tmp = extract1(Tmp, Tmp.find('-'), std::string::npos);
  Tmp = extract1(Tmp, Tmp.find('-'), std::string::npos);
  return extract(Tmp, 0, std::string::npos);
}

void Triple::setTriple(const std::string &Str) {
  Data = Str;
  Arch = InvalidArch;
}

void Triple::setArch(ArchType Kind) {
  setArchName(getArchTypeName(Kind));
}

void Triple::setVendor(VendorType Kind) {
  setVendorName(getVendorTypeName(Kind));
}

void Triple::setOS(OSType Kind) {
  setOSName(getOSTypeName(Kind));
}

void Triple::setArchName(const std::string &Str) {
  setTriple(Str + "-" + getVendorName() + "-" + getOSAndEnvironmentName());
}

void Triple::setVendorName(const std::string &Str) {
  setTriple(getArchName() + "-" + Str + "-" + getOSAndEnvironmentName());
}

void Triple::setOSName(const std::string &Str) {
  if (hasEnvironment())
    setTriple(getArchName() + "-" + getVendorName() + "-" + Str +
              "-" + getEnvironmentName());
  else
    setTriple(getArchName() + "-" + getVendorName() + "-" + Str);
}

void Triple::setEnvironmentName(const std::string &Str) {
  setTriple(getArchName() + "-" + getVendorName() + "-" + getOSName() + 
            "-" + Str);
}

void Triple::setOSAndEnvironmentName(const std::string &Str) {
  setTriple(getArchName() + "-" + getVendorName() + "-" + Str);
}
