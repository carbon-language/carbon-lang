//===-- Twine.cpp - Fast Temporary String Concatenation -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Twine.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

std::string Twine::str() const {
  // If we're storing only a std::string, just return it.
  if (LHSKind == StdStringKind && RHSKind == EmptyKind)
    return *static_cast<const std::string*>(LHS);

  // Otherwise, flatten and copy the contents first.
  SmallString<256> Vec;
  return toStringRef(Vec).str();
}

void Twine::toVector(SmallVectorImpl<char> &Out) const {
  raw_svector_ostream OS(Out);
  print(OS);
}

StringRef Twine::toStringRef(SmallVectorImpl<char> &Out) const {
  if (isSingleStringRef())
    return getSingleStringRef();
  toVector(Out);
  return StringRef(Out.data(), Out.size());
}

StringRef Twine::toNullTerminatedStringRef(SmallVectorImpl<char> &Out) const {
  if (isUnary()) {
    switch (getLHSKind()) {
    case CStringKind:
      // Already null terminated, yay!
      return StringRef(static_cast<const char*>(LHS));
    case StdStringKind: {
      const std::string *str = static_cast<const std::string*>(LHS);
      return StringRef(str->c_str(), str->size());
    }
    default:
      break;
    }
  }
  toVector(Out);
  Out.push_back(0);
  Out.pop_back();
  return StringRef(Out.data(), Out.size());
}

void Twine::printOneChild(raw_ostream &OS, const void *Ptr,
                          NodeKind Kind) const {
  switch (Kind) {
  case Twine::NullKind: break;
  case Twine::EmptyKind: break;
  case Twine::TwineKind:
    static_cast<const Twine*>(Ptr)->print(OS);
    break;
  case Twine::CStringKind:
    OS << static_cast<const char*>(Ptr);
    break;
  case Twine::StdStringKind:
    OS << *static_cast<const std::string*>(Ptr);
    break;
  case Twine::StringRefKind:
    OS << *static_cast<const StringRef*>(Ptr);
    break;
  case Twine::DecUIKind:
    OS << (unsigned)(uintptr_t)Ptr;
    break;
  case Twine::DecIKind:
    OS << (int)(intptr_t)Ptr;
    break;
  case Twine::DecULKind:
    OS << *static_cast<const unsigned long*>(Ptr);
    break;
  case Twine::DecLKind:
    OS << *static_cast<const long*>(Ptr);
    break;
  case Twine::DecULLKind:
    OS << *static_cast<const unsigned long long*>(Ptr);
    break;
  case Twine::DecLLKind:
    OS << *static_cast<const long long*>(Ptr);
    break;
  case Twine::UHexKind:
    OS.write_hex(*static_cast<const uint64_t*>(Ptr));
    break;
  }
}

void Twine::printOneChildRepr(raw_ostream &OS, const void *Ptr,
                              NodeKind Kind) const {
  switch (Kind) {
  case Twine::NullKind:
    OS << "null"; break;
  case Twine::EmptyKind:
    OS << "empty"; break;
  case Twine::TwineKind:
    OS << "rope:";
    static_cast<const Twine*>(Ptr)->printRepr(OS);
    break;
  case Twine::CStringKind:
    OS << "cstring:\""
       << static_cast<const char*>(Ptr) << "\"";
    break;
  case Twine::StdStringKind:
    OS << "std::string:\""
       << static_cast<const std::string*>(Ptr) << "\"";
    break;
  case Twine::StringRefKind:
    OS << "stringref:\""
       << static_cast<const StringRef*>(Ptr) << "\"";
    break;
  case Twine::DecUIKind:
    OS << "decUI:\"" << (unsigned)(uintptr_t)Ptr << "\"";
    break;
  case Twine::DecIKind:
    OS << "decI:\"" << (int)(intptr_t)Ptr << "\"";
    break;
  case Twine::DecULKind:
    OS << "decUL:\"" << *static_cast<const unsigned long*>(Ptr) << "\"";
    break;
  case Twine::DecLKind:
    OS << "decL:\"" << *static_cast<const long*>(Ptr) << "\"";
    break;
  case Twine::DecULLKind:
    OS << "decULL:\"" << *static_cast<const unsigned long long*>(Ptr) << "\"";
    break;
  case Twine::DecLLKind:
    OS << "decLL:\"" << *static_cast<const long long*>(Ptr) << "\"";
    break;
  case Twine::UHexKind:
    OS << "uhex:\"" << static_cast<const uint64_t*>(Ptr) << "\"";
    break;
  }
}

void Twine::print(raw_ostream &OS) const {
  printOneChild(OS, LHS, getLHSKind());
  printOneChild(OS, RHS, getRHSKind());
}

void Twine::printRepr(raw_ostream &OS) const {
  OS << "(Twine ";
  printOneChildRepr(OS, LHS, getLHSKind());
  OS << " ";
  printOneChildRepr(OS, RHS, getRHSKind());
  OS << ")";
}

void Twine::dump() const {
  print(llvm::dbgs());
}

void Twine::dumpRepr() const {
  printRepr(llvm::dbgs());
}
