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
  SmallString<256> Vec;
  toVector(Vec);
  return std::string(Vec.begin(), Vec.end());
}

void Twine::toVector(SmallVectorImpl<char> &Out) const {
  raw_svector_ostream OS(Out);
  print(OS);
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
    OS << *static_cast<const unsigned int*>(Ptr);
    break;
  case Twine::DecIKind:
    OS << *static_cast<const int*>(Ptr);
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
    OS << "decUI:\"" << *static_cast<const unsigned int*>(Ptr) << "\"";
    break;
  case Twine::DecIKind:
    OS << "decI:\"" << *static_cast<const int*>(Ptr) << "\"";
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
