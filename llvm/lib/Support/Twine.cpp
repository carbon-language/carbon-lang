//===-- Twine.cpp - Fast Temporary String Concatenation -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

std::string Twine::str() const {
  std::string Res;
  raw_string_ostream OS(Res);
  print(OS);
  return Res;
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
  case Twine::CStringKind: 
    OS << static_cast<const char*>(Ptr); 
    break;
  case Twine::StdStringKind:
    OS << *static_cast<const std::string*>(Ptr); 
    break;
  case Twine::StringRefKind:
    OS << *static_cast<const StringRef*>(Ptr); 
    break;
  case Twine::TwineKind:
    static_cast<const Twine*>(Ptr)->print(OS); 
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
  case Twine::CStringKind:
    OS << "cstring:\"" 
       << static_cast<const char*>(Ptr) << "\""; 
    break;
  case Twine::StdStringKind:
    OS << "std::string:\"" 
       << *static_cast<const std::string*>(Ptr) << "\""; 
    break;
  case Twine::StringRefKind:
    OS << "stringref:\"" 
       << *static_cast<const StringRef*>(Ptr) << "\""; 
    break;
  case Twine::TwineKind:
    OS << "rope:";
    static_cast<const Twine*>(Ptr)->printRepr(OS);
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
  print(llvm::errs());
}

void Twine::dumpRepr() const {
  printRepr(llvm::errs());
}
