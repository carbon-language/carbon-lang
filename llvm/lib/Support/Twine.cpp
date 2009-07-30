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
  // FIXME: This is very inefficient, since we are creating a large raw_ostream
  // buffer -- hitting malloc, which we were supposed to avoid -- all when we
  // have this pretty little small vector available.
  //
  // The best way to fix this is to make raw_svector_ostream do the right thing
  // and be efficient, by augmenting the base raw_ostream with the ability to
  // have the buffer managed by a concrete implementation.
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
  case Twine::UDec32Kind:
    OS << *static_cast<const uint32_t*>(Ptr);
    break;
  case Twine::SDec32Kind:
    OS << *static_cast<const int32_t*>(Ptr);
    break;
  case Twine::UDec64Kind:
    OS << *static_cast<const uint64_t*>(Ptr);
    break;
  case Twine::SDec64Kind:
    OS << *static_cast<const int64_t*>(Ptr);
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
  case Twine::UDec32Kind:
    OS << "udec32:" << static_cast<const uint64_t*>(Ptr) << "\"";
    break;
  case Twine::SDec32Kind:
    OS << "sdec32:" << static_cast<const int64_t*>(Ptr) << "\"";
    break;
  case Twine::UDec64Kind:
    OS << "udec64:" << static_cast<const uint64_t*>(Ptr) << "\"";
    break;
  case Twine::SDec64Kind:
    OS << "sdec64:" << static_cast<const int64_t*>(Ptr) << "\"";
    break;
  case Twine::UHexKind:
    OS << "uhex:" << static_cast<const uint64_t*>(Ptr) << "\"";
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
