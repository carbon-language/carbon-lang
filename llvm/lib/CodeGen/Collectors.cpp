//===-- Collectors.cpp - Garbage collector registry -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the static data members of the CollectorRegistry class.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/Collectors.h"

using namespace llvm;

template<> CollectorRegistry::node *CollectorRegistry::Head = 0;
template<> CollectorRegistry::node *CollectorRegistry::Tail = 0;
template<> CollectorRegistry::listener *CollectorRegistry::ListenerHead = 0;
template<> CollectorRegistry::listener *CollectorRegistry::ListenerTail = 0;

template<> GCMetadataPrinterRegistry::node *GCMetadataPrinterRegistry::Head = 0;
template<> GCMetadataPrinterRegistry::node *GCMetadataPrinterRegistry::Tail = 0;
template<> GCMetadataPrinterRegistry::listener *
GCMetadataPrinterRegistry::ListenerHead = 0;
template<> GCMetadataPrinterRegistry::listener *
GCMetadataPrinterRegistry::ListenerTail = 0;
