//===- lib/Target/SparcV9/MappingInfo.h -------------------------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// Data structures to support the Reoptimizer's Instruction-to-MachineInstr
// mapping information gatherer.
//
//===----------------------------------------------------------------------===//

#ifndef MAPPINGINFO_H
#define MAPPINGINFO_H

#include <iosfwd>
#include <vector>
#include <string>

namespace llvm {

class Pass;

Pass *getMappingInfoAsmPrinterPass(std::ostream &out);
Pass *createInternalGlobalMapperPass();

class MappingInfo {
  struct byteVector : public std::vector <unsigned char> {
    void dumpAssembly (std::ostream &Out);
  };
  std::string comment;
  std::string symbolPrefix;
  unsigned functionNumber;
  byteVector bytes;
public:
  void outByte (unsigned char b) { bytes.push_back (b); }
  MappingInfo (std::string Comment, std::string SymbolPrefix,
	           unsigned FunctionNumber) : comment(Comment),
        	   symbolPrefix(SymbolPrefix), functionNumber(FunctionNumber) {}
  void dumpAssembly (std::ostream &Out);
  unsigned char *getBytes (unsigned &length) {
	length = bytes.size(); return &bytes[0];
  }
};

} // End llvm namespace

#endif
