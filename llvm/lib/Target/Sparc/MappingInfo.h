//===- llvm/Reoptimizer/Mapping/MappingInfo.h ------------------*- C++ -*--=////
//
// Data structures to support the Reoptimizer's Instruction-to-MachineInstr
// mapping information gatherer.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_REOPTIMIZER_MAPPING_MAPPINGINFO_H
#define LLVM_REOPTIMIZER_MAPPING_MAPPINGINFO_H

#include <iosfwd>
#include <vector>
#include <string>
class Pass;

Pass *getMappingInfoCollector(std::ostream &out);

class MappingInfo {
  class byteVector : public std::vector <unsigned char> {
  public:
	void dumpAssembly (std::ostream &Out);
  };
  std::string comment;
  std::string symbolPrefix;
  unsigned functionNumber;
  byteVector bytes;
public:
  void outByte (unsigned char b) { bytes.push_back (b); }
  MappingInfo (std::string _comment, std::string _symbolPrefix,
	           unsigned _functionNumber) : comment(_comment),
        	   symbolPrefix(_symbolPrefix), functionNumber(_functionNumber) { }
  void dumpAssembly (std::ostream &Out);
  unsigned char *getBytes (unsigned int &length) {
	length = bytes.size(); return &bytes[0];
  }
};

#endif
