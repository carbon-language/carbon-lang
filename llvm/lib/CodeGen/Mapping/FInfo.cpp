#include "llvm/Reoptimizer/Mapping/FInfo.h"
#include "llvm/Pass.h"
#include "llvm/Module.h"

namespace {
  class FunctionInfo : public Pass {
    std::ostream &Out;
  public:
    FunctionInfo(std::ostream &out) : Out(out){}
    const char* getPassName() const{ return "Sparc FunctionInfo"; }
    bool run(Module &M);
  private:
    void writePrologue(const char *area, const char *label);
    void writeEpilogue(const char *area, const char *label);
                       
  };
}

Pass *getFunctionInfo(std::ostream &out){
  return new FunctionInfo(out);
}

bool FunctionInfo::run(Module &M){
  unsigned f;
  
  writePrologue("FUNCTION MAP", "FunctionBB");
  f=0;
  for(Module::iterator FI=M.begin(), FE=M.end(); FE!=FI; ++FI){
    if(FI->isExternal()) continue;
    Out << "\t.xword BBMIMap"<<f<<"\n";
    ++f;
  }
  writeEpilogue("FUNCTION MAP", "FunctionBB");
  
  writePrologue("FUNCTION MAP", "FunctionLI");
  f=0;
  for(Module::iterator FI=M.begin(), FE=M.end(); FE!=FI; ++FI){
    if(FI->isExternal()) continue;
    Out << "\t.xword LMIMap"<<f<<"\n";
    ++f;
  }
  writeEpilogue("FUNCTION MAP", "FunctionLI");
  
  
  return false;
}


void FunctionInfo::writePrologue(const char *area,
				    const char *label){
  Out << "\n\n\n!"<<area<<"\n";   
  Out << "\t.section \".rodata\"\n\t.align 8\n";  
  Out << "\t.global "<<label<<"\n";    
  Out << "\t.type "<<label<<",#object\n"; 
  Out << label<<":\n"; 
  //Out << "\t.word .end_"<<label<<"-"<<label<<"\n";
}

void FunctionInfo::writeEpilogue(const char *area,
				    const char *label){
  Out << ".end_" << label << ":\n";    
  Out << "\t.size " << label << ", .end_" 
      << label << "-" << label << "\n\n\n\n";
  
  //Out << "\n\n!" << area << " Length\n";
  //Out << "\t.section \".bbdata\",#alloc,#write\n";                                     
  //Out << "\t.global " << label << "_length\n";          
  //Out << "\t.align 4\n";
  //Out << "\t.type " << label << "_length,#object\n";
  //Out << "\t.size "<< label <<"_length,4\n";
  //Out << label <<" _length:\n";
  //Out << "\t.word\t.end_"<<label<<"-"<<label<<"\n\n\n\n";               
}
