//===--- SerializationTest.cpp - Experimental Object Serialization --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Ted Kremenek and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements prototype code for serialization of objects in clang.
//  It is not intended yet for public use, but simply is a placeholder to
//  experiment with new serialization features.  Serialization will eventually
//  be integrated as a proper component of the clang libraries.
//
//===----------------------------------------------------------------------===//

#include "ASTConsumers.h"
#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "llvm/System/Path.h"
#include "llvm/Bitcode/BitstreamWriter.h"
#include <fstream>
#include <iostream>

using namespace clang;
using llvm::BitstreamWriter;
using std::cerr;
using std::cout;
using std::endl;
using std::flush;

namespace llvm {  
template<typename T> struct IntrospectionTrait {
  struct Flags { 
    enum { isPod = false, UniqueInstances = false, UniqueRefs = false };
  };
  
  template<typename Introspector>
  struct Ops {
    static inline void Introspect(T& X, Introspector& I) {
      assert (false && "Introspect not implemented.");
    }
  };
};
}

namespace {
class SerializationTest : public ASTConsumer {
  IdentifierTable* IdTable;
  unsigned MainFileID;
public:
  void Initialize(ASTContext& Context, unsigned mainFileID) {
    IdTable = &Context.Idents;
    MainFileID = mainFileID;
    RunSerializationTest();
  }
  
  void RunSerializationTest();
  bool WriteAll(llvm::sys::Path& Filename);
  
  virtual void HandleTopLevelDecl(Decl *D) {}
};

class Writer {
  std::vector<unsigned char> Buffer;
  BitstreamWriter Stream;
  std::ostream& Out;
public:
  
  enum { IdentifierTableBID = 0x8 };
  
  Writer(std::ostream& out) : Stream(Buffer), Out(out) {
    Buffer.reserve(256*1024);
    
    // Emit the file header.
    Stream.Emit((unsigned)'B', 8);
    Stream.Emit((unsigned)'C', 8);
    Stream.Emit(0xC, 4);
    Stream.Emit(0xF, 4);
    Stream.Emit(0xE, 4);
    Stream.Emit(0x0, 4);
  }
  
  ~Writer() {
    Out.write((char*)&Buffer.front(), Buffer.size());
    Out.flush();
  }
  
  template <typename T> inline void operator()(T& x) {
    llvm::IntrospectionTrait<T>::template Ops<Writer>::Introspect(x,*this);
  }
    
  template <typename T> inline void operator()(T& x, unsigned bits) {
    llvm::IntrospectionTrait<T>::template Ops<Writer>::Introspect(x,bits,*this);
  }
  
  template <typename T> inline void operator()(const T& x) {
    operator()(const_cast<T&>(x));
  }
  
  template <typename T> inline void operator()(const T& x, unsigned bits) {
    operator()(const_cast<T&>(x),bits);
  }  
  
  inline void operator()(bool X) { Stream.Emit(X,1); }
  inline void operator()(unsigned X) { Stream.Emit(X,32); }
  inline void operator()(unsigned X, unsigned bits, bool VBR=false) {
    if (VBR) Stream.Emit(X,bits);
    else Stream.Emit(X,bits);
  }
  
  inline BitstreamWriter& getStream() {
    return Stream;
  }
  
  template <typename T> inline void EnterSubblock(unsigned CodeLen) {
    Stream.EnterSubblock(8,CodeLen);
  }
  
  inline void ExitBlock() { Stream.ExitBlock(); }
  
};  
  
} // end anonymous namespace  

//===----------------------------------------------------------------------===//
// External Interface.
//===----------------------------------------------------------------------===//

ASTConsumer* clang::CreateSerializationTest() {
  return new SerializationTest();
}
  
//===----------------------------------------------------------------------===//
// Serialization "Driver" code.
//===----------------------------------------------------------------------===//

void SerializationTest::RunSerializationTest() { 
  std::string ErrMsg;
  llvm::sys::Path Filename = llvm::sys::Path::GetTemporaryDirectory(&ErrMsg);

  if (Filename.isEmpty()) {
    cerr << "Error: " << ErrMsg << "\n";
    return;
  }
  
  Filename.appendComponent("test.cfe_bc");
  
  if (Filename.makeUnique(true,&ErrMsg)) {
    cerr << "Error: " << ErrMsg << "\n";
    return;
  }
  
  if (!WriteAll(Filename))
    return;
  
  cout << "Wrote file: " << Filename.c_str() << "\n";
}

bool SerializationTest::WriteAll(llvm::sys::Path& Filename) {  
  std::ofstream Out(Filename.c_str());
  
  if (!Out) {
    cerr << "Error: Cannot open " << Filename.c_str() << "\n";
    return false;
  }
    
  Writer W(Out);
  W(*IdTable);

  W.getStream().FlushToWord();
  return true;
}

//===----------------------------------------------------------------------===//
// Serialization Methods.
//===----------------------------------------------------------------------===//

namespace llvm {

struct IntrospectionPrimitivesFlags {
  enum { isPod = true, UniqueInstances = false, UniqueRefs = false };
};
  

template<> struct
IntrospectionTrait<bool>::Flags : public IntrospectionPrimitivesFlags {};
  
template<> struct
IntrospectionTrait<unsigned>::Flags : public IntrospectionPrimitivesFlags {};

template<> struct
IntrospectionTrait<short>::Flags : public IntrospectionPrimitivesFlags {};



template<> 
struct IntrospectionTrait<clang::IdentifierInfo>::Flags {
  enum { isPod = false,  // Cannot copy via memcpy.  Must use copy-ctor.    
         hasUniqueInstances = true, // Two pointers with different
                                    // addreses point to objects
                                    // that are not equal to each other.    
         hasUniqueReferences = true // Two (non-temporary) pointers                                    
                                    // will point to distinct instances.
  };
};
  
template<> template<typename Introspector>
struct IntrospectionTrait<clang::IdentifierInfo>::Ops<Introspector> {
  static void Introspect(clang::IdentifierInfo& X, Introspector& I) {
//    I(X.getTokenID());
    I(X.getBuiltinID(),9); // FIXME: do 9 bit specialization.
//    I(X.getObjCKeywordID());
    I(X.hasMacroDefinition());
    I(X.isExtensionToken());
    I(X.isPoisoned());
    I(X.isOtherTargetMacro());
    I(X.isCPlusPlusOperatorKeyword());
    I(X.isNonPortableBuiltin());
  }
};
  
template<> template<>
struct IntrospectionTrait<clang::IdentifierTable>::Ops<Writer> {
  static void Introspect(clang::IdentifierTable& X, Writer& W) {
    W.EnterSubblock<clang::IdentifierTable>(1);
/*        
    for (clang::IdentifierTable::iterator I = X.begin(), E = X.end();
         I != E; ++I)
      W(I->getValue());
   */ 
    W.ExitBlock();
  }
};

  

  
} // end namespace llvm

