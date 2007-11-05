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
#include "llvm/Support/Streams.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Bitcode/Serialize.h"
#include "llvm/Bitcode/Deserialize.h"
#include "llvm/System/TimeValue.h"
#include <stdio.h>

//===----------------------------------------------------------------------===//
// Driver code.
//===----------------------------------------------------------------------===//

using namespace clang;
using llvm::sys::TimeValue;

namespace {

template<typename T> struct Janitor {
  T* Obj;
  Janitor(T* obj) : Obj(obj) {}
  ~Janitor() { delete Obj; }
};

class SerializationTest : public ASTConsumer {
  ASTContext* Context;
  llvm::sys::Path Filename;
  std::vector<unsigned char>* Buffer;
  llvm::BitstreamWriter* OBStream;
  llvm::Serializer* serializer;
  
  void DeserializeTest();
public:
  
  SerializationTest(llvm::sys::Path filename);
  ~SerializationTest();

  virtual void Initialize(ASTContext& context, unsigned);    
  virtual void HandleTopLevelDecl(Decl *D);
};
  
} // end anonymous namespace

ASTConsumer* clang::CreateSerializationTest() {
  std::string ErrMsg;
  llvm::sys::Path Filename = llvm::sys::Path::GetTemporaryDirectory(&ErrMsg);
  
  if (Filename.isEmpty()) {
    llvm::cerr << "Error: " << ErrMsg << "\n";
    return NULL;
  }
  
  Filename.appendComponent("test.ast");
  
  if (Filename.makeUnique(true,&ErrMsg)) {
    llvm::cerr << "Error: " << ErrMsg << "\n";
    return NULL;
  }
  
  return new SerializationTest(Filename);
}

SerializationTest::SerializationTest(llvm::sys::Path filename)
  : Filename(filename), OBStream(NULL), serializer(NULL) {
      
    // Reserve 256K for bitstream buffer.
    Buffer = new std::vector<unsigned char>();
    assert (Buffer && "Could not allocate buffer.");
    Buffer->reserve(256*1024);
    
    // Open bitstream and write preamble.    
    OBStream = new llvm::BitstreamWriter(*Buffer);
    assert (OBStream && "could not create bitstream for serialization");
    
    OBStream->Emit((unsigned)'B', 8);
    OBStream->Emit((unsigned)'C', 8);
    OBStream->Emit(0xC, 4);
    OBStream->Emit(0xF, 4);
    OBStream->Emit(0xE, 4);
    OBStream->Emit(0x0, 4);
    
    // Open serializer.
    serializer = new llvm::Serializer(*OBStream,0);
    assert (serializer && "could not create serializer");
}
  

void SerializationTest::Initialize(ASTContext& context, unsigned) {
  llvm::cerr << "[ " << TimeValue::now().toString() << " ] "
             << "Faux-serializing: SourceManager et al.\n";

  serializer->EnterBlock();
  // "Fake" emit the SourceManager, etc.
  Context = &context;
  serializer->EmitPtr(&context.SourceMgr);
  serializer->EmitPtr(&context.Target);
  serializer->EmitPtr(&context.Idents);
  serializer->EmitPtr(&context.Selectors);  

  llvm::cerr << "[ " << TimeValue::now().toString() << " ] "
             << "Serializing: ASTContext.\n";


  serializer->EmitOwnedPtr(&context);
  serializer->ExitBlock();
}

void SerializationTest::HandleTopLevelDecl(Decl *D) {
  llvm::cerr << "[ " << TimeValue::now().toString() << " ] "
             << "Serializing: Decl.\n";
  
  serializer->EnterBlock();  
  serializer->EmitOwnedPtr(D);
  serializer->ExitBlock();
}

SerializationTest::~SerializationTest() {
  delete serializer;
  delete OBStream;
  
  if (FILE *fp = fopen(Filename.c_str(),"wb")) {
    fwrite((char*)&Buffer->front(), sizeof(char), Buffer->size(), fp);
    delete Buffer;
    fclose(fp);
  }
  else { 
    llvm::cerr << "Error: Cannot open " << Filename.c_str() << "\n";
    delete Buffer;
    return;
  }

  llvm::cerr << "[ " << TimeValue::now().toString() << " ] "
             << "Commited bitstream to disk: " << Filename.c_str() << "\n";
  
  DeserializeTest();
}

void SerializationTest::DeserializeTest() {

  llvm::MemoryBuffer* MBuffer = 
    llvm::MemoryBuffer::getFile(Filename.c_str(), strlen(Filename.c_str()));
  
  if(!MBuffer) {
    llvm::cerr << "ERROR: Cannot read file for deserialization.\n";
    return;
  }
  
  Janitor<llvm::MemoryBuffer> AutoReleaseBuffer(MBuffer);
  
  if (MBuffer->getBufferSize() & 0x3) {
    llvm::cerr << "ERROR: AST file should be a multiple of 4 bytes in length.\n";
    return;
  }
  
  unsigned char *BufPtr = (unsigned char *)MBuffer->getBufferStart();
  llvm::BitstreamReader IBStream(BufPtr,BufPtr+MBuffer->getBufferSize());
  
  // Sniff for the signature.
  if (IBStream.Read(8) != 'B' ||
      IBStream.Read(8) != 'C' ||
      IBStream.Read(4) != 0xC ||
      IBStream.Read(4) != 0xF ||
      IBStream.Read(4) != 0xE ||
      IBStream.Read(4) != 0x0) {
    llvm::cerr << "ERROR: Invalid AST-bitcode signature.\n";
    return;
  }
  
  llvm::Deserializer deserializer(IBStream);
  
  
  // "Fake" read the SourceManager, etc.  
  llvm::cerr << "[ " << TimeValue::now().toString() << " ] "
             << "Faux-Deserializing: SourceManager et al.\n";
  
  deserializer.RegisterPtr(&Context->SourceMgr);
  deserializer.RegisterPtr(&Context->Target);
  deserializer.RegisterPtr(&Context->Idents);
  deserializer.RegisterPtr(&Context->Selectors);  
  
  llvm::cerr << "[ " << TimeValue::now().toString() << " ] "
             << "Deserializing: ASTContext.\n";

  // Deserialize the AST context.
  deserializer.ReadOwnedPtr<ASTContext>();
//  ASTContext* context = 
}
