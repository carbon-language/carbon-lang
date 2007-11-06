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
#include <stdio.h>
#include <list>

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
  std::list<Decl*> Decls;
  
  enum { ContextBlock = 0x1, DeclBlock = 0x3 };

public:  
  SerializationTest() : Context(NULL) {};
  ~SerializationTest();

  virtual void Initialize(ASTContext& context, unsigned) {
      Context = &context;
  }
  
  virtual void HandleTopLevelDecl(Decl *D) {
    Decls.push_back(D);
  }

private:
  void Serialize(llvm::sys::Path& Filename);
  void Deserialize(llvm::sys::Path& Filename);
};
  
} // end anonymous namespace

ASTConsumer* clang::CreateSerializationTest() {  
  return new SerializationTest();
}

static void WritePreamble(llvm::BitstreamWriter& Stream) {
  Stream.Emit((unsigned)'B', 8);
  Stream.Emit((unsigned)'C', 8);
  Stream.Emit(0xC, 4);
  Stream.Emit(0xF, 4);
  Stream.Emit(0xE, 4);
  Stream.Emit(0x0, 4);
}

static bool ReadPremable(llvm::BitstreamReader& Stream) {
  return Stream.Read(8) != 'B' ||
         Stream.Read(8) != 'C' ||
         Stream.Read(4) != 0xC ||
         Stream.Read(4) != 0xF ||
         Stream.Read(4) != 0xE ||
         Stream.Read(4) != 0x0;
}

void SerializationTest::Serialize(llvm::sys::Path& Filename) {
  
  // Reserve 256K for bitstream buffer.
  std::vector<unsigned char> Buffer;
  Buffer.reserve(256*1024);
  
  // Create bitstream and write preamble.    
  llvm::BitstreamWriter Stream(Buffer);
  WritePreamble(Stream);
  
  // Create serializer.
  llvm::Serializer Sezr(Stream);
  
  // ===---------------------------------------------------===/
  //      Serialize the "Translation Unit" metadata.
  // ===---------------------------------------------------===/

  Sezr.EnterBlock(ContextBlock);

  // "Fake" emit the SourceManager.
  llvm::cerr << "Faux-serializing: SourceManager.\n";
  Sezr.EmitPtr(&Context->SourceMgr);
  
  // "Fake" emit the Target.
  llvm::cerr << "Faux-serializing: Target.\n";
  Sezr.EmitPtr(&Context->Target);

  // "Fake" emit Selectors.
  llvm::cerr << "Faux-serializing: Selectors.\n";
  Sezr.EmitPtr(&Context->Selectors);
  
  // Emit the Identifier Table.
  llvm::cerr << "Serializing: IdentifierTable.\n";  
  Sezr.EmitOwnedPtr(&Context->Idents);
  
  // Emit the ASTContext.
  llvm::cerr << "Serializing: ASTContext.\n";  
  Sezr.EmitOwnedPtr(Context);
  
  Sezr.ExitBlock();  
  
  // ===---------------------------------------------------===/
  //      Serialize the top-level decls.
  // ===---------------------------------------------------===/  
  
  Sezr.EnterBlock(DeclBlock);
  
  for (std::list<Decl*>::iterator I=Decls.begin(), E=Decls.end(); I!=E; ++I) {
    llvm::cerr << "Serializing: Decl.\n";    
    Sezr.EmitOwnedPtr(*I);
  }

  Sezr.ExitBlock();
  
  // ===---------------------------------------------------===/
  //      Finalize serialization: write the bits to disk.
  // ===---------------------------------------------------===/ 
  
  if (FILE *fp = fopen(Filename.c_str(),"wb")) {
    fwrite((char*)&Buffer.front(), sizeof(char), Buffer.size(), fp);
    fclose(fp);
  }
  else { 
    llvm::cerr << "Error: Cannot open " << Filename.c_str() << "\n";
    return;
  }
  
  llvm::cerr << "Commited bitstream to disk: " << Filename.c_str() << "\n";
}


void SerializationTest::Deserialize(llvm::sys::Path& Filename) {
  
  // Create the memory buffer that contains the contents of the file.
  
  using llvm::MemoryBuffer;
  
  MemoryBuffer* MBuffer = MemoryBuffer::getFile(Filename.c_str(),
                                                strlen(Filename.c_str()));
  
  if(!MBuffer) {
    llvm::cerr << "ERROR: Cannot read file for deserialization.\n";
    return;
  }
  
  // Create an "autocollector" object to release the memory buffer upon
  // termination of the current scope.
  Janitor<MemoryBuffer> AutoReleaseBuffer(MBuffer);
  
  // Check if the file is of the proper length.
  if (MBuffer->getBufferSize() & 0x3) {
    llvm::cerr << "ERROR: AST file length should be a multiple of 4 bytes.\n";
    return;
  }
  
  // Create the bitstream reader.
  unsigned char *BufPtr = (unsigned char *) MBuffer->getBufferStart();
  llvm::BitstreamReader Stream(BufPtr,BufPtr+MBuffer->getBufferSize());
  
  // Sniff for the signature in the bitcode file.
  if (!ReadPremable(Stream)) {
    llvm::cerr << "ERROR: Invalid AST-bitcode signature.\n";
    return;
  }  
  
  // Create the Dezr.
  llvm::Deserializer Dezr(Stream);
  
  // ===---------------------------------------------------===/
  //      Deserialize the "Translation Unit" metadata.
  // ===---------------------------------------------------===/
  
  // "Fake" read the SourceManager.
  llvm::cerr << "Faux-Deserializing: SourceManager.\n";
  Dezr.RegisterPtr(&Context->SourceMgr);

  // "Fake" read the TargetInfo.
  llvm::cerr << "Faux-Deserializing: Target.\n";
  Dezr.RegisterPtr(&Context->Target);

  // "Fake" read the Selectors.
  llvm::cerr << "Faux-Deserializing: Selectors.\n";
  Dezr.RegisterPtr(&Context->Selectors);  
  
  // Read the identifier table.
  llvm::cerr << "Deserializing: IdentifierTable\n";
  Dezr.ReadOwnedPtr<IdentifierTable>();
  
  // Read the ASTContext.  
  llvm::cerr << "Deserializing: ASTContext.\n";
  Dezr.ReadOwnedPtr<ASTContext>();
  
  // Create a printer to "consume" our deserialized ASTS.
  ASTConsumer* Printer = CreateASTPrinter();
  Janitor<ASTConsumer> PrinterJanitor(Printer);  
  
  // The remaining objects in the file are top-level decls.
  while (!Dezr.AtEnd()) {
    llvm::cerr << "Deserializing: Decl.\n";
    Decl* decl = Dezr.ReadOwnedPtr<Decl>();
    Printer->HandleTopLevelDecl(decl);    
  }
}
  

SerializationTest::~SerializationTest() {
    
  std::string ErrMsg;
  llvm::sys::Path Filename = llvm::sys::Path::GetTemporaryDirectory(&ErrMsg);
  
  if (Filename.isEmpty()) {
    llvm::cerr << "Error: " << ErrMsg << "\n";
    return;
  }
  
  Filename.appendComponent("test.ast");
  
  if (Filename.makeUnique(true,&ErrMsg)) {
    llvm::cerr << "Error: " << ErrMsg << "\n";
    return;
  }
  
  Serialize(Filename);
  Deserialize(Filename);
}
