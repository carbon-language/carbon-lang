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

//===----------------------------------------------------------------------===//
// Driver code.
//===----------------------------------------------------------------------===//

using namespace clang;

namespace {
  template<typename T>
  struct Janitor {
    T* Obj;
    Janitor(T* obj) : Obj(obj) {}
    ~Janitor() { delete Obj; }
  };
} // end anonymous namespace

namespace {
  class SerializationTest : public ASTConsumer {
    IdentifierTable* IdTable;
    unsigned MainFileID;
  public:
    void Initialize(ASTContext& Context, unsigned mainFileID) {
      IdTable = &Context.Idents;
      MainFileID = mainFileID;
    }
    
    ~SerializationTest() {
      RunSerializationTest();
    }
    
    void RunSerializationTest();
    bool WriteTable(llvm::sys::Path& Filename, IdentifierTable* T);
    IdentifierTable* ReadTable(llvm::sys::Path& Filename);
    
    virtual void HandleTopLevelDecl(Decl *D) {}
  };
} // end anonymous namespace

ASTConsumer* clang::CreateSerializationTest() {
  return new SerializationTest();
}

void SerializationTest::RunSerializationTest() { 
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
  
  llvm::cerr << "Writing out Identifier table\n";
  WriteTable(Filename,IdTable);
  llvm::cerr << "Reading in Identifier Table\n";
  IdentifierTable* T = ReadTable(Filename);
  Janitor<IdentifierTable> roger(T);
  
  Filename.appendSuffix("2");
  llvm::cerr << "Writing out Identifier table (2)\n";
  WriteTable(Filename,T);
  llvm::cerr << "Reading in Identifier Table (2)\n";
  Janitor<IdentifierTable> wilco(ReadTable(Filename));
}

bool SerializationTest::WriteTable(llvm::sys::Path& Filename,
                                   IdentifierTable* T) {
  if (!T)
    return false;
  
  std::vector<unsigned char> Buffer;
  Buffer.reserve(256*1024);
  
  llvm::BitstreamWriter Stream(Buffer);
  
  Stream.Emit((unsigned)'B', 8);
  Stream.Emit((unsigned)'C', 8);
  Stream.Emit(0xC, 4);
  Stream.Emit(0xF, 4);
  Stream.Emit(0xE, 4);
  Stream.Emit(0x0, 4);

  llvm::Serializer S(Stream);
  S.Emit(*T);
  S.Flush();
  
  if (FILE *fp = fopen(Filename.c_str(),"wb")) {
    fwrite((char*)&Buffer.front(), sizeof(char), Buffer.size(), fp);
    fclose(fp);
  }
  else { 
    llvm::cerr << "Error: Cannot open " << Filename.c_str() << "\n";
    return false;
  }
  
  llvm::cerr << "Wrote file: " << Filename.c_str() << "\n";
  return true;
}


IdentifierTable* SerializationTest::ReadTable(llvm::sys::Path& Filename) {
  llvm::MemoryBuffer* Buffer = 
    llvm::MemoryBuffer::getFile(Filename.c_str(), strlen(Filename.c_str()));
  
  if(!Buffer) {
    llvm::cerr << "Error reading file\n";
    return NULL;
  }
  
  Janitor<llvm::MemoryBuffer> AutoReleaseBuffer(Buffer);
  
  if (Buffer->getBufferSize() & 0x3) {
    llvm::cerr << "AST file should be a multiple of 4 bytes in length\n";
    return NULL;
  }
  
  unsigned char *BufPtr = (unsigned char *)Buffer->getBufferStart();
  llvm::BitstreamReader Stream(BufPtr,BufPtr+Buffer->getBufferSize());
  
  // Sniff for the signature.
  if (Stream.Read(8) != 'B' ||
      Stream.Read(8) != 'C' ||
      Stream.Read(4) != 0xC ||
      Stream.Read(4) != 0xF ||
      Stream.Read(4) != 0xE ||
      Stream.Read(4) != 0x0) {
    llvm::cerr << "Invalid AST-bitcode signature\n";
    return NULL;
  }
  
  llvm::Deserializer D(Stream);

  llvm::cerr << "Materializing identifier table.\n";
  return D.Materialize<IdentifierTable>();
}
