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
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/SourceManager.h"
#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/CFG.h"
#include "clang.h"
#include "llvm/System/Path.h"
#include "llvm/Support/Streams.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Bitcode/Serialize.h"
#include "llvm/Bitcode/Deserialize.h"
#include <fstream>
#include <stdio.h>
#include <list>

using namespace clang;

//===----------------------------------------------------------------------===//
// Utility classes
//===----------------------------------------------------------------------===//

namespace {

template<typename T> class Janitor {
  T* Obj;
public:
  explicit Janitor(T* obj) : Obj(obj) {}
  ~Janitor() { delete Obj; }
  operator T*() const { return Obj; }
  T* operator->() { return Obj; }
};
  
//===----------------------------------------------------------------------===//
// Driver code.
//===----------------------------------------------------------------------===//

class SerializationTest : public ASTConsumer {
  ASTContext* Context;
  Diagnostic &Diags;
  FileManager &FMgr;
  const LangOptions& LangOpts;
  std::list<Decl*> Decls;
  
  enum { BasicMetadataBlock = 1,
         ASTContextBlock = 2,
         DeclsBlock = 3 };

public:  
  SerializationTest(Diagnostic &d, FileManager& fmgr, const LangOptions& LOpts)
    : Context(NULL), Diags(d), FMgr(fmgr), LangOpts(LOpts) {};
  
  ~SerializationTest();

  virtual void Initialize(ASTContext& context, unsigned) {
      Context = &context;
  }
  
  virtual void HandleTopLevelDecl(Decl *D) {
    Decls.push_back(D);
  }

private:
  void Serialize(llvm::sys::Path& Filename, llvm::sys::Path& FNameDeclPrint);
  void Deserialize(llvm::sys::Path& Filename, llvm::sys::Path& FNameDeclPrint);
};
  
} // end anonymous namespace

ASTConsumer*
clang::CreateSerializationTest(Diagnostic &Diags, FileManager& FMgr,
                               const LangOptions &LOpts) {  
  return new SerializationTest(Diags,FMgr,LOpts);
}

static void WritePreamble(llvm::BitstreamWriter& Stream) {
  Stream.Emit((unsigned)'B', 8);
  Stream.Emit((unsigned)'C', 8);
  Stream.Emit(0xC, 4);
  Stream.Emit(0xF, 4);
  Stream.Emit(0xE, 4);
  Stream.Emit(0x0, 4);
}

static bool ReadPreamble(llvm::BitstreamReader& Stream) {
  return Stream.Read(8) != 'B' ||
         Stream.Read(8) != 'C' ||
         Stream.Read(4) != 0xC ||
         Stream.Read(4) != 0xF ||
         Stream.Read(4) != 0xE ||
         Stream.Read(4) != 0x0;
}

void SerializationTest::Serialize(llvm::sys::Path& Filename,
                                  llvm::sys::Path& FNameDeclPrint) {
  
  // Reserve 256K for bitstream buffer.
  std::vector<unsigned char> Buffer;
  Buffer.reserve(256*1024);
  
  // Create bitstream and write preamble.    
  llvm::BitstreamWriter Stream(Buffer);
  WritePreamble(Stream);
  
  // Create serializer.
  llvm::Serializer Sezr(Stream);
  
  // ===---------------------------------------------------===/
  //      Serialize the top-level decls.
  // ===---------------------------------------------------===/  
  
  Sezr.EnterBlock(DeclsBlock);
    
  { // Create a printer to "consume" our deserialized ASTS.

    Janitor<ASTConsumer> Printer(CreateASTPrinter());
    std::ofstream DeclPP(FNameDeclPrint.c_str());
    assert (DeclPP && "Could not open file for printing out decls.");
    Janitor<ASTConsumer> FilePrinter(CreateASTPrinter(&DeclPP));
    
    for (std::list<Decl*>::iterator I=Decls.begin(), E=Decls.end(); I!=E; ++I) {
      llvm::cerr << "Serializing: Decl.\n";   
      
      // Only serialize the head of a decl chain.  The ASTConsumer interfaces
      // provides us with each top-level decl, including those nested in
      // a decl chain, so we may be passed decls that are already serialized.
      if (!Sezr.isRegistered(*I)) {
        Printer->HandleTopLevelDecl(*I);
        FilePrinter->HandleTopLevelDecl(*I);
        
        if (FunctionDecl* FD = dyn_cast<FunctionDecl>(*I))
          if (FD->getBody()) {
            // Construct and print a CFG.
            Janitor<CFG> cfg(CFG::buildCFG(FD->getBody()));
            cfg->print(DeclPP);
          }
        
        // Serialize the decl.
        Sezr.EmitOwnedPtr(*I);
      }
    }
  }
  
  Sezr.ExitBlock();
  
  // ===---------------------------------------------------===/
  //      Serialize the "Translation Unit" metadata.
  // ===---------------------------------------------------===/

  // Emit ASTContext.
  Sezr.EnterBlock(ASTContextBlock);  
  llvm::cerr << "Serializing: ASTContext.\n";  
  Sezr.EmitOwnedPtr(Context);  
  Sezr.ExitBlock();  
  
  
  Sezr.EnterBlock(BasicMetadataBlock);

  // Block for SourceManager and Target.  Allows easy skipping around
  // to the Selectors during deserialization.
  Sezr.EnterBlock();

  // "Fake" emit the SourceManager.
  llvm::cerr << "Serializing: SourceManager.\n";
  Sezr.Emit(Context->SourceMgr);
  
  // Emit the Target.
  llvm::cerr << "Serializing: Target.\n";
  Sezr.EmitPtr(&Context->Target);
  Sezr.EmitCStr(Context->Target.getTargetTriple());

  Sezr.ExitBlock();

  // Emit the Selectors.
  llvm::cerr << "Serializing: Selectors.\n";
  Sezr.Emit(Context->Selectors);
  
  // Emit the Identifier Table.
  llvm::cerr << "Serializing: IdentifierTable.\n";  
  Sezr.Emit(Context->Idents);

  Sezr.ExitBlock();  
  
  // ===---------------------------------------------------===/
  // Finalize serialization: write the bits to disk.
  if (FILE* fp = fopen(Filename.c_str(),"wb")) {
    fwrite((char*)&Buffer.front(), sizeof(char), Buffer.size(), fp);
    fclose(fp);
  }
  else { 
    llvm::cerr << "Error: Cannot open " << Filename.c_str() << "\n";
    return;
  }
  
  llvm::cerr << "Commited bitstream to disk: " << Filename.c_str() << "\n";
}


void SerializationTest::Deserialize(llvm::sys::Path& Filename,
                                    llvm::sys::Path& FNameDeclPrint) {
  
  // Create the memory buffer that contains the contents of the file.
  
  using llvm::MemoryBuffer;
  
  Janitor<MemoryBuffer> MBuffer(MemoryBuffer::getFile(Filename.c_str(),
                                              strlen(Filename.c_str())));
  
  if(!MBuffer) {
    llvm::cerr << "ERROR: Cannot read file for deserialization.\n";
    return;
  }
  
  // Check if the file is of the proper length.
  if (MBuffer->getBufferSize() & 0x3) {
    llvm::cerr << "ERROR: AST file length should be a multiple of 4 bytes.\n";
    return;
  }
  
  // Create the bitstream reader.
  unsigned char *BufPtr = (unsigned char *) MBuffer->getBufferStart();
  llvm::BitstreamReader Stream(BufPtr,BufPtr+MBuffer->getBufferSize());
  
  // Sniff for the signature in the bitcode file.
  if (ReadPreamble(Stream)) {
    llvm::cerr << "ERROR: Invalid AST-bitcode signature.\n";
    return;
  }
    
  // Create the deserializer.
  llvm::Deserializer Dezr(Stream);
  
  // ===---------------------------------------------------===/
  //      Deserialize the "Translation Unit" metadata.
  // ===---------------------------------------------------===/
  
  // Skip to the BasicMetaDataBlock.  First jump to ASTContextBlock
  // (which will appear earlier) and record its location.
  
  bool FoundBlock = Dezr.SkipToBlock(ASTContextBlock);
  assert (FoundBlock);

  llvm::Deserializer::Location ASTContextBlockLoc =
    Dezr.getCurrentBlockLocation();
  
  FoundBlock = Dezr.SkipToBlock(BasicMetadataBlock);
  assert (FoundBlock);
  
  // Read the SourceManager.
  llvm::cerr << "Deserializing: SourceManager.\n";
  SourceManager::CreateAndRegister(Dezr,FMgr);

  { // Read the TargetInfo.
    llvm::cerr << "Deserializing: Target.\n";
    llvm::SerializedPtrID PtrID = Dezr.ReadPtrID();
    char* triple = Dezr.ReadCStr(NULL,0,true);
    std::vector<std::string> triples;
    triples.push_back(triple);
    delete [] triple;
    Dezr.RegisterPtr(PtrID,CreateTargetInfo(triples,&Diags));
  }
    
  // For Selectors, we must read the identifier table first because the
  //  SelectorTable depends on the identifiers being already deserialized.
  llvm::Deserializer::Location SelectorBlockLoc =
    Dezr.getCurrentBlockLocation();
    
  Dezr.SkipBlock();
  
  // Read the identifier table.
  llvm::cerr << "Deserializing: IdentifierTable\n";
  IdentifierTable::CreateAndRegister(Dezr);
  
  // Now jump back and read the selectors.
  llvm::cerr << "Deserializing: Selectors\n";
  Dezr.JumpTo(SelectorBlockLoc);
  SelectorTable::CreateAndRegister(Dezr);
  
  // Now jump back to ASTContextBlock and read the ASTContext.
  llvm::cerr << "Deserializing: ASTContext.\n";
  Dezr.JumpTo(ASTContextBlockLoc);
  Dezr.ReadOwnedPtr<ASTContext>();
    
  // "Rewind" the stream.  Find the block with the serialized top-level decls.
  Dezr.Rewind();
  FoundBlock = Dezr.SkipToBlock(DeclsBlock);
  assert (FoundBlock);
  llvm::Deserializer::Location DeclBlockLoc = Dezr.getCurrentBlockLocation();
  
  // Create a printer to "consume" our deserialized ASTS.
  ASTConsumer* Printer = CreateASTPrinter();
  Janitor<ASTConsumer> PrinterJanitor(Printer);  
  std::ofstream DeclPP(FNameDeclPrint.c_str());
  assert (DeclPP && "Could not open file for printing out decls.");
  Janitor<ASTConsumer> FilePrinter(CreateASTPrinter(&DeclPP));
  
  // The remaining objects in the file are top-level decls.
  while (!Dezr.FinishedBlock(DeclBlockLoc)) {
    llvm::cerr << "Deserializing: Decl.\n";
    Decl* decl = Dezr.ReadOwnedPtr<Decl>();
    Printer->HandleTopLevelDecl(decl);
    FilePrinter->HandleTopLevelDecl(decl);
    
    if (FunctionDecl* FD = dyn_cast<FunctionDecl>(decl))
      if (FD->getBody()) {
        // Construct and print a CFG.
        Janitor<CFG> cfg(CFG::buildCFG(FD->getBody()));
        cfg->print(DeclPP);
      }
  }
}
  
namespace {
  class TmpDirJanitor {
    llvm::sys::Path& Dir;
  public:
    explicit TmpDirJanitor(llvm::sys::Path& dir) : Dir(dir) {}

    ~TmpDirJanitor() { 
      llvm::cerr << "Removing: " << Dir.c_str() << '\n';
      Dir.eraseFromDisk(true); 
    }
  };
}

SerializationTest::~SerializationTest() {

  std::string ErrMsg;
  llvm::sys::Path Dir = llvm::sys::Path::GetTemporaryDirectory(&ErrMsg);
  
  if (Dir.isEmpty()) {
    llvm::cerr << "Error: " << ErrMsg << "\n";
    return;
  }
  
  TmpDirJanitor RemoveTmpOnExit(Dir);
    
  llvm::sys::Path FNameDeclBefore = Dir;
  FNameDeclBefore.appendComponent("test.decl_before.txt");

  if (FNameDeclBefore.makeUnique(true,&ErrMsg)) {
    llvm::cerr << "Error: " << ErrMsg << "\n";
    return;
  }
  
  llvm::sys::Path FNameDeclAfter = Dir;
  FNameDeclAfter.appendComponent("test.decl_after.txt");
  
  if (FNameDeclAfter.makeUnique(true,&ErrMsg)) {
    llvm::cerr << "Error: " << ErrMsg << "\n";
    return;
  }

  llvm::sys::Path ASTFilename = Dir;
  ASTFilename.appendComponent("test.ast");
  
  if (ASTFilename.makeUnique(true,&ErrMsg)) {
    llvm::cerr << "Error: " << ErrMsg << "\n";
    return;
  }
  
  // Serialize and then deserialize the ASTs.
  Serialize(ASTFilename, FNameDeclBefore);
  Deserialize(ASTFilename, FNameDeclAfter);
  
  // Read both pretty-printed files and compare them.
  
  using llvm::MemoryBuffer;
  
  Janitor<MemoryBuffer>
    MBufferSer(MemoryBuffer::getFile(FNameDeclBefore.c_str(),
                                     strlen(FNameDeclBefore.c_str())));
  
  if(!MBufferSer) {
    llvm::cerr << "ERROR: Cannot read pretty-printed file (pre-pickle).\n";
    return;
  }
  
  Janitor<MemoryBuffer>
    MBufferDSer(MemoryBuffer::getFile(FNameDeclAfter.c_str(),
                                      strlen(FNameDeclAfter.c_str())));
  
  if(!MBufferDSer) {
    llvm::cerr << "ERROR: Cannot read pretty-printed file (post-pickle).\n";
    return;
  }
  
  const char *p1 = MBufferSer->getBufferStart();
  const char *e1 = MBufferSer->getBufferEnd();
  const char *p2 = MBufferDSer->getBufferStart();
  const char *e2 = MBufferDSer->getBufferEnd();

  if (MBufferSer->getBufferSize() == MBufferDSer->getBufferSize())
    for ( ; p1 != e1 ; ++p1, ++p2  )
      if (*p1 != *p2) break;
  
  if (p1 != e1 || p2 != e2 )
    llvm::cerr << "ERROR: Pretty-printed files are not the same.\n";
  else
    llvm::cerr << "SUCCESS: Pretty-printed files are the same.\n";
}
