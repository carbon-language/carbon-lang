//===--- SerializationTest.cpp - Experimental Object Serialization --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements prototype code for serialization of objects in clang.
//  It is not intended yet for public use, but simply is a placeholder to
//  experiment with new serialization features.  Serialization will eventually
//  be integrated as a proper component of the clang libraries.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/CFG.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclGroup.h"
#include "clang-cc.h"
#include "ASTConsumers.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/Support/Streams.h"
#include "llvm/System/Path.h"
#include <fstream>
#include <cstring>
using namespace clang;

//===----------------------------------------------------------------------===//
// Driver code.
//===----------------------------------------------------------------------===//

namespace {
  
class SerializationTest : public ASTConsumer {
  Diagnostic &Diags;
  FileManager &FMgr;  
public:  
  SerializationTest(Diagnostic &d, FileManager& fmgr)
                    : Diags(d), FMgr(fmgr) {}
  
  ~SerializationTest() {}
  
  virtual void HandleTranslationUnit(ASTContext &C);
  
private:
  bool Serialize(llvm::sys::Path& Filename, llvm::sys::Path& FNameDeclPrint,
                 ASTContext &Ctx);
  
  bool Deserialize(llvm::sys::Path& Filename, llvm::sys::Path& FNameDeclPrint);
};
  
} // end anonymous namespace

ASTConsumer*
clang::CreateSerializationTest(Diagnostic &Diags, FileManager& FMgr) {  
  return new SerializationTest(Diags, FMgr);
}


bool SerializationTest::Serialize(llvm::sys::Path& Filename,
                                  llvm::sys::Path& FNameDeclPrint,
                                  ASTContext &Ctx) {
  { 
    // Pretty-print the decls to a temp file.
    std::string Err;
    llvm::raw_fd_ostream DeclPP(FNameDeclPrint.c_str(), true, Err);
    assert (Err.empty() && "Could not open file for printing out decls.");
    llvm::OwningPtr<ASTConsumer> FilePrinter(CreateASTPrinter(&DeclPP));
    
    TranslationUnitDecl *TUD = Ctx.getTranslationUnitDecl();
    for (DeclContext::decl_iterator I = TUD->decls_begin(), E =TUD->decls_end();
         I != E; ++I)
      FilePrinter->HandleTopLevelDecl(DeclGroupRef(*I));
  }
  
  // Serialize the translation unit.
  
  // Reserve 256K for bitstream buffer.
  std::vector<unsigned char> Buffer;
  Buffer.reserve(256*1024);
  
  Ctx.EmitASTBitcodeBuffer(Buffer);
  
  // Write the bits to disk. 
  if (FILE* fp = fopen(Filename.c_str(),"wb")) {
    fwrite((char*)&Buffer.front(), sizeof(char), Buffer.size(), fp);
    fclose(fp);
    return true;
  }
  
  return false;
}

bool SerializationTest::Deserialize(llvm::sys::Path& Filename,
                                    llvm::sys::Path& FNameDeclPrint) {
  
  // Deserialize the translation unit.
  ASTContext *NewCtx;
  
  {
    // Create the memory buffer that contains the contents of the file.  
    llvm::OwningPtr<llvm::MemoryBuffer> 
      MBuffer(llvm::MemoryBuffer::getFile(Filename.c_str()));
  
    if (!MBuffer)
      return false;
    
    NewCtx = ASTContext::ReadASTBitcodeBuffer(*MBuffer, FMgr);
  }

  if (!NewCtx)
    return false;
  
  {
    // Pretty-print the deserialized decls to a temp file.
    std::string Err;
    llvm::raw_fd_ostream DeclPP(FNameDeclPrint.c_str(), true, Err);
    assert (Err.empty() && "Could not open file for printing out decls.");
    llvm::OwningPtr<ASTConsumer> FilePrinter(CreateASTPrinter(&DeclPP));
    
    TranslationUnitDecl *TUD = NewCtx->getTranslationUnitDecl();
    for (DeclContext::decl_iterator I = TUD->decls_begin(), E = TUD->decls_end();
         I != E; ++I)
      FilePrinter->HandleTopLevelDecl(DeclGroupRef(*I));
  }

  delete NewCtx;
  
  return true;
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

void SerializationTest::HandleTranslationUnit(ASTContext &Ctx) {

  std::string ErrMsg;
  llvm::sys::Path Dir = llvm::sys::Path::GetTemporaryDirectory(&ErrMsg);
  
  if (Dir.isEmpty()) {
    llvm::cerr << "Error: " << ErrMsg << "\n";
    return;
  }
  
  TmpDirJanitor RemoveTmpOnExit(Dir);
    
  llvm::sys::Path FNameDeclBefore = Dir;
  FNameDeclBefore.appendComponent("test.decl_before.txt");

  if (FNameDeclBefore.makeUnique(true, &ErrMsg)) {
    llvm::cerr << "Error: " << ErrMsg << "\n";
    return;
  }
  
  llvm::sys::Path FNameDeclAfter = Dir;
  FNameDeclAfter.appendComponent("test.decl_after.txt");
  
  if (FNameDeclAfter.makeUnique(true, &ErrMsg)) {
    llvm::cerr << "Error: " << ErrMsg << "\n";
    return;
  }

  llvm::sys::Path ASTFilename = Dir;
  ASTFilename.appendComponent("test.ast");
  
  if (ASTFilename.makeUnique(true, &ErrMsg)) {
    llvm::cerr << "Error: " << ErrMsg << "\n";
    return;
  }
  
  // Serialize and then deserialize the ASTs.
  bool status = Serialize(ASTFilename, FNameDeclBefore, Ctx);
  assert (status && "Serialization failed.");  
  status = Deserialize(ASTFilename, FNameDeclAfter);
  assert (status && "Deserialization failed.");
  
  // Read both pretty-printed files and compare them.
  
  using llvm::MemoryBuffer;
  
  llvm::OwningPtr<MemoryBuffer>
    MBufferSer(MemoryBuffer::getFile(FNameDeclBefore.c_str()));
  
  if(!MBufferSer) {
    llvm::cerr << "ERROR: Cannot read pretty-printed file (pre-pickle).\n";
    return;
  }
  
  llvm::OwningPtr<MemoryBuffer>
    MBufferDSer(MemoryBuffer::getFile(FNameDeclAfter.c_str()));
  
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
