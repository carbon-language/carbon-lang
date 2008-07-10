//===--- TranslationUnit.cpp - Abstraction for Translation Units ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
// FIXME: This should eventually be moved out of the driver, or replaced
//        with its eventual successor.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/TranslationUnit.h"

#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/SourceManager.h"
#include "clang/AST/AST.h"

#include "llvm/Bitcode/Serialize.h"
#include "llvm/Bitcode/Deserialize.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/System/Path.h"

#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/DenseSet.h"

using namespace clang;

enum { BasicMetadataBlock = 1,
       ASTContextBlock = 2,
       DeclsBlock = 3 };

TranslationUnit::~TranslationUnit() {
  if (OwnsDecls) {
    llvm::DenseSet<Decl*> Killed;
    for (iterator I=begin(), E=end(); I!=E; ++I) {
      if (Killed.count(*I)) continue;

      Killed.insert(*I);
      
      // FIXME: This is a horrible hack.  Because there is no clear ownership
      //  role between ObjCInterfaceDecls and the ObjCPropertyDecls that they
      //  reference, we need to destroy ObjCPropertyDecls here.  This will
      //  eventually be fixed when the ownership of ObjCPropertyDecls gets
      //  cleaned up.
      if (ObjCInterfaceDecl* IDecl = dyn_cast<ObjCInterfaceDecl>(*I))
        for (ObjCInterfaceDecl::classprop_iterator ID=IDecl->classprop_begin(),
             ED=IDecl->classprop_end(); ID!=ED; ++ID) {
          if (!*ID || Killed.count(*ID)) continue;
          Killed.insert(*ID);
          (*ID)->Destroy(*Context);
        }
      
      // FIXME: This is a horrible hack.  Because there is no clear ownership
      //  role between ObjCProtocolDecls and the ObjCPropertyDecls that they
      //  reference, we need to destroy ObjCPropertyDecls here.  This will
      //  eventually be fixed when the ownership of ObjCPropertyDecls gets
      //  cleaned up.
      if (ObjCProtocolDecl* PDecl = dyn_cast<ObjCProtocolDecl>(*I))
        for (ObjCProtocolDecl::classprop_iterator ID=PDecl->classprop_begin(),
             ED=PDecl->classprop_end(); ID!=ED; ++ID) {
          if (!*ID || Killed.count(*ID)) continue;
          Killed.insert(*ID);
          (*ID)->Destroy(*Context);
        }
            
      // FIXME: There is no clear ownership policy now for ObjCInterfaceDecls
      //  referenced by ObjCClassDecls.  Some of them can be forward decls that
      //  are never later defined (and forward decls can be referenced by
      //  multiple ObjCClassDecls) or the ObjCInterfaceDecl later
      //  becomes a real definition. 
      //  Ideally we should have separate objects for forward declarations and
      //  definitions, obviating this problem.  Because of this situation,
      //  referenced ObjCInterfaceDecls are destroyed here.      
      if (ObjCClassDecl* CDecl = dyn_cast<ObjCClassDecl>(*I))
        for (ObjCClassDecl::iterator ID=CDecl->begin(),
             ED=CDecl->end(); ID!=ED; ++ID) {          
          if (!*ID || Killed.count(*ID)) continue;
          Killed.insert(*ID);
          (*ID)->Destroy(*Context);
        }
      
      // FIXME: There is no clear ownership policy now for ObjCProtocolDecls
      //  referenced by ObjCForwardProtocolDecl.  Some of them can be forward 
      //  decls that are never later defined (and forward decls can be
      //  referenced by multiple ObjCClassDecls) or the ObjCProtocolDecl 
      //  later becomes a real definition. 
      //  Ideally we should have separate objects for forward declarations and
      //  definitions, obviating this problem.  Because of this situation,
      //  referenced ObjCProtocolDecls are destroyed here.  
      if (ObjCForwardProtocolDecl* FDec = dyn_cast<ObjCForwardProtocolDecl>(*I))
        for (ObjCForwardProtocolDecl::iterator ID=FDec->begin(),
             ED=FDec->end(); ID!=ED; ++ID) {          
          if (!*ID || Killed.count(*ID)) continue;
          Killed.insert(*ID);
          (*ID)->Destroy(*Context);
        }
      
            
      (*I)->Destroy(*Context);
    }
  }

  if (OwnsMetaData && Context) {
    // The ASTContext object has the sole references to the IdentifierTable
    // Selectors, and the Target information.  Go and delete them, since
    // the TranslationUnit effectively owns them.
    
    delete &(Context->Idents);
    delete &(Context->Selectors);
    delete &(Context->Target);
    delete Context;
  }  
}

bool clang::EmitASTBitcodeFile(const TranslationUnit* TU,                                
                               const llvm::sys::Path& Filename) {

  return TU ? EmitASTBitcodeFile(*TU, Filename) : false;
}
  
bool clang::EmitASTBitcodeBuffer(const TranslationUnit* TU, 
                                 std::vector<unsigned char>& Buffer) {

  return TU ? EmitASTBitcodeBuffer(*TU, Buffer) : false;
}

bool clang::EmitASTBitcodeStream(const TranslationUnit* TU, 
                                 std::ostream& Stream) {

  return TU ? EmitASTBitcodeStream(*TU, Stream) : false;
}

bool clang::EmitASTBitcodeBuffer(const TranslationUnit& TU, 
                                 std::vector<unsigned char>& Buffer) {
  // Create bitstream.
  llvm::BitstreamWriter Stream(Buffer);
  
  // Emit the preamble.
  Stream.Emit((unsigned)'B', 8);
  Stream.Emit((unsigned)'C', 8);
  Stream.Emit(0xC, 4);
  Stream.Emit(0xF, 4);
  Stream.Emit(0xE, 4);
  Stream.Emit(0x0, 4);
  
  { 
    // Create serializer.  Placing it in its own scope assures any necessary
    // finalization of bits to the buffer in the serializer's dstor.    
    llvm::Serializer Sezr(Stream);  
    
    // Emit the translation unit.
    TU.Emit(Sezr);
  }
  
  return true;
}

bool clang::EmitASTBitcodeStream(const TranslationUnit& TU, 
                                 std::ostream& Stream) {  
  
  // Reserve 256K for bitstream buffer.
  std::vector<unsigned char> Buffer;
  Buffer.reserve(256*1024);
  
  EmitASTBitcodeBuffer(TU,Buffer);
  
  // Write the bits to disk.
  Stream.write((char*)&Buffer.front(), Buffer.size());
  return true;
}

bool clang::EmitASTBitcodeFile(const TranslationUnit& TU, 
                               const llvm::sys::Path& Filename) {  
  
  // Reserve 256K for bitstream buffer.
  std::vector<unsigned char> Buffer;
  Buffer.reserve(256*1024);
  
  EmitASTBitcodeBuffer(TU,Buffer);
  
  // Write the bits to disk. 
  if (FILE* fp = fopen(Filename.c_str(),"wb")) {
    fwrite((char*)&Buffer.front(), sizeof(char), Buffer.size(), fp);
    fclose(fp);
    return true;
  }

  return false;  
}

void TranslationUnit::Emit(llvm::Serializer& Sezr) const {

  // ===---------------------------------------------------===/
  //      Serialize the top-level decls.
  // ===---------------------------------------------------===/  
  
  Sezr.EnterBlock(DeclsBlock);

  // Only serialize the head of a decl chain.  The ASTConsumer interfaces
  // provides us with each top-level decl, including those nested in
  // a decl chain, so we may be passed decls that are already serialized.  
  for (const_iterator I=begin(), E=end(); I!=E; ++I) 
      if (!Sezr.isRegistered(*I))
        Sezr.EmitOwnedPtr(*I);
  
  Sezr.ExitBlock();
  
  // ===---------------------------------------------------===/
  //      Serialize the "Translation Unit" metadata.
  // ===---------------------------------------------------===/

  // Emit ASTContext.
  Sezr.EnterBlock(ASTContextBlock);  
  Sezr.EmitOwnedPtr(Context);  
  Sezr.ExitBlock();    
  
  Sezr.EnterBlock(BasicMetadataBlock);
  
  // Block for SourceManager and Target.  Allows easy skipping
  // around to the block for the Selectors during deserialization.
  Sezr.EnterBlock();
    
  // Emit the SourceManager.
  Sezr.Emit(Context->getSourceManager());
    
  // Emit the Target.
  Sezr.EmitPtr(&Context->Target);
  Sezr.EmitCStr(Context->Target.getTargetTriple());
  
  Sezr.ExitBlock(); // exit "BasicMetadataBlock"
  
  // Emit the Selectors.
  Sezr.Emit(Context->Selectors);
  
  // Emit the Identifier Table.
  Sezr.Emit(Context->Idents);
  
  Sezr.ExitBlock();  // exit "ASTContextBlock"
}

TranslationUnit*
clang::ReadASTBitcodeBuffer(llvm::MemoryBuffer& MBuffer, FileManager& FMgr) {

  // Check if the file is of the proper length.
  if (MBuffer.getBufferSize() & 0x3) {
    // FIXME: Provide diagnostic: "Length should be a multiple of 4 bytes."
    return NULL;
  }
  
  // Create the bitstream reader.
  unsigned char *BufPtr = (unsigned char *) MBuffer.getBufferStart();
  llvm::BitstreamReader Stream(BufPtr,BufPtr+MBuffer.getBufferSize());
  
  if (Stream.Read(8) != 'B' ||
      Stream.Read(8) != 'C' ||
      Stream.Read(4) != 0xC ||
      Stream.Read(4) != 0xF ||
      Stream.Read(4) != 0xE ||
      Stream.Read(4) != 0x0) {
    // FIXME: Provide diagnostic.
    return NULL;
  }
  
  // Create the deserializer.
  llvm::Deserializer Dezr(Stream);
  
  return TranslationUnit::Create(Dezr,FMgr);
}

TranslationUnit*
clang::ReadASTBitcodeFile(const llvm::sys::Path& Filename, FileManager& FMgr) {
  
  // Create the memory buffer that contains the contents of the file.  
  llvm::OwningPtr<llvm::MemoryBuffer> 
    MBuffer(llvm::MemoryBuffer::getFile(Filename.c_str()));
  
  if (!MBuffer) {
    // FIXME: Provide diagnostic.
    return NULL;
  }
  
  return ReadASTBitcodeBuffer(*MBuffer, FMgr);
}

TranslationUnit* TranslationUnit::Create(llvm::Deserializer& Dezr,
                                         FileManager& FMgr) {
  
  // Create the translation unit object.
  TranslationUnit* TU = new TranslationUnit();
  
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
  SourceManager::CreateAndRegister(Dezr,FMgr);
    
  { // Read the TargetInfo.
    llvm::SerializedPtrID PtrID = Dezr.ReadPtrID();
    char* triple = Dezr.ReadCStr(NULL,0,true);
    Dezr.RegisterPtr(PtrID, TargetInfo::CreateTargetInfo(std::string(triple)));
    delete [] triple;
  }
  
  // For Selectors, we must read the identifier table first because the
  //  SelectorTable depends on the identifiers being already deserialized.
  llvm::Deserializer::Location SelectorBlkLoc = Dezr.getCurrentBlockLocation();  
  Dezr.SkipBlock();
  
  // Read the identifier table.
  IdentifierTable::CreateAndRegister(Dezr);
  
  // Now jump back and read the selectors.
  Dezr.JumpTo(SelectorBlkLoc);
  SelectorTable::CreateAndRegister(Dezr);
  
  // Now jump back to ASTContextBlock and read the ASTContext.
  Dezr.JumpTo(ASTContextBlockLoc);
  TU->Context = Dezr.ReadOwnedPtr<ASTContext>();
  
  // "Rewind" the stream.  Find the block with the serialized top-level decls.
  Dezr.Rewind();
  FoundBlock = Dezr.SkipToBlock(DeclsBlock);
  assert (FoundBlock);
  llvm::Deserializer::Location DeclBlockLoc = Dezr.getCurrentBlockLocation();
  
  while (!Dezr.FinishedBlock(DeclBlockLoc))
    TU->AddTopLevelDecl(Dezr.ReadOwnedPtr<Decl>(*TU->Context));

  return TU;
}

