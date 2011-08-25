//===--- ModuleManager.cpp - Module Manager ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the ModuleManager class, which manages a set of loaded
//  modules for the ASTReader.
//
//===----------------------------------------------------------------------===//
#include "clang/Serialization/ModuleManager.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/system_error.h"

using namespace clang;
using namespace serialization;

Module *ModuleManager::lookup(StringRef Name) {
  const FileEntry *Entry = FileMgr.getFile(Name);
  return Modules[Entry];
}

llvm::MemoryBuffer *ModuleManager::lookupBuffer(StringRef Name) {
  const FileEntry *Entry = FileMgr.getFile(Name);
  return InMemoryBuffers[Entry];
}

std::pair<Module *, bool>
ModuleManager::addModule(StringRef FileName, ModuleKind Type, 
                         Module *ImportedBy, std::string &ErrorStr) {
  const FileEntry *Entry = FileMgr.getFile(FileName);
  if (!Entry && FileName != "-") {
    ErrorStr = "file not found";
    return std::make_pair(static_cast<Module*>(0), false);
  }
  
  // Check whether we already loaded this module, before 
  Module *&ModuleEntry = Modules[Entry];
  bool NewModule = false;
  if (!ModuleEntry) {
    // Allocate a new module.
    Module *New = new Module(Type);
    New->FileName = FileName.str();
    Chain.push_back(New);
    NewModule = true;
    ModuleEntry = New;
    
    // Load the contents of the module
    if (llvm::MemoryBuffer *Buffer = lookupBuffer(FileName)) {
      // The buffer was already provided for us.
      assert(Buffer && "Passed null buffer");
      New->Buffer.reset(Buffer);
    } else {
      // Open the AST file.
      llvm::error_code ec;
      if (FileName == "-") {
        ec = llvm::MemoryBuffer::getSTDIN(New->Buffer);
        if (ec)
          ErrorStr = ec.message();
      } else
        New->Buffer.reset(FileMgr.getBufferForFile(FileName, &ErrorStr));
      
      if (!New->Buffer)
        return std::make_pair(static_cast<Module*>(0), false);
    }
    
    // Initialize the stream
    New->StreamFile.init((const unsigned char *)New->Buffer->getBufferStart(),
                         (const unsigned char *)New->Buffer->getBufferEnd());     }
  
  if (ImportedBy) {
    ModuleEntry->ImportedBy.insert(ImportedBy);
    ImportedBy->Imports.insert(ModuleEntry);
  } else {
    ModuleEntry->DirectlyImported = true;
  }
  
  return std::make_pair(ModuleEntry, NewModule);
}

void ModuleManager::addInMemoryBuffer(StringRef FileName, 
                                      llvm::MemoryBuffer *Buffer) {
  
  const FileEntry *Entry = FileMgr.getVirtualFile(FileName, 
                                                  Buffer->getBufferSize(), 0);
  InMemoryBuffers[Entry] = Buffer;
}

ModuleManager::ModuleManager(const FileSystemOptions &FSO) : FileMgr(FSO) { }

ModuleManager::~ModuleManager() {
  for (unsigned i = 0, e = Chain.size(); i != e; ++i)
    delete Chain[e - i - 1];
}

void ModuleManager::visit(bool (*Visitor)(Module &M, void *UserData), 
                          void *UserData) {
  unsigned N = size();
  
  // Record the number of incoming edges for each module. When we
  // encounter a module with no incoming edges, push it into the queue
  // to seed the queue.
  SmallVector<Module *, 4> Queue;
  Queue.reserve(N);
  llvm::DenseMap<Module *, unsigned> UnusedIncomingEdges; 
  for (ModuleIterator M = begin(), MEnd = end(); M != MEnd; ++M) {
    if (unsigned Size = (*M)->ImportedBy.size())
      UnusedIncomingEdges[*M] = Size;
    else
      Queue.push_back(*M);
  }
  
  llvm::SmallPtrSet<Module *, 4> Skipped;
  unsigned QueueStart = 0;
  while (QueueStart < Queue.size()) {
    Module *CurrentModule = Queue[QueueStart++];
    
    // Check whether this module should be skipped.
    if (Skipped.count(CurrentModule))
      continue;
    
    if (Visitor(*CurrentModule, UserData)) {
      // The visitor has requested that cut off visitation of any
      // module that the current module depends on. To indicate this
      // behavior, we mark all of the reachable modules as having N
      // incoming edges (which is impossible otherwise).
      SmallVector<Module *, 4> Stack;
      Stack.push_back(CurrentModule);
      Skipped.insert(CurrentModule);
      while (!Stack.empty()) {
        Module *NextModule = Stack.back();
        Stack.pop_back();
        
        // For any module that this module depends on, push it on the
        // stack (if it hasn't already been marked as visited).
        for (llvm::SetVector<Module *>::iterator 
             M = NextModule->Imports.begin(),
             MEnd = NextModule->Imports.end();
             M != MEnd; ++M) {
          if (Skipped.insert(*M))
            Stack.push_back(*M);
        }
      }
      continue;
    }
    
    // For any module that this module depends on, push it on the
    // stack (if it hasn't already been marked as visited).
    for (llvm::SetVector<Module *>::iterator M = CurrentModule->Imports.begin(),
         MEnd = CurrentModule->Imports.end();
         M != MEnd; ++M) {
      
      // Remove our current module as an impediment to visiting the
      // module we depend on. If we were the last unvisited module
      // that depends on this particular module, push it into the
      // queue to be visited.
      unsigned &NumUnusedEdges = UnusedIncomingEdges[*M];
      if (NumUnusedEdges && (--NumUnusedEdges == 0))
        Queue.push_back(*M);
    }
  }
}

/// \brief Perform a depth-first visit of the current module.
static bool visitDepthFirst(Module &M, 
                            bool (*Visitor)(Module &M, bool Preorder, 
                                            void *UserData), 
                            void *UserData,
                            llvm::SmallPtrSet<Module *, 4> &Visited) {
  // Preorder visitation
  if (Visitor(M, /*Preorder=*/true, UserData))
    return true;
  
  // Visit children
  for (llvm::SetVector<Module *>::iterator IM = M.Imports.begin(),
       IMEnd = M.Imports.end();
       IM != IMEnd; ++IM) {
    if (!Visited.insert(*IM))
      continue;
    
    if (visitDepthFirst(**IM, Visitor, UserData, Visited))
      return true;
  }  
  
  // Postorder visitation
  return Visitor(M, /*Preorder=*/false, UserData);
}

void ModuleManager::visitDepthFirst(bool (*Visitor)(Module &M, bool Preorder, 
                                                    void *UserData), 
                                    void *UserData) {
  llvm::SmallPtrSet<Module *, 4> Visited;
  for (unsigned I = 0, N = Chain.size(); I != N; ++I) {
    if (!Visited.insert(Chain[I]))
      continue;
    
    if (::visitDepthFirst(*Chain[I], Visitor, UserData, Visited))
      return;
  }
}
