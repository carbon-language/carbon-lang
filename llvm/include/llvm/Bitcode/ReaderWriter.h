//===-- llvm/Bitcode/ReaderWriter.h - Bitcode reader/writers ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This header defines interfaces to read and write LLVM bitcode files/streams.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_BITCODE_H
#define LLVM_BITCODE_H

#include <string>

namespace llvm {
  class Module;
  class MemoryBuffer;
  class ModulePass;
  class BitstreamWriter;
  class LLVMContext;
  class raw_ostream;
  
  /// getLazyBitcodeModule - Read the header of the specified bitcode buffer
  /// and prepare for lazy deserialization of function bodies.  If successful,
  /// this takes ownership of 'buffer' and returns a non-null pointer.  On
  /// error, this returns null, *does not* take ownership of Buffer, and fills
  /// in *ErrMsg with an error description if ErrMsg is non-null.
  Module *getLazyBitcodeModule(MemoryBuffer *Buffer,
                               LLVMContext& Context,
                               std::string *ErrMsg = 0);

  /// getBitcodeTargetTriple - Read the header of the specified bitcode
  /// buffer and extract just the triple information. If successful,
  /// this returns a string and *does not* take ownership
  /// of 'buffer'. On error, this returns "", and fills in *ErrMsg
  /// if ErrMsg is non-null.
  std::string getBitcodeTargetTriple(MemoryBuffer *Buffer,
                                     LLVMContext& Context,
                                     std::string *ErrMsg = 0);

  /// ParseBitcodeFile - Read the specified bitcode file, returning the module.
  /// If an error occurs, this returns null and fills in *ErrMsg if it is
  /// non-null.  This method *never* takes ownership of Buffer.
  Module *ParseBitcodeFile(MemoryBuffer *Buffer, LLVMContext& Context,
                           std::string *ErrMsg = 0);

  /// WriteBitcodeToFile - Write the specified module to the specified
  /// raw output stream.  For streams where it matters, the given stream
  /// should be in "binary" mode.
  void WriteBitcodeToFile(const Module *M, raw_ostream &Out);

  /// WriteBitcodeToStream - Write the specified module to the specified
  /// raw output stream.
  void WriteBitcodeToStream(const Module *M, BitstreamWriter &Stream);

  /// createBitcodeWriterPass - Create and return a pass that writes the module
  /// to the specified ostream.
  ModulePass *createBitcodeWriterPass(raw_ostream &Str);

  /// createBitcodeVerifierPass - Create a pass that writes a module to disk and
  /// then reads the module back in to verify bitcode serialization and
  /// deserialization.
  ModulePass *createBitcodeVerifierPass(raw_ostream &Str);

  /// isBitcodeWrapper - Return true if the given bytes are the magic bytes
  /// for an LLVM IR bitcode wrapper.
  ///
  static inline bool isBitcodeWrapper(const unsigned char *BufPtr,
                                      const unsigned char *BufEnd) {
    // See if you can find the hidden message in the magic bytes :-).
    // (Hint: it's a little-endian encoding.)
    return BufPtr != BufEnd &&
           BufPtr[0] == 0xDE &&
           BufPtr[1] == 0xC0 &&
           BufPtr[2] == 0x17 &&
           BufPtr[3] == 0x0B;
  }

  /// isRawBitcode - Return true if the given bytes are the magic bytes for
  /// raw LLVM IR bitcode (without a wrapper).
  ///
  static inline bool isRawBitcode(const unsigned char *BufPtr,
                                  const unsigned char *BufEnd) {
    // These bytes sort of have a hidden message, but it's not in
    // little-endian this time, and it's a little redundant.
    return BufPtr != BufEnd &&
           BufPtr[0] == 'B' &&
           BufPtr[1] == 'C' &&
           BufPtr[2] == 0xc0 &&
           BufPtr[3] == 0xde;
  }

  /// isBitcode - Return true if the given bytes are the magic bytes for
  /// LLVM IR bitcode, either with or without a wrapper.
  ///
  static bool inline isBitcode(const unsigned char *BufPtr,
                               const unsigned char *BufEnd) {
    return isBitcodeWrapper(BufPtr, BufEnd) ||
           isRawBitcode(BufPtr, BufEnd);
  }

  /// SkipBitcodeWrapperHeader - Some systems wrap bc files with a special
  /// header for padding or other reasons.  The format of this header is:
  ///
  /// struct bc_header {
  ///   uint32_t Magic;         // 0x0B17C0DE
  ///   uint32_t Version;       // Version, currently always 0.
  ///   uint32_t BitcodeOffset; // Offset to traditional bitcode file.
  ///   uint32_t BitcodeSize;   // Size of traditional bitcode file.
  ///   ... potentially other gunk ...
  /// };
  /// 
  /// This function is called when we find a file with a matching magic number.
  /// In this case, skip down to the subsection of the file that is actually a
  /// BC file.
  static inline bool SkipBitcodeWrapperHeader(unsigned char *&BufPtr,
                                              unsigned char *&BufEnd) {
    enum {
      KnownHeaderSize = 4*4,  // Size of header we read.
      OffsetField = 2*4,      // Offset in bytes to Offset field.
      SizeField = 3*4         // Offset in bytes to Size field.
    };
    
    // Must contain the header!
    if (BufEnd-BufPtr < KnownHeaderSize) return true;
    
    unsigned Offset = ( BufPtr[OffsetField  ]        |
                       (BufPtr[OffsetField+1] << 8)  |
                       (BufPtr[OffsetField+2] << 16) |
                       (BufPtr[OffsetField+3] << 24));
    unsigned Size   = ( BufPtr[SizeField    ]        |
                       (BufPtr[SizeField  +1] << 8)  |
                       (BufPtr[SizeField  +2] << 16) |
                       (BufPtr[SizeField  +3] << 24));
    
    // Verify that Offset+Size fits in the file.
    if (Offset+Size > unsigned(BufEnd-BufPtr))
      return true;
    BufPtr += Offset;
    BufEnd = BufPtr+Size;
    return false;
  }
} // End llvm namespace

#endif
