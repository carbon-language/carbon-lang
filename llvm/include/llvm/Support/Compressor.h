//===- llvm/Support/Compressor.h --------------------------------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file declares the llvm::Compressor class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_COMPRESSOR_H
#define LLVM_SUPPORT_COMPRESSOR_H

#include "llvm/Support/DataTypes.h"

namespace llvm {

  /// This class provides an abstraction for compressing a block of memory using
  /// a standard compression utility such as bzip2 or libz. This interface
  /// allows us to abstract the notion of compression and deal with alternate
  /// compression scheme availability depending on the configured platform. This
  /// facility will always favor a bzip2 implementation if its available.
  /// Otherwise, libz will be used if it is available. If neither zlib nor bzip2
  /// are available, a very simple algorithm provided by the Compressor class
  /// will be used. The type of compression used can be determined by inspecting 
  /// the first byte of the compressed output. ASCII values '0', '1', and '2', 
  /// denote the compression type as given in the Algorithm enumeration below.
  /// The Compressor is intended for use with memory mapped files where the 
  /// entire data block to be compressed or decompressed is available in 
  /// memory. However, output can be gathered in repeated calls to a callback.
  /// @since 1.4
  /// @brief An abstraction for memory to memory data (de)compression
  class Compressor {
    /// @name Types
    /// @{
    public:
      enum Algorithm {
        COMP_TYPE_SIMPLE = '0',  ///< Use simple but ubiquitous algorithm
        COMP_TYPE_ZLIB = '1',    ///< Use zlib algorithm, if available
        COMP_TYPE_BZIP2 = '2',   ///< Use bzip2 algorithm (preferred)
      };

      /// A callback function type used by the Compressor to get the next chunk 
      /// of data to which (de)compressed output will be written. This function
      /// must be written by the caller to provide the buffering of the output
      /// data.
      /// @returns 0 for success, 1 for failure
      /// @throws nothing
      /// @brief Output callback function type
      typedef unsigned (OutputDataCallback)(char*& buffer, unsigned& size,
                                            void* context);

    /// @}
    /// @name Methods
    /// @{
    public:
      /// This function does the compression work. The block of memory starting
      /// at \p in and extending for \p size bytes is compressed. The compressed
      /// output is written to memory blocks returned by the \p cb callback. The
      /// caller must provide an implementation of the OutputDataCallback
      /// function type and provide its address as \p cb. Note that the callback
      /// function will be called as many times as necessary to complete the
      /// compression of the \p in block but that the total size will generally
      /// be less than \p size. It is a good idea to provide as large a value to
      /// the callback's \p size parameter as possible so that fewer calls to
      /// the callback are made. The \p hint parameter tells the function which
      /// kind of compression to start with. However, if its not available on
      /// the platform, the algorithm "falls back" from bzip2 -> zlib -> simple.
      /// @throws std::string if an error occurs
      /// @returns the total size of the compressed data
      /// @brief Compress a block of memory.
      static uint64_t compress(char* in, unsigned size, OutputDataCallback* cb,
                               Algorithm hint = COMP_TYPE_BZIP2,
                               void* context = 0);

      /// This function does the decompression work. The block of memory
      /// starting at \p in and extending for \p size bytes is decompressed. The
      /// decompressed output is written to memory blocks returned by the \p cb
      /// callback. The caller must provide an implementation of the
      /// OutputDataCallback function type and provide its address as \p cb.
      /// Note that the callback function will be called as many times as
      /// necessary to complete the compression of the \p in block but that the
      /// total size will generally be greater than \p size. It is a good idea
      /// to provide as large a value to the callback's \p size parameter as 
      /// possible so that fewer calls to the callback are made.
      /// @throws std::string if an error occurs
      /// @returns the total size of the decompressed data
      /// @brief Decompress a block of memory.
      static uint64_t decompress(char *in, unsigned size, 
                                 OutputDataCallback* cb, void* context = 0);

    /// @}
  };
}

// vim: sw=2 ai

#endif
