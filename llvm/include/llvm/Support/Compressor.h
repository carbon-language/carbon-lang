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
#include <ostream>

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

    /// @}
    /// @name High Level Interface
    /// @{
    public:
      /// This method compresses a block of memory pointed to by \p in with 
      /// size \p size to a block of memory, \p out, that is allocated with 
      /// malloc. It is the caller's responsibility to free \p out. The \p hint
      /// indicates which type of compression the caller would *prefer*.
      /// @throws std::string explaining error if a compression error occurs
      /// @returns The size of the output buffer \p out.
      /// @brief Compress memory to a new memory buffer.
      static uint64_t compressToNewBuffer(
        const char* in,           ///< The buffer to be compressed
        unsigned size,            ///< The size of the buffer to be compressed
        char*&out,                ///< The returned output buffer
        Algorithm hint            ///< Hint for type of compression to perform
          = COMP_TYPE_BZIP2
      );

      /// This method compresses a block of memory pointed to by \p in with 
      /// size \p size to a stream. The stream \p out must be open and ready for
      /// writing when this method is called. The stream will not be closed by
      /// this method.  The \p hint argument indicates which type of 
      /// compression the caller would *prefer*.
      /// @returns The amount of data written to \p out.
      /// @brief Compress memory to a file.
      static uint64_t compressToStream(
        const char*in,            ///< The buffer to be compressed
        unsigned size,            ///< The size of the buffer to be compressed
        std::ostream& out,        ///< The output stream to write data on
        Algorithm hint            ///< Hint for type of compression to perform
          = COMP_TYPE_BZIP2
      );

      /// This method decompresses a block of memory pointed to by \p in with 
      /// size \p size to a new block of memory, \p out, \p that was allocated
      /// by malloc. It is the caller's responsibility to free \p out. 
      /// @returns The size of the output buffer \p out.
      /// @brief Decompress memory to a new memory buffer.
      static uint64_t decompressToNewBuffer(
        const char *in,           ///< The buffer to be decompressed
        unsigned size,            ///< Size of the buffer to be decompressed
        char*&out                 ///< The returned output buffer
      );

      /// This method decompresses a block of memory pointed to by \p in with 
      /// size \p size to a stream. The stream \p out must be open and ready for
      /// writing when this method is called. The stream will not be closed by
      /// this method. 
      /// @returns The amount of data written to \p out.
      /// @brief Decompress memory to a stream.
      static uint64_t decompressToStream(
        const char *in,           ///< The buffer to be decompressed
        unsigned size,            ///< Size of the buffer to be decompressed
        std::ostream& out         ///< The stream to write write data on
      );

    /// @}
    /// @name Low Level Interface
    /// @{
    public:
      /// A callback function type used by the Compressor's low level interface
      /// to get the next chunk of data to which (de)compressed output will be 
      /// written. This callback completely abstracts the notion of how to 
      /// handle the output data of compression or decompression. The callback
      /// is responsible for determining both the storage location and the size 
      /// of the output. The callback may also do other things with the data
      /// such as write it, transmit it, etc. Note that providing very small
      /// values for \p size will make the compression run very inefficiently.
      /// It is recommended that \p size be chosen based on the some multiple or
      /// fraction of the object being decompressed or compressed, respetively.
      /// @returns 0 for success, 1 for failure
      /// @throws nothing
      /// @brief Output callback function type
      typedef unsigned (OutputDataCallback)(char*& buffer, unsigned& size,
                                            void* context);

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
      static uint64_t compress(
        const char* in,            ///< The buffer to be compressed
        unsigned size,             ///< The size of the buffer to be compressed
        OutputDataCallback* cb,    ///< Call back for memory allocation
        Algorithm hint             ///< Hint for type of compression to perform
          = COMP_TYPE_BZIP2,
        void* context              ///< Context for callback
          = 0
      );

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
      static uint64_t decompress(
        const char *in,              ///< The buffer to be decompressed
        unsigned size,               ///< Size of the buffer to be decompressed
        OutputDataCallback* cb,      ///< Call back for memory allocation
        void* context                ///< Context for callback
          = 0
      );

    /// @}
  };
}

// vim: sw=2 ai

#endif
