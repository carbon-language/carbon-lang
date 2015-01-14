//===-- PipePosix.h ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Host_posix_PipePosix_h_
#define liblldb_Host_posix_PipePosix_h_
#if defined(__cplusplus)

#include "lldb/Host/PipeBase.h"

namespace lldb_private {

//----------------------------------------------------------------------
/// @class PipePosix PipePosix	.h "lldb/Host/posix/PipePosix.h"
/// @brief A posix-based implementation of Pipe, a class that abtracts
///        unix style pipes.
///
/// A class that abstracts the LLDB core from host pipe functionality.
//----------------------------------------------------------------------
class PipePosix : public PipeBase
{
public:
    static int kInvalidDescriptor;

    PipePosix();

    ~PipePosix() override;

    Error
    CreateNew(bool child_process_inherit) override;
    Error
    CreateNew(llvm::StringRef name, bool child_process_inherit) override;
    Error
    OpenAsReader(llvm::StringRef name, bool child_process_inherit) override;
    Error
    OpenAsWriterWithTimeout(llvm::StringRef name, bool child_process_inherit, const std::chrono::microseconds &timeout) override;

    bool
    CanRead() const override;
    bool
    CanWrite() const override;

    int
    GetReadFileDescriptor() const override;
    int
    GetWriteFileDescriptor() const override;
    int
    ReleaseReadFileDescriptor() override;
    int
    ReleaseWriteFileDescriptor() override;

    // Close both descriptors
    void
    Close() override;

    Error
    Delete(llvm::StringRef name) override;

    Error
    Write(const void *buf, size_t size, size_t &bytes_written) override;
    Error
    ReadWithTimeout(void *buf, size_t size, const std::chrono::microseconds &timeout, size_t &bytes_read) override;

private:
    void
    CloseReadFileDescriptor();
    void
    CloseWriteFileDescriptor();

    int m_fds[2];
};

} // namespace lldb_private

#endif  // #if defined(__cplusplus)
#endif  // liblldb_Host_posix_PipePosix_h_
