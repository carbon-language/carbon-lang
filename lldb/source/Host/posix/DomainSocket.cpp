//===-- DomainSocket.cpp ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/posix/DomainSocket.h"

#include "lldb/Host/FileSystem.h"

#include <sys/socket.h>
#include <sys/un.h>

using namespace lldb;
using namespace lldb_private;

namespace {

const int kDomain = AF_UNIX;
const int kType   = SOCK_STREAM;

bool SetSockAddr(llvm::StringRef name, const size_t name_offset, sockaddr_un* saddr_un)
{
    if (name.size() + name_offset > sizeof(saddr_un->sun_path))
        return false;

    saddr_un->sun_family = kDomain;
    memset(saddr_un->sun_path, 0, sizeof(saddr_un->sun_path));

    strncpy(&saddr_un->sun_path[name_offset], name.data(), name.size());
#if defined(__APPLE__) || defined(__FreeBSD__) || defined(__NetBSD__)
    saddr_un->sun_len = SUN_LEN (saddr_un);
#endif
    return true;
}

}

DomainSocket::DomainSocket(NativeSocket socket)
    : Socket(socket, ProtocolUnixDomain, true)
{
}

DomainSocket::DomainSocket(bool child_processes_inherit, Error &error)
    : DomainSocket(CreateSocket(kDomain, kType, 0, child_processes_inherit, error))
{
}

DomainSocket::DomainSocket(SocketProtocol protocol, bool child_processes_inherit, Error &error)
    : Socket(CreateSocket(kDomain, kType, 0, child_processes_inherit, error), protocol, true)
{
}

Error
DomainSocket::Connect(llvm::StringRef name)
{
    sockaddr_un saddr_un;
    if (!SetSockAddr(name, GetNameOffset(), &saddr_un))
        return Error("Failed to set socket address");

    Error error;
    if (::connect(GetNativeSocket(), (struct sockaddr *)&saddr_un, sizeof(saddr_un)) < 0)
        SetLastError (error);

    return error;
}

Error
DomainSocket::Listen(llvm::StringRef name, int backlog)
{
    sockaddr_un saddr_un;
    if (!SetSockAddr(name, GetNameOffset(), &saddr_un))
        return Error("Failed to set socket address");

    DeleteSocketFile(name);

    Error error;
    if (::bind(GetNativeSocket(), (struct sockaddr *)&saddr_un, sizeof(saddr_un)) == 0)
        if (::listen(GetNativeSocket(), backlog) == 0)
            return error;

    SetLastError(error);
    return error;
}

Error
DomainSocket::Accept(llvm::StringRef name, bool child_processes_inherit, Socket *&socket)
{
    Error error;
    auto conn_fd = AcceptSocket(GetNativeSocket(), nullptr, nullptr, child_processes_inherit, error);
    if (error.Success())
        socket = new DomainSocket(conn_fd);

    return error;
}

size_t
DomainSocket::GetNameOffset() const
{
    return 0;
}

void
DomainSocket::DeleteSocketFile(llvm::StringRef name)
{
    FileSystem::Unlink(FileSpec{name, true});
}
