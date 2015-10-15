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

#ifdef __ANDROID__
// Android does not have SUN_LEN
#ifndef SUN_LEN
#define SUN_LEN(ptr) ((size_t) (((struct sockaddr_un *) 0)->sun_path) + strlen((ptr)->sun_path))
#endif
#endif // #ifdef __ANDROID__

namespace {

const int kDomain = AF_UNIX;
const int kType   = SOCK_STREAM;

void SetSockAddr(llvm::StringRef name, sockaddr_un* saddr_un)
{
    saddr_un->sun_family = kDomain;
    ::strncpy(saddr_un->sun_path, name.data(), sizeof(saddr_un->sun_path) - 1);
    saddr_un->sun_path[sizeof(saddr_un->sun_path) - 1] = '\0';
#if defined(__APPLE__) || defined(__FreeBSD__) || defined(__NetBSD__)
    saddr_un->sun_len = SUN_LEN (saddr_un);
#endif
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

Error
DomainSocket::Connect(llvm::StringRef name)
{
    sockaddr_un saddr_un;
    SetSockAddr(name, &saddr_un);

    Error error;
    if (::connect(GetNativeSocket(), (struct sockaddr *)&saddr_un, SUN_LEN (&saddr_un)) < 0)
        SetLastError (error);

    return error;
}

Error
DomainSocket::Listen(llvm::StringRef name, int backlog)
{
    sockaddr_un saddr_un;
    SetSockAddr(name, &saddr_un);

    FileSystem::Unlink(FileSpec{name, true});

    Error error;
    if (::bind(GetNativeSocket(), (struct sockaddr *)&saddr_un, SUN_LEN (&saddr_un)) == 0)
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
