//===-- RNBSocket.cpp -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Created by Greg Clayton on 12/12/07.
//
//===----------------------------------------------------------------------===//

#include "RNBSocket.h"
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <termios.h>
#include "DNBLog.h"
#include "DNBError.h"

#if defined (__arm__)
#include "lockdown.h"
#endif

/* Once we have a RNBSocket object with a port # specified,
   this function is called to wait for an incoming connection.
   This function blocks while waiting for that connection.  */

rnb_err_t
RNBSocket::Listen (in_port_t listen_port_num)
{
    //DNBLogThreadedIf(LOG_RNB_COMM, "%8u RNBSocket::%s called", (uint32_t)m_timer.ElapsedMicroSeconds(true), __FUNCTION__);
    // Disconnect without saving errno
    Disconnect (false);

    DNBError err;
    int listen_port = ::socket (AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (listen_port == -1)
        err.SetError(errno, DNBError::POSIX);

    if (err.Fail() || DNBLogCheckLogBit(LOG_RNB_COMM))
        err.LogThreaded("::socket ( domain = AF_INET, type = SOCK_STREAM, protocol = IPPROTO_TCP ) => socket = %i", listen_port);

    if (err.Fail())
        return rnb_err;

    // enable local address reuse
    SetSocketOption (listen_port, SOL_SOCKET, SO_REUSEADDR, 1);

    struct sockaddr_in sa;
    ::memset (&sa, 0, sizeof sa);
    sa.sin_len = sizeof sa;
    sa.sin_family = AF_INET;
    sa.sin_port = htons (listen_port_num);
    sa.sin_addr.s_addr = htonl (INADDR_ANY);

    int error = ::bind (listen_port, (struct sockaddr *) &sa, sizeof(sa));
    if (error == -1)
        err.SetError(errno, DNBError::POSIX);

    if (err.Fail() || DNBLogCheckLogBit(LOG_RNB_COMM))
        err.LogThreaded("::bind ( socket = %i, (struct sockaddr *) &sa, sizeof(sa)) )", listen_port);

    if (err.Fail())
    {
        ClosePort (listen_port, false);
        return rnb_err;
    }

    error = ::listen (listen_port, 1);
    if (error == -1)
        err.SetError(errno, DNBError::POSIX);

    if (err.Fail() || DNBLogCheckLogBit(LOG_RNB_COMM))
        err.LogThreaded("::listen ( socket = %i, backlog = 1 )", listen_port);

    if (err.Fail())
    {
        ClosePort (listen_port, false);
        return rnb_err;
    }

    m_conn_port = ::accept (listen_port, NULL, 0);
    if (m_conn_port == -1)
        err.SetError(errno, DNBError::POSIX);

    if (err.Fail() || DNBLogCheckLogBit(LOG_RNB_COMM))
        err.LogThreaded("::accept ( socket = %i, address = NULL, address_len = 0 )", listen_port);

    if (err.Fail())
    {
        ClosePort (listen_port, false);
        return rnb_err;
    }
    else
    {
        // We are done with the listen port
        ClosePort (listen_port, false);

        // Keep our TCP packets coming without any delays.
        SetSocketOption (m_conn_port, IPPROTO_TCP, TCP_NODELAY, 1);
    }

    return rnb_success;
}

#if defined (__arm__)
rnb_err_t
RNBSocket::ConnectToService()
{
    DNBLog("Connecting to com.apple.%s service...", DEBUGSERVER_PROGRAM_NAME);
    // Disconnect from any previous connections
    Disconnect(false);

    m_conn_port = ::lockdown_checkin (NULL, NULL);
    if (m_conn_port == -1)
    {
        DNBLogThreadedIf(LOG_RNB_COMM, "::lockdown_checkin(NULL, NULL) failed");
        return rnb_not_connected;
    }
    m_conn_port_from_lockdown = true;
    return rnb_success;
}
#endif

rnb_err_t
RNBSocket::OpenFile (const char *path)
{
    DNBError err;
    m_conn_port = open (path, O_RDWR);
    if (m_conn_port == -1)
    {
        err.SetError(errno, DNBError::POSIX);
        err.LogThreaded ("can't open file '%s'", path);
        return rnb_not_connected;
    }
    else
    {
        struct termios stdin_termios;

        if (::tcgetattr (m_conn_port, &stdin_termios) == 0)
        {
            stdin_termios.c_lflag &= ~ECHO;     // Turn off echoing
            stdin_termios.c_lflag &= ~ICANON;   // Get one char at a time
            ::tcsetattr (m_conn_port, TCSANOW, &stdin_termios);
        }
    }
    return rnb_success;
}

int
RNBSocket::SetSocketOption(int fd, int level, int option_name, int option_value)
{
    return ::setsockopt(fd, level, option_name, &option_value, sizeof(option_value));
}

rnb_err_t
RNBSocket::Disconnect (bool save_errno)
{
    if (m_conn_port_from_lockdown)
        m_conn_port_from_lockdown = false;
    return ClosePort (m_conn_port, save_errno);
}


rnb_err_t
RNBSocket::Read (std::string &p)
{
    char buf[1024];
    p.clear();

    // Note that BUF is on the stack so we must be careful to keep any
    // writes to BUF from overflowing or we'll have security issues.

    if (m_conn_port == -1)
        return rnb_err;

    //DNBLogThreadedIf(LOG_RNB_COMM, "%8u RNBSocket::%s calling read()", (uint32_t)m_timer.ElapsedMicroSeconds(true), __FUNCTION__);
    DNBError err;
    int bytesread = read (m_conn_port, buf, sizeof (buf));
    if (bytesread <= 0)
        err.SetError(errno, DNBError::POSIX);
    else
        p.append(buf, bytesread);

    if (err.Fail() || DNBLogCheckLogBit(LOG_RNB_COMM))
        err.LogThreaded("::read ( %i, %p, %zu ) => %i", m_conn_port, buf, sizeof (buf), bytesread);

    // Our port went away - we have to mark this so IsConnected will return the truth.
    if (bytesread == 0)
    {
        m_conn_port = -1;
        return rnb_not_connected;
    }
    else if (bytesread == -1)
    {
        m_conn_port = -1;
        return rnb_err;
    }
    // Strip spaces from the end of the buffer
    while (!p.empty() && isspace (p[p.size() - 1]))
        p.erase (p.size () - 1);

    // Most data in the debugserver packets valid printable characters...
    DNBLogThreadedIf(LOG_RNB_COMM, "read: %s", p.c_str());
    return rnb_success;
}

rnb_err_t
RNBSocket::Write (const void *buffer, size_t length)
{
    if (m_conn_port == -1)
        return rnb_err;

    DNBError err;
    int bytessent = send (m_conn_port, buffer, length, 0);
    if (bytessent < 0)
        err.SetError(errno, DNBError::POSIX);

    if (err.Fail() || DNBLogCheckLogBit(LOG_RNB_COMM))
        err.LogThreaded("::send ( socket = %i, buffer = %p, length = %zu, flags = 0 ) => %i", m_conn_port, buffer, length, bytessent);

    if (bytessent < 0)
        return rnb_err;

    if (bytessent != length)
        return rnb_err;

    DNBLogThreadedIf(LOG_RNB_PACKETS, "putpkt: %*s", length, (char *)buffer);   // All data is string based in debugserver, so this is safe
    DNBLogThreadedIf(LOG_RNB_COMM, "sent: %*s", length, (char *)buffer);

    return rnb_success;
}


rnb_err_t
RNBSocket::ClosePort (int& fd, bool save_errno)
{
    int close_err = 0;
    if (fd > 0)
    {
        errno = 0;
        close_err = close (fd);
        fd = -1;
    }
    return close_err != 0 ? rnb_err : rnb_success;
}


